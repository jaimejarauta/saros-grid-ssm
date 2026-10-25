"""Spatio-Temporal Selective State Space Model for Transmission Grid Forecasting.

Combines non-Euclidean graph spatial diffusion with continuous-time diagonal selective
state space recurrence for nodal renewable injection and curtailment bottleneck prediction.
"""

from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSpatialDiffusion(nn.Module):
    """K-hop graph spatial diffusion operator reflecting physical transmission power flow."""
    def __init__(self, d_in: int, d_out: int, num_hops: int = 2):
        super().__init__()
        self.num_hops = num_hops
        self.d_in = d_in
        self.d_out = d_out
        
        # Learnable filter weights for each spatial hop k in {0, ..., num_hops}
        self.hop_weights = nn.ParameterList([
            nn.Parameter(torch.empty(d_in, d_out)) for _ in range(num_hops + 1)
        ])
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.norm = nn.LayerNorm(d_out)
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.hop_weights:
            nn.init.xavier_uniform_(w)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass for spatial diffusion.
        
        Args:
            x: Node features of shape (B, T, N, D_in)
            adj_norm: Normalized symmetric adjacency matrix (N, N)
        Returns:
            Diffused features of shape (B, T, N, D_out)
        """
        # P_0 = X
        p_curr = x
        out = torch.matmul(p_curr, self.hop_weights[0])

        # Successive Chebyshev / Adjacency polynomial diffusion hops
        for k in range(1, self.num_hops + 1):
            # p_k = A_norm * p_{k-1} along the node dimension N
            # (N, N) @ (B, T, N, D_in) -> (B, T, N, D_in)
            p_next = torch.einsum("nm,btmd->btnd", adj_norm, p_curr)
            out = out + torch.matmul(p_next, self.hop_weights[k])
            p_curr = p_next

        out = out + self.bias
        return self.norm(F.gelu(out))

class SelectiveSSMBlock(nn.Module):
    """Selective continuous-time diagonal state space recurrence layer.
    
    Implements input-dependent time-step discretization (Mamba/S4D formulation):
        A_c = -exp(alpha)  (BIBO stable diagonal transition)
        Delta_t = softplus(W_delta x_t + b_delta)
        B_t = W_B x_t,  C_t = W_C x_t
        h_t = exp(Delta_t A_c) * h_{t-1} + (Delta_t B_t) * x_t
        y_t = C_t^T h_t + D x_t
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        bidirectional: bool = True,
        dropout: float = 0.05
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional

        # Continuous diagonal transition matrix A_c parameterization
        # log_alpha enforces Re(A_c) = -exp(log_alpha) < 0 for unconditional stability
        self.log_alpha_fwd = nn.Parameter(torch.linspace(0.1, 2.0, d_state).unsqueeze(0))
        if bidirectional:
            self.log_alpha_bwd = nn.Parameter(torch.linspace(0.1, 2.0, d_state).unsqueeze(0))

        # Input-dependent selective projections (Delta, B, C)
        self.proj_delta_fwd = nn.Linear(d_model, d_state)
        self.proj_b_fwd = nn.Linear(d_model, d_state)
        self.proj_c_fwd = nn.Linear(d_model, d_state)

        if bidirectional:
            self.proj_delta_bwd = nn.Linear(d_model, d_state)
            self.proj_b_bwd = nn.Linear(d_model, d_state)
            self.proj_c_bwd = nn.Linear(d_model, d_state)

        # Feedthrough skip connection D
        self.d_skip = nn.Parameter(torch.ones(d_model))

        # Output projection and normalization
        out_dim = d_model * 2 if bidirectional else d_model
        self.out_proj = nn.Linear(out_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _selective_scan(
        self,
        x: torch.Tensor,
        log_alpha: torch.Tensor,
        proj_delta: nn.Linear,
        proj_b: nn.Linear,
        proj_c: nn.Linear,
        reverse: bool = False
    ) -> torch.Tensor:
        """Performs selective continuous-time SSM scan along sequence length dimension.
        
        Args:
            x: Tensor of shape (BatchSize, SeqLen, D_model)
        Returns:
            Output tensor of shape (BatchSize, SeqLen, D_model)
        """
        b_size, seq_len, _ = x.shape
        device = x.device

        # Continuous transition A_c = -exp(log_alpha) (1, d_state)
        a_cont = -torch.exp(log_alpha)

        # Precompute input-dependent projections
        delta = F.softplus(proj_delta(x))   # (B, T, d_state)
        b_t = proj_b(x)                     # (B, T, d_state)
        c_t = proj_c(x)                     # (B, T, d_state)

        # Discretize: A_bar = exp(delta * A_c), B_bar = delta * B_t
        a_bar = torch.exp(delta * a_cont.unsqueeze(0))  # (B, T, d_state)
        b_bar = delta * b_t                            # (B, T, d_state)

        h_state = torch.zeros(b_size, self.d_state, device=device, dtype=x.dtype)
        y_steps = []

        step_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        for t in step_indices:
            # Scalar channel modulation: take mean across d_model for state input
            x_step_input = x[:, t, :].mean(dim=-1, keepdim=True)
            # h_t = A_bar * h_{t-1} + B_bar * x_t
            h_state = a_bar[:, t, :] * h_state + b_bar[:, t, :] * x_step_input
            # y_t = C_t * h_t
            y_state = h_state * c_t[:, t, :]  # (B, d_state)
            y_steps.append(y_state)

        if reverse:
            y_steps = y_steps[::-1]

        y_stacked = torch.stack(y_steps, dim=1)  # (B, T, d_state)
        # Tile or project to d_model
        if self.d_state != self.d_model:
            # Linear expansion to d_model
            rep_factor = math.ceil(self.d_model / self.d_state)
            y_out = y_stacked.repeat(1, 1, rep_factor)[:, :, :self.d_model]
        else:
            y_out = y_stacked

        return y_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for selective temporal SSM.
        
        Args:
            x: Input tensor of shape (B, T, N, D_model)
        Returns:
            Output tensor of shape (B, T, N, D_model)
        """
        b, t, n, d = x.shape
        # Flatten spatial nodes into batch: (B * N, T, D)
        x_flat = x.permute(0, 2, 1, 3).reshape(b * n, t, d)

        # Forward selective scan
        y_fwd = self._selective_scan(
            x_flat, self.log_alpha_fwd, self.proj_delta_fwd, self.proj_b_fwd, self.proj_c_fwd, reverse=False
        )

        if self.bidirectional:
            # Backward selective scan
            y_bwd = self._selective_scan(
                x_flat, self.log_alpha_bwd, self.proj_delta_bwd, self.proj_b_bwd, self.proj_c_bwd, reverse=True
            )
            y_combined = torch.cat([y_fwd, y_bwd], dim=-1)
        else:
            y_combined = y_fwd

        # Project back to d_model and add skip connection
        out = self.out_proj(y_combined) + x_flat * self.d_skip
        out = self.dropout(out)
        out = self.norm(out)

        # Reshape back to (B, T, N, D_model)
        return out.reshape(b, n, t, d).permute(0, 2, 1, 3)

class GridForecastingHead(nn.Module):
    """Dual forecasting head: continuous renewable MW injection and curtailment probability."""
    def __init__(self, d_model: int, out_features: int = 2):
        super().__init__()
        # Continuous injection head (Solar and Wind MW)
        self.injection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_features),
            nn.Softplus()  # Enforces non-negative MW injection
        )
        # Discrete curtailment bottleneck classification head
        self.curtailment_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Probability in [0, 1]
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps hidden representations to dispatch outputs.
        
        Args:
            h: Hidden state of shape (B, T, N, D_model)
        Returns:
            injection_mw: (B, T, N, 2)
            curtailment_prob: (B, T, N, 1)
        """
        injection = self.injection_head(h)
        curtailment = self.curtailment_head(h)
        return injection, curtailment

class GridSpatioTemporalSSM(nn.Module):
    """Spatio-Temporal Selective State Space Model for Transmission Grid Dispatch."""
    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 2,
        d_model: int = 64,
        d_state: int = 16,
        num_spatial_hops: int = 2,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.05
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.num_layers = num_layers

        self.spatial_layers = nn.ModuleList([
            GraphSpatialDiffusion(d_model, d_model, num_hops=num_spatial_hops)
            for _ in range(num_layers)
        ])
        self.temporal_layers = nn.ModuleList([
            SelectiveSSMBlock(d_model, d_state=d_state, bidirectional=bidirectional, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.head = GridForecastingHead(d_model, out_features=out_features)

    def forward(
        self,
        x: torch.Tensor,
        adj_norm: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass for 15-minute dispatch forecasting.
        
        Args:
            x: Input tensor of shape (B, T, N, in_features)
            adj_norm: Normalized graph adjacency matrix of shape (N, N)
        Returns:
            Dictionary containing:
                "injection_mw": Predicted renewable injections (B, T, N, 2)
                "curtailment_prob": Predicted bottleneck probability (B, T, N, 1)
        """
        h = self.in_proj(x)

        for s_layer, t_layer in zip(self.spatial_layers, self.temporal_layers):
            # 1. Spatial diffusion across transmission corridors with residual
            h = h + s_layer(h, adj_norm)
            # 2. Selective temporal state space recurrence with residual
            h = h + t_layer(h)

        inj_pred, curt_pred = self.head(h)

        return {
            "injection_mw": inj_pred,
            "curtailment_prob": curt_pred
        }
