"""Unit tests for GridSpatioTemporalSSM architecture, spatial diffusion, and selective recurrence."""

import torch
import torch.nn as nn
from saros_grid_ssm.model import (
    GraphSpatialDiffusion,
    SelectiveSSMBlock,
    GridForecastingHead,
    GridSpatioTemporalSSM,
)
from saros_grid_ssm.ree_data import build_spanish_grid_adjacency, normalize_adjacency

def test_graph_spatial_diffusion_shape():
    """Verify K-hop graph spatial diffusion preserves shape and propagates across nodes."""
    torch.manual_seed(42)
    b, t, n, d_in, d_out = 2, 8, 10, 4, 16
    adj = build_spanish_grid_adjacency(num_nodes=n)
    adj_norm = torch.from_numpy(normalize_adjacency(adj)).float()

    diff_layer = GraphSpatialDiffusion(d_in=d_in, d_out=d_out, num_hops=2)
    x = torch.randn(b, t, n, d_in)

    out = diff_layer(x, adj_norm)
    assert out.shape == (b, t, n, d_out), f"Expected shape {(b, t, n, d_out)}, got {out.shape}"
    assert torch.isfinite(out).all(), "Non-finite outputs detected in spatial diffusion"

def test_selective_ssm_recurrence():
    """Verify continuous-time selective SSM runs forward and backward scans stably."""
    torch.manual_seed(42)
    b, t, n, d_model = 2, 16, 5, 32
    ssm_block = SelectiveSSMBlock(d_model=d_model, d_state=8, bidirectional=True)
    x = torch.randn(b, t, n, d_model)

    out = ssm_block(x)
    assert out.shape == (b, t, n, d_model), f"Expected shape {(b, t, n, d_model)}, got {out.shape}"
    assert torch.isfinite(out).all(), "Non-finite outputs detected in selective SSM"

def test_forecasting_head_bounds():
    """Verify output heads produce non-negative injection MW and [0, 1] curtailment probabilities."""
    torch.manual_seed(42)
    b, t, n, d_model = 2, 8, 5, 32
    head = GridForecastingHead(d_model=d_model, out_features=2)
    h = torch.randn(b, t, n, d_model)

    injection, curtailment = head(h)
    assert injection.shape == (b, t, n, 2)
    assert curtailment.shape == (b, t, n, 1)
    # Non-negative MW injection via Softplus
    assert (injection >= 0.0).all(), "Negative MW injection predicted"
    # Probability in [0, 1] via Sigmoid
    assert (curtailment >= 0.0).all() and (curtailment <= 1.0).all(), "Curtailment probability out of [0, 1]"

def test_full_grid_ssm_forward_and_backward():
    """End-to-end forward and backward gradient flow check."""
    torch.manual_seed(42)
    b, t, n = 2, 12, 6
    adj = build_spanish_grid_adjacency(num_nodes=n)
    adj_norm = torch.from_numpy(normalize_adjacency(adj)).float()

    model = GridSpatioTemporalSSM(
        in_features=4,
        out_features=2,
        d_model=16,
        d_state=8,
        num_spatial_hops=1,
        num_layers=1,
        bidirectional=True
    )
    x = torch.randn(b, t, n, 4)
    preds = model(x, adj_norm)

    assert "injection_mw" in preds and "curtailment_prob" in preds
    inj = preds["injection_mw"]
    curt = preds["curtailment_prob"]
    assert inj.shape == (b, t, n, 2)
    assert curt.shape == (b, t, n, 1)

    loss = inj.sum() + curt.sum()
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"

def test_bibo_stability_eigenvalues():
    """Verify that continuous diagonal transition elements have strictly negative real parts."""
    model = GridSpatioTemporalSSM(in_features=4, out_features=2, d_model=16, d_state=8)
    for layer in model.temporal_layers:
        a_fwd = -torch.exp(layer.log_alpha_fwd)
        assert (a_fwd < 0.0).all(), "BIBO stability violation: continuous A has non-negative elements"
        if layer.bidirectional:
            a_bwd = -torch.exp(layer.log_alpha_bwd)
            assert (a_bwd < 0.0).all(), "BIBO stability violation in backward scan"

if __name__ == "__main__":
    test_graph_spatial_diffusion_shape()
    test_selective_ssm_recurrence()
    test_forecasting_head_bounds()
    test_full_grid_ssm_forward_and_backward()
    test_bibo_stability_eigenvalues()
    print("All test_model.py tests passed successfully.")
