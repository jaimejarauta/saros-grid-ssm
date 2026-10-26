# Note 01: Spatio-Temporal Continuous SSMs for Electricity Transmission Networks

**Author:** Jaime Jarauta  
**Status:** Pre-registered active theory note  
**Target:** `src/saros_grid_ssm/model.py`

---

## 1. Physical Motivation & Mathematical Formulation

The Spanish high-voltage electricity transmission grid (operated by Red Eléctrica de España, REE) operates under 15-minute dispatch and settlement intervals ($T = 96$ dispatch steps per day). With high penetration of utility-scale solar PV in southern and central Spain (Andalusia, Extremadura, Castile-La Mancha) and wind farms in the Ebro valley and northwest (Aragon, Castile and León, Galicia), power flows are subject to localized transmission bottlenecks and renewable curtailment.

### 1.1 Non-Euclidean Grid Topology & Graph Spatial Diffusion

Let $G = (V, E, W)$ represent the transmission grid graph with $N = 50$ major transmission nodes. The physical line parameters (susceptance and conductance) determine the grid adjacency matrix $A \in \mathbb{R}^{N \times N}$, where $A_{ij} > 0$ if a $400\text{ kV}$ or $220\text{ kV}$ corridor connects node $i$ and node $j$.

To avoid high computational overhead and instability, we employ symmetric normalized spatial diffusion:
$$\tilde{A} = \tilde{D}^{-1/2} (A + I_N) \tilde{D}^{-1/2}, \quad \tilde{D}_{ii} = 1 + \sum_{j} A_{ij}$$

For nodal feature tensor $X_t \in \mathbb{R}^{B \times N \times D_{\text{in}}}$, localized $K$-hop spatial diffusion is computed via finite polynomial diffusion filters:
$$H_t^{(s)} = \sum_{k=0}^{K} \tilde{A}^k X_t W_k$$
where $W_k \in \mathbb{R}^{D_{\text{in}} \times D_{\text{model}}}$ are learnable diffusion filter weights. This operator models physical nodal power flow redistribution across adjacent transmission lines in $O(K \cdot |E| \cdot D)$ operations without requiring $O(N^2)$ dense cross-attention.

---

## 2. Selective Continuous State Space Temporal Dynamics

Electricity generation profiles exhibit multi-scale diurnal oscillations (24-hour solar cycle, morning and evening load peaks, nocturnal wind surges). Standard recurrent models suffer from gradient decay, while Transformers scale quadratically with sequence length $O(T^2)$.

We define the continuous-time state space dynamics for each node latent vector $x(t) \in \mathbb{R}^D$:
$$\frac{dh(t)}{dt} = A_c h(t) + B_c x(t)$$
$$y(t) = C_c h(t) + D x(t)$$

### 2.1 Diagonal Continuous Transition & Input-Dependent Selection

To ensure numerical stability and bounded state growth, $A_c$ is parameterized as a diagonal matrix with strictly negative real components:
$$A_c = -\exp(\alpha), \quad \alpha \in \mathbb{R}^{D_{\text{state}}}$$
guaranteeing continuous BIBO stability: $\text{Re}(\lambda_i(A_c)) < 0$.

Under input-dependent selection (Mamba/S4D formulation), the sampling step size $\Delta_t \in \mathbb{R}_{>0}$ and projection operators $B_t \in \mathbb{R}^{D_{\text{state}}}$, $C_t \in \mathbb{R}^{D_{\text{state}}}$ are computed dynamically from the spatially diffused representations:
$$\Delta_t = \text{softplus}(W_\Delta H_t^{(s)} + b_\Delta)$$
$$B_t = W_B H_t^{(s)}, \quad C_t = W_C H_t^{(s)}$$

Discretization via Zero-Order Hold (ZOH) yields:
$$\bar{A}_t = \exp(\Delta_t A_c) = \exp(-\Delta_t \cdot \exp(\alpha))$$
$$\bar{B}_t = (\Delta_t A_c)^{-1} (\exp(\Delta_t A_c) - I) \cdot (\Delta_t B_t) \approx \Delta_t B_t$$

State recurrence over time steps $t = 1, \dots, T$:
$$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot x_t$$
$$y_t = \sum_{d} C_{t, d} \cdot h_{t, d} + D x_t$$

### 2.2 Bidirectional Day-Ahead Conditioning

For day-ahead dispatch planning where the entire 96-timestep diurnal schedule is predicted simultaneously, we employ bidirectional selective SSM layers:
1. Forward recurrence: encodes causal generation history and solar ramp-up.
2. Backward recurrence: propagates nocturnal load commitments and cooling demands.
3. Dual forecasting heads:
   - Nodal injection $\hat{Y}_{\text{MW}} \in \mathbb{R}^{B \times T \times N \times 2}$ (predicted Solar and Wind MW).
   - Curtailment risk $\hat{P}_{\text{curtail}} \in \mathbb{R}^{B \times T \times N \times 1}$ with Sigmoid activation.

---

## 3. Invariants & Pre-Registered Criteria

1. **Spectral Stability Invariant:** Real eigenvalues of continuous transition matrix satisfy $\text{Re}(\lambda_i) \le -\epsilon < 0$, guaranteeing non-exploding state propagation across unbounded rollouts.
2. **Sub-Quadratic Complexity:** Total computational cost scales as $O(B \cdot T \cdot (K |E| + N D_{\text{state}}))$, strictly linear in time horizon $T$ and transmission nodes $N$.
3. **Target wMAPE:** Day-ahead renewable injection error across 50 nodes satisfies $\text{wMAPE} < 6.5\%$.
4. **Curtailment Detection:** Classification performance on transmission bottleneck curtailment events achieves $F_1 \ge 0.75$.

## Run log 2026-10-26: forecasting setup

Before: same-step targets, bidirectional SSM, isolated nodes. Copying the inputs scored 0.98% wMAPE vs the model's
1.83%, and the pointwise rule labelled curtailment exactly, so neither the graph nor the SSM could matter.

Changes: targets at t+4 (1 h) from inputs <= t (`grid.horizon`), unidirectional SSM, signed neighbour coupling in
the generator (`coupling=0.5`: neighbours' net balance absorbs or congests exports), baselines in the gate
(copy: potentials and local rule at t; yesterday: observed value at t+H-96).

| Arm (seed 2026, 300 steps, CPU) | wMAPE | Curtailment F1 |
|---|---|---|
| Copy baseline | 18.59% | 0.557 |
| Yesterday baseline | 16.33% | 0.809 |
| SSM, 0 graph hops | 4.48% | 0.879 |
| SSM, 2 graph hops | 3.94% | 0.936 |

Rule: pass if wMAPE < 6.5%, F1 >= 0.75, and both beat the best baseline. Status: pass, one seed (exploratory).
Caveat: data is still synthetic; the next step is real ESIOS data with the same horizon and baselines.
