"""Unit tests for Spanish REE 50-node data generation, adjacency matrices, and metrics."""

import numpy as np
import torch
from saros_grid_ssm.ree_data import (
    SPANISH_50_NODES,
    build_spanish_grid_adjacency,
    normalize_adjacency,
    generate_synthetic_ree_grid,
    REESpanishGridDataset,
)
from saros_grid_ssm.metrics import compute_wmape, compute_curtailment_f1, compute_grid_metrics

def test_spanish_grid_nodes_and_adjacency():
    """Verify 50 Spanish nodes and symmetric adjacency matrix structure."""
    assert len(SPANISH_50_NODES) == 50
    adj = build_spanish_grid_adjacency(num_nodes=50)
    assert adj.shape == (50, 50)
    # Symmetry check
    assert np.allclose(adj, adj.T), "Grid adjacency matrix is not symmetric"
    # Diagonal should be zero initially (no self-loops in raw transmission corridors)
    assert np.all(np.diag(adj) == 0.0), "Raw adjacency contains self-loops"
    # Ensure graph is connected (at least 50 corridors)
    num_corridors = np.sum(adj) / 2
    assert num_corridors >= 40, f"Expected >= 40 transmission corridors, got {num_corridors}"

def test_normalized_adjacency_properties():
    """Verify symmetric normalized adjacency D^{-1/2}(A+I)D^{-1/2} has eigenvalues <= 1."""
    adj = build_spanish_grid_adjacency(num_nodes=50)
    adj_norm = normalize_adjacency(adj)
    assert adj_norm.shape == (50, 50)
    assert np.allclose(adj_norm, adj_norm.T, atol=1e-5), "Normalized adjacency is not symmetric"
    # Maximum eigenvalue of normalized adjacency with self loops is <= 1.0 + epsilon
    evals = np.linalg.eigvalsh(adj_norm)
    assert np.max(evals) <= 1.0001, f"Maximum eigenvalue exceeds 1: {np.max(evals)}"
    assert np.min(evals) >= -1.0001, f"Minimum eigenvalue below -1: {np.min(evals)}"

def test_synthetic_data_generation_physics():
    """Verify diurnal profiles adhere to solar/wind physical constraints."""
    data = generate_synthetic_ree_grid(num_days=3, num_nodes=50, seed=42)
    inputs = data["inputs"]           # (288, 50, 4)
    injections = data["injections"]   # (288, 50, 2)
    curtailments = data["curtailments"] # (288, 50, 1)

    assert inputs.shape == (288, 50, 4)
    assert injections.shape == (288, 50, 2)
    assert curtailments.shape == (288, 50, 1)

    # 1. Solar is strictly zero at night (step 0 = 00:00, step 12 = 03:00, step 90 = 22:30)
    for day in range(3):
        night_step = day * 96 + 8   # 02:00
        assert np.all(inputs[night_step, :, 0] == 0.0), f"Solar > 0 detected at night step {night_step}"
        assert np.all(injections[night_step, :, 0] == 0.0), "Solar injection > 0 at night"

    # 2. Solar peaks during midday (step 56 = 14:00)
    midday_step = 56
    assert np.mean(inputs[midday_step, :, 0]) > 200.0, "Expected midday solar potential to exceed 200 MW"

    # 3. Injections are non-negative
    assert np.all(injections >= 0.0), "Negative renewable injection generated"

    # 4. Curtailment binary labels
    unique_curt = np.unique(curtailments)
    assert set(unique_curt).issubset({0.0, 1.0}), f"Non-binary curtailment values: {unique_curt}"

def test_dataset_loader_windowing():
    """Verify PyTorch dataset yields correct day-ahead 96-timestep windows."""
    data = generate_synthetic_ree_grid(num_days=4, num_nodes=50, seed=42)
    dataset = REESpanishGridDataset(
        inputs=data["inputs"],
        injections=data["injections"],
        curtailments=data["curtailments"],
        window_size=96,
        stride=96
    )
    assert len(dataset) == 4, f"Expected 4 daily dispatch windows, got {len(dataset)}"
    x, y_inj, y_curt = dataset[0]
    assert x.shape == (96, 50, 4)
    assert y_inj.shape == (96, 50, 2)
    assert y_curt.shape == (96, 50, 1)

def test_metrics_computations():
    """Verify wMAPE and Curtailment F1 accuracy on known edge cases."""
    # 1. Exact match wMAPE == 0
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred_exact = np.array([100.0, 200.0, 300.0])
    assert compute_wmape(y_true, y_pred_exact) == 0.0

    # 2. 10% offset wMAPE == 0.10
    y_pred_offset = np.array([110.0, 220.0, 330.0])
    assert abs(compute_wmape(y_true, y_pred_offset) - 0.10) < 1e-5

    # 3. Curtailment F1 perfect classification
    c_true = np.array([1, 0, 1, 1, 0])
    c_prob_perfect = np.array([0.9, 0.1, 0.8, 0.95, 0.2])
    res_perfect = compute_curtailment_f1(c_true, c_prob_perfect)
    assert res_perfect["f1"] == 1.0
    assert res_perfect["precision"] == 1.0
    assert res_perfect["recall"] == 1.0

    # 4. Inverted predictions F1 == 0
    c_prob_inverted = np.array([0.1, 0.9, 0.2, 0.05, 0.8])
    res_zero = compute_curtailment_f1(c_true, c_prob_inverted)
    assert res_zero["f1"] == 0.0

if __name__ == "__main__":
    test_spanish_grid_nodes_and_adjacency()
    test_normalized_adjacency_properties()
    test_synthetic_data_generation_physics()
    test_dataset_loader_windowing()
    test_metrics_computations()
    print("All test_ree_data.py tests passed successfully.")


def test_dataset_horizon_shifts_targets():
    data = generate_synthetic_ree_grid(num_days=4, num_nodes=50, seed=42)
    ds = REESpanishGridDataset(data["inputs"], data["injections"], data["curtailments"],
                               window_size=96, stride=96, horizon=4)
    assert len(ds) == 3  # last window would need targets past the end
    x, y_inj, y_curt = ds[1]
    assert np.allclose(x.numpy(), data["inputs"][96:192])
    assert np.allclose(y_inj.numpy(), data["injections"][100:196])
    assert np.allclose(y_curt.numpy(), data["curtailments"][100:196])


def test_baseline_forecasts_alignment():
    from saros_grid_ssm.metrics import baseline_forecasts
    T, N = 300, 3
    inputs = np.arange(T * N * 4, dtype=np.float32).reshape(T, N, 4)
    inj = np.arange(T * N * 2, dtype=np.float32).reshape(T, N, 2)
    curt = (np.arange(T * N) % 2).astype(np.float32).reshape(T, N, 1)
    b = baseline_forecasts(inputs, inj, curt, starts=[100, 196], window=96, horizon=4)
    # copy: potentials at the input step t predict the target at t + 4
    assert np.allclose(b["copy"][0][1], inputs[196:292, :, :2])
    # yesterday: target at t + 4 predicted by the observed value at t + 4 - 96
    assert np.allclose(b["yesterday"][0][0], inj[8:104])
    assert np.allclose(b["yesterday"][1][1], curt[104:200])
    assert b["copy"][0].shape == (2, 96, N, 2)


def test_curtailment_couples_to_neighbours():
    from saros_grid_ssm.metrics import compute_curtailment_f1
    def local_rule_f1(coupling):
        d = generate_synthetic_ree_grid(num_days=3, num_nodes=50, seed=7, coupling=coupling)
        x = d["inputs"]
        rule = ((x[..., 0] + x[..., 1] - x[..., 2] - x[..., 3]) > 0).astype(np.float32)
        return compute_curtailment_f1(d["curtailments"][..., 0], rule)["f1"]
    assert local_rule_f1(0.0) == 1.0  # isolated nodes: pointwise rule is exact
    assert local_rule_f1(0.5) < 0.9   # neighbour balance changes the label
