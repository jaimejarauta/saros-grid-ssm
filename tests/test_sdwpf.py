"""Tests for the SDWPF (KDD Cup 2022) loader: cleaning mask, tensors, kNN graph, windows, masked metrics."""

import numpy as np
import pandas as pd
import pytest
import torch

from saros_grid_ssm.sdwpf import build_tensors, knn_adjacency, ForecastWindows, masked_scores


def fake_frames(n_turb=4, n_days=3):
    rows = []
    rng = np.random.RandomState(0)
    for t in range(1, n_turb + 1):
        for d in range(1, n_days + 1):
            for s in range(144):
                rows.append({"TurbID": t, "Day": d, "Tmstamp": f"{s // 6:02d}:{(s % 6) * 10:02d}",
                             "Wspd": 5.0 + rng.rand(), "Wdir": 0.0, "Etmp": 20.0, "Itmp": 30.0, "Ndir": 10.0,
                             "Pab1": 1.0, "Pab2": 1.0, "Pab3": 1.0, "Prtv": 0.0, "Patv": 500.0 + t})
    data = pd.DataFrame(rows)
    data.loc[0, "Patv"] = np.nan                      # missing -> masked
    data.loc[1, ["Patv", "Wspd"]] = [-5.0, 6.0]       # Patv <= 0 with Wspd > 2.5 -> masked
    data.loc[2, "Pab1"] = 95.0                        # pitch > 89 -> masked
    loc = pd.DataFrame({"TurbID": range(1, n_turb + 1), "x": [0, 1, 10, 11], "y": [0, 0, 0, 0]})
    return data, loc


def test_build_tensors_shapes_and_mask():
    data, loc = fake_frames()
    x, y, valid = build_tensors(data)
    T, N = 3 * 144, 4
    assert x.shape[:2] == (T, N) and y.shape == (T, N) and valid.shape == (T, N)
    assert not valid[0, 0] and not valid[1, 0] and not valid[2, 0] and valid[3, 0]
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    assert (y >= 0).all()


def test_knn_adjacency_symmetric_no_self():
    _, loc = fake_frames()
    a = knn_adjacency(loc, k=1)
    assert np.allclose(a, a.T) and np.all(np.diag(a) == 0)
    assert a[0, 1] == 1 and a[2, 3] == 1 and a[0, 2] == 0


def test_windows_are_causal():
    data, _ = fake_frames()
    x, y, valid = build_tensors(data)
    ds = ForecastWindows(x, y, valid, window=36, horizons=(6, 36), starts=range(0, 300, 7))
    xi, yi, vi, _ = ds[0]
    assert xi.shape == (36, 4, x.size(-1)) and yi.shape == (4, 2)
    assert torch.equal(yi[:, 0], y[36 - 1 + 6]) and torch.equal(yi[:, 1], y[36 - 1 + 36])
    assert all(s + 36 - 1 + 36 < x.size(0) for s in ds.starts)


def test_masked_scores():
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pred = torch.tensor([[2.0, 2.0], [3.0, 100.0]])
    valid = torch.tensor([[True, True], [True, False]])
    s = masked_scores(pred[..., None], y[..., None], valid[..., None], scale=1.0)
    name = list(s)[0]
    assert s[name]["mae"] == pytest.approx(1 / 3) and s[name]["rmse"] == pytest.approx((1 / 3) ** 0.5)


def test_paired_bootstrap_ci():
    from saros_grid_ssm.sdwpf import paired_bootstrap
    a = torch.linspace(1.0, 2.0, 200)
    lo, hi = paired_bootstrap(a, a + 0.5, n=500, seed=0)  # b worse by exactly 0.5
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    noise = torch.randn(200, generator=torch.Generator().manual_seed(0))
    lo, hi = paired_bootstrap(a, a + noise, n=500, seed=0)
    assert lo < 0 < hi
