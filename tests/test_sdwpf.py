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


def test_split_targets_never_cross_boundaries():
    from saros_grid_ssm.sdwpf import split_starts
    W, H, d_tr, d_va, days = 144, (6, 36, 144), 8, 9, 10
    tr, va, te = split_starts(days * 144, W, H, d_tr, d_va, train_stride=1, eval_stride=1)
    tgt = lambda ss, h: [s + W - 1 + h for s in ss]
    assert max(tgt(tr, 144)) < d_tr * 144 <= min(tgt(va, 6))
    assert max(tgt(va, 144)) < d_va * 144 <= min(tgt(te, 6))
    assert max(tgt(te, 144)) < days * 144


def test_climatology_by_time_of_day():
    from saros_grid_ssm.sdwpf import climatology
    tod = torch.arange(3 * 144) % 144
    y = torch.stack([tod.float(), 2 * tod.float()], -1)  # (T, 2)
    valid = torch.ones_like(y, dtype=torch.bool)
    y[144 + 5, 0], valid[144 + 5, 0] = 1e6, False      # invalid outlier is ignored
    y[2 * 144:] = -1.0                                  # outside the training span
    c = climatology(y, valid, t_end=2 * 144)
    assert c.shape == (144, 2) and torch.allclose(c[:, 0], tod[:144].float()) and torch.allclose(c[:, 1], 2 * tod[:144].float())


def test_day_means():
    from saros_grid_ssm.sdwpf import day_means
    err = torch.tensor([1.0, 3.0, 10.0, 20.0, 30.0])
    steps = torch.tensor([0, 143, 144, 200, 287])
    assert torch.allclose(day_means(err, steps), torch.tensor([2.0, 20.0]))
