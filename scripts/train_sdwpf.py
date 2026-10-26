#!/usr/bin/env python3
"""SDWPF study (Colab run 2): spatio-temporal SSM vs persistence, seasonal-naive, LightGBM and GRU on real wind power.
Horizons 1 h / 6 h / 24 h from a 24 h input window; split by day (train 1-196, val 197-220, test 221-245).
Each arm/seed writes its own result file, so a disconnect only repeats the arm in progress.
Gate (2026-10-26): SSM-2hop beats LightGBM and GRU on the KDD score at >= 2 of 3 horizons (mean over seeds,
paired bootstrap CI over test windows excludes 0) and beats SSM-0hop; otherwise the graph claim is dropped.
Graph WaveNet from the plan is not implemented; GRU is the learned no-graph baseline.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from saros_grid_ssm.model import GridSpatioTemporalSSM
from saros_grid_ssm.ree_data import normalize_adjacency
from saros_grid_ssm.runtime import Run, file_digest
from saros_grid_ssm.sdwpf import (STEPS_PER_DAY, ForecastWindows, build_tensors, download, knn_adjacency,
                                  masked_scores, paired_bootstrap)

PATV = 9  # index of scaled Patv in the feature vector


class GRUForecaster(nn.Module):
    """Per-turbine GRU over the window, no graph."""

    def __init__(self, f_in, n_h, d=64):
        super().__init__()
        self.gru = nn.GRU(f_in, d, num_layers=2, batch_first=True)
        self.head = nn.Linear(d, n_h)

    def forward(self, x, _adj=None):
        B, W, N, Fd = x.shape
        h, _ = self.gru(x.permute(0, 2, 1, 3).reshape(B * N, W, Fd))
        return F.softplus(self.head(h[:, -1])).view(B, N, -1)


class SSMForecaster(nn.Module):
    """GridSpatioTemporalSSM, read out at the last input step; one output per horizon."""

    def __init__(self, f_in, n_h, hops, d=64):
        super().__init__()
        self.net = GridSpatioTemporalSSM(in_features=f_in, out_features=n_h, d_model=d, d_state=16,
                                         num_spatial_hops=hops, num_layers=2, bidirectional=False)

    def forward(self, x, adj):
        return self.net(x, adj)["injection_mw"][:, -1]


def predict(model, ds, adj, device, bs=16):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _, _, _ in DataLoader(ds, batch_size=bs):
            preds.append(model(xb.to(device), adj).float().cpu())
    model.train()
    return torch.cat(preds)


def stack_targets(ds):
    ys, vs = zip(*[(ds[i][1], ds[i][2]) for i in range(len(ds))])
    return torch.stack(ys), torch.stack(vs)


def train_neural(model, train_ds, val_ds, adj, a, device, run, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=True)
    y_val, v_val = stack_targets(val_ds)
    best, best_state, step, t0 = float("inf"), None, 0, time.time()
    while step < a.steps:
        for xb, yb, vb, _ in loader:
            xb, yb, vb = xb.to(device), yb.to(device), vb.to(device)
            loss = (F.smooth_l1_loss(model(xb, adj), yb, reduction="none", beta=0.05) * vb).sum() / vb.sum().clamp_min(1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
            if step % a.eval_every == 0 or step == a.steps:
                s = masked_scores(predict(model, val_ds, adj, device), y_val, v_val)
                val = np.mean([v["score"] for v in s.values()])
                if val < best:
                    best, best_state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
                print(f"  [{tag}] step {step} loss {loss.item():.4f} val score {val:.1f} kW (best {best:.1f}) "
                      f"{time.time() - t0:.0f}s")
            if step == 100:
                run.log_throughput(step, a.steps, "steps")
            if step >= a.steps:
                break
    model.load_state_dict(best_state)
    return model


def lightgbm_preds(x, y, valid, train_starts, test_ds, horizons, W):
    """One LightGBM per horizon on per-turbine lag features (no graph); None if lightgbm is missing."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("lightgbm not installed; skipping that arm")
        return None
    lags = [0, 1, 2, 3, 6, 12, 36, 72, 143]

    def feats(t_end):  # (N, F) features at input end step t_end
        cols = [x[t_end - l, :, PATV] for l in lags] + [x[t_end - l, :, 0] for l in (0, 1, 6, 36)] + \
               [x[t_end, :, -2], x[t_end, :, -1], x[t_end - 35:t_end + 1, :, PATV].mean(0)]
        return torch.stack(cols, -1).numpy()

    rng = np.random.RandomState(0)
    tr = rng.choice(np.array(list(train_starts)), size=min(4000, len(train_starts)), replace=False)
    X_tr = np.concatenate([feats(s + W - 1) for s in tr])
    X_te = np.stack([feats(s + W - 1) for s in test_ds.starts])  # (Nw, N, F)
    out = []
    for h in horizons:
        Y = np.concatenate([y[s + W - 1 + h].numpy() for s in tr])
        M = np.concatenate([valid[s + W - 1 + h].numpy() for s in tr])
        params = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "bagging_fraction": 0.8,
                  "bagging_freq": 1, "feature_fraction": 0.8, "verbose": -1, "seed": 0}
        m = lgb.train(params, lgb.Dataset(X_tr[M], Y[M]), num_boost_round=600)
        out.append(m.predict(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape[0], X_te.shape[1]))
    return torch.tensor(np.clip(np.stack(out, -1), 0, None), dtype=torch.float32)


def per_window_mae(pred, y, valid):
    """Mean abs error (kW) per test window over valid points and horizons; for paired comparisons."""
    err = (pred - y).abs() * 1500.0 * valid
    return err.sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arms", nargs="+", default=["persistence", "seasonal", "lightgbm", "gru", "ssm0", "ssm2"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--window", type=int, default=144)
    p.add_argument("--horizons", type=int, nargs="+", default=[6, 36, 144])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--train-stride", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-days", type=int, default=None, help="smoke tests only")
    p.add_argument("--budget-hours", type=float, default=6.0)
    a = p.parse_args()

    data_csv, loc_csv = download(a.data_dir)
    run = Run(a.out_dir, vars(a), budget_hours=a.budget_hours)
    run.data_digests["sdwpf_245days_v1.csv"] = file_digest(data_csv)
    df = pd.read_csv(data_csv)
    if a.max_days:
        df = df[df["Day"] <= a.max_days]
    x, y, valid = build_tensors(df)
    adj = torch.tensor(normalize_adjacency(knn_adjacency(pd.read_csv(loc_csv, encoding="utf-8-sig"), k=8)),
                       device=a.device)
    days = x.size(0) // STEPS_PER_DAY
    d_tr, d_va = int(days * 0.8), int(days * 0.9)
    mean = x[:d_tr * STEPS_PER_DAY].mean((0, 1))
    std = x[:d_tr * STEPS_PER_DAY].std((0, 1)).clamp_min(1e-6)
    keep = torch.zeros(x.size(-1), dtype=torch.bool); keep[PATV] = True  # Patv stays in capacity units
    xn = torch.where(keep, x, (x - mean) / std)

    W, H = a.window, a.horizons
    span = W + max(H)
    tr_starts = range(0, d_tr * STEPS_PER_DAY - span, a.train_stride)
    va_starts = range(d_tr * STEPS_PER_DAY - W, d_va * STEPS_PER_DAY - span + W, 36)
    te_starts = range(d_va * STEPS_PER_DAY - W, x.size(0) - span, 36)  # inputs may reach back into val
    train_ds = ForecastWindows(xn, y, valid, W, H, tr_starts)
    val_ds = ForecastWindows(xn, y, valid, W, H, va_starts)
    test_ds = ForecastWindows(xn, y, valid, W, H, te_starts)
    y_te, v_te = stack_targets(test_ds)
    print(f"windows: train {len(train_ds)} val {len(val_ds)} test {len(test_ds)} | turbines {x.size(1)}")

    results, preds = {}, {}
    for arm in a.arms:
        for seed in (a.seeds if arm in ("gru", "ssm0", "ssm2") else [0]):
            tag = f"{arm}_s{seed}"
            path = os.path.join(a.out_dir, f"{tag}.pt")
            if os.path.exists(path):  # resume: arm already finished
                pr = torch.load(path)
            else:
                torch.manual_seed(seed)
                if arm == "persistence":
                    pr = torch.stack([y[s + W - 1].repeat(len(H), 1).T for s in test_ds.starts])
                elif arm == "seasonal":
                    pr = torch.stack([torch.stack([y[s + W - 1 + h - STEPS_PER_DAY] for h in H], -1) for s in test_ds.starts])
                elif arm == "lightgbm":
                    pr = lightgbm_preds(x, y, valid, tr_starts, test_ds, H, W)
                    if pr is None:
                        continue
                else:
                    model = (GRUForecaster(x.size(-1), len(H)) if arm == "gru"
                             else SSMForecaster(x.size(-1), len(H), hops=int(arm[-1]))).to(a.device)
                    model = train_neural(model, train_ds, val_ds, adj, a, a.device, run, tag)
                    pr = predict(model, test_ds, adj, a.device)
                torch.save(pr, path)
            preds[tag] = pr
            results[tag] = masked_scores(pr, y_te, v_te, horizons=H)
            print(f"{tag:<16} " + " | ".join(f"{k} {v['score']:.1f}" for k, v in results[tag].items()))
            run.check_budget(len(results), len(a.arms) * len(a.seeds))

    def mean_over_seeds(arm):
        tags = [t for t in results if t.startswith(arm + "_")]
        return {h: np.mean([results[t][h]["score"] for t in tags]) for h in results[tags[0]]} if tags else None

    summary = {arm: mean_over_seeds(arm) for arm in a.arms if mean_over_seeds(arm)}
    gate = None
    if all(k in summary for k in ("ssm2", "ssm0", "gru")):
        rivals = [r for r in ("lightgbm", "gru") if r in summary]
        wins = sum(all(summary["ssm2"][h] < summary[r][h] for r in rivals) for h in summary["ssm2"])
        ssm2 = torch.stack([per_window_mae(preds[t], y_te, v_te) for t in preds if t.startswith("ssm2_")]).mean(0)
        cis = {}
        for r in rivals + ["ssm0"]:
            other = torch.stack([per_window_mae(preds[t], y_te, v_te) for t in preds if t.startswith(r + "_")]).mean(0)
            cis[r] = paired_bootstrap(ssm2, other)
        graph_helps = all(summary["ssm2"][h] < summary["ssm0"][h] for h in summary["ssm2"]) and cis["ssm0"][0] > 0
        gate = {"horizon_wins_vs_all_rivals": wins, "ci_vs": cis, "graph_helps": graph_helps,
                "passed": wins >= 2 and all(cis[r][0] > 0 for r in rivals) and graph_helps}
        print(f"gate: {gate}")
    run.write_manifest({"per_run": results, "mean_over_seeds": summary, "gate": gate})


if __name__ == "__main__":
    main()
