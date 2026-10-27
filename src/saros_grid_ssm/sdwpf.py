"""SDWPF (Baidu KDD Cup 2022) wind power data: 134 turbines, 10-min SCADA, 245 days, turbine coordinates.
Mirror: huggingface.co/datasets/aigrids/WindFarm_raw (licence not stated on the mirror; see the Scientific Data
article, doi:10.1038/s41597-024-03427-5). Validity rules follow the official KDD Cup evaluation.
"""

import os
import tarfile
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

STEPS_PER_DAY = 144
CAPACITY_KW = 1500.0  # rated turbine power; targets are scaled to [0, ~1]
HORIZON_NAMES = {6: "1h", 36: "6h", 144: "24h"}
FEATURES = ["Wspd", "Wdir_sin", "Wdir_cos", "Etmp", "Itmp", "Ndir_sin", "Ndir_cos", "Pab", "Prtv", "Patv", "tod_sin", "tod_cos"]


def download(cache_dir: str) -> Tuple[str, str]:
    """Paths of the 245-day SCADA csv and the location csv (downloads and extracts once)."""
    data = os.path.join(cache_dir, "sdwpf_kddcup", "sdwpf_245days_v1.csv")
    loc = os.path.join(cache_dir, "sdwpf_kddcup", "sdwpf_baidukddcup2022_turb_location.csv")
    if not os.path.exists(data):
        from huggingface_hub import hf_hub_download
        tar = hf_hub_download("aigrids/WindFarm_raw", "sdwpf_kddcup.tar.gz", repo_type="dataset", local_dir=cache_dir)
        with tarfile.open(tar) as tf:
            tf.extractall(cache_dir, members=[m for m in tf.getmembers() if m.name.endswith(".csv")])
    return data, loc


def build_tensors(df) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x (T, N, F) unnormalized features, y (T, N) Patv / capacity, valid (T, N) official evaluation mask)."""
    df = df.sort_values(["TurbID", "Day", "Tmstamp"])
    n_turb = df["TurbID"].nunique()
    T = len(df) // n_turb
    raw = {c: df[c].to_numpy(dtype=np.float64).reshape(n_turb, T).T for c in
           ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]}
    pab = np.maximum(np.maximum(raw["Pab1"], raw["Pab2"]), raw["Pab3"])
    missing = np.isnan(np.stack(list(raw.values()))).any(0)
    invalid = (missing | ((raw["Patv"] <= 0) & (raw["Wspd"] > 2.5)) | (pab > 89)
               | (np.abs(raw["Wdir"]) > 180) | (np.abs(raw["Ndir"]) > 720))

    def ffill(a):  # forward fill along time, then zeros for leading gaps
        out = a.copy()
        for t in range(1, out.shape[0]):
            out[t] = np.where(np.isnan(out[t]), out[t - 1], out[t])
        return np.nan_to_num(out)

    r = {k: ffill(v) for k, v in raw.items()}
    pab = ffill(pab)
    tod = 2 * np.pi * (np.arange(T) % STEPS_PER_DAY) / STEPS_PER_DAY
    tod = np.repeat(tod[:, None], n_turb, 1)
    feats = [r["Wspd"], np.sin(np.radians(r["Wdir"])), np.cos(np.radians(r["Wdir"])), r["Etmp"], r["Itmp"],
             np.sin(np.radians(r["Ndir"])), np.cos(np.radians(r["Ndir"])), pab, r["Prtv"],
             np.clip(r["Patv"], 0, None) / CAPACITY_KW, np.sin(tod), np.cos(tod)]
    x = torch.tensor(np.stack(feats, -1), dtype=torch.float32)
    y = torch.tensor(np.clip(r["Patv"], 0, None) / CAPACITY_KW, dtype=torch.float32)
    return x, y, torch.tensor(~invalid)


def knn_adjacency(loc, k: int = 8) -> np.ndarray:
    """Symmetric 0/1 adjacency linking each turbine to its k nearest neighbours by (x, y)."""
    xy = loc.sort_values("TurbID")[["x", "y"]].to_numpy(dtype=np.float64)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    a = np.zeros_like(d, dtype=np.float32)
    for i, nbrs in enumerate(np.argsort(d, axis=1)[:, :k]):
        a[i, nbrs] = 1.0
    return np.maximum(a, a.T)


class ForecastWindows(Dataset):
    """Inputs x[s : s+W]; targets y at s+W-1+h for each horizon h (strictly after the last input step)."""

    def __init__(self, x, y, valid, window: int, horizons: Sequence[int], starts: Iterable[int]):
        self.x, self.y, self.valid, self.window, self.horizons = x, y, valid, window, list(horizons)
        last = x.size(0) - window - max(self.horizons)
        self.starts = [s for s in starts if 0 <= s <= last]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s, W = self.starts[i], self.window
        idx = [s + W - 1 + h for h in self.horizons]
        return self.x[s:s + W], self.y[idx].T, self.valid[idx].T, s


def masked_scores(pred, y, valid, scale: float = CAPACITY_KW, horizons: Sequence[int] = (6, 36, 144)) -> Dict:
    """Per-horizon MAE / RMSE in kW over valid points, plus the KDD score (MAE + RMSE) / 2. pred, y: (..., H)."""
    out = {}
    for j, h in enumerate(list(horizons)[:pred.size(-1)]):
        m = valid[..., j]
        err = (pred[..., j] - y[..., j])[m] * scale
        mae, rmse = err.abs().mean().item(), err.pow(2).mean().sqrt().item()
        out[HORIZON_NAMES.get(h, f"h{h}")] = {"mae": mae, "rmse": rmse, "score": (mae + rmse) / 2}
    return out


def paired_bootstrap(err_a: torch.Tensor, err_b: torch.Tensor, n: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """95% CI of mean(err_b - err_a) over paired items (e.g. test windows); > 0 means a is better."""
    g = torch.Generator().manual_seed(seed)
    d = (err_b - err_a).double()
    idx = torch.randint(0, d.numel(), (n, d.numel()), generator=g)
    means = d[idx].mean(1)
    return means.quantile(0.025).item(), means.quantile(0.975).item()


def split_starts(T: int, window: int, horizons: Sequence[int], d_tr: int, d_va: int, train_stride: int = 6,
                 eval_stride: int = 36) -> Tuple[range, range, range]:
    """Window starts whose every target lies inside its split (train days < d_tr <= val days < d_va <= test).
    Inputs may reach back into the previous split; targets never cross a boundary."""
    span, lo = window + max(horizons), window - 1 + min(horizons)
    tr = range(0, d_tr * STEPS_PER_DAY - span + 1, train_stride)
    va = range(d_tr * STEPS_PER_DAY - lo, d_va * STEPS_PER_DAY - span + 1, eval_stride)
    te = range(d_va * STEPS_PER_DAY - lo, T - span + 1, eval_stride)
    return tr, va, te


def climatology(y: torch.Tensor, valid: torch.Tensor, t_end: int) -> torch.Tensor:
    """(STEPS_PER_DAY, N) mean of valid y per time of day over steps < t_end; the honest no-skill 24 h forecast."""
    tod = torch.arange(t_end) % STEPS_PER_DAY
    w = valid[:t_end].float()
    num = torch.zeros(STEPS_PER_DAY, y.size(1)).index_add_(0, tod, y[:t_end] * w)
    den = torch.zeros(STEPS_PER_DAY, y.size(1)).index_add_(0, tod, w)
    return num / den.clamp_min(1)


def day_means(err: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """Mean of per-window errors within each calendar day of `steps`; overlapping windows are not independent,
    so days are the bootstrap unit."""
    _, inv = (torch.as_tensor(steps) // STEPS_PER_DAY).unique(return_inverse=True)
    return torch.zeros(int(inv.max()) + 1).index_add_(0, inv, err.float()) / torch.bincount(inv).float()
