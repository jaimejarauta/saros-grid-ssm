"""Evaluation metrics for REE grid renewable injection forecasting and curtailment detection."""

from typing import Dict, Union
import numpy as np
import torch

def compute_wmape(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-5
) -> float:
    """Weighted Mean Absolute Percentage Error (wMAPE).
    
    Formula: sum(|y_true - y_pred|) / (sum(|y_true|) + eps)
    """
    if isinstance(y_true, torch.Tensor):
        diff = torch.abs(y_true - y_pred).sum()
        denom = torch.abs(y_true).sum() + eps
        return (diff / denom).item()
    else:
        diff = np.sum(np.abs(y_true - y_pred))
        denom = np.sum(np.abs(y_true)) + eps
        return float(diff / denom)

def compute_curtailment_f1(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred_prob: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """Precision, recall, and F1 score for curtailment bottleneck event classification."""
    if isinstance(y_true, torch.Tensor) and isinstance(y_pred_prob, torch.Tensor):
        y_true_b = (y_true > 0.5)
        y_pred_b = (y_pred_prob >= threshold)
        tp = int((y_true_b & y_pred_b).sum().item())
        fp = int((~y_true_b & y_pred_b).sum().item())
        fn = int((y_true_b & ~y_pred_b).sum().item())
        tn = int((~y_true_b & ~y_pred_b).sum().item())
        total_len = y_true.numel()
    else:
        if isinstance(y_true, torch.Tensor):
            y_true_np = (y_true > 0.5).detach().cpu().numpy().astype(int).flatten()
        else:
            y_true_np = (y_true > 0.5).astype(int).flatten()

        if isinstance(y_pred_prob, torch.Tensor):
            y_pred_np = (y_pred_prob >= threshold).detach().cpu().numpy().astype(int).flatten()
        else:
            y_pred_np = (y_pred_prob >= threshold).astype(int).flatten()

        tp = int(np.sum((y_true_np == 1) & (y_pred_np == 1)))
        fp = int(np.sum((y_true_np == 0) & (y_pred_np == 1)))
        fn = int(np.sum((y_true_np == 1) & (y_pred_np == 0)))
        tn = int(np.sum((y_true_np == 0) & (y_pred_np == 0)))
        total_len = len(y_true_np)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(1, total_len)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

def compute_grid_metrics(
    injection_true: Union[torch.Tensor, np.ndarray],
    injection_pred: Union[torch.Tensor, np.ndarray],
    curtailment_true: Union[torch.Tensor, np.ndarray],
    curtailment_pred_prob: Union[torch.Tensor, np.ndarray],
    curtail_threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes joint telemetry for injection wMAPE, MAE, RMSE and curtailment classification."""
    wmape = compute_wmape(injection_true, injection_pred)
    
    if isinstance(injection_true, torch.Tensor):
        mae = torch.mean(torch.abs(injection_true - injection_pred)).item()
        rmse = torch.sqrt(torch.mean((injection_true - injection_pred) ** 2)).item()
    else:
        mae = float(np.mean(np.abs(injection_true - injection_pred)))
        rmse = float(np.sqrt(np.mean((injection_true - injection_pred) ** 2)))

    curtail_metrics = compute_curtailment_f1(
        curtailment_true, curtailment_pred_prob, threshold=curtail_threshold
    )

    return {
        "wmape": float(wmape),
        "mae": float(mae),
        "rmse": float(rmse),
        "curtailment_f1": curtail_metrics["f1"],
        "curtailment_precision": curtail_metrics["precision"],
        "curtailment_recall": curtail_metrics["recall"],
        "curtailment_accuracy": curtail_metrics["accuracy"],
    }


def baseline_forecasts(inputs, injections, curtailments, starts, window: int, horizon: int):
    """Forecasts for targets at t+horizon, t in each window. Returns {name: (injection (W,T,N,2), curtailment (W,T,N,1))}.
    copy: potentials and the curtailment rule at t. yesterday: observed value at t + horizon - 96 (same slot, one day back).
    """
    import numpy as np
    sl = lambda arr, off: np.stack([arr[s + off:s + off + window] for s in starts])
    x = sl(inputs, 0)
    rule = (x[..., 0] + x[..., 1] - x[..., 2] - x[..., 3] > 0).astype(np.float32)[..., None]
    return {
        "copy": (x[..., :2], rule),
        "yesterday": (sl(injections, horizon - 96), sl(curtailments, horizon - 96)),
    }
