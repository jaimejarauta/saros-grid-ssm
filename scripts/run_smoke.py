#!/usr/bin/env python3
"""Execution script for Saros-Grid-SSM smoke test on 50-Node REE Spanish Grid."""

import argparse
import json
import os
import sys
import time
import subprocess
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from saros_grid_ssm.model import GridSpatioTemporalSSM
from saros_grid_ssm.ree_data import generate_synthetic_ree_grid, REESpanishGridDataset
from saros_grid_ssm.metrics import compute_wmape, compute_curtailment_f1, compute_grid_metrics

def get_git_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"

def main():
    parser = argparse.ArgumentParser(description="Saros-Grid-SSM Smoke Test Runner")
    parser.add_argument("--config", type=str, default="configs/smoke_test.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda, mps, cpu)")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases telemetry logging")
    parser.add_argument("--wandb-project", type=str, default="moonshot-saros-grid-ssm", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run display name")
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"], help="W&B run mode override")
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(REPO_ROOT, args.config)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Device Auto-Detection: CUDA -> Apple Silicon MPS -> CPU
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    print("=" * 70)
    print("SAROS-GRID-SSM: SPATIO-TEMPORAL REE 50-NODE TRANSMISSION SMOKE TEST")
    print(f"Device: {device} ({dev_name})")
    print("=" * 70)

    # Set reproducibility seed
    seed = cfg["training"].get("seed", 2026)
    torch.manual_seed(seed)

    # Initialize W&B tracking if requested or WANDB_API_KEY is present
    use_wandb = args.wandb or (os.environ.get("WANDB_API_KEY") is not None)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            mode = args.wandb_mode or os.environ.get("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "offline")
            run_name = args.wandb_name or f"saros-grid-ssm-smoke-seed{seed}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                group="saros-grid-smoke",
                job_type="smoke",
                name=run_name,
                config=cfg,
                mode=mode,
                dir=os.environ.get("WANDB_DIR", "/tmp/wandb"),
            )
            print(f"[W&B] Initialized run '{run_name}' in project '{args.wandb_project}' (mode: {mode})")
        except Exception as e:
            print(f"[W&B Warning] Failed to initialize wandb: {e}. Continuing without W&B.")
            use_wandb = False

    # 2. Generate Synthetic REE Grid Data (14 days = 1344 timesteps @ 15 min)
    print("\n[1/4] Generating synthetic REE 50-node transmission dataset...")
    num_nodes = cfg["grid"]["num_nodes"]
    raw_data = generate_synthetic_ree_grid(num_days=21, num_nodes=num_nodes, seed=seed)
    adj_norm = torch.from_numpy(raw_data["adjacency_norm"]).float().to(device)

    # Split train and validation (14 days train, 7 days val)
    train_split = 14 * 96
    train_dataset = REESpanishGridDataset(
        inputs=raw_data["inputs"][:train_split],
        injections=raw_data["injections"][:train_split],
        curtailments=raw_data["curtailments"][:train_split],
        window_size=96,
        stride=48
    )
    val_dataset = REESpanishGridDataset(
        inputs=raw_data["inputs"][train_split:],
        injections=raw_data["injections"][train_split:],
        curtailments=raw_data["curtailments"][train_split:],
        window_size=96,
        stride=96
    )

    batch_size = cfg["training"].get("batch_size", 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    print(f"Dataset generated: Train windows = {len(train_dataset)}, Val windows = {len(val_dataset)}")
    print(f"Grid topology: {num_nodes} nodes, {int(raw_data['adjacency'].sum() // 2)} transmission corridors")

    # 3. Model Instantiation
    print("\n[2/4] Initializing GridSpatioTemporalSSM...")
    model = GridSpatioTemporalSSM(
        in_features=cfg["grid"]["in_features"],
        out_features=cfg["grid"]["out_features"],
        d_model=cfg["model"]["d_model"],
        d_state=cfg["model"]["d_state"],
        num_spatial_hops=cfg["model"]["num_spatial_hops"],
        num_layers=cfg["model"]["num_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated. Trainable parameters: {param_count:,}")

    # 4. Training Loop
    print("\n[3/4] Running smoke training loop...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    huber_loss_fn = nn.SmoothL1Loss(beta=10.0)
    bce_loss_fn = nn.BCELoss()

    max_steps = args.steps if args.steps is not None else cfg["training"].get("steps", 50)
    step = 0
    t0 = time.time()
    model.train()

    while step < max_steps:
        for x_b, y_inj_b, y_curt_b in train_loader:
            x_b = x_b.to(device)
            y_inj_b = y_inj_b.to(device)
            y_curt_b = y_curt_b.to(device)

            optimizer.zero_grad()
            preds = model(x_b, adj_norm)

            # Joint loss: Renewable injection MW + Curtailment bottleneck risk
            loss_inj = huber_loss_fn(preds["injection_mw"], y_inj_b)
            loss_curt = bce_loss_fn(preds["curtailment_prob"], y_curt_b)
            total_loss = loss_inj + 5.0 * loss_curt

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step += 1
            if step % 10 == 0 or step == max_steps:
                wmape_batch = compute_wmape(y_inj_b, preds["injection_mw"])
                print(f"Step [{step:03d}/{max_steps:03d}] | Total Loss: {total_loss.item():.4f} | "
                      f"Inj Loss: {loss_inj.item():.4f} | Curt Loss: {loss_curt.item():.4f} | "
                      f"Batch wMAPE: {wmape_batch * 100:.2f}%")
                if use_wandb:
                    wandb.log({
                        "step": step,
                        "train/total_loss": total_loss.item(),
                        "train/loss_inj": loss_inj.item(),
                        "train/loss_curt": loss_curt.item(),
                        "train/batch_wmape": wmape_batch,
                    })

            if step >= max_steps:
                break

    train_duration = time.time() - t0
    print(f"Training completed in {train_duration:.2f} seconds ({train_duration / max_steps * 1000:.1f} ms/step).")

    # 5. Validation Evaluation
    print("\n[4/4] Evaluating on held-out validation dispatch days...")
    model.eval()
    with torch.no_grad():
        for x_val, y_inj_val, y_curt_val in val_loader:
            x_val = x_val.to(device)
            y_inj_val = y_inj_val.to(device)
            y_curt_val = y_curt_val.to(device)

            preds_val = model(x_val, adj_norm)

            metrics = compute_grid_metrics(
                injection_true=y_inj_val,
                injection_pred=preds_val["injection_mw"],
                curtailment_true=y_curt_val,
                curtailment_pred_prob=preds_val["curtailment_prob"]
            )

    wmape_pct = metrics["wmape"] * 100.0
    gate_threshold_pct = cfg["pre_registered_rules"]["gate_threshold"] * 100.0
    curtail_f1 = metrics["curtailment_f1"]
    f1_threshold = cfg["pre_registered_rules"].get("curtailment_f1_threshold", 0.75)
    gate_passed = (
        metrics["wmape"] < cfg["pre_registered_rules"]["gate_threshold"]
        and curtail_f1 >= f1_threshold
    )

    print("\n" + "=" * 70)
    print("PRE-REGISTERED GATE 1 ASSESSMENT:")
    print(f"  Day-Ahead Renewable Injection wMAPE : {wmape_pct:.2f}%  (Target: < {gate_threshold_pct:.2f}%)")
    print(f"  Mean Absolute Error (MAE)           : {metrics['mae']:.2f} MW")
    print(f"  Root Mean Square Error (RMSE)       : {metrics['rmse']:.2f} MW")
    print(f"  Curtailment Bottleneck F1 Score     : {curtail_f1:.4f}  (Target: >= {cfg['pre_registered_rules']['curtailment_f1_threshold']:.2f})")
    print(f"  Curtailment Precision / Recall      : {metrics['curtailment_precision']:.4f} / {metrics['curtailment_recall']:.4f}")
    print(f"  Gate 1 Status                       : {'PASSED' if gate_passed else 'FAILED'}")
    print("=" * 70)

    if use_wandb and wandb_run:
        wandb.summary["wmape"] = metrics["wmape"]
        wandb.summary["mae"] = metrics["mae"]
        wandb.summary["rmse"] = metrics["rmse"]
        wandb.summary["curtailment_f1"] = metrics["curtailment_f1"]
        wandb.summary["curtailment_precision"] = metrics["curtailment_precision"]
        wandb.summary["curtailment_recall"] = metrics["curtailment_recall"]
        wandb.summary["gate1_passed"] = gate_passed
        wandb.finish()

    # 6. Save Manifest
    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "git_commit": get_git_hash(),
        "device": str(device),
        "device_name": dev_name,
        "parameters": param_count,
        "train_steps": max_steps,
        "train_duration_sec": train_duration,
        "metrics": metrics,
        "gate_passed": gate_passed,
        "config": cfg
    }

    manifest_path = os.path.join(results_dir, "smoke_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nExecution manifest saved to: {manifest_path}")

    return 0 if gate_passed else 1

if __name__ == "__main__":
    sys.exit(main())
