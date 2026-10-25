# Saros Grid SSM

Spatio-temporal selective state space models for electrical transmission grid dispatch and curtailment risk forecasting.

## Overview
Saros Grid SSM combines $K$-hop Chebyshev graph diffusion and bidirectional selective state space models (SSMs) to model high-voltage transmission networks (reflecting the Spanish REE Iberian grid). It jointly forecasts 15-minute nodal renewable injections (solar and wind MW) and transmission corridor bottleneck curtailments.

## Pre-Registered Gate Criteria
- **Gate 1 (Smoke / Dispatch):** Day-Ahead renewable injection wMAPE $< 6.50\%$ and curtailment bottleneck $F_1 \ge 0.75$.
- **Kill Rule:** wMAPE $\ge 8.50\%$ or $F_1 < 0.50$.

## Quickstart

### Local (Apple Silicon MPS / NVIDIA CUDA):
```bash
# Run smoke test (50 nodes x 96 intervals)
python scripts/run_smoke.py --config configs/smoke_test.yaml --steps 50 --wandb
```

### Google Colab:
Open `notebooks/colab_runner_saros_grid_ssm.ipynb` in Colab on a standard T4 GPU. Includes automatic machine teardown (`runtime.unassign()`) to protect compute credits upon completion.

## Telemetry
Native Weights & Biases support via `--wandb` (project: `moonshot-saros-grid-ssm`).

## License
MIT

