# Getting Started: saros-grid-ssm

This document defines the exact workflow, metrics, and branching protocols for developing **Saros-Grid-SSM** (Idea 07: Solar/Wind Spatio-Temporal SSM for Spanish REE Electricity Grid).

---

## 1. Environment & Setup

### Local GPU & LXC Containers
```bash
# 1. Create and activate virtual environment via uv
uv venv .venv
source .venv/bin/activate

# 2. Install dependencies (strictly uv pip)
uv pip install -e ".[dev]"

# 3. Verify PyTorch installation and GPU memory
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 4. Launch with high-performance Tensor Core execution
.venv/bin/python scripts/run_smoke.py --config configs/smoke_test.yaml
```

### Google Colab (Free T4 or Pro A100/L4)
1. Open [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) in Google Colab.
2. Ensure GPU runtime is selected (`Runtime -> Change runtime type -> T4 GPU`).
3. Run all cells: the notebook auto-detects T4 Tensor Cores, installs dependencies with `uv`, runs the REE 50-node 15-minute dispatch smoke test, and evaluates wMAPE and curtailment F1 against pre-registered criteria.

---

## 2. Git & Branching Protocol

- **Protected Branch:** `main`. **Never commit, rebase, or push directly to `main`.**
- **Active Feature Branch:** `feature/initial-implementation`.
- **Workflow for New Tasks:**
  ```bash
  git checkout feature/initial-implementation
  git checkout -b feat/task-name
  # Implement code with tests (TDD)
  .venv/bin/python -m pytest tests/
  git add <touched_files>
  git commit -m "feat: implement spatio-temporal selective recurrence"
  # Never git push. Notify the lead researcher when ready.
  ```
- **Commit Format:** `<type>: <what changed>` where type is one of `feat`, `fix`, `docs`, `refactor`, `test`, `exp`.
- **Attribution Rule:** **Zero AI or Claude attribution anywhere** (no trailers, no commit notes, no comments).

---

## 3. Core Hypothesis, Metrics & Pre-Registered Criteria

### The Question
Can a continuous-time Spatio-Temporal Selective State Space Model (SSM) combining non-Euclidean graph spatial diffusion with diagonal linear state space recurrence predict day-ahead nodal renewable injections across the 50 major transmission nodes of the Spanish REE electricity grid with $\text{wMAPE} < 6.5\%$ while outperforming spatial GCNs and attention baselines without $O(N^2)$ compute?

### Key Performance Indicators (KPIs)
1. **Weighted Mean Absolute Percentage Error (wMAPE):**
   $$\text{wMAPE} = \frac{\sum_{n=1}^{50} \sum_{t=1}^{96} |y_{n,t} - \hat{y}_{n,t}|}{\sum_{n=1}^{50} \sum_{t=1}^{96} |y_{n,t}|} < 6.5\%$$
   evaluated on day-ahead (96 timesteps @ 15 min) nodal solar and wind dispatch.
2. **Curtailment Bottleneck Detection F1:** Binary classification $F_1 \ge 0.75$ on transmission export constraint curtailment events.
3. **Persistence Baseline Contrast:**
   Outperforms the 24-hour seasonal persistence baseline ($y_t = y_{t-96}$, same dispatch period yesterday) by at least $25\%$ relative reduction in wMAPE:
   $$\frac{\text{wMAPE}_{\text{persist}} - \text{wMAPE}_{\text{ssm}}}{\text{wMAPE}_{\text{persist}}} \ge 0.25$$
4. **Spectral Stability Invariant:** Real eigenvalues of continuous transition matrix satisfy $\text{Re}(\lambda_i) \le 0$, ensuring asymptotic BIBO stability and zero phase drift over extended multi-week rollouts.

### Pre-Registered Rules
- **Kill Criterion (Day 3 Gate):** If on the REE 50-node day-ahead benchmark, the model achieves $\text{wMAPE} \ge 8.5\%$ OR fails to beat the 24-hour persistence baseline by at least $15\%$, OR achieves curtailment $F_1 < 0.50$, kill immediately and archive as a controlled null.
- **Pulse Criterion:** If the model achieves $\text{wMAPE} < 6.5\%$ and curtailment $F_1 \ge 0.78$ while maintaining sub-quadratic $O(T \cdot N \cdot D)$ memory scaling, proceed with Phase 2 (OMIE marginal pricing integration and battery storage dispatch optimization).

---

## 4. Weights & Biases (W&B) Setup

```python
import wandb

wandb.init(
    project="moonshot-saros-grid-ssm",
    group="smoke-test-ree-50nodes",
    job_type="train",
    name="saros-grid-ssm-seed2026",
    config={
        "num_nodes": 50,
        "timesteps": 96,
        "d_model": 64,
        "d_state": 16,
        "spatial_hops": 2,
        "bidirectional": True,
        "gate_wmape": 0.065,
        "manifest_digest": "sha256_here"
    }
)
```

---

## 5. Day 2 Smoke Test Execution

Before launching on a shared machine:
```bash
# 1. Busy check
ps -eo pid,etime,pcpu,comm,args --sort=-pcpu | head
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# 2. Run with watchdog
./scripts/watchdog.sh .venv/bin/python scripts/run_smoke.py --config configs/smoke_test.yaml
```
