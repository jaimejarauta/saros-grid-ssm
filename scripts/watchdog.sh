#!/usr/bin/env bash
# Unattended execution watchdog for long-running GPU jobs: saros-grid-ssm
# Enforces pre-flight busy checks, environment sandboxing, and persistent logging.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${REPO_ROOT}/results/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

echo "=== [Watchdog] Starting job at ${TIMESTAMP} ===" | tee -a "${LOG_FILE}"
echo "Repository: ${REPO_ROOT}" | tee -a "${LOG_FILE}"
echo "Git Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')" | tee -a "${LOG_FILE}"

# 1. Pre-flight busy check
if command -v nvidia-smi &> /dev/null; then
    BUSY_GPUS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
    if [ -n "${BUSY_GPUS}" ]; then
        echo "[Watchdog WARNING] GPU has active compute processes:" | tee -a "${LOG_FILE}"
        echo "${BUSY_GPUS}" | tee -a "${LOG_FILE}"
    else
        echo "[Watchdog] GPU check: IDLE" | tee -a "${LOG_FILE}"
    fi
fi

# 2. Virtual environment resolution
VENV_PY="${REPO_ROOT}/.venv/bin/python"
if [ ! -f "${VENV_PY}" ]; then
    SHARED_VENV="${REPO_ROOT}/../.shared_venv/bin/python"
    if [ -f "${SHARED_VENV}" ]; then
        VENV_PY="${SHARED_VENV}"
    else
        VENV_PY="python3"
    fi
fi
echo "[Watchdog] Using Python: ${VENV_PY}" | tee -a "${LOG_FILE}"

# 3. Environment isolation (mitigate disk usage on shared containers)
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${WANDB_DIR}"

# 4. Job Execution with exit trapping
CMD=("$@")
if [ ${#CMD[@]} -eq 0 ]; then
    CMD=("${VENV_PY}" "scripts/run_smoke.py")
fi

echo "[Watchdog] Executing: ${CMD[*]}" | tee -a "${LOG_FILE}"

EXIT_CODE=0
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}" || EXIT_CODE=$?

END_TIME=$(date +"%Y%m%d_%H%M%S")
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "=== [Watchdog] Job completed successfully at ${END_TIME} ===" | tee -a "${LOG_FILE}"
else
    echo "=== [Watchdog ERROR] Job failed with exit code ${EXIT_CODE} at ${END_TIME} ===" | tee -a "${LOG_FILE}"
fi

exit ${EXIT_CODE}
