#!/usr/bin/env bash
# Robust environment activator for LRDBenchmark
# Prefers conda env 'lrdbenchmark'; falls back to local .venv

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH-}"

# Try to activate conda environment if available
if command -v conda >/dev/null 2>&1; then
  # Initialize conda with error handling
  if eval "$(conda shell.bash hook)" 2>/dev/null; then
    if conda env list | awk '{print $1}' | grep -qx "lrdbenchmark_gpu"; then
      # Use a more robust activation method
      source $(conda info --base)/etc/profile.d/conda.sh
      conda activate lrdbenchmark_gpu
      echo "[env] Activated conda environment: lrdbenchmark_gpu"
    elif conda env list | awk '{print $1}' | grep -qx "lrdbenchmark"; then
      source $(conda info --base)/etc/profile.d/conda.sh
      conda activate lrdbenchmark
      echo "[env] Activated conda environment: lrdbenchmark"
    else
      echo "[env] Conda detected but environment 'lrdbenchmark_gpu' or 'lrdbenchmark' not found; will use .venv if present."
    fi
  else
    echo "[env] Conda hook failed; will use .venv if present."
  fi
fi

# If not in conda, activate or create .venv
if [[ -z "${CONDA_DEFAULT_ENV-}" ]]; then
  if [[ -d "${PROJECT_ROOT}/.venv" ]]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.venv/bin/activate"
    echo "[env] Activated Python venv: ${PROJECT_ROOT}/.venv"
  else
    echo "[env] Creating local Python venv under ${PROJECT_ROOT}/.venv ..."
    python3 -m venv "${PROJECT_ROOT}/.venv"
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.venv/bin/activate"
    python -m pip install --upgrade pip
    if [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
      python -m pip install -e "${PROJECT_ROOT}"
    fi
    if [[ -f "${PROJECT_ROOT}/docs/requirements.txt" ]]; then
      python -m pip install -r "${PROJECT_ROOT}/docs/requirements.txt" || true
    fi
    echo "[env] Activated Python venv: ${PROJECT_ROOT}/.venv"
  fi
fi

# Common environment tweaks
export MPLBACKEND=Agg

# Final status line
if [[ -n "${VIRTUAL_ENV-}" ]]; then
  ENV_NAME="$(basename "${VIRTUAL_ENV}")"
elif [[ -n "${CONDA_DEFAULT_ENV-}" ]]; then
  ENV_NAME="${CONDA_DEFAULT_ENV}"
else
  ENV_NAME="system"
fi
echo "[env] Environment ready: ${ENV_NAME}"











