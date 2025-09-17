#!/usr/bin/env bash
# Clean environment activator for LRDBenchmark
# Avoids conda activation issues by using direct Python path

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH-}"

# Find conda installation
CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
fi

# Try to use the GPU environment directly without conda activation
if [[ -n "$CONDA_BASE" && -d "$CONDA_BASE/envs/lrdbenchmark_gpu" ]]; then
    # Use direct Python path to avoid conda activation issues
    export PATH="$CONDA_BASE/envs/lrdbenchmark_gpu/bin:$PATH"
    export CONDA_DEFAULT_ENV="lrdbenchmark_gpu"
    export CONDA_PREFIX="$CONDA_BASE/envs/lrdbenchmark_gpu"
    echo "[env] Using lrdbenchmark_gpu environment (direct path)"
elif [[ -n "$CONDA_BASE" && -d "$CONDA_BASE/envs/lrdbenchmark" ]]; then
    # Fallback to regular lrdbenchmark environment
    export PATH="$CONDA_BASE/envs/lrdbenchmark/bin:$PATH"
    export CONDA_DEFAULT_ENV="lrdbenchmark"
    export CONDA_PREFIX="$CONDA_BASE/envs/lrdbenchmark"
    echo "[env] Using lrdbenchmark environment (direct path)"
else
    # Use system Python or create venv
    if [[ -d "${PROJECT_ROOT}/.venv" ]]; then
        source "${PROJECT_ROOT}/.venv/bin/activate"
        echo "[env] Activated Python venv: ${PROJECT_ROOT}/.venv"
    else
        echo "[env] Creating local Python venv under ${PROJECT_ROOT}/.venv ..."
        python3 -m venv "${PROJECT_ROOT}/.venv"
        source "${PROJECT_ROOT}/.venv/bin/activate"
        python -m pip install --upgrade pip
        if [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
            python -m pip install -e "${PROJECT_ROOT}"
        fi
        echo "[env] Activated Python venv: ${PROJECT_ROOT}/.venv"
    fi
fi

# Common environment tweaks
export MPLBACKEND=Agg

# Final status
if [[ -n "${VIRTUAL_ENV-}" ]]; then
    ENV_NAME="$(basename "${VIRTUAL_ENV}")"
elif [[ -n "${CONDA_DEFAULT_ENV-}" ]]; then
    ENV_NAME="${CONDA_DEFAULT_ENV}"
else
    ENV_NAME="system"
fi
echo "[env] Environment ready: ${ENV_NAME}"
