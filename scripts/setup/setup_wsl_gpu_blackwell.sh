#!/bin/bash
# Dedicated WSL Setup for JAX GPU on Blackwell (RTX 5070)
# Created on 2026-03-10

set -euo pipefail

ENV_NAME="lrdbenchmark_gpu_env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_PATH="${PROJECT_ROOT}/${ENV_NAME}"

echo "================================================"
echo "LRDBenchmark Blackwell GPU Setup (WSL)"
echo "Targeting RTX 5070 (sm_120)"
echo "================================================"

# Check for Blackwell Support
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "Error: nvidia-smi not found. Ensure NVIDIA drivers are installed on Windows."
    exit 1
fi

# Create fresh venv
echo "Creating fresh virtual environment at ${ENV_PATH}..."
rm -rf "${ENV_PATH}"
python3 -m venv "${ENV_PATH}"

# Activate
source "${ENV_PATH}/bin/activate"

# Upgrade toolchain
pip install --upgrade pip setuptools wheel

# Install JAX with Blackwell support (CUDA 13.2)
# In 2026, JAX 0.9.x is the latest.
echo "Installing JAX 0.9.x with CUDA 13.2 support..."
pip install --upgrade "jax[cuda13]==0.9.1"

# Install PyTorch for Blackwell (CUDA 13.2 support)
echo "Installing PyTorch with CUDA 13.2 support..."
# Check for cu132 or cu130
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132 || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
echo "Installing LRDBenchmark dependencies..."
cd "${PROJECT_ROOT}"
pip install -e .[accel-all]

# Verify JAX GPU
echo "================================================"
echo "Verifying JAX GPU Support"
echo "================================================"
python -c "
import jax
import sys
print(f'JAX Version: {jax.__version__}')
backend = jax.default_backend()
print(f'Detected Backend: {backend}')
devices = jax.devices()
print(f'Devices: {devices}')

if backend == 'gpu':
    print('✅ JAX GPU SUCCESS: GPU is enabled and active.')
else:
    print('❌ JAX GPU FAILURE: JAX is still using CPU fallback.')
    sys.exit(1)
"

echo ""
echo "Setup Complete!"
echo "To use this environment, run: source ${ENV_NAME}/bin/activate"
