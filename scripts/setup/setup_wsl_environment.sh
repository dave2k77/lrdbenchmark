#!/bin/bash
# WSL Environment Setup Script for LRDBenchmark with RTX 5070 GPU Support
# This script sets up a Python virtual environment in WSL with GPU support

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="lrdbenchmark_venv"
ENV_PATH="${SCRIPT_DIR}/${ENV_NAME}"
PYTHON_VERSION="3.11"

echo "================================================"
echo "LRDBenchmark WSL Environment Setup"
echo "GPU: NVIDIA RTX 5070"
echo "================================================"
echo ""

# Check if running in WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Warning: This doesn't appear to be a WSL environment."
    echo "Continuing anyway, but GPU support may not work correctly."
    echo ""
fi

# Check for NVIDIA GPU
echo "Checking NVIDIA GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo "Warning: nvidia-smi not found. GPU support may not be available."
    echo "Make sure NVIDIA drivers are installed in WSL."
    echo "For WSL2, you need:"
    echo "  1. Latest NVIDIA driver on Windows (516.40 or newer)"
    echo "  2. CUDA toolkit in WSL (optional for PyTorch/JAX)"
    echo ""
fi

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.10-3.12"
    echo "On Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $(python3 --version)"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) and sys.version_info < (3, 13) else 1)"; then
    echo "Warning: Python ${PYTHON_VER} is recommended (3.10-3.12)"
    echo "Current version may work but is not fully tested."
fi
echo ""

# Create virtual environment
if [ -d "${ENV_PATH}" ]; then
    echo "Virtual environment already exists at ${ENV_PATH}"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "${ENV_PATH}"
    else
        echo "Using existing virtual environment."
        echo ""
        echo "To activate the environment, run:"
        echo "  source ${ENV_PATH}/bin/activate"
        echo "or:"
        echo "  source activate_env.sh"
        exit 0
    fi
fi

echo "Creating Python virtual environment..."
python3 -m venv "${ENV_PATH}"
echo "Virtual environment created at ${ENV_PATH}"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source "${ENV_PATH}/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel
echo ""

# Detect CUDA version for PyTorch
echo "Detecting CUDA version for PyTorch installation..."
CUDA_VERSION="cu121"  # Default to CUDA 12.1 for RTX 5070

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "CUDA toolkit version: ${NVCC_VERSION}"
    
    # Map CUDA version to PyTorch CUDA version
    if [[ $(echo "$NVCC_VERSION >= 12.1" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        CUDA_VERSION="cu121"
    elif [[ $(echo "$NVCC_VERSION >= 12.0" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        CUDA_VERSION="cu121"
    elif [[ $(echo "$NVCC_VERSION >= 11.8" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        CUDA_VERSION="cu118"
    fi
else
    echo "CUDA toolkit not found, but PyTorch will use CUDA from system driver"
    echo "Installing PyTorch with CUDA 12.1 support (compatible with RTX 5070)"
fi

echo "Selected PyTorch CUDA version: ${CUDA_VERSION}"
echo ""

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA ${CUDA_VERSION} support..."
if [ "${CUDA_VERSION}" == "cu121" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi
echo ""

# Install JAX with CUDA support (will fallback to CPU if needed)
echo "Installing JAX with CUDA support..."
# Try CUDA 12 first (for RTX 5070)
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 2>/dev/null || {
    echo "JAX CUDA 12 installation failed, trying CUDA 11..."
    pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 2>/dev/null || {
        echo "JAX CUDA installation failed, installing CPU-only version..."
        pip install jax jaxlib
    }
}
echo ""

# Install core dependencies
echo "Installing core LRDBenchmark dependencies..."
pip install numpy>=1.26.0
pip install scipy>=1.10.0
pip install scikit-learn>=1.2.0
pip install pandas>=1.5.0
pip install pywavelets>=1.3.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install psutil>=5.8.0
pip install networkx>=2.6.0
pip install joblib>=1.1.0
echo ""

# Install additional acceleration libraries
echo "Installing additional acceleration libraries..."
pip install numba>=0.60.0
echo ""

# Install project in editable mode
echo "Installing LRDBenchmark in editable mode..."
pip install -e .[accel-all]
echo ""

# Verify GPU setup
echo "================================================"
echo "Verifying GPU Setup"
echo "================================================"
echo ""

python << 'EOF'
import sys

print("Testing PyTorch GPU support...")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("  ⚠️  Warning: CUDA not available in PyTorch")
        print("  This may be OK if CUDA toolkit is not installed in WSL")
        print("  PyTorch may still work using the Windows driver")
except Exception as e:
    print(f"  Error: {e}")
    sys.exit(1)

print("\nTesting JAX GPU support...")
try:
    import jax
    print(f"JAX version: {jax.__version__}")
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    print(f"JAX backend: {jax.default_backend()}")
    
    if jax.default_backend() == 'gpu':
        print("  JAX GPU support: ✅ Working")
    else:
        print("  JAX GPU support: ⚠️ CPU fallback (this is OK for RTX 5070)")
except Exception as e:
    print(f"  Warning: JAX setup issue: {e}")
    print("  JAX may fall back to CPU mode")

print("\nTesting core dependencies...")
try:
    import numpy as np
    import scipy
    import pandas as pd
    import sklearn
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print("  Core dependencies: ✅ All working")
except Exception as e:
    print(f"  Error: {e}")
    sys.exit(1)

print("\n✅ GPU setup verification complete!")
EOF

VERIFY_EXIT_CODE=$?
if [ $VERIFY_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "⚠️  Some verification tests failed. The environment may still work,"
    echo "but GPU support may be limited. Check the output above for details."
    echo ""
    echo "You can continue to use the environment, but GPU features may not work."
    echo "To troubleshoot, run: python verify_wsl_gpu_setup.py"
    # Don't exit with error - allow user to proceed
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Virtual environment location: ${ENV_PATH}"
echo ""
echo "To activate the environment, run:"
echo "  source ${ENV_PATH}/bin/activate"
echo "or:"
echo "  source activate_env.sh"
echo ""
echo "To deactivate, simply run:"
echo "  deactivate"
echo ""
echo "GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    echo "  ✅ NVIDIA driver detected"
else
    echo "  ⚠️  NVIDIA driver not found"
fi

python << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"  ✅ PyTorch GPU: Available (CUDA {torch.version.cuda})")
else:
    print(f"  ⚠️  PyTorch GPU: Not available")

import jax
backend = jax.default_backend()
if backend == 'gpu':
    print(f"  ✅ JAX GPU: Available")
else:
    print(f"  ⚠️  JAX GPU: CPU fallback (normal for RTX 5070)")
EOF

echo ""
echo "For more information, see:"
echo "  - docs/development_summaries/GPU_CONFIGURATION_GUIDE.md"
echo "  - config/CONDA_FREE_SETUP.md"
echo ""

