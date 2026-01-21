# WSL Quick Start Guide

Quick reference for setting up LRDBenchmark in WSL with RTX 5070 GPU support.

## Prerequisites

- WSL2 installed and running
- NVIDIA driver 516.40+ installed on Windows
- Python 3.10-3.12 available in WSL

## Quick Setup (3 Steps)

### 1. Run Setup Script

```bash
./setup_wsl_environment.sh
```

This creates the virtual environment and installs all dependencies.

### 2. Verify Setup

```bash
python verify_wsl_gpu_setup.py
```

### 3. Activate Environment

```bash
source activate_env.sh
```

## Expected Output

After setup, you should see:
- ✅ PyTorch with CUDA 12.1 support (RTX 5070 compatible)
- ⚠️  JAX may use CPU fallback (normal for RTX 5070 sm_120)
- ✅ All LRDBenchmark dependencies installed

## Quick Test

```python
import torch
print(f"PyTorch CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Need Help?

- Full guide: See `WSL_SETUP_GUIDE.md`
- Troubleshooting: Run `python verify_wsl_gpu_setup.py`
- GPU config: See `docs/development_summaries/GPU_CONFIGURATION_GUIDE.md`

## Notes

- **PyTorch GPU**: ✅ Works perfectly with RTX 5070
- **JAX GPU**: ⚠️ May use CPU (this is OK, neural networks use PyTorch)
- **Performance**: 10-50x speedup for neural network training on GPU

