# WSL Environment Setup Guide for LRDBenchmark with RTX 5070 GPU

This guide provides step-by-step instructions for setting up a working virtual environment for LRDBenchmark in WSL (Windows Subsystem for Linux) with support for NVIDIA RTX 5070 GPU.

## Prerequisites

### 1. WSL Installation

Ensure you have WSL2 installed and configured:

```bash
# Check WSL version
wsl --version

# If WSL2 is not installed, install it:
# In PowerShell (as Administrator):
wsl --install
```

### 2. NVIDIA Drivers

**Important**: For WSL2, you need NVIDIA drivers installed on **both** Windows and WSL:

#### Windows Side:
1. Install the latest NVIDIA driver for Windows (version 516.40 or newer)
2. The driver must support WSL2 GPU passthrough
3. Download from: https://www.nvidia.com/Download/index.aspx

#### WSL Side:
The NVIDIA driver in Windows automatically provides GPU access in WSL2. No separate driver installation needed in WSL, but you may want the CUDA toolkit:

```bash
# Optional: Install CUDA toolkit in WSL (for development)
# This is not strictly required for PyTorch/JAX as they bundle CUDA libraries
```

### 3. Verify GPU Access in WSL

Before proceeding, verify that your GPU is accessible from WSL:

```bash
# Check NVIDIA driver
nvidia-smi

# You should see your RTX 5070 listed
# If not, ensure:
# 1. Latest NVIDIA driver is installed on Windows
# 2. WSL2 is being used (not WSL1)
# 3. NVIDIA WSL driver support is enabled
```

## Quick Setup

### Automated Setup Script

The easiest way to set up the environment is using the provided script:

```bash
# Make script executable (if needed)
chmod +x setup_wsl_environment.sh

# Run the setup script
./setup_wsl_environment.sh
```

This script will:
- ✅ Check WSL environment
- ✅ Verify NVIDIA GPU availability
- ✅ Create a Python virtual environment
- ✅ Install PyTorch with CUDA 12.1 support (compatible with RTX 5070)
- ✅ Install JAX (with automatic CPU fallback if GPU not supported)
- ✅ Install all LRDBenchmark dependencies
- ✅ Verify GPU setup

### Manual Setup (Alternative)

If you prefer to set up manually or the script fails:

#### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd /path/to/lrdbenchmark

# Create virtual environment
python3 -m venv lrdbenchmark_venv

# Activate virtual environment
source lrdbenchmark_venv/bin/activate
```

#### Step 2: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

#### Step 3: Install PyTorch with CUDA Support

For RTX 5070, use CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Step 4: Install JAX

Try CUDA 12 first, but it will fall back to CPU if needed:

```bash
# Try CUDA 12
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# If that fails, install CPU version (still works fine)
# pip install jax jaxlib
```

#### Step 5: Install Core Dependencies

```bash
pip install numpy>=1.26.0 scipy>=1.10.0 scikit-learn>=1.2.0 \
            pandas>=1.5.0 pywavelets>=1.3.0 matplotlib>=3.5.0 \
            seaborn>=0.11.0 psutil>=5.8.0 networkx>=2.6.0 joblib>=1.1.0 \
            numba>=0.60.0
```

#### Step 6: Install LRDBenchmark

```bash
# Install in editable mode with all acceleration extras
pip install -e .[accel-all]
```

## Verification

After setup, verify that everything works correctly:

```bash
# Run the verification script
python verify_wsl_gpu_setup.py

# Or manually test:
python << EOF
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import jax
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

import lrdbenchmark
print(f"LRDBenchmark version: {lrdbenchmark.__version__}")
EOF
```

## Activation

To activate the environment in future sessions:

```bash
# Option 1: Use the activation script
source activate_env.sh

# Option 2: Activate directly
source lrdbenchmark_venv/bin/activate
```

## GPU Compatibility Notes

### RTX 5070 Architecture

The NVIDIA RTX 5070 uses the Blackwell architecture with compute capability **sm_120**. This is a very new architecture:

- **PyTorch**: ✅ Fully supported (CUDA 12.1+)
- **JAX**: ⚠️ May fall back to CPU (sm_120 support may be limited)
- **Performance**: ✅ PyTorch GPU acceleration works perfectly

### Why JAX May Use CPU

JAX may not yet support the sm_120 architecture. This is **normal and expected**. The CPU fallback works fine for JAX operations, and LRDBenchmark uses PyTorch for neural network estimators anyway.

### Expected Performance

- **Neural Network Training**: 10-50x speedup on GPU (PyTorch)
- **Parallel Classical Estimators**: Works on JAX CPU (stable)
- **Large Data Generation**: 5-20x speedup on GPU (PyTorch)

## Troubleshooting

### Issue: `nvidia-smi` not found

**Solution**: 
1. Ensure latest NVIDIA driver is installed on Windows
2. Restart WSL: `wsl --shutdown` (in PowerShell), then restart WSL
3. Check Windows driver version is 516.40+

### Issue: PyTorch CUDA not available

**Solution**:
```bash
# Verify CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Out of Memory (OOM) errors

**Solution**: The RTX 5070 has 7.5 GB VRAM. For large operations:
- Reduce batch size
- Process data in smaller chunks
- Use CPU fallback for very large datasets

### Issue: JAX GPU not working

**This is expected!** JAX may not support sm_120 yet. The CPU fallback works fine:
```python
# JAX will automatically use CPU if GPU not available
import jax
print(jax.default_backend())  # May show 'cpu'
```

This does not affect LRDBenchmark performance, as neural networks use PyTorch (which works with GPU).

### Issue: Virtual environment not activating

**Solution**:
```bash
# Use full path
source /path/to/lrdbenchmark/lrdbenchmark_venv/bin/activate

# Or check permissions
chmod +x lrdbenchmark_venv/bin/activate
```

### Issue: Python version mismatch

**Solution**: LRDBenchmark requires Python 3.10-3.12:
```bash
# Check Python version
python3 --version

# If needed, install correct version (Ubuntu/Debian):
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip

# Create venv with specific version
python3.11 -m venv lrdbenchmark_venv
```

## Environment Variables

You can control GPU behavior with environment variables:

```bash
# Force CPU-only mode (for testing)
export LRDBENCHMARK_FORCE_CPU=1

# JAX-specific settings
export JAX_PLATFORMS=cpu  # Force JAX to use CPU

# PyTorch-specific settings
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
```

## Next Steps

After successful setup:

1. **Test the installation**:
   ```bash
   python verify_wsl_gpu_setup.py
   ```

2. **Run a quick benchmark**:
   ```python
   from lrdbenchmark import ComprehensiveBenchmark
   
   benchmark = ComprehensiveBenchmark(runtime_profile="quick")
   summary = benchmark.run_comprehensive_benchmark(
       data_length=256,
       benchmark_type="classical",
       save_results=False,
   )
   print(summary)
   ```

3. **Check GPU acceleration**:
   ```python
   import torch
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   ```

## Additional Resources

- [GPU Configuration Guide](docs/development_summaries/GPU_CONFIGURATION_GUIDE.md)
- [Conda-Free Setup Guide](config/CONDA_FREE_SETUP.md)
- [LRDBenchmark Documentation](https://lrdbenchmark.readthedocs.io/)

## Support

If you encounter issues:

1. Run the verification script: `python verify_wsl_gpu_setup.py`
2. Check the troubleshooting section above
3. Review GPU configuration guide: `docs/development_summaries/GPU_CONFIGURATION_GUIDE.md`
4. Check NVIDIA driver compatibility: https://docs.nvidia.com/cuda/wsl-user-guide/

---

**Last Updated**: 2024
**GPU**: NVIDIA RTX 5070 (sm_120)
**Recommended CUDA**: 12.1+
**Python**: 3.10-3.12

