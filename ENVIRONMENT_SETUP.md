# LRDBenchmark Environment Setup

## 🎯 Overview

This document describes the dedicated conda environment setup for the LRDBenchmark project, including all required dependencies and tools for comprehensive long-range dependence analysis.

## 🚀 Quick Start

### Activate Environment
```bash
# Navigate to project directory
cd /home/davianc/Documents/LRDBenchmark

# Activate environment using the provided script
./activate_lrdbenchmark_env.sh
```

### Manual Activation
```bash
# Source conda
source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate lrdbenchmark
```

## 📦 Environment Details

### Python Version
- **Python**: 3.11.13
- **Conda Environment**: `lrdbenchmark`
- **Location**: `/home/davianc/miniconda3/envs/lrdbenchmark`

### GPU Support
- **GPU**: NVIDIA GeForce RTX 5070 Laptop GPU (7.5 GB VRAM)
- **CUDA**: 13.0 available, PyTorch 2.5.1+cu121 with CUDA 12.1
- **JAX**: 0.7.2 with CUDA support (CPU fallback available)
- **Status**: ✅ GPU acceleration ready for LRDBenchmark

### Core Dependencies

#### Scientific Computing
- **NumPy**: 2.0.1 - Numerical computing
- **SciPy**: 1.16.0 - Scientific computing
- **Pandas**: 2.3.3 - Data manipulation
- **Matplotlib**: 3.10.6 - Plotting
- **Seaborn**: 0.13.2 - Statistical visualization

#### Machine Learning & Neural Networks
- **Scikit-learn**: 1.7.2 - Machine learning
- **PyTorch**: 2.8.0+cpu - Deep learning framework
- **Torchvision**: 0.23.0+cpu - Computer vision
- **Torchaudio**: 2.8.0+cpu - Audio processing

#### Advanced Computing
- **JAX**: 0.7.2 - High-performance ML
- **JAXlib**: 0.7.2 - JAX backend
- **Numba**: 0.62.1 - JIT compilation
- **Statsmodels**: 0.14.5 - Statistical modeling
- **Arch**: 7.2.0 - Econometric modeling

#### Signal Processing
- **PyWavelets**: 1.9.0 - Wavelet analysis
- **NetworkX**: 3.3 - Graph analysis

#### Development Tools
- **Jupyter**: 7.4.5 - Interactive notebooks
- **Notebook**: 7.4.5 - Jupyter notebook server
- **IPython**: 8.28.0 - Enhanced Python shell

## 🧪 Testing the Environment

### Basic Import Test
```python
import lrdbenchmark
print("LRDBenchmark version:", lrdbenchmark.__version__)
```

### Comprehensive Test
```python
# Test all major components
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import torch
import jax
import numba
import statsmodels
import arch
import pywt
import networkx as nx
import lrdbenchmark

print("✅ All packages imported successfully!")
```

### GPU Test
```python
# Test GPU availability
import torch
import jax

print(f"PyTorch CUDA: {torch.cuda.is_available()}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# Test GPU computation
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    y = torch.mm(x, x.t())
    print(f"✅ PyTorch GPU test: {y.shape} on {device}")
else:
    print("❌ PyTorch GPU not available")

print("✅ GPU test completed!")
```

## 📚 Available Notebooks

The environment includes 5 comprehensive demonstration notebooks:

1. **`01_data_generation_and_visualisation.ipynb`**
   - Data generation from all models (FBM, FGN, ARFIMA, MRW, Alpha-Stable)
   - Comprehensive visualisations
   - Quality assessment

2. **`02_estimation_and_validation.ipynb`**
   - All estimator categories (Classical, ML, Neural)
   - Statistical validation
   - Performance comparison

3. **`03_custom_models_and_estimators.ipynb`**
   - Extending the library
   - Custom data models and estimators
   - Best practices

4. **`04_comprehensive_benchmarking.ipynb`**
   - Full benchmarking system
   - Contamination testing
   - Performance metrics

5. **`05_leaderboard_generation.ipynb`**
   - Leaderboard creation
   - Performance ranking
   - Results visualisation

## 🛠️ Usage Examples

### Running Jupyter Notebooks
```bash
# Activate environment
./activate_lrdbenchmark_env.sh

# Start Jupyter
jupyter notebook
```

### Running Python Scripts
```bash
# Activate environment
./activate_lrdbenchmark_env.sh

# Run script
python your_script.py
```

### Installing Additional Packages
```bash
# Activate environment
./activate_lrdbenchmark_env.sh

# Install package
pip install package_name
```

## 🔧 Environment Management

### Deactivating
```bash
conda deactivate
```

### Updating Packages
```bash
# Activate environment
./activate_lrdbenchmark_env.sh

# Update specific package
pip install --upgrade package_name

# Update all packages
pip list --outdated
```

### Recreating Environment
```bash
# Remove existing environment
conda env remove -n lrdbenchmark

# Recreate from requirements
conda create -n lrdbenchmark python=3.11
conda activate lrdbenchmark
pip install -r requirements.txt
```

## 📋 Requirements File

The environment includes a `requirements.txt` file with all installed packages for reproducibility.

## ⚠️ Notes

- The environment uses CPU-only PyTorch for compatibility
- JAX is configured for CPU usage
- All packages are compatible with Python 3.11
- The environment is optimized for long-range dependence analysis

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure environment is activated
2. **Version Conflicts**: Use `pip install --upgrade package_name`
3. **Memory Issues**: Consider reducing data size in notebooks
4. **Performance**: Use JAX for GPU acceleration if available

### Getting Help

- Check the LRDBenchmark documentation
- Review notebook examples
- Check package-specific documentation
- Use the provided test scripts

## 🎉 Success Indicators

When the environment is properly set up, you should see:
- ✅ LRDBenchmark version displayed
- ✅ All imports working without errors
- ✅ Jupyter notebooks running smoothly
- ✅ All demonstration notebooks executable

---

**Environment Status**: ✅ Ready for LRDBenchmark development and analysis
