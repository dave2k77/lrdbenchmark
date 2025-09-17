# Conda-Free LRDBenchmark Setup

This setup avoids conda entirely to prevent terminal crashes and uses only system Python with virtual environments.

## 🚀 Quick Setup

### Step 1: Run the setup script
```bash
python3 setup_conda_free.py
```

This will:
- Create a virtual environment at `~/lrdbenchmark_venv`
- Install JAX with CUDA support
- Install all required packages
- Test JAX functionality
- Create an activation script

### Step 2: Activate the environment
```bash
source activate_lrdbenchmark.sh
```

### Step 3: Test the setup
```bash
python3 test_conda_free.py
```

## 🔧 Manual Setup (if script fails)

### Create virtual environment
```bash
python3 -m venv ~/lrdbenchmark_venv
source ~/lrdbenchmark_venv/bin/activate
```

### Install JAX with CUDA
```bash
pip install --upgrade pip
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Install other packages
```bash
pip install numpy scipy matplotlib pandas seaborn torch torchvision scikit-learn plotly joblib numba sympy pytest sphinx black flake8 mypy pre-commit
```

### Test JAX
```bash
python -c "import jax; print(f'JAX devices: {jax.devices()}'); print(f'JAX backend: {jax.default_backend()}')"
```

## 🎯 Benefits

- ✅ No conda dependency
- ✅ No terminal crashes
- ✅ Clean virtual environment
- ✅ JAX GPU support
- ✅ All required packages included

## 🔍 Troubleshooting

### If JAX GPU doesn't work
```bash
# Check CUDA installation
nvidia-smi

# Try CPU-only JAX
pip uninstall jax jaxlib
pip install jax jaxlib
```

### If virtual environment doesn't activate
```bash
# Use full path
source ~/lrdbenchmark_venv/bin/activate

# Or use Python directly
~/lrdbenchmark_venv/bin/python your_script.py
```

### If packages are missing
```bash
# Activate environment first
source ~/lrdbenchmark_venv/bin/activate

# Install missing package
pip install package_name
```

## 📁 Files Created

- `~/lrdbenchmark_venv/` - Virtual environment directory
- `activate_lrdbenchmark.sh` - Activation script
- `setup_conda_free.py` - Setup script
- `test_conda_free.py` - Test script

## 🎉 Success Indicators

When everything works, you should see:
- `JAX devices: [CudaDevice(id=0)]` (GPU) or `[CpuDevice(id=0)]` (CPU)
- `JAX backend: gpu` (GPU) or `cpu` (CPU)
- All package imports successful
- No terminal crashes
