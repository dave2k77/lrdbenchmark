#!/usr/bin/env python3
"""
Create a clean conda environment without problematic compiler packages
"""

import subprocess
import sys
import os

def run_command(cmd, check=True):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return result

def create_clean_environment():
    """Create a clean conda environment"""
    print("Creating clean conda environment...")
    
    # Remove existing environment if it exists
    run_command(['conda', 'env', 'remove', '-n', 'lrdbenchmark_clean', '-y'], check=False)
    
    # Create new environment with minimal packages
    result = run_command([
        'conda', 'create', '-n', 'lrdbenchmark_clean', 
        'python=3.11', 
        'numpy', 'scipy', 'matplotlib', 'pandas',
        '-c', 'conda-forge', '-y'
    ])
    
    if result is None:
        return False
    
    print("Installing JAX with CUDA support...")
    
    # Install JAX with CUDA support using pip
    result = run_command([
        'conda', 'run', '-n', 'lrdbenchmark_clean',
        'pip', 'install', 'jax[cuda12_pip]', 
        '-f', 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
    ])
    
    if result is None:
        return False
    
    print("Installing additional packages...")
    
    # Install other required packages
    packages = [
        'torch', 'torchvision', 'scikit-learn', 'seaborn', 'plotly',
        'joblib', 'numba', 'sympy', 'pytest', 'sphinx', 'black', 'flake8'
    ]
    
    for package in packages:
        result = run_command([
            'conda', 'run', '-n', 'lrdbenchmark_clean',
            'pip', 'install', package
        ])
        if result is None:
            print(f"Warning: Failed to install {package}")
    
    print("Testing JAX...")
    
    # Test JAX
    test_script = '''
import jax
import jax.numpy as jnp
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# Test computation
x = jnp.array([1, 2, 3, 4, 5])
y = jnp.sin(x)
print(f"Test computation: {y}")

# Check GPU
gpu_devices = [d for d in jax.devices() if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
if gpu_devices:
    print(f"✅ JAX GPU detected: {gpu_devices}")
else:
    print("⚠️  JAX running on CPU only")
'''
    
    result = run_command([
        'conda', 'run', '-n', 'lrdbenchmark_clean',
        'python', '-c', test_script
    ])
    
    if result is None:
        return False
    
    print("✅ Clean environment created successfully!")
    print("To use it, run: conda activate lrdbenchmark_clean")
    return True

if __name__ == "__main__":
    success = create_clean_environment()
    sys.exit(0 if success else 1)
