#!/usr/bin/env python3
"""
Python-based environment switcher to avoid bash issues
"""

import os
import sys
import subprocess
from pathlib import Path

def find_conda_base():
    """Find conda installation base directory"""
    try:
        result = subprocess.run(['conda', 'info', '--base'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def setup_environment():
    """Set up the environment using direct Python paths"""
    project_root = Path(__file__).parent.absolute()
    
    # Add project to Python path
    pythonpath = os.environ.get('PYTHONPATH', '')
    if str(project_root) not in pythonpath:
        os.environ['PYTHONPATH'] = f"{project_root}:{pythonpath}" if pythonpath else str(project_root)
    
    # Try to find and use conda environment
    conda_base = find_conda_base()
    if conda_base:
        gpu_env = Path(conda_base) / "envs" / "lrdbenchmark_gpu"
        reg_env = Path(conda_base) / "envs" / "lrdbenchmark"
        
        if reg_env.exists():
            # Use primary environment
            env_bin = reg_env / "bin"
            os.environ['PATH'] = f"{env_bin}:{os.environ['PATH']}"
            os.environ['CONDA_DEFAULT_ENV'] = "lrdbenchmark"
            os.environ['CONDA_PREFIX'] = str(reg_env)
            print(f"[env] Using lrdbenchmark environment: {reg_env}")
            return str(env_bin / "python")
        elif gpu_env.exists():
            # Fallback to GPU clone
            env_bin = gpu_env / "bin"
            os.environ['PATH'] = f"{env_bin}:{os.environ['PATH']}"
            os.environ['CONDA_DEFAULT_ENV'] = "lrdbenchmark_gpu"
            os.environ['CONDA_PREFIX'] = str(gpu_env)
            print(f"[env] Using lrdbenchmark_gpu environment: {gpu_env}")
            return str(env_bin / "python")
    
    # Fallback to system Python
    print("[env] Using system Python")
    return sys.executable

def test_jax():
    """Test JAX functionality"""
    try:
        import jax
        import jax.numpy as jnp
        
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
        print(f"JAX backend: {jax.default_backend()}")
        
        # Test computation
        x = jnp.array([1, 2, 3, 4, 5])
        y = jnp.sin(x)
        print(f"Test computation: sin([1,2,3,4,5]) = {y}")
        
        # Check GPU
        gpu_devices = [d for d in jax.devices() if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        if gpu_devices:
            print(f"✅ JAX GPU detected: {gpu_devices}")
            return True
        else:
            print("⚠️  JAX running on CPU only")
            return False
            
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
        return False

if __name__ == "__main__":
    print("Setting up environment...")
    python_path = setup_environment()
    print(f"Using Python: {python_path}")
    
    print("\nTesting JAX...")
    success = test_jax()
    
    if success:
        print("\n✅ Environment setup successful!")
    else:
        print("\n❌ Environment setup failed!")
        sys.exit(1)
