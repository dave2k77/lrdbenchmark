#!/usr/bin/env python3
"""
Test script for conda-free LRDBenchmark setup
"""

import sys
import os
from pathlib import Path

def test_environment():
    """Test the conda-free environment"""
    print("🧪 Testing conda-free LRDBenchmark environment...")
    print("=" * 50)
    
    # Test Python
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Test JAX
    try:
        import jax
        import jax.numpy as jnp
        
        print(f"\nJAX version: {jax.__version__}")
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
            return True
            
    except ImportError as e:
        print(f"❌ JAX import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
        return False

def test_other_packages():
    """Test other required packages"""
    print("\n📦 Testing other packages...")
    
    packages = [
        ('numpy', 'np'),
        ('scipy', 'sp'),
        ('matplotlib', 'plt'),
        ('pandas', 'pd'),
        ('torch', 'torch'),
        ('sklearn', 'sklearn'),
    ]
    
    success = True
    for package, alias in packages:
        try:
            exec(f"import {package} as {alias}")
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ {package} import failed: {e}")
            success = False
    
    return success

if __name__ == "__main__":
    print("🚀 LRDBenchmark Conda-Free Test")
    print("=" * 50)
    
    # Test environment
    env_ok = test_environment()
    
    # Test other packages
    packages_ok = test_other_packages()
    
    print("\n" + "=" * 50)
    if env_ok and packages_ok:
        print("🎉 All tests passed! Environment is ready.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)
