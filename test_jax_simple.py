#!/usr/bin/env python3
"""
Simple JAX GPU test script that avoids conda activation issues
"""

import os
import sys

def test_jax_gpu():
    """Test JAX GPU functionality without conda activation"""
    try:
        # Set environment variables to avoid conda issues
        os.environ.pop('CONDA_DEFAULT_ENV', None)
        os.environ.pop('CONDA_PREFIX', None)
        os.environ.pop('CONDA_PROMPT_MODIFIER', None)
        
        import jax
        import jax.numpy as jnp
        
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
        print(f"JAX backend: {jax.default_backend()}")
        
        # Test basic JAX functionality
        x = jnp.array([1, 2, 3, 4, 5])
        y = jnp.sin(x)
        print(f"Test computation: sin([1,2,3,4,5]) = {y}")
        
        # Check if GPU is available
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
    success = test_jax_gpu()
    sys.exit(0 if success else 1)
