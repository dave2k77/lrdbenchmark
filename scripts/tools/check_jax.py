import jax
import sys
import os

print(f"JAX Version: {jax.__version__}")
try:
    import jaxlib
    print(f"Jaxlib Version: {jaxlib.__version__}")
except ImportError:
    print("Jaxlib not found")

try:
    print(f"Default Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
except Exception as e:
    print(f"Error getting jax devices: {e}")

# Check for CUDA-related environment variables
for key in os.environ:
    if "CUDA" in key:
        print(f"{key}: {os.environ[key]}")
