import torch
import jax
import jax.numpy as jnp

print("--- PyTorch ---")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    x = torch.randn(100, 100).cuda()
    y = x @ x.T
    print(f"PyTorch GPU Computation: Success!")

print("\n--- JAX ---")
print(f"JAX Backend: {jax.default_backend()}")
print(f"JAX Devices: {jax.devices()}")
try:
    x = jnp.ones((100, 100))
    y = jnp.sum(x)
    print(f"JAX Computation Result: {y}")
except Exception as e:
    print(f"JAX Computation Failed: {e}")
