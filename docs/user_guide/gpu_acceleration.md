# GPU Acceleration Guide

## Overview

LRDBenchmark supports optional GPU acceleration for neural network estimators and data generation. GPU acceleration is **purely optional** - the library works perfectly in CPU-only mode and will automatically fall back to CPU when GPU is not available.

## Quick Decision Guide

**Should I use GPU?**
- ✅ **Yes** if you have: NVIDIA GPU with CUDA support, large datasets (>10k points), or need fast neural network inference
- ❌ **No** if you have: CPU-only environment, small datasets (<1k points), or want maximum compatibility

## Installation Options

### CPU-Only Installation (Recommended for most users)
```bash
pip install lrdbenchmark
```
This installs only the core dependencies and works on any system.

### GPU-Accelerated Installation
```bash
# PyTorch GPU support (neural networks)
pip install lrdbenchmark[accel-pytorch]

# JAX GPU support (data generation)
pip install lrdbenchmark[accel-jax]

# All GPU acceleration
pip install lrdbenchmark[accel-all]
```

## Usage Patterns

### 1. CPU-Only Usage (Default)
```python
from lrdbenchmark import FBMModel, RSEstimator

# All estimators default to CPU
model = FBMModel(H=0.7)
data = model.generate(length=1000)
estimator = RSEstimator()
result = estimator.estimate(data)
```

### 2. Optional GPU Usage
```python
from lrdbenchmark import FBMModel, gpu_is_available

# Check GPU availability
if gpu_is_available():
    print("GPU acceleration available")
    # Use GPU-accelerated estimators
else:
    print("Using CPU-only mode")
    # Use classical estimators
```

### 3. Explicit GPU Control
```python
# Neural networks with GPU control
from lrdbenchmark.models.pretrained_models import TransformerPretrainedModel

# Force CPU usage
model = TransformerPretrainedModel(use_gpu=False)

# Try GPU (with fallback)
model = TransformerPretrainedModel(use_gpu=True)
```

## GPU Requirements

### Hardware Requirements
- **NVIDIA GPU** with CUDA support
- **Minimum**: 2GB VRAM
- **Recommended**: 4GB+ VRAM for large datasets
- **Optimal**: 8GB+ VRAM for production workloads

### Software Requirements
- **CUDA**: Version 11.0+ (for PyTorch) or 12.0+ (for JAX)
- **PyTorch**: 1.9.0+ with CUDA support
- **JAX**: 0.3.0+ with CUDA support (optional)

## Performance Expectations

### Speedup Factors
- **Neural Networks**: 5-20x faster on GPU
- **Data Generation**: 2-5x faster on GPU
- **Classical Estimators**: No GPU benefit (CPU-optimized)

### Memory Usage
- **Small datasets** (<1k points): GPU overhead not worth it
- **Medium datasets** (1k-10k points): 2-5x speedup
- **Large datasets** (>10k points): 5-20x speedup

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Error: CUDA out of memory
# Solution: Use CPU fallback
from lrdbenchmark.models.pretrained_models import TransformerPretrainedModel
model = TransformerPretrainedModel(use_gpu=False)
```

#### 2. GPU Not Detected
```python
# Check GPU availability
from lrdbenchmark import gpu_is_available, get_device_info

print(f"GPU Available: {gpu_is_available()}")
print(f"Device Info: {get_device_info()}")
```

#### 3. JAX GPU Issues
```python
# Force JAX to use CPU
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

# Or use PyTorch instead
from lrdbenchmark.models.pretrained_models import TransformerPretrainedModel
model = TransformerPretrainedModel(use_gpu=True)  # Uses PyTorch
```

### Environment Variables

```bash
# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"

# Enable GPU debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
```

## Best Practices

### 1. Start with CPU-Only
```python
# Always test CPU-first
from lrdbenchmark import RSEstimator, DFAEstimator
estimator = RSEstimator()  # CPU-only, always works
```

### 2. Add GPU Gradually
```python
# Add GPU acceleration for specific use cases
if gpu_is_available() and len(data) > 1000:
    # Use GPU for large datasets
    model = TransformerPretrainedModel(use_gpu=True)
else:
    # Use CPU for small datasets
    estimator = RSEstimator()
```

### 3. Monitor Memory Usage
```python
from lrdbenchmark import get_device_info, clear_cache

# Check memory before large operations
info = get_device_info()
if info['memory_free'] < 1.0:  # Less than 1GB free
    clear_cache()

# Clear cache after operations
clear_cache()
```

### 4. Handle Errors Gracefully
```python
try:
    model = TransformerPretrainedModel(use_gpu=True)
    result = model.estimate(data)
except Exception as e:
    print(f"GPU failed: {e}")
    # Fallback to CPU
    estimator = RSEstimator()
    result = estimator.estimate(data)
```

## Configuration Examples

### Production Deployment
```python
# Production: CPU-first with optional GPU
from lrdbenchmark import gpu_is_available

def get_optimal_estimator(data_size):
    if gpu_is_available() and data_size > 5000:
        return TransformerPretrainedModel(use_gpu=True)
    else:
        return RSEstimator()  # Always reliable
```

### Development Environment
```python
# Development: Force CPU for consistency
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# All operations will use CPU
from lrdbenchmark import FBMModel, RSEstimator
```

### Research Environment
```python
# Research: Use GPU when available
from lrdbenchmark import gpu_is_available, get_device_info

if gpu_is_available():
    info = get_device_info()
    print(f"Using GPU: {info['device_name']}")
    print(f"Memory: {info['memory_free']:.1f}GB free")
```

## Performance Monitoring

### Check GPU Status
```python
from lrdbenchmark import get_device_info

info = get_device_info()
print(f"GPU Available: {info['available']}")
print(f"Device Count: {info['device_count']}")
print(f"Device Name: {info['device_name']}")
print(f"Memory Total: {info['memory_total']:.1f}GB")
print(f"Memory Free: {info['memory_free']:.1f}GB")
```

### Monitor Performance
```python
import time
from lrdbenchmark import gpu_is_available

def benchmark_estimator(estimator, data, name):
    start = time.time()
    result = estimator.estimate(data)
    elapsed = time.time() - start
    
    gpu_status = "GPU" if gpu_is_available() else "CPU"
    print(f"{name} ({gpu_status}): {elapsed:.3f}s")
    return result
```

## Migration Guide

### From GPU-Required to GPU-Optional
```python
# Old code (GPU required)
model = TransformerPretrainedModel()  # Assumed GPU

# New code (GPU optional)
model = TransformerPretrainedModel(use_gpu=False)  # Explicit CPU
# or
model = TransformerPretrainedModel(use_gpu=True)   # Try GPU with fallback
```

### From JAX to PyTorch
```python
# Old code (JAX)
import jax.numpy as jnp
data = jnp.array(data)

# New code (PyTorch with fallback)
from lrdbenchmark.models.pretrained_models import TransformerPretrainedModel
model = TransformerPretrainedModel(use_gpu=True)  # Uses PyTorch
```

## Support

### Getting Help
- **Documentation**: [https://lrdbenchmark.readthedocs.io/](https://lrdbenchmark.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/dave2k77/lrdbenchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dave2k77/lrdbenchmark/discussions)

### Common Solutions
1. **GPU not working**: Use CPU-only mode (default)
2. **Memory errors**: Reduce batch size or use CPU
3. **Import errors**: Install with `pip install lrdbenchmark[accel-pytorch]`
4. **Performance issues**: Check GPU memory and use appropriate batch sizes

### Reporting Issues
When reporting GPU-related issues, include:
- GPU model and driver version
- CUDA version
- PyTorch version
- Error messages and stack traces
- System specifications
