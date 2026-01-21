# lrdbenchmark Examples

This directory contains progressive examples demonstrating lrdbenchmark usage from basic to production-ready patterns. The library provides 20 estimators (13 classical, 3 ML, 4 neural networks) for comprehensive long-range dependence analysis.

## Example Progression

### 1. Quick Start (`01_quickstart.py`)
**5-line basic usage**
- Generate synthetic fBm data
- Estimate Hurst parameter with R/S analysis
- Minimal dependencies

### 2. CPU-Only Configuration (`02_cpu_only.py`)
**CPU-only usage patterns**
- Multiple classical estimators
- No GPU dependencies
- Reliable for all environments

### 3. Optional GPU Usage (`03_gpu_optional.py`)
**GPU acceleration with fallback**
- Check GPU availability
- Automatic CPU fallback
- Neural network estimators

### 4. Production Deployment (`04_production.py`)
**Production-ready patterns**
- Error handling and recovery
- Performance monitoring
- Batch processing
- Comprehensive benchmarking

## Running Examples

### Basic Installation
```bash
# Install core package only
pip install lrdbenchmark

# Run CPU-only examples
python examples/01_quickstart.py
python examples/02_cpu_only.py
```

### With GPU Acceleration
```bash
# Install with GPU support
pip install lrdbenchmark[accel-pytorch]

# Run GPU examples
python examples/03_gpu_optional.py
python examples/04_production.py
```

### Full Installation
```bash
# Install all optional dependencies
pip install lrdbenchmark[accel-all]

# Run all examples
for example in examples/*.py; do
    echo "Running $example"
    python "$example"
    echo "---"
done
```

## Example Outputs

### Quick Start
```
Estimated H: 0.698
```

### CPU-Only
```
CPU-Only Estimation Results:
----------------------------------------
    R/S: H = 0.698
    DFA: H = 0.701
    GPH: H = 0.695

True H: 0.700
All estimators should give results close to 0.7
```

### GPU Optional
```
GPU Available: True
GPU Info: {'available': True, 'device_count': 1, ...}

Classical Estimator (CPU):
R/S Analysis: H = 0.698

Neural Network Estimator:
CNN: H = 0.702
✓ GPU acceleration used

True H: 0.700
```

### Production
```
lrdbenchmark Production Example
==================================================
Generating test data...
Generated 1000 data points

Running estimators...
✓     R/S: H = 0.698 (took 0.001s)
✓     DFA: H = 0.701 (took 0.002s)
✓     GPH: H = 0.695 (took 0.003s)

Summary: 3/3 estimators successful

Running comprehensive benchmark...
✓ Benchmark completed successfully
  Results: 3 estimator categories

Production workflow completed!
```

## Troubleshooting

### Import Errors
If you get import errors, ensure you have the required dependencies:
```bash
pip install lrdbenchmark[accel-pytorch]  # For neural networks
pip install lrdbenchmark[accel-jax]      # For JAX acceleration
```

### GPU Issues
If GPU examples fail, they will automatically fall back to CPU. For CPU-only usage:
```python
# Force CPU-only mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Memory Issues
For large datasets or memory-constrained environments:
```python
# Use smaller batch sizes
from lrdbenchmark.gpu import suggest_batch_size
batch_size = suggest_batch_size(data_size=10000, sequence_length=1000)
```

## Next Steps

After running these examples, explore:
- [Documentation](https://lrdbenchmark.readthedocs.io/)
- [API Reference](https://lrdbenchmark.readthedocs.io/api/)
- [Notebooks](../notebooks/) for interactive examples
- [Benchmark Scripts](../scripts/benchmarks/) for performance analysis
