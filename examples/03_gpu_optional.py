#!/usr/bin/env python3
"""
LRDBenchmark Optional GPU Usage Example

This example demonstrates how to use GPU acceleration when available,
with automatic fallback to CPU when GPU is not available.
"""

import numpy as np
from lrdbenchmark import FBMModel, RSEstimator, gpu_is_available, get_device_info

# Check GPU availability
print(f"GPU Available: {gpu_is_available()}")
if gpu_is_available():
    info = get_device_info()
    print(f"GPU Info: {info}")

# Generate synthetic data
fbm = FBMModel(H=0.7, sigma=1.0)
data = fbm.generate(n=1000, seed=42)

# Test classical estimator (always CPU)
print("\nClassical Estimator (CPU):")
rs_estimator = RSEstimator()
result = rs_estimator.estimate(data)
print(f"R/S Analysis: H = {result['hurst_parameter']:.3f}")

# Test neural network estimators (GPU optional)
try:
    from lrdbenchmark.analysis.machine_learning.cnn_estimator_unified import CNNEstimator
    
    print("\nNeural Network Estimator:")
    cnn_estimator = CNNEstimator()
    result = cnn_estimator.estimate(data)
    print(f"CNN: H = {result['hurst_parameter']:.3f}")
    
    if gpu_is_available():
        print("✓ GPU acceleration used")
    else:
        print("✓ CPU fallback used")
        
except ImportError:
    print("\nNeural network estimators not available")
    print("Install with: pip install lrdbenchmark[accel-pytorch]")

print(f"\nTrue H: 0.700")
