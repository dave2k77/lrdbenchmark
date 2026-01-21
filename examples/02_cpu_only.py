#!/usr/bin/env python3
"""
LRDBenchmark CPU-Only Configuration Example

This example shows how to use LRDBenchmark in CPU-only mode,
ensuring no GPU dependencies are required.
"""

import numpy as np
from lrdbenchmark import FBMModel, RSEstimator, DFAEstimator, GPHEstimator

# Generate synthetic data
fbm = FBMModel(H=0.7, sigma=1.0)
data = fbm.generate(length=1000, seed=42)

# Test multiple classical estimators (all CPU-only)
estimators = {
    'R/S': RSEstimator(),
    'DFA': DFAEstimator(), 
    'GPH': GPHEstimator()
}

print("CPU-Only Estimation Results:")
print("-" * 40)

for name, estimator in estimators.items():
    result = estimator.estimate(data)
    print(f"{name:>6}: H = {result['hurst_parameter']:.3f}")

print(f"\nTrue H: 0.700")
print("All estimators should give results close to 0.7")
