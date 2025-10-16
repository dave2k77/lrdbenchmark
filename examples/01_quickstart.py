#!/usr/bin/env python3
"""
LRDBenchmark Quick Start Example

This example demonstrates the basic usage of LRDBenchmark in just 5 lines.
"""

import numpy as np
from lrdbenchmark import FBMModel, RSEstimator

# Generate synthetic fractional Brownian motion
fbm = FBMModel(H=0.7, sigma=1.0)
data = fbm.generate(length=1000, seed=42)

# Estimate Hurst parameter using R/S analysis
estimator = RSEstimator()
result = estimator.estimate(data)

print(f"Estimated H: {result['hurst_parameter']:.3f}")  # Should be ~0.7
