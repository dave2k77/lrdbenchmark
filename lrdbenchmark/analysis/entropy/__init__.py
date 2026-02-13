"""
Entropy-based estimators for LRDBench.

This module provides Multiscale Entropy (MSE) and Multivariate Multiscale
Entropy (mvMSE) estimators that measure regularity/complexity as a proxy
for long-range dependence (LRD).
"""

from .mse_estimator import MSEEstimator
from .mvmse_estimator import MultivariateMSEEstimator

__all__ = [
    "MSEEstimator",
    "MultivariateMSEEstimator",
]
