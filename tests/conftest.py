"""
Pytest configuration and fixtures for LRDBenchmark tests.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from lrdbenchmark.random_manager import get_random_manager, initialise_global_rng


@pytest.fixture(autouse=True)
def reset_random_manager():
    """Reset the global RNG manager before each test."""
    initialise_global_rng(42)
    yield


@pytest.fixture
def sample_data():
    """Generate sample time series data for testing."""
    rng = get_random_manager().spawn_generator("tests:sample_data")
    return rng.normal(size=1000)


@pytest.fixture
def fbm_data():
    """Generate fractional Brownian motion data for testing."""
    rng = get_random_manager().spawn_generator("tests:fbm_data")
    # Simple fBm-like data
    n = 1000
    H = 0.7
    dt = 1.0
    t = np.arange(n) * dt
    
    # Generate fBm using approximate method
    increments = rng.normal(size=n - 1)
    fbm = np.zeros(n)
    fbm[0] = 0
    
    for i in range(1, n):
        fbm[i] = fbm[i-1] + dt**H * increments[i-1]
    
    return fbm


@pytest.fixture
def short_data():
    """Generate short time series for edge case testing."""
    rng = get_random_manager().spawn_generator("tests:short_data")
    return rng.normal(size=50)


@pytest.fixture
def long_data():
    """Generate long time series for performance testing."""
    rng = get_random_manager().spawn_generator("tests:long_data")
    return rng.normal(size=10000)


@pytest.fixture
def contaminated_data():
    """Generate data with contamination for robustness testing."""
    rng = get_random_manager().spawn_generator("tests:contaminated_data")
    base_data = rng.normal(size=1000)
    
    # Add outliers
    contaminated = base_data.copy()
    outlier_indices = rng.choice(1000, size=50, replace=False)
    contaminated[outlier_indices] += 5 * rng.normal(size=50)
    
    return contaminated


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_gpu_available():
    """Mock GPU availability for testing."""
    import lrdbenchmark.gpu as gpu_module
    
    original_is_available = gpu_module.is_available
    gpu_module.is_available = lambda: True
    
    yield
    
    gpu_module.is_available = original_is_available


@pytest.fixture
def mock_gpu_unavailable():
    """Mock GPU unavailability for testing."""
    import lrdbenchmark.gpu as gpu_module
    
    original_is_available = gpu_module.is_available
    gpu_module.is_available = lambda: False
    
    yield
    
    gpu_module.is_available = original_is_available


@pytest.fixture(scope="session")
def test_models():
    """Create test data models for integration testing."""
    try:
        from lrdbenchmark.models.data_models import FBMModel, FGNModel, ARFIMAModel
        
        models = {
            'fbm': FBMModel(H=0.7, sigma=1.0),
            'fgn': FGNModel(H=0.7, sigma=1.0),
            'arfima': ARFIMAModel(d=0.2, phi=[0.5], theta=[0.3])
        }
        return models
    except ImportError:
        pytest.skip("Data models not available")


@pytest.fixture(scope="session")
def test_estimators():
    """Create test estimators for integration testing."""
    try:
        from lrdbenchmark import RSEstimator, DFAEstimator, GPHEstimator
        
        estimators = {
            'rs': RSEstimator(),
            'dfa': DFAEstimator(),
            'gph': GPHEstimator()
        }
        return estimators
    except ImportError:
        pytest.skip("Estimators not available")


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark testing."""
    return {
        'n_samples': 100,
        'n_trials': 5,
        'use_gpu': False,
        'clear_memory_between_runs': True
    }
