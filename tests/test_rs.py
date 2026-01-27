import numpy as np
import pytest

from lrdbenchmark.analysis.temporal.rs_estimator import RSEstimator


class TestRSEstimator:
    """Test cases for RSEstimator."""
    
    def test_valid_parameters(self):
        """Test valid parameter initialization."""
        estimator = RSEstimator(min_window_size=10, max_window_size=100)
        # RSEstimator stores parameters internally - access via .parameters
        assert estimator.parameters['min_block_size'] == 10
        assert estimator.parameters['max_block_size'] == 100
        assert estimator.parameters['window_sizes'] is None
        assert estimator.parameters['overlap'] is False
    
    def test_invalid_min_window_size(self):
        """Test invalid minimum window size."""
        with pytest.raises(ValueError, match="min_block_size must be at least 4"):
            RSEstimator(min_window_size=3)
    
    def test_invalid_max_window_size(self):
        """Test invalid maximum window size."""
        with pytest.raises(ValueError, match="max_block_size must be greater than min_block_size"):
            RSEstimator(min_window_size=10, max_window_size=5)
    
    def test_invalid_window_sizes_positive(self):
        """Test invalid window sizes (non-positive)."""
        with pytest.raises(ValueError, match="Window sizes must be positive"):
            RSEstimator(window_sizes=[-3, 10, 20])
    
    def test_invalid_window_sizes_count(self):
        """Test invalid window sizes count."""
        with pytest.raises(ValueError, match="Need at least 3 window sizes"):
            RSEstimator(window_sizes=[10, 20])
    
    def test_estimation_length_and_type(self):
        """Test estimation returns correct length and type."""
        estimator = RSEstimator()
        
        # Generate test data (fBm-like)
        np.random.seed(42)
        data = np.cumsum(np.random.normal(0, 1, 1000))
        
        results = estimator.estimate(data)
        
        assert isinstance(results, dict)
        assert 'hurst_parameter' in results
        assert 'block_sizes' in results  # Current API uses block_sizes
        assert 'rs_values' in results
        assert 'r_squared' in results
        assert 'std_error' in results
        assert 'confidence_interval' in results
        
        assert isinstance(results['hurst_parameter'], float)
        assert isinstance(results['block_sizes'], list)
        assert isinstance(results['rs_values'], list)
        assert len(results['block_sizes']) == len(results['rs_values'])
        assert len(results['block_sizes']) >= 3
    
    def test_estimation_with_short_data(self):
        """Test estimation with insufficient data."""
        estimator = RSEstimator()
        data = np.random.normal(0, 1, 15)  # Too short
        
        with pytest.raises(ValueError, match="Need at least 3"):
            estimator.estimate(data)
    
    def test_estimation_with_large_window(self):
        """Test estimation with window size too large."""
        estimator = RSEstimator(min_window_size=1000, max_window_size=2000)
        data = np.random.normal(0, 1, 100)  # Too short for large windows
        
        with pytest.raises(ValueError, match="Need at least 3"):
            estimator.estimate(data)
    
    def test_reproducibility(self):
        """Test that estimation is reproducible with same seed."""
        estimator1 = RSEstimator()
        estimator2 = RSEstimator()
        
        np.random.seed(42)
        data1 = np.cumsum(np.random.normal(0, 1, 1000))
        
        np.random.seed(42)
        data2 = np.cumsum(np.random.normal(0, 1, 1000))
        
        results1 = estimator1.estimate(data1)
        results2 = estimator2.estimate(data2)
        
        assert np.allclose(results1['hurst_parameter'], results2['hurst_parameter'])
        assert np.allclose(results1['r_squared'], results2['r_squared'])
    
    def test_custom_window_sizes(self):
        """Test estimation with custom window sizes."""
        window_sizes = [10, 20, 40, 80]
        estimator = RSEstimator(window_sizes=window_sizes)
        
        np.random.seed(42)
        data = np.cumsum(np.random.normal(0, 1, 1000))
        
        results = estimator.estimate(data)
        
        # Results use block_sizes key
        assert results['block_sizes'] == window_sizes
        assert len(results['rs_values']) == len(window_sizes)
    
    def test_confidence_intervals(self):
        """Test confidence interval in results."""
        estimator = RSEstimator()
        
        np.random.seed(42)
        data = np.cumsum(np.random.normal(0, 1, 1000))
        
        results = estimator.estimate(data)
        ci = results['confidence_interval']
        
        assert isinstance(ci, list)
        assert len(ci) == 2
        assert ci[0] < ci[1]
    
    def test_estimation_quality(self):
        """Test estimation quality metrics are in results."""
        estimator = RSEstimator()
        
        np.random.seed(42)
        data = np.cumsum(np.random.normal(0, 1, 1000))
        
        results = estimator.estimate(data)
        
        # Quality metrics are directly in results
        assert 'r_squared' in results
        assert 'p_value' in results
        assert 'std_error' in results
        
        assert 0 <= results['r_squared'] <= 1
        assert 0 <= results['p_value'] <= 1
        assert results['std_error'] > 0
    
    def test_string_representations(self):
        """Test string representations exist."""
        estimator = RSEstimator(min_window_size=10, max_window_size=100)
        
        # Check that repr doesn't raise
        repr_str = repr(estimator)
        assert 'RSEstimator' in repr_str
    
    def test_hurst_value_range(self):
        """Test that Hurst estimates are in reasonable range."""
        estimator = RSEstimator()
        
        np.random.seed(42)
        # Generate fBm-like data (should give H near 0.5 for random walk)
        data = np.cumsum(np.random.normal(0, 1, 2000))
        
        results = estimator.estimate(data)
        
        # R/S can produce estimates outside [0,1] and is known to be biased
        # Just check it's finite and roughly in expected range
        assert 0.0 < results['hurst_parameter'] < 1.5
    
    def test_optimization_info(self):
        """Test get_optimization_info returns expected structure."""
        estimator = RSEstimator()
        
        info = estimator.get_optimization_info()
        
        assert 'current_framework' in info
        assert 'jax_available' in info
        assert 'numba_available' in info
