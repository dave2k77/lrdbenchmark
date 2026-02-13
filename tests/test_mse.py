"""
Tests for Multiscale Entropy (MSE) and Multivariate MSE estimators.
"""

import numpy as np
import pytest

from lrdbenchmark.analysis.entropy.mse_estimator import MSEEstimator
from lrdbenchmark.analysis.entropy.mvmse_estimator import MultivariateMSEEstimator


# ======================================================================
# MSEEstimator Tests
# ======================================================================
class TestMSEEstimator:
    """Test suite for the univariate MSE estimator."""

    def test_valid_parameters(self):
        """Test valid parameter initialisation."""
        est = MSEEstimator(m=2, r=0.15, max_scale=10)
        assert est.parameters["m"] == 2
        assert est.parameters["r"] == 0.15
        assert est.parameters["max_scale"] == 10

    def test_invalid_m(self):
        """Test that m < 1 raises ValueError."""
        with pytest.raises(ValueError, match="m must be a positive integer"):
            MSEEstimator(m=0)

    def test_invalid_r(self):
        """Test that r <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="r must be positive"):
            MSEEstimator(r=-0.1)

    def test_invalid_max_scale(self):
        """Test that max_scale < 2 raises ValueError."""
        with pytest.raises(ValueError, match="max_scale must be an integer >= 2"):
            MSEEstimator(max_scale=1)

    def test_short_data_raises(self):
        """Test that very short data raises ValueError."""
        est = MSEEstimator()
        with pytest.raises(ValueError, match="too short"):
            est.estimate(np.random.randn(20))

    def test_results_structure(self):
        """Test that estimate() returns all expected keys."""
        est = MSEEstimator(max_scale=5)
        np.random.seed(42)
        data = np.cumsum(np.random.randn(500))

        results = est.estimate(data)

        assert isinstance(results, dict)
        expected_keys = {
            "complexity_index", "hurst_parameter", "entropy_values",
            "scales", "slope", "r_squared", "method", "m", "r",
            "max_scale_used", "n_valid_scales",
        }
        assert expected_keys.issubset(results.keys())
        assert results["method"] == "RCMSE"

    def test_results_types(self):
        """Test that result values have the correct types."""
        est = MSEEstimator(max_scale=5)
        np.random.seed(42)
        data = np.cumsum(np.random.randn(500))
        results = est.estimate(data)

        assert isinstance(results["complexity_index"], float)
        assert isinstance(results["hurst_parameter"], float)
        assert isinstance(results["entropy_values"], list)
        assert isinstance(results["scales"], list)
        assert len(results["entropy_values"]) == len(results["scales"])

    def test_hurst_range(self):
        """Test that hurst_parameter is in [0, 1]."""
        est = MSEEstimator(max_scale=5)
        np.random.seed(42)
        data = np.cumsum(np.random.randn(500))
        results = est.estimate(data)

        assert 0.0 <= results["hurst_parameter"] <= 1.0

    def test_reproducibility(self):
        """Test that estimation is reproducible with the same data."""
        est1 = MSEEstimator(max_scale=5)
        est2 = MSEEstimator(max_scale=5)

        np.random.seed(42)
        data = np.cumsum(np.random.randn(500))

        r1 = est1.estimate(data.copy())
        r2 = est2.estimate(data.copy())

        assert np.isclose(r1["complexity_index"], r2["complexity_index"])
        assert np.isclose(r1["hurst_parameter"], r2["hurst_parameter"])

    def test_get_results(self):
        """Test that get_results() works after estimate()."""
        est = MSEEstimator(max_scale=5)
        np.random.seed(42)
        data = np.cumsum(np.random.randn(500))

        est.estimate(data)
        results = est.get_results()
        assert "complexity_index" in results

    def test_get_results_before_estimate(self):
        """Test that get_results() raises before estimate()."""
        est = MSEEstimator()
        with pytest.raises(ValueError, match="No results available"):
            est.get_results()

    def test_repr(self):
        """Test string representation."""
        est = MSEEstimator(m=3, r=0.2, max_scale=15)
        repr_str = repr(est)
        assert "MSEEstimator" in repr_str

    def test_white_noise_vs_correlated(self):
        """White noise should produce lower CI than correlated 1/f-like signal.

        This is the fundamental discriminating property of MSE for LRD.
        """
        np.random.seed(42)
        white_noise = np.random.randn(1000)
        pink_noise = np.cumsum(np.random.randn(1000))  # random walk ≈ 1/f

        est_wn = MSEEstimator(max_scale=10)
        est_pn = MSEEstimator(max_scale=10)

        r_wn = est_wn.estimate(white_noise)
        r_pn = est_pn.estimate(pink_noise)

        # 1/f noise should sustain entropy across scales → higher CI
        # White noise entropy decays → lower CI
        # The Hurst estimate for the walk should be higher than for white noise
        assert r_pn["hurst_parameter"] >= r_wn["hurst_parameter"]


# ======================================================================
# MultivariateMSEEstimator Tests
# ======================================================================
class TestMultivariateMSEEstimator:
    """Test suite for the multivariate MSE estimator."""

    def test_valid_parameters(self):
        """Test valid parameter initialisation."""
        est = MultivariateMSEEstimator(m=[2, 2], r=0.15, max_scale=10)
        assert est.parameters["m"] == [2, 2]
        assert est.parameters["r"] == 0.15

    def test_default_m(self):
        """Test that default m is [2, 2]."""
        est = MultivariateMSEEstimator()
        assert est.parameters["m"] == [2, 2]

    def test_invalid_m_short(self):
        """Test that m with < 2 elements raises ValueError."""
        with pytest.raises(ValueError, match="list of at least 2"):
            MultivariateMSEEstimator(m=[2])

    def test_invalid_m_values(self):
        """Test that m with negative values raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            MultivariateMSEEstimator(m=[0, 2])

    def test_invalid_r(self):
        """Test that r <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="r must be positive"):
            MultivariateMSEEstimator(r=0)

    def test_1d_input_raises(self):
        """Test that 1-D input raises ValueError."""
        est = MultivariateMSEEstimator()
        with pytest.raises(ValueError, match="2-D input"):
            est.estimate(np.random.randn(500))

    def test_single_channel_raises(self):
        """Test that single-channel 2-D input raises ValueError."""
        est = MultivariateMSEEstimator()
        with pytest.raises(ValueError, match="at least 2 channels"):
            est.estimate(np.random.randn(500, 1))

    def test_m_channel_mismatch_raises(self):
        """Test that mismatched m length and channel count raises."""
        est = MultivariateMSEEstimator(m=[2, 2])
        with pytest.raises(ValueError, match="must match the number of channels"):
            est.estimate(np.random.randn(500, 3))

    def test_short_data_raises(self):
        """Test that very short data raises ValueError."""
        est = MultivariateMSEEstimator()
        with pytest.raises(ValueError, match="too short"):
            est.estimate(np.random.randn(20, 2))

    def test_results_structure(self):
        """Test that estimate() returns all expected keys."""
        est = MultivariateMSEEstimator(m=[2, 2], max_scale=5)
        np.random.seed(42)
        data = np.column_stack([
            np.cumsum(np.random.randn(500)),
            np.cumsum(np.random.randn(500)),
        ])

        results = est.estimate(data)

        expected_keys = {
            "complexity_index", "hurst_parameter", "entropy_values",
            "scales", "slope", "r_squared", "n_channels", "method",
            "m", "r", "max_scale_used", "n_valid_scales",
        }
        assert expected_keys.issubset(results.keys())
        assert results["method"] == "MvMSEn"
        assert results["n_channels"] == 2

    def test_results_types(self):
        """Test that result values have correct types."""
        est = MultivariateMSEEstimator(m=[2, 2], max_scale=5)
        np.random.seed(42)
        data = np.column_stack([
            np.cumsum(np.random.randn(500)),
            np.cumsum(np.random.randn(500)),
        ])
        results = est.estimate(data)

        assert isinstance(results["complexity_index"], float)
        assert isinstance(results["hurst_parameter"], float)
        assert isinstance(results["n_channels"], int)

    def test_hurst_range(self):
        """Test that hurst_parameter is in [0, 1]."""
        est = MultivariateMSEEstimator(m=[2, 2], max_scale=5)
        np.random.seed(42)
        data = np.column_stack([
            np.cumsum(np.random.randn(500)),
            np.cumsum(np.random.randn(500)),
        ])
        results = est.estimate(data)
        assert 0.0 <= results["hurst_parameter"] <= 1.0

    def test_three_channels(self):
        """Test with 3 channels and matching m."""
        est = MultivariateMSEEstimator(m=[2, 2, 2], max_scale=5)
        np.random.seed(42)
        data = np.column_stack([
            np.cumsum(np.random.randn(500)),
            np.cumsum(np.random.randn(500)),
            np.cumsum(np.random.randn(500)),
        ])
        results = est.estimate(data)
        assert results["n_channels"] == 3

    def test_repr(self):
        """Test string representation."""
        est = MultivariateMSEEstimator(m=[2, 2], r=0.2)
        repr_str = repr(est)
        assert "MultivariateMSEEstimator" in repr_str
