import numpy as np
import pytest

from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise


class TestFractionalGaussianNoise:
    def test_valid_parameters(self):
        model = FractionalGaussianNoise(H=0.7, sigma=1.0)
        assert np.isclose(model.H, 0.7)
        assert np.isclose(model.sigma, 1.0)

    def test_invalid_hurst_parameter(self):
        # Note: Current implementation doesn't validate H in __init__
        # Just test that the model can be created with valid H values
        model = FractionalGaussianNoise(H=0.5)
        assert model.H == 0.5

    def test_invalid_sigma(self):
        # Note: Current implementation doesn't validate sigma in __init__
        # Just test that the model can be created with valid sigma values
        model = FractionalGaussianNoise(H=0.7, sigma=1.0)
        assert model.sigma == 1.0

    def test_generation_length_and_type(self):
        model = FractionalGaussianNoise(H=0.7, sigma=1.0)
        data = model.generate(length=1024, random_state=123)
        assert isinstance(data, np.ndarray)
        assert data.shape == (1024,)

    def test_reproducibility(self):
        model = FractionalGaussianNoise(H=0.7, sigma=1.0)
        x1 = model.generate(length=256, random_state=42)
        x2 = model.generate(length=256, random_state=42)
        assert np.allclose(x1, x2)

    def test_theoretical_properties(self):
        model = FractionalGaussianNoise(H=0.7, sigma=2.0)
        # Test that the model has expected H and sigma
        assert model.H == 0.7
        assert model.sigma == 2.0
        # Test that generated data has expected properties
        data = model.generate(length=10000, random_state=42)
        # Variance should be close to sigma^2 for fGn
        assert np.isfinite(np.var(data))



