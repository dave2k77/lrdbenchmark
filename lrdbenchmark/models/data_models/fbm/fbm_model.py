import numpy as np
from typing import Any, Dict, Optional
from ..base_model import BaseModel
from ..fgn.fgn_model import FractionalGaussianNoise


class FractionalBrownianMotion(BaseModel):
    """
    Fractional Brownian Motion (fBm) generator.
    
    Generates fBm by cumulatively summing fGn increments.
    """
    def __init__(self, H: float = 0.7, sigma: float = 1.0, method: str = "davies_harte", **kwargs):
        self.H = H
        self.sigma = sigma
        self.method = method
        self.fgn = FractionalGaussianNoise(H=H, sigma=sigma)
        super().__init__(H=H, sigma=sigma, method=method, **kwargs)

    def _validate_parameters(self) -> None:
        """Validate model parameters."""
        if not (0 < self.H < 1):
            raise ValueError(f"Hurst parameter H must be in (0, 1), got {self.H}")
        if self.sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {self.sigma}")
        valid_methods = ["davies_harte", "circulant", "cholesky"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got {self.method}")

    def generate(
        self, 
        length: Optional[int] = None,
        seed: Optional[int] = None,
        n: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate fBm time series.
        
        Parameters
        ----------
        length : int, optional
            Length of the time series to generate (preferred parameter name)
        seed : int, optional
            Random seed for reproducibility
        n : int, optional
            Alternate parameter name for length (for backward compatibility)
        rng : np.random.Generator, optional
            Random number generator instance
        random_state : int, optional
            Seed for random number generator if rng is not provided (deprecated, use seed)
            
        Returns
        -------
        np.ndarray
            Generated fBm time series
        """
        # Handle length parameter (support both 'length' and 'n')
        if length is None and n is not None:
            length = n
        if length is None:
            raise ValueError("Either 'length' or 'n' must be provided")
        
        # Handle seed parameter (support both 'seed' and 'random_state')
        effective_seed = seed if seed is not None else random_state
        
        # Generate fGn increments
        noise = self.fgn.generate(length, seed=effective_seed, rng=rng)
        
        # Cumulate to get fBm
        fbm = np.cumsum(noise)
        
        return fbm

    def get_theoretical_properties(self) -> Dict[str, Any]:
        """
        Get theoretical properties of the fBm model.
        
        Returns
        -------
        dict
            Dictionary containing theoretical properties
        """
        return {
            "hurst_parameter": self.H,
            "variance": self.sigma ** 2,
            "stationary": False,  # fBm is not stationary
            "long_range_dependent": self.H > 0.5,
            "long_range_dependence": self.H > 0.5,  # alternate key for compatibility
            "self_similar": True,
            "self_similarity_exponent": self.H,
            "stationary_increments": True,
            "gaussian": True,
            "mean": 0.0,
        }

    def get_increments(self, data: np.ndarray) -> np.ndarray:
        """
        Compute the increments of the fBm process.
        
        Parameters
        ----------
        data : np.ndarray
            fBm time series data
            
        Returns
        -------
        np.ndarray
            Increments (differences) of the input data
        """
        return np.diff(data)

    def __str__(self) -> str:
        """String representation of the model."""
        return f"FractionalBrownianMotion(H={self.H}, sigma={self.sigma})"

    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"FractionalBrownianMotion(H={self.H}, sigma={self.sigma}, method={self.method})"
