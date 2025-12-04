import numpy as np
from scipy.fft import fft, ifft
from typing import Any, Dict, Optional, Union, Generator

from ..base_model import BaseModel


class FractionalGaussianNoise(BaseModel):
    """
    Fractional Gaussian Noise (fGn) generator using the Davies-Harte method.
    
    Generates exact fGn with Hurst parameter H.
    """
    def __init__(
        self, 
        H: float = 0.7, 
        sigma: float = 1.0, 
        **kwargs
    ):
        self.H = H
        self.sigma = sigma
        super().__init__(H=H, sigma=sigma, **kwargs)

    def _validate_parameters(self) -> None:
        """Validate model parameters."""
        if not (0 < self.H < 1):
            raise ValueError(f"Hurst parameter H must be in (0, 1), got {self.H}")
        if self.sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {self.sigma}")

    def generate(
        self, 
        length: Optional[int] = None, 
        seed: Optional[int] = None,
        n: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate fGn time series.
        
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
            Generated fGn time series
        """
        # Handle length parameter (support both 'length' and 'n')
        if length is None and n is not None:
            length = n
        if length is None:
            raise ValueError("Either 'length' or 'n' must be provided")
        
        # Handle seed parameter (support both 'seed' and 'random_state')
        effective_seed = seed if seed is not None else random_state
        
        if rng is None:
            rng = np.random.default_rng(effective_seed)
            
        # Davies-Harte method
        N = length
        
        # 1. Calculate autocovariance function
        k = np.arange(N)
        # Autocovariance of fGn: gamma(k) = (sigma^2 / 2) * (|k+1|^2H - 2|k|^2H + |k-1|^2H)
        gamma = (self.sigma**2 / 2.0) * (np.abs(k + 1)**(2 * self.H) - 2 * np.abs(k)**(2 * self.H) + np.abs(k - 1)**(2 * self.H))
        
        # 2. Construct the first row of the circulant matrix C
        # We use M = 2N
        k_extended = np.arange(N + 1)
        gamma_ext = (self.sigma**2 / 2.0) * (np.abs(k_extended + 1)**(2 * self.H) - 2 * np.abs(k_extended)**(2 * self.H) + np.abs(k_extended - 1)**(2 * self.H))
        
        # Circulant first row: g0, g1, ..., g(N-1), g(N), g(N-1), ..., g1
        first_row = np.concatenate([gamma_ext[:N], [gamma_ext[N]], gamma_ext[1:N][::-1]])
        
        # 3. Compute eigenvalues of C
        eigenvals = fft(first_row).real
        
        # Check for negative eigenvalues (numerical instability or invalid H)
        if np.any(eigenvals < 0):
            if np.min(eigenvals) > -1e-9:
                eigenvals[eigenvals < 0] = 0
            else:
                # Warning: Davies-Harte failure (should not happen for valid fGn)
                eigenvals[eigenvals < 0] = 0
                
        # 4. Generate complex Gaussian noise
        # V = (randn + j * randn)
        V = rng.standard_normal(2 * N) + 1j * rng.standard_normal(2 * N)
        
        # 5. Compute IFFT
        Y = ifft(np.sqrt(eigenvals) * V)
        
        # 6. Take real part and scale
        # The factor sqrt(2N) ensures the correct variance
        fgn = Y[:N].real * np.sqrt(2 * N)
        
        return fgn

    def get_theoretical_properties(self) -> Dict[str, Any]:
        """
        Get theoretical properties of the fGn model.
        
        Returns
        -------
        dict
            Dictionary containing theoretical properties
        """
        return {
            "hurst_parameter": self.H,
            "variance": self.sigma ** 2,
            "stationary": True,
            "long_range_dependent": self.H > 0.5,
            "mean": 0.0,
        }
