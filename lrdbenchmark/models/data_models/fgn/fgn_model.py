import numpy as np
from scipy.fft import fft, ifft
from typing import Optional, Union, Generator

class FractionalGaussianNoise:
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
        
    def generate(
        self, 
        length: int, 
        rng: Optional[np.random.Generator] = None,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate fGn time series.
        
        Parameters
        ----------
        length : int
            Length of the time series to generate
        rng : np.random.Generator, optional
            Random number generator instance
        random_state : int, optional
            Seed for random number generator if rng is not provided
            
        Returns
        -------
        np.ndarray
            Generated fGn time series
        """
        if rng is None:
            rng = np.random.default_rng(random_state)
            
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
