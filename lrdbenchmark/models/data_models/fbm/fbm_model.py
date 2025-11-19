import numpy as np
from typing import Optional
from ..fgn.fgn_model import FractionalGaussianNoise

class FractionalBrownianMotion:
    """
    Fractional Brownian Motion (fBm) generator.
    
    Generates fBm by cumulatively summing fGn increments.
    """
    def __init__(self, H: float = 0.7, sigma: float = 1.0, **kwargs):
        self.H = H
        self.sigma = sigma
        self.fgn = FractionalGaussianNoise(H=H, sigma=sigma, **kwargs)

    def generate(
        self, 
        length: int, 
        rng: Optional[np.random.Generator] = None,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate fBm time series.
        
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
            Generated fBm time series
        """
        # Generate fGn increments
        noise = self.fgn.generate(length, rng=rng, random_state=random_state)
        
        # Cumulate to get fBm
        fbm = np.cumsum(noise)
        
        return fbm
