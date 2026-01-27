#!/usr/bin/env python3
"""
Critical Regime Generators for LRDBenchmark.

Physics-motivated generators for testing LRD estimators in critical,
nonequilibrium, and heavy-tailed regimes where classical assumptions fail.

Classes:
    - OrnsteinUhlenbeckProcess: OU with time-varying friction (transient criticality)
    - SubordinatedProcess: Subordinated Brownian motion (nonequilibrium)
    - FractionalLevyMotion: Heavy-tailed, non-Gaussian LRD (α<2 stable)
    - SOCAvalancheModel: Bak-Tang-Wiesenfeld self-organized criticality
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.special import gamma as gamma_func
from enum import Enum


class OrnsteinUhlenbeckProcess:
    """
    Ornstein-Uhlenbeck process with time-varying friction coefficient.
    
    Models transient criticality where the system transitions between
    different relaxation regimes.
    
    The SDE is:
        dX_t = -θ(t) * X_t * dt + σ * dW_t
    
    where θ(t) is the time-varying friction coefficient.
    
    Example
    -------
    >>> gen = OrnsteinUhlenbeckProcess(theta_start=0.1, theta_end=1.0)
    >>> result = gen.generate(1000)
    """
    
    def __init__(
        self,
        theta_start: float = 0.1,
        theta_end: float = 1.0,
        sigma: float = 1.0,
        transition_type: str = 'linear',
        dt: float = 0.01,
        random_state: Optional[int] = None
    ):
        """
        Initialize OU process with time-varying friction.
        
        Parameters
        ----------
        theta_start : float
            Initial friction coefficient (low = critical-like)
        theta_end : float
            Final friction coefficient
        sigma : float
            Noise intensity
        transition_type : str
            How θ transitions: 'linear', 'exponential', 'step'
        dt : float
            Time step for simulation
        random_state : int, optional
            Random seed
        """
        self.theta_start = theta_start
        self.theta_end = theta_end
        self.sigma = sigma
        self.transition_type = transition_type
        self.dt = dt
        self.rng = np.random.default_rng(random_state)
    
    def _get_theta_trajectory(self, length: int) -> np.ndarray:
        """Get time-varying friction coefficient."""
        t = np.linspace(0, 1, length)
        
        if self.transition_type == 'linear':
            return self.theta_start + (self.theta_end - self.theta_start) * t
        elif self.transition_type == 'exponential':
            rate = 3.0
            return self.theta_start + (self.theta_end - self.theta_start) * (1 - np.exp(-rate * t))
        elif self.transition_type == 'step':
            theta = np.full(length, self.theta_start)
            theta[length//2:] = self.theta_end
            return theta
        else:
            raise ValueError(f"Unknown transition type: {self.transition_type}")
    
    def generate(
        self,
        length: int,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate OU process with time-varying friction.
        
        Returns dict with 'signal', 'theta_trajectory', 'metadata'.
        """
        local_rng = np.random.default_rng(seed) if seed is not None else self.rng
        
        theta_traj = self._get_theta_trajectory(length)
        
        # Euler-Maruyama simulation
        x = np.zeros(length)
        x[0] = local_rng.normal(0, self.sigma / np.sqrt(2 * self.theta_start))
        
        sqrt_dt = np.sqrt(self.dt)
        
        for i in range(1, length):
            theta = theta_traj[i-1]
            drift = -theta * x[i-1] * self.dt
            diffusion = self.sigma * sqrt_dt * local_rng.normal()
            x[i] = x[i-1] + drift + diffusion
        
        return {
            'signal': x,
            'theta_trajectory': theta_traj,
            'metadata': {
                'process_type': 'OrnsteinUhlenbeck',
                'theta_start': self.theta_start,
                'theta_end': self.theta_end,
                'sigma': self.sigma,
                'transition_type': self.transition_type,
                'stationary': False
            }
        }


class SubordinatedProcess:
    """
    Subordinated Brownian motion for modeling nonequilibrium phenomena.
    
    The process X(S(t)) where X is Brownian motion and S(t) is an
    inverse stable subordinator, producing subdiffusive behavior.
    
    This models systems with trapping events and anomalous diffusion
    where classical ergodicity breaks down.
    
    Example
    -------
    >>> gen = SubordinatedProcess(alpha=0.7)
    >>> result = gen.generate(1000)
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        sigma: float = 1.0,
        random_state: Optional[int] = None
    ):
        """
        Initialize subordinated process.
        
        Parameters
        ----------
        alpha : float
            Subordinator index (0 < alpha < 1). Lower = more trapping.
        sigma : float
            Diffusion coefficient of parent Brownian motion
        random_state : int, optional
            Random seed
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        
        self.alpha = alpha
        self.sigma = sigma
        self.rng = np.random.default_rng(random_state)
    
    def _generate_stable_subordinator(
        self,
        length: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Generate one-sided stable Lévy process (subordinator)."""
        # Use Chambers-Mallows-Stuck algorithm for positive stable
        u = rng.uniform(0, np.pi, length)
        e = rng.exponential(1.0, length)
        
        # Positive stable with index alpha
        s = (np.sin(self.alpha * u) / np.cos(u)**(1/self.alpha)) * \
            (np.cos(u * (1 - self.alpha)) / e)**((1 - self.alpha)/self.alpha)
        
        return np.cumsum(s)
    
    def generate(
        self,
        length: int,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate subordinated Brownian motion.
        
        Returns dict with 'signal', 'operational_time', 'metadata'.
        """
        local_rng = np.random.default_rng(seed) if seed is not None else self.rng
        
        # Generate subordinator (operational time)
        subordinator = self._generate_stable_subordinator(length * 2, local_rng)
        
        # Generate parent Brownian motion (on operational time scale)
        bm = np.cumsum(local_rng.normal(0, self.sigma, length * 2))
        
        # Inverse subordinator (first passage times)
        physical_times = np.linspace(0, subordinator[-1], length)
        
        # Interpolate BM at inverse subordinator times
        indices = np.searchsorted(subordinator, physical_times)
        indices = np.clip(indices, 0, len(bm) - 1)
        signal = bm[indices]
        
        return {
            'signal': signal,
            'operational_time': subordinator[:length],
            'metadata': {
                'process_type': 'SubordinatedBrownian',
                'alpha': self.alpha,
                'sigma': self.sigma,
                'subdiffusive': True,
                'ergodic': False
            }
        }


class FractionalLevyMotion:
    """
    Linear Fractional Stable Motion (LFSM) via FFT-based spectral method.
    
    Generates heavy-tailed, non-Gaussian processes with long-range dependence
    by applying fractional integration to symmetric α-stable noise in the
    frequency domain.
    
    The algorithm:
        1. Generate symmetric α-stable noise Z
        2. FFT to frequency domain: Z̃ = FFT(Z)
        3. Apply spectral kernel: X̃ = Z̃ * |ω|^{-d} where d = H - 1/α
        4. IFFT back to time domain: X = IFFT(X̃)
    
    When α = 2 (Gaussian case), d = H - 0.5, recovering fractional Brownian motion.
    
    Parameters
    ----------
    H : float
        Hurst (self-similarity) parameter, 0 < H < 1
    alpha : float
        Stability index, 0 < alpha <= 2. α=2 is Gaussian (fBm).
    beta : float
        Skewness parameter, -1 <= beta <= 1. Use 0 for symmetric.
    scale : float
        Scale parameter for the stable distribution
    use_hpfracc : bool
        If True, attempt to use hpfracc library for optimized operations.
        Falls back to NumPy if hpfracc is not available.
    random_state : int, optional
        Random seed for reproducibility
    
    Example
    -------
    >>> gen = FractionalLevyMotion(H=0.7, alpha=1.5)
    >>> result = gen.generate(1000)
    >>> signal = result['signal']
    
    Notes
    -----
    The relationship between H (Hurst parameter), α (stability index), and
    d (fractional integration order) is: d = H - 1/α
    
    For LFSM, the valid parameter range requires: 0 < H < 1 and 1/α < H < 1
    to ensure d > 0 (fractional integration, not differentiation).
    
    References
    ----------
    Samorodnitsky, G. & Taqqu, M. S. (1994). Stable Non-Gaussian Random
    Processes: Stochastic Models with Infinite Variance. Chapman & Hall.
    """
    
    # Try to import hpfracc at class level
    _hpfracc_available = None
    _hpfracc_module = None
    
    @classmethod
    def _check_hpfracc(cls) -> bool:
        """Check if hpfracc is available (cached)."""
        if cls._hpfracc_available is None:
            try:
                import hpfracc
                cls._hpfracc_module = hpfracc
                cls._hpfracc_available = True
            except ImportError:
                cls._hpfracc_available = False
        return cls._hpfracc_available
    
    def __init__(
        self,
        H: float = 0.7,
        alpha: float = 1.5,
        beta: float = 0.0,
        scale: float = 1.0,
        use_hpfracc: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize Linear Fractional Stable Motion generator.
        
        Parameters
        ----------
        H : float
            Hurst-like self-similarity parameter (0 < H < 1)
        alpha : float
            Stability index (0 < alpha <= 2). α=2 is Gaussian.
        beta : float
            Skewness parameter (-1 <= beta <= 1)
        scale : float
            Scale parameter
        use_hpfracc : bool
            Whether to use hpfracc library if available
        random_state : int, optional
            Random seed
        """
        if not 0 < H < 1:
            raise ValueError("H must be in (0, 1)")
        if not 0 < alpha <= 2:
            raise ValueError("alpha must be in (0, 2]")
        if not -1 <= beta <= 1:
            raise ValueError("beta must be in [-1, 1]")
        
        self.H = H
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.use_hpfracc = use_hpfracc and self._check_hpfracc()
        self.rng = np.random.default_rng(random_state)
        
        # Compute fractional integration order: d = H - 1/alpha
        # This is the key relationship for LFSM
        self.d = H - 1.0 / alpha
    
    def _generate_stable_rv(
        self,
        size: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate stable random variables using Chambers-Mallows-Stuck algorithm.
        
        For symmetric α-stable (beta=0), this produces the Lévy driver noise.
        """
        u = rng.uniform(-np.pi/2, np.pi/2, size)
        e = rng.exponential(1.0, size)
        
        if self.alpha == 2:
            # Gaussian case
            return rng.normal(0, self.scale * np.sqrt(2), size)
        
        if self.alpha == 1:
            # Cauchy case
            return self.scale * np.tan(u)
        
        # General stable case (Chambers-Mallows-Stuck)
        b = np.arctan(self.beta * np.tan(np.pi * self.alpha / 2)) / self.alpha
        s = (1 + self.beta**2 * np.tan(np.pi * self.alpha / 2)**2)**(1/(2*self.alpha))
        
        x = s * (np.sin(self.alpha * (u + b)) / np.cos(u)**(1/self.alpha)) * \
            (np.cos(u - self.alpha * (u + b)) / e)**((1 - self.alpha)/self.alpha)
        
        return self.scale * x
    
    def _apply_spectral_kernel(
        self,
        noise: np.ndarray
    ) -> np.ndarray:
        """
        Apply fractional integration kernel |ω|^{-d} in frequency domain.
        
        This is the core of the spectral method for LFSM generation.
        """
        n = len(noise)
        
        # FFT of the stable noise
        noise_fft = np.fft.fft(noise)
        
        # Construct frequency array
        freq = np.fft.fftfreq(n)
        
        # Build spectral kernel: |ω|^{-d}
        # Handle DC component (ω=0) to avoid division by zero
        omega = 2 * np.pi * freq
        with np.errstate(divide='ignore', invalid='ignore'):
            kernel = np.abs(omega) ** (-self.d)
        
        # Set DC component to 0 (removes mean, standard for LFSM)
        kernel[0] = 0.0
        
        # Handle any remaining infinities at very low frequencies
        kernel = np.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply kernel in frequency domain
        result_fft = noise_fft * kernel
        
        # Inverse FFT to get time-domain signal
        result = np.fft.ifft(result_fft)
        
        # Take real part (imaginary should be negligible for real input)
        return np.real(result)
    
    def _apply_spectral_kernel_hpfracc(
        self,
        noise: np.ndarray
    ) -> np.ndarray:
        """
        Apply fractional integration using hpfracc's optimized methods.
        
        Uses Riemann-Liouville fractional integral when available.
        """
        try:
            hpfracc = self._hpfracc_module
            
            # Use hpfracc's fractional integral if available
            # The RL integral of order d corresponds to |ω|^{-d} in frequency domain
            if hasattr(hpfracc, 'riemann_liouville_integral'):
                t = np.linspace(0, 1, len(noise))
                result = hpfracc.riemann_liouville_integral(t, noise, self.d)
                return result
            elif hasattr(hpfracc, 'optimized_riemann_liouville_integral'):
                t = np.linspace(0, 1, len(noise))
                from hpfracc import FractionalOrder
                order = FractionalOrder(self.d)
                result = hpfracc.optimized_riemann_liouville_integral(t, noise, order)
                return result
            else:
                # Fall back to numpy implementation
                return self._apply_spectral_kernel(noise)
        except Exception:
            # On any error, fall back to numpy
            return self._apply_spectral_kernel(noise)
    
    def generate(
        self,
        length: int,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate Linear Fractional Stable Motion.
        
        Uses FFT-based spectral method with kernel |ω|^{-d} where d = H - 1/α.
        
        Parameters
        ----------
        length : int
            Length of the time series to generate
        seed : int, optional
            Random seed for this generation (overrides constructor seed)
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'signal': The generated LFSM time series
            - 'metadata': Process parameters and properties
        """
        local_rng = np.random.default_rng(seed) if seed is not None else self.rng
        
        # Step 1: Generate symmetric α-stable innovations (Lévy driver)
        innovations = self._generate_stable_rv(length, local_rng)
        
        # Steps 2-4: Apply spectral fractional integration
        if self.use_hpfracc and self._hpfracc_available:
            signal = self._apply_spectral_kernel_hpfracc(innovations)
        else:
            signal = self._apply_spectral_kernel(innovations)
        
        return {
            'signal': signal,
            'metadata': {
                'process_type': 'LinearFractionalStableMotion',
                'H': self.H,
                'alpha': self.alpha,
                'beta': self.beta,
                'scale': self.scale,
                'd': self.d,  # Fractional integration order
                'heavy_tailed': self.alpha < 2,
                'infinite_variance': self.alpha < 2,
                'method': 'spectral_fft',
                'used_hpfracc': self.use_hpfracc and self._hpfracc_available
            }
        }


class SOCAvalancheModel:
    """
    Self-Organized Criticality avalanche model (Bak-Tang-Wiesenfeld).
    
    Simulates a sandpile model producing scale-free avalanche dynamics.
    The resulting time series of avalanche sizes exhibits power-law
    correlations characteristic of critical systems.
    
    Example
    -------
    >>> gen = SOCAvalancheModel(grid_size=64)
    >>> result = gen.generate(1000)
    """
    
    def __init__(
        self,
        grid_size: int = 32,
        threshold: int = 4,
        random_state: Optional[int] = None
    ):
        """
        Initialize SOC sandpile model.
        
        Parameters
        ----------
        grid_size : int
            Size of square lattice
        threshold : int
            Toppling threshold (typically 4 for 2D)
        random_state : int, optional
            Random seed
        """
        self.grid_size = grid_size
        self.threshold = threshold
        self.rng = np.random.default_rng(random_state)
    
    def _run_sandpile(
        self,
        n_avalanches: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Run sandpile simulation and record avalanche sizes."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        avalanche_sizes = []
        
        for _ in range(n_avalanches):
            # Add grain at random site
            x = rng.integers(0, self.grid_size)
            y = rng.integers(0, self.grid_size)
            grid[x, y] += 1
            
            # Topple until stable
            avalanche_size = 0
            while np.any(grid >= self.threshold):
                unstable = grid >= self.threshold
                avalanche_size += np.sum(unstable)
                
                # Topple unstable sites
                topple_count = grid[unstable] // self.threshold
                grid[unstable] -= topple_count * self.threshold
                
                # Distribute to neighbors
                unstable_coords = np.argwhere(unstable)
                for cx, cy in unstable_coords:
                    if cx > 0:
                        grid[cx-1, cy] += 1
                    if cx < self.grid_size - 1:
                        grid[cx+1, cy] += 1
                    if cy > 0:
                        grid[cx, cy-1] += 1
                    if cy < self.grid_size - 1:
                        grid[cx, cy+1] += 1
            
            avalanche_sizes.append(avalanche_size)
        
        return np.array(avalanche_sizes)
    
    def generate(
        self,
        length: int,
        seed: Optional[int] = None,
        warmup: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate time series of avalanche sizes from SOC sandpile.
        
        Parameters
        ----------
        length : int
            Number of avalanche events to generate
        seed : int, optional
            Random seed
        warmup : int
            Number of initial events to discard (reach critical state)
        
        Returns dict with 'signal', 'metadata'.
        """
        local_rng = np.random.default_rng(seed) if seed is not None else self.rng
        
        # Run with warmup
        total_events = warmup + length
        all_sizes = self._run_sandpile(total_events, local_rng)
        
        # Discard warmup
        signal = all_sizes[warmup:].astype(np.float64)
        
        # Normalize for analysis
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        return {
            'signal': signal,
            'raw_avalanche_sizes': all_sizes[warmup:],
            'metadata': {
                'process_type': 'SOCAvalanche',
                'grid_size': self.grid_size,
                'threshold': self.threshold,
                'warmup': warmup,
                'critical': True,
                'power_law_distributed': True
            }
        }


# Convenience factory function
def create_critical_regime_process(
    process_type: str,
    **kwargs
) -> Any:
    """
    Factory function for critical regime processes.
    
    Parameters
    ----------
    process_type : str
        'ornstein_uhlenbeck', 'subordinated', 'fractional_levy', 'soc_avalanche'
    **kwargs
        Process-specific parameters
    """
    process_map = {
        'ornstein_uhlenbeck': OrnsteinUhlenbeckProcess,
        'ou': OrnsteinUhlenbeckProcess,
        'subordinated': SubordinatedProcess,
        'subordinated_brownian': SubordinatedProcess,
        'fractional_levy': FractionalLevyMotion,
        'levy': FractionalLevyMotion,
        'lfsm': FractionalLevyMotion,
        'linear_fractional_stable_motion': FractionalLevyMotion,
        'soc_avalanche': SOCAvalancheModel,
        'soc': SOCAvalancheModel,
        'sandpile': SOCAvalancheModel
    }
    
    process_type = process_type.lower().replace(' ', '_').replace('-', '_')
    
    if process_type not in process_map:
        raise ValueError(
            f"Unknown process type '{process_type}'. "
            f"Available: ornstein_uhlenbeck, subordinated, fractional_levy, lfsm, soc_avalanche"
        )
    
    return process_map[process_type](**kwargs)


# Alias for explicit naming
LinearFractionalStableMotion = FractionalLevyMotion
