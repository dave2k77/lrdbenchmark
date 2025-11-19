#!/usr/bin/env python3
"""
Unified Whittle Estimator for Spectral Analysis.

This module implements the Whittle estimator using scipy.optimize.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal, optimize
from typing import Dict, Any, Optional, Union, Tuple
import warnings

class WhittleEstimator:
    """
    Whittle estimator for Hurst parameter estimation.
    
    Minimizes the Whittle likelihood function based on the fGn spectral density.
    """

    def __init__(
        self,
        min_freq_ratio: float = 0.01,
        max_freq_ratio: float = 0.5,
        **kwargs
    ):
        self.min_freq_ratio = min_freq_ratio
        self.max_freq_ratio = max_freq_ratio
        self.results = {}

    def estimate(self, data: Union[np.ndarray, list]) -> Dict[str, Any]:
        """
        Estimate Hurst parameter using Whittle method.

        Parameters
        ----------
        data : array-like
            Time series data.

        Returns
        -------
        dict
            Dictionary containing estimation results
        """
        data = np.asarray(data)
        n = len(data)
        
        # 1. Compute Periodogram
        # We use the standard periodogram (no windowing usually for Whittle, or Hann)
        # Whittle approximation assumes raw periodogram.
        freqs, psd = signal.periodogram(data, window='boxcar', detrend='constant', scaling='density')
        
        # 2. Select frequencies
        # Discard DC (freq=0) and Nyquist if needed, and apply ratio limits
        # Frequencies are in [0, 0.5] (cycles/sample)
        mask = (freqs > self.min_freq_ratio) & (freqs <= self.max_freq_ratio)
        freqs_sel = freqs[mask]
        psd_sel = psd[mask]
        
        if len(freqs_sel) < 10:
            warnings.warn("Insufficient frequency points for Whittle estimation.")
            return {"hurst_parameter": 0.5} # Fallback
            
        # 3. Define Negative Log-Likelihood
        # Whittle likelihood: sum( log(f) + I/f )
        # f(lambda) = C * f_shape(lambda, H)
        # We can profile out C.
        # Q(H) = sum( I / f_shape ) / m
        # log L = m * log(Q) + sum( log(f_shape) )
        
        def fgn_spectrum_shape(freqs, H):
            # Approximate fGn spectrum sum
            # f(lambda) ~ |lambda|^(1-2H) is the local approximation
            # Exact: C * (1-cos(lam)) * sum |2pi k + lam|^(-2H-1)
            # Let's use the expansion with k = -2 to 2
            lam = 2 * np.pi * freqs
            s = np.zeros_like(lam)
            for k in range(-2, 3):
                term = np.abs(2 * np.pi * k + lam)
                s += term ** (-2 * H - 1)
            return (1 - np.cos(lam)) * s

        def neg_log_likelihood(H):
            if H <= 0.01 or H >= 0.99:
                return np.inf
            
            f_shape = fgn_spectrum_shape(freqs_sel, H)
            # Profile out Scale C
            # C_hat = mean( I / f_shape )
            ratio = psd_sel / f_shape
            C_hat = np.mean(ratio)
            
            # Full likelihood (ignoring constants)
            # L = sum( log(C * f_shape) + I / (C * f_shape) )
            #   = m*log(C) + sum(log(f_shape)) + (1/C) * sum(I/f_shape)
            #   = m*log(C) + sum(log(f_shape)) + m
            
            nll = len(freqs_sel) * np.log(C_hat) + np.sum(np.log(f_shape))
            return nll

        # 4. Optimize
        res = optimize.minimize_scalar(neg_log_likelihood, bounds=(0.01, 0.99), method='bounded')
        
        hurst = res.x
        
        # Calculate scale
        f_shape = fgn_spectrum_shape(freqs_sel, hurst)
        scale = np.mean(psd_sel / f_shape)
        
        self.results = {
            "hurst_parameter": float(hurst),
            "d_parameter": float(hurst - 0.5),
            "scale_parameter": float(scale),
            "method": "Whittle_Exact_Scipy",
            "optimization_success": res.success,
            "frequencies": freqs_sel.tolist(),
            "periodogram": psd_sel.tolist(),
            "model_spectrum": (scale * f_shape).tolist()
        }
        
        return self.results
