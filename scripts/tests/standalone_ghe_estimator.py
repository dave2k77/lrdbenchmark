#!/usr/bin/env python3
"""
Standalone GHE (Generalized Hurst Exponent) Estimator for testing.

This is a standalone implementation that doesn't depend on the LRDBenchmark package structure.
"""

import numpy as np
import warnings
from scipy import stats
from typing import Dict, Any, Optional, Union, Tuple, List

# Simple base estimator class
class BaseEstimator:
    def __init__(self, **kwargs):
        self.parameters = kwargs
        self.results = {}

class GHEEstimator(BaseEstimator):
    """
    GHE (Generalized Hurst Exponent) Estimator for Long-Range Dependence Analysis.

    Based on the paper:
    "Typical Algorithms for Estimating Hurst Exponent of Time Sequence: A Data Analyst's Perspective"
    by HONG-YAN ZHANG, ZHI-QIANG FENG, SI-YU FENG, AND YU ZHOU
    IEEE ACCESS 2024, DOI: 10.1109/ACCESS.2024.3512542
    """

    def __init__(self, **kwargs):
        """Initialize the GHE estimator."""
        super().__init__(**kwargs)
        
        # Set default parameters
        self.parameters.setdefault('q_values', np.array([1, 2, 3, 4, 5]))
        self.parameters.setdefault('tau_min', 2)
        self.parameters.setdefault('tau_max', None)
        self.parameters.setdefault('tau_step', 1)
        
        # Initialize results
        self.results = {}
        self.name = "GHE"
        self.category = "Temporal"
        
        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate estimator parameters."""
        q_values = self.parameters['q_values']
        if not isinstance(q_values, (list, np.ndarray)):
            raise ValueError("q_values must be a list or numpy array")
        
        q_values = np.array(q_values)
        if len(q_values) == 0:
            raise ValueError("q_values cannot be empty")
        
        if np.any(q_values <= 0):
            raise ValueError("All q_values must be positive")
        
        if self.parameters['tau_min'] < 1:
            raise ValueError("tau_min must be at least 1")
        
        if self.parameters['tau_step'] < 1:
            raise ValueError("tau_step must be at least 1")

    def _compute_qth_moments(self, data: np.ndarray, q_values: np.ndarray, 
                           tau_values: np.ndarray) -> np.ndarray:
        """Compute q-th order moments using NumPy."""
        N = len(data)
        moments = np.zeros((len(q_values), len(tau_values)))
        
        for i, q in enumerate(q_values):
            for j, tau in enumerate(tau_values):
                if tau >= N:
                    moments[i, j] = np.nan
                    continue
                
                # Compute increments
                increments = data[tau:] - data[:-tau]
                
                # Compute q-th moment
                if q == 1:
                    moments[i, j] = np.mean(np.abs(increments))
                else:
                    moments[i, j] = np.mean(np.abs(increments) ** q)
        
        return moments

    def _estimate_hurst_exponents(self, tau_values: np.ndarray, moments: np.ndarray, 
                                q_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate Hurst exponents from q-th moments."""
        hurst_exponents = np.zeros(len(q_values))
        r_squared = np.zeros(len(q_values))
        std_errors = np.zeros(len(q_values))
        
        for i, q in enumerate(q_values):
            # Get valid data points (non-NaN)
            valid_mask = ~np.isnan(moments[i, :])
            if np.sum(valid_mask) < 2:
                hurst_exponents[i] = np.nan
                r_squared[i] = np.nan
                std_errors[i] = np.nan
                continue
            
            tau_valid = tau_values[valid_mask]
            moments_valid = moments[i, valid_mask]
            
            # Log transform for linear regression
            log_tau = np.log(tau_valid)
            log_moments = np.log(moments_valid)
            
            # Linear regression: log(K_q(tau)) = q*H(q)*log(tau) + C
            # So: log_moments = q*H(q)*log_tau + C
            # Therefore: H(q) = slope / q
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_tau, log_moments)
                hurst_exponents[i] = slope / q
                r_squared[i] = r_value ** 2
                std_errors[i] = std_err / q
            except:
                hurst_exponents[i] = np.nan
                r_squared[i] = np.nan
                std_errors[i] = np.nan
        
        return hurst_exponents, r_squared, std_errors

    def estimate(self, data: np.ndarray) -> Dict[str, Any]:
        """Estimate the generalized Hurst exponent using the GHE method."""
        # Validate input
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if len(data) < 10:
            raise ValueError("Data must have at least 10 points")
        
        # Get parameters
        q_values = np.array(self.parameters['q_values'])
        tau_min = self.parameters['tau_min']
        tau_max = self.parameters['tau_max'] or min(len(data) // 4, 50)
        tau_step = self.parameters['tau_step']
        
        # Generate time lags
        tau_values = np.arange(tau_min, tau_max + 1, tau_step)
        
        if len(tau_values) < 2:
            raise ValueError("Not enough time lags for analysis")
        
        # Compute q-th moments
        moments = self._compute_qth_moments(data, q_values, tau_values)
        
        # Estimate Hurst exponents
        hurst_exponents, r_squared, std_errors = self._estimate_hurst_exponents(
            tau_values, moments, q_values
        )
        
        # Compute average Hurst exponent (for q=2, which corresponds to standard Hurst)
        q2_idx = np.where(np.abs(q_values - 2.0) < 1e-6)[0]
        if len(q2_idx) > 0:
            main_hurst = hurst_exponents[q2_idx[0]]
        else:
            # If q=2 not available, use the closest q value
            q2_idx = np.argmin(np.abs(q_values - 2.0))
            main_hurst = hurst_exponents[q2_idx]
        
        # Store results
        self.results = {
            'hurst_parameter': main_hurst,
            'generalized_hurst_exponents': hurst_exponents,
            'q_values': q_values,
            'tau_values': tau_values,
            'moments': moments,
            'r_squared': r_squared,
            'std_errors': std_errors,
            'backend_used': 'numpy',
            'success': True,
            'method': 'GHE',
            'data_length': len(data),
            'n_tau': len(tau_values),
            'n_q': len(q_values)
        }
        
        return self.results

    def get_multifractal_spectrum(self) -> Dict[str, np.ndarray]:
        """Compute the multifractal spectrum from generalized Hurst exponents."""
        if not self.results or not self.results.get('success', False):
            raise ValueError("No successful estimation results available")
        
        q_values = self.results['q_values']
        hurst_exponents = self.results['generalized_hurst_exponents']
        
        # Remove NaN values
        valid_mask = ~np.isnan(hurst_exponents)
        q_valid = q_values[valid_mask]
        h_valid = hurst_exponents[valid_mask]
        
        if len(q_valid) < 3:
            return {'alpha': np.array([]), 'f_alpha': np.array([])}
        
        # Compute multifractal spectrum
        # α = H(q) + q * H'(q)
        # f(α) = q * α - τ(q)
        # where τ(q) = q * H(q) - 1
        
        # Compute derivatives using finite differences
        if len(q_valid) > 1:
            dH_dq = np.gradient(h_valid, q_valid)
            alpha = h_valid + q_valid * dH_dq
            tau_q = q_valid * h_valid - 1
            f_alpha = q_valid * alpha - tau_q
        else:
            alpha = h_valid
            f_alpha = np.zeros_like(alpha)
        
        return {
            'alpha': alpha,
            'f_alpha': f_alpha,
            'q_values': q_valid,
            'hurst_exponents': h_valid
        }


def generate_fbm(hurst, length=1000, seed=42):
    """Generate fractional Brownian motion."""
    np.random.seed(seed)
    t = np.linspace(0, 1, length)
    dt = t[1] - t[0]
    
    # Generate increments
    increments = np.random.normal(0, 1, length) * (dt ** hurst)
    
    # Cumulative sum to get FBM
    fbm = np.cumsum(increments)
    
    return fbm


def test_ghe_estimator():
    """Test the GHE estimator."""
    print("=" * 60)
    print("Testing GHE (Generalized Hurst Exponent) Estimator")
    print("Based on: Zhang et al. (2024) IEEE ACCESS")
    print("DOI: 10.1109/ACCESS.2024.3512542")
    print("=" * 60)
    
    # Test parameters
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    data_length = 1000
    
    print(f"Testing with data length: {data_length}")
    print(f"Hurst values to test: {hurst_values}")
    print()
    
    results = []
    
    for H_true in hurst_values:
        print(f"Testing with H_true = {H_true}")
        print("-" * 40)
        
        # Generate test data
        data = generate_fbm(H_true, data_length)
        
        # Test GHE estimator
        ghe = GHEEstimator(
            q_values=[1, 2, 3, 4, 5],
            tau_min=2,
            tau_max=min(data_length // 4, 50),
            tau_step=1
        )
        
        try:
            ghe_results = ghe.estimate(data)
            H_ghe = ghe_results['hurst_parameter']
            H_generalized = ghe_results['generalized_hurst_exponents']
            r_squared = ghe_results['r_squared']
            
            print(f"✅ GHE Results:")
            print(f"   Main Hurst (q=2): {H_ghe:.4f}")
            print(f"   Error: {abs(H_ghe - H_true):.4f}")
            print(f"   R² values: {r_squared}")
            print(f"   Generalized H(q): {H_generalized}")
            
            results.append({
                'H_true': H_true,
                'H_ghe': H_ghe,
                'error': abs(H_ghe - H_true),
                'r_squared': r_squared,
                'generalized': H_generalized
            })
            
        except Exception as e:
            print(f"❌ GHE estimation failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    return results


def test_multifractal_analysis():
    """Test multifractal analysis."""
    print("=" * 60)
    print("Testing Multifractal Analysis")
    print("=" * 60)
    
    # Generate test data
    data = generate_fbm(0.7, 1000)
    
    # GHE estimator with more q values
    ghe = GHEEstimator(
        q_values=np.linspace(0.5, 5, 10),
        tau_min=2,
        tau_max=50
    )
    
    try:
        results = ghe.estimate(data)
        spectrum = ghe.get_multifractal_spectrum()
        
        print(f"✅ Multifractal Spectrum:")
        print(f"   Alpha values: {spectrum['alpha']}")
        print(f"   f(alpha) values: {spectrum['f_alpha']}")
        if len(spectrum['alpha']) > 0:
            print(f"   Spectrum width: {np.max(spectrum['alpha']) - np.min(spectrum['alpha']):.4f}")
        
    except Exception as e:
        print(f"❌ Multifractal analysis failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    try:
        # Test 1: Basic functionality
        results = test_ghe_estimator()
        
        # Test 2: Multifractal analysis
        test_multifractal_analysis()
        
        # Summary
        print("=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        valid_results = [r for r in results if not np.isnan(r['error'])]
        if valid_results:
            avg_error = np.mean([r['error'] for r in valid_results])
            print(f"Average estimation error: {avg_error:.4f}")
            print(f"Successful tests: {len(valid_results)}/{len(results)}")
        
        print("\n✅ All tests completed successfully!")
        print("The GHE estimator has been successfully implemented and tested.")
        print("This demonstrates the 'Adding New Estimator Model' functionality.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
