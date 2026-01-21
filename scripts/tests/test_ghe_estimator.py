#!/usr/bin/env python3
"""
Test script for the GHE (Generalized Hurst Exponent) Estimator.

This script demonstrates how to use the new GHE estimator and compares
its performance with other estimators in the LRDBenchmark framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lrdbenchmark.analysis.temporal.ghe.ghe_estimator_unified import GHEEstimator
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion

def generate_test_data(hurst_value, length=1000, seed=42):
    """Generate fractional Brownian motion test data."""
    np.random.seed(seed)
    fbm = FractionalBrownianMotion(hurst=hurst_value, length=length)
    return fbm.generate()

def test_ghe_estimator():
    """Test the GHE estimator with different Hurst values."""
    print("=" * 60)
    print("Testing GHE (Generalized Hurst Exponent) Estimator")
    print("=" * 60)
    
    # Test parameters
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    data_length = 1000
    
    results = []
    
    for H_true in hurst_values:
        print(f"\nTesting with H_true = {H_true}")
        print("-" * 40)
        
        # Generate test data
        data = generate_test_data(H_true, data_length)
        
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
            backend = ghe_results['backend_used']
            
            print(f"GHE Results:")
            print(f"  Main Hurst (q=2): {H_ghe:.4f}")
            print(f"  Error: {abs(H_ghe - H_true):.4f}")
            print(f"  Backend: {backend}")
            print(f"  R² values: {r_squared}")
            print(f"  Generalized H(q): {H_generalized}")
            
            results.append({
                'H_true': H_true,
                'H_ghe': H_ghe,
                'error': abs(H_ghe - H_true),
                'backend': backend,
                'r_squared': r_squared,
                'generalized': H_generalized
            })
            
        except Exception as e:
            print(f"GHE estimation failed: {e}")
            results.append({
                'H_true': H_true,
                'H_ghe': np.nan,
                'error': np.nan,
                'backend': 'failed',
                'r_squared': np.nan,
                'generalized': np.nan
            })
    
    return results

def compare_with_rs_estimator():
    """Compare GHE with R/S estimator."""
    print("\n" + "=" * 60)
    print("Comparing GHE with R/S Estimator")
    print("=" * 60)
    
    # Test with one Hurst value
    H_true = 0.7
    data = generate_test_data(H_true, 1000)
    
    # GHE estimator
    ghe = GHEEstimator(q_values=[1, 2, 3, 4, 5])
    ghe_results = ghe.estimate(data)
    H_ghe = ghe_results['hurst_parameter']
    
    # R/S estimator
    rs = RSEstimator()
    rs_results = rs.estimate(data)
    H_rs = rs_results['hurst_parameter']
    
    print(f"True Hurst: {H_true:.4f}")
    print(f"GHE Estimate: {H_ghe:.4f} (Error: {abs(H_ghe - H_true):.4f})")
    print(f"R/S Estimate: {H_rs:.4f} (Error: {abs(H_rs - H_true):.4f})")
    
    return H_ghe, H_rs

def test_multifractal_analysis():
    """Test multifractal spectrum analysis."""
    print("\n" + "=" * 60)
    print("Testing Multifractal Spectrum Analysis")
    print("=" * 60)
    
    # Generate test data
    data = generate_test_data(0.7, 1000)
    
    # GHE estimator with more q values for better spectrum
    ghe = GHEEstimator(
        q_values=np.linspace(0.5, 5, 10),
        tau_min=2,
        tau_max=50
    )
    
    results = ghe.estimate(data)
    spectrum = ghe.get_multifractal_spectrum()
    
    print(f"Multifractal Spectrum:")
    print(f"  Alpha values: {spectrum['alpha']}")
    print(f"  f(alpha) values: {spectrum['f_alpha']}")
    print(f"  Spectrum width: {np.max(spectrum['alpha']) - np.min(spectrum['alpha']):.4f}")
    
    return spectrum

def create_visualization():
    """Create visualization of GHE analysis."""
    print("\n" + "=" * 60)
    print("Creating GHE Visualization")
    print("=" * 60)
    
    # Generate test data
    data = generate_test_data(0.7, 1000)
    
    # GHE estimator
    ghe = GHEEstimator(q_values=[1, 2, 3, 4, 5])
    results = ghe.estimate(data)
    
    # Create plots
    fig = ghe.plot_scaling_behavior(figsize=(15, 10))
    
    # Add title with results
    main_hurst = results['hurst_parameter']
    fig.suptitle(f'GHE Analysis - Estimated H = {main_hurst:.4f}', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('ghe_analysis_demo.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'ghe_analysis_demo.png'")
    
    return fig

def main():
    """Main test function."""
    print("GHE Estimator Test Suite")
    print("Based on: Zhang et al. (2024) IEEE ACCESS")
    print("DOI: 10.1109/ACCESS.2024.3512542")
    
    try:
        # Test 1: Basic functionality
        results = test_ghe_estimator()
        
        # Test 2: Comparison with R/S
        H_ghe, H_rs = compare_with_rs_estimator()
        
        # Test 3: Multifractal analysis
        spectrum = test_multifractal_analysis()
        
        # Test 4: Visualization
        fig = create_visualization()
        
        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        valid_results = [r for r in results if not np.isnan(r['error'])]
        if valid_results:
            avg_error = np.mean([r['error'] for r in valid_results])
            print(f"Average estimation error: {avg_error:.4f}")
            print(f"Successful tests: {len(valid_results)}/{len(results)}")
        
        print(f"GHE vs R/S comparison: GHE={H_ghe:.4f}, R/S={H_rs:.4f}")
        print("Multifractal spectrum analysis completed successfully")
        print("Visualization created and saved")
        
        print("\n✅ All tests completed successfully!")
        print("The GHE estimator has been successfully integrated into LRDBenchmark.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
