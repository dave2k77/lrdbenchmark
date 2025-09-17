#!/usr/bin/env python3
"""
GHE Estimator Benchmark Demo

This script demonstrates the integration of the GHE (Generalized Hurst Exponent) 
estimator into the LRDBenchmark framework and compares its performance with 
other classical estimators.

Based on the paper:
"Typical Algorithms for Estimating Hurst Exponent of Time Sequence: A Data Analyst's Perspective"
by HONG-YAN ZHANG, ZHI-QIANG FENG, SI-YU FENG, AND YU ZHOU
IEEE ACCESS 2024, DOI: 10.1109/ACCESS.2024.3512542
"""

import numpy as np
import time
import sys
import os
from typing import Dict, List, Any
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import estimators
from lrdbenchmark.analysis.temporal.ghe.ghe_estimator_unified import GHEEstimator
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.temporal.higuchi.higuchi_estimator_unified import HiguchiEstimator

# Import data models
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise

def generate_test_data(model_type: str, hurst: float, length: int, seed: int = 42) -> np.ndarray:
    """Generate test data using specified model."""
    np.random.seed(seed)
    
    if model_type == "fbm":
        fbm = FractionalBrownianMotion(H=hurst, sigma=1.0)
        return fbm.generate(length)
    elif model_type == "fgn":
        fgn = FractionalGaussianNoise(H=hurst, sigma=1.0)
        return fgn.generate(length)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def benchmark_estimator(estimator, data: np.ndarray, hurst_true: float, 
                       estimator_name: str) -> Dict[str, Any]:
    """Benchmark a single estimator."""
    start_time = time.time()
    
    try:
        results = estimator.estimate(data)
        execution_time = time.time() - start_time
        
        hurst_estimated = results.get('hurst_parameter', np.nan)
        error = abs(hurst_estimated - hurst_true) if not np.isnan(hurst_estimated) else np.nan
        success = not np.isnan(hurst_estimated) and results.get('success', False)
        
        return {
            'estimator': estimator_name,
            'hurst_true': hurst_true,
            'hurst_estimated': hurst_estimated,
            'error': error,
            'execution_time': execution_time,
            'success': success,
            'backend': results.get('backend_used', 'unknown'),
            'additional_info': {
                'method': results.get('method', estimator_name),
                'data_length': len(data),
                'r_squared': results.get('r_squared', None),
                'generalized_hurst': results.get('generalized_hurst_exponents', None) if estimator_name == 'GHE' else None
            }
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            'estimator': estimator_name,
            'hurst_true': hurst_true,
            'hurst_estimated': np.nan,
            'error': np.nan,
            'execution_time': execution_time,
            'success': False,
            'backend': 'failed',
            'error_message': str(e),
            'additional_info': {}
        }

def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing GHE with other estimators."""
    print("=" * 80)
    print("GHE Estimator Comprehensive Benchmark")
    print("Based on: Zhang et al. (2024) IEEE ACCESS")
    print("DOI: 10.1109/ACCESS.2024.3512542")
    print("=" * 80)
    
    # Test parameters
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    data_lengths = [500, 1000, 2000]
    models = ["fbm", "fgn"]
    
    # Initialize estimators
    estimators = {
        'GHE': GHEEstimator(q_values=[1, 2, 3, 4, 5], tau_min=2, tau_max=50),
        'R/S': RSEstimator(),
        'DFA': DFAEstimator(),
        'Higuchi': HiguchiEstimator()
    }
    
    all_results = []
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Testing with {model.upper()} data")
        print(f"{'='*60}")
        
        for length in data_lengths:
            print(f"\nData length: {length}")
            print("-" * 40)
            
            for hurst in hurst_values:
                print(f"Hurst = {hurst:.1f}: ", end="")
                
                # Generate test data
                data = generate_test_data(model, hurst, length)
                
                # Test each estimator
                for name, estimator in estimators.items():
                    result = benchmark_estimator(estimator, data, hurst, name)
                    all_results.append(result)
                    
                    if result['success']:
                        print(f"{name}({result['error']:.3f}) ", end="")
                    else:
                        print(f"{name}(FAIL) ", end="")
                
                print()  # New line after each Hurst value
    
    return all_results

def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and summarize benchmark results."""
    print("\n" + "=" * 80)
    print("Benchmark Results Analysis")
    print("=" * 80)
    
    # Group results by estimator
    estimator_results = {}
    for result in results:
        estimator = result['estimator']
        if estimator not in estimator_results:
            estimator_results[estimator] = []
        estimator_results[estimator].append(result)
    
    # Calculate statistics for each estimator
    print(f"\n{'Estimator':<12} {'Success Rate':<12} {'Avg Error':<12} {'Avg Time (s)':<15} {'Backend':<10}")
    print("-" * 70)
    
    for estimator, est_results in estimator_results.items():
        successful = [r for r in est_results if r['success']]
        success_rate = len(successful) / len(est_results) * 100
        
        if successful:
            avg_error = np.mean([r['error'] for r in successful])
            avg_time = np.mean([r['execution_time'] for r in est_results])
            backend = successful[0]['backend']
        else:
            avg_error = np.nan
            avg_time = np.mean([r['execution_time'] for r in est_results])
            backend = 'failed'
        
        print(f"{estimator:<12} {success_rate:>10.1f}% {avg_error:>10.4f} {avg_time:>13.4f} {backend:<10}")
    
    # Detailed GHE analysis
    ghe_results = [r for r in results if r['estimator'] == 'GHE' and r['success']]
    if ghe_results:
        print(f"\n{'='*60}")
        print("GHE Detailed Analysis")
        print(f"{'='*60}")
        
        print(f"Total GHE tests: {len([r for r in results if r['estimator'] == 'GHE'])}")
        print(f"Successful GHE tests: {len(ghe_results)}")
        print(f"GHE Success rate: {len(ghe_results) / len([r for r in results if r['estimator'] == 'GHE']) * 100:.1f}%")
        
        if ghe_results:
            print(f"Average GHE error: {np.mean([r['error'] for r in ghe_results]):.4f}")
            print(f"GHE error std: {np.std([r['error'] for r in ghe_results]):.4f}")
            
            # Show R² values
            r_squared_values = []
            for r in ghe_results:
                if 'r_squared' in r['additional_info'] and r['additional_info']['r_squared'] is not None:
                    r_squared_values.extend(r['additional_info']['r_squared'])
            
            if r_squared_values:
                print(f"Average R²: {np.mean(r_squared_values):.4f}")
                print(f"R² std: {np.std(r_squared_values):.4f}")

def test_multifractal_capabilities():
    """Test GHE multifractal analysis capabilities."""
    print(f"\n{'='*60}")
    print("GHE Multifractal Analysis Test")
    print(f"{'='*60}")
    
    # Generate test data
    data = generate_test_data("fbm", 0.7, 2000)
    
    # Test with different q values
    q_configs = [
        ([1, 2, 3, 4, 5], "Standard"),
        (np.linspace(0.5, 5, 10), "Extended"),
        (np.linspace(0.1, 10, 20), "Comprehensive")
    ]
    
    for q_values, config_name in q_configs:
        print(f"\n{config_name} q-values: {len(q_values)} points")
        print("-" * 40)
        
        try:
            ghe = GHEEstimator(q_values=q_values, tau_min=2, tau_max=100)
            results = ghe.estimate(data)
            spectrum = ghe.get_multifractal_spectrum()
            
            print(f"Main Hurst: {results['hurst_parameter']:.4f}")
            print(f"Spectrum width: {np.max(spectrum['alpha']) - np.min(spectrum['alpha']):.4f}")
            print(f"Valid spectrum points: {len(spectrum['alpha'])}")
            
        except Exception as e:
            print(f"Failed: {e}")

def main():
    """Main benchmark function."""
    print("GHE Estimator Integration Demo")
    print("Testing 'Adding New Estimator Model' functionality")
    
    try:
        # Run comprehensive benchmark
        results = run_comprehensive_benchmark()
        
        # Analyze results
        analyze_results(results)
        
        # Test multifractal capabilities
        test_multifractal_capabilities()
        
        print(f"\n{'='*80}")
        print("✅ GHE Estimator Integration Complete!")
        print("The GHE estimator has been successfully integrated into LRDBenchmark.")
        print("This demonstrates the framework's capability to add new estimator models.")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
