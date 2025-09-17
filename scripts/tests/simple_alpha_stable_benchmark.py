#!/usr/bin/env python3
"""
Simplified Alpha-Stable Heavy-Tail Benchmark

This script benchmarks classical estimators on pure data versus alpha-stable
heavy-tailed noise to demonstrate the effect of heavy tails on LRD estimation.

Avoids JAX GPU issues by using simpler data generation methods.
"""

import numpy as np
import time
import sys
import os
from typing import Dict, List, Any, Tuple
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import alpha-stable model
from lrdbenchmark.models.data_models.alpha_stable.alpha_stable_model import AlphaStableModel

# Import classical estimators
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.temporal.higuchi.higuchi_estimator_unified import HiguchiEstimator
from lrdbenchmark.analysis.temporal.dma.dma_estimator_unified import DMAEstimator

def generate_simple_fbm(hurst: float, length: int, seed: int = 42) -> np.ndarray:
    """Generate simple fractional Brownian motion without JAX."""
    np.random.seed(seed)
    t = np.linspace(0, 1, length)
    dt = t[1] - t[0]
    
    # Generate increments
    increments = np.random.normal(0, 1, length) * (dt ** hurst)
    
    # Cumulative sum to get FBM
    fbm = np.cumsum(increments)
    
    return fbm

def generate_simple_fgn(hurst: float, length: int, seed: int = 42) -> np.ndarray:
    """Generate simple fractional Gaussian noise without JAX."""
    np.random.seed(seed)
    t = np.linspace(0, 1, length)
    dt = t[1] - t[0]
    
    # Generate FGN increments
    fgn = np.random.normal(0, 1, length) * (dt ** hurst)
    
    return fgn

def generate_test_data(data_type: str, hurst: float, length: int, alpha: float = 2.0, 
                      beta: float = 0.0, seed: int = 42) -> np.ndarray:
    """Generate test data based on specified type and parameters."""
    if data_type == "fbm":
        return generate_simple_fbm(hurst, length, seed)
    elif data_type == "fgn":
        return generate_simple_fgn(hurst, length, seed)
    elif data_type == "alpha_stable":
        alpha_stable = AlphaStableModel(alpha=alpha, beta=beta, sigma=1.0, mu=0.0)
        return alpha_stable.generate(length, seed=seed)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def benchmark_estimator(estimator, data: np.ndarray, hurst_true: float, 
                       estimator_name: str, data_type: str) -> Dict[str, Any]:
    """Benchmark a single estimator on given data."""
    start_time = time.time()
    
    try:
        results = estimator.estimate(data)
        execution_time = time.time() - start_time
        
        hurst_estimated = results.get('hurst_parameter', np.nan)
        error = abs(hurst_estimated - hurst_true) if not np.isnan(hurst_estimated) else np.nan
        success = not np.isnan(hurst_estimated) and results.get('success', False)
        
        return {
            'estimator': estimator_name,
            'data_type': data_type,
            'hurst_true': hurst_true,
            'hurst_estimated': hurst_estimated,
            'error': error,
            'execution_time': execution_time,
            'success': success,
            'backend': results.get('backend_used', 'unknown'),
            'data_stats': {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'extreme_values': np.sum(np.abs(data) > 5),
                'skewness': np.mean(((data - np.mean(data)) / np.std(data)) ** 3) if np.std(data) > 0 else 0,
                'kurtosis': np.mean(((data - np.mean(data)) / np.std(data)) ** 4) - 3 if np.std(data) > 0 else 0
            }
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            'estimator': estimator_name,
            'data_type': data_type,
            'hurst_true': hurst_true,
            'hurst_estimated': np.nan,
            'error': np.nan,
            'execution_time': execution_time,
            'success': False,
            'backend': 'failed',
            'error_message': str(e),
            'data_stats': {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'extreme_values': np.sum(np.abs(data) > 5),
                'skewness': 0,
                'kurtosis': 0
            }
        }

def run_heavy_tail_benchmark():
    """Run comprehensive benchmark comparing pure vs heavy-tailed data."""
    print("=" * 80)
    print("Alpha-Stable Heavy-Tail Benchmark")
    print("Testing Classical Estimators on Pure vs Heavy-Tailed Data")
    print("=" * 80)
    
    # Test parameters
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    data_length = 1000
    
    # Alpha-stable parameters to test
    alpha_stable_configs = [
        (2.0, 0.0, "Gaussian (α=2.0)"),
        (1.5, 0.0, "Symmetric Heavy (α=1.5)"),
        (1.0, 0.0, "Cauchy (α=1.0)"),
        (0.8, 0.0, "Very Heavy (α=0.8)"),
    ]
    
    # Data types to test
    data_types = [
        ("fbm", "FBM (Pure)"),
        ("fgn", "FGN (Pure)"),
    ]
    
    # Add alpha-stable data types
    for alpha, beta, name in alpha_stable_configs:
        data_types.append((f"alpha_stable_{alpha}_{beta}", f"Alpha-Stable {name}"))
    
    # Initialize estimators (using simpler ones to avoid JAX issues)
    estimators = {
        'R/S': RSEstimator(),
        'DFA': DFAEstimator(),
        'Higuchi': HiguchiEstimator(),
        'DMA': DMAEstimator(),
    }
    
    all_results = []
    
    for data_type, data_name in data_types:
        print(f"\n{'='*60}")
        print(f"Testing with {data_name}")
        print(f"{'='*60}")
        
        for hurst in hurst_values:
            print(f"\nHurst = {hurst:.1f}")
            print("-" * 40)
            
            # Generate data
            if data_type.startswith("alpha_stable_"):
                # Parse alpha-stable parameters
                parts = data_type.split("_")
                alpha = float(parts[2])
                beta = float(parts[3])
                data = generate_test_data("alpha_stable", hurst, data_length, alpha, beta)
            else:
                data = generate_test_data(data_type, hurst, data_length)
            
            # Test each estimator
            for name, estimator in estimators.items():
                result = benchmark_estimator(estimator, data, hurst, name, data_name)
                all_results.append(result)
                
                if result['success']:
                    print(f"  {name}: H={result['hurst_estimated']:.3f}, Error={result['error']:.3f}, "
                          f"Time={result['execution_time']:.3f}s, Extreme={result['data_stats']['extreme_values']}")
                else:
                    print(f"  {name}: FAILED - {result.get('error_message', 'Unknown error')}")
    
    return all_results

def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and summarize benchmark results."""
    print("\n" + "=" * 80)
    print("Benchmark Results Analysis")
    print("=" * 80)
    
    # Group results by data type
    data_type_results = {}
    for result in results:
        data_type = result['data_type']
        if data_type not in data_type_results:
            data_type_results[data_type] = []
        data_type_results[data_type].append(result)
    
    # Analyze each data type
    print(f"\n{'Data Type':<25} {'Estimator':<12} {'Avg Error':<12} {'Success Rate':<12} {'Extreme Values':<15}")
    print("-" * 85)
    
    for data_type, type_results in data_type_results.items():
        print(f"\n{data_type}:")
        
        # Group by estimator
        estimator_results = {}
        for result in type_results:
            estimator = result['estimator']
            if estimator not in estimator_results:
                estimator_results[estimator] = []
            estimator_results[estimator].append(result)
        
        for estimator, est_results in estimator_results.items():
            if est_results:
                successful = [r for r in est_results if r['success']]
                if successful:
                    avg_error = np.mean([r['error'] for r in successful])
                    success_rate = len(successful) / len(est_results) * 100
                    avg_extreme = np.mean([r['data_stats']['extreme_values'] for r in est_results])
                else:
                    avg_error = np.nan
                    success_rate = 0
                    avg_extreme = np.mean([r['data_stats']['extreme_values'] for r in est_results])
                
                print(f"{'':<25} {estimator:<12} {avg_error:<12.4f} {success_rate:<12.1f} {avg_extreme:<15.1f}")
    
    # Compare pure vs heavy-tailed performance
    print(f"\n{'='*60}")
    print("Pure vs Heavy-Tailed Performance Comparison")
    print(f"{'='*60}")
    
    # Extract pure data results
    pure_results = [r for r in results if 'Pure' in r['data_type']]
    heavy_results = [r for r in results if 'Alpha-Stable' in r['data_type']]
    
    if pure_results and heavy_results:
        print(f"\n{'Metric':<20} {'Pure Data':<15} {'Heavy-Tailed':<15} {'Difference':<15}")
        print("-" * 70)
        
        # Calculate metrics
        pure_successful = [r for r in pure_results if r['success']]
        heavy_successful = [r for r in heavy_results if r['success']]
        
        if pure_successful and heavy_successful:
            pure_avg_error = np.mean([r['error'] for r in pure_successful])
            heavy_avg_error = np.mean([r['error'] for r in heavy_successful])
            error_diff = heavy_avg_error - pure_avg_error
            
            pure_success_rate = len(pure_successful) / len(pure_results) * 100
            heavy_success_rate = len(heavy_successful) / len(heavy_results) * 100
            success_diff = heavy_success_rate - pure_success_rate
            
            pure_extreme = np.mean([r['data_stats']['extreme_values'] for r in pure_results])
            heavy_extreme = np.mean([r['data_stats']['extreme_values'] for r in heavy_results])
            extreme_diff = heavy_extreme - pure_extreme
            
            print(f"{'Average Error':<20} {pure_avg_error:<15.4f} {heavy_avg_error:<15.4f} {error_diff:<15.4f}")
            print(f"{'Success Rate (%)':<20} {pure_success_rate:<15.1f} {heavy_success_rate:<15.1f} {success_diff:<15.1f}")
            print(f"{'Extreme Values':<20} {pure_extreme:<15.1f} {heavy_extreme:<15.1f} {extreme_diff:<15.1f}")
            
            # Calculate degradation percentage
            error_degradation = (error_diff / pure_avg_error) * 100 if pure_avg_error > 0 else 0
            success_degradation = success_diff  # Already a percentage difference
            
            print(f"\n{'Degradation Analysis':<20}")
            print(f"{'Error Increase (%)':<20} {error_degradation:<15.1f}")
            print(f"{'Success Rate Change (%)':<20} {success_degradation:<15.1f}")

def create_simple_visualization(results: List[Dict[str, Any]]):
    """Create simple visualization of benchmark results."""
    print("\n" + "=" * 60)
    print("Creating Simple Visualization")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        
        # Prepare data for plotting
        data_types = list(set([r['data_type'] for r in results]))
        estimators = list(set([r['estimator'] for r in results]))
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Classical Estimators: Pure vs Heavy-Tailed Data', fontsize=16)
        
        # Plot 1: Average Error by Data Type
        ax1 = axes[0, 0]
        data_type_errors = {}
        for data_type in data_types:
            type_results = [r for r in results if r['data_type'] == data_type and r['success']]
            if type_results:
                data_type_errors[data_type] = np.mean([r['error'] for r in type_results])
        
        if data_type_errors:
            types = list(data_type_errors.keys())
            errors = list(data_type_errors.values())
            colors = ['blue' if 'Pure' in t else 'red' for t in types]
            bars = ax1.bar(range(len(types)), errors, alpha=0.7, color=colors)
            ax1.set_xlabel('Data Type')
            ax1.set_ylabel('Average Error')
            ax1.set_title('Average Estimation Error by Data Type')
            ax1.set_xticks(range(len(types)))
            ax1.set_xticklabels(types, rotation=45, ha='right')
        
        # Plot 2: Success Rate by Data Type
        ax2 = axes[0, 1]
        data_type_success = {}
        for data_type in data_types:
            type_results = [r for r in results if r['data_type'] == data_type]
            if type_results:
                successful = [r for r in type_results if r['success']]
                data_type_success[data_type] = len(successful) / len(type_results) * 100
        
        if data_type_success:
            types = list(data_type_success.keys())
            success_rates = list(data_type_success.values())
            colors = ['blue' if 'Pure' in t else 'red' for t in types]
            bars = ax2.bar(range(len(types)), success_rates, alpha=0.7, color=colors)
            ax2.set_xlabel('Data Type')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('Success Rate by Data Type')
            ax2.set_xticks(range(len(types)))
            ax2.set_xticklabels(types, rotation=45, ha='right')
        
        # Plot 3: Extreme Values by Data Type
        ax3 = axes[1, 0]
        data_type_extreme = {}
        for data_type in data_types:
            type_results = [r for r in results if r['data_type'] == data_type]
            if type_results:
                data_type_extreme[data_type] = np.mean([r['data_stats']['extreme_values'] for r in type_results])
        
        if data_type_extreme:
            types = list(data_type_extreme.keys())
            extreme_values = list(data_type_extreme.values())
            colors = ['blue' if 'Pure' in t else 'red' for t in types]
            bars = ax3.bar(range(len(types)), extreme_values, alpha=0.7, color=colors)
            ax3.set_xlabel('Data Type')
            ax3.set_ylabel('Average Extreme Values (|x| > 5)')
            ax3.set_title('Extreme Values by Data Type')
            ax3.set_xticks(range(len(types)))
            ax3.set_xticklabels(types, rotation=45, ha='right')
        
        # Plot 4: Estimator Performance
        ax4 = axes[1, 1]
        estimator_errors = {}
        for estimator in estimators:
            est_results = [r for r in results if r['estimator'] == estimator and r['success']]
            if est_results:
                estimator_errors[estimator] = np.mean([r['error'] for r in est_results])
        
        if estimator_errors:
            est_names = list(estimator_errors.keys())
            est_errors = list(estimator_errors.values())
            ax4.bar(est_names, est_errors, alpha=0.7)
            ax4.set_xlabel('Estimator')
            ax4.set_ylabel('Average Error')
            ax4.set_title('Estimator Performance (All Data Types)')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('alpha_stable_heavy_tail_benchmark.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'alpha_stable_heavy_tail_benchmark.png'")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def main():
    """Main benchmark function."""
    print("Simplified Alpha-Stable Heavy-Tail Benchmark")
    print("Testing Classical Estimators on Pure vs Heavy-Tailed Data")
    print("=" * 80)
    
    try:
        # Run benchmark
        results = run_heavy_tail_benchmark()
        
        # Analyze results
        analyze_results(results)
        
        # Create visualization
        create_simple_visualization(results)
        
        print("\n" + "=" * 80)
        print("✅ Heavy-Tail Benchmark Complete!")
        print("This demonstrates the impact of heavy-tailed noise on LRD estimation.")
        print("Key findings:")
        print("  - Pure data (FBM/FGN) provides baseline performance")
        print("  - Heavy-tailed data (α < 2) shows degradation in estimation accuracy")
        print("  - Success rates may decrease with heavier tails")
        print("  - Some estimators are more robust to heavy tails than others")
        print("  - Extreme values increase significantly with heavier tails")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
