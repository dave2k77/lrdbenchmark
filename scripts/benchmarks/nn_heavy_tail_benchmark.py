#!/usr/bin/env python3
"""
Neural Network Heavy-Tail Benchmark

This script tests how Neural Network estimators perform on pure data versus 
alpha-stable heavy-tailed data to demonstrate their robustness to extreme values.

Tests: CNN, LSTM, GRU, Transformer on FBM/FGN vs Alpha-Stable data.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import data models
from lrdbenchmark.models.data_models.alpha_stable.alpha_stable_model import AlphaStableModel

# Import Neural Network estimators
from lrdbenchmark.analysis.machine_learning.cnn_estimator_unified import CNNEstimator
from lrdbenchmark.analysis.machine_learning.lstm_estimator_unified import LSTMEstimator
from lrdbenchmark.analysis.machine_learning.gru_estimator_unified import GRUEstimator
from lrdbenchmark.analysis.machine_learning.transformer_estimator_unified import TransformerEstimator

def generate_simple_fbm(hurst: float, length: int, seed: int = 42) -> np.ndarray:
    """Generate simple fractional Brownian motion."""
    np.random.seed(seed)
    t = np.linspace(0, 1, length)
    dt = t[1] - t[0]
    
    # Generate increments
    increments = np.random.normal(0, dt**hurst, length-1)
    
    # Cumulative sum to get FBM
    fbm = np.cumsum(np.concatenate([[0], increments]))
    return fbm

def generate_simple_fgn(hurst: float, length: int, seed: int = 42) -> np.ndarray:
    """Generate simple fractional Gaussian noise."""
    fbm = generate_simple_fbm(hurst, length + 1, seed)
    return np.diff(fbm)

def analyze_data_characteristics(data: np.ndarray, name: str) -> Dict[str, Any]:
    """Analyze data characteristics."""
    stats = {
        'name': name,
        'length': len(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'kurtosis': np.mean(((data - np.mean(data)) / np.std(data))**4) - 3,
        'extreme_values': np.sum(np.abs(data) > 5),
        'extreme_ratio': np.sum(np.abs(data) > 5) / len(data),
        'finite_variance': np.isfinite(np.var(data)),
        'finite_mean': np.isfinite(np.mean(data))
    }
    return stats

def test_nn_estimator(estimator, data: np.ndarray, true_hurst: float, name: str) -> Dict[str, Any]:
    """Test a Neural Network estimator on data."""
    try:
        # Set optimization to numpy to avoid JAX issues
        estimator.use_optimization = 'numpy'
        
        # Estimate Hurst
        result = estimator.estimate(data)
        
        if result is not None and 'hurst_parameter' in result:
            estimated_hurst = result['hurst_parameter']
            error = abs(estimated_hurst - true_hurst)
            success = True
        else:
            estimated_hurst = np.nan
            error = np.nan
            success = False
            
    except Exception as e:
        estimated_hurst = np.nan
        error = np.nan
        success = False
        print(f"    Error with {name}: {str(e)[:100]}...")
    
    return {
        'estimator': name,
        'true_hurst': true_hurst,
        'estimated_hurst': estimated_hurst,
        'error': error,
        'success': success
    }

def run_nn_heavy_tail_benchmark():
    """Run the Neural Network heavy-tail benchmark."""
    print("🧠 Neural Network Heavy-Tail Benchmark")
    print("=" * 50)
    
    # Test parameters
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    data_lengths = [1000, 2000]
    alpha_values = [2.0, 1.5, 1.0, 0.8]  # 2.0 = Gaussian, <2.0 = heavy-tailed
    
    # Initialize Neural Network estimators
    estimators = {
        'CNN': CNNEstimator(),
        'LSTM': LSTMEstimator(),
        'GRU': GRUEstimator(),
        'Transformer': TransformerEstimator()
    }
    
    results = []
    
    print(f"\n📊 Testing {len(estimators)} Neural Network estimators on {len(hurst_values)} Hurst values")
    print(f"   Data lengths: {data_lengths}")
    print(f"   Alpha values: {alpha_values}")
    
    for hurst in hurst_values:
        print(f"\n🎯 Testing Hurst = {hurst}")
        print("-" * 30)
        
        for length in data_lengths:
            print(f"\n  📏 Data length = {length}")
            
            # Generate pure data (FBM)
            fbm_data = generate_simple_fbm(hurst, length, seed=42)
            fgn_data = generate_simple_fgn(hurst, length, seed=42)
            
            # Analyze pure data characteristics
            fbm_stats = analyze_data_characteristics(fbm_data, f"FBM (H={hurst})")
            fgn_stats = analyze_data_characteristics(fgn_data, f"FGN (H={hurst})")
            
            print(f"    Pure FBM: kurtosis={fbm_stats['kurtosis']:.3f}, extreme_values={fbm_stats['extreme_values']}")
            print(f"    Pure FGN: kurtosis={fgn_stats['kurtosis']:.3f}, extreme_values={fgn_stats['extreme_values']}")
            
            # Test Neural Network estimators on pure data
            print(f"    🧠 Testing Neural Network estimators on pure data:")
            for est_name, estimator in estimators.items():
                fbm_result = test_nn_estimator(estimator, fbm_data, hurst, f"{est_name}_FBM")
                fgn_result = test_nn_estimator(estimator, fgn_data, hurst, f"{est_name}_FGN")
                
                results.extend([fbm_result, fgn_result])
                
                if fbm_result['success']:
                    print(f"      {est_name} FBM: {fbm_result['estimated_hurst']:.3f} (error: {fbm_result['error']:.3f})")
                else:
                    print(f"      {est_name} FBM: FAILED")
                    
                if fgn_result['success']:
                    print(f"      {est_name} FGN: {fgn_result['estimated_hurst']:.3f} (error: {fgn_result['error']:.3f})")
                else:
                    print(f"      {est_name} FGN: FAILED")
            
            # Test on alpha-stable data
            for alpha in alpha_values:
                print(f"\n    🔥 Testing alpha-stable data (α={alpha}):")
                
                # Generate alpha-stable data
                alpha_model = AlphaStableModel(
                    alpha=alpha,
                    beta=0.0,  # Symmetric
                    sigma=1.0,
                    mu=0.0,
                    use_optimization='numpy'
                )
                
                alpha_data = alpha_model.generate(length, seed=42)
                alpha_stats = analyze_data_characteristics(alpha_data, f"Alpha-Stable (α={alpha})")
                
                print(f"      Data: kurtosis={alpha_stats['kurtosis']:.1f}, extreme_values={alpha_stats['extreme_values']}")
                
                # Test Neural Network estimators on alpha-stable data
                for est_name, estimator in estimators.items():
                    alpha_result = test_nn_estimator(estimator, alpha_data, hurst, f"{est_name}_Alpha{alpha}")
                    results.append(alpha_result)
                    
                    if alpha_result['success']:
                        print(f"        {est_name}: {alpha_result['estimated_hurst']:.3f} (error: {alpha_result['error']:.3f})")
                    else:
                        print(f"        {est_name}: FAILED")
    
    return results

def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and visualize results."""
    print(f"\n📈 Results Analysis")
    print("=" * 50)
    
    # Convert to DataFrame-like structure for analysis
    data_types = []
    estimators = []
    errors = []
    successes = []
    
    for result in results:
        if 'FBM' in result['estimator'] or 'FGN' in result['estimator']:
            data_types.append('Pure')
        else:
            data_types.append('Alpha-Stable')
        
        estimator_name = result['estimator'].split('_')[0]
        estimators.append(estimator_name)
        errors.append(result['error'])
        successes.append(result['success'])
    
    # Calculate success rates
    print(f"\n🎯 Success Rates by Data Type:")
    pure_success = np.mean([s for d, s in zip(data_types, successes) if d == 'Pure'])
    alpha_success = np.mean([s for d, s in zip(data_types, successes) if d == 'Alpha-Stable'])
    
    print(f"  Pure Data (FBM/FGN): {pure_success:.1%}")
    print(f"  Alpha-Stable Data: {alpha_success:.1%}")
    
    # Calculate success rates by estimator
    print(f"\n🧠 Success Rates by Estimator:")
    for estimator in set(estimators):
        est_success = np.mean([s for e, s in zip(estimators, successes) if e == estimator])
        print(f"  {estimator}: {est_success:.1%}")
    
    # Calculate error statistics
    print(f"\n📊 Error Statistics (successful estimates only):")
    successful_errors = [e for e, s in zip(errors, successes) if s and not np.isnan(e)]
    
    if successful_errors:
        print(f"  Mean Error: {np.mean(successful_errors):.3f}")
        print(f"  Median Error: {np.median(successful_errors):.3f}")
        print(f"  Std Error: {np.std(successful_errors):.3f}")
        print(f"  Min Error: {np.min(successful_errors):.3f}")
        print(f"  Max Error: {np.max(successful_errors):.3f}")
    else:
        print("  No successful estimates to analyze")
    
    # Create visualization
    create_visualization(results, data_types, estimators, errors, successes)

def create_visualization(results: List[Dict[str, Any]], data_types: List[str], 
                        estimators: List[str], errors: List[float], successes: List[bool]):
    """Create visualization of results."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Neural Network Estimators: Pure vs Alpha-Stable Data Performance', fontsize=16, fontweight='bold')
        
        # 1. Success rate by data type
        data_type_counts = {}
        for dt in set(data_types):
            data_type_counts[dt] = {
                'success': sum(1 for d, s in zip(data_types, successes) if d == dt and s),
                'total': sum(1 for d in data_types if d == dt)
            }
        
        labels = list(data_type_counts.keys())
        success_counts = [data_type_counts[dt]['success'] for dt in labels]
        total_counts = [data_type_counts[dt]['total'] for dt in labels]
        success_rates = [s/t for s, t in zip(success_counts, total_counts)]
        
        bars1 = ax1.bar(labels, success_rates, color=['skyblue', 'lightcoral'], alpha=0.7)
        ax1.set_title('Success Rate by Data Type')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 2. Success rate by estimator
        estimator_counts = {}
        for est in set(estimators):
            estimator_counts[est] = {
                'success': sum(1 for e, s in zip(estimators, successes) if e == est and s),
                'total': sum(1 for e in estimators if e == est)
            }
        
        est_labels = list(estimator_counts.keys())
        est_success_counts = [estimator_counts[est]['success'] for est in est_labels]
        est_total_counts = [estimator_counts[est]['total'] for est in est_labels]
        est_success_rates = [s/t for s, t in zip(est_success_counts, est_total_counts)]
        
        bars2 = ax2.bar(est_labels, est_success_rates, color=['lightgreen', 'orange', 'purple', 'red'], alpha=0.7)
        ax2.set_title('Success Rate by Estimator')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars2, est_success_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 3. Error distribution (successful estimates only)
        successful_errors = [e for e, s in zip(errors, successes) if s and not np.isnan(e)]
        
        if successful_errors:
            ax3.hist(successful_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_title('Error Distribution (Successful Estimates)')
            ax3.set_xlabel('Absolute Error')
            ax3.set_ylabel('Frequency')
            ax3.axvline(np.mean(successful_errors), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(successful_errors):.3f}')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No successful estimates', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Error Distribution (No Data)')
        
        # 4. Error by data type (box plot)
        pure_errors = [e for d, e, s in zip(data_types, errors, successes) 
                      if d == 'Pure' and s and not np.isnan(e)]
        alpha_errors = [e for d, e, s in zip(data_types, errors, successes) 
                       if d == 'Alpha-Stable' and s and not np.isnan(e)]
        
        if pure_errors or alpha_errors:
            box_data = []
            box_labels = []
            if pure_errors:
                box_data.append(pure_errors)
                box_labels.append('Pure')
            if alpha_errors:
                box_data.append(alpha_errors)
                box_labels.append('Alpha-Stable')
            
            ax4.boxplot(box_data, labels=box_labels)
            ax4.set_title('Error Distribution by Data Type')
            ax4.set_ylabel('Absolute Error')
        else:
            ax4.text(0.5, 0.5, 'No successful estimates', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Error Distribution by Data Type (No Data)')
        
        plt.tight_layout()
        plt.savefig('nn_heavy_tail_benchmark.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved as 'nn_heavy_tail_benchmark.png'")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")

if __name__ == "__main__":
    print("🚀 Starting Neural Network Heavy-Tail Benchmark...")
    print("   Testing Neural Network estimators on pure vs alpha-stable data")
    print("   Focus: Robustness to heavy-tailed noise")
    
    try:
        results = run_nn_heavy_tail_benchmark()
        analyze_results(results)
        
        print(f"\n✅ Neural Network Heavy-Tail Benchmark completed!")
        print(f"   Total tests: {len(results)}")
        print(f"   Successful estimates: {sum(1 for r in results if r['success'])}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
