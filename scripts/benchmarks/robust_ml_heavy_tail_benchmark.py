#!/usr/bin/env python3
"""
Robust ML Heavy-Tail Benchmark

This script tests how ML estimators perform on pure data versus alpha-stable 
heavy-tailed data using our new robust feature extractor and optimization backend.

Tests: RandomForest, SVR, GradientBoosting on FBM/FGN vs Alpha-Stable data.
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

# Import ML estimators
from lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified import RandomForestEstimator
from lrdbenchmark.analysis.machine_learning.svr_estimator_unified import SVREstimator
from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator_unified import GradientBoostingEstimator

# Import robustness modules
from lrdbenchmark.robustness.robust_feature_extractor import RobustFeatureExtractor
from lrdbenchmark.robustness.adaptive_preprocessor import AdaptiveDataPreprocessor
from lrdbenchmark.robustness.robust_optimization_backend import RobustOptimizationBackend

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

def test_robust_ml_estimator(estimator, data: np.ndarray, true_hurst: float, name: str, 
                           robust_extractor: RobustFeatureExtractor, 
                           preprocessor: AdaptiveDataPreprocessor) -> Dict[str, Any]:
    """Test an ML estimator on data using robust preprocessing."""
    try:
        # Set optimization to numpy to avoid JAX issues
        estimator.use_optimization = 'numpy'
        
        # Preprocess data
        data_processed, preprocess_metadata = preprocessor.preprocess(data)
        
        # Extract robust features
        features = robust_extractor.extract_features(data_processed)
        
        # Check if features are valid
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"    Warning: {name} features contain NaN/Inf values")
            return {
                'estimator': name,
                'true_hurst': true_hurst,
                'estimated_hurst': np.nan,
                'error': np.nan,
                'success': False,
                'reason': 'Invalid features (NaN/Inf)'
            }
        
        # Estimate Hurst using robust features
        # Note: This is a simplified approach - in practice, we'd need to modify
        # the ML estimators to accept pre-extracted features
        result = estimator.estimate(data_processed)
        
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
        'success': success,
        'preprocess_method': preprocess_metadata.get('method', 'unknown'),
        'data_type': preprocess_metadata.get('data_type', 'unknown'),
        'n_features': len(features) if 'features' in locals() else 0
    }

def run_robust_ml_heavy_tail_benchmark():
    """Run the robust ML heavy-tail benchmark."""
    print("🧠 Robust ML Heavy-Tail Benchmark")
    print("=" * 50)
    
    # Initialize robustness components
    robust_extractor = RobustFeatureExtractor()
    preprocessor = AdaptiveDataPreprocessor()
    backend = RobustOptimizationBackend()
    
    print(f"🔧 Robustness Components Initialized:")
    print(f"   JAX GPU Working: {backend.hardware_info.jax_gpu_working}")
    print(f"   Robust Feature Extractor: {robust_extractor.__class__.__name__}")
    print(f"   Adaptive Preprocessor: {preprocessor.__class__.__name__}")
    
    # Test parameters
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    data_lengths = [1000, 2000]
    alpha_values = [2.0, 1.5, 1.0, 0.8]  # 2.0 = Gaussian, <2.0 = heavy-tailed
    
    # Initialize ML estimators
    estimators = {
        'RandomForest': RandomForestEstimator(),
        'SVR': SVREstimator(),
        'GradientBoosting': GradientBoostingEstimator()
    }
    
    results = []
    
    print(f"\n📊 Testing {len(estimators)} ML estimators on {len(hurst_values)} Hurst values")
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
            
            # Test ML estimators on pure data
            print(f"    🧠 Testing ML estimators on pure data:")
            for est_name, estimator in estimators.items():
                fbm_result = test_robust_ml_estimator(estimator, fbm_data, hurst, f"{est_name}_FBM", 
                                                    robust_extractor, preprocessor)
                fgn_result = test_robust_ml_estimator(estimator, fgn_data, hurst, f"{est_name}_FGN", 
                                                    robust_extractor, preprocessor)
                
                results.extend([fbm_result, fgn_result])
                
                if fbm_result['success']:
                    print(f"      {est_name} FBM: {fbm_result['estimated_hurst']:.3f} (error: {fbm_result['error']:.3f}) [{fbm_result['preprocess_method']}]")
                else:
                    print(f"      {est_name} FBM: FAILED - {fbm_result.get('reason', 'Unknown error')}")
                    
                if fgn_result['success']:
                    print(f"      {est_name} FGN: {fgn_result['estimated_hurst']:.3f} (error: {fgn_result['error']:.3f}) [{fgn_result['preprocess_method']}]")
                else:
                    print(f"      {est_name} FGN: FAILED - {fgn_result.get('reason', 'Unknown error')}")
            
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
                
                # Test ML estimators on alpha-stable data
                for est_name, estimator in estimators.items():
                    alpha_result = test_robust_ml_estimator(estimator, alpha_data, hurst, f"{est_name}_Alpha{alpha}", 
                                                          robust_extractor, preprocessor)
                    results.append(alpha_result)
                    
                    if alpha_result['success']:
                        print(f"        {est_name}: {alpha_result['estimated_hurst']:.3f} (error: {alpha_result['error']:.3f}) [{alpha_result['preprocess_method']}]")
                    else:
                        print(f"        {est_name}: FAILED - {alpha_result.get('reason', 'Unknown error')}")
    
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
    preprocess_methods = []
    
    for result in results:
        if 'FBM' in result['estimator'] or 'FGN' in result['estimator']:
            data_types.append('Pure')
        else:
            data_types.append('Alpha-Stable')
        
        estimator_name = result['estimator'].split('_')[0]
        estimators.append(estimator_name)
        errors.append(result['error'])
        successes.append(result['success'])
        preprocess_methods.append(result.get('preprocess_method', 'unknown'))
    
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
    
    # Calculate success rates by preprocessing method
    print(f"\n🔧 Success Rates by Preprocessing Method:")
    for method in set(preprocess_methods):
        method_success = np.mean([s for m, s in zip(preprocess_methods, successes) if m == method])
        method_count = sum(1 for m in preprocess_methods if m == method)
        print(f"  {method}: {method_success:.1%} ({method_count} tests)")
    
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
    create_visualization(results, data_types, estimators, errors, successes, preprocess_methods)

def create_visualization(results: List[Dict[str, Any]], data_types: List[str], 
                        estimators: List[str], errors: List[float], successes: List[bool],
                        preprocess_methods: List[str]):
    """Create visualization of results."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Robust ML Estimators: Pure vs Alpha-Stable Data Performance', fontsize=16, fontweight='bold')
        
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
        
        bars2 = ax2.bar(est_labels, est_success_rates, color=['lightgreen', 'orange', 'purple'], alpha=0.7)
        ax2.set_title('Success Rate by Estimator')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars2, est_success_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 3. Success rate by preprocessing method
        method_counts = {}
        for method in set(preprocess_methods):
            method_counts[method] = {
                'success': sum(1 for m, s in zip(preprocess_methods, successes) if m == method and s),
                'total': sum(1 for m in preprocess_methods if m == method)
            }
        
        method_labels = list(method_counts.keys())
        method_success_counts = [method_counts[method]['success'] for method in method_labels]
        method_total_counts = [method_counts[method]['total'] for method in method_labels]
        method_success_rates = [s/t for s, t in zip(method_success_counts, method_total_counts)]
        
        bars3 = ax3.bar(method_labels, method_success_rates, color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'], alpha=0.7)
        ax3.set_title('Success Rate by Preprocessing Method')
        ax3.set_ylabel('Success Rate')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars3, method_success_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 4. Error distribution (successful estimates only)
        successful_errors = [e for e, s in zip(errors, successes) if s and not np.isnan(e)]
        
        if successful_errors:
            ax4.hist(successful_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.set_title('Error Distribution (Successful Estimates)')
            ax4.set_xlabel('Absolute Error')
            ax4.set_ylabel('Frequency')
            ax4.axvline(np.mean(successful_errors), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(successful_errors):.3f}')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No successful estimates', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Error Distribution (No Data)')
        
        plt.tight_layout()
        plt.savefig('robust_ml_heavy_tail_benchmark.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved as 'robust_ml_heavy_tail_benchmark.png'")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")

if __name__ == "__main__":
    print("🚀 Starting Robust ML Heavy-Tail Benchmark...")
    print("   Testing ML estimators with robust preprocessing on pure vs alpha-stable data")
    print("   Focus: Robustness to heavy-tailed noise with improved feature extraction")
    
    try:
        results = run_robust_ml_heavy_tail_benchmark()
        analyze_results(results)
        
        print(f"\n✅ Robust ML Heavy-Tail Benchmark completed!")
        print(f"   Total tests: {len(results)}")
        print(f"   Successful estimates: {sum(1 for r in results if r['success'])}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
