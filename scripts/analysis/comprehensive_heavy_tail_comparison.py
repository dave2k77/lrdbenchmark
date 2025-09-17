#!/usr/bin/env python3
"""
Comprehensive Heavy-Tail Comparison

This script compares Classical, ML, and Neural Network estimators on heavy-tail data
to understand their relative performance and robustness.
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

# Import Classical estimators
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.temporal.higuchi.higuchi_estimator_unified import HiguchiEstimator
from lrdbenchmark.analysis.temporal.dma.dma_estimator_unified import DMAEstimator

# Import ML estimators
from lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified import RandomForestEstimator
from lrdbenchmark.analysis.machine_learning.svr_estimator_unified import SVREstimator
from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator_unified import GradientBoostingEstimator

# Import Neural Network estimators
from lrdbenchmark.analysis.machine_learning.cnn_estimator_unified import CNNEstimator
from lrdbenchmark.analysis.machine_learning.lstm_estimator_unified import LSTMEstimator
from lrdbenchmark.analysis.machine_learning.gru_estimator_unified import GRUEstimator
from lrdbenchmark.analysis.machine_learning.transformer_estimator_unified import TransformerEstimator

# Import robustness modules
from lrdbenchmark.robustness.adaptive_preprocessor import AdaptiveDataPreprocessor

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

def test_estimator(estimator, data: np.ndarray, true_hurst: float, name: str, 
                  preprocessor: AdaptiveDataPreprocessor = None) -> Dict[str, Any]:
    """Test an estimator on data."""
    try:
        # Set optimization to numpy to avoid JAX issues
        estimator.use_optimization = 'numpy'
        
        # Preprocess data if preprocessor provided
        if preprocessor is not None:
            data_processed, preprocess_metadata = preprocessor.preprocess(data)
        else:
            data_processed = data
            preprocess_metadata = {'method': 'none', 'data_type': 'raw'}
        
        # Estimate Hurst
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
        'category': get_estimator_category(name),
        'true_hurst': true_hurst,
        'estimated_hurst': estimated_hurst,
        'error': error,
        'success': success,
        'preprocess_method': preprocess_metadata.get('method', 'none'),
        'data_type': preprocess_metadata.get('data_type', 'raw')
    }

def get_estimator_category(name: str) -> str:
    """Get the category of an estimator."""
    if name in ['RSEstimator', 'DFAEstimator', 'HiguchiEstimator', 'DMAEstimator']:
        return 'Classical'
    elif name in ['RandomForest', 'SVR', 'GradientBoosting']:
        return 'ML'
    elif name in ['CNN', 'LSTM', 'GRU', 'Transformer']:
        return 'Neural Network'
    else:
        return 'Unknown'

def run_comprehensive_heavy_tail_comparison():
    """Run comprehensive comparison of all estimator types on heavy-tail data."""
    print("🧠 Comprehensive Heavy-Tail Comparison")
    print("=" * 60)
    print("Comparing Classical, ML, and Neural Network estimators on heavy-tail data")
    
    # Initialize robustness components
    preprocessor = AdaptiveDataPreprocessor()
    
    # Test parameters
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    data_lengths = [1000, 2000]
    alpha_values = [2.0, 1.5, 1.0, 0.8]  # 2.0 = Gaussian, <2.0 = heavy-tailed
    
    # Initialize all estimators
    classical_estimators = {
        'RSEstimator': RSEstimator(use_optimization='numpy'),
        'DFAEstimator': DFAEstimator(use_optimization='numpy'),
        'HiguchiEstimator': HiguchiEstimator(use_optimization='numpy'),
        'DMAEstimator': DMAEstimator(use_optimization='numpy')
    }
    
    ml_estimators = {
        'RandomForest': RandomForestEstimator(use_optimization='numpy'),
        'SVR': SVREstimator(use_optimization='numpy'),
        'GradientBoosting': GradientBoostingEstimator(use_optimization='numpy')
    }
    
    nn_estimators = {
        'CNN': CNNEstimator(use_optimization='numpy'),
        'LSTM': LSTMEstimator(use_optimization='numpy'),
        'GRU': GRUEstimator(use_optimization='numpy'),
        'Transformer': TransformerEstimator(use_optimization='numpy')
    }
    
    all_estimators = {**classical_estimators, **ml_estimators, **nn_estimators}
    
    results = []
    
    print(f"\n📊 Testing {len(all_estimators)} estimators on {len(hurst_values)} Hurst values")
    print(f"   Data lengths: {data_lengths}")
    print(f"   Alpha values: {alpha_values}")
    print(f"   Classical: {len(classical_estimators)}, ML: {len(ml_estimators)}, NN: {len(nn_estimators)}")
    
    for hurst in hurst_values:
        print(f"\n🎯 Testing Hurst = {hurst}")
        print("-" * 40)
        
        for length in data_lengths:
            print(f"\n  📏 Data length = {length}")
            
            # Test on pure data (FBM)
            fbm_data = generate_simple_fbm(hurst, length, seed=42)
            fbm_stats = analyze_data_characteristics(fbm_data, f"FBM (H={hurst})")
            
            print(f"    Pure FBM: kurtosis={fbm_stats['kurtosis']:.3f}, extreme_values={fbm_stats['extreme_values']}")
            
            # Test all estimators on pure data
            print(f"    🧠 Testing all estimators on pure data:")
            for est_name, estimator in all_estimators.items():
                result = test_estimator(estimator, fbm_data, hurst, est_name, preprocessor)
                results.append(result)
                
                if result['success']:
                    print(f"      {est_name}: {result['estimated_hurst']:.3f} (error: {result['error']:.3f}) [{result['preprocess_method']}]")
                else:
                    print(f"      {est_name}: FAILED")
            
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
                
                # Test all estimators on alpha-stable data
                for est_name, estimator in all_estimators.items():
                    result = test_estimator(estimator, alpha_data, hurst, est_name, preprocessor)
                    results.append(result)
                    
                    if result['success']:
                        print(f"        {est_name}: {result['estimated_hurst']:.3f} (error: {result['error']:.3f}) [{result['preprocess_method']}]")
                    else:
                        print(f"        {est_name}: FAILED")
    
    return results

def analyze_comprehensive_results(results: List[Dict[str, Any]]):
    """Analyze comprehensive results."""
    print(f"\n📈 Comprehensive Results Analysis")
    print("=" * 60)
    
    # Convert to DataFrame-like structure for analysis
    categories = []
    estimators = []
    errors = []
    successes = []
    data_types = []
    alpha_values = []
    
    for result in results:
        categories.append(result['category'])
        estimators.append(result['estimator'])
        errors.append(result['error'])
        successes.append(result['success'])
        
        if 'FBM' in result['estimator'] or 'Alpha' in result['estimator']:
            if 'Alpha' in result['estimator']:
                # Extract alpha value from estimator name
                alpha_val = 2.0  # Default for pure data
                data_types.append('Alpha-Stable')
            else:
                alpha_val = 2.0
                data_types.append('Pure')
        else:
            alpha_val = 2.0
            data_types.append('Pure')
        
        alpha_values.append(alpha_val)
    
    # Calculate success rates by category
    print(f"\n🎯 Success Rates by Category:")
    for category in set(categories):
        cat_success = np.mean([s for c, s in zip(categories, successes) if c == category])
        cat_count = sum(1 for c in categories if c == category)
        print(f"  {category}: {cat_success:.1%} ({cat_count} tests)")
    
    # Calculate success rates by data type
    print(f"\n📊 Success Rates by Data Type:")
    for dt in set(data_types):
        dt_success = np.mean([s for d, s in zip(data_types, successes) if d == dt])
        dt_count = sum(1 for d in data_types if d == dt)
        print(f"  {dt}: {dt_success:.1%} ({dt_count} tests)")
    
    # Calculate error statistics by category
    print(f"\n📊 Error Statistics by Category:")
    for category in set(categories):
        cat_errors = [e for c, e, s in zip(categories, errors, successes) if c == category and s and not np.isnan(e)]
        if cat_errors:
            print(f"  {category}:")
            print(f"    Mean Error: {np.mean(cat_errors):.3f}")
            print(f"    Median Error: {np.median(cat_errors):.3f}")
            print(f"    Std Error: {np.std(cat_errors):.3f}")
            print(f"    Min Error: {np.min(cat_errors):.3f}")
            print(f"    Max Error: {np.max(cat_errors):.3f}")
        else:
            print(f"  {category}: No successful estimates")
    
    # Calculate error statistics by data type
    print(f"\n📊 Error Statistics by Data Type:")
    for dt in set(data_types):
        dt_errors = [e for d, e, s in zip(data_types, errors, successes) if d == dt and s and not np.isnan(e)]
        if dt_errors:
            print(f"  {dt}:")
            print(f"    Mean Error: {np.mean(dt_errors):.3f}")
            print(f"    Median Error: {np.median(dt_errors):.3f}")
            print(f"    Std Error: {np.std(dt_errors):.3f}")
        else:
            print(f"  {dt}: No successful estimates")
    
    # Create comprehensive visualization
    create_comprehensive_visualization(results, categories, estimators, errors, successes, data_types)

def create_comprehensive_visualization(results: List[Dict[str, Any]], categories: List[str], 
                                     estimators: List[str], errors: List[float], successes: List[bool],
                                     data_types: List[str]):
    """Create comprehensive visualization of results."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Heavy-Tail Comparison: Classical vs ML vs Neural Network', fontsize=16, fontweight='bold')
        
        # 1. Success rate by category
        category_counts = {}
        for cat in set(categories):
            category_counts[cat] = {
                'success': sum(1 for c, s in zip(categories, successes) if c == cat and s),
                'total': sum(1 for c in categories if c == cat)
            }
        
        cat_labels = list(category_counts.keys())
        cat_success_counts = [category_counts[cat]['success'] for cat in cat_labels]
        cat_total_counts = [category_counts[cat]['total'] for cat in cat_labels]
        cat_success_rates = [s/t for s, t in zip(cat_success_counts, cat_total_counts)]
        
        bars1 = ax1.bar(cat_labels, cat_success_rates, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
        ax1.set_title('Success Rate by Estimator Category')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, cat_success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 2. Error distribution by category
        classical_errors = [e for c, e, s in zip(categories, errors, successes) if c == 'Classical' and s and not np.isnan(e)]
        ml_errors = [e for c, e, s in zip(categories, errors, successes) if c == 'ML' and s and not np.isnan(e)]
        nn_errors = [e for c, e, s in zip(categories, errors, successes) if c == 'Neural Network' and s and not np.isnan(e)]
        
        ax2.hist([classical_errors, ml_errors, nn_errors], bins=20, alpha=0.7, 
                label=['Classical', 'ML', 'Neural Network'], color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('Error Distribution by Category')
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Success rate by data type
        data_type_counts = {}
        for dt in set(data_types):
            data_type_counts[dt] = {
                'success': sum(1 for d, s in zip(data_types, successes) if d == dt and s),
                'total': sum(1 for d in data_types if d == dt)
            }
        
        dt_labels = list(data_type_counts.keys())
        dt_success_counts = [data_type_counts[dt]['success'] for dt in dt_labels]
        dt_total_counts = [data_type_counts[dt]['total'] for dt in dt_labels]
        dt_success_rates = [s/t for s, t in zip(dt_success_counts, dt_total_counts)]
        
        bars3 = ax3.bar(dt_labels, dt_success_rates, color=['lightblue', 'orange'], alpha=0.7)
        ax3.set_title('Success Rate by Data Type')
        ax3.set_ylabel('Success Rate')
        ax3.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars3, dt_success_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 4. Top performers by category
        category_errors = {}
        for cat in set(categories):
            cat_errors = [e for c, e, s in zip(categories, errors, successes) if c == cat and s and not np.isnan(e)]
            if cat_errors:
                category_errors[cat] = np.mean(cat_errors)
            else:
                category_errors[cat] = np.nan
        
        cat_names = list(category_errors.keys())
        cat_mean_errors = [category_errors[cat] for cat in cat_names]
        
        bars4 = ax4.bar(cat_names, cat_mean_errors, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
        ax4.set_title('Mean Error by Category')
        ax4.set_ylabel('Mean Absolute Error')
        
        # Add value labels on bars
        for bar, error in zip(bars4, cat_mean_errors):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{error:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('comprehensive_heavy_tail_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved as 'comprehensive_heavy_tail_comparison.png'")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Heavy-Tail Comparison...")
    print("   Comparing Classical, ML, and Neural Network estimators on heavy-tail data")
    print("   Focus: Understanding relative performance and robustness")
    
    try:
        results = run_comprehensive_heavy_tail_comparison()
        analyze_comprehensive_results(results)
        
        print(f"\n✅ Comprehensive Heavy-Tail Comparison completed!")
        print(f"   Total tests: {len(results)}")
        print(f"   Successful estimates: {sum(1 for r in results if r['success'])}")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
