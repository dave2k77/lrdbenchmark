#!/usr/bin/env python3
"""
Test Robustness Improvements

This script tests the robustness improvements for handling heavy-tailed data
and JAX GPU compatibility issues.
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

# Import robustness modules
from lrdbenchmark.robustness.robust_optimization_backend import RobustOptimizationBackend
from lrdbenchmark.robustness.robust_feature_extractor import RobustFeatureExtractor

# Import data models
from lrdbenchmark.models.data_models.alpha_stable.alpha_stable_model import AlphaStableModel

def generate_test_data():
    """Generate various types of test data."""
    np.random.seed(42)
    
    data_types = {}
    
    # 1. Normal data
    data_types['normal'] = np.random.normal(0, 1, 1000)
    
    # 2. Heavy-tailed data (alpha-stable)
    alpha_model = AlphaStableModel(alpha=1.0, beta=0.0, sigma=1.0, mu=0.0, use_optimization='numpy')
    data_types['heavy_tailed'] = alpha_model.generate(1000, seed=42)
    
    # 3. Data with extreme outliers
    normal_data = np.random.normal(0, 1, 1000)
    extreme_outliers = np.array([50, -50, 100, -100, 200])
    data_types['extreme_outliers'] = np.concatenate([normal_data, extreme_outliers])
    
    # 4. Data with NaN values
    normal_data = np.random.normal(0, 1, 1000)
    normal_data[100:105] = np.nan  # Add some NaN values
    data_types['with_nan'] = normal_data
    
    # 5. Very short data
    data_types['short'] = np.random.normal(0, 1, 5)
    
    return data_types

def test_robust_optimization_backend():
    """Test the robust optimization backend."""
    print("🔧 Testing Robust Optimization Backend")
    print("=" * 50)
    
    backend = RobustOptimizationBackend()
    
    # Test hardware detection
    print(f"Hardware Info:")
    print(f"  GPU Available: {backend.hardware_info.has_gpu}")
    print(f"  JAX GPU Working: {backend.hardware_info.jax_gpu_working}")
    print(f"  CPU Cores: {backend.hardware_info.cpu_cores}")
    print(f"  Memory: {backend.hardware_info.memory_gb:.1f} GB")
    
    # Test framework selection
    test_cases = [
        (100, "small_computation"),
        (1000, "medium_computation"),
        (10000, "large_computation"),
        (100000, "very_large_computation")
    ]
    
    print(f"\nFramework Selection Tests:")
    for data_size, comp_type in test_cases:
        framework = backend.select_optimal_framework(data_size, comp_type)
        recommendation = backend.get_framework_recommendation(data_size, comp_type)
        
        print(f"  Data size {data_size:6d} ({comp_type:20s}): {framework.value:8s} - {recommendation['reasoning']}")
    
    return backend

def test_robust_feature_extractor():
    """Test the robust feature extractor."""
    print(f"\n🧠 Testing Robust Feature Extractor")
    print("=" * 50)
    
    extractor = RobustFeatureExtractor()
    data_types = generate_test_data()
    
    results = {}
    
    for data_name, data in data_types.items():
        print(f"\nTesting {data_name} data:")
        print(f"  Data shape: {data.shape}")
        print(f"  Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        print(f"  Has NaN: {np.any(np.isnan(data))}")
        print(f"  Has Inf: {np.any(np.isinf(data))}")
        
        try:
            # Extract features
            features = extractor.extract_features(data)
            print(f"  Features extracted: {len(features)}")
            print(f"  Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
            print(f"  Has NaN features: {np.any(np.isnan(features))}")
            
            results[data_name] = {
                'success': True,
                'n_features': len(features),
                'has_nan': np.any(np.isnan(features)),
                'feature_range': (np.min(features), np.max(features))
            }
            
        except Exception as e:
            print(f"  ❌ Feature extraction failed: {e}")
            results[data_name] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def test_heavy_tail_robustness():
    """Test robustness specifically on heavy-tailed data."""
    print(f"\n🔥 Testing Heavy-Tail Robustness")
    print("=" * 50)
    
    extractor = RobustFeatureExtractor()
    
    # Test different alpha values
    alpha_values = [2.0, 1.5, 1.0, 0.8, 0.5]
    results = {}
    
    for alpha in alpha_values:
        print(f"\nTesting α = {alpha}:")
        
        # Generate alpha-stable data
        alpha_model = AlphaStableModel(
            alpha=alpha,
            beta=0.0,
            sigma=1.0,
            mu=0.0,
            use_optimization='numpy'
        )
        
        data = alpha_model.generate(1000, seed=42)
        
        # Analyze data characteristics
        print(f"  Data characteristics:")
        print(f"    Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        print(f"    Extreme values (|x| > 5): {np.sum(np.abs(data) > 5)}")
        
        try:
            # Calculate kurtosis (might be infinite)
            kurtosis = np.mean(((data - np.mean(data)) / np.std(data))**4) - 3
            print(f"    Kurtosis: {kurtosis:.3f}")
        except:
            print(f"    Kurtosis: Infinite/NaN")
        
        # Extract features
        try:
            features = extractor.extract_features(data)
            print(f"  ✅ Features extracted: {len(features)}")
            print(f"    Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
            print(f"    Has NaN features: {np.any(np.isnan(features))}")
            
            results[alpha] = {
                'success': True,
                'n_features': len(features),
                'has_nan': np.any(np.isnan(features)),
                'extreme_values': np.sum(np.abs(data) > 5)
            }
            
        except Exception as e:
            print(f"  ❌ Feature extraction failed: {e}")
            results[alpha] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def create_robustness_visualization(backend_results, feature_results, heavy_tail_results):
    """Create visualization of robustness test results."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Robustness Improvements Test Results', fontsize=16, fontweight='bold')
        
        # 1. Feature extraction success rates
        data_types = list(feature_results.keys())
        success_rates = [1 if feature_results[dt]['success'] else 0 for dt in data_types]
        
        bars1 = ax1.bar(data_types, success_rates, color=['green' if s else 'red' for s in success_rates])
        ax1.set_title('Feature Extraction Success Rate by Data Type')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.0%}', ha='center', va='bottom')
        
        # 2. Heavy-tail robustness
        alpha_vals = list(heavy_tail_results.keys())
        heavy_tail_success = [1 if heavy_tail_results[alpha]['success'] else 0 for alpha in alpha_vals]
        
        bars2 = ax2.bar([f'α={alpha}' for alpha in alpha_vals], heavy_tail_success, 
                       color=['green' if s else 'red' for s in heavy_tail_success])
        ax2.set_title('Heavy-Tail Robustness by Alpha Value')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, rate in zip(bars2, heavy_tail_success):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.0%}', ha='center', va='bottom')
        
        # 3. Feature count by data type
        feature_counts = [feature_results[dt]['n_features'] if feature_results[dt]['success'] else 0 
                         for dt in data_types]
        
        bars3 = ax3.bar(data_types, feature_counts, color='skyblue', alpha=0.7)
        ax3.set_title('Number of Features Extracted by Data Type')
        ax3.set_ylabel('Feature Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars3, feature_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        # 4. Extreme values vs success
        extreme_values = [heavy_tail_results[alpha]['extreme_values'] if heavy_tail_results[alpha]['success'] else 0
                         for alpha in alpha_vals]
        success_colors = ['green' if heavy_tail_results[alpha]['success'] else 'red' 
                         for alpha in alpha_vals]
        
        bars4 = ax4.bar([f'α={alpha}' for alpha in alpha_vals], extreme_values, 
                       color=success_colors, alpha=0.7)
        ax4.set_title('Extreme Values vs Success Rate')
        ax4.set_ylabel('Number of Extreme Values')
        
        # Add value labels
        for bar, count in zip(bars4, extreme_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('robustness_improvements_test.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved as 'robustness_improvements_test.png'")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")

def main():
    """Run all robustness tests."""
    print("🚀 Testing Robustness Improvements")
    print("=" * 60)
    print("Testing improved robustness for heavy-tailed data handling")
    print("and JAX GPU compatibility issues")
    
    try:
        # Test robust optimization backend
        backend_results = test_robust_optimization_backend()
        
        # Test robust feature extractor
        feature_results = test_robust_feature_extractor()
        
        # Test heavy-tail robustness
        heavy_tail_results = test_heavy_tail_robustness()
        
        # Create visualization
        create_robustness_visualization(backend_results, feature_results, heavy_tail_results)
        
        # Summary
        print(f"\n📈 Test Summary")
        print("=" * 50)
        
        # Feature extraction summary
        feature_success = sum(1 for r in feature_results.values() if r['success'])
        print(f"Feature Extraction: {feature_success}/{len(feature_results)} data types successful")
        
        # Heavy-tail summary
        heavy_tail_success = sum(1 for r in heavy_tail_results.values() if r['success'])
        print(f"Heavy-Tail Robustness: {heavy_tail_success}/{len(heavy_tail_results)} alpha values successful")
        
        # Overall success rate
        total_tests = len(feature_results) + len(heavy_tail_results)
        total_success = feature_success + heavy_tail_success
        print(f"Overall Success Rate: {total_success}/{total_tests} ({total_success/total_tests:.1%})")
        
        print(f"\n✅ Robustness improvement tests completed!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
