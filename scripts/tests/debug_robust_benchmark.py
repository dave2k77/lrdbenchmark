#!/usr/bin/env python3
"""
Debug Robust Benchmark

This script tests the robust benchmark approach to see why it's failing.
"""

import numpy as np
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import robustness modules
from lrdbenchmark.robustness.robust_feature_extractor import RobustFeatureExtractor
from lrdbenchmark.robustness.adaptive_preprocessor import AdaptiveDataPreprocessor

# Import ML estimators
from lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified import RandomForestEstimator

def test_robust_preprocessing():
    """Test robust preprocessing on pure data."""
    print("🔍 Testing Robust Preprocessing on Pure Data")
    print("=" * 50)
    
    # Generate pure data
    data = np.random.normal(0, 1, 1000)
    
    # Initialize robust components
    robust_extractor = RobustFeatureExtractor()
    preprocessor = AdaptiveDataPreprocessor()
    
    print(f"Original data shape: {data.shape}")
    print(f"Original data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    
    # Test preprocessing
    try:
        data_processed, preprocess_metadata = preprocessor.preprocess(data)
        print(f"✅ Preprocessing successful!")
        print(f"   Processed data shape: {data_processed.shape}")
        print(f"   Processed data range: [{np.min(data_processed):.3f}, {np.max(data_processed):.3f}]")
        print(f"   Preprocessing method: {preprocess_metadata.get('method', 'unknown')}")
        print(f"   Data type: {preprocess_metadata.get('data_type', 'unknown')}")
        
        # Test feature extraction
        features = robust_extractor.extract_features(data_processed)
        print(f"   Features extracted: {len(features)}")
        print(f"   Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
        print(f"   Has NaN features: {np.any(np.isnan(features))}")
        
        return True, data_processed, features
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_ml_estimator_with_robust_preprocessing():
    """Test ML estimator with robust preprocessing."""
    print(f"\n🔍 Testing ML Estimator with Robust Preprocessing")
    print("=" * 50)
    
    # Generate pure data
    data = np.random.normal(0, 1, 1000)
    
    # Initialize robust components
    robust_extractor = RobustFeatureExtractor()
    preprocessor = AdaptiveDataPreprocessor()
    
    # Initialize ML estimator
    estimator = RandomForestEstimator(use_optimization='numpy')
    
    try:
        # Preprocess data
        data_processed, preprocess_metadata = preprocessor.preprocess(data)
        print(f"Data preprocessed: {preprocess_metadata.get('method', 'unknown')}")
        
        # Extract robust features
        features = robust_extractor.extract_features(data_processed)
        print(f"Features extracted: {len(features)}")
        
        # Test estimation on preprocessed data
        result = estimator.estimate(data_processed)
        
        print(f"✅ ML Estimation with robust preprocessing successful!")
        print(f"   Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Estimation with robust preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_estimator_without_robust_preprocessing():
    """Test ML estimator without robust preprocessing (original approach)."""
    print(f"\n🔍 Testing ML Estimator without Robust Preprocessing")
    print("=" * 50)
    
    # Generate pure data
    data = np.random.normal(0, 1, 1000)
    
    # Initialize ML estimator
    estimator = RandomForestEstimator(use_optimization='numpy')
    
    try:
        # Test estimation on original data
        result = estimator.estimate(data)
        
        print(f"✅ ML Estimation without robust preprocessing successful!")
        print(f"   Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Estimation without robust preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_robust_benchmark_approach():
    """Test the exact approach used in our robust benchmark."""
    print(f"\n🔍 Testing Robust Benchmark Approach")
    print("=" * 50)
    
    # Generate pure data
    data = np.random.normal(0, 1, 1000)
    
    # Initialize robust components
    robust_extractor = RobustFeatureExtractor()
    preprocessor = AdaptiveDataPreprocessor()
    
    # Initialize ML estimator
    estimator = RandomForestEstimator(use_optimization='numpy')
    
    try:
        # This is the exact approach from our robust benchmark
        estimator.use_optimization = 'numpy'
        
        # Preprocess data
        data_processed, preprocess_metadata = preprocessor.preprocess(data)
        
        # Extract robust features
        features = robust_extractor.extract_features(data_processed)
        
        # Check if features are valid
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print("❌ Features contain NaN/Inf values")
            return False
        
        # Estimate Hurst using robust features
        # Note: This is where the issue might be - we're not using the features!
        result = estimator.estimate(data_processed)
        
        if result is not None and 'hurst_parameter' in result:
            estimated_hurst = result['hurst_parameter']
            success = True
        else:
            estimated_hurst = np.nan
            success = False
        
        print(f"✅ Robust benchmark approach successful!")
        print(f"   Estimated Hurst: {estimated_hurst}")
        print(f"   Success: {success}")
        print(f"   Result keys: {list(result.keys()) if result else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust benchmark approach failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Debugging Robust Benchmark Approach")
    print("=" * 60)
    print("Testing why robust benchmark fails on pure data")
    
    # Test robust preprocessing
    preprocess_success, data_processed, features = test_robust_preprocessing()
    
    # Test ML estimator with robust preprocessing
    ml_robust_success = test_ml_estimator_with_robust_preprocessing()
    
    # Test ML estimator without robust preprocessing
    ml_original_success = test_ml_estimator_without_robust_preprocessing()
    
    # Test robust benchmark approach
    robust_benchmark_success = test_robust_benchmark_approach()
    
    print(f"\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Robust Preprocessing: {'✅' if preprocess_success else '❌'}")
    print(f"ML + Robust Preprocessing: {'✅' if ml_robust_success else '❌'}")
    print(f"ML Original Approach: {'✅' if ml_original_success else '❌'}")
    print(f"Robust Benchmark Approach: {'✅' if robust_benchmark_success else '❌'}")
    
    if preprocess_success and ml_original_success and not robust_benchmark_success:
        print(f"\n🔍 Root Cause: Issue is in the robust benchmark approach itself")
    elif not preprocess_success:
        print(f"\n🔍 Root Cause: Robust preprocessing is failing")
    elif not ml_original_success:
        print(f"\n🔍 Root Cause: ML estimator is failing even without robust preprocessing")
    else:
        print(f"\n🔍 All approaches working - issue must be in the benchmark script logic")

if __name__ == "__main__":
    main()
