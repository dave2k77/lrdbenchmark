#!/usr/bin/env python3
"""
Debug Preprocessed Data

This script tests what happens when we pass preprocessed data to ML estimators.
"""

import numpy as np
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import robustness modules
from lrdbenchmark.robustness.adaptive_preprocessor import AdaptiveDataPreprocessor

# Import ML estimators
from lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified import RandomForestEstimator

def test_original_data():
    """Test ML estimator on original data."""
    print("🔍 Testing ML Estimator on Original Data")
    print("=" * 50)
    
    # Generate original data
    data = np.random.normal(0, 1, 1000)
    
    # Initialize ML estimator
    estimator = RandomForestEstimator(use_optimization='numpy')
    
    try:
        result = estimator.estimate(data)
        print(f"✅ Original data estimation successful!")
        print(f"   Hurst: {result.get('hurst_parameter', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Original data estimation failed: {e}")
        return False

def test_standardized_data():
    """Test ML estimator on standardized data."""
    print(f"\n🔍 Testing ML Estimator on Standardized Data")
    print("=" * 50)
    
    # Generate original data
    data = np.random.normal(0, 1, 1000)
    
    # Standardize data (like our preprocessor does)
    data_standardized = (data - np.mean(data)) / np.std(data)
    
    print(f"Original data stats: mean={np.mean(data):.3f}, std={np.std(data):.3f}")
    print(f"Standardized data stats: mean={np.mean(data_standardized):.3f}, std={np.std(data_standardized):.3f}")
    
    # Initialize ML estimator
    estimator = RandomForestEstimator(use_optimization='numpy')
    
    try:
        result = estimator.estimate(data_standardized)
        print(f"✅ Standardized data estimation successful!")
        print(f"   Hurst: {result.get('hurst_parameter', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Standardized data estimation failed: {e}")
        return False

def test_winsorized_data():
    """Test ML estimator on winsorized data."""
    print(f"\n🔍 Testing ML Estimator on Winsorized Data")
    print("=" * 50)
    
    # Generate data with extreme values
    data = np.random.normal(0, 1, 1000)
    data[0:5] = [10, -10, 15, -15, 20]  # Add extreme values
    
    # Winsorize data (like our preprocessor does for heavy-tailed data)
    q1, q99 = np.percentile(data, [1, 99])
    data_winsorized = np.clip(data, q1, q99)
    
    print(f"Original data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    print(f"Winsorized data range: [{np.min(data_winsorized):.3f}, {np.max(data_winsorized):.3f}]")
    print(f"Winsorized {np.sum((data < q1) | (data > q99))} extreme values")
    
    # Initialize ML estimator
    estimator = RandomForestEstimator(use_optimization='numpy')
    
    try:
        result = estimator.estimate(data_winsorized)
        print(f"✅ Winsorized data estimation successful!")
        print(f"   Hurst: {result.get('hurst_parameter', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Winsorized data estimation failed: {e}")
        return False

def test_robust_preprocessor_data():
    """Test ML estimator on data processed by our robust preprocessor."""
    print(f"\n🔍 Testing ML Estimator on Robust Preprocessor Data")
    print("=" * 50)
    
    # Generate original data
    data = np.random.normal(0, 1, 1000)
    
    # Use our robust preprocessor
    preprocessor = AdaptiveDataPreprocessor()
    data_processed, preprocess_metadata = preprocessor.preprocess(data)
    
    print(f"Original data stats: mean={np.mean(data):.3f}, std={np.std(data):.3f}")
    print(f"Processed data stats: mean={np.mean(data_processed):.3f}, std={np.std(data_processed):.3f}")
    print(f"Preprocessing method: {preprocess_metadata.get('method', 'unknown')}")
    print(f"Data type: {preprocess_metadata.get('data_type', 'unknown')}")
    
    # Initialize ML estimator
    estimator = RandomForestEstimator(use_optimization='numpy')
    
    try:
        result = estimator.estimate(data_processed)
        print(f"✅ Robust preprocessor data estimation successful!")
        print(f"   Hurst: {result.get('hurst_parameter', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Robust preprocessor data estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Debugging Preprocessed Data Issues")
    print("=" * 60)
    print("Testing how different data preprocessing affects ML estimators")
    
    # Test original data
    original_success = test_original_data()
    
    # Test standardized data
    standardized_success = test_standardized_data()
    
    # Test winsorized data
    winsorized_success = test_winsorized_data()
    
    # Test robust preprocessor data
    robust_success = test_robust_preprocessor_data()
    
    print(f"\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Original Data: {'✅' if original_success else '❌'}")
    print(f"Standardized Data: {'✅' if standardized_success else '❌'}")
    print(f"Winsorized Data: {'✅' if winsorized_success else '❌'}")
    print(f"Robust Preprocessor Data: {'✅' if robust_success else '❌'}")
    
    if original_success and not robust_success:
        print(f"\n🔍 Root Cause: Robust preprocessor is changing data in a way that breaks ML estimators")
    elif not original_success:
        print(f"\n🔍 Root Cause: ML estimators are failing even on original data")
    else:
        print(f"\n🔍 All data types working - issue must be elsewhere")

if __name__ == "__main__":
    main()
