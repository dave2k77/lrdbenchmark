#!/usr/bin/env python3
"""
Debug Estimator Chain

This script tests the estimator chain to understand why they're failing on pure data.
"""

import numpy as np
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_rs_estimator_directly():
    """Test R/S estimator directly to see if it works."""
    print("🔍 Testing R/S Estimator Directly")
    print("=" * 40)
    
    try:
        from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
        
        # Create estimator with numpy optimization
        rs_estimator = RSEstimator(use_optimization='numpy')
        
        # Generate simple test data
        data = np.random.normal(0, 1, 1000)
        
        print(f"Data shape: {data.shape}")
        print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        
        # Test estimation
        result = rs_estimator.estimate(data)
        
        print(f"✅ R/S Estimation successful!")
        print(f"   Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ R/S Estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_estimator_chain():
    """Test ML estimator chain to see where it fails."""
    print(f"\n🔍 Testing ML Estimator Chain")
    print("=" * 40)
    
    try:
        from lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified import RandomForestEstimator
        
        # Create estimator
        rf_estimator = RandomForestEstimator(use_optimization='numpy')
        
        # Generate simple test data
        data = np.random.normal(0, 1, 1000)
        
        print(f"Data shape: {data.shape}")
        print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        
        # Test estimation
        result = rf_estimator.estimate(data)
        
        print(f"✅ ML Estimation successful!")
        print(f"   Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classical_estimator_directly():
    """Test classical estimator directly."""
    print(f"\n🔍 Testing Classical Estimator Directly")
    print("=" * 40)
    
    try:
        from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
        
        # Create estimator with numpy optimization
        rs_estimator = RSEstimator(use_optimization='numpy')
        
        # Generate simple test data
        data = np.random.normal(0, 1, 1000)
        
        print(f"Data shape: {data.shape}")
        print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        
        # Test estimation
        result = rs_estimator.estimate(data)
        
        print(f"✅ Classical Estimation successful!")
        print(f"   Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Classical Estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Debugging Estimator Chain")
    print("=" * 50)
    print("Testing why estimators fail on pure data")
    
    # Test R/S estimator directly
    rs_success = test_rs_estimator_directly()
    
    # Test ML estimator chain
    ml_success = test_ml_estimator_chain()
    
    # Test classical estimator
    classical_success = test_classical_estimator_directly()
    
    print(f"\n📊 Test Results Summary")
    print("=" * 30)
    print(f"R/S Estimator Direct: {'✅' if rs_success else '❌'}")
    print(f"ML Estimator Chain: {'✅' if ml_success else '❌'}")
    print(f"Classical Estimator: {'✅' if classical_success else '❌'}")
    
    if not rs_success:
        print(f"\n🔍 Root Cause: R/S estimator is failing, which breaks the ML fallback chain")
    elif not ml_success:
        print(f"\n🔍 Root Cause: ML estimator chain is failing despite R/S working")
    else:
        print(f"\n🔍 All estimators working - issue must be elsewhere")

if __name__ == "__main__":
    main()
