#!/usr/bin/env python3
"""
Test script for LRDBenchmark package structure
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all package imports"""
    print("Testing LRDBenchmark package imports...")
    
    try:
        import lrdbenchmark
        print("✓ Main package import: SUCCESS")
    except Exception as e:
        print(f"✗ Main package import: FAILED - {e}")
        return False
    
    # Test data models
    try:
        from lrdbenchmark.models.data_models import FBMModel, FGNModel, ARFIMAModel, MRWModel
        print("✓ Data models import: SUCCESS")
        
        # Test data generation
        fbm = FBMModel(H=0.6)
        data = fbm.generate(n=100)
        print(f"✓ FBM data generation: SUCCESS (generated {len(data)} points)")
        
    except Exception as e:
        print(f"✗ Data models import: FAILED - {e}")
        return False
    
    # Test classical estimators
    try:
        from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
        from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
        print("✓ Classical estimators import: SUCCESS")
    except Exception as e:
        print(f"✗ Classical estimators import: FAILED - {e}")
        return False
    
    # Test ML estimators
    try:
        from lrdbenchmark.analysis.machine_learning import RandomForestEstimator, SVREstimator
        print("✓ ML estimators import: SUCCESS")
    except Exception as e:
        print(f"✗ ML estimators import: FAILED - {e}")
        return False
    
    # Test neural network factory
    try:
        from lrdbenchmark.analysis.machine_learning.neural_network_factory import NeuralNetworkFactory
        print("✓ Neural network factory import: SUCCESS")
    except Exception as e:
        print(f"✗ Neural network factory import: FAILED - {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All imports successful! Package structure is working.")
    else:
        print("\n⚠️  Some imports failed. Check the errors above.")
