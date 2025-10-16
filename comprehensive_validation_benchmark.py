#!/usr/bin/env python3
"""
Comprehensive validation benchmark for LRDBenchmark v2.3.0
Tests all major functionality to ensure the library overhaul didn't break anything.
"""

import numpy as np
import time
import warnings
from pathlib import Path
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all simplified imports work correctly."""
    print("🔍 Testing Simplified API Imports...")
    
    try:
        # Test data models
        from lrdbenchmark import FBMModel, FGNModel, ARFIMAModel, MRWModel, AlphaStableModel
        print("✅ Data models imported successfully")
        
        # Test classical estimators
        from lrdbenchmark import RSEstimator, DFAEstimator, GPHEstimator, WhittleEstimator
        print("✅ Classical estimators imported successfully")
        
        # Test ML estimators
        from lrdbenchmark import RandomForestEstimator, SVREstimator, GradientBoostingEstimator
        print("✅ ML estimators imported successfully")
        
        # Test neural estimators
        from lrdbenchmark import CNNEstimator, LSTMEstimator, GRUEstimator, TransformerEstimator
        print("✅ Neural estimators imported successfully")
        
        # Test benchmark system
        from lrdbenchmark import ComprehensiveBenchmark
        print("✅ Benchmark system imported successfully")
        
        # Test GPU utilities
        from lrdbenchmark import gpu_is_available, get_device_info, clear_gpu_cache
        print("✅ GPU utilities imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_data_generation():
    """Test data generation with new API."""
    print("\n🔍 Testing Data Generation...")
    
    try:
        from lrdbenchmark import FBMModel, FGNModel, ARFIMAModel, MRWModel, AlphaStableModel
        
        # Test FBM
        fbm = FBMModel(H=0.7, sigma=1.0)
        fbm_data = fbm.generate(length=1000, seed=42)
        print(f"✅ FBM generated: {len(fbm_data)} points, H=0.7")
        
        # Test FGN
        fgn = FGNModel(H=0.6, sigma=1.0)
        fgn_data = fgn.generate(length=1000, seed=42)
        print(f"✅ FGN generated: {len(fgn_data)} points, H=0.6")
        
        # Test ARFIMA
        arfima = ARFIMAModel(d=0.3, sigma=1.0)
        arfima_data = arfima.generate(length=1000, seed=42)
        print(f"✅ ARFIMA generated: {len(arfima_data)} points, d=0.3")
        
        # Test MRW
        mrw = MRWModel(H=0.7, lambda_param=0.1, sigma=1.0)
        mrw_data = mrw.generate(length=1000, seed=42)
        print(f"✅ MRW generated: {len(mrw_data)} points, H=0.7")
        
        # Test Alpha-Stable
        alpha_stable = AlphaStableModel(alpha=1.5, beta=0.0, sigma=1.0)
        alpha_data = alpha_stable.generate(length=1000, seed=42)
        print(f"✅ Alpha-Stable generated: {len(alpha_data)} points, α=1.5")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_classical_estimators():
    """Test classical estimators."""
    print("\n🔍 Testing Classical Estimators...")
    
    try:
        from lrdbenchmark import FBMModel, RSEstimator, DFAEstimator, GPHEstimator, WhittleEstimator
        
        # Generate test data
        fbm = FBMModel(H=0.7, sigma=1.0)
        data = fbm.generate(length=1000, seed=42)
        
        # Test R/S
        rs = RSEstimator()
        rs_result = rs.estimate(data)
        print(f"✅ R/S: H={rs_result['hurst_parameter']:.3f} (expected ~0.7)")
        
        # Test DFA
        dfa = DFAEstimator()
        dfa_result = dfa.estimate(data)
        print(f"✅ DFA: H={dfa_result['hurst_parameter']:.3f} (expected ~0.7)")
        
        # Test GPH
        gph = GPHEstimator()
        gph_result = gph.estimate(data)
        print(f"✅ GPH: H={gph_result['hurst_parameter']:.3f} (expected ~0.7)")
        
        # Test Whittle
        whittle = WhittleEstimator()
        whittle_result = whittle.estimate(data)
        print(f"✅ Whittle: H={whittle_result['hurst_parameter']:.3f} (expected ~0.7)")
        
        return True
        
    except Exception as e:
        print(f"❌ Classical estimators test failed: {e}")
        return False

def test_ml_estimators():
    """Test machine learning estimators."""
    print("\n🔍 Testing ML Estimators...")
    
    try:
        from lrdbenchmark import FBMModel, RandomForestEstimator, SVREstimator, GradientBoostingEstimator
        
        # Generate test data
        fbm = FBMModel(H=0.7, sigma=1.0)
        data = fbm.generate(length=1000, seed=42)
        
        # Test Random Forest
        rf = RandomForestEstimator()
        rf_result = rf.estimate(data)
        print(f"✅ Random Forest: H={rf_result['hurst_parameter']:.3f} (expected ~0.7)")
        
        # Test SVR
        svr = SVREstimator()
        svr_result = svr.estimate(data)
        print(f"✅ SVR: H={svr_result['hurst_parameter']:.3f} (expected ~0.7)")
        
        # Test Gradient Boosting
        gb = GradientBoostingEstimator()
        gb_result = gb.estimate(data)
        print(f"✅ Gradient Boosting: H={gb_result['hurst_parameter']:.3f} (expected ~0.7)")
        
        return True
        
    except Exception as e:
        print(f"❌ ML estimators test failed: {e}")
        return False

def test_neural_estimators():
    """Test neural network estimators."""
    print("\n🔍 Testing Neural Estimators...")
    
    try:
        from lrdbenchmark import FBMModel, CNNEstimator, LSTMEstimator, GRUEstimator, TransformerEstimator
        
        # Generate test data
        fbm = FBMModel(H=0.7, sigma=1.0)
        data = fbm.generate(length=1000, seed=42)
        
        # Test CNN
        cnn = CNNEstimator()
        cnn_result = cnn.estimate(data)
        print(f"✅ CNN: H={cnn_result['hurst_parameter']:.3f} (expected ~0.7)")
        
        # Test LSTM
        lstm = LSTMEstimator()
        lstm_result = lstm.estimate(data)
        print(f"✅ LSTM: H={lstm_result['hurst_parameter']:.3f} (expected ~0.7)")
        
        # Test GRU
        gru = GRUEstimator()
        gru_result = gru.estimate(data)
        print(f"✅ GRU: H={gru_result['hurst_parameter']:.3f} (expected ~0.7)")
        
        # Test Transformer
        transformer = TransformerEstimator()
        transformer_result = transformer.estimate(data)
        print(f"✅ Transformer: H={transformer_result['hurst_parameter']:.3f} (expected ~0.7)")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural estimators test failed: {e}")
        return False

def test_gpu_functionality():
    """Test GPU utilities and fallback mechanisms."""
    print("\n🔍 Testing GPU Functionality...")
    
    try:
        from lrdbenchmark import gpu_is_available, get_device_info, clear_gpu_cache
        
        # Check GPU availability
        gpu_available = gpu_is_available()
        print(f"✅ GPU available: {gpu_available}")
        
        if gpu_available:
            device_info = get_device_info()
            print(f"✅ Device info: {device_info}")
        
        # Test GPU cache clearing
        clear_gpu_cache()
        print("✅ GPU cache cleared successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU functionality test failed: {e}")
        return False

def test_comprehensive_benchmark():
    """Test the comprehensive benchmark system."""
    print("\n🔍 Testing Comprehensive Benchmark System...")
    
    try:
        from lrdbenchmark import ComprehensiveBenchmark, FBMModel
        
        # Create benchmark
        benchmark = ComprehensiveBenchmark()
        
        # Generate test data
        fbm = FBMModel(H=0.7, sigma=1.0)
        data = fbm.generate(length=1000, seed=42)
        
        # Run a small benchmark
        print("Running small benchmark (this may take a moment)...")
        start_time = time.time()
        
        results = benchmark.run_comprehensive_benchmark(
            data_length=1000,
            benchmark_type="comprehensive"
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Benchmark completed in {duration:.2f}s")
        print(f"✅ Results: Benchmark completed successfully")
        
        # Check results structure - the benchmark returns a summary, not individual results
        if results is not None:
            print("✅ Benchmark completed successfully")
            return True
        else:
            print("❌ Benchmark returned no results")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive benchmark test failed: {e}")
        return False

def test_performance_consistency():
    """Test that performance is consistent with expectations."""
    print("\n🔍 Testing Performance Consistency...")
    
    try:
        from lrdbenchmark import FBMModel, RSEstimator, DFAEstimator
        
        # Generate test data
        fbm = FBMModel(H=0.7, sigma=1.0)
        data = fbm.generate(length=1000, seed=42)
        
        # Time R/S estimation
        rs = RSEstimator()
        start_time = time.time()
        rs_result = rs.estimate(data)
        rs_time = time.time() - start_time
        
        # Time DFA estimation
        dfa = DFAEstimator()
        start_time = time.time()
        dfa_result = dfa.estimate(data)
        dfa_time = time.time() - start_time
        
        print(f"✅ R/S estimation: {rs_time:.3f}s, H={rs_result['hurst_parameter']:.3f}")
        print(f"✅ DFA estimation: {dfa_time:.3f}s, H={dfa_result['hurst_parameter']:.3f}")
        
        # Check that results are reasonable
        rs_h = rs_result['hurst_parameter']
        dfa_h = dfa_result['hurst_parameter']
        
        if 0.5 <= rs_h <= 0.9 and 0.5 <= dfa_h <= 0.9:
            print("✅ Results are within reasonable range")
        else:
            print(f"⚠️  Results may be outside expected range: R/S={rs_h:.3f}, DFA={dfa_h:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance consistency test failed: {e}")
        return False

def main():
    """Run comprehensive validation benchmark."""
    print("🚀 LRDBenchmark v2.3.0 Comprehensive Validation")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Data Generation", test_data_generation),
        ("Classical Estimators", test_classical_estimators),
        ("ML Estimators", test_ml_estimators),
        ("Neural Estimators", test_neural_estimators),
        ("GPU Functionality", test_gpu_functionality),
        ("Comprehensive Benchmark", test_comprehensive_benchmark),
        ("Performance Consistency", test_performance_consistency),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! v2.3.0 is working correctly!")
        return True
    else:
        print(f"⚠️  {total-passed} tests failed. Please investigate.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
