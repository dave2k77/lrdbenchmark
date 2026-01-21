
import sys
import os
import numpy as np
import time

sys.path.append(os.getcwd())

from lrdbenchmark.generation import TimeSeriesGenerator
from lrdbenchmark.analysis.spectral.periodogram_estimator import PeriodogramEstimator

def verify_spectral():
    print("Generating FGN noise (H=0.7, Length=16384)...")
    # Using longer series for better spectral resolution at low freq
    gen = TimeSeriesGenerator(random_state=42)
    res = gen.generate(model='fgn', length=16384, params={'H': 0.7}, preprocess=True)
    signal = res['signal']
    
    # Test cases:
    # 1. Welch (JAX vs NumPy)
    print("\n--- Testing Welch Method ---")
    backends = ['numpy', 'jax']
    results = {}
    
    for backend in backends:
        print(f"\nTesting backend: {backend}")
        try:
            est = PeriodogramEstimator(use_optimization=backend, use_welch=True)
            start_time = time.time()
            res_dict = est.estimate(signal)
            dur = time.time() - start_time
            
            h = res_dict['hurst_parameter']
            fw = res_dict['optimization_framework']
            
            print(f"  H = {h:.4f}")
            print(f"  Framework: {fw}")
            print(f"  Method: {res_dict['method']}")
            print(f"  Time: {dur:.4f}s")
            
            results[backend] = h
        except Exception as e:
            print(f"  FAILED: {e}")

    # Check Consistency
    if 'numpy' in results and 'jax' in results:
        diff = abs(results['numpy'] - results['jax'])
        print(f"  Difference JAX vs NumPy: {diff:.6f}")
        if diff < 1e-4:
            print("  OK: Consistent")
        else:
            print("  WARNING: Inconsistent!")

    # 2. Multitaper Fallback in JAX
    print("\n--- Testing Multitaper Fallback (JAX) ---")
    try:
        est = PeriodogramEstimator(use_optimization='jax', use_multitaper=True)
        res_dict = est.estimate(signal)
        print(f"  H = {res_dict['hurst_parameter']:.4f}")
        print(f"  Framework Used: {res_dict['optimization_framework']}")
        if res_dict['optimization_framework'] == 'numpy (fallback)':
             print("  OK: Correctly fell back to NumPy")
        else:
             print("  WARNING: Did not fallback as expected (or JAX implemented it?)")
    except Exception as e:
        print(f"  FAILED: {e}")

if __name__ == "__main__":
    verify_spectral()
