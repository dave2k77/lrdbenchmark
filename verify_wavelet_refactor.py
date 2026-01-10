
import sys
import os
import numpy as np
import time

sys.path.append(os.getcwd())

from lrdbenchmark.generation import TimeSeriesGenerator
from lrdbenchmark.analysis.wavelet.log_variance_estimator import WaveletLogVarianceEstimator

def verify_wavelet():
    print("Generating FGN noise (H=0.7, Length=16384)...")
    gen = TimeSeriesGenerator(random_state=42)
    # Using L=16k for wavelet scales
    res = gen.generate(model='fgn', length=16384, params={'H': 0.7}, preprocess=True)
    signal = res['signal']
    
    backends = ['numpy', 'jax']
    results = {}
    
    print("\n--- Testing WaveletLogVarianceEstimator ---")
    
    for backend in backends:
        print(f"\nTesting backend: {backend}")
        try:
            est = WaveletLogVarianceEstimator(use_optimization=backend, wavelet='db4')
            start_time = time.time()
            res_dict = est.estimate(signal)
            dur = time.time() - start_time
            
            h = res_dict['hurst_parameter']
            fw = res_dict['optimization_framework']
            slope = res_dict.get('slope', 0)
            
            print(f"  H = {h:.4f}")
            print(f"  Slope = {slope:.4f}")
            print(f"  Framework: {fw}")
            print(f"  Time: {dur:.4f}s")
            
            results[backend] = h
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Check Consistency
    if 'numpy' in results and 'jax' in results:
        diff = abs(results['numpy'] - results['jax'])
        print(f"  Difference JAX vs NumPy: {diff:.6f}")
        if diff < 1e-4:
            print("  OK: Consistent")
        else:
            print("  WARNING: Inconsistent!")

if __name__ == "__main__":
    verify_wavelet()
