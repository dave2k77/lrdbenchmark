
import sys
import os
import numpy as np
import time

sys.path.append(os.getcwd())

from lrdbenchmark.generation import TimeSeriesGenerator
from lrdbenchmark.analysis.spectral.whittle_estimator import WhittleEstimator

def verify_whittle():
    print("Generating FGN noise (H=0.7, Length=4096)...")
    gen = TimeSeriesGenerator(random_state=42)
    # Whittle expects FGN (Short Memory? No, FGN is LRD).
    # Whittle fGn spectrum shape assumes fGn input.
    res = gen.generate(model='fgn', length=4096, params={'H': 0.7}, preprocess=True)
    signal = res['signal']
    
    backends = ['numpy', 'jax']
    results = {}
    
    for backend in backends:
        print(f"\nTesting backend: {backend}")
        try:
            est = WhittleEstimator(use_optimization=backend)
            start_time = time.time()
            res_dict = est.estimate(signal)
            dur = time.time() - start_time
            
            h = res_dict['hurst_parameter']
            fw = res_dict['optimization_framework']
            
            print(f"  H = {h:.4f}")
            print(f"  Framework: {fw}")
            print(f"  Time: {dur:.4f}s")
            
            results[backend] = h
        except Exception as e:
            print(f"  FAILED: {e}")

    # Check Consistency
    if 'numpy' in results and 'jax' in results:
        diff = abs(results['numpy'] - results['jax'])
        print(f"  Difference JAX vs NumPy: {diff:.6f}")
        if diff < 1e-3:
            print("  OK: Consistent")
        else:
            print("  WARNING: Inconsistent!")

if __name__ == "__main__":
    verify_whittle()
