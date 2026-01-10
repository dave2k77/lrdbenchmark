
import sys
import os
import numpy as np
import time

sys.path.append(os.getcwd())

from lrdbenchmark.generation import TimeSeriesGenerator
from lrdbenchmark.analysis.multifractal.mfdfa_estimator import MFDFAEstimator

def verify_mfdfa():
    print("Generating FGN noise (H=0.7, Length=4096)...")
    gen = TimeSeriesGenerator(random_state=42)
    res = gen.generate(model='fgn', length=4096, params={'H': 0.7}, preprocess=True)
    signal = res['signal']
    
    backends = ['numpy', 'jax']
    results = {}
    
    print("\n--- Testing MFDFAEstimator ---")
    q_orders = [-2, 0, 2]
    
    for backend in backends:
        print(f"\nTesting backend: {backend}")
        try:
            est = MFDFAEstimator(use_optimization=backend, q_orders=q_orders, order=1)
            start_time = time.time()
            res_dict = est.estimate(signal)
            dur = time.time() - start_time
            
            h_qs = res_dict['h_qs']
            width = res_dict['width']
            fw = res_dict['optimization_framework']
            
            print(f"  H(q) = {h_qs}")
            print(f"  Width = {width:.4f}")
            print(f"  Framework: {fw}")
            print(f"  Time: {dur:.4f}s")
            
            results[backend] = h_qs
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Check Consistency
    if 'numpy' in results and 'jax' in results:
        diff = np.max(np.abs(np.array(results['numpy']) - np.array(results['jax'])))
        print(f"  Max Diff JAX vs NumPy: {diff:.6f}")
        if diff < 1e-4:
            print("  OK: Consistent")
        else:
            print("  WARNING: Inconsistent!")

if __name__ == "__main__":
    verify_mfdfa()
