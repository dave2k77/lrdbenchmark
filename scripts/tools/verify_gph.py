
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from lrdbenchmark.analysis.spectral.gph.gph_estimator_unified import GPHEstimator

def test_gph_failure():
    print("Testing GPH Failure Mode (Small Data)...")
    # Very small data to trigger "insufficient data" path
    data = np.random.randn(10)
    
    estimator = GPHEstimator(use_optimization='numpy')
    try:
        result = estimator.estimate(data)
        print(f"NumPy Result (Small Data): {result}")
    except ValueError as e:
        print(f"SUCCESS: NumPy correctly raised error: {e}")
    except Exception as e:
        print(f"FAILURE: NumPy raised unexpected error: {e}")

def test_gph_numba_integration():
    print("\nTesting GPH Numba Integration...")
    try:
        import numba
        print("Numba is available.")
    except ImportError:
        print("Numba not available, skipping.")
        return

    data = np.random.randn(2000)
    estimator = GPHEstimator(use_optimization='numba')
    
    # Call _estimate_numba directly to see exception
    print("Calling _estimate_numba directly...")
    try:
        result = estimator._estimate_numba(data)
        print("SUCCESS: _estimate_numba execution successful.")
    except Exception as e:
        print(f"FAILURE: _estimate_numba failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Check if estimate() wrappers it correctly
    try:
        result = estimator.estimate(data)
        if result.get('method') == 'numba':
            print("SUCCESS: GPH used Numba implementation.")
        else:
             print(f"WARNING: GPH did not use Numba (used {result.get('method')}).")
    except Exception as e:
        print(f"FAILURE: GPH estimate() failed: {e}")

if __name__ == "__main__":
    test_gph_failure()
    test_gph_numba_integration()
