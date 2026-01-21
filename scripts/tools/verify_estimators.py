import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from lrdbenchmark.analysis.benchmark import ComprehensiveBenchmark

def verify_estimators():
    print("Initializing Benchmark...")
    bench = ComprehensiveBenchmark()
    
    print("\nGetting ML Estimators...")
    try:
        ml_ests = bench.get_estimators_by_type("ML")
        print(f"Found {len(ml_ests)} ML estimators: {list(ml_ests.keys())}")
    except Exception as e:
        print(f"Error getting ML estimators: {e}")
        ml_ests = {}

    print("\nGetting NN Estimators...")
    try:
        nn_ests = bench.get_estimators_by_type("neural")
        print(f"Found {len(nn_ests)} NN estimators: {list(nn_ests.keys())}")
    except Exception as e:
        print(f"Error getting NN estimators: {e}")
        nn_ests = {}
    
    # Test data
    data_lengths = [2048]
    
    print("\nTesting Inference...")
    estimators = {**ml_ests, **nn_ests}
    
    for name, est in estimators.items():
        print(f"\nTesting {name}...")
        for N in data_lengths:
            try:
                data = np.random.randn(N)
                res = est.estimate(data)
                h = res.get('hurst_parameter')
                print(f"  Length {N}: Success, H={h}")
            except Exception as e:
                print(f"  Length {N}: Failed - {e}")

if __name__ == "__main__":
    verify_estimators()
