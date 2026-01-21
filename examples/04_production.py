#!/usr/bin/env python3
"""
LRDBenchmark Production Deployment Example

This example shows production-ready patterns for LRDBenchmark,
including error handling, performance monitoring, and batch processing.
"""

import numpy as np
import time
from lrdbenchmark import FBMModel, RSEstimator, DFAEstimator, GPHEstimator
from lrdbenchmark.analysis.benchmark import ComprehensiveBenchmark
from lrdbenchmark.exceptions import EstimatorError, GPUMemoryError

def safe_estimate(estimator, data, estimator_name):
    """Safely estimate with error handling."""
    try:
        start_time = time.time()
        result = estimator.estimate(data)
        execution_time = time.time() - start_time
        
        return {
            'success': True,
            'result': result,
            'execution_time': execution_time,
            'estimator': estimator_name
        }
    except GPUMemoryError as e:
        print(f"⚠️ GPU memory error with {estimator_name}: {e}")
        return {
            'success': False,
            'error': 'GPU_MEMORY_ERROR',
            'estimator': estimator_name
        }
    except EstimatorError as e:
        print(f"⚠️ Estimator error with {estimator_name}: {e}")
        return {
            'success': False,
            'error': 'ESTIMATOR_ERROR',
            'estimator': estimator_name
        }
    except Exception as e:
        print(f"⚠️ Unexpected error with {estimator_name}: {e}")
        return {
            'success': False,
            'error': 'UNEXPECTED_ERROR',
            'estimator': estimator_name
        }

def main():
    """Main production workflow."""
    print("LRDBenchmark Production Example")
    print("=" * 50)
    
    # Generate test data
    print("Generating test data...")
    fbm = FBMModel(H=0.7, sigma=1.0)
    data = fbm.generate(length=1000, seed=42)
    print(f"Generated {len(data)} data points")
    
    # Initialize estimators
    estimators = {
        'R/S': RSEstimator(),
        'DFA': DFAEstimator(),
        'GPH': GPHEstimator()
    }
    
    # Process with error handling
    print("\nRunning estimators...")
    results = []
    
    for name, estimator in estimators.items():
        result = safe_estimate(estimator, data, name)
        results.append(result)
        
        if result['success']:
            h_est = result['result']['hurst_parameter']
            exec_time = result['execution_time']
            print(f"✓ {name:>6}: H = {h_est:.3f} (took {exec_time:.3f}s)")
        else:
            print(f"✗ {name:>6}: Failed - {result['error']}")
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nSummary: {successful}/{len(results)} estimators successful")
    
    # Benchmark example
    print("\nRunning comprehensive benchmark...")
    try:
        benchmark = ComprehensiveBenchmark()
        
        # Small benchmark for demo
        benchmark_results = benchmark.run_classical_estimators(
            data_models=['fbm'],
            n_samples=10,
            n_trials=3
        )
        
        print("✓ Benchmark completed successfully")
        print(f"  Results: {len(benchmark_results)} estimator categories")
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
    
    print("\nProduction workflow completed!")

if __name__ == "__main__":
    main()
