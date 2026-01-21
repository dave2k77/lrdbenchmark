Comprehensive Adaptive Estimators API
=====================================

LRDBenchmark provides a comprehensive suite of 13 adaptive classical estimators that automatically select the optimal computation framework (GPU/JAX, CPU/Numba, or NumPy) based on data characteristics and hardware availability.

Optimization Backend
--------------------

The intelligent optimization backend automatically selects the best computation framework for each estimator based on data size, hardware availability, and performance characteristics.

.. autoclass:: lrdbenchmark.analysis.optimization_backend.OptimizationBackend
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lrdbenchmark.analysis.optimization_backend.OptimizationFramework
   :members:
   :undoc-members:
   :show-inheritance:

Comprehensive Adaptive Estimators
---------------------------------

All comprehensive adaptive estimators inherit from the base class and provide automatic framework selection, robust error handling, and performance profiling.

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Temporal Estimators
-------------------

Comprehensive Adaptive R/S Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveRS
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Comprehensive Adaptive DFA
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveDFA
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Comprehensive Adaptive DMA
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveDMA
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Comprehensive Adaptive Higuchi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveHiguchi
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Spectral Estimators
-------------------

Comprehensive Adaptive GPH
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveGPH
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Comprehensive Adaptive Whittle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveWhittle
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Comprehensive Adaptive Periodogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptivePeriodogram
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Wavelet Estimators
------------------

Comprehensive Adaptive CWT
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveCWT
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Comprehensive Adaptive Wavelet Variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveWaveletVar
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Comprehensive Adaptive Wavelet Log Variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveWaveletLogVar
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Comprehensive Adaptive Wavelet Whittle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveWaveletWhittle
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Multifractal Estimators
-----------------------

Comprehensive Adaptive MFDFA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveMFDFA
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Comprehensive Adaptive Wavelet Leaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.ComprehensiveAdaptiveWaveletLeaders
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate
   .. automethod:: get_performance_info

Factory Functions
-----------------

Get All Comprehensive Adaptive Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: lrdbenchmark.analysis.comprehensive_adaptive_estimators.get_all_comprehensive_adaptive_classical_estimators

Usage Examples
--------------

Basic Usage with Automatic Framework Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.analysis.comprehensive_adaptive_estimators import get_all_comprehensive_adaptive_classical_estimators
   from lrdbenchmark import FBMModel
   import numpy as np
   
   # Get all comprehensive adaptive estimators
   estimators = get_all_comprehensive_adaptive_classical_estimators()
   
   # Generate test data
   model = FBMModel(H=0.7, sigma=1.0)
   data = model.generate(1000, seed=42)
   
   # Test a few key estimators
   test_estimators = ['Comprehensive_RS', 'Comprehensive_DFA', 'Comprehensive_GPH', 'Comprehensive_CWT']
   
   for name in test_estimators:
       estimator = estimators[name]
       result = estimator.estimate(data)
       print(f"{name}: Hurst={result['hurst_parameter']:.4f}, "
             f"Framework={result['framework_used']}, "
             f"Time={result['execution_time']:.4f}s")

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.analysis.comprehensive_adaptive_estimators import ComprehensiveAdaptiveRS
   from lrdbenchmark import FBMModel
   import numpy as np
   
   # Create estimator with performance profiling
   estimator = ComprehensiveAdaptiveRS(enable_profiling=True)
   
   # Generate test data
   model = FBMModel(H=0.7, sigma=1.0)
   data = model.generate(1000, seed=42)
   
   # Run multiple estimations
   for i in range(5):
       result = estimator.estimate(data)
       print(f"Run {i+1}: Framework={result['framework_used']}, "
             f"Time={result['execution_time']:.4f}s")
   
   # Get performance information
   perf_info = estimator.get_performance_info()
   print(f"\nPerformance Summary:")
   print(f"Total runs: {perf_info['total_runs']}")
   print(f"Frameworks used: {perf_info['frameworks_used']}")

EEG Contamination Testing
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.analysis.comprehensive_adaptive_estimators import ComprehensiveAdaptiveDFA
   from lrdbenchmark import ContaminationFactory, ConfoundingScenario
   from lrdbenchmark import FBMModel
   import numpy as np
   
   # Create estimator and contamination factory
   estimator = ComprehensiveAdaptiveDFA()
   contamination_factory = ContaminationFactory()
   
   # Generate pure data
   model = FBMModel(H=0.7, sigma=1.0)
   pure_data = model.generate(1000, seed=42)
   
   # Test with EEG contamination scenarios
   eeg_scenarios = [
       ConfoundingScenario.EEG_OCULAR_ARTIFACTS,
       ConfoundingScenario.EEG_MUSCLE_ARTIFACTS,
       ConfoundingScenario.EEG_CARDIAC_ARTIFACTS,
       ConfoundingScenario.EEG_60HZ_NOISE
   ]
   
   print("EEG Contamination Testing:")
   print(f"Pure data H estimate: {estimator.estimate(pure_data)['hurst_parameter']:.4f}")
   
   for scenario in eeg_scenarios:
       contaminated_data, description = contamination_factory.apply_confounding(
           pure_data, scenario, intensity=0.3
       )
       result = estimator.estimate(contaminated_data)
       print(f"{scenario.value}: H={result['hurst_parameter']:.4f}")

Framework Selection Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.analysis.comprehensive_adaptive_estimators import ComprehensiveAdaptiveGPH
   from lrdbenchmark import FBMModel
   import numpy as np
   
   # Test with different data sizes to see framework selection
   model = FBMModel(H=0.7, sigma=1.0)
   
   data_sizes = [100, 500, 1000, 5000, 10000]
   
   for size in data_sizes:
       data = model.generate(size, seed=42)
       estimator = ComprehensiveAdaptiveGPH()
       result = estimator.estimate(data)
       
       print(f"Data size: {size:5d}, "
             f"Framework: {result['framework_used']:15s}, "
             f"Time: {result['execution_time']:.4f}s, "
             f"H: {result['hurst_parameter']:.4f}")

Error Handling and Robustness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.analysis.comprehensive_adaptive_estimators import ComprehensiveAdaptiveCWT
   import numpy as np
   
   estimator = ComprehensiveAdaptiveCWT()
   
   # Test with various data conditions
   test_cases = {
       'Short data': np.random.randn(50),
       'Medium data': np.random.randn(500),
       'Long data': np.random.randn(5000),
       'Noisy data': np.random.randn(1000) + 0.5 * np.random.randn(1000),
       'Trend data': np.cumsum(np.random.randn(1000)) + np.linspace(0, 10, 1000)
   }
   
   for name, data in test_cases.items():
       try:
           result = estimator.estimate(data)
           print(f"{name:15s}: H={result['hurst_parameter']:.4f}, "
                 f"Framework={result['framework_used']}")
       except Exception as e:
           print(f"{name:15s}: Failed - {e}")

Best Practices
--------------

1. **Automatic Framework Selection**: Let the system automatically choose the best framework
2. **Performance Profiling**: Enable profiling for performance monitoring
3. **Error Handling**: Always handle potential estimation errors gracefully
4. **Data Validation**: Ensure data meets minimum requirements for reliable estimates
5. **Multiple Estimators**: Compare results from different estimator types
6. **Contamination Testing**: Test robustness with realistic contamination scenarios

.. note::
   The comprehensive adaptive estimators automatically handle framework selection,
   error recovery, and performance optimization. They provide the most robust
   and efficient way to use classical LRD estimators.

.. warning::
   While the adaptive system handles most edge cases automatically, very short
   data (< 50 points) may still produce unreliable results regardless of the
   framework used.
