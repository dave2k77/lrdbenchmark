Comprehensive Adaptive Estimators Demo
======================================

This comprehensive demo showcases the new adaptive estimator system with automatic framework selection, robust error handling, and EEG contamination testing.

Overview
--------

The comprehensive adaptive estimators provide:

* **Automatic Framework Selection**: GPU/JAX, CPU/Numba, or NumPy based on data characteristics
* **Robust Error Handling**: Adaptive parameter selection and progressive fallbacks
* **Performance Profiling**: Built-in performance monitoring and optimization
* **EEG Contamination Testing**: Realistic artifact scenarios for biomedical applications
* **Mathematical Verification**: All estimators verified against theoretical foundations

Basic Usage
-----------

.. code-block:: python

   from lrdbenchmark.analysis.comprehensive_adaptive_estimators import get_all_comprehensive_adaptive_classical_estimators
   from lrdbenchmark import FBMModel
   import numpy as np
   
   # Get all comprehensive adaptive estimators
   estimators = get_all_comprehensive_adaptive_classical_estimators()
   
   # Generate test data
   model = FBMModel(H=0.7, sigma=1.0)
   data = model.generate(1000, seed=42)
   
   print("Comprehensive Adaptive Estimators Demo")
   print("=" * 50)
   print(f"Generated FBM data with true H = 0.7")
   print(f"Data length: {len(data)}")
   print(f"Data mean: {data.mean():.3f}, std: {data.std():.3f}")
   
   # Test all estimators
   results = {}
   for name, estimator in estimators.items():
       try:
           result = estimator.estimate(data)
           results[name] = result
           print(f"{name:25s}: H={result['hurst_parameter']:.4f}, "
                 f"Framework={result['framework_used']:15s}, "
                 f"Time={result['execution_time']:.4f}s")
       except Exception as e:
           print(f"{name:25s}: Failed - {e}")
   
   # Calculate average error
   errors = []
   for name, result in results.items():
       error = abs(result['hurst_parameter'] - 0.7)
       errors.append(error)
   
   print(f"\nAverage absolute error: {np.mean(errors):.4f}")
   print(f"Best estimator: {min(results.keys(), key=lambda k: abs(results[k]['hurst_parameter'] - 0.7))}")

Performance Comparison
----------------------

.. code-block:: python

   from lrdbenchmark.analysis.comprehensive_adaptive_estimators import ComprehensiveAdaptiveRS, ComprehensiveAdaptiveDFA
   from lrdbenchmark import FBMModel
   import numpy as np
   import time
   
   # Generate test data
   model = FBMModel(H=0.7, sigma=1.0)
   data = model.generate(5000, seed=42)
   
   print("Performance Comparison Demo")
   print("=" * 40)
   print(f"Data length: {len(data)}")
   
   # Test R/S estimator
   rs_estimator = ComprehensiveAdaptiveRS(enable_profiling=True)
   start_time = time.time()
   rs_result = rs_estimator.estimate(data)
   rs_time = time.time() - start_time
   
   # Test DFA estimator
   dfa_estimator = ComprehensiveAdaptiveDFA(enable_profiling=True)
   start_time = time.time()
   dfa_result = dfa_estimator.estimate(data)
   dfa_time = time.time() - start_time
   
   print(f"R/S Analysis:")
   print(f"  Hurst estimate: {rs_result['hurst_parameter']:.4f}")
   print(f"  Framework used: {rs_result['framework_used']}")
   print(f"  Execution time: {rs_time:.4f}s")
   print(f"  Framework reasoning: {rs_result['framework_reasoning']}")
   
   print(f"\nDFA Analysis:")
   print(f"  Hurst estimate: {dfa_result['hurst_parameter']:.4f}")
   print(f"  Framework used: {dfa_result['framework_used']}")
   print(f"  Execution time: {dfa_time:.4f}s")
   print(f"  Framework reasoning: {dfa_result['framework_reasoning']}")
   
   # Get performance information
   rs_perf = rs_estimator.get_performance_info()
   dfa_perf = dfa_estimator.get_performance_info()
   
   print(f"\nPerformance Summary:")
   print(f"R/S total runs: {rs_perf['total_runs']}")
   print(f"DFA total runs: {dfa_perf['total_runs']}")

EEG Contamination Testing
-------------------------

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
   
   # Get pure data estimate
   pure_result = estimator.estimate(pure_data)
   pure_h = pure_result['hurst_parameter']
   
   print("EEG Contamination Testing Demo")
   print("=" * 40)
   print(f"Pure data H estimate: {pure_h:.4f}")
   print(f"Pure data framework: {pure_result['framework_used']}")
   
   # EEG contamination scenarios
   eeg_scenarios = [
       ConfoundingScenario.EEG_OCULAR_ARTIFACTS,
       ConfoundingScenario.EEG_MUSCLE_ARTIFACTS,
       ConfoundingScenario.EEG_CARDIAC_ARTIFACTS,
       ConfoundingScenario.EEG_ELECTRODE_POPPING,
       ConfoundingScenario.EEG_ELECTRODE_DRIFT,
       ConfoundingScenario.EEG_60HZ_NOISE,
       ConfoundingScenario.EEG_SWEAT_ARTIFACTS,
       ConfoundingScenario.EEG_MOVEMENT_ARTIFACTS
   ]
   
   print(f"\nEEG Contamination Results:")
   print(f"{'Scenario':<25s} {'H Estimate':<12s} {'Error':<8s} {'Framework':<15s}")
   print("-" * 70)
   
   for scenario in eeg_scenarios:
       contaminated_data, description = contamination_factory.apply_confounding(
           pure_data, scenario, intensity=0.3
       )
       result = estimator.estimate(contaminated_data)
       
       h_est = result['hurst_parameter']
       error = abs(h_est - pure_h)
       framework = result['framework_used']
       
       print(f"{scenario.value:<25s} {h_est:<12.4f} {error:<8.4f} {framework:<15s}")
   
   # Calculate robustness metrics
   errors = []
   for scenario in eeg_scenarios:
       contaminated_data, _ = contamination_factory.apply_confounding(
           pure_data, scenario, intensity=0.3
       )
       result = estimator.estimate(contaminated_data)
       error = abs(result['hurst_parameter'] - pure_h)
       errors.append(error)
   
   print(f"\nRobustness Summary:")
   print(f"Average error: {np.mean(errors):.4f}")
   print(f"Max error: {np.max(errors):.4f}")
   print(f"Min error: {np.min(errors):.4f}")
   print(f"Std error: {np.std(errors):.4f}")

Data Length Robustness Testing
------------------------------

.. code-block:: python

   from lrdbenchmark.analysis.comprehensive_adaptive_estimators import ComprehensiveAdaptiveGPH, ComprehensiveAdaptiveCWT
   from lrdbenchmark import FBMModel
   import numpy as np
   
   # Create estimators
   gph_estimator = ComprehensiveAdaptiveGPH()
   cwt_estimator = ComprehensiveAdaptiveCWT()
   
   # Generate test data
   model = FBMModel(H=0.7, sigma=1.0)
   
   # Test different data lengths
   data_lengths = [50, 100, 200, 500, 1000, 2000, 5000]
   
   print("Data Length Robustness Testing")
   print("=" * 40)
   print(f"{'Length':<8s} {'GPH H':<8s} {'GPH Error':<10s} {'CWT H':<8s} {'CWT Error':<10s}")
   print("-" * 50)
   
   for length in data_lengths:
       data = model.generate(length, seed=42)
       
       # Test GPH
       try:
           gph_result = gph_estimator.estimate(data)
           gph_h = gph_result['hurst_parameter']
           gph_error = abs(gph_h - 0.7)
           gph_str = f"{gph_h:.4f}"
           gph_err_str = f"{gph_error:.4f}"
       except Exception as e:
           gph_str = "Failed"
           gph_err_str = "N/A"
       
       # Test CWT
       try:
           cwt_result = cwt_estimator.estimate(data)
           cwt_h = cwt_result['hurst_parameter']
           cwt_error = abs(cwt_h - 0.7)
           cwt_str = f"{cwt_h:.4f}"
           cwt_err_str = f"{cwt_error:.4f}"
       except Exception as e:
           cwt_str = "Failed"
           cwt_err_str = "N/A"
       
       print(f"{length:<8d} {gph_str:<8s} {gph_err_str:<10s} {cwt_str:<8s} {cwt_err_str:<10s}")

Framework Selection Analysis
----------------------------

.. code-block:: python

   from lrdbenchmark.analysis.comprehensive_adaptive_estimators import ComprehensiveAdaptiveRS
   from lrdbenchmark import FBMModel
   import numpy as np
   
   # Create estimator with profiling
   estimator = ComprehensiveAdaptiveRS(enable_profiling=True)
   
   # Generate test data
   model = FBMModel(H=0.7, sigma=1.0)
   
   # Test with different data sizes
   data_sizes = [100, 500, 1000, 2000, 5000, 10000]
   
   print("Framework Selection Analysis")
   print("=" * 40)
   print(f"{'Data Size':<10s} {'Framework':<15s} {'Time (s)':<10s} {'H Estimate':<12s}")
   print("-" * 50)
   
   for size in data_sizes:
       data = model.generate(size, seed=42)
       result = estimator.estimate(data)
       
       print(f"{size:<10d} {result['framework_used']:<15s} "
             f"{result['execution_time']:<10.4f} {result['hurst_parameter']:<12.4f}")
   
   # Get performance information
   perf_info = estimator.get_performance_info()
   print(f"\nPerformance Summary:")
   print(f"Total runs: {perf_info['total_runs']}")
   print(f"Frameworks used: {perf_info['frameworks_used']}")

Comprehensive Benchmark
-----------------------

.. code-block:: python

   from lrdbenchmark.analysis.comprehensive_adaptive_estimators import get_all_comprehensive_adaptive_classical_estimators
   from lrdbenchmark import ContaminationFactory, ConfoundingScenario
   from lrdbenchmark import FBMModel
   from lrdbenchmark import FGNModel
   import numpy as np
   import pandas as pd
   
   # Get all estimators
   estimators = get_all_comprehensive_adaptive_classical_estimators()
   contamination_factory = ContaminationFactory()
   
   # Test scenarios
   data_models = {
       'FBM': FBMModel(H=0.7, sigma=1.0),
       'FGN': FGNModel(H=0.7, sigma=1.0)
   }
   
   contamination_scenarios = [
       ConfoundingScenario.PURE,
       ConfoundingScenario.GAUSSIAN_NOISE,
       ConfoundingScenario.EEG_OCULAR_ARTIFACTS,
       ConfoundingScenario.EEG_MUSCLE_ARTIFACTS
   ]
   
   data_lengths = [500, 1000, 2000]
   
   print("Comprehensive Benchmark Demo")
   print("=" * 40)
   
   # Store results
   all_results = []
   
   for model_name, model in data_models.items():
       for length in data_lengths:
           for scenario in contamination_scenarios:
               # Generate data
               data = model.generate(length, seed=42)
               
               # Apply contamination
               if scenario != ConfoundingScenario.PURE:
                   data, description = contamination_factory.apply_confounding(
                       data, scenario, intensity=0.3
                   )
               
               # Test all estimators
               for est_name, estimator in estimators.items():
                   try:
                       result = estimator.estimate(data)
                       all_results.append({
                           'Model': model_name,
                           'Length': length,
                           'Scenario': scenario.value,
                           'Estimator': est_name,
                           'H_Estimate': result['hurst_parameter'],
                           'Framework': result['framework_used'],
                           'Time': result['execution_time'],
                           'Success': True
                       })
                   except Exception as e:
                       all_results.append({
                           'Model': model_name,
                           'Length': length,
                           'Scenario': scenario.value,
                           'Estimator': est_name,
                           'H_Estimate': np.nan,
                           'Framework': 'Failed',
                           'Time': 0,
                           'Success': False
                       })
   
   # Create DataFrame
   df = pd.DataFrame(all_results)
   
   # Calculate success rate
   success_rate = df['Success'].mean() * 100
   print(f"Overall success rate: {success_rate:.1f}%")
   
   # Calculate average error (for successful estimates)
   successful_df = df[df['Success'] == True]
   if len(successful_df) > 0:
       # Calculate error for FBM (true H = 0.7)
       fbm_df = successful_df[successful_df['Model'] == 'FBM']
       if len(fbm_df) > 0:
           fbm_df['Error'] = abs(fbm_df['H_Estimate'] - 0.7)
           avg_error = fbm_df['Error'].mean()
           print(f"Average error (FBM): {avg_error:.4f}")
   
   # Framework usage statistics
   framework_usage = df['Framework'].value_counts()
   print(f"\nFramework usage:")
   for framework, count in framework_usage.items():
       print(f"  {framework}: {count} ({count/len(df)*100:.1f}%)")
   
   # Estimator performance
   estimator_success = df.groupby('Estimator')['Success'].mean() * 100
   print(f"\nEstimator success rates:")
   for estimator, success_rate in estimator_success.items():
       print(f"  {estimator}: {success_rate:.1f}%")

Best Practices
--------------

1. **Use Comprehensive Adaptive Estimators**: They provide the most robust and efficient estimation
2. **Enable Performance Profiling**: Monitor framework selection and performance
3. **Test with Contamination**: Use realistic contamination scenarios for robustness testing
4. **Compare Multiple Estimators**: Different estimators may perform better for different data types
5. **Validate with Known Data**: Test on synthetic data with known Hurst parameters
6. **Monitor Framework Selection**: Understand which frameworks are being used and why

.. note::
   The comprehensive adaptive estimators automatically handle most edge cases and
   provide the most reliable way to estimate Hurst parameters from time series data.

.. warning::
   While the adaptive system is robust, very short data (< 50 points) may still
   produce unreliable results regardless of the framework used.
