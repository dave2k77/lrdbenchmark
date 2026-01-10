Estimators API
=============

lrdbenchmark provides a comprehensive suite of 18 estimators for detecting and quantifying long-range dependence in time series data.

Base Estimator
-------------

.. autoclass:: lrdbenchmark.analysis.base_estimator.BaseEstimator
   :members:
   :undoc-members:
   :show-inheritance:

Temporal Estimators
------------------

Detrended Fluctuation Analysis (DFA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.temporal.dfa_estimator.DFAEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Detrended Moving Average (DMA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.temporal.dma_estimator.DMAEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Higuchi Method
~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.temporal.higuchi_estimator.HiguchiEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Generalised Hurst Exponent (GHE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.temporal.ghe_estimator.GHEEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

R/S Analysis
~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.temporal.rs_estimator.RSEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Spectral Estimators
------------------

Geweke-Porter-Hudak (GPH)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.spectral.gph_estimator.GPHEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Periodogram
~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.spectral.periodogram_estimator.PeriodogramEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Whittle Estimator
~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.spectral.whittle_estimator.WhittleEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Wavelet Estimators
-----------------

Continuous Wavelet Transform (CWT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.wavelet.cwt_estimator.CWTEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Wavelet Variance
~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.wavelet.variance_estimator.WaveletVarianceEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Wavelet Log-Variance
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.wavelet.log_variance_estimator.WaveletLogVarianceEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Wavelet Whittle
~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.wavelet.whittle_estimator.WaveletWhittleEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Multifractal Estimators
-----------------------

Multifractal Detrended Fluctuation Analysis (MFDFA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.multifractal.mfdfa_estimator.MFDFAEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Multifractal Wavelet Leaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.multifractal.wavelet_leaders_estimator.MultifractalWaveletLeadersEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: estimate

Machine Learning Estimators
---------------------------

For detailed documentation of machine learning estimators, see :doc:`machine_learning_estimators`.

The following ML estimators are available:

* **Random Forest**: Ensemble tree-based estimation with feature importance
* **Support Vector Regression**: SVM-based estimation with RBF kernel
* **Gradient Boosting**: Boosted tree estimation with comprehensive feature engineering

See the dedicated :doc:`machine_learning_estimators` page for complete API documentation, performance metrics, and usage examples.

Neural Network Estimators
-------------------------

For detailed documentation of neural network estimators, see :doc:`neural_network_factory`.

The following neural network architectures are available:

* **CNN**: Convolutional Neural Networks for spatial pattern recognition
* **LSTM**: Long Short-Term Memory networks for temporal sequences
* **GRU**: Gated Recurrent Units for efficient temporal modeling
* **Transformer**: Attention-based architectures for complex patterns

See the dedicated :doc:`neural_network_factory` page for complete API documentation, architecture details, and usage examples.

High Performance Estimators
---------------------------

.. note::
   The unified estimators automatically select optimal computation frameworks (JAX, Numba, or NumPy)
   based on data characteristics and hardware availability. For advanced users, the high-performance
   modules in ``lrdbenchmark.analysis.high_performance.jax`` and ``lrdbenchmark.analysis.high_performance.numba``
   provide direct access to optimized implementations (e.g., ``DFAEstimatorJAX``, ``DFAEstimatorNumba``),
   but the unified estimators are recommended for most use cases as they automatically select the best
   framework based on data characteristics and hardware availability.

Backend Modules (Strategy Pattern)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following backend modules provide modular implementations for each estimator family,
enabling JAX GPU acceleration or Numba JIT compilation:

**Temporal Backends**

* ``lrdbenchmark.analysis.temporal.dfa_backends``: NumPy, JAX, Numba implementations for DFA.
* ``lrdbenchmark.analysis.temporal.rs_backends``: NumPy, JAX, Numba implementations for R/S.

**Spectral Backends**

* ``lrdbenchmark.analysis.spectral.spectral_backends``: NumPy, JAX implementations for Periodogram, Welch, Whittle.

**Wavelet Backends**

* ``lrdbenchmark.analysis.wavelet.wavelet_backends``: NumPy, JAX implementations for DWT-based variance estimators.

**Multifractal Backends**

* ``lrdbenchmark.analysis.multifractal.mfdfa_backends``: NumPy, JAX implementations for MFDFA.
* ``lrdbenchmark.analysis.multifractal.wavelet_leaders_backends``: NumPy, JAX implementations for Wavelet Leaders.

**Usage Example (Backend Selection)**

.. code-block:: python

   from lrdbenchmark import DFAEstimator

   # Auto-select best backend (default)
   estimator = DFAEstimator(use_optimization='auto')

   # Force JAX for GPU acceleration
   estimator_jax = DFAEstimator(use_optimization='jax')

   # Force NumPy for maximum compatibility
   estimator_numpy = DFAEstimator(use_optimization='numpy')


Usage Examples
--------------

Basic Estimator Usage
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import DFAEstimator
   from lrdbenchmark import GPHEstimator
   from lrdbenchmark import FBMModel
   
   # Generate test data with known Hurst parameter
   model = FBMModel(H=0.7, sigma=1.0)
   data = model.generate(1000, seed=42)
   
   print(f"Generated FBM data with true H = 0.7")
   print(f"Data length: {len(data)}")
   print(f"Data mean: {data.mean():.3f}, std: {data.std():.3f}")
   
   # Use DFA estimator
   dfa = DFAEstimator()
   H_dfa = dfa.estimate(data)
   print(f"DFA H estimate: {H_dfa:.3f}")
   print(f"DFA error: {abs(H_dfa - 0.7):.3f}")
   
   # Use GPH estimator
   gph = GPHEstimator()
   H_gph = gph.estimate(data)
   print(f"GPH H estimate: {H_gph:.3f}")
   print(f"GPH error: {abs(H_gph - 0.7):.3f}")
   
   # Compare estimates
   print(f"\nEstimate comparison:")
   print(f"True H: 0.700")
   print(f"DFA:    {H_dfa:.3f} (error: {abs(H_dfa - 0.7):.3f})")
   print(f"GPH:    {H_gph:.3f} (error: {abs(H_gph - 0.7):.3f})")

Multiple Estimators Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import DFAEstimator
   from lrdbenchmark import RSEstimator
   from lrdbenchmark import GPHEstimator
   from lrdbenchmark import WaveletVarianceEstimator
   from lrdbenchmark import HiguchiEstimator
   from lrdbenchmark import FBMModel, FGNModel
   import pandas as pd
   
   # Define estimators to test
   estimators = {
       'DFA': DFAEstimator(),
       'R/S': RSEstimator(),
       'GPH': GPHEstimator(),
       'Wavelet Variance': WaveletVarianceEstimator(),
       'Higuchi': HiguchiEstimator()
   }
   
   # Test on different data models
   test_cases = {
       'FBM (H=0.7)': FBMModel(H=0.7, sigma=1.0),
       'FBM (H=0.3)': FBMModel(H=0.3, sigma=1.0),
       'FGN (H=0.8)': FGNModel(H=0.8, sigma=1.0)
   }
   
   # Store results
   all_results = []
   
   for case_name, model in test_cases.items():
       print(f"\n=== Testing {case_name} ===")
       data = model.generate(1000, seed=42)
       true_H = model.H
       
       case_results = {'Case': case_name, 'True_H': true_H}
       
       for name, estimator in estimators.items():
           try:
               H_est = estimator.estimate(data)
               error = abs(H_est - true_H)
               case_results[name] = H_est
               case_results[f'{name}_error'] = error
               print(f"  {name}: H = {H_est:.3f} (error: {error:.3f})")
           except Exception as e:
               print(f"  {name}: Error - {e}")
               case_results[name] = None
               case_results[f'{name}_error'] = None
       
       all_results.append(case_results)
   
   # Create summary DataFrame
   df = pd.DataFrame(all_results)
   print(f"\n=== SUMMARY ===")
   print(df.round(3))
   
   # Calculate average errors
   error_columns = [col for col in df.columns if col.endswith('_error')]
   avg_errors = df[error_columns].mean()
   print(f"\n=== AVERAGE ERRORS ===")
   for col in error_columns:
       estimator = col.replace('_error', '')
       print(f"{estimator}: {avg_errors[col]:.3f}")

Machine Learning Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import RandomForestEstimator
   from lrdbenchmark import GradientBoostingEstimator
   from lrdbenchmark import CNNEstimator
   from lrdbenchmark import FBMModel, FGNModel, ARFIMAModel
   import numpy as np
   from sklearn.model_selection import train_test_split
   
   # Generate comprehensive training dataset
   print("Generating training dataset...")
   training_data = []
   training_labels = []
   
   # Create diverse training data
   H_values = np.linspace(0.3, 0.9, 15)  # 15 different H values
   models = {
       'FBM': FBMModel,
       'FGN': FGNModel,
       'ARFIMA': lambda H: ARFIMAModel(d=H-0.5, p=1, q=1)
   }
   
   for H in H_values:
       for model_name, model_class in models.items():
           if model_name == 'ARFIMA':
               model = model_class(H)
           else:
               model = model_class(H=H, sigma=1.0)
           
           # Generate multiple realizations
           for i in range(20):
               data = model.generate(1000, seed=int(H*1000 + i))
               training_data.append(data)
               training_labels.append(H)
   
   print(f"Generated {len(training_data)} training samples")
   print(f"H range: {min(training_labels):.1f} to {max(training_labels):.1f}")
   
   # Split into training and validation sets
   X_train, X_val, y_train, y_val = train_test_split(
       training_data, training_labels, test_size=0.2, random_state=42
   )
   
   # Train Random Forest estimator
   print("\nTraining Random Forest estimator...")
   rf_estimator = RandomForestEstimator(
       n_estimators=100,
       max_depth=10,
       random_state=42
   )
   rf_estimator.fit(X_train, y_train)
   
   # Train Gradient Boosting estimator
   print("Training Gradient Boosting estimator...")
   gb_estimator = GradientBoostingEstimator(
       n_estimators=100,
       learning_rate=0.1,
       max_depth=5,
       random_state=42
   )
   gb_estimator.fit(X_train, y_train)
   
   # Evaluate on validation set
   print("\n=== Validation Results ===")
   rf_val_pred = rf_estimator.estimate(X_val)
   gb_val_pred = gb_estimator.estimate(X_val)
   
   rf_mae = np.mean(np.abs(np.array(rf_val_pred) - np.array(y_val)))
   gb_mae = np.mean(np.abs(np.array(gb_val_pred) - np.array(y_val)))
   
   print(f"Random Forest MAE: {rf_mae:.3f}")
   print(f"Gradient Boosting MAE: {gb_mae:.3f}")
   
   # Test on new data
   print("\n=== Test on New Data ===")
   test_cases = [
       ('FBM (H=0.6)', FBMModel(H=0.6, sigma=1.0)),
       ('FGN (H=0.4)', FGNModel(H=0.4, sigma=1.0)),
       ('ARFIMA (H=0.75)', ARFIMAModel(d=0.25, p=1, q=1))
   ]
   
   for test_name, test_model in test_cases:
       test_data = test_model.generate(1000, seed=999)
       
       H_rf = rf_estimator.estimate([test_data])[0]
       H_gb = gb_estimator.estimate([test_data])[0]
       
       if 'H=' in test_name:
           true_H = float(test_name.split('H=')[1].split(')')[0])
       else:
           true_H = 0.75  # For ARFIMA
       
       print(f"{test_name}:")
       print(f"  True H: {true_H:.3f}")
       print(f"  RF estimate: {H_rf:.3f} (error: {abs(H_rf - true_H):.3f})")
       print(f"  GB estimate: {H_gb:.3f} (error: {abs(H_gb - true_H):.3f})")

High Performance Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import DFAEstimator, RSEstimator, FBMModel
   import numpy as np
   
   # Generate large dataset
   model = FBMModel(H=0.7, sigma=1.0)
   data = model.generate(10000, seed=42)
   
   # Unified estimators automatically use optimal backend (JAX/Numba/NumPy)
   dfa = DFAEstimator(use_optimization='auto')  # Auto-selects best framework
   result = dfa.estimate(data)
   print(f"DFA H estimate: {result['hurst_parameter']:.3f}")
   
   # R/S estimator with automatic optimization
   rs = RSEstimator(use_optimization='auto')
   result = rs.estimate(data)
   print(f"R/S H estimate: {result['hurst_parameter']:.3f}")

Parameter Tuning
~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import DFAEstimator
   from lrdbenchmark import GPHEstimator
   from lrdbenchmark import FBMModel
   
   # Generate test data
   model = FBMModel(H=0.7, sigma=1.0)
   data = model.generate(1000, seed=42)
   
   # DFA with custom parameters
   dfa = DFAEstimator(
       min_scale=4,
       max_scale=100,
       num_scales=20,
       polynomial_order=2
   )
   H_dfa = dfa.estimate(data)
   
   # GPH with custom parameters
   gph = GPHEstimator(
       num_frequencies=50,
       min_frequency=0.01,
       max_frequency=0.5
   )
   H_gph = gph.estimate(data)
   
   print(f"DFA (custom): H = {H_dfa:.3f}")
   print(f"GPH (custom): H = {H_gph:.3f}")

# Note: All estimators are documented above in their respective sections
# No duplicate documentation needed

Error Handling
--------------

.. code-block:: python

   from lrdbenchmark import DFAEstimator
   from lrdbenchmark import GPHEstimator
   
   # Test with insufficient data
   short_data = [1, 2, 3, 4, 5]  # Too short for most estimators
   
   dfa = DFAEstimator()
   try:
       H_dfa = dfa.estimate(short_data)
       print(f"DFA H estimate: {H_dfa:.3f}")
   except ValueError as e:
       print(f"DFA error: {e}")
   
   gph = GPHEstimator()
   try:
       H_gph = gph.estimate(short_data)
       print(f"GPH H estimate: {H_gph:.3f}")
   except ValueError as e:
       print(f"GPH error: {e}")

Performance Comparison
----------------------

.. code-block:: python

   import time
   from lrdbenchmark import DFAEstimator, FBMModel
   
   # Generate test data
   model = FBMModel(H=0.7, sigma=1.0)
   data = model.generate(5000, seed=42)
   
   # DFA with automatic optimization (will use JAX/Numba if available)
   dfa = DFAEstimator(use_optimization='auto')
   start_time = time.time()
   result = dfa.estimate(data)
   dfa_time = time.time() - start_time
   
   # DFA with explicit NumPy backend (no optimization)
   dfa_numpy = DFAEstimator(use_optimization='numpy')
   start_time = time.time()
   result_numpy = dfa_numpy.estimate(data)
   dfa_numpy_time = time.time() - start_time
   
   print(f"Optimized DFA: H = {result['hurst_parameter']:.3f}, Time = {dfa_time:.4f}s")
   print(f"NumPy DFA: H = {result_numpy['hurst_parameter']:.3f}, Time = {dfa_numpy_time:.4f}s")
   if dfa_numpy_time > 0:
       print(f"Speedup: {dfa_numpy_time/dfa_time:.2f}x")

Best Practices
--------------

1. **Data Length**: Use at least 1000 samples for reliable estimates
2. **Parameter Selection**: Choose appropriate scale ranges for your data
3. **Multiple Estimators**: Compare results from different estimator types
4. **Error Handling**: Always handle potential estimation errors
5. **Performance**: Use high-performance estimators for large datasets
6. **Validation**: Test on synthetic data with known Hurst parameters

.. note::
   Different estimators may give slightly different results due to their
   underlying assumptions and methodologies. It's recommended to use
   multiple estimators and compare their results.

.. warning::
   Some estimators require specific data characteristics (e.g., stationarity
   for spectral methods). Always check the estimator documentation for
   requirements and limitations.
