Machine Learning Estimators
============================

LRDBenchmark provides production-ready machine learning estimators for Long-Range Dependence (LRD) estimation. These estimators achieve **excellent performance** with perfect robustness, with **Gradient Boosting achieving the best ML performance** at 0.193 MAE.

Overview
--------

The machine learning estimators use advanced feature engineering with 50-70 engineered features per model, including:

* **Statistical Features**: Mean, standard deviation, skewness, kurtosis
* **Time Series Features**: Autocorrelation at multiple lags, variance of increments
* **Spectral Features**: Power spectrum analysis, frequency band ratios, spectral slope
* **DFA Features**: Detrended fluctuation analysis with slope calculation
* **Wavelet Features**: Wavelet variance at different scales, wavelet slope
* **R/S Analysis Features**: Rescaled range analysis with slope calculation
* **Additional Features**: Trend analysis, seasonality detection, entropy measures

SVR Estimator
-------------

Support Vector Regression estimator with RBF kernel and comprehensive feature engineering.

.. autoclass:: lrdbenchmark.analysis.machine_learning.svr_estimator.SVREstimator
   :members:
   :undoc-members:
   :show-inheritance:

**Performance**: 0.202 MAE, 100% success rate, 0.009s execution time

**Key Features**:
* RBF kernel with configurable parameters (C, gamma, epsilon)
* 50+ engineered features including spectral and DFA analysis
* Model persistence with save/load functionality
* Robust error handling with fallback to R/S analysis

**Example Usage**:

.. code-block:: python

   from lrdbenchmark import SVREstimator
   import numpy as np

   # Initialize estimator
   svr = SVREstimator(kernel='rbf', C=1.0, gamma='scale')

   # Generate training data
   X_train = np.random.randn(100, 500)
   y_train = np.random.uniform(0.2, 0.8, 100)

   # Train model
   svr.train(X_train, y_train)

   # Make prediction
   new_data = np.random.randn(1, 500)
   prediction = svr.predict(new_data)

Gradient Boosting Estimator
---------------------------

Gradient Boosting Regressor with comprehensive feature engineering - **Best Overall Performance**.

.. autoclass:: lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator.GradientBoostingEstimator
   :members:
   :undoc-members:
   :show-inheritance:

**Performance**: 0.193 MAE (**Best ML**), 100% success rate, 0.013s execution time

**Key Features**:
* Configurable parameters (n_estimators, learning_rate, max_depth)
* 60+ engineered features including advanced spectral and DFA analysis
* Feature importance analysis
* Model persistence with save/load functionality
* Robust error handling with fallback to R/S analysis

**Example Usage**:

.. code-block:: python

   from lrdbenchmark import GradientBoostingEstimator
   import numpy as np

   # Initialize estimator
   gb = GradientBoostingEstimator(n_estimators=50, learning_rate=0.1)

   # Generate training data
   X_train = np.random.randn(100, 500)
   y_train = np.random.uniform(0.2, 0.8, 100)

   # Train model
   gb.train(X_train, y_train)

   # Make prediction
   new_data = np.random.randn(1, 500)
   prediction = gb.predict(new_data)

   # Get feature importance
   importance = gb.get_feature_importance()

Random Forest Estimator
-----------------------

Random Forest Regressor with comprehensive feature engineering and feature importance analysis.

.. autoclass:: lrdbenchmark.analysis.machine_learning.random_forest_estimator.RandomForestEstimator
   :members:
   :undoc-members:
   :show-inheritance:

**Performance**: 0.202 MAE, 100% success rate, 2.099s execution time

**Key Features**:
* Configurable parameters (n_estimators, max_depth, min_samples_split)
* 70+ engineered features including fractal dimension and approximate entropy
* Feature importance analysis
* Model persistence with save/load functionality
* Robust error handling with fallback to R/S analysis

**Example Usage**:

.. code-block:: python

   from lrdbenchmark import RandomForestEstimator
   import numpy as np

   # Initialize estimator
   rf = RandomForestEstimator(n_estimators=50, max_depth=5)

   # Generate training data
   X_train = np.random.randn(100, 500)
   y_train = np.random.uniform(0.2, 0.8, 100)

   # Train model
   rf.train(X_train, y_train)

   # Make prediction
   new_data = np.random.randn(1, 500)
   prediction = rf.predict(new_data)

   # Get feature importance
   importance = rf.get_feature_importance()

Production ML System
--------------------

Production-ready ML system with train-once, apply-many workflow and intelligent framework selection.

.. autoclass:: lrdbenchmark.analysis.machine_learning.production_ml_system.ProductionMLSystem
   :members:
   :undoc-members:
   :show-inheritance:

**Key Features**:
* Intelligent framework selection (JAX, PyTorch, Numba)
* Train-once, apply-many workflow
* Model persistence and caching
* Batch prediction capabilities
* Production-ready deployment

**Example Usage**:

.. code-block:: python

   from lrdbenchmark.analysis.machine_learning.production_ml_system import ProductionMLSystem, ProductionConfig
   import numpy as np

   # Configure system
   config = ProductionConfig(
       model_type="cnn",
       input_length=500,
       hidden_dims=[64, 32],
       learning_rate=0.001,
       epochs=20
   )

   # Initialize system
   system = ProductionMLSystem(config)

   # Generate training data
   X_train = np.random.randn(100, 500)
   y_train = np.random.uniform(0.2, 0.8, 100)

   # Train model
   system.train(X_train, y_train)

   # Make prediction
   new_data = np.random.randn(1, 500)
   prediction = system.predict(new_data)

Performance Comparison
----------------------

| Method | Mean Error | Execution Time | Success Rate | Category |
|--------|------------|----------------|--------------|----------|
| **LSTM** | **0.097** | 0.0012s | 100% | Neural Networks |
| **CNN** | **0.103** | 0.0064s | 100% | Neural Networks |
| **Transformer** | **0.106** | 0.0026s | 100% | Neural Networks |
| **GRU** | **0.108** | 0.0007s | 100% | Neural Networks |
| **R/S** | **0.099** | 0.348s | 100% | Classical |
| **GradientBoosting** | **0.193** | 0.013s | 100% | ML |
| **SVR** | **0.202** | 0.009s | 100% | ML |
| **Whittle** | 0.200 | 0.0002s | 100% | Classical |
| **Periodogram** | 0.205 | 0.0005s | 100% | Classical |
| **CWT** | 0.269 | 0.063s | 100% | Classical |

Key Advantages
--------------

* **Excellent Performance**: Strong performance with perfect robustness
* **Advanced Feature Engineering**: 50-70 engineered features per model
* **Production Ready**: Model persistence, error handling, and deployment capabilities
* **Comprehensive Testing**: 100% success rate across all test cases
* **Research Quality**: Publication-ready results with detailed performance metrics

Best Practices
--------------

1. **For Highest Accuracy**: Use LSTM Neural Network (0.097 MAE)
2. **For Fast ML Performance**: Use SVR (0.009s execution time)
3. **For Feature Analysis**: Use Random Forest (feature importance available)
4. **For Production Deployment**: Use Production ML System with train-once, apply-many workflow
5. **For Real-time Applications**: Use GRU Neural Network (0.0007s execution time)

See Also
--------

* :doc:`../examples/comprehensive_adaptive_demo` - Complete usage examples
* :doc:`../research/theory` - Theoretical foundations
* :doc:`../research/validation` - Validation methodology
