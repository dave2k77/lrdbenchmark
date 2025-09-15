Quick Start Guide
================

This guide will get you up and running with lrdbenchmark in minutes.

Basic Usage
-----------

Generate synthetic data and run a benchmark:

.. code-block:: python

   import numpy as np
   from lrdbenchmark import FBMModel, RSEstimator
   
   # Generate Fractional Brownian Motion data
   model = FBMModel(H=0.7, sigma=1.0)
   data = model.generate(1000, seed=42)
   
   # Estimate Hurst parameter using R/S analysis
   rs_estimator = RSEstimator()
   hurst_estimate = rs_estimator.estimate(data)
   
   print(f"True Hurst: 0.7, Estimated: {hurst_estimate:.3f}")

Neural Network Usage
--------------------

LRDBenchmark provides a comprehensive neural network factory with 4 architectures that achieve excellent speed-accuracy trade-offs:

.. code-block:: python

   from lrdbenchmark.analysis.machine_learning.neural_network_factory import (
       NeuralNetworkFactory, NNArchitecture, NNConfig, create_all_benchmark_networks
   )
   import numpy as np

   # Create neural network factory
   factory = NeuralNetworkFactory()

   # Create a specific network
   config = NNConfig(
       architecture=NNArchitecture.TRANSFORMER,
       input_length=500,
       hidden_dims=[64, 32],
       learning_rate=0.001,
       epochs=50
   )
   network = factory.create_network(config)

   # Generate training data
   X_train = np.random.randn(100, 500)  # 100 samples of length 500
   y_train = np.random.uniform(0.2, 0.8, 100)  # True Hurst parameters

   # Train the network (train-once, apply-many workflow)
   history = network.train_model(X_train, y_train)

   # Make predictions on new data
   new_data = np.random.randn(1, 500)
   prediction = network.predict(new_data)

   print(f"Neural Network Prediction: {prediction[0]:.3f}")

   # Create all benchmark networks
   all_networks = create_all_benchmark_networks(input_length=500)
   for name, network in all_networks.items():
       print(f"Created {name} network")

Machine Learning Usage
----------------------

LRDBenchmark provides production-ready machine learning estimators:

.. code-block:: python

   from lrdbenchmark.analysis.machine_learning.svr_estimator import SVREstimator
   from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator import GradientBoostingEstimator
   from lrdbenchmark.analysis.machine_learning.random_forest_estimator import RandomForestEstimator
   import numpy as np

   # Generate training data
   X_train = np.random.randn(100, 500)  # 100 samples of length 500
   y_train = np.random.uniform(0.2, 0.8, 100)  # True Hurst parameters

   # Train ML models
   svr = SVREstimator(kernel='rbf', C=1.0)
   svr.train(X_train, y_train)

   gb = GradientBoostingEstimator(n_estimators=50, learning_rate=0.1)
   gb.train(X_train, y_train)

   rf = RandomForestEstimator(n_estimators=50, max_depth=5)
   rf.train(X_train, y_train)

   # Make predictions on new data
   new_data = np.random.randn(1, 500)
   svr_pred = svr.predict(new_data)
   gb_pred = gb.predict(new_data)
   rf_pred = rf.predict(new_data)

   print(f"SVR: {svr_pred:.3f}, Gradient Boosting: {gb_pred:.3f}, Random Forest: {rf_pred:.3f}")

Production ML System
~~~~~~~~~~~~~~~~~~~~

For production deployment, use the production ML system with train-once, apply-many workflow:

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

   print(f"CNN Prediction: {prediction.hurst_parameter:.3f}")

Data Models
-----------

LRDBenchmark provides several synthetic data models:

.. code-block:: python

   from lrdbenchmark import FBMModel, FGNModel, ARFIMAModel, MRWModel
   
   # Fractional Brownian Motion
   fbm = FBMModel(H=0.7, sigma=1.0)
   fbm_data = fbm.generate(1000)
   
   # Fractional Gaussian Noise
   fgn = FGNModel(H=0.6, sigma=1.0)
   fgn_data = fgn.generate(1000)
   
   # ARFIMA process
   arfima = ARFIMAModel(d=0.3, sigma=1.0)
   arfima_data = arfima.generate(1000)
   
   # Multifractal Random Walk
   mrw = MRWModel(H=0.7, lambda_param=0.1, sigma=1.0)
   mrw_data = mrw.generate(1000)

Individual Estimators
---------------------

Use specific estimators directly:

.. code-block:: python

   from lrdbenchmark.analysis.temporal.dfa.dfa_estimator import DFAEstimator
   from lrdbenchmark.analysis.spectral.gph.gph_estimator import GPHEstimator
   
   # Detrended Fluctuation Analysis
   dfa = DFAEstimator()
   H_dfa = dfa.estimate(data)
   
   # Geweke-Porter-Hudak estimator
   gph = GPHEstimator()
   H_gph = gph.estimate(data)
   
   print(f"DFA H estimate: {H_dfa:.3f}")
   print(f"GPH H estimate: {H_gph:.3f}")

Analytics System
----------------

Track usage and performance:

.. code-block:: python

   from lrdbenchmark import FBMModel, RSEstimator
   
   # Generate data and run analysis
   model = FBMModel(H=0.7)
   data = model.generate(1000)
   
   # Estimate Hurst parameter
   rs_estimator = RSEstimator()
   hurst_estimate = rs_estimator.estimate(data)
   
   print(f"Hurst estimate: {hurst_estimate:.3f}")

Enhanced ML and Neural Network Estimators
-----------------------------------------

Use the new enhanced estimators with pre-trained models:

.. code-block:: python

   from lrdbenchmark import (
       CNNEstimator, LSTMEstimator, GRUEstimator, TransformerEstimator,
       RandomForestEstimator, SVREstimator, GradientBoostingEstimator
   )
   
   # Enhanced CNN with residual connections and attention
   cnn = CNNEstimator()
   H_cnn = cnn.estimate(data)
   
   # Enhanced LSTM with bidirectional architecture
   lstm = LSTMEstimator()
   H_lstm = lstm.estimate(data)
   
   # Enhanced GRU with attention mechanisms
   gru = GRUEstimator()
   H_gru = gru.estimate(data)
   
   # Enhanced Transformer with self-attention
   transformer = TransformerEstimator()
   H_transformer = transformer.estimate(data)
   
   # Traditional ML estimators
   rf = RandomForestEstimator()
   H_rf = rf.estimate(data)
   
   svr = SVREstimator()
   H_svr = svr.estimate(data)
   
   gb = GradientBoostingEstimator()
   H_gb = gb.estimate(data)
   
   print(f"CNN H estimate: {H_cnn:.3f}")
   print(f"LSTM H estimate: {H_lstm:.3f}")
   print(f"GRU H estimate: {H_gru:.3f}")
   print(f"Transformer H estimate: {H_transformer:.3f}")

Advanced Usage
--------------

Custom benchmark configuration:

.. code-block:: python

   from lrdbenchmark import FBMModel, RSEstimator, DFAEstimator
   
   # Generate data
   model = FBMModel(H=0.7)
   data = model.generate(2000)
   
   # Test multiple estimators
   rs_estimator = RSEstimator()
   dfa_estimator = DFAEstimator()
   
   rs_hurst = rs_estimator.estimate(data)
   dfa_hurst = dfa_estimator.estimate(data)
   
   print(f"R/S estimate: {rs_hurst:.3f}")
   print(f"DFA estimate: {dfa_hurst:.3f}")

Integration with HPFracc
------------------------

Compare with fractional neural networks:

.. code-block:: python

   # This requires hpfracc to be installed
   try:
       from scripts.hpfracc_proper_benchmark import HPFraccProperBenchmark
       
       # Create benchmark
       benchmark = HPFraccProperBenchmark(
           series_length=1000,
           batch_size=32,
           input_window=10,
           prediction_horizon=1
       )
       
       # Run comparison
       results = benchmark.run_benchmark()
       
       # Generate report
       report = benchmark.generate_report()
       print(report)
       
   except ImportError:
       print("HPFracc not available. Install with: pip install hpfracc")

Visualization
-------------

Plot results and data:

.. code-block:: python

   import matplotlib.pyplot as plt
   from lrdbenchmark import FBMModel
   
   # Generate data with different H values
   H_values = [0.3, 0.5, 0.7, 0.9]
   datasets = {}
   
   for H in H_values:
       model = FBMModel(H=H, sigma=1.0)
       datasets[f'H={H}'] = model.generate(1000)
   
   # Plot
   plt.figure(figsize=(12, 8))
   for name, data in datasets.items():
       plt.plot(data[:200], label=name, alpha=0.7)
   
   plt.title('Fractional Brownian Motion with Different H Values')
   plt.xlabel('Time')
   plt.ylabel('Value')
   plt.legend()
   plt.grid(True)
   plt.show()

Performance Tips
----------------

1. **Use GPU acceleration** when available
2. **Batch processing** for large datasets
3. **Enable analytics** for monitoring
4. **Use appropriate data lengths** (1000+ samples recommended)

Next Steps
-----------

* :doc:`installation` - Detailed installation guide
* :doc:`api/data_models` - Learn about data models
* :doc:`api/estimators` - Explore available estimators
* :doc:`examples/comprehensive_demo` - More examples and use cases
