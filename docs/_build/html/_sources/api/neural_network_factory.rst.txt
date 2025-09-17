Neural Network Factory API
===========================

The Neural Network Factory provides a comprehensive framework for creating and managing various neural network architectures for Hurst parameter estimation. This module implements train-once, apply-many workflows with proper GPU memory management and model persistence.

.. automodule:: lrdbenchmark.analysis.machine_learning.neural_network_factory
   :members:

Neural Network Architectures
-----------------------------

The factory supports 8 different neural network architectures:

Feedforward Network
~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.machine_learning.neural_network_factory.FeedforwardNetwork
   :members:

Convolutional Network
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.machine_learning.neural_network_factory.ConvolutionalNetwork
   :members:

LSTM Network
~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.machine_learning.neural_network_factory.LSTMNetwork
   :members:

Bidirectional LSTM Network
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.machine_learning.neural_network_factory.BidirectionalLSTMNetwork
   :members:

GRU Network
~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.machine_learning.neural_network_factory.GRUNetwork
   :members:

Transformer Network
~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.machine_learning.neural_network_factory.TransformerNetwork
   :members:

ResNet Network
~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.machine_learning.neural_network_factory.ResNetNetwork
   :members:

Hybrid CNN-LSTM Network
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.analysis.machine_learning.neural_network_factory.HybridCNNLSTMNetwork
   :members:

Factory Class
--------------

.. autoclass:: lrdbenchmark.analysis.machine_learning.neural_network_factory.NeuralNetworkFactory
   :members:

Configuration
--------------

.. autoclass:: lrdbenchmark.analysis.machine_learning.neural_network_factory.NNConfig
   :members:

Architecture Enumeration
------------------------

.. autoclass:: lrdbenchmark.analysis.machine_learning.neural_network_factory.NNArchitecture
   :members:

Convenience Functions
---------------------

.. autofunction:: lrdbenchmark.analysis.machine_learning.neural_network_factory.create_feedforward_network

.. autofunction:: lrdbenchmark.analysis.machine_learning.neural_network_factory.create_cnn_network

.. autofunction:: lrdbenchmark.analysis.machine_learning.neural_network_factory.create_lstm_network

.. autofunction:: lrdbenchmark.analysis.machine_learning.neural_network_factory.create_all_benchmark_networks

Usage Examples
--------------

Creating a Single Network
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.analysis.machine_learning.neural_network_factory import (
       NeuralNetworkFactory, NNArchitecture, NNConfig
   )
   
   # Create configuration
   config = NNConfig(
       architecture=NNArchitecture.TRANSFORMER,
       input_length=500,
       hidden_dims=[64, 32],
       learning_rate=0.001,
       epochs=50
   )
   
   # Create network
   factory = NeuralNetworkFactory()
   network = factory.create_network(config)

Creating All Benchmark Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.analysis.machine_learning.neural_network_factory import create_all_benchmark_networks
   
   # Create all available networks
   networks = create_all_benchmark_networks(input_length=500)
   
   for name, network in networks.items():
       print(f"Created {name} network")

Training and Prediction
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Generate training data
   X_train = np.random.randn(100, 500)
   y_train = np.random.uniform(0.2, 0.8, 100)
   
   # Train the network
   history = network.train_model(X_train, y_train)
   
   # Make predictions
   new_data = np.random.randn(1, 500)
   prediction = network.predict(new_data)
   
   print(f"Prediction: {prediction[0]:.3f}")

Model Persistence
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Models are automatically saved after training
   # To load a saved model:
   network.load_model()  # Loads from default path
   
   # Or specify a custom path:
   network.load_model("path/to/model.pth")

Performance Characteristics
---------------------------

Based on comprehensive benchmarking, the neural network architectures show the following performance:

+------------------+-----------+----------------+------------------+
| Architecture     | MAE       | Execution Time | Success Rate     |
+==================+===========+================+==================+
| Transformer      | 0.1802    | 0.7ms          | 100%             |
+------------------+-----------+----------------+------------------+
| LSTM             | 0.1833    | 0.3ms          | 100%             |
+------------------+-----------+----------------+------------------+
| Bidirectional    | 0.1834    | 0.3ms          | 100%             |
| LSTM             |           |                |                  |
+------------------+-----------+----------------+------------------+
| Convolutional    | 0.1844    | 0.0ms          | 100%             |
+------------------+-----------+----------------+------------------+
| GRU              | 0.1849    | 0.2ms          | 100%             |
+------------------+-----------+----------------+------------------+
| ResNet           | 0.1859    | 0.1ms          | 100%             |
+------------------+-----------+----------------+------------------+
| Feedforward      | 0.1946    | 0.0ms          | 100%             |
+------------------+-----------+----------------+------------------+

Key Features
------------

* **Train-once, Apply-many**: Models are trained once and can be used for multiple predictions
* **GPU Memory Management**: Batch processing prevents CUDA out-of-memory issues
* **Model Persistence**: Automatic saving and loading of trained models
* **Device Management**: Automatic GPU/CPU selection and device placement
* **Consistent Performance**: All architectures achieve similar accuracy levels
* **Fast Inference**: Ultra-fast prediction times (0.0-0.7ms per sample)
* **Production Ready**: Robust error handling and memory management
