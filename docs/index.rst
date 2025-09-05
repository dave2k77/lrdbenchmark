.. LRDBench documentation master file, created by
   sphinx-quickstart on Sun Aug 25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LRDBenchmark's documentation!
========================================

.. image:: https://img.shields.io/pypi/v/lrdbenchmark.svg
   :target: https://pypi.org/project/lrdbenchmark/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/lrdbenchmark.svg
   :target: https://pypi.org/project/lrdbenchmark/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://readthedocs.org/projects/lrdbenchmark/badge/?version=latest
   :target: https://lrdbenchmark.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

**LRDBenchmark** is a comprehensive benchmarking framework for long-range dependence (LRD) analysis in time series data. It provides a unified platform for evaluating and comparing various estimators and models for detecting and quantifying long-range dependence patterns.

🏆 **Latest Results: Comprehensive Three-Way Comparison - Classical vs ML vs Neural Networks!**

Our latest comprehensive three-way benchmark shows:
- **R/S (Classical): Best individual performance** (0.0997 MAE)
- **Neural Networks: Excellent speed-accuracy trade-offs** (0.1802-0.1946 MAE, 0.0-0.7ms execution time)
- **17 estimators tested**: 7 Classical, 3 ML, 7 Neural Network approaches
- **88.2% overall success rate** across all approaches
- **Neural Network Factory**: 8 architectures with train-once, apply-many workflows
- **Production-ready systems** with GPU memory management and model persistence

Key Features
------------

* **Neural Network Factory**: 8 architectures (FFN, CNN, LSTM, GRU, Transformer, ResNet, etc.) with train-once, apply-many workflows
* **Three-Way Comparison**: Classical, ML, and Neural Network approaches comprehensively benchmarked
* **Best Individual Performance**: R/S (Classical) with 0.0997 MAE
* **Neural Network Excellence**: Consistent high performance (0.1802-0.1946 MAE) with ultra-fast inference (0.0-0.7ms)
* **Production-Ready ML Models**: SVR, Gradient Boosting, Random Forest with 50-70 engineered features
* **Comprehensive Classical Estimators**: 7 adaptive estimators with automatic optimization framework selection
* **Intelligent Backend System**: Automatic GPU/JAX, CPU/Numba, or NumPy selection based on data characteristics
* **GPU Memory Management**: Batch processing and memory optimization for neural networks
* **Model Persistence**: Automatic model saving and loading for production deployment
* **Robust Error Handling**: Adaptive parameter selection and progressive fallback mechanisms
* **EEG Contamination Testing**: 8 realistic artifact scenarios for biomedical applications
* **Multiple Data Models**: FBM, FGN, ARFIMA, MRW with configurable parameters
* **High Performance**: GPU-accelerated implementations with JAX and Numba backends
* **Analytics System**: Built-in usage tracking and performance monitoring
* **Extensible Architecture**: Easy integration of new estimators and models
* **Production Ready**: Pre-trained models and comprehensive testing
* **Research Ready**: Publication-quality results with mathematical verification

Quick Start
-----------

Install with `pip install lrdbenchmark` and see the Quick Start Guide for detailed examples.

Installation & Setup
--------------------

.. toctree::
   :maxdepth: 2

   installation
   quickstart

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api/machine_learning_estimators
   api/neural_network_factory
   api/adaptive_estimators
   api/contamination_factory
   api/estimators
   api/data_models
   api/benchmark
   api/analytics

Research & Theory
-----------------

.. toctree::
   :maxdepth: 2

   research/theory
   research/validation

Examples & Demos
----------------

.. toctree::
   :maxdepth: 2

   examples/comprehensive_adaptive_demo
   examples/comprehensive_demo

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

