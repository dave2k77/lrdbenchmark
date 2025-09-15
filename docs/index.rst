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
- **LSTM (Neural Networks): Best individual performance** (0.097 MAE)
- **Neural Networks: Dominate top 4 positions** (0.097-0.108 MAE, 0.0007-0.0064s execution time)
- **15 estimators tested**: 8 Classical, 3 ML, 4 Neural Network approaches
- **100% overall success rate** across all approaches
- **Neural Network Factory**: 4 architectures with unified implementation
- **Production-ready systems** with perfect robustness and train-once, apply-many workflows

🔥 **Heavy-Tail Robustness: Exceptional Performance on Alpha-Stable Data!**

Our comprehensive heavy-tail analysis reveals:
- **Machine Learning Dominance**: 0.208 mean error (GradientBoosting: 0.201 MAE)
- **Neural Network Excellence**: 0.247 mean error (LSTM: 0.245 MAE)
- **Classical Reliability**: 0.409 mean error (DFA: 0.346 MAE)
- **Perfect Robustness**: 100% success rate on extreme heavy-tail data (α=0.8)
- **440 test scenarios**: Alpha-stable distributions from Gaussian (α=2.0) to extreme heavy-tailed (α=0.8)
- **Adaptive Preprocessing**: Intelligent handling of all heavy-tail characteristics

Key Features
------------

* **Neural Network Factory**: 4 architectures (CNN, LSTM, GRU, Transformer) with unified implementation
* **Three-Way Comparison**: Classical, ML, and Neural Network approaches comprehensively benchmarked
* **Best Individual Performance**: LSTM (Neural Networks) with 0.097 MAE
* **Neural Network Excellence**: Dominate top 4 positions (0.097-0.108 MAE) with ultra-fast inference (0.0007-0.0064s)
* **Production-Ready ML Models**: SVR, Gradient Boosting, Random Forest with train-once, apply-many workflows
* **Comprehensive Classical Estimators**: 8 adaptive estimators with automatic optimization framework selection
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

