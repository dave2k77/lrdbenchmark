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

🚀 **Comprehensive Estimator Suite**

LRDBenchmark provides state-of-the-art implementations across three categories:

- **Classical Methods**: 8+ estimators including R/S, DFA, DMA, Higuchi, GPH, Whittle, CWT, and Generalized Hurst Exponent
- **Machine Learning**: Random Forest, SVR, and Gradient Boosting with optimized hyperparameters
- **Neural Networks**: LSTM, GRU, CNN, and Transformer architectures with pre-trained models
- **Heavy-Tail Robustness**: α-stable distribution modeling with adaptive preprocessing
- **High-Performance Computing**: Intelligent backend selection (JAX → Numba → NumPy)

📚 **Demonstration Notebooks**

Comprehensive Jupyter notebooks showcase all library features:

- **Data Generation & Visualization**: All stochastic models (FBM, FGN, ARFIMA, MRW, Alpha-Stable)
- **Estimation & Validation**: All estimator categories with statistical validation
- **Custom Models & Estimators**: Library extensibility and custom implementations
- **Comprehensive Benchmarking**: Full benchmarking system with contamination testing
- **Leaderboard Generation**: Performance rankings and comparative analysis

Key Features
------------

* **Comprehensive Estimator Suite**: 15+ estimators across classical, ML, and neural network approaches
* **Unified ML Feature Engineering**: 76-feature extraction pipeline with pre-trained model support
* **Neural Network Factory**: 4 architectures (CNN, LSTM, GRU, Transformer) with automatic device selection
* **Production-Ready ML Models**: SVR (29 features), Gradient Boosting (54 features), Random Forest (76 features)
* **Heavy-Tail Robustness**: α-stable distribution modeling with adaptive preprocessing techniques
* **Intelligent Backend System**: Automatic GPU/JAX, CPU/Numba, or NumPy selection based on data characteristics
* **High-Performance Computing**: GPU-accelerated implementations with graceful fallbacks
* **Multiple Data Models**: FBM, FGN, ARFIMA, MRW, and α-stable processes with configurable parameters
* **Comprehensive Benchmarking**: End-to-end scripts with statistical analysis and confidence intervals
* **Analytics System**: Built-in usage tracking and performance monitoring
* **Extensible Architecture**: Easy integration of new estimators and models
* **Production Ready**: Pre-trained models, model persistence, and comprehensive testing

Quick Start
-----------

Install with `pip install lrdbenchmark` and see the Quick Start Guide for detailed examples.

Installation & Setup
--------------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   installation
   quickstart

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   api/machine_learning_estimators
   api/neural_network_factory
   api/adaptive_estimators
   api/contamination_factory
   api/estimators
   api/data_models
   api/benchmark
   api/analytics

Theory & Background
-------------------

For theoretical foundations and validation techniques, see the comprehensive examples in the API documentation and quickstart guide.

Examples & Demos
----------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   examples/comprehensive_adaptive_demo
   examples/comprehensive_demo

Demonstration Notebooks
-----------------------

Comprehensive Jupyter notebooks demonstrating all library features:

.. toctree::
   :maxdepth: 2
   :titlesonly:

   notebooks/notebooks_overview

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

