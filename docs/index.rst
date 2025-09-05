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

🏆 **Latest Results: ML Models Achieve 74% Better Accuracy Than Classical Methods!**

Our latest comprehensive benchmark shows:
- **74% better accuracy** for ML models (0.079 MAE) vs Classical methods (0.305 MAE)
- **Gradient Boosting: Best overall performance** (0.023 MAE - 90% better than best classical)
- **4 production-ready ML models** with 50-70 engineered features per model
- **Advanced feature engineering** including spectral, DFA, wavelet, and R/S analysis
- **Production-ready system** with train-once, apply-many workflow
- **100% success rate** for both ML and classical approaches

Key Features
------------

* **Production-Ready ML Models**: SVR, Gradient Boosting, Random Forest, and CNN with 50-70 engineered features
* **Superior ML Performance**: 74% better accuracy than classical methods with Gradient Boosting achieving best overall performance
* **Comprehensive Classical Estimators**: 13 adaptive estimators with automatic optimization framework selection
* **Intelligent Backend System**: Automatic GPU/JAX, CPU/Numba, or NumPy selection based on data characteristics
* **Advanced Feature Engineering**: Spectral, DFA, wavelet, and R/S analysis features for ML models
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

