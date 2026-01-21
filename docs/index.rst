.. LRDBench documentation master file, created by
   sphinx-quickstart on Sun Aug 25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the lrdbenchmark documentation
=========================================

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

**lrdbenchmark** delivers reproducible benchmarking, diagnostics, and reporting for long-range dependence (LRD) analysis. It combines stochastic data models, twenty estimators, contamination-aware preprocessing, stratified reporting, and provenance capture in one toolkit.

What lrdbenchmark provides
--------------------------

* **Twenty estimators with a unified API** – 13 classical (temporal, spectral, wavelet, multifractal), 3 machine-learning, and 4 neural estimators.
* **Runtime profiles** – switch between the lightweight ``quick`` profile (used automatically under pytest/CI) and the exhaustive ``full`` profile for publication-grade studies.
* **Robust benchmarking** – contamination models, adaptive preprocessing, stratified summaries, non-parametric significance testing, and uncertainty calibration.
* **Nonstationarity testing** – time-varying H generators (regime switching, continuous drift, structural breaks), critical regime models (OU, fractional Lévy, SOC), and structural break detection (CUSUM, Chow test, ICSS).
* **Surrogate data testing** – IAAFT, phase randomization, and AR surrogates for hypothesis testing of LRD and nonlinearity.
* **Coverage probability analysis** – Monte Carlo estimation of CI coverage with studentized bootstrap and calibration diagnostics.
* **Adaptive acceleration** – automatic CPU mode with optional JAX → Numba → NumPy fallbacks and GPU support when requested.
* **Containerized experiments** – Docker support for reproducible cloud/HPC benchmarking.
* **Documentation-first tutorials** – the tutorial series now lives inside the docs and is mirrored by Markdown notebooks for interactive exploration.


Quick start
-----------

Install with ``pip install lrdbenchmark`` and see :doc:`quickstart` for end-to-end examples, including how to opt into the different runtime profiles.

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

   api/generation
   api/diagnostics
   api/uncertainty
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

Examples & demos
----------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   examples/comprehensive_adaptive_demo
   examples/comprehensive_demo
   examples/leaderboard

Domain Guides
-------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   domain/preprocessing_guidelines

Demonstration materials
-----------------------

The original interactive curriculum is available in two forms:

* :doc:`tutorials/index` – the canonical documentation narrative.
* :doc:`notebooks/notebooks_overview` – guidance on using the Markdown notebook sources bundled in ``notebooks/markdown/``.

Tutorial Series
---------------

Structured learning path through the lrdbenchmark workflow:

.. toctree::
   :maxdepth: 2
   :titlesonly:

   tutorials/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

