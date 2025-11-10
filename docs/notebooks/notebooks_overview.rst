Demonstration Notebooks Overview
=================================

LRDBenchmark includes comprehensive Jupyter notebooks that demonstrate all library features through practical examples. These notebooks are designed to be self-contained, well-documented, and provide a complete learning path from basic concepts to advanced applications.

Notebook Structure
------------------

The demonstration notebooks follow a progressive learning structure:

1. **Data Generation & Visualization** - Understanding stochastic models
2. **Estimation & Validation** - Learning estimator categories and statistical validation
3. **Custom Models & Estimators** - Library extensibility and custom implementations
4. **Comprehensive Benchmarking** - Full benchmarking system with contamination testing
5. **Leaderboard Generation** - Performance rankings and comparative analysis

Tutorial Series
---------------

The instructional material previously hosted exclusively as notebooks is now published as a structured tutorial sequence. For the canonical, documentation-friendly narrative, follow the multi-part series:

- :doc:`/tutorials/tutorial_01_synthetic_data` — Generating synthetic and realistic LRD data, including contamination scenarios
- :doc:`/tutorials/tutorial_02_estimators` — Estimating the Hurst parameter with statistical validation and uncertainty quantification
- :doc:`/tutorials/tutorial_03_customization` — Extending LRDBenchmark with custom data generators and estimators
- :doc:`/tutorials/tutorial_04_benchmarking` — Running comprehensive and contamination-aware benchmarks
- :doc:`/tutorials/tutorial_05_leaderboards` — Building stratified leaderboards and publishing results

Legacy Notebooks
----------------

The original Jupyter notebooks remain available in the repository under ``notebooks/`` for interactive experimentation. They mirror the content of the tutorial series and can be launched locally when a live environment is preferred.

Getting Started with Notebooks
==============================

Prerequisites
-------------

- Python 3.8+
- Jupyter Notebook or JupyterLab
- LRDBenchmark installed (``pip install lrdbenchmark``)

Running the Notebooks
--------------------

1. **Clone the repository**:
   .. code-block:: bash
   
      git clone https://github.com/dave2k77/lrdbenchmark.git
      cd lrdbenchmark

2. **Install dependencies**:
   .. code-block:: bash
   
      pip install -e .
      pip install jupyter matplotlib seaborn

3. **Start Jupyter**:
   .. code-block:: bash
   
      jupyter notebook notebooks/

4. **Run notebooks in order** for the complete learning experience

Notebook Features
-----------------

- **Self-contained**: Each notebook runs independently
- **Progressive complexity**: Build from simple to advanced concepts
- **Practical focus**: Real-world workflows, not just API documentation
- **Reproducible**: Fixed seeds, saved outputs
- **Well-documented**: Extensive markdown and comments
- **Visual**: Rich plots and tables throughout
- **Educational**: Explain why, not just how

Output Files
------------

Each notebook generates:
- **Visualizations**: High-quality plots saved to `outputs/` directory
- **Data files**: CSV/JSON exports of results
- **Performance metrics**: Detailed analysis tables
- **Publication-ready figures**: LaTeX/PNG formats

Cross-References
----------------

The notebooks are designed to work together:
- **Notebook 1** → **Notebook 2**: Use generated data for estimation
- **Notebook 2** → **Notebook 3**: Apply estimators to custom models
- **Notebook 3** → **Notebook 4**: Include custom components in benchmarks
- **Notebook 4** → **Notebook 5**: Use benchmark results for leaderboards

This creates a complete workflow from data generation to performance analysis.

Support and Contributing
=========================

- **Issues**: Report problems with notebooks on `GitHub Issues <https://github.com/dave2k77/lrdbenchmark/issues>`_
- **Discussions**: Ask questions on `GitHub Discussions <https://github.com/dave2k77/lrdbenchmark/discussions>`_
- **Contributing**: Submit improvements via pull requests

The notebooks are actively maintained and updated with each library release.
