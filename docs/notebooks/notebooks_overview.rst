Demonstration Notebooks Overview
=================================

lrdbenchmark ships a Markdown-based notebook set (converted from the original Jupyter notebooks) that mirrors the tutorial series published in this documentation. Each notebook is self-contained, well annotated, and follows the same progression as the narrative material.

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
- :doc:`/tutorials/tutorial_03_customization` — Extending lrdbenchmark with custom data generators and estimators
- :doc:`/tutorials/tutorial_04_benchmarking` — Running comprehensive and contamination-aware benchmarks
- :doc:`/tutorials/tutorial_05_leaderboards` — Building stratified leaderboards and publishing results

Notebook storage format
-----------------------

To keep the repository lightweight and diff-friendly, notebooks are distributed as Markdown files in ``notebooks/markdown/`` together with exported figures and data artefacts. They can be opened directly in editors that understand MyST/Markdown notebooks, or converted back to ``.ipynb`` format with tools such as `Jupytext <https://jupytext.readthedocs.io/>`_.

Getting Started with Notebooks
==============================

Prerequisites
-------------

- Python 3.10–3.12
- Jupyter Notebook or JupyterLab
- lrdbenchmark installed (``pip install lrdbenchmark``)

Converting back to ``.ipynb``
-----------------------------

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/dave2k77/lrdbenchmark.git
      cd lrdbenchmark

2. **Install the optional tooling**:

   .. code-block:: bash

      pip install -e .
      pip install jupytext jupyter matplotlib seaborn

3. **Convert and launch**:

   .. code-block:: bash

      jupytext --to notebook notebooks/markdown/01_data_generation_and_visualisation.md
      jupyter notebook notebooks/markdown/

4. **Open the converted notebooks** in order for the complete learning experience.

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

Each notebook generates the same artefacts as the original ``.ipynb`` versions:

- **Visualisations**: High-quality plots saved to ``outputs/``.
- **Data files**: CSV/JSON exports of results.
- **Performance metrics**: Detailed analysis tables.
- **Publication-ready figures**: LaTeX/PNG formats.

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
