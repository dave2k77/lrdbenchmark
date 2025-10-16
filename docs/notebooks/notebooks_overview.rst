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

Available Notebooks
-------------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   01_data_generation_and_visualisation
   02_estimation_and_validation
   03_custom_models_and_estimators
   04_comprehensive_benchmarking
   05_leaderboard_generation

01. Data Generation and Visualization
=====================================

**File**: `notebooks/01_data_generation_and_visualisation.ipynb`

**Purpose**: Demonstrate all available data models and visualization techniques

**Key Content**:
- Introduction to long-range dependence concepts
- Generate data from all models: FBM, FGN, ARFIMA, MRW, Alpha-Stable
- Visual comparison of different Hurst parameters (H=0.3, 0.5, 0.7, 0.9)
- Time series plots, autocorrelation functions, power spectral density
- Data quality assessment (statistics, stationarity tests)
- Theoretical properties vs empirical observations

**Data Models Covered**:
- `FBMModel` - self-similar Gaussian process (Fractional Brownian Motion)
- `FGNModel` - increments of FBM (Fractional Gaussian Noise)
- `ARFIMAModel` - autoregressive fractionally integrated moving average
- `MRWModel` - multifractal processes (Multifractal Random Walk)
- `AlphaStableModel` - heavy-tailed distributions

02. Estimation and Statistical Validation
==========================================

**File**: `notebooks/02_estimation_and_validation.ipynb`

**Purpose**: Demonstrate all estimator categories with statistical validation

**Key Content**:
- Overview of estimator categories (Classical, ML, Neural)
- Classical estimators: R/S, DFA, DMA, Higuchi, GPH, Whittle, Periodogram, CWT
- Machine learning estimators: Random Forest, SVR, Gradient Boosting
- Neural network estimators: CNN, LSTM, GRU, Transformer
- Statistical validation: confidence intervals, bootstrap methods
- Performance comparison across estimator categories
- Decision guidelines for estimator selection

**Estimator Categories**:
- **Temporal**: R/S Analysis, DFA, DMA, Higuchi
- **Spectral**: GPH, Whittle, Periodogram
- **Wavelet**: CWT, Wavelet Variance, Log Variance, Wavelet Whittle
- **Multifractal**: MFDFA, Wavelet Leaders
- **ML**: Random Forest, SVR, Gradient Boosting
- **Neural**: CNN, LSTM, GRU, Transformer

03. Custom Models and Estimators
=================================

**File**: `notebooks/03_custom_models_and_estimators.ipynb`

**Purpose**: Show users how to extend the library with custom data models and estimators

**Key Content**:
- Understanding the `BaseModel` interface for data models
- Creating a custom data model (Fractional Ornstein-Uhlenbeck process)
- Implementing required methods: `generate()`, `_validate_parameters()`, `get_theoretical_properties()`
- Creating a custom classical estimator (Variance-Based Hurst Estimator)
- Integrating custom models into the benchmark framework
- Best practices for extensibility

**Base Classes**:
- `BaseModel` from `lrdbenchmark/models/data_models/base_model.py`
- Estimator patterns from existing unified estimators
- `MLModelFactory` for ML extensions

04. Comprehensive Benchmarking
=============================

**File**: `notebooks/04_comprehensive_benchmarking.ipynb`

**Purpose**: Demonstrate the full benchmarking system with different configurations

**Key Content**:
- Introduction to the `ComprehensiveBenchmark` class
- Running classical-only benchmarks
- Running ML-only benchmarks
- Running neural-only benchmarks
- Running comprehensive (all estimators) benchmarks
- Contamination robustness testing:
  - Additive Gaussian noise
  - Multiplicative noise
  - Outliers
  - Trend contamination
  - Seasonal patterns
  - Missing data
- Performance metrics: MAE, execution time, success rate
- Visualizing benchmark results

**Benchmark Types**:
- `classical`: 13+ classical statistical estimators
- `ML`: 3 machine learning estimators
- `neural`: 4 neural network estimators
- `comprehensive`: All estimators combined

05. Leaderboard Generation
==========================

**File**: `notebooks/05_leaderboard_generation.ipynb`

**Purpose**: Demonstrate leaderboard creation and performance ranking

**Key Content**:
- Loading benchmark results from multiple runs
- Creating overall performance leaderboards
- Category-wise rankings (Classical vs ML vs Neural)
- Composite scoring system:
  - Accuracy score (MAE-based)
  - Speed score (execution time)
  - Robustness score (contamination performance)
  - Realistic performance score
- Weighted composite rankings
- Creating publication-ready tables
- Generating performance comparison plots
- Exporting leaderboards to LaTeX/Markdown/CSV

**Leaderboard Types**:
- Overall leaderboard (all estimators)
- Category leaderboards (Classical, ML, Neural)
- Robustness leaderboard (contamination resistance)
- Speed leaderboard (execution time)
- Accuracy leaderboard (pure MAE)

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
   
      git clone https://github.com/dave2k77/LRDBenchmark.git
      cd LRDBenchmark

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

- **Issues**: Report problems with notebooks on `GitHub Issues <https://github.com/dave2k77/LRDBenchmark/issues>`_
- **Discussions**: Ask questions on `GitHub Discussions <https://github.com/dave2k77/LRDBenchmark/discussions>`_
- **Contributing**: Submit improvements via pull requests

The notebooks are actively maintained and updated with each library release.
