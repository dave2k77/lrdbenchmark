# LRDBenchmark Demonstration Notebooks

This directory contains comprehensive demonstration notebooks showing all major features of the LRDBenchmark library for long-range dependence (LRD) analysis.

## Overview

The LRDBenchmark library provides a comprehensive framework for:
- **Data Generation**: Multiple stochastic processes with long-range dependence
- **Estimation**: Classical, Machine Learning, and Neural Network estimators
- **Benchmarking**: Systematic performance evaluation and comparison
- **Extensibility**: Custom data models and estimators
- **Leaderboards**: Performance rankings and statistical analysis

## Notebooks

### 1. Data Generation and Visualisation (`01_data_generation_and_visualisation.ipynb`)

**Purpose**: Demonstrate all available data models and visualisation techniques

**Key Features**:
- Introduction to long-range dependence concepts
- Generate data from all models: FBM, FGN, ARFIMA, MRW, Alpha-Stable
- Visual comparison of different Hurst parameters
- Time series plots, autocorrelation functions, power spectral density
- Data quality assessment and statistical validation

**Data Models Covered**:
- `FractionalBrownianMotion` - self-similar Gaussian process
- `FractionalGaussianNoise` - increments of FBM
- `ARFIMAModel` - autoregressive fractionally integrated moving average
- `MultifractalRandomWalk` - multifractal processes
- `AlphaStableModel` - heavy-tailed distributions

**Prerequisites**: Basic understanding of time series analysis

---

### 2. Estimation and Statistical Validation (`02_estimation_and_validation.ipynb`)

**Purpose**: Demonstrate all estimator categories with statistical validation

**Key Features**:
- Overview of estimator categories (Classical, ML, Neural)
- Classical estimators: R/S, DFA, DMA, Higuchi, GPH, Whittle, Periodogram, CWT
- Machine learning estimators: Random Forest, SVR, Gradient Boosting
- Neural network estimators: CNN, LSTM, GRU, Transformer
- Statistical validation: confidence intervals, bootstrap methods
- Performance comparison and decision guidelines

**Estimator Categories**:
- **Temporal**: R/S Analysis, DFA, DMA, Higuchi
- **Spectral**: GPH, Whittle, Periodogram
- **Wavelet**: CWT, Wavelet Variance, Log Variance, Wavelet Whittle
- **Multifractal**: MFDFA, Wavelet Leaders
- **ML**: Random Forest, SVR, Gradient Boosting
- **Neural**: CNN, LSTM, GRU, Transformer

**Prerequisites**: Notebook 1 (Data Generation)

---

### 3. Custom Models and Estimators (`03_custom_models_and_estimators.ipynb`)

**Purpose**: Show users how to extend the library with custom data models and estimators

**Key Features**:
- Understanding the `BaseModel` interface for data models
- Creating a custom data model (Fractional Ornstein-Uhlenbeck process)
- Implementing required methods: `generate()`, `_validate_parameters()`, `get_theoretical_properties()`
- Creating a custom classical estimator (Variance-Based Hurst Estimator)
- Creating a custom ML-based estimator using the ML factory
- Integration with the benchmark framework
- Testing and validation of custom implementations

**Base Classes**:
- `BaseModel` from `lrdbenchmark/models/data_models/base_model.py`
- Estimator patterns from existing unified estimators
- `MLModelFactory` for ML extensions
- `NeuralNetworkFactory` for neural network extensions

**Prerequisites**: Notebooks 1 and 2 (Data Generation and Estimation)

---

### 4. Comprehensive Estimator Benchmarking (`04_comprehensive_benchmarking.ipynb`)

**Purpose**: Demonstrate the full benchmarking system with different configurations

**Key Features**:
- Introduction to the `ComprehensiveBenchmark` class
- Running classical-only, ML-only, neural-only, and comprehensive benchmarks
- Contamination robustness testing:
  - Additive Gaussian noise
  - Multiplicative noise
  - Outliers
  - Trend contamination
  - Seasonal patterns
  - Missing data
- Performance metrics: MAE, execution time, success rate
- Advanced metrics: convergence rate, mean signed error, bias, stability
- Data length sensitivity analysis
- Hurst parameter sensitivity analysis
- Visualising benchmark results and interpreting performance metrics

**Benchmark Types**:
- `classical`: 13+ classical statistical estimators
- `ML`: 3 machine learning estimators
- `neural`: 4 neural network estimators
- `comprehensive`: All estimators combined

**Prerequisites**: Notebook 2 (Estimation and Validation)

---

### 5. Leaderboard Generation (`05_leaderboard_generation.ipynb`)

**Purpose**: Demonstrate leaderboard creation and performance ranking

**Key Features**:
- Loading benchmark results from multiple runs
- Creating overall performance leaderboards
- Category-wise rankings (Classical vs ML vs Neural)
- Composite scoring system:
  - Accuracy score (MAE-based)
  - Speed score (execution time)
  - Robustness score (contamination performance)
  - Realistic performance score
- Weighted composite rankings
- Contamination robustness rankings
- Creating publication-ready tables
- Generating performance comparison plots
- Heatmaps of estimator performance
- Radar charts for multi-metric comparison
- Exporting leaderboards to LaTeX/Markdown/CSV
- Real-world application recommendations

**Leaderboard Types**:
- Overall leaderboard (all estimators)
- Category leaderboards (Classical, ML, Neural)
- Robustness leaderboard (contamination resistance)
- Speed leaderboard (execution time)
- Accuracy leaderboard (pure MAE)

**Prerequisites**: Notebook 4 (Comprehensive Benchmarking)

---

## Prerequisites

### System Requirements
- Python 3.8+
- NumPy, SciPy, Pandas, Matplotlib, Seaborn
- Jupyter Notebook or JupyterLab
- Optional: JAX for GPU acceleration

### Installation
```bash
# Install LRDBenchmark
pip install lrdbenchmark

# Or for development
git clone https://github.com/dave2k77/LRDBenchmark.git
cd LRDBenchmark
pip install -e .
```

### Dependencies
```bash
# Core dependencies
pip install numpy scipy pandas matplotlib seaborn

# Optional: Enhanced performance
pip install jax jaxlib

# Optional: Documentation
pip install sphinx sphinx-rtd-theme
```

## Usage

### Running the Notebooks

1. **Start with Notebook 1**: Data Generation and Visualisation
2. **Follow the sequence**: Each notebook builds on the previous ones
3. **Run all cells**: Execute cells in order for best results
4. **Save outputs**: Generated figures and data are saved to `outputs/` directory

### Expected Runtime
- **Notebook 1**: 5-10 minutes
- **Notebook 2**: 10-15 minutes
- **Notebook 3**: 5-10 minutes
- **Notebook 4**: 15-30 minutes
- **Notebook 5**: 10-20 minutes

### Output Files
Each notebook generates:
- **Figures**: High-resolution PNG files in `outputs/`
- **Data**: CSV files with results and statistics
- **Tables**: LaTeX tables for publications
- **Logs**: Detailed execution logs

## Key Features Demonstrated

### 1. Data Generation
- **5 Data Models**: FBM, FGN, ARFIMA, MRW, Alpha-Stable
- **Parameter Control**: Precise Hurst parameter specification
- **Quality Assessment**: Statistical validation of generated data
- **Visualization**: Comprehensive plots and analysis

### 2. Estimation
- **3 Categories**: Classical, ML, Neural estimators
- **20+ Estimators**: Complete coverage of LRD estimation methods
- **Statistical Validation**: Confidence intervals and significance tests
- **Performance Comparison**: Accuracy, speed, and robustness analysis

### 3. Extensibility
- **Custom Data Models**: Fractional Ornstein-Uhlenbeck process
- **Custom Estimators**: Variance-based Hurst estimator
- **Integration**: Seamless integration with existing framework
- **Best Practices**: Guidelines for extensibility

### 4. Benchmarking
- **Comprehensive Testing**: All estimator categories
- **Contamination Testing**: Robustness under various scenarios
- **Performance Metrics**: Accuracy, speed, reliability
- **Statistical Analysis**: Confidence intervals and significance tests

### 5. Leaderboards
- **Performance Rankings**: Multi-category comparisons
- **Composite Scoring**: Weighted performance metrics
- **Visualization**: Publication-ready plots and tables
- **Export Formats**: CSV, JSON, LaTeX

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure LRDBenchmark is properly installed
2. **Memory Issues**: Reduce data length or number of runs
3. **Performance**: Use JAX acceleration for large datasets
4. **Visualization**: Check matplotlib backend settings

### Getting Help

- **Documentation**: [LRDBenchmark Docs](https://lrdbenchmark.readthedocs.io/)
- **GitHub**: [LRDBenchmark Repository](https://github.com/dave2k77/LRDBenchmark)
- **Issues**: [GitHub Issues](https://github.com/dave2k77/LRDBenchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dave2k77/LRDBenchmark/discussions)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with modern Python scientific computing stack
- Leverages JAX for high-performance computing
- Inspired by the need for reproducible LRD analysis
- Community-driven development and validation

---

**Made with ❤️ for the time series analysis community**
