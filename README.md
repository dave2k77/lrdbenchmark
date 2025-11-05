# LRDBenchmark

A comprehensive, reproducible framework for Long-Range Dependence (LRD) estimation and benchmarking across Classical, Machine Learning, and Neural Network methods.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version 2.3.1](https://img.shields.io/badge/version-2.3.1-green.svg)](https://pypi.org/project/lrdbenchmark/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17534599.svg)](https://doi.org/10.5281/zenodo.17534599)

## 🎉 **v2.3.1 - Release Highlights**

**✅ Enhanced Stability & Performance:**
- **100% Test Coverage**: Comprehensive validation across all 20 estimators
- **Robust GPU Fallback**: Graceful CPU fallback when CUDA memory is exhausted
- **Simplified API**: Unified imports with `from lrdbenchmark import ...`
- **Fixed JAX Issues**: Resolved CUDA backend initialization errors
- **Enhanced Error Handling**: Custom exception hierarchy with actionable messages

**✅ New Features:**
- **Lazy GPU Initialization**: CPU-first approach with optional GPU acceleration
- **Unified Feature Extractor**: 76-feature pipeline for ML estimators
- **Missing Estimator Modules**: Complete coverage of all 20 estimators
- **Progressive Examples**: CPU-only, GPU-optional, and production patterns

**✅ Improved Compatibility:**
- **Broader Python Support**: Python 3.8-3.12 compatibility (Python 3.13 not yet supported)
- **Optional Dependencies**: GPU acceleration libraries are truly optional
- **Enhanced Documentation**: Updated examples and API references

## 🚀 Features

**Comprehensive Estimator Suite:**
- **13 Classical Methods**: R/S, DFA, DMA, Higuchi, Periodogram, GPH, Whittle, CWT, Wavelet Variance, Wavelet Log Variance, Wavelet Whittle, MFDFA, Multifractal Wavelet Leaders
- **Unified ML Feature Engineering**: 76-feature extraction pipeline with pre-trained model support
- **3 Machine Learning Models**: Random Forest (76 features), SVR (29 features), Gradient Boosting (54 features)
- **4 Neural Network Architectures**: LSTM, GRU, CNN, Transformer with automatic device selection
- **Total: 20 Estimators** across all categories

**Robust Heavy-Tail Analysis:**
- α-stable distribution modeling for heavy-tailed time series
- Adaptive preprocessing: standardization, winsorization, log-winsorization, detrending
- Contamination-aware estimation with intelligent fallback mechanisms

**High-Performance Computing:**
- Intelligent optimization backend with graceful fallbacks: JAX → Numba → NumPy
- GPU acceleration support where available
- Optimized implementations for large-scale analysis

**Comprehensive Benchmarking:**
- End-to-end benchmarking scripts with statistical analysis
- Confidence intervals, significance tests, and effect size calculations
- Performance leaderboards and comparative analysis tools

**📚 Demonstration Notebooks:**
- **5 Comprehensive Jupyter Notebooks** showcasing all library features
- **Data Generation & Visualization**: All stochastic models with comprehensive plots
- **Estimation & Validation**: All estimator categories with statistical validation
- **Custom Models & Estimators**: Library extensibility and custom implementations
- **Comprehensive Benchmarking**: Full benchmarking system with contamination testing
- **Leaderboard Generation**: Performance rankings and comparative analysis

## 📦 Installation

### Basic Installation (CPU-only)
```bash
pip install lrdbenchmark
```

### With GPU Acceleration (Optional)
```bash
# All acceleration libraries
pip install lrdbenchmark[accel-all]

# Specific acceleration libraries
pip install lrdbenchmark[accel-jax]      # JAX acceleration
pip install lrdbenchmark[accel-pytorch]  # PyTorch acceleration
pip install lrdbenchmark[accel-numba]    # Numba acceleration
```

### Development Installation
```bash
git clone https://github.com/dave2k77/lrdbenchmark.git
cd lrdbenchmark
pip install -e .
```

## 🔧 Quick Start

### Basic Usage

```python
from lrdbenchmark import FBMModel, RSEstimator

# Generate synthetic fractional Brownian motion
fbm = FBMModel(H=0.7, sigma=1.0)
x = fbm.generate(length=1000, seed=42)

# Estimate Hurst parameter using R/S analysis
estimator = RSEstimator()
result = estimator.estimate(x)
print(f"Estimated H: {result['hurst_parameter']:.3f}")  # ~0.7
```

### Advanced Benchmarking

```python
from lrdbenchmark import ComprehensiveBenchmark

# Run comprehensive benchmark across all estimators
benchmark = ComprehensiveBenchmark()
results = benchmark.run_comprehensive_benchmark(
    data_length=1000,
    benchmark_type='comprehensive'  # Options: 'comprehensive', 'classical', 'ML', 'neural'
)
benchmark.print_summary(results)
```

### Machine Learning Estimation

```python
from lrdbenchmark import FBMModel, RandomForestEstimator

# Generate synthetic data
fbm = FBMModel(H=0.7, sigma=1.0)
x = fbm.generate(length=1000, seed=42)

# Use pre-trained ML estimator
ml_estimator = RandomForestEstimator()
result = ml_estimator.estimate(x)
print(f"Estimated H: {result['hurst_parameter']:.3f}")
```

### Neural Network Estimation

```python
from lrdbenchmark import FBMModel, LSTMEstimator

# Generate synthetic data
fbm = FBMModel(H=0.7, sigma=1.0)
x = fbm.generate(length=1000, seed=42)

# Use neural network estimator (auto-detects GPU/CPU)
nn_estimator = LSTMEstimator()
result = nn_estimator.estimate(x)
print(f"Estimated H: {result['hurst_parameter']:.3f}")
```


## 📚 Documentation

- **📖 Full Documentation**: [https://lrdbenchmark.readthedocs.io/](https://lrdbenchmark.readthedocs.io/)
- **🚀 Quick Start Guide**: [`docs/quickstart.rst`](docs/quickstart.rst)
- **💡 Examples**: [`docs/examples/`](docs/examples/) and [`examples/`](examples/)
- **🔧 API Reference**: [API Documentation](https://lrdbenchmark.readthedocs.io/en/latest/api/)
- **📓 Demonstration Notebooks**: [`notebooks/`](notebooks/) - 5 comprehensive Jupyter notebooks showcasing all features

## 🏗️ Project Structure

```
lrdbenchmark/
├── lrdbenchmark/           # Main package
│   ├── analysis/           # Estimator implementations
│   ├── models/            # Data generation models
│   ├── analytics/         # Performance monitoring
│   └── robustness/        # Heavy-tail robustness tools
├── notebooks/             # Demonstration notebooks (5 comprehensive Jupyter notebooks)
├── scripts/               # Benchmarking and analysis scripts
├── examples/              # Usage examples
├── docs/                  # Documentation
├── tests/                 # Test suite
├── tools/                 # Development utilities
└── config/                # Configuration files
```

## 🛠️ Available Estimators

### Classical Methods (13 estimators)
- **Temporal** (4): R/S Analysis, DFA, DMA, Higuchi
- **Spectral** (3): Periodogram, GPH (Geweke-Porter-Hudak), Whittle
- **Wavelet** (4): CWT (Continuous Wavelet Transform), Wavelet Variance, Wavelet Log Variance, Wavelet Whittle
- **Multifractal** (2): MFDFA, Multifractal Wavelet Leaders

### Machine Learning
- **Random Forest** - Ensemble tree-based estimation
- **Support Vector Regression** - SVM-based estimation
- **Gradient Boosting** - Boosted tree estimation

### Neural Networks
- **LSTM** - Long Short-Term Memory networks
- **GRU** - Gated Recurrent Units
- **CNN** - Convolutional Neural Networks
- **Transformer** - Attention-based architectures

## 📓 Demonstration Notebooks

LRDBenchmark includes 5 comprehensive Jupyter notebooks that demonstrate all library features:

### 1. Data Generation and Visualization
**File**: `notebooks/01_data_generation_and_visualisation.ipynb`

Demonstrates all available data models with comprehensive visualizations:
- **FBM/FGN**: Fractional Brownian Motion and Gaussian Noise
- **ARFIMA**: Autoregressive Fractionally Integrated Moving Average
- **MRW**: Multifractal Random Walk
- **Alpha-Stable**: Heavy-tailed distributions
- **Visualizations**: Time series, ACF, PSD, distributions
- **Quality Assessment**: Statistical validation and theoretical properties

### 2. Estimation and Statistical Validation
**File**: `notebooks/02_estimation_and_validation.ipynb`

Covers all estimator categories with statistical validation:
- **Classical** (13): R/S, DFA, DMA, Higuchi, GPH, Whittle, Periodogram, CWT, Wavelet Variance, Wavelet Log Variance, Wavelet Whittle, MFDFA, Multifractal Wavelet Leaders
- **Machine Learning** (3): Random Forest, SVR, Gradient Boosting
- **Neural Networks** (4): CNN, LSTM, GRU, Transformer
- **Statistical Validation**: Confidence intervals, bootstrap methods
- **Performance Comparison**: Accuracy, speed, and reliability analysis

### 3. Custom Models and Estimators
**File**: `notebooks/03_custom_models_and_estimators.ipynb`

Shows how to extend the library with custom components:
- **Custom Data Models**: Fractional Ornstein-Uhlenbeck process
- **Custom Estimators**: Variance-Based Hurst Estimator
- **Library Extensibility**: Base classes and integration patterns
- **Best Practices**: Guidelines for custom implementations

### 4. Comprehensive Benchmarking
**File**: `notebooks/04_comprehensive_benchmarking.ipynb`

Demonstrates the full benchmarking system:
- **Benchmark Types**: Classical, ML, Neural, Comprehensive
- **Contamination Testing**: Noise, outliers, trends, seasonal patterns
- **Performance Metrics**: MAE, execution time, success rate
- **Statistical Analysis**: Confidence intervals and significance tests

### 5. Leaderboard Generation
**File**: `notebooks/05_leaderboard_generation.ipynb`

Shows performance ranking and comparative analysis:
- **Performance Rankings**: Overall and category-wise leaderboards
- **Composite Scoring**: Accuracy, speed, and robustness metrics
- **Visualization**: Performance plots and comparison tables
- **Export Options**: CSV, JSON, LaTeX formats

### Getting Started with Notebooks

```bash
# Clone the repository
git clone https://github.com/dave2k77/lrdbenchmark.git
cd lrdbenchmark

# Install dependencies
pip install -e .
pip install jupyter matplotlib seaborn

# Start Jupyter
jupyter notebook notebooks/
```

Each notebook is self-contained, well-documented, and provides a complete learning path from basic concepts to advanced applications.

## 🧪 Testing

Run the test suite:

```bash
# Basic tests
python -m pytest tests/

# With coverage
python -m pytest tests/ --cov=lrdbenchmark --cov-report=html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📖 Citation

If you use LRDBenchmark in your research, please cite it:

```bibtex
@software{chin2024lrdbenchmark,
  author = {Chin, Davian R.},
  title = {LRDBenchmark: A Comprehensive Framework for Long-Range Dependence Estimation},
  version = {2.3.1},
  doi = {10.5281/zenodo.17534599},
  url = {https://github.com/dave2k77/lrdbenchmark},
  year = {2024}
}
```

**DOI**: [10.5281/zenodo.17534599](https://doi.org/10.5281/zenodo.17534599)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern Python scientific computing stack
- Leverages JAX for high-performance computing
- Inspired by the need for reproducible LRD analysis
- Community-driven development and validation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/dave2k77/lrdbenchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dave2k77/lrdbenchmark/discussions)
- **Documentation**: [ReadTheDocs](https://lrdbenchmark.readthedocs.io/)

---

**Made with ❤️ for the time series analysis community**









