# LRDBenchmark

A comprehensive, reproducible framework for Long-Range Dependence (LRD) estimation and benchmarking across Classical, Machine Learning, and Neural Network methods.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Features

**Comprehensive Estimator Suite:**
- **8+ Classical Methods**: R/S, DFA, DMA, Higuchi, Periodogram, GPH, Whittle, CWT, and more
- **Unified ML Feature Engineering**: 76-feature extraction pipeline with pre-trained model support
- **3 Machine Learning Models**: Random Forest (76 features), SVR (29 features), Gradient Boosting (54 features)
- **4 Neural Network Architectures**: LSTM, GRU, CNN, Transformer with automatic device selection
- **Generalized Hurst Exponent (GHE)**: Advanced multifractal analysis capabilities

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

# Run comprehensive benchmark across multiple estimators
benchmark = ComprehensiveBenchmark()
results = benchmark.run_classical_estimators(
    data_models=['fbm', 'fgn', 'arfima'],
    n_samples=1000,
    n_trials=100
)
benchmark.generate_leaderboard(results)
```

### Heavy-Tail Robustness Analysis

```python
from lrdbenchmark import AlphaStableModel, AdaptivePreprocessor

# Generate heavy-tailed α-stable process
alpha_stable = AlphaStableModel(alpha=1.5, beta=0.0, scale=1.0)
x = alpha_stable.generate(length=1000, seed=42)

# Apply adaptive preprocessing for robust estimation
preprocessor = AdaptivePreprocessor()
x_processed = preprocessor.preprocess(x, method='auto')

# Estimate with robust preprocessing
estimator = RSEstimator()
result = estimator.estimate(x_processed)
```

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install lrdbenchmark
```

### Development Installation

```bash
git clone https://github.com/dave2k77/LRDBenchmark.git
cd LRDBenchmark
pip install -e .
```

### Optional Dependencies

For enhanced performance and additional features:

```bash
# GPU acceleration (JAX)
pip install "lrdbenchmark[jax]"

# Documentation building
pip install "lrdbenchmark[docs]"

# Development tools
pip install "lrdbenchmark[dev]"
```

## 📚 Documentation

- **📖 Full Documentation**: [https://lrdbenchmark.readthedocs.io/](https://lrdbenchmark.readthedocs.io/)
- **🚀 Quick Start Guide**: [`docs/quickstart.rst`](docs/quickstart.rst)
- **💡 Examples**: [`docs/examples/`](docs/examples/) and [`examples/`](examples/)
- **🔧 API Reference**: [API Documentation](https://lrdbenchmark.readthedocs.io/en/latest/api/)
- **📓 Demonstration Notebooks**: [`notebooks/`](notebooks/) - 5 comprehensive Jupyter notebooks showcasing all features

## 🏗️ Project Structure

```
LRDBenchmark/
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

### Classical Methods
- **R/S Analysis** - Rescaled Range analysis
- **DFA** - Detrended Fluctuation Analysis  
- **DMA** - Detrended Moving Average
- **Higuchi** - Higuchi's fractal dimension method
- **Periodogram** - Periodogram-based estimation
- **GPH** - Geweke and Porter-Hudak estimator
- **Whittle** - Whittle maximum likelihood
- **CWT** - Continuous Wavelet Transform
- **GHE** - Generalized Hurst Exponent

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
- **Classical**: R/S, DFA, DMA, Higuchi, GPH, Whittle, Periodogram, CWT
- **Machine Learning**: Random Forest, SVR, Gradient Boosting
- **Neural Networks**: CNN, LSTM, GRU, Transformer
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
git clone https://github.com/dave2k77/LRDBenchmark.git
cd LRDBenchmark

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern Python scientific computing stack
- Leverages JAX for high-performance computing
- Inspired by the need for reproducible LRD analysis
- Community-driven development and validation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/dave2k77/LRDBenchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dave2k77/LRDBenchmark/discussions)
- **Documentation**: [ReadTheDocs](https://lrdbenchmark.readthedocs.io/)

---

**Made with ❤️ for the time series analysis community**









