# LRDBenchmark: A Comprehensive Framework for Long-Range Dependence Estimation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1000/xyz-blue.svg)](https://doi.org/10.1000/xyz)

A comprehensive and reproducible framework for benchmarking Long-Range Dependence (LRD) estimation methods with intelligent optimization backend, comprehensive adaptive classical estimators, and production-ready machine learning models.

## 🎯 Overview

LRDBenchmark provides a standardized platform for evaluating and comparing LRD estimators with automatic framework selection (GPU/JAX, CPU/Numba, NumPy), robust error handling, and realistic contamination testing. Our latest comprehensive benchmark shows **ML models achieve 74% better accuracy than classical methods** with **Gradient Boosting achieving the best overall performance** at 0.023 MAE.

### Key Features

- **🔬 Comprehensive Classical Estimators**: 13 adaptive estimators with automatic optimization framework selection
- **🤖 Production-Ready ML Models**: SVR, Gradient Boosting, Random Forest, and CNN with 50-70 engineered features
- **🧠 Intelligent Backend System**: Automatic GPU/JAX, CPU/Numba, or NumPy selection based on data characteristics
- **🛡️ Robust Error Handling**: Adaptive parameter selection and progressive fallback mechanisms
- **🧪 EEG Contamination Testing**: 8 realistic artifact scenarios for biomedical applications
- **📊 Mathematical Verification**: All estimators verified against theoretical foundations
- **⚡ High Performance**: GPU-accelerated implementations with JAX and Numba backends
- **🔄 Reproducible**: Complete code, data, and results available
- **📈 Research Ready**: Publication-quality results with comprehensive testing
- **🏆 Superior ML Performance**: 74% better accuracy than classical methods

## 🏆 Latest Results

Our comprehensive benchmark of **800 test cases** comparing ML vs Classical methods reveals:

- **74% Better Accuracy**: ML models (0.079 MAE) vs Classical methods (0.305 MAE)
- **Best Overall Performance**: Gradient Boosting (0.023 MAE - 90% better than best classical)
- **4 Production-Ready ML Models**: SVR, Gradient Boosting, Random Forest, CNN
- **Advanced Feature Engineering**: 50-70 engineered features per ML model
- **100% Success Rate**: Both ML and classical approaches
- **Production-Ready System**: Train-once, apply-many workflow with model persistence

## 📊 Performance Summary

| Method | Mean Error | Execution Time | Success Rate | Training Time |
|--------|------------|----------------|--------------|---------------|
| **GradientBoosting** | **0.023** | 17.5ms | 100% | 1.75s |
| **RandomForest** | **0.044** | 852.0ms | 100% | 84.15s |
| **SVR** | **0.079** | 14.5ms | 100% | 1.46s |
| **CNN** | **0.170** | 2.0ms | 100% | 2.01s |
| **ML Average** | **0.079** | 222.0ms | 100% | - |
| **Whittle** | 0.227 | 0.2ms | 100% | - |
| **RS (R/S)** | 0.248 | 8.5ms | 100% | - |
| **GPH** | 0.306 | 0.4ms | 100% | - |
| **DFA** | 0.447 | 14.8ms | 100% | - |
| **Classical Average** | 0.305 | 6.0ms | 100% | - |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LRDBenchmark.git
cd LRDBenchmark

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbenchmark.analysis.temporal.rs.rs_estimator import RSEstimator

# Generate synthetic data
fbm = FractionalBrownianMotion(hurst=0.8, length=1000)
data = fbm.generate()

# Estimate Hurst parameter
rs_estimator = RSEstimator()
hurst_estimate = rs_estimator.estimate(data)

print(f"True Hurst: 0.8, Estimated: {hurst_estimate:.3f}")
```

### Machine Learning Usage

```python
from lrdbenchmark.analysis.machine_learning.svr_estimator import SVREstimator
from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator import GradientBoostingEstimator
from lrdbenchmark.analysis.machine_learning.random_forest_estimator import RandomForestEstimator
import numpy as np

# Generate training data
X_train = np.random.randn(100, 500)  # 100 samples of length 500
y_train = np.random.uniform(0.2, 0.8, 100)  # True Hurst parameters

# Train ML models
svr = SVREstimator(kernel='rbf', C=1.0)
svr.train(X_train, y_train)

gb = GradientBoostingEstimator(n_estimators=50, learning_rate=0.1)
gb.train(X_train, y_train)

rf = RandomForestEstimator(n_estimators=50, max_depth=5)
rf.train(X_train, y_train)

# Make predictions on new data
new_data = np.random.randn(1, 500)
svr_pred = svr.predict(new_data)
gb_pred = gb.predict(new_data)
rf_pred = rf.predict(new_data)

print(f"SVR: {svr_pred:.3f}, Gradient Boosting: {gb_pred:.3f}, Random Forest: {rf_pred:.3f}")
```

### Run ML vs Classical Benchmark

```bash
# Run comprehensive ML vs Classical benchmark
python final_ml_vs_classical_benchmark.py

# Run simple ML benchmark
python simple_ml_vs_classical_benchmark.py

# Test individual ML estimators
python test_proper_ml_estimators.py
```

### Run Complete Benchmark

```bash
# Run comprehensive benchmark
python comprehensive_all_estimators_benchmark.py

# Analyze results
python analyze_all_estimators_results.py

# Generate publication figures
python generate_publication_figures.py
```

## 📁 Repository Structure

```
LRDBenchmark/
├── lrdbenchmark/                 # Main package
│   ├── models/                   # Data models and estimators
│   │   ├── data_models/         # Stochastic processes (FBM, FGN, ARFIMA, MRW)
│   │   └── estimators/          # Base estimator classes
│   └── analysis/                # Analysis modules
│       ├── temporal/            # Temporal estimators (DFA, R/S, DMA, Higuchi)
│       ├── spectral/            # Spectral estimators (Whittle, GPH, Periodogram)
│       ├── wavelet/             # Wavelet estimators (CWT, Wavelet Variance)
│       ├── multifractal/        # Multifractal estimators (MFDFA, Wavelet Leaders)
│       └── machine_learning/    # ML and neural network estimators
├── tests/                       # Unit tests
├── benchmarks/                  # Benchmark scripts
├── results/                     # Benchmark results
├── figures/                     # Generated figures
├── docs/                        # Documentation
├── manuscript.tex               # LaTeX manuscript
├── references.bib               # Bibliography
└── supplementary_materials.md   # Supplementary materials
```

## 🔬 Implemented Estimators

### Machine Learning Estimators (4) - **NEW!**
- **SVR**: Support Vector Regression with 50+ engineered features (0.079 MAE)
- **Gradient Boosting**: Best overall performance (0.023 MAE - 90% better than classical)
- **Random Forest**: High accuracy with feature importance (0.044 MAE)
- **CNN**: Convolutional Neural Network with production system (0.170 MAE)

### Classical Estimators (13)
- **Temporal**: DFA, R/S, DMA, Higuchi
- **Spectral**: Whittle, GPH, Periodogram
- **Wavelet**: CWT, Wavelet Variance, Wavelet Log Variance, Wavelet Whittle
- **Multifractal**: MFDFA, Wavelet Leaders


### Neural Network Estimators (3)
- **CNN**: Convolutional neural network for time series
- **LSTM**: Long short-term memory network
- **Transformer**: Attention-based architecture

## 📊 Data Models

### Fractional Brownian Motion (FBM)
Continuous-time Gaussian process with self-similarity property.

### Fractional Gaussian Noise (FGN)
Increment process of FBM with long-range dependence.

### ARFIMA Process
AutoRegressive Fractionally Integrated Moving Average with fractional differencing.

### Multifractal Random Walk (MRW)
Incorporates multifractal properties through cascade processes.

## 📈 Results and Visualizations

The framework generates comprehensive visualizations:

- **Figure 1**: Category performance comparison
- **Figure 2**: Individual estimator analysis
- **Figure 3**: Contamination effects
- **Figure 4**: Data length effects
- **Figure 5**: Comprehensive summary and recommendations

All figures are publication-ready with high resolution (300 DPI) and professional styling.

## 🧪 Experimental Design

### Factors
- **Data Models**: 4 levels (FBM, FGN, ARFIMA, MRW)
- **Estimators**: 12 levels (all implemented estimators)
- **Hurst Parameters**: 5 levels (0.6, 0.7, 0.8, 0.9, 0.95)
- **Data Lengths**: 2 levels (1000, 2000 points)
- **Contamination**: 3 levels (0%, 10%, 20% additive noise)
- **Replications**: 10 per condition

### Metrics
- **Accuracy**: Mean absolute error, relative error
- **Efficiency**: Execution time, memory usage
- **Robustness**: Performance under contamination
- **Reliability**: Success rate, consistency

## 🔧 Extending the Framework

### Adding New Estimators

```python
from lrdbenchmark.models.estimators.base_estimator import BaseEstimator

class MyEstimator(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.name = "MyEstimator"
        self.category = "Custom"
    
    def estimate(self, data):
        # Implement your estimation logic
        return hurst_estimate
```

### Adding New Data Models

```python
from lrdbenchmark.models.data_models.base_data_model import BaseDataModel

class MyDataModel(BaseDataModel):
    def __init__(self, hurst, length, **kwargs):
        super().__init__(hurst, length)
        self.name = "MyDataModel"
    
    def generate(self):
        # Implement your data generation logic
        return data
```

## 📚 Documentation

- **Manuscript**: `manuscript.tex` - Complete research paper
- **Supplementary Materials**: `supplementary_materials.md` - Detailed analysis
- **API Documentation**: Available in `docs/` directory
- **Examples**: See `examples/` directory for usage examples

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
black lrdbenchmark/
isort lrdbenchmark/
flake8 lrdbenchmark/
```

## 📄 Citation

If you use LRDBenchmark in your research, please cite:

```bibtex
@article{yourname2024,
  title={LRDBenchmark: A Comprehensive and Reproducible Framework for Long-Range Dependence Estimation},
  author={Your Name},
  journal={Journal Name},
  year={2024},
  publisher={Publisher}
}
```

## 📞 Contact

- **Email**: your.email@institution.edu
- **Issues**: [GitHub Issues](https://github.com/yourusername/LRDBenchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/LRDBenchmark/discussions)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the developers of the open-source libraries that made this work possible:
- NumPy, SciPy, scikit-learn for scientific computing
- PyTorch for neural network implementations
- Matplotlib, Seaborn for visualization
- And many others listed in `requirements.txt`

## 🔗 Related Work

- [Long-Range Dependence in Time Series](https://example.com)
- [Machine Learning for Time Series Analysis](https://example.com)
- [Benchmarking Statistical Methods](https://example.com)

---

**LRDBenchmark** - Setting the standard for Long-Range Dependence estimation benchmarking.