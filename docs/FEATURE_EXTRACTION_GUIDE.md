# Unified Feature Extraction Guide

## 🎯 Overview

The LRDBenchmark library includes a comprehensive unified feature extraction pipeline designed specifically for Long-Range Dependence (LRD) analysis. This guide explains the 76-feature set, its relevance to LRD detection, and how to use it effectively.

## 🚀 Key Features

### Unified 76-Feature Pipeline
- **Comprehensive Coverage**: Statistical, temporal, spectral, and wavelet features
- **Pre-trained Model Support**: Works seamlessly with existing ML models
- **Feature Subsets**: Optimized subsets for different algorithms (29, 54, 76 features)
- **Automatic Device Selection**: GPU acceleration with CPU fallback
- **Performance Optimized**: NumPy/SciPy acceleration with Numba JIT compilation

### Supported ML Models
- **Random Forest**: 76 features (full feature set)
- **SVR**: 29 features (statistical and basic temporal features)
- **Gradient Boosting**: 54 features (statistical, temporal, and spectral features)

## 📊 Feature Categories

### 1. Basic Statistical Features (10 features)
- **Mean, Standard Deviation, Variance**: Central tendency and spread
- **Min, Max, Median**: Distribution bounds and center
- **Quartiles (25th, 75th percentiles)**: Distribution shape
- **Skewness, Kurtosis**: Distribution asymmetry and tail behavior

**LRD Relevance**: These features capture the fundamental statistical properties that change with long-range dependence.

### 2. Autocorrelation Features (7 features)
- **Lags**: 1, 2, 5, 10, 20, 50, 100 time steps
- **Purpose**: Measure temporal dependencies at different scales

**LRD Relevance**: Autocorrelation decay patterns are fundamental to LRD detection. Long-range dependence shows persistent autocorrelation.

### 3. Multi-scale Variance of Increments (20 features)
- **Scales**: 1, 2, 4, 8, 16 time steps
- **Metrics per scale**: Variance, Mean Absolute Error, Standard Deviation, Range
- **Purpose**: Capture scale-dependent behavior characteristic of LRD

**LRD Relevance**: LRD processes show specific scaling relationships in their increments that these features capture.

### 4. Spectral Features (10 features)
- **Power Spectral Density**: Low, mid, high frequency bands
- **Spectral Slope**: Power-law exponent (key LRD indicator)
- **Spectral Centroid, Spread**: Frequency distribution characteristics
- **Spectral Entropy**: Complexity measure
- **Dominant Frequency, Power**: Peak frequency characteristics
- **Spectral Rolloff**: 95% power frequency

**LRD Relevance**: LRD processes exhibit 1/f^β power spectra, making spectral features crucial for detection.

### 5. DFA-inspired Features (8 features)
- **Detrended Fluctuation**: At scales 4, 8, 16, 32
- **Detrending Metrics**: Standard deviation, mean absolute error
- **Trend Analysis**: Slope and correlation with time

**LRD Relevance**: DFA is a gold standard for LRD detection. These features capture DFA-like behavior.

### 6. Wavelet-inspired Features (6 features)
- **Multi-resolution Variance**: At levels 1, 2, 3
- **Haar Wavelet Coefficients**: Approximation and detail variances
- **Detail Mean Absolute Error**: Wavelet detail statistics

**LRD Relevance**: Wavelets naturally capture multi-scale behavior, essential for LRD analysis.

### 7. Higher-order Statistics (8 features)
- **Quantiles**: 10th, 90th percentiles, inter-quantile range
- **Mean Absolute Deviation**: Robust spread measure
- **Difference Statistics**: Variance, mean absolute error, skewness, kurtosis of first differences

**LRD Relevance**: Higher-order moments capture non-Gaussian behavior often present in LRD processes.

### 8. Range-based Features (7 features)
- **R/S Statistics**: At multiple segmentations (2, 4, 8 segments)
- **Overall R/S**: Full series R/S statistic
- **Range Statistics**: Absolute range, normalized range, mean absolute differences

**LRD Relevance**: R/S analysis is a classical LRD method. These features capture R/S-like behavior.

## 🔧 Usage Examples

### Basic Feature Extraction
```python
from lrdbenchmark.analysis.machine_learning.unified_feature_extractor import UnifiedFeatureExtractor
import numpy as np

# Generate sample data
data = np.random.randn(1000)

# Extract 76 features
features = UnifiedFeatureExtractor.extract_features(data, feature_set="full_76")
print(f"Extracted {len(features)} features")

# Get feature names
feature_names = UnifiedFeatureExtractor.get_feature_names("full_76")
print(f"Feature names: {feature_names[:5]}...")  # First 5 names
```

### Using with ML Estimators
```python
from lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified import RandomForestEstimator
from lrdbenchmark.analysis.machine_learning.svr_estimator_unified import SVREstimator
from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator_unified import GradientBoostingEstimator

# Initialize estimators (they use unified feature extraction automatically)
rf_estimator = RandomForestEstimator()  # Uses 76 features
svr_estimator = SVREstimator()         # Uses 29 features
gb_estimator = GradientBoostingEstimator()  # Uses 54 features

# Estimate Hurst parameter
data = np.random.randn(1000)
result = rf_estimator.estimate(data)
print(f"Hurst parameter: {result['hurst_parameter']}")
```

### Feature Subsets
```python
# Extract different feature subsets
features_29 = UnifiedFeatureExtractor.extract_features(data, feature_set="svr_29")
features_54 = UnifiedFeatureExtractor.extract_features(data, feature_set="gradient_boosting_54")
features_76 = UnifiedFeatureExtractor.extract_features(data, feature_set="full_76")

print(f"SVR features: {len(features_29)}")
print(f"Gradient Boosting features: {len(features_54)}")
print(f"Random Forest features: {len(features_76)}")
```

## 📈 Performance Characteristics

### Computational Efficiency
- **Feature Extraction Time**: ~1-5ms for 1000-point series
- **Memory Usage**: Minimal (in-place operations where possible)
- **Scalability**: Linear with data length
- **GPU Acceleration**: Automatic when available

### Feature Quality
- **Robustness**: Handles NaN/Inf values gracefully
- **Deterministic**: Same input always produces same output
- **Normalized**: Features are scaled appropriately for ML models
- **Comprehensive**: Covers all major LRD detection approaches

## 🎯 Best Practices

### Data Preparation
```python
# Ensure data is clean
data = np.array(data, dtype=np.float64)
data = data[~np.isnan(data)]  # Remove NaN values
data = data[~np.isinf(data)]  # Remove infinite values

# Minimum length recommendation
if len(data) < 100:
    print("Warning: Data length < 100 may affect feature quality")
```

### Feature Selection
```python
# For different use cases
if use_case == "quick_analysis":
    features = UnifiedFeatureExtractor.extract_features(data, "svr_29")
elif use_case == "comprehensive_analysis":
    features = UnifiedFeatureExtractor.extract_features(data, "full_76")
elif use_case == "balanced_analysis":
    features = UnifiedFeatureExtractor.extract_features(data, "gradient_boosting_54")
```

### Model Integration
```python
# Custom estimator using unified features
class CustomEstimator(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.feature_extractor = UnifiedFeatureExtractor()
    
    def estimate(self, data):
        # Extract features
        features = self.feature_extractor.extract_features(data, "full_76")
        
        # Your custom estimation logic
        hurst_estimate = self._custom_estimation(features)
        
        return {
            'hurst_parameter': hurst_estimate,
            'method': 'custom',
            'features_used': 76
        }
```

## 🔍 Feature Importance Analysis

### Most Important Features for LRD Detection
1. **Spectral Slope**: Direct measure of 1/f^β behavior
2. **R/S Statistics**: Classical LRD indicators
3. **DFA Features**: Detrended fluctuation analysis
4. **Autocorrelation**: Temporal dependency measures
5. **Multi-scale Variance**: Scaling behavior

### Feature Correlations
- **High Correlation**: Features within same category (e.g., spectral features)
- **Low Correlation**: Features across categories (e.g., statistical vs. spectral)
- **Optimal Subset**: 29-feature subset balances performance and accuracy

## 🚀 Advanced Usage

### Custom Feature Engineering
```python
# Extend the feature extractor
class CustomFeatureExtractor(UnifiedFeatureExtractor):
    @staticmethod
    def extract_custom_features(data):
        # Add your custom features
        custom_features = []
        
        # Example: Custom LRD measure
        custom_features.append(your_custom_lrd_measure(data))
        
        return custom_features
    
    @classmethod
    def extract_features(cls, data, feature_set="custom"):
        if feature_set == "custom":
            base_features = super().extract_features(data, "full_76")
            custom_features = cls.extract_custom_features(data)
            return np.concatenate([base_features, custom_features])
        else:
            return super().extract_features(data, feature_set)
```

### Batch Processing
```python
# Process multiple time series
def process_batch(data_list):
    results = []
    for data in data_list:
        features = UnifiedFeatureExtractor.extract_features(data, "full_76")
        results.append(features)
    return np.array(results)
```

## 📚 References

### Theoretical Background
- **Long-Range Dependence**: Beran (1994), Taqqu (2003)
- **Feature Engineering**: Guyon & Elisseeff (2003)
- **Spectral Analysis**: Priestley (1981)
- **Wavelet Analysis**: Mallat (2009)

### Implementation Details
- **NumPy/SciPy**: Harris et al. (2020)
- **Numba JIT**: Lam et al. (2015)
- **PyTorch**: Paszke et al. (2019)
- **JAX**: Bradbury et al. (2018)

## 🎉 Summary

The unified feature extraction pipeline provides:

✅ **Comprehensive Coverage**: 76 features covering all major LRD detection approaches
✅ **Pre-trained Model Support**: Works seamlessly with existing ML models
✅ **Performance Optimized**: Fast extraction with GPU acceleration
✅ **Flexible Usage**: Multiple feature subsets for different algorithms
✅ **Production Ready**: Robust, tested, and well-documented

This feature extraction system enables state-of-the-art LRD detection using machine learning approaches while maintaining the theoretical rigor of classical methods.
