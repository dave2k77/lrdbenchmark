"""
LRDBenchmark: Long-Range Dependence Benchmarking Toolkit

A comprehensive toolkit for benchmarking long-range dependence estimators
on synthetic and real-world time series data.
"""

import os
import warnings
import logging

# Set CPU-only mode by default to avoid GPU issues
# Users can override by setting these environment variables explicitly
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
if 'JAX_PLATFORMS' not in os.environ:
    os.environ['JAX_PLATFORMS'] = 'cpu'
# Prevent JAX from discovering CUDA plugins when in CPU-only mode
if 'XLA_PYTHON_CLIENT_PREALLOCATE' not in os.environ:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Disable JAX CUDA plugin discovery to prevent CUDA_ERROR_NO_DEVICE errors
if 'JAX_PLATFORM_NAME' not in os.environ:
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# Suppress JAX CUDA warnings and errors when using CPU-only mode
warnings.filterwarnings('ignore', category=UserWarning, module='jax')
warnings.filterwarnings('ignore', message='.*Jax plugin configuration error.*')
warnings.filterwarnings('ignore', message='.*CUDA_ERROR_NO_DEVICE.*')
warnings.filterwarnings('ignore', message='.*operation cuInit.*failed.*')

# Suppress JAX logging errors - set to CRITICAL to hide plugin initialization errors
logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL)
logging.getLogger('jax_plugins').setLevel(logging.CRITICAL)

__version__ = "2.3.1"
__author__ = "LRDBench Development Team"
__email__ = "lrdbench@example.com"

# Core data models
try:
    from .models.data_models import FBMModel, FGNModel, ARFIMAModel, MRWModel, AlphaStableModel
except ImportError as e:
    print(f"Warning: Could not import data models: {e}")
    FBMModel = None
    FGNModel = None
    ARFIMAModel = None
    MRWModel = None
    AlphaStableModel = None

# Classical estimators
try:
    # Temporal estimators
    from .analysis.temporal.rs.rs_estimator_unified import RSEstimator
    from .analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
    from .analysis.temporal.dma.dma_estimator_unified import DMAEstimator
    from .analysis.temporal.higuchi.higuchi_estimator_unified import HiguchiEstimator
    
    # Spectral estimators
    from .analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator
    from .analysis.spectral.gph.gph_estimator_unified import GPHEstimator
    from .analysis.spectral.periodogram.periodogram_estimator_unified import PeriodogramEstimator
    
    # Wavelet estimators
    from .analysis.wavelet.cwt.cwt_estimator_unified import CWTEstimator
    from .analysis.wavelet.variance.variance_estimator_unified import WaveletVarianceEstimator
    from .analysis.wavelet.log_variance.log_variance_estimator_unified import WaveletLogVarianceEstimator
    from .analysis.wavelet.whittle.whittle_estimator_unified import WaveletWhittleEstimator
    
    # Multifractal estimators
    from .analysis.multifractal.mfdfa.mfdfa_estimator_unified import MFDFAEstimator
    from .analysis.multifractal.wavelet_leaders.wavelet_leaders_estimator_unified import MultifractalWaveletLeadersEstimator
    
except ImportError as e:
    print(f"Warning: Could not import classical estimators: {e}")
    # Temporal estimators
    RSEstimator = None
    DFAEstimator = None
    DMAEstimator = None
    HiguchiEstimator = None
    
    # Spectral estimators
    WhittleEstimator = None
    GPHEstimator = None
    PeriodogramEstimator = None
    
    # Wavelet estimators
    CWTEstimator = None
    WaveletVarianceEstimator = None
    WaveletLogVarianceEstimator = None
    WaveletWhittleEstimator = None
    
    # Multifractal estimators
    MFDFAEstimator = None
    MultifractalWaveletLeadersEstimator = None

# Machine Learning estimators
try:
    from .analysis.machine_learning import (
        RandomForestEstimator,
        SVREstimator,
        GradientBoostingEstimator,
        CNNEstimator,
        LSTMEstimator,
        GRUEstimator,
        TransformerEstimator,
    )
except ImportError as e:
    print(f"Warning: Could not import ML estimators: {e}")
    RandomForestEstimator = None
    SVREstimator = None
    GradientBoostingEstimator = None
    CNNEstimator = None
    LSTMEstimator = None
    GRUEstimator = None
    TransformerEstimator = None

# Neural Network Factory
try:
    from .analysis.machine_learning.neural_network_factory import NeuralNetworkFactory
except ImportError as e:
    print(f"Warning: Could not import neural network factory: {e}")
    NeuralNetworkFactory = None

# Benchmark system
try:
    from .analysis.benchmark import ComprehensiveBenchmark
except ImportError as e:
    print(f"Warning: Could not import benchmark system: {e}")
    ComprehensiveBenchmark = None

# GPU utilities
try:
    from .gpu import is_available as gpu_is_available, get_device_info, clear_cache, suggest_batch_size, get_safe_device
    from .gpu_memory import get_gpu_memory_info, clear_gpu_cache, monitor_gpu_memory
except ImportError as e:
    print(f"Warning: Could not import GPU utilities: {e}")
    gpu_is_available = lambda: False
    get_device_info = lambda: {'available': False}
    clear_cache = lambda: None
    suggest_batch_size = lambda data_size, seq_len: min(32, data_size)
    get_safe_device = lambda use_gpu=False: 'cpu'
    get_gpu_memory_info = lambda: {'torch_available': False, 'jax_available': False}
    clear_gpu_cache = lambda: None
    monitor_gpu_memory = lambda op_name="operation": None

# Main exports
__all__ = [
    # Data models
    "FBMModel",
    "FGNModel", 
    "ARFIMAModel",
    "MRWModel",
    "AlphaStableModel",
    # Classical estimators
    "RSEstimator",
    "DFAEstimator", 
    "DMAEstimator",
    "HiguchiEstimator",
    "WhittleEstimator",
    "GPHEstimator",
    "PeriodogramEstimator",
    "CWTEstimator",
    "WaveletVarianceEstimator",
    "WaveletLogVarianceEstimator",
    "WaveletWhittleEstimator",
    "MFDFAEstimator",
    "MultifractalWaveletLeadersEstimator",
    # Machine Learning estimators
    "RandomForestEstimator",
    "SVREstimator",
    "GradientBoostingEstimator",
    "CNNEstimator",
    "LSTMEstimator",
    "GRUEstimator",
    "TransformerEstimator",
    # Neural Network Factory
    "NeuralNetworkFactory",
    # Benchmark system
    "ComprehensiveBenchmark",
    # GPU utilities
    "gpu_is_available",
    "get_device_info", 
    "clear_cache",
    "suggest_batch_size",
    "get_safe_device",
    "get_gpu_memory_info",
    "clear_gpu_cache",
    "monitor_gpu_memory",
    # Version info
    "__version__",
    "__author__",
    "__email__",
]
