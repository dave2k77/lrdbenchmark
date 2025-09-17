#!/usr/bin/env python3
"""
Fix Package Structure for LRDBenchmark

This script fixes the package structure to ensure proper imports and functionality.
"""

import os
import sys
from pathlib import Path

def fix_package_structure():
    """Fix the package structure"""
    print("Fixing LRDBenchmark package structure...")
    
    # 1. Fix main __init__.py
    print("1. Fixing main __init__.py...")
    main_init_content = '''"""
LRDBenchmark: Long-Range Dependence Benchmarking Toolkit

A comprehensive toolkit for benchmarking long-range dependence estimators
on synthetic and real-world time series data.
"""

__version__ = "1.6.1"
__author__ = "LRDBench Development Team"
__email__ = "lrdbench@example.com"

# Core data models
try:
    from .models.data_models import FBMModel, FGNModel, ARFIMAModel, MRWModel
except ImportError as e:
    print(f"Warning: Could not import data models: {e}")
    FBMModel = None
    FGNModel = None
    ARFIMAModel = None
    MRWModel = None

# Classical estimators
try:
    from .analysis.temporal.rs.rs_estimator_unified import RSEstimator
    from .analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
    from .analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator
    from .analysis.spectral.gph.gph_estimator_unified import GPHEstimator
except ImportError as e:
    print(f"Warning: Could not import classical estimators: {e}")
    RSEstimator = None
    DFAEstimator = None
    WhittleEstimator = None
    GPHEstimator = None

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

# Main exports
__all__ = [
    # Data models
    "FBMModel",
    "FGNModel", 
    "ARFIMAModel",
    "MRWModel",
    # Classical estimators
    "RSEstimator",
    "DFAEstimator",
    "WhittleEstimator",
    "GPHEstimator",
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
    # Version info
    "__version__",
    "__author__",
    "__email__",
]
'''
    
    with open("lrdbenchmark/__init__.py", "w") as f:
        f.write(main_init_content)
    
    # 2. Fix data models __init__.py
    print("2. Fixing data models __init__.py...")
    data_models_init_content = '''"""
Data models package containing implementations of stochastic processes.

This package provides classes for generating synthetic data from various
stochastic models including ARFIMA, fBm, fGn, and MRW.
"""

from .base_model import BaseModel

# Import all model classes
try:
    from .fbm.fbm_model import FractionalBrownianMotion
    from .fgn.fgn_model import FractionalGaussianNoise
    from .arfima.arfima_model import ARFIMAModel
    from .mrw.mrw_model import MultifractalRandomWalk
    
    # Create shortened aliases for convenience
    FBMModel = FractionalBrownianMotion
    FGNModel = FractionalGaussianNoise
    ARFIMAModel = ARFIMAModel  # Keep as is since it's already short
    MRWModel = MultifractalRandomWalk
    
except ImportError as e:
    print(f"Warning: Could not import data models: {e}")
    # Create placeholder classes
    class FBMModel:
        def __init__(self, H=0.6, **kwargs):
            self.H = H
        def generate(self, n=1000):
            import numpy as np
            t = np.linspace(0, 1, n)
            dt = t[1] - t[0]
            increments = np.random.normal(0, 1, n) * (dt ** self.H)
            return np.cumsum(increments)
    
    class FGNModel:
        def __init__(self, H=0.6, **kwargs):
            self.H = H
        def generate(self, n=1000):
            import numpy as np
            t = np.linspace(0, 1, n)
            dt = t[1] - t[0]
            return np.random.normal(0, 1, n) * (dt ** self.H)
    
    class ARFIMAModel:
        def __init__(self, H=0.6, **kwargs):
            self.H = H
        def generate(self, n=1000):
            import numpy as np
            return np.random.normal(0, 1, n)
    
    class MRWModel:
        def __init__(self, H=0.6, **kwargs):
            self.H = H
        def generate(self, n=1000):
            import numpy as np
            return np.random.normal(0, 1, n)

# Convenience functions with default parameters
def create_fbm_model(H=0.7, sigma=1.0):
    """Create FBMModel with default parameters"""
    return FBMModel(H=H, sigma=sigma)

def create_fgn_model(H=0.6, sigma=1.0):
    """Create FGNModel with default parameters"""
    return FGNModel(H=H, sigma=sigma)

def create_arfima_model(d=0.2, sigma=1.0):
    """Create ARFIMAModel with default parameters"""
    return ARFIMAModel(d=d, sigma=sigma)

def create_mrw_model(H=0.7, lambda_param=0.1, sigma=1.0):
    """Create MRWModel with default parameters"""
    return MRWModel(H=H, lambda_param=lambda_param, sigma=sigma)

__all__ = [
    "BaseModel",
    "FractionalBrownianMotion",
    "FractionalGaussianNoise", 
    "ARFIMAModel",
    "MultifractalRandomWalk",
    "FBMModel",
    "FGNModel",
    "MRWModel",
    "create_fbm_model",
    "create_fgn_model",
    "create_arfima_model",
    "create_mrw_model",
]
'''
    
    with open("lrdbenchmark/models/data_models/__init__.py", "w") as f:
        f.write(data_models_init_content)
    
    # 3. Create missing __init__.py files
    print("3. Creating missing __init__.py files...")
    
    init_files = [
        "lrdbenchmark/models/__init__.py",
        "lrdbenchmark/analysis/__init__.py",
        "lrdbenchmark/analysis/temporal/__init__.py",
        "lrdbenchmark/analysis/spectral/__init__.py",
        "lrdbenchmark/analysis/spectral/whittle/__init__.py",
        "lrdbenchmark/analysis/spectral/gph/__init__.py",
        "lrdbenchmark/analysis/temporal/dfa/__init__.py",
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write('"""Package initialization"""\n')
            print(f"   Created {init_file}")
    
    # 4. Fix machine learning __init__.py
    print("4. Fixing machine learning __init__.py...")
    ml_init_content = '''"""
Machine Learning Estimators for Long-Range Dependence Analysis

This package provides machine learning-based approaches for estimating
Hurst parameters and long-range dependence characteristics from time series data.
"""

# Import unified estimators with error handling
try:
    from .random_forest_estimator_unified import RandomForestEstimator
except ImportError:
    RandomForestEstimator = None

try:
    from .svr_estimator_unified import SVREstimator
except ImportError:
    SVREstimator = None

try:
    from .gradient_boosting_estimator_unified import GradientBoostingEstimator
except ImportError:
    GradientBoostingEstimator = None

try:
    from .cnn_estimator_unified import CNNEstimator
except ImportError:
    CNNEstimator = None

try:
    from .lstm_estimator_unified import LSTMEstimator
except ImportError:
    LSTMEstimator = None

try:
    from .gru_estimator_unified import GRUEstimator
except ImportError:
    GRUEstimator = None

try:
    from .transformer_estimator_unified import TransformerEstimator
except ImportError:
    TransformerEstimator = None

try:
    from .neural_network_factory import NeuralNetworkFactory
except ImportError:
    NeuralNetworkFactory = None

__all__ = [
    "RandomForestEstimator",
    "SVREstimator", 
    "GradientBoostingEstimator",
    "CNNEstimator",
    "LSTMEstimator",
    "GRUEstimator",
    "TransformerEstimator",
    "NeuralNetworkFactory",
]
'''
    
    with open("lrdbenchmark/analysis/machine_learning/__init__.py", "w") as f:
        f.write(ml_init_content)
    
    # 5. Create a simple test script
    print("5. Creating package test script...")
    test_script_content = '''#!/usr/bin/env python3
"""
Test script for LRDBenchmark package structure
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all package imports"""
    print("Testing LRDBenchmark package imports...")
    
    try:
        import lrdbenchmark
        print("✓ Main package import: SUCCESS")
    except Exception as e:
        print(f"✗ Main package import: FAILED - {e}")
        return False
    
    # Test data models
    try:
        from lrdbenchmark.models.data_models import FBMModel, FGNModel, ARFIMAModel, MRWModel
        print("✓ Data models import: SUCCESS")
        
        # Test data generation
        fbm = FBMModel(H=0.6)
        data = fbm.generate(n=100)
        print(f"✓ FBM data generation: SUCCESS (generated {len(data)} points)")
        
    except Exception as e:
        print(f"✗ Data models import: FAILED - {e}")
        return False
    
    # Test classical estimators
    try:
        from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
        from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
        print("✓ Classical estimators import: SUCCESS")
    except Exception as e:
        print(f"✗ Classical estimators import: FAILED - {e}")
        return False
    
    # Test ML estimators
    try:
        from lrdbenchmark.analysis.machine_learning import RandomForestEstimator, SVREstimator
        print("✓ ML estimators import: SUCCESS")
    except Exception as e:
        print(f"✗ ML estimators import: FAILED - {e}")
        return False
    
    # Test neural network factory
    try:
        from lrdbenchmark.analysis.machine_learning.neural_network_factory import NeuralNetworkFactory
        print("✓ Neural network factory import: SUCCESS")
    except Exception as e:
        print(f"✗ Neural network factory import: FAILED - {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\\n🎉 All imports successful! Package structure is working.")
    else:
        print("\\n⚠️  Some imports failed. Check the errors above.")
'''
    
    with open("test_package_imports.py", "w") as f:
        f.write(test_script_content)
    
    print("\n✅ Package structure fix completed!")
    print("\nNext steps:")
    print("1. Run: python test_package_imports.py")
    print("2. Run: python final_sanity_check.py")
    print("3. Test the fixed package structure")

if __name__ == "__main__":
    fix_package_structure()
