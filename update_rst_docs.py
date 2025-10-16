#!/usr/bin/env python3
"""
Update all RST documentation files to use simplified API imports.
"""

import os
import re
from pathlib import Path

def update_rst_file(file_path):
    """Update a single RST file."""
    print(f"📝 Updating {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Store original content for comparison
        original_content = content
        
        # Update verbose imports to simplified ones
        replacements = [
            # Data models
            (r'from lrdbenchmark\.models\.data_models\.fbm\.fbm_model import FBMModel', 
             'from lrdbenchmark import FBMModel'),
            (r'from lrdbenchmark\.models\.data_models\.fgn\.fgn_model import FGNModel', 
             'from lrdbenchmark import FGNModel'),
            (r'from lrdbenchmark\.models\.data_models\.mrw\.mrw_model import MRWModel', 
             'from lrdbenchmark import MRWModel'),
            (r'from lrdbenchmark\.models\.data_models\.alpha_stable\.alpha_stable_model import AlphaStableModel', 
             'from lrdbenchmark import AlphaStableModel'),
            (r'from lrdbenchmark\.models\.data_models\.arfima\.arfima_model import ARFIMAModel', 
             'from lrdbenchmark import ARFIMAModel'),
            
            # Estimators
            (r'from lrdbenchmark\.analysis\.temporal\.rs\.rs_estimator import RSEstimator', 
             'from lrdbenchmark import RSEstimator'),
            (r'from lrdbenchmark\.analysis\.temporal\.dfa\.dfa_estimator import DFAEstimator', 
             'from lrdbenchmark import DFAEstimator'),
            (r'from lrdbenchmark\.analysis\.spectral\.gph\.gph_estimator import GPHEstimator', 
             'from lrdbenchmark import GPHEstimator'),
            (r'from lrdbenchmark\.analysis\.spectral\.whittle\.whittle_estimator import WhittleEstimator', 
             'from lrdbenchmark import WhittleEstimator'),
            (r'from lrdbenchmark\.analysis\.temporal\.higuchi\.higuchi_estimator import HiguchiEstimator', 
             'from lrdbenchmark import HiguchiEstimator'),
            (r'from lrdbenchmark\.analysis\.wavelet\.variance\.wavelet_variance_estimator import WaveletVarianceEstimator', 
             'from lrdbenchmark import WaveletVarianceEstimator'),
            
            # Machine Learning
            (r'from lrdbenchmark\.analysis\.machine_learning\.svr_estimator import SVREstimator', 
             'from lrdbenchmark import SVREstimator'),
            (r'from lrdbenchmark\.analysis\.machine_learning\.random_forest_estimator import RandomForestEstimator', 
             'from lrdbenchmark import RandomForestEstimator'),
            (r'from lrdbenchmark\.analysis\.machine_learning\.gradient_boosting_estimator import GradientBoostingEstimator', 
             'from lrdbenchmark import GradientBoostingEstimator'),
            (r'from lrdbenchmark\.analysis\.machine_learning\.cnn_estimator import CNNEstimator', 
             'from lrdbenchmark import CNNEstimator'),
            (r'from lrdbenchmark\.analysis\.machine_learning\.lstm_estimator import LSTMEstimator', 
             'from lrdbenchmark import LSTMEstimator'),
            (r'from lrdbenchmark\.analysis\.machine_learning\.gru_estimator import GRUEstimator', 
             'from lrdbenchmark import GRUEstimator'),
            (r'from lrdbenchmark\.analysis\.machine_learning\.transformer_estimator import TransformerEstimator', 
             'from lrdbenchmark import TransformerEstimator'),
            
            # Neural Network Factory
            (r'from lrdbenchmark\.analysis\.machine_learning\.neural_network_factory import NeuralNetworkFactory', 
             'from lrdbenchmark import NeuralNetworkFactory'),
            (r'from lrdbenchmark\.analysis\.machine_learning\.neural_network_factory import NNArchitecture', 
             'from lrdbenchmark import NNArchitecture'),
            (r'from lrdbenchmark\.analysis\.machine_learning\.neural_network_factory import NNConfig', 
             'from lrdbenchmark import NNConfig'),
            (r'from lrdbenchmark\.analysis\.machine_learning\.neural_network_factory import create_all_benchmark_networks', 
             'from lrdbenchmark import create_all_benchmark_networks'),
            
            # Benchmark
            (r'from lrdbenchmark\.analysis\.benchmark import ComprehensiveBenchmark', 
             'from lrdbenchmark import ComprehensiveBenchmark'),
            
            # Contamination
            (r'from lrdbenchmark\.models\.contamination\.contamination_factory import ContaminationFactory', 
             'from lrdbenchmark import ContaminationFactory'),
            (r'from lrdbenchmark\.models\.contamination\.contamination_factory import ConfoundingScenario', 
             'from lrdbenchmark import ConfoundingScenario'),
            (r'from lrdbenchmark\.models\.contamination\.contamination_factory import ConfoundingProfile', 
             'from lrdbenchmark import ConfoundingProfile'),
            
            # Analytics
            (r'from lrdbenchmark\.analytics import AnalyticsDashboard', 
             'from lrdbenchmark import AnalyticsDashboard'),
            (r'from lrdbenchmark\.analytics import PerformanceMonitor', 
             'from lrdbenchmark import PerformanceMonitor'),
            (r'from lrdbenchmark\.analytics import ErrorAnalyzer', 
             'from lrdbenchmark import ErrorAnalyzer'),
            (r'from lrdbenchmark\.analytics import WorkflowAnalyzer', 
             'from lrdbenchmark import WorkflowAnalyzer'),
            (r'from lrdbenchmark\.analytics import UsageTracker', 
             'from lrdbenchmark import UsageTracker'),
        ]
        
        # Apply all replacements
        for old_pattern, new_pattern in replacements:
            content = re.sub(old_pattern, new_pattern, content)
        
        # Update class names in usage
        content = re.sub(r'FractionalBrownianMotion\(', 'FBMModel(', content)
        content = re.sub(r'FractionalGaussianNoise\(', 'FGNModel(', content)
        content = re.sub(r'MultifractalRandomWalk\(', 'MRWModel(', content)
        
        # Update method calls
        content = re.sub(r'\.generate\(n=', '.generate(length=', content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated {file_path}")
            return True
        else:
            print(f"ℹ️  No changes needed for {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    """Update all RST files in the docs directory."""
    print("🔄 Updating RST Documentation Files")
    print("=" * 50)
    
    docs_dir = Path("docs")
    if not docs_dir.exists():
        print("❌ docs directory not found!")
        return
    
    # Find all RST files
    rst_files = list(docs_dir.glob("**/*.rst"))
    
    print(f"📚 Found {len(rst_files)} RST files to check")
    
    updated_count = 0
    for rst_file in rst_files:
        if update_rst_file(rst_file):
            updated_count += 1
    
    print(f"\n✅ Updated {updated_count} RST files")
    print("\n📋 Summary of changes:")
    print("- Updated verbose imports to simplified API")
    print("- Updated class names (FractionalBrownianMotion → FBMModel)")
    print("- Updated method calls (generate(n= → generate(length=)")
    print("- Applied to all RST documentation files")

if __name__ == "__main__":
    main()
