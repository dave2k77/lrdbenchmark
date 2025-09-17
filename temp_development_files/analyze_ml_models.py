#!/usr/bin/env python3
"""
Comprehensive analysis of machine learning models in LRDBenchmark.

This script evaluates the implementation, efficiency, and correctness of all ML models.
"""

import numpy as np
import time
import warnings
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def analyze_ml_models():
    """Analyze all machine learning models in the framework."""
    
    print("🔍 COMPREHENSIVE MACHINE LEARNING MODELS ANALYSIS")
    print("=" * 60)
    
    # Test data
    test_data = {
        "short": np.random.randn(100),
        "medium": np.random.randn(1000), 
        "long": np.random.randn(5000)
    }
    
    # ML Models to analyze
    ml_models = {
        "Random Forest": "lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified.RandomForestEstimator",
        "SVR": "lrdbenchmark.analysis.machine_learning.svr_estimator_unified.SVREstimator", 
        "Gradient Boosting": "lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator_unified.GradientBoostingEstimator",
        "CNN": "lrdbenchmark.analysis.machine_learning.cnn_estimator_unified.CNNEstimator",
        "LSTM": "lrdbenchmark.analysis.machine_learning.lstm_estimator_unified.LSTMEstimator",
        "GRU": "lrdbenchmark.analysis.machine_learning.gru_estimator_unified.GRUEstimator",
        "Transformer": "lrdbenchmark.analysis.machine_learning.transformer_estimator_unified.TransformerEstimator"
    }
    
    results = {}
    
    for model_name, model_path in ml_models.items():
        print(f"\n📊 Analyzing {model_name}")
        print("-" * 40)
        
        try:
            # Import the model
            module_path, class_name = model_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            ModelClass = getattr(module, class_name)
            
            # Test different optimization frameworks
            frameworks = ["numpy", "numba", "jax"]
            model_results = {}
            
            for framework in frameworks:
                print(f"  Testing {framework.upper()} framework...")
                
                try:
                    # Initialize model
                    start_time = time.time()
                    model = ModelClass(use_optimization=framework)
                    init_time = time.time() - start_time
                    
                    # Test on different data sizes
                    data_results = {}
                    
                    for data_name, data in test_data.items():
                        try:
                            start_time = time.time()
                            result = model.estimate(data)
                            exec_time = time.time() - start_time
                            
                            data_results[data_name] = {
                                "success": True,
                                "execution_time": exec_time,
                                "hurst_estimate": result.get("hurst_parameter", 0.5),
                                "method": result.get("method", "unknown"),
                                "fallback_used": result.get("fallback_used", False),
                                "optimization_framework": result.get("optimization_framework", framework)
                            }
                            
                        except Exception as e:
                            data_results[data_name] = {
                                "success": False,
                                "error": str(e),
                                "execution_time": 0.0
                            }
                    
                    model_results[framework] = {
                        "initialization_time": init_time,
                        "data_results": data_results,
                        "optimization_info": model.get_optimization_info() if hasattr(model, 'get_optimization_info') else {}
                    }
                    
                except Exception as e:
                    model_results[framework] = {
                        "error": str(e),
                        "initialization_time": 0.0
                    }
            
            results[model_name] = model_results
            
        except Exception as e:
            print(f"  ❌ Failed to import {model_name}: {e}")
            results[model_name] = {"import_error": str(e)}
    
    return results

def analyze_pretrained_models():
    """Analyze the pretrained model implementations."""
    
    print("\n🧠 PRETRAINED MODELS ANALYSIS")
    print("=" * 40)
    
    pretrained_models = {
        "CNN Pretrained": "lrdbenchmark.models.pretrained_models.cnn_pretrained.CNNPretrainedModel",
        "Transformer Pretrained": "lrdbenchmark.models.pretrained_models.transformer_pretrained.TransformerPretrainedModel",
        "Random Forest Pretrained": "lrdbenchmark.models.pretrained_models.ml_pretrained.RandomForestPretrainedModel",
        "SVR Pretrained": "lrdbenchmark.models.pretrained_models.ml_pretrained.SVREstimatorPretrainedModel",
        "Gradient Boosting Pretrained": "lrdbenchmark.models.pretrained_models.ml_pretrained.GradientBoostingPretrainedModel"
    }
    
    test_data = np.random.randn(500)  # Standard test data
    
    pretrained_results = {}
    
    for model_name, model_path in pretrained_models.items():
        print(f"\n📈 Testing {model_name}")
        print("-" * 30)
        
        try:
            # Import the model
            module_path, class_name = model_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            ModelClass = getattr(module, class_name)
            
            # Initialize and test
            start_time = time.time()
            model = ModelClass()
            init_time = time.time() - start_time
            
            start_time = time.time()
            result = model.estimate(test_data)
            exec_time = time.time() - start_time
            
            pretrained_results[model_name] = {
                "success": True,
                "initialization_time": init_time,
                "execution_time": exec_time,
                "hurst_estimate": result.get("hurst_parameter", 0.5),
                "method": result.get("method", "unknown"),
                "model_info": result.get("model_info", {}),
                "confidence_interval": result.get("confidence_interval", [0.4, 0.6])
            }
            
            print(f"  ✅ Success: H={result.get('hurst_parameter', 0.5):.3f}, Time={exec_time:.3f}s")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            pretrained_results[model_name] = {
                "success": False,
                "error": str(e)
            }
    
    return pretrained_results

def evaluate_architecture_quality():
    """Evaluate the quality of neural network architectures."""
    
    print("\n🏗️ NEURAL NETWORK ARCHITECTURE ANALYSIS")
    print("=" * 50)
    
    architectures = {
        "CNN": {
            "file": "lrdbenchmark/models/pretrained_models/cnn_pretrained.py",
            "class": "SimpleCNN1D",
            "layers": [
                "Conv1d(1, 16, kernel_size=5, padding=2)",
                "ReLU + MaxPool1d(2)",
                "Conv1d(16, 32, kernel_size=5, padding=2)", 
                "ReLU + MaxPool1d(2)",
                "Conv1d(32, 64, kernel_size=5, padding=2)",
                "ReLU + AdaptiveAvgPool1d(1)",
                "Linear(64, 128) + ReLU + Dropout(0.3)",
                "Linear(128, 64) + ReLU + Dropout(0.3)",
                "Linear(64, 1) + Sigmoid"
            ],
            "parameters": "~50K parameters (estimated)",
            "input_handling": "Fixed length (500), padding/truncation",
            "normalization": "Z-score normalization",
            "output": "Sigmoid (0-1 range)"
        },
        "Transformer": {
            "file": "lrdbenchmark/models/pretrained_models/transformer_pretrained.py",
            "class": "SimpleTransformer",
            "layers": [
                "Linear(1, 64) - Input projection",
                "Positional encoding (learned)",
                "TransformerEncoderLayer(d_model=64, nhead=4, num_layers=2)",
                "Global average pooling",
                "Linear(64, 64) + ReLU + Dropout(0.2)",
                "Linear(64, 32) + ReLU + Dropout(0.2)", 
                "Linear(32, 1) + Sigmoid"
            ],
            "parameters": "~25K parameters (estimated)",
            "input_handling": "Fixed length (500), padding/truncation",
            "normalization": "Z-score normalization",
            "output": "Sigmoid (0-1 range)"
        }
    }
    
    for arch_name, arch_info in architectures.items():
        print(f"\n🔧 {arch_name} Architecture:")
        print(f"  File: {arch_info['file']}")
        print(f"  Class: {arch_info['class']}")
        print(f"  Parameters: {arch_info['parameters']}")
        print(f"  Input Handling: {arch_info['input_handling']}")
        print(f"  Normalization: {arch_info['normalization']}")
        print(f"  Output: {arch_info['output']}")
        print("  Layers:")
        for layer in arch_info['layers']:
            print(f"    - {layer}")

def generate_analysis_report(results, pretrained_results):
    """Generate a comprehensive analysis report."""
    
    print("\n📋 COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 50)
    
    # Overall statistics
    total_models = len(results)
    successful_models = sum(1 for r in results.values() if not r.get("import_error"))
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total ML Models: {total_models}")
    print(f"  Successfully Imported: {successful_models}")
    print(f"  Import Success Rate: {successful_models/total_models*100:.1f}%")
    
    # Framework performance analysis
    print(f"\n⚡ FRAMEWORK PERFORMANCE:")
    framework_stats = {"numpy": 0, "numba": 0, "jax": 0}
    framework_times = {"numpy": [], "numba": [], "jax": []}
    
    for model_name, model_results in results.items():
        if "import_error" not in model_results:
            for framework, framework_results in model_results.items():
                if "error" not in framework_results:
                    framework_stats[framework] += 1
                    for data_name, data_results in framework_results.get("data_results", {}).items():
                        if data_results.get("success"):
                            framework_times[framework].append(data_results["execution_time"])
    
    for framework, count in framework_stats.items():
        avg_time = np.mean(framework_times[framework]) if framework_times[framework] else 0
        print(f"  {framework.upper()}: {count} successful models, avg time: {avg_time:.3f}s")
    
    # Model-specific analysis
    print(f"\n🎯 MODEL-SPECIFIC ANALYSIS:")
    for model_name, model_results in results.items():
        if "import_error" not in model_results:
            print(f"\n  {model_name}:")
            
            # Check for fallback usage
            fallback_count = 0
            total_tests = 0
            
            for framework, framework_results in model_results.items():
                if "error" not in framework_results:
                    for data_name, data_results in framework_results.get("data_results", {}).items():
                        total_tests += 1
                        if data_results.get("fallback_used", False):
                            fallback_count += 1
            
            fallback_rate = fallback_count / total_tests * 100 if total_tests > 0 else 0
            print(f"    Fallback Usage Rate: {fallback_rate:.1f}%")
            
            # Check optimization framework selection
            for framework, framework_results in model_results.items():
                if "error" not in framework_results:
                    opt_info = framework_results.get("optimization_info", {})
                    current_framework = opt_info.get("current_framework", "unknown")
                    print(f"    {framework.upper()} Framework: {current_framework}")
    
    # Pretrained models analysis
    print(f"\n🧠 PRETRAINED MODELS ANALYSIS:")
    pretrained_success = sum(1 for r in pretrained_results.values() if r.get("success"))
    print(f"  Successfully Tested: {pretrained_success}/{len(pretrained_results)}")
    
    for model_name, model_results in pretrained_results.items():
        if model_results.get("success"):
            exec_time = model_results["execution_time"]
            hurst_est = model_results["hurst_estimate"]
            print(f"  {model_name}: H={hurst_est:.3f}, Time={exec_time:.3f}s")
    
    # Issues and recommendations
    print(f"\n⚠️ ISSUES IDENTIFIED:")
    issues = []
    
    # Check for import errors
    for model_name, model_results in results.items():
        if "import_error" in model_results:
            issues.append(f"Import error in {model_name}: {model_results['import_error']}")
    
    # Check for high fallback usage
    for model_name, model_results in results.items():
        if "import_error" not in model_results:
            fallback_count = 0
            total_tests = 0
            for framework, framework_results in model_results.items():
                if "error" not in framework_results:
                    for data_name, data_results in framework_results.get("data_results", {}).items():
                        total_tests += 1
                        if data_results.get("fallback_used", False):
                            fallback_count += 1
            
            if total_tests > 0 and fallback_count / total_tests > 0.5:
                issues.append(f"High fallback usage in {model_name}: {fallback_count/total_tests*100:.1f}%")
    
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  No major issues identified")
    
    print(f"\n💡 RECOMMENDATIONS:")
    recommendations = [
        "1. Implement proper training pipelines for all ML models",
        "2. Add comprehensive parameter validation",
        "3. Implement proper feature extraction for time series data",
        "4. Add model persistence and loading mechanisms",
        "5. Implement proper error handling and logging",
        "6. Add comprehensive unit tests for all models",
        "7. Optimize neural network architectures for time series",
        "8. Implement proper data preprocessing pipelines",
        "9. Add model evaluation and validation metrics",
        "10. Implement proper GPU/CPU optimization"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")

def main():
    """Main analysis function."""
    
    # Run comprehensive analysis
    results = analyze_ml_models()
    pretrained_results = analyze_pretrained_models()
    evaluate_architecture_quality()
    generate_analysis_report(results, pretrained_results)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
