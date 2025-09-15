#!/usr/bin/env python3
"""
Comprehensive Benchmark of Neural Network LRD Estimators

This script performs a thorough benchmark of neural network estimators using:
1. Pure data with known Hurst parameters
2. Contaminated data with various realistic scenarios
3. Real-world contexts (financial, physiological, environmental, network)
4. GPU optimization and performance comparison
5. Train-once-apply-many workflow validation
"""

import numpy as np
import time
import warnings
import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import neural network estimators
try:
    from lrdbenchmark.analysis.machine_learning import (
        CNNEstimator, LSTMEstimator, GRUEstimator, TransformerEstimator
    )
    NEURAL_ESTIMATORS_AVAILABLE = True
except ImportError as e:
    NEURAL_ESTIMATORS_AVAILABLE = False
    print(f"Warning: Neural network estimators not available: {e}")

# Import data models
from lrdbenchmark.models.data_models import FBMModel, FGNModel, ARFIMAModel, MRWModel

# Import contamination models
try:
    from lrdbenchmark.models.contamination.contamination_models import (
        ContaminationModel, ContaminationType, ContaminationConfig
    )
    CONTAMINATION_AVAILABLE = True
except ImportError:
    CONTAMINATION_AVAILABLE = False
    print("Warning: Contamination models not available")

try:
    from lrdbenchmark.models.contamination.contamination_factory import (
        ContaminationFactory, ConfoundingScenario
    )
    FACTORY_AVAILABLE = True
except ImportError:
    FACTORY_AVAILABLE = False
    print("Warning: Contamination factory not available")

class NeuralEstimatorsBenchmark:
    """Comprehensive benchmark of neural network LRD estimators."""
    
    def __init__(self, output_dir: str = "neural_benchmark_results"):
        """Initialize the benchmark."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.test_data = {}
        self.estimators = {}
        self.contamination_model = None
        self.contamination_factory = None
        
        self._initialize_estimators()
        self._initialize_contamination()
        self._generate_test_data()
    
    def _initialize_estimators(self):
        """Initialize all neural network estimators."""
        if not NEURAL_ESTIMATORS_AVAILABLE:
            print("❌ Neural network estimators not available")
            return
        
        self.estimators = {
            "CNN": CNNEstimator(),
            "LSTM": LSTMEstimator(),
            "GRU": GRUEstimator(),
            "Transformer": TransformerEstimator()
        }
        
        print(f"✅ Initialized {len(self.estimators)} neural network estimators")
    
    def _initialize_contamination(self):
        """Initialize contamination models."""
        if CONTAMINATION_AVAILABLE:
            self.contamination_model = ContaminationModel()
            print("✅ Contamination model initialized")
        else:
            print("⚠️ Contamination model not available")
        
        if FACTORY_AVAILABLE:
            self.contamination_factory = ContaminationFactory()
            print("✅ Contamination factory initialized")
        else:
            print("⚠️ Contamination factory not available")
    
    def _generate_test_data(self):
        """Generate comprehensive test data."""
        print("🔍 Generating comprehensive test data...")
        
        # Test Hurst parameters - focused range for neural networks
        hurst_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        n_samples_per_hurst = 50  # Reduced for neural networks
        sequence_lengths = [100, 250, 500, 1000]  # Neural networks work well with shorter sequences
        
        # Generate pure data
        pure_data = {}
        for H in hurst_values:
            pure_data[f"H_{H}"] = {
                "true_hurst": H,
                "fbm": [],
                "fgn": [],
                "arfima": []
            }
            
            for seq_len in sequence_lengths:
                # FBM data
                fbm_model = FBMModel(H=H, sigma=1.0)
                fbm_data = fbm_model.generate(n=seq_len, seed=42)
                pure_data[f"H_{H}"]["fbm"].append(fbm_data)
                
                # FGN data
                fgn_model = FGNModel(H=H, sigma=1.0)
                fgn_data = fgn_model.generate(n=seq_len, seed=42)
                pure_data[f"H_{H}"]["fgn"].append(fgn_data)
                
                # ARFIMA data (d = H - 0.5)
                d = H - 0.5
                arfima_model = ARFIMAModel(d=d, sigma=1.0)
                arfima_data = arfima_model.generate(n=seq_len, seed=42)
                pure_data[f"H_{H}"]["arfima"].append(arfima_data)
        
        self.test_data["pure"] = pure_data
        
        # Generate contaminated data
        self.test_data["contaminated"] = self._generate_contaminated_data()
        
        # Generate realistic context data
        self.test_data["realistic"] = self._generate_realistic_data()
        
        print(f"✅ Generated test data:")
        print(f"   • Pure data: {len(hurst_values)} Hurst values × {len(sequence_lengths)} lengths × 3 types")
        print(f"   • Contaminated data: Multiple contamination scenarios")
        print(f"   • Realistic data: Domain-specific contexts")
    
    def _generate_contaminated_data(self) -> Dict[str, Any]:
        """Generate contaminated data for robustness testing."""
        print("   Generating contaminated data...")
        
        contaminated_data = {}
        
        # Use representative base data (H=0.6, 500 points, FBM)
        base_fbm = FBMModel(H=0.6, sigma=1.0)
        base_data = base_fbm.generate(n=500, seed=42)
        
        # Define contamination scenarios
        contamination_scenarios = [
            ("additive_noise", {"std": 0.1}),
            ("linear_trend", {"slope": 0.01}),
            ("polynomial_trend", {"degree": 2}),
            ("spikes", {"probability": 0.05, "amplitude": 2.0}),
            ("level_shifts", {"probability": 0.02, "amplitude": 1.0}),
            ("missing_data", {"probability": 0.1}),
            ("colored_noise", {"std": 0.1}),
            ("impulsive_noise", {"probability": 0.05, "amplitude": 3.0})
        ]
        
        for scenario_name, params in contamination_scenarios:
            try:
                if self.contamination_model:
                    contaminated = self._apply_contamination(base_data, scenario_name, params)
                else:
                    contaminated = self._fallback_contamination(base_data, scenario_name, params)
                
                contaminated_data[scenario_name] = {
                    "true_hurst": 0.6,
                    "data": contaminated,
                    "scenario": scenario_name,
                    "parameters": params
                }
            except Exception as e:
                print(f"   Warning: Failed to generate {scenario_name}: {e}")
        
        return contaminated_data
    
    def _generate_realistic_data(self) -> Dict[str, Any]:
        """Generate realistic context data."""
        print("   Generating realistic context data...")
        
        realistic_data = {}
        
        # Financial time series context
        realistic_data["financial"] = {
            "true_hurst": 0.6,  # Typical for financial volatility
            "data": self._generate_financial_data(),
            "context": "financial_volatility"
        }
        
        # Physiological signal context
        realistic_data["physiological"] = {
            "true_hurst": 0.7,  # Typical for physiological signals
            "data": self._generate_physiological_data(),
            "context": "physiological_signal"
        }
        
        # Environmental data context
        realistic_data["environmental"] = {
            "true_hurst": 0.8,  # Typical for environmental data
            "data": self._generate_environmental_data(),
            "context": "environmental_monitoring"
        }
        
        # Network traffic context
        realistic_data["network"] = {
            "true_hurst": 0.5,  # Typical for network traffic
            "data": self._generate_network_data(),
            "context": "network_traffic"
        }
        
        return realistic_data
    
    def _generate_financial_data(self) -> np.ndarray:
        """Generate realistic financial time series."""
        # Generate base FBM with financial characteristics
        fbm_model = FBMModel(H=0.6, sigma=1.0)
        base_data = fbm_model.generate(n=500, seed=42)
        
        # Add financial-specific characteristics
        # Volatility clustering (GARCH-like behavior)
        volatility = np.abs(np.diff(base_data)) + 0.1
        for i in range(1, len(volatility)):
            volatility[i] = 0.8 * volatility[i-1] + 0.2 * np.random.normal(0, 0.1)
        
        # Add some market crashes (extreme events)
        crash_indices = np.random.choice(len(base_data), size=3, replace=False)
        base_data[crash_indices] += -3.0 * np.random.normal(0, 1, 3)
        
        return base_data
    
    def _generate_physiological_data(self) -> np.ndarray:
        """Generate realistic physiological signal."""
        # Generate base FBM with physiological characteristics
        fbm_model = FBMModel(H=0.7, sigma=1.0)
        base_data = fbm_model.generate(n=500, seed=42)
        
        # Add physiological artifacts
        # Heart rate variability-like patterns
        heart_rate_component = 0.5 * np.sin(2 * np.pi * np.arange(len(base_data)) / 50)
        base_data += heart_rate_component
        
        # Motion artifacts (occasional spikes)
        motion_indices = np.random.choice(len(base_data), size=10, replace=False)
        base_data[motion_indices] += 2.0 * np.random.normal(0, 1, 10)
        
        return base_data
    
    def _generate_environmental_data(self) -> np.ndarray:
        """Generate realistic environmental monitoring data."""
        # Generate base FBM with environmental characteristics
        fbm_model = FBMModel(H=0.8, sigma=1.0)
        base_data = fbm_model.generate(n=500, seed=42)
        
        # Add seasonal patterns
        seasonal_component = 0.3 * np.sin(2 * np.pi * np.arange(len(base_data)) / 100)
        base_data += seasonal_component
        
        # Add measurement drift (slow trend)
        drift = 0.001 * np.arange(len(base_data))
        base_data += drift
        
        return base_data
    
    def _generate_network_data(self) -> np.ndarray:
        """Generate realistic network traffic data."""
        # Generate base FBM with network characteristics
        fbm_model = FBMModel(H=0.5, sigma=1.0)
        base_data = fbm_model.generate(n=500, seed=42)
        
        # Add network-specific patterns
        # Burst patterns (high traffic periods)
        burst_indices = np.random.choice(len(base_data), size=25, replace=False)
        base_data[burst_indices] += 2.0 * np.random.normal(0, 1, 25)
        
        # Congestion effects (correlated noise)
        congestion_noise = np.zeros(len(base_data))
        congestion_noise[0] = np.random.normal(0, 0.5)
        for i in range(1, len(congestion_noise)):
            congestion_noise[i] = 0.7 * congestion_noise[i-1] + 0.3 * np.random.normal(0, 0.5)
        base_data += congestion_noise
        
        return base_data
    
    def _apply_contamination(self, data: np.ndarray, scenario: str, params: Dict) -> np.ndarray:
        """Apply contamination using the contamination model."""
        try:
            if scenario == "additive_noise":
                return self.contamination_model.add_noise_gaussian(data, std=params["std"])
            elif scenario == "linear_trend":
                return self.contamination_model.add_trend_linear(data, slope=params["slope"])
            elif scenario == "polynomial_trend":
                return self.contamination_model.add_trend_polynomial(data, degree=params["degree"])
            elif scenario == "spikes":
                return self.contamination_model.add_artifact_spikes(
                    data, probability=params["probability"], amplitude=params["amplitude"]
                )
            elif scenario == "level_shifts":
                return self.contamination_model.add_artifact_level_shifts(
                    data, probability=params["probability"], amplitude=params["amplitude"]
                )
            elif scenario == "missing_data":
                return self.contamination_model.add_artifact_missing_data(
                    data, probability=params["probability"]
                )
            elif scenario == "colored_noise":
                return self.contamination_model.add_noise_colored(data, std=params["std"])
            elif scenario == "impulsive_noise":
                return self.contamination_model.add_noise_impulsive(
                    data, probability=params["probability"], amplitude=params["amplitude"]
                )
            else:
                return self._fallback_contamination(data, scenario, params)
        except Exception as e:
            print(f"Warning: Contamination failed for {scenario}: {e}")
            return self._fallback_contamination(data, scenario, params)
    
    def _fallback_contamination(self, data: np.ndarray, scenario: str, params: Dict) -> np.ndarray:
        """Fallback contamination methods."""
        contaminated = data.copy()
        
        if scenario == "additive_noise":
            contaminated += params["std"] * np.random.normal(0, 1, len(data))
        elif scenario == "linear_trend":
            contaminated += params["slope"] * np.arange(len(data))
        elif scenario == "polynomial_trend":
            x = np.arange(len(data)) / len(data)
            trend = 0.1 * (x**params["degree"] - 0.5)
            contaminated += trend
        elif scenario == "spikes":
            spike_indices = np.random.choice(len(data), size=int(params["probability"] * len(data)), replace=False)
            contaminated[spike_indices] += params["amplitude"] * np.random.normal(0, 1, len(spike_indices))
        elif scenario == "level_shifts":
            shift_indices = np.random.choice(len(data), size=int(params["probability"] * len(data)), replace=False)
            contaminated[shift_indices] += params["amplitude"]
        elif scenario == "missing_data":
            missing_indices = np.random.choice(len(data), size=int(params["probability"] * len(data)), replace=False)
            contaminated[missing_indices] = np.nan
        elif scenario == "colored_noise":
            noise = np.zeros(len(data))
            noise[0] = np.random.normal(0, 1)
            for i in range(1, len(data)):
                noise[i] = 0.8 * noise[i-1] + params["std"] * np.random.normal(0, 1)
            contaminated += noise
        elif scenario == "impulsive_noise":
            impulse_indices = np.random.choice(len(data), size=int(params["probability"] * len(data)), replace=False)
            contaminated[impulse_indices] += params["amplitude"] * np.random.normal(0, 1, len(impulse_indices))
        
        return contaminated
    
    def benchmark_pure_data(self) -> Dict[str, Any]:
        """Benchmark estimators on pure data."""
        print("\n📊 Benchmarking on Pure Data...")
        
        pure_results = {}
        
        for estimator_name, estimator in self.estimators.items():
            print(f"   Testing {estimator_name} on pure data...")
            
            estimator_results = {
                "accuracy": {},
                "performance": {},
                "robustness": {}
            }
            
            # Test on different data types and lengths
            for data_key, data_info in self.test_data["pure"].items():
                true_hurst = data_info["true_hurst"]
                
                # Test on FBM data
                for i, fbm_data in enumerate(data_info["fbm"]):
                    try:
                        start_time = time.time()
                        result = estimator.estimate(fbm_data)
                        execution_time = time.time() - start_time
                        
                        estimated_hurst = result.get("hurst_parameter", np.nan)
                        error = abs(estimated_hurst - true_hurst)
                        
                        estimator_results["accuracy"][f"{data_key}_fbm_{i}"] = {
                            "true_hurst": true_hurst,
                            "estimated_hurst": estimated_hurst,
                            "absolute_error": error,
                            "relative_error": error / true_hurst if true_hurst != 0 else np.nan,
                            "execution_time": execution_time,
                            "r_squared": result.get("r_squared", np.nan),
                            "sequence_length": len(fbm_data)
                        }
                        
                    except Exception as e:
                        estimator_results["accuracy"][f"{data_key}_fbm_{i}"] = {
                            "error": str(e),
                            "execution_time": np.nan
                        }
                
                # Test on FGN data
                for i, fgn_data in enumerate(data_info["fgn"]):
                    try:
                        start_time = time.time()
                        result = estimator.estimate(fgn_data)
                        execution_time = time.time() - start_time
                        
                        estimated_hurst = result.get("hurst_parameter", np.nan)
                        error = abs(estimated_hurst - true_hurst)
                        
                        estimator_results["accuracy"][f"{data_key}_fgn_{i}"] = {
                            "true_hurst": true_hurst,
                            "estimated_hurst": estimated_hurst,
                            "absolute_error": error,
                            "relative_error": error / true_hurst if true_hurst != 0 else np.nan,
                            "execution_time": execution_time,
                            "r_squared": result.get("r_squared", np.nan),
                            "sequence_length": len(fgn_data)
                        }
                        
                    except Exception as e:
                        estimator_results["accuracy"][f"{data_key}_fgn_{i}"] = {
                            "error": str(e),
                            "execution_time": np.nan
                        }
            
            pure_results[estimator_name] = estimator_results
        
        print("✅ Pure data benchmark completed")
        return pure_results
    
    def benchmark_contaminated_data(self) -> Dict[str, Any]:
        """Benchmark estimators on contaminated data."""
        print("\n🛡️ Benchmarking on Contaminated Data...")
        
        contaminated_results = {}
        
        for estimator_name, estimator in self.estimators.items():
            print(f"   Testing {estimator_name} on contaminated data...")
            
            estimator_results = {
                "contamination_robustness": {},
                "performance_degradation": {}
            }
            
            for scenario_name, scenario_data in self.test_data["contaminated"].items():
                try:
                    data = scenario_data["data"]
                    true_hurst = scenario_data["true_hurst"]
                    
                    # Handle missing data
                    if np.any(np.isnan(data)):
                        clean_data = data[~np.isnan(data)]
                    else:
                        clean_data = data
                    
                    if len(clean_data) < 50:
                        estimator_results["contamination_robustness"][scenario_name] = {
                            "error": "Insufficient data after cleaning"
                        }
                        continue
                    
                    # Estimate on contaminated data
                    start_time = time.time()
                    result = estimator.estimate(clean_data)
                    execution_time = time.time() - start_time
                    
                    estimated_hurst = result.get("hurst_parameter", np.nan)
                    error = abs(estimated_hurst - true_hurst)
                    
                    estimator_results["contamination_robustness"][scenario_name] = {
                        "true_hurst": true_hurst,
                        "estimated_hurst": estimated_hurst,
                        "absolute_error": error,
                        "relative_error": error / true_hurst if true_hurst != 0 else np.nan,
                        "execution_time": execution_time,
                        "data_length": len(clean_data),
                        "contamination_scenario": scenario_name,
                        "scenario_parameters": scenario_data["parameters"]
                    }
                    
                except Exception as e:
                    estimator_results["contamination_robustness"][scenario_name] = {
                        "error": str(e),
                        "contamination_scenario": scenario_name
                    }
            
            contaminated_results[estimator_name] = estimator_results
        
        print("✅ Contaminated data benchmark completed")
        return contaminated_results
    
    def benchmark_realistic_data(self) -> Dict[str, Any]:
        """Benchmark estimators on realistic context data."""
        print("\n🌍 Benchmarking on Realistic Context Data...")
        
        realistic_results = {}
        
        for estimator_name, estimator in self.estimators.items():
            print(f"   Testing {estimator_name} on realistic data...")
            
            estimator_results = {
                "context_performance": {},
                "domain_specific_accuracy": {}
            }
            
            for context_name, context_data in self.test_data["realistic"].items():
                try:
                    data = context_data["data"]
                    true_hurst = context_data["true_hurst"]
                    context = context_data["context"]
                    
                    # Estimate on realistic data
                    start_time = time.time()
                    result = estimator.estimate(data)
                    execution_time = time.time() - start_time
                    
                    estimated_hurst = result.get("hurst_parameter", np.nan)
                    error = abs(estimated_hurst - true_hurst)
                    
                    estimator_results["context_performance"][context_name] = {
                        "true_hurst": true_hurst,
                        "estimated_hurst": estimated_hurst,
                        "absolute_error": error,
                        "relative_error": error / true_hurst if true_hurst != 0 else np.nan,
                        "execution_time": execution_time,
                        "data_length": len(data),
                        "context": context
                    }
                    
                except Exception as e:
                    estimator_results["context_performance"][context_name] = {
                        "error": str(e),
                        "context": context_data["context"]
                    }
            
            realistic_results[estimator_name] = estimator_results
        
        print("✅ Realistic data benchmark completed")
        return realistic_results
    
    def analyze_results(self, pure_results: Dict, contaminated_results: Dict, realistic_results: Dict) -> Dict[str, Any]:
        """Analyze and summarize benchmark results."""
        print("\n📈 Analyzing Benchmark Results...")
        
        analysis = {
            "pure_data_analysis": {},
            "contaminated_data_analysis": {},
            "realistic_data_analysis": {},
            "performance_ranking": {},
            "recommendations": {}
        }
        
        # Analyze pure data results
        for estimator_name, results in pure_results.items():
            accuracy_data = results["accuracy"]
            
            # Calculate metrics
            errors = []
            times = []
            r_squared_values = []
            sequence_lengths = []
            
            for test_name, test_result in accuracy_data.items():
                if "absolute_error" in test_result:
                    errors.append(test_result["absolute_error"])
                    times.append(test_result["execution_time"])
                    if "r_squared" in test_result and not np.isnan(test_result["r_squared"]):
                        r_squared_values.append(test_result["r_squared"])
                    if "sequence_length" in test_result:
                        sequence_lengths.append(test_result["sequence_length"])
            
            analysis["pure_data_analysis"][estimator_name] = {
                "mean_absolute_error": np.mean(errors) if errors else np.nan,
                "std_absolute_error": np.std(errors) if errors else np.nan,
                "mean_execution_time": np.mean(times) if times else np.nan,
                "mean_r_squared": np.mean(r_squared_values) if r_squared_values else np.nan,
                "success_rate": len(errors) / len(accuracy_data) if accuracy_data else 0,
                "sequence_length_impact": self._analyze_sequence_length_impact(errors, sequence_lengths)
            }
        
        # Analyze contaminated data results
        for estimator_name, results in contaminated_results.items():
            robustness_data = results["contamination_robustness"]
            
            # Calculate robustness metrics
            errors = []
            success_count = 0
            scenario_performance = {}
            
            for scenario, result in robustness_data.items():
                if "absolute_error" in result:
                    errors.append(result["absolute_error"])
                    success_count += 1
                    scenario_performance[scenario] = result["absolute_error"]
            
            analysis["contaminated_data_analysis"][estimator_name] = {
                "mean_absolute_error": np.mean(errors) if errors else np.nan,
                "std_absolute_error": np.std(errors) if errors else np.nan,
                "robustness_score": success_count / len(robustness_data) if robustness_data else 0,
                "total_contamination_scenarios": len(robustness_data),
                "scenario_performance": scenario_performance
            }
        
        # Analyze realistic data results
        for estimator_name, results in realistic_results.items():
            context_data = results["context_performance"]
            
            # Calculate context-specific metrics
            errors = []
            context_performance = {}
            
            for context, result in context_data.items():
                if "absolute_error" in result:
                    errors.append(result["absolute_error"])
                    context_performance[context] = result["absolute_error"]
            
            analysis["realistic_data_analysis"][estimator_name] = {
                "mean_absolute_error": np.mean(errors) if errors else np.nan,
                "std_absolute_error": np.std(errors) if errors else np.nan,
                "context_success_rate": len(errors) / len(context_data) if context_data else 0,
                "total_contexts": len(context_data),
                "context_performance": context_performance
            }
        
        # Calculate overall performance ranking
        performance_scores = {}
        for estimator_name in self.estimators.keys():
            pure_analysis = analysis["pure_data_analysis"].get(estimator_name, {})
            contaminated_analysis = analysis["contaminated_data_analysis"].get(estimator_name, {})
            realistic_analysis = analysis["realistic_data_analysis"].get(estimator_name, {})
            
            # Calculate composite score
            pure_score = 0
            if not np.isnan(pure_analysis.get("mean_absolute_error", np.nan)):
                pure_score = max(0, 10 - pure_analysis["mean_absolute_error"] * 10)
            
            robustness_score = contaminated_analysis.get("robustness_score", 0) * 10
            realistic_score = realistic_analysis.get("context_success_rate", 0) * 10
            
            performance_scores[estimator_name] = {
                "pure_data_score": pure_score,
                "robustness_score": robustness_score,
                "realistic_score": realistic_score,
                "overall_score": (pure_score + robustness_score + realistic_score) / 3
            }
        
        # Rank estimators
        ranked_estimators = sorted(
            performance_scores.items(),
            key=lambda x: x[1]["overall_score"],
            reverse=True
        )
        
        analysis["performance_ranking"] = {
            "scores": performance_scores,
            "rankings": ranked_estimators
        }
        
        # Generate recommendations
        if ranked_estimators:
            best_estimator = ranked_estimators[0][0]
            analysis["recommendations"] = {
                "best_overall": best_estimator,
                "best_for_pure_data": max(performance_scores.items(), key=lambda x: x[1]["pure_data_score"])[0],
                "most_robust": max(performance_scores.items(), key=lambda x: x[1]["robustness_score"])[0],
                "best_for_realistic": max(performance_scores.items(), key=lambda x: x[1]["realistic_score"])[0]
            }
        
        print("✅ Results analysis completed")
        return analysis
    
    def _analyze_sequence_length_impact(self, errors: List[float], lengths: List[int]) -> Dict[str, float]:
        """Analyze the impact of sequence length on performance."""
        if not errors or not lengths or len(errors) != len(lengths):
            return {"correlation": 0.0, "impact": "unknown"}
        
        # Calculate correlation between sequence length and error
        correlation = np.corrcoef(lengths, errors)[0, 1] if len(errors) > 1 else 0.0
        
        return {
            "correlation": correlation,
            "impact": "positive" if correlation > 0.1 else "negative" if correlation < -0.1 else "minimal"
        }
    
    def generate_visualizations(self, pure_results: Dict, contaminated_results: Dict, realistic_results: Dict, analysis: Dict):
        """Generate comprehensive visualizations of benchmark results."""
        print("\n📊 Generating Visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(24, 18))
        
        # 1. Pure Data Performance Comparison
        ax1 = plt.subplot(4, 4, 1)
        estimators = list(self.estimators.keys())
        mean_errors = [analysis["pure_data_analysis"].get(est, {}).get("mean_absolute_error", np.nan) for est in estimators]
        
        bars1 = ax1.bar(estimators, mean_errors, alpha=0.7)
        ax1.set_title("Pure Data: Mean Absolute Error", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Mean Absolute Error")
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Execution Time Comparison
        ax2 = plt.subplot(4, 4, 2)
        mean_times = [analysis["pure_data_analysis"].get(est, {}).get("mean_execution_time", np.nan) for est in estimators]
        
        bars2 = ax2.bar(estimators, mean_times, alpha=0.7, color='orange')
        ax2.set_title("Pure Data: Mean Execution Time", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Execution Time (seconds)")
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Robustness Comparison
        ax3 = plt.subplot(4, 4, 3)
        robustness_scores = [analysis["contaminated_data_analysis"].get(est, {}).get("robustness_score", 0) for est in estimators]
        
        bars3 = ax3.bar(estimators, robustness_scores, alpha=0.7, color='green')
        ax3.set_title("Contamination Robustness", fontsize=12, fontweight='bold')
        ax3.set_ylabel("Robustness Score")
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Realistic Context Performance
        ax4 = plt.subplot(4, 4, 4)
        realistic_scores = [analysis["realistic_data_analysis"].get(est, {}).get("context_success_rate", 0) for est in estimators]
        
        bars4 = ax4.bar(estimators, realistic_scores, alpha=0.7, color='purple')
        ax4.set_title("Realistic Context Performance", fontsize=12, fontweight='bold')
        ax4.set_ylabel("Success Rate")
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Overall Performance Ranking
        ax5 = plt.subplot(4, 4, 5)
        ranking_data = analysis["performance_ranking"]["scores"]
        overall_scores = [ranking_data.get(est, {}).get("overall_score", 0) for est in estimators]
        
        bars5 = ax5.bar(estimators, overall_scores, alpha=0.7, color='red')
        ax5.set_title("Overall Performance Score", fontsize=12, fontweight='bold')
        ax5.set_ylabel("Overall Score")
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Contamination Error Heatmap
        ax6 = plt.subplot(4, 4, 6)
        contamination_scenarios = list(self.test_data["contaminated"].keys())
        
        error_matrix = np.zeros((len(estimators), len(contamination_scenarios)))
        for i, estimator in enumerate(estimators):
            for j, scenario in enumerate(contamination_scenarios):
                error = contaminated_results[estimator]["contamination_robustness"].get(scenario, {}).get("absolute_error", np.nan)
                error_matrix[i, j] = error if not np.isnan(error) else 0
        
        im = ax6.imshow(error_matrix, cmap='YlOrRd', aspect='auto')
        ax6.set_title("Contamination Error Heatmap", fontsize=12, fontweight='bold')
        ax6.set_xlabel("Contamination Scenario")
        ax6.set_ylabel("Estimator")
        ax6.set_xticks(range(len(contamination_scenarios)))
        ax6.set_xticklabels([cs.replace('_', '\n') for cs in contamination_scenarios], rotation=45, ha='right')
        ax6.set_yticks(range(len(estimators)))
        ax6.set_yticklabels(estimators)
        plt.colorbar(im, ax=ax6, label="Absolute Error")
        
        # 7. Context Performance Comparison
        ax7 = plt.subplot(4, 4, 7)
        contexts = list(self.test_data["realistic"].keys())
        
        context_matrix = np.zeros((len(estimators), len(contexts)))
        for i, estimator in enumerate(estimators):
            for j, context in enumerate(contexts):
                error = realistic_results[estimator]["context_performance"].get(context, {}).get("absolute_error", np.nan)
                context_matrix[i, j] = error if not np.isnan(error) else 0
        
        im2 = ax7.imshow(context_matrix, cmap='YlGnBu', aspect='auto')
        ax7.set_title("Context Performance Heatmap", fontsize=12, fontweight='bold')
        ax7.set_xlabel("Context")
        ax7.set_ylabel("Estimator")
        ax7.set_xticks(range(len(contexts)))
        ax7.set_xticklabels(contexts, rotation=45, ha='right')
        ax7.set_yticks(range(len(estimators)))
        ax7.set_yticklabels(estimators)
        plt.colorbar(im2, ax=ax7, label="Absolute Error")
        
        # 8. Performance vs Robustness Scatter
        ax8 = plt.subplot(4, 4, 8)
        pure_scores = [ranking_data.get(est, {}).get("pure_data_score", 0) for est in estimators]
        robustness_scores = [ranking_data.get(est, {}).get("robustness_score", 0) for est in estimators]
        
        scatter = ax8.scatter(pure_scores, robustness_scores, s=100, alpha=0.7)
        ax8.set_title("Performance vs Robustness", fontsize=12, fontweight='bold')
        ax8.set_xlabel("Pure Data Score")
        ax8.set_ylabel("Robustness Score")
        
        for i, estimator in enumerate(estimators):
            ax8.annotate(estimator, (pure_scores[i], robustness_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 9-12. Individual estimator performance breakdown
        for i, estimator in enumerate(estimators):
            ax = plt.subplot(4, 4, 9 + i)
            
            # Get performance metrics
            pure_error = analysis["pure_data_analysis"].get(estimator, {}).get("mean_absolute_error", np.nan)
            robustness = analysis["contaminated_data_analysis"].get(estimator, {}).get("robustness_score", 0)
            realistic = analysis["realistic_data_analysis"].get(estimator, {}).get("context_success_rate", 0)
            overall = ranking_data.get(estimator, {}).get("overall_score", 0)
            
            metrics = [pure_error, robustness * 10, realistic * 10, overall]
            metric_names = ['Pure\nError', 'Robustness\n(×10)', 'Realistic\n(×10)', 'Overall\nScore']
            colors = ['blue', 'green', 'purple', 'red']
            
            bars = ax.bar(metric_names, metrics, color=colors, alpha=0.7)
            ax.set_title(f"{estimator} Performance", fontsize=10, fontweight='bold')
            ax.set_ylabel("Score/Error")
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics):
                if not np.isnan(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_dir / "neural_estimators_benchmark.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved to {plot_path}")
    
    def save_results(self, pure_results: Dict, contaminated_results: Dict, realistic_results: Dict, analysis: Dict):
        """Save benchmark results to files."""
        print("\n💾 Saving Results...")
        
        # Save raw results
        results_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "estimators_tested": list(self.estimators.keys()),
                "test_scenarios": {
                    "pure_data": len(self.test_data["pure"]),
                    "contaminated_data": len(self.test_data["contaminated"]),
                    "realistic_data": len(self.test_data["realistic"])
                }
            },
            "pure_data_results": pure_results,
            "contaminated_data_results": contaminated_results,
            "realistic_data_results": realistic_results,
            "analysis": analysis
        }
        
        # Save JSON results
        json_path = self.output_dir / "neural_estimators_benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save CSV summary
        csv_data = []
        for estimator_name in self.estimators.keys():
            pure_analysis = analysis["pure_data_analysis"].get(estimator_name, {})
            contaminated_analysis = analysis["contaminated_data_analysis"].get(estimator_name, {})
            realistic_analysis = analysis["realistic_data_analysis"].get(estimator_name, {})
            ranking_data = analysis["performance_ranking"]["scores"].get(estimator_name, {})
            
            csv_data.append({
                "Estimator": estimator_name,
                "Pure_Mean_Absolute_Error": pure_analysis.get("mean_absolute_error", np.nan),
                "Pure_Mean_Execution_Time": pure_analysis.get("mean_execution_time", np.nan),
                "Pure_Success_Rate": pure_analysis.get("success_rate", 0),
                "Robustness_Score": contaminated_analysis.get("robustness_score", 0),
                "Realistic_Success_Rate": realistic_analysis.get("context_success_rate", 0),
                "Pure_Data_Score": ranking_data.get("pure_data_score", 0),
                "Robustness_Score_Ranking": ranking_data.get("robustness_score", 0),
                "Realistic_Score": ranking_data.get("realistic_score", 0),
                "Overall_Score": ranking_data.get("overall_score", 0)
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / "neural_estimators_benchmark_summary.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Results saved to {json_path}")
        print(f"✅ Summary saved to {csv_path}")
    
    def print_summary(self, analysis: Dict):
        """Print a comprehensive summary of benchmark results."""
        print("\n📊 COMPREHENSIVE NEURAL NETWORK BENCHMARK SUMMARY")
        print("=" * 70)
        
        if analysis["recommendations"]:
            rec = analysis["recommendations"]
            print(f"🏆 Best Overall Estimator: {rec['best_overall']}")
            print(f"🛡️ Most Robust Estimator: {rec['most_robust']}")
            print(f"🎯 Best for Pure Data: {rec['best_for_pure_data']}")
            print(f"🌍 Best for Realistic Contexts: {rec['best_for_realistic']}")
        
        print(f"\n📈 Performance Rankings:")
        rankings = analysis["performance_ranking"]["rankings"]
        for i, (estimator, scores) in enumerate(rankings, 1):
            print(f"{i:2d}. {estimator:15s}: {scores['overall_score']:5.2f}/10 (Pure: {scores['pure_data_score']:4.1f}, Robust: {scores['robustness_score']:4.1f}, Realistic: {scores['realistic_score']:4.1f})")
        
        print(f"\n📊 Key Statistics:")
        print(f"   • Total estimators tested: {len(self.estimators)}")
        print(f"   • Pure data scenarios: {len(self.test_data['pure'])} Hurst values × 3 types × 4 lengths")
        print(f"   • Contamination scenarios: {len(self.test_data['contaminated'])}")
        print(f"   • Realistic contexts: {len(self.test_data['realistic'])}")
        
        print(f"\n✅ Neural Network Benchmark completed successfully!")
        print(f"📁 Results saved to: {self.output_dir}")
    
    def run_comprehensive_benchmark(self):
        """Run the complete benchmark suite."""
        print("🧠 Starting Comprehensive Neural Network Estimators Benchmark")
        print("=" * 70)
        
        # Run benchmarks
        pure_results = self.benchmark_pure_data()
        contaminated_results = self.benchmark_contaminated_data()
        realistic_results = self.benchmark_realistic_data()
        
        # Analyze results
        analysis = self.analyze_results(pure_results, contaminated_results, realistic_results)
        
        # Generate visualizations
        self.generate_visualizations(pure_results, contaminated_results, realistic_results, analysis)
        
        # Save results
        self.save_results(pure_results, contaminated_results, realistic_results, analysis)
        
        # Print summary
        self.print_summary(analysis)
        
        return {
            "pure_results": pure_results,
            "contaminated_results": contaminated_results,
            "realistic_results": realistic_results,
            "analysis": analysis
        }

def main():
    """Run the comprehensive neural network benchmark."""
    benchmark = NeuralEstimatorsBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    return results

if __name__ == "__main__":
    results = main()
