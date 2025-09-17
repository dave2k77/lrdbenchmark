#!/usr/bin/env python3
"""
Comprehensive Benchmark of Classical LRD Estimators

This script performs a thorough benchmark of classical LRD estimators using both
pure data and contaminated data to assess their robustness and performance
under realistic conditions.
"""

import numpy as np
import time
import warnings
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Import data models
from lrdbenchmark.models.data_models import FBMModel, FGNModel, ARFIMAModel, MRWModel

# Import classical estimators
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.temporal.dma.dma_estimator_unified import DMAEstimator
from lrdbenchmark.analysis.temporal.higuchi.higuchi_estimator_unified import HiguchiEstimator
from lrdbenchmark.analysis.spectral.gph.gph_estimator_unified import GPHEstimator
from lrdbenchmark.analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator
from lrdbenchmark.analysis.spectral.periodogram.periodogram_estimator_unified import PeriodogramEstimator
from lrdbenchmark.analysis.wavelet.cwt.cwt_estimator_unified import CWTEstimator

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

class ClassicalEstimatorsBenchmark:
    """Comprehensive benchmark of classical LRD estimators."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize the benchmark."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.test_data = {}
        self.estimators = {}
        self.contamination_model = None
        
        self._initialize_estimators()
        self._initialize_contamination()
        self._generate_test_data()
    
    def _initialize_estimators(self):
        """Initialize all classical estimators."""
        self.estimators = {
            # Temporal estimators
            "R/S": RSEstimator(min_block_size=10, num_blocks=15),
            "DFA": DFAEstimator(min_scale=10, num_scales=15, order=1),
            "DMA": DMAEstimator(min_scale=10, num_scales=15),
            "Higuchi": HiguchiEstimator(min_k=2, max_k=20),
            
            # Spectral estimators
            "GPH": GPHEstimator(min_freq_ratio=0.01, max_freq_ratio=0.1),
            "Whittle": WhittleEstimator(),
            "Periodogram": PeriodogramEstimator(),
            
            # Wavelet estimators
            "CWT": CWTEstimator(),
        }
        
        print(f"✅ Initialized {len(self.estimators)} classical estimators")
    
    def _initialize_contamination(self):
        """Initialize contamination models."""
        if CONTAMINATION_AVAILABLE:
            self.contamination_model = ContaminationModel()
            print("✅ Contamination model initialized")
        else:
            print("⚠️ Contamination model not available")
    
    def _generate_test_data(self):
        """Generate test data with known Hurst parameters."""
        print("🔍 Generating test data with known Hurst parameters...")
        
        # Test Hurst parameters
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        n_samples = 1000
        
        for H in hurst_values:
            print(f"   Generating data with H = {H}")
            
            # FBM data
            fbm_model = FBMModel(H=H, sigma=1.0)
            fbm_data = fbm_model.generate(n=n_samples, seed=42)
            
            # FGN data
            fgn_model = FGNModel(H=H, sigma=1.0)
            fgn_data = fgn_model.generate(n=n_samples, seed=42)
            
            # ARFIMA data (d = H - 0.5)
            d = H - 0.5
            arfima_model = ARFIMAModel(d=d, sigma=1.0)
            arfima_data = arfima_model.generate(n=n_samples, seed=42)
            
            self.test_data[f"H_{H}"] = {
                "true_hurst": H,
                "fbm": fbm_data,
                "fgn": fgn_data,
                "arfima": arfima_data
            }
        
        print(f"✅ Generated test data for {len(hurst_values)} Hurst values")
    
    def _generate_contaminated_data(self, base_data: np.ndarray, contamination_type: str) -> np.ndarray:
        """Generate contaminated data using available contamination methods."""
        if not CONTAMINATION_AVAILABLE or self.contamination_model is None:
            # Fallback contamination methods
            return self._fallback_contamination(base_data, contamination_type)
        
        try:
            if contamination_type == "additive_noise":
                return self.contamination_model.add_noise_gaussian(base_data, std=0.1)
            elif contamination_type == "linear_trend":
                return self.contamination_model.add_trend_linear(base_data, slope=0.01)
            elif contamination_type == "polynomial_trend":
                return self.contamination_model.add_trend_polynomial(base_data, degree=2)
            elif contamination_type == "spikes":
                return self.contamination_model.add_artifact_spikes(base_data, probability=0.05, amplitude=2.0)
            elif contamination_type == "level_shifts":
                return self.contamination_model.add_artifact_level_shifts(base_data, probability=0.02, amplitude=1.0)
            elif contamination_type == "missing_data":
                return self.contamination_model.add_artifact_missing_data(base_data, probability=0.1)
            elif contamination_type == "colored_noise":
                return self.contamination_model.add_noise_colored(base_data, alpha=1.0, std=0.1)
            elif contamination_type == "impulsive_noise":
                return self.contamination_model.add_noise_impulsive(base_data, probability=0.05, amplitude=3.0)
            else:
                return self._fallback_contamination(base_data, contamination_type)
        except Exception as e:
            print(f"Warning: Contamination failed for {contamination_type}: {e}")
            return self._fallback_contamination(base_data, contamination_type)
    
    def _fallback_contamination(self, base_data: np.ndarray, contamination_type: str) -> np.ndarray:
        """Fallback contamination methods when main contamination model is not available."""
        contaminated = base_data.copy()
        
        if contamination_type == "additive_noise":
            contaminated += 0.1 * np.random.normal(0, 1, len(base_data))
        elif contamination_type == "linear_trend":
            contaminated += 0.01 * np.arange(len(base_data))
        elif contamination_type == "polynomial_trend":
            x = np.arange(len(base_data)) / len(base_data)
            trend = 0.1 * (x**2 - 0.5)
            contaminated += trend
        elif contamination_type == "spikes":
            spike_indices = np.random.choice(len(base_data), size=int(0.05 * len(base_data)), replace=False)
            contaminated[spike_indices] += 2.0 * np.random.normal(0, 1, len(spike_indices))
        elif contamination_type == "level_shifts":
            shift_indices = np.random.choice(len(base_data), size=int(0.02 * len(base_data)), replace=False)
            contaminated[shift_indices] += 1.0
        elif contamination_type == "missing_data":
            missing_indices = np.random.choice(len(base_data), size=int(0.1 * len(base_data)), replace=False)
            contaminated[missing_indices] = np.nan
        elif contamination_type == "colored_noise":
            # Simple colored noise using AR(1)
            noise = np.zeros(len(base_data))
            noise[0] = np.random.normal(0, 1)
            for i in range(1, len(base_data)):
                noise[i] = 0.8 * noise[i-1] + 0.1 * np.random.normal(0, 1)
            contaminated += noise
        elif contamination_type == "impulsive_noise":
            impulse_indices = np.random.choice(len(base_data), size=int(0.05 * len(base_data)), replace=False)
            contaminated[impulse_indices] += 3.0 * np.random.normal(0, 1, len(impulse_indices))
        
        return contaminated
    
    def benchmark_pure_data(self) -> Dict[str, Any]:
        """Benchmark estimators on pure (uncontaminated) data."""
        print("\n📊 Benchmarking on Pure Data...")
        
        pure_results = {}
        
        for estimator_name, estimator in self.estimators.items():
            print(f"   Testing {estimator_name} on pure data...")
            
            estimator_results = {
                "accuracy": {},
                "performance": {},
                "robustness": {}
            }
            
            # Test on different data types
            for data_key, data_info in self.test_data.items():
                true_hurst = data_info["true_hurst"]
                
                # Test on FBM data
                try:
                    start_time = time.time()
                    result = estimator.estimate(data_info["fbm"])
                    execution_time = time.time() - start_time
                    
                    estimated_hurst = result.get("hurst_parameter", np.nan)
                    error = abs(estimated_hurst - true_hurst)
                    
                    estimator_results["accuracy"][f"{data_key}_fbm"] = {
                        "true_hurst": true_hurst,
                        "estimated_hurst": estimated_hurst,
                        "absolute_error": error,
                        "relative_error": error / true_hurst if true_hurst != 0 else np.nan,
                        "execution_time": execution_time,
                        "r_squared": result.get("r_squared", np.nan)
                    }
                    
                except Exception as e:
                    estimator_results["accuracy"][f"{data_key}_fbm"] = {
                        "error": str(e),
                        "execution_time": np.nan
                    }
                
                # Test on FGN data
                try:
                    start_time = time.time()
                    result = estimator.estimate(data_info["fgn"])
                    execution_time = time.time() - start_time
                    
                    estimated_hurst = result.get("hurst_parameter", np.nan)
                    error = abs(estimated_hurst - true_hurst)
                    
                    estimator_results["accuracy"][f"{data_key}_fgn"] = {
                        "true_hurst": true_hurst,
                        "estimated_hurst": estimated_hurst,
                        "absolute_error": error,
                        "relative_error": error / true_hurst if true_hurst != 0 else np.nan,
                        "execution_time": execution_time,
                        "r_squared": result.get("r_squared", np.nan)
                    }
                    
                except Exception as e:
                    estimator_results["accuracy"][f"{data_key}_fgn"] = {
                        "error": str(e),
                        "execution_time": np.nan
                    }
            
            pure_results[estimator_name] = estimator_results
        
        print("✅ Pure data benchmark completed")
        return pure_results
    
    def benchmark_contaminated_data(self) -> Dict[str, Any]:
        """Benchmark estimators on contaminated data."""
        print("\n🛡️ Benchmarking on Contaminated Data...")
        
        # Define contamination types
        contamination_types = [
            "additive_noise",
            "linear_trend", 
            "polynomial_trend",
            "spikes",
            "level_shifts",
            "missing_data",
            "colored_noise",
            "impulsive_noise"
        ]
        
        contaminated_results = {}
        
        for estimator_name, estimator in self.estimators.items():
            print(f"   Testing {estimator_name} on contaminated data...")
            
            estimator_results = {
                "contamination_robustness": {},
                "performance_degradation": {}
            }
            
            # Test on one representative dataset (H=0.7 FBM)
            base_data = self.test_data["H_0.7"]["fbm"]
            true_hurst = 0.7
            
            for cont_type in contamination_types:
                try:
                    # Generate contaminated data
                    contaminated_data = self._generate_contaminated_data(base_data, cont_type)
                    
                    # Handle missing data
                    if np.any(np.isnan(contaminated_data)):
                        clean_data = contaminated_data[~np.isnan(contaminated_data)]
                    else:
                        clean_data = contaminated_data
                    
                    if len(clean_data) < 100:
                        estimator_results["contamination_robustness"][cont_type] = {
                            "error": "Insufficient data after cleaning"
                        }
                        continue
                    
                    # Estimate on contaminated data
                    start_time = time.time()
                    result = estimator.estimate(clean_data)
                    execution_time = time.time() - start_time
                    
                    estimated_hurst = result.get("hurst_parameter", np.nan)
                    error = abs(estimated_hurst - true_hurst)
                    
                    estimator_results["contamination_robustness"][cont_type] = {
                        "true_hurst": true_hurst,
                        "estimated_hurst": estimated_hurst,
                        "absolute_error": error,
                        "relative_error": error / true_hurst if true_hurst != 0 else np.nan,
                        "execution_time": execution_time,
                        "data_length": len(clean_data),
                        "contamination_applied": cont_type
                    }
                    
                except Exception as e:
                    estimator_results["contamination_robustness"][cont_type] = {
                        "error": str(e),
                        "contamination_applied": cont_type
                    }
            
            contaminated_results[estimator_name] = estimator_results
        
        print("✅ Contaminated data benchmark completed")
        return contaminated_results
    
    def analyze_results(self, pure_results: Dict, contaminated_results: Dict) -> Dict[str, Any]:
        """Analyze and summarize benchmark results."""
        print("\n📈 Analyzing Benchmark Results...")
        
        analysis = {
            "pure_data_analysis": {},
            "contaminated_data_analysis": {},
            "robustness_analysis": {},
            "performance_ranking": {},
            "recommendations": {}
        }
        
        # Analyze pure data results
        for estimator_name, results in pure_results.items():
            accuracy_data = results["accuracy"]
            
            # Calculate average metrics
            errors = []
            times = []
            r_squared_values = []
            
            for test_name, test_result in accuracy_data.items():
                if "absolute_error" in test_result:
                    errors.append(test_result["absolute_error"])
                    times.append(test_result["execution_time"])
                    if "r_squared" in test_result and not np.isnan(test_result["r_squared"]):
                        r_squared_values.append(test_result["r_squared"])
            
            analysis["pure_data_analysis"][estimator_name] = {
                "mean_absolute_error": np.mean(errors) if errors else np.nan,
                "std_absolute_error": np.std(errors) if errors else np.nan,
                "mean_execution_time": np.mean(times) if times else np.nan,
                "mean_r_squared": np.mean(r_squared_values) if r_squared_values else np.nan,
                "success_rate": len(errors) / len(accuracy_data) if accuracy_data else 0
            }
        
        # Analyze contaminated data results
        for estimator_name, results in contaminated_results.items():
            robustness_data = results["contamination_robustness"]
            
            # Calculate robustness metrics
            errors = []
            success_count = 0
            
            for cont_type, cont_result in robustness_data.items():
                if "absolute_error" in cont_result:
                    errors.append(cont_result["absolute_error"])
                    success_count += 1
            
            analysis["contaminated_data_analysis"][estimator_name] = {
                "mean_absolute_error": np.mean(errors) if errors else np.nan,
                "std_absolute_error": np.std(errors) if errors else np.nan,
                "robustness_score": success_count / len(robustness_data) if robustness_data else 0,
                "total_contamination_types": len(robustness_data)
            }
        
        # Calculate overall performance ranking
        performance_scores = {}
        for estimator_name in self.estimators.keys():
            pure_analysis = analysis["pure_data_analysis"].get(estimator_name, {})
            contaminated_analysis = analysis["contaminated_data_analysis"].get(estimator_name, {})
            
            # Calculate composite score (lower error is better, higher success rate is better)
            pure_score = 0
            if not np.isnan(pure_analysis.get("mean_absolute_error", np.nan)):
                pure_score = max(0, 10 - pure_analysis["mean_absolute_error"] * 10)
            
            robustness_score = contaminated_analysis.get("robustness_score", 0) * 10
            
            performance_scores[estimator_name] = {
                "pure_data_score": pure_score,
                "robustness_score": robustness_score,
                "overall_score": (pure_score + robustness_score) / 2
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
                "recommendations": {
                    "general_purpose": best_estimator,
                    "high_accuracy": "Whittle" if "Whittle" in performance_scores else best_estimator,
                    "real_time": "R/S" if "R/S" in performance_scores else best_estimator,
                    "robust": max(performance_scores.items(), key=lambda x: x[1]["robustness_score"])[0]
                }
            }
        
        print("✅ Results analysis completed")
        return analysis
    
    def generate_visualizations(self, pure_results: Dict, contaminated_results: Dict, analysis: Dict):
        """Generate comprehensive visualizations of benchmark results."""
        print("\n📊 Generating Visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Pure Data Performance Comparison
        ax1 = plt.subplot(3, 3, 1)
        estimators = list(self.estimators.keys())
        mean_errors = [analysis["pure_data_analysis"].get(est, {}).get("mean_absolute_error", np.nan) for est in estimators]
        
        bars1 = ax1.bar(estimators, mean_errors, alpha=0.7)
        ax1.set_title("Pure Data: Mean Absolute Error", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Mean Absolute Error")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, mean_errors):
            if not np.isnan(value):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Execution Time Comparison
        ax2 = plt.subplot(3, 3, 2)
        mean_times = [analysis["pure_data_analysis"].get(est, {}).get("mean_execution_time", np.nan) for est in estimators]
        
        bars2 = ax2.bar(estimators, mean_times, alpha=0.7, color='orange')
        ax2.set_title("Pure Data: Mean Execution Time", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Execution Time (seconds)")
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Robustness Comparison
        ax3 = plt.subplot(3, 3, 3)
        robustness_scores = [analysis["contaminated_data_analysis"].get(est, {}).get("robustness_score", 0) for est in estimators]
        
        bars3 = ax3.bar(estimators, robustness_scores, alpha=0.7, color='green')
        ax3.set_title("Contamination Robustness", fontsize=12, fontweight='bold')
        ax3.set_ylabel("Robustness Score")
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Overall Performance Ranking
        ax4 = plt.subplot(3, 3, 4)
        ranking_data = analysis["performance_ranking"]["scores"]
        overall_scores = [ranking_data.get(est, {}).get("overall_score", 0) for est in estimators]
        
        bars4 = ax4.bar(estimators, overall_scores, alpha=0.7, color='purple')
        ax4.set_title("Overall Performance Score", fontsize=12, fontweight='bold')
        ax4.set_ylabel("Overall Score")
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. R-squared Comparison
        ax5 = plt.subplot(3, 3, 5)
        r_squared_values = [analysis["pure_data_analysis"].get(est, {}).get("mean_r_squared", np.nan) for est in estimators]
        
        bars5 = ax5.bar(estimators, r_squared_values, alpha=0.7, color='red')
        ax5.set_title("Pure Data: Mean R-squared", fontsize=12, fontweight='bold')
        ax5.set_ylabel("R-squared")
        ax5.set_ylim(0, 1)
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Success Rate Comparison
        ax6 = plt.subplot(3, 3, 6)
        success_rates = [analysis["pure_data_analysis"].get(est, {}).get("success_rate", 0) for est in estimators]
        
        bars6 = ax6.bar(estimators, success_rates, alpha=0.7, color='brown')
        ax6.set_title("Pure Data: Success Rate", fontsize=12, fontweight='bold')
        ax6.set_ylabel("Success Rate")
        ax6.set_ylim(0, 1)
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. Contamination Error Heatmap
        ax7 = plt.subplot(3, 3, 7)
        contamination_types = [
            "additive_noise", "linear_trend", "polynomial_trend", "spikes",
            "level_shifts", "missing_data", "colored_noise", "impulsive_noise"
        ]
        
        error_matrix = np.zeros((len(estimators), len(contamination_types)))
        for i, estimator in enumerate(estimators):
            for j, cont_type in enumerate(contamination_types):
                error = contaminated_results[estimator]["contamination_robustness"].get(cont_type, {}).get("absolute_error", np.nan)
                error_matrix[i, j] = error if not np.isnan(error) else 0
        
        im = ax7.imshow(error_matrix, cmap='YlOrRd', aspect='auto')
        ax7.set_title("Contamination Error Heatmap", fontsize=12, fontweight='bold')
        ax7.set_xlabel("Contamination Type")
        ax7.set_ylabel("Estimator")
        ax7.set_xticks(range(len(contamination_types)))
        ax7.set_xticklabels([ct.replace('_', '\n') for ct in contamination_types], rotation=45, ha='right')
        ax7.set_yticks(range(len(estimators)))
        ax7.set_yticklabels(estimators)
        
        # Add colorbar
        plt.colorbar(im, ax=ax7, label="Absolute Error")
        
        # 8. Performance vs Robustness Scatter
        ax8 = plt.subplot(3, 3, 8)
        pure_scores = [ranking_data.get(est, {}).get("pure_data_score", 0) for est in estimators]
        robustness_scores = [ranking_data.get(est, {}).get("robustness_score", 0) for est in estimators]
        
        scatter = ax8.scatter(pure_scores, robustness_scores, s=100, alpha=0.7)
        ax8.set_title("Performance vs Robustness", fontsize=12, fontweight='bold')
        ax8.set_xlabel("Pure Data Score")
        ax8.set_ylabel("Robustness Score")
        
        # Add estimator labels
        for i, estimator in enumerate(estimators):
            ax8.annotate(estimator, (pure_scores[i], robustness_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Create summary text
        summary_text = "BENCHMARK SUMMARY\\n\\n"
        if analysis["recommendations"]:
            rec = analysis["recommendations"]
            summary_text += f"Best Overall: {rec['best_overall']}\\n"
            summary_text += f"Most Robust: {rec['most_robust']}\\n"
            summary_text += f"Best for Pure Data: {rec['best_for_pure_data']}\\n\\n"
        
        summary_text += f"Total Estimators: {len(self.estimators)}\\n"
        summary_text += f"Total Test Cases: {len(self.test_data) * 2}\\n"
        summary_text += f"Contamination Types: 8\\n"
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_dir / "classical_estimators_benchmark.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved to {plot_path}")
    
    def save_results(self, pure_results: Dict, contaminated_results: Dict, analysis: Dict):
        """Save benchmark results to files."""
        print("\n💾 Saving Results...")
        
        # Save raw results
        results_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "estimators_tested": list(self.estimators.keys()),
                "test_data_keys": list(self.test_data.keys()),
                "contamination_types": [
                    "additive_noise", "linear_trend", "polynomial_trend", "spikes",
                    "level_shifts", "missing_data", "colored_noise", "impulsive_noise"
                ]
            },
            "pure_data_results": pure_results,
            "contaminated_data_results": contaminated_results,
            "analysis": analysis
        }
        
        # Save JSON results
        json_path = self.output_dir / "classical_estimators_benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save CSV summary
        csv_data = []
        for estimator_name in self.estimators.keys():
            pure_analysis = analysis["pure_data_analysis"].get(estimator_name, {})
            contaminated_analysis = analysis["contaminated_data_analysis"].get(estimator_name, {})
            ranking_data = analysis["performance_ranking"]["scores"].get(estimator_name, {})
            
            csv_data.append({
                "Estimator": estimator_name,
                "Mean_Absolute_Error_Pure": pure_analysis.get("mean_absolute_error", np.nan),
                "Mean_Execution_Time_Pure": pure_analysis.get("mean_execution_time", np.nan),
                "Mean_R_squared_Pure": pure_analysis.get("mean_r_squared", np.nan),
                "Success_Rate_Pure": pure_analysis.get("success_rate", 0),
                "Mean_Absolute_Error_Contaminated": contaminated_analysis.get("mean_absolute_error", np.nan),
                "Robustness_Score": contaminated_analysis.get("robustness_score", 0),
                "Pure_Data_Score": ranking_data.get("pure_data_score", 0),
                "Robustness_Score_Ranking": ranking_data.get("robustness_score", 0),
                "Overall_Score": ranking_data.get("overall_score", 0)
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / "classical_estimators_benchmark_summary.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Results saved to {json_path}")
        print(f"✅ Summary saved to {csv_path}")
    
    def run_comprehensive_benchmark(self):
        """Run the complete benchmark suite."""
        print("🚀 Starting Comprehensive Classical Estimators Benchmark")
        print("=" * 70)
        
        # Run benchmarks
        pure_results = self.benchmark_pure_data()
        contaminated_results = self.benchmark_contaminated_data()
        
        # Analyze results
        analysis = self.analyze_results(pure_results, contaminated_results)
        
        # Generate visualizations
        self.generate_visualizations(pure_results, contaminated_results, analysis)
        
        # Save results
        self.save_results(pure_results, contaminated_results, analysis)
        
        # Print summary
        self.print_summary(analysis)
        
        return {
            "pure_results": pure_results,
            "contaminated_results": contaminated_results,
            "analysis": analysis
        }
    
    def print_summary(self, analysis: Dict):
        """Print a comprehensive summary of benchmark results."""
        print("\n📊 COMPREHENSIVE BENCHMARK SUMMARY")
        print("=" * 70)
        
        if analysis["recommendations"]:
            rec = analysis["recommendations"]
            print(f"🏆 Best Overall Estimator: {rec['best_overall']}")
            print(f"🛡️ Most Robust Estimator: {rec['most_robust']}")
            print(f"🎯 Best for Pure Data: {rec['best_for_pure_data']}")
        
        print(f"\n📈 Performance Rankings:")
        rankings = analysis["performance_ranking"]["rankings"]
        for i, (estimator, scores) in enumerate(rankings, 1):
            print(f"{i:2d}. {estimator:12s}: {scores['overall_score']:5.2f}/10 (Pure: {scores['pure_data_score']:4.1f}, Robust: {scores['robustness_score']:4.1f})")
        
        print(f"\n📊 Key Statistics:")
        print(f"   • Total estimators tested: {len(self.estimators)}")
        print(f"   • Test Hurst values: {[0.3, 0.5, 0.7, 0.9]}")
        print(f"   • Data types: FBM, FGN, ARFIMA")
        print(f"   • Contamination types: 8")
        print(f"   • Total test cases: {len(self.test_data) * 2 * len(self.estimators)}")
        
        print(f"\n✅ Benchmark completed successfully!")
        print(f"📁 Results saved to: {self.output_dir}")

def main():
    """Run the comprehensive benchmark."""
    benchmark = ClassicalEstimatorsBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    return results

if __name__ == "__main__":
    results = main()
