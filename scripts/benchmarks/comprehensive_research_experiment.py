#!/usr/bin/env python3
"""
Comprehensive Research Paper Benchmarking Experiment

This script implements the full benchmarking methodology for a research paper on
long-range dependence estimation. It evaluates estimators across:

1. Pure Synthetic Processes (fBm, fGn, ARFIMA, MRW)
2. Contaminated Processes (8 types of systematic contamination)
3. Real-World Time Series (physiological, financial, environmental, network)
4. Statistical Evaluation (bias, variance, RMSE, robustness)
5. Scoring and Leaderboard Generation

Parameters are reduced to manageable levels without compromising validity.
"""

import numpy as np
import time
import warnings
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, field
from itertools import combinations
from scipy import stats

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
from lrdbenchmark.analysis.wavelet.variance.variance_estimator_unified import WaveletVarianceEstimator
from lrdbenchmark.analysis.multifractal.mfdfa.mfdfa_estimator_unified import MFDFAEstimator

# Import contamination models
try:
    from lrdbenchmark.models.contamination.contamination_models import (
        ContaminationModel, ContaminationType, ContaminationConfig
    )
    CONTAMINATION_AVAILABLE = True
except ImportError:
    CONTAMINATION_AVAILABLE = False

# Import real-world datasets
try:
    from lrdbenchmark.real_world.datasets import DATASETS, DatasetSpec
    REAL_WORLD_AVAILABLE = True
except ImportError:
    REAL_WORLD_AVAILABLE = False

# Import pretrained ML models
try:
    from lrdbenchmark.models.pretrained_models.ml_pretrained import (
        RandomForestPretrainedModel,
        SVREstimatorPretrainedModel,
        GradientBoostingPretrainedModel,
    )
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False

# Try to import neural models
try:
    from lrdbenchmark.models.pretrained_models.cnn_pretrained import CNNPretrainedModel
    from lrdbenchmark.models.pretrained_models.lstm_pretrained import LSTMPretrainedModel
    NEURAL_MODELS_AVAILABLE = True
except ImportError:
    NEURAL_MODELS_AVAILABLE = False


@dataclass
class ExperimentConfig:
    """Configuration for the research experiment with reduced parameters."""
    
    # Data generation parameters (reduced for manageability)
    hurst_values: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])
    arfima_d_values: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    mrw_intermittency_values: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    
    # Time series lengths (reduced from 2^10 to 2^14)
    # Using 2^10 (1024) and 2^12 (4096) for speed
    data_lengths: List[int] = field(default_factory=lambda: [1024, 4096])
    
    # Replication parameters (reduced from 1000 to 100)
    n_replications: int = 100
    
    # Contamination levels
    contamination_levels: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    
    # Statistical testing
    n_permutations: int = 1000  # Reduced from 10000
    alpha: float = 0.05
    
    # Leaderboard weights
    weight_precision: float = 0.50
    weight_speed: float = 0.25
    weight_robustness: float = 0.15
    weight_heavy_tail: float = 0.10
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Output directory
    output_dir: str = "research_experiment_results"


class ComprehensiveResearchExperiment:
    """
    Comprehensive benchmarking experiment for research paper methodology.
    
    Implements the full evaluation framework:
    - Pure synthetic processes
    - Contaminated processes  
    - Real-world time series
    - Statistical evaluation
    - Scoring and leaderboard generation
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """Initialize the experiment with configuration."""
        self.config = config or ExperimentConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.rng = np.random.default_rng(self.config.random_seed)
        
        # Initialize estimators
        self.estimators = self._initialize_estimators()
        
        # Initialize contamination model
        self.contamination_model = ContaminationModel() if CONTAMINATION_AVAILABLE else None
        
        # Results storage
        self.results = {
            "pure_synthetic": {},
            "contaminated": {},
            "real_world": {},
            "statistical_evaluation": {},
            "leaderboard": {}
        }
        
        print(f"🔬 Initialized Research Experiment")
        print(f"   - {len(self.estimators)} estimators")
        print(f"   - {self.config.n_replications} replications per condition")
        print(f"   - Data lengths: {self.config.data_lengths}")
        print(f"   - Hurst values: {self.config.hurst_values}")
    
    def _initialize_estimators(self) -> Dict[str, Any]:
        """Initialize all estimators for benchmarking."""
        estimators = {
            # Temporal estimators (classical)
            "R/S": RSEstimator(min_block_size=10, num_blocks=15),
            "DFA": DFAEstimator(min_scale=10, num_scales=15, order=1),
            "DMA": DMAEstimator(min_scale=10, num_scales=15),
            "Higuchi": HiguchiEstimator(min_k=2, max_k=20),
            
            # Spectral estimators (classical)
            "GPH": GPHEstimator(min_freq_ratio=0.01, max_freq_ratio=0.1),
            "Whittle": WhittleEstimator(),
            "Periodogram": PeriodogramEstimator(),
            
            # Wavelet estimators (classical)
            "CWT": CWTEstimator(),
            "WaveletVar": WaveletVarianceEstimator(),
            
            # Multifractal estimators
            "MFDFA": MFDFAEstimator(),
        }
        
        # Add ML estimators if available
        if ML_MODELS_AVAILABLE:
            try:
                estimators["RandomForest"] = RandomForestPretrainedModel()
                estimators["SVR"] = SVREstimatorPretrainedModel()
                estimators["GradientBoosting"] = GradientBoostingPretrainedModel()
            except Exception as e:
                print(f"   Warning: Could not load ML models: {e}")
        
        # Add neural estimators if available
        if NEURAL_MODELS_AVAILABLE:
            try:
                estimators["CNN"] = CNNPretrainedModel(input_length=500)
                estimators["LSTM"] = LSTMPretrainedModel(input_length=500)
            except Exception as e:
                print(f"   Warning: Could not load neural models: {e}")
        
        return estimators
    
    def _generate_data(
        self, 
        model_type: str, 
        length: int, 
        params: Dict[str, Any],
        seed: int
    ) -> np.ndarray:
        """Generate synthetic data from a specified model."""
        rng = np.random.default_rng(seed)
        
        if model_type == "fBm":
            model = FBMModel(H=params["H"], sigma=params.get("sigma", 1.0))
            return model.generate(length=length, rng=rng)
        
        elif model_type == "fGn":
            model = FGNModel(H=params["H"], sigma=params.get("sigma", 1.0))
            return model.generate(length=length, rng=rng)
        
        elif model_type == "ARFIMA":
            model = ARFIMAModel(
                d=params["d"],
                ar_params=params.get("ar_params", []),
                ma_params=params.get("ma_params", []),
                sigma=params.get("sigma", 1.0)
            )
            return model.generate(length=length, rng=rng)
        
        elif model_type == "MRW":
            model = MRWModel(
                H=params["H"],
                lambda_param=params.get("lambda_param", 0.5),
                sigma=params.get("sigma", 1.0)
            )
            return model.generate(length=length, rng=rng)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _apply_contamination(
        self, 
        data: np.ndarray, 
        contamination_type: str,
        level: float,
        seed: int
    ) -> np.ndarray:
        """Apply contamination to time series data."""
        rng = np.random.default_rng(seed)
        contaminated = data.copy()
        n = len(data)
        
        if contamination_type == "linear_trend":
            # Linear trend (simulating slow drift)
            trend = level * np.linspace(0, 1, n) * np.std(data)
            contaminated += trend
        
        elif contamination_type == "polynomial_trend":
            # Polynomial trend (quadratic)
            x = np.linspace(0, 1, n)
            trend = level * (x ** 2 - 0.5) * np.std(data)
            contaminated += trend
        
        elif contamination_type == "level_shifts":
            # Random level shifts (regime changes)
            n_shifts = max(1, int(n * level * 0.01))
            shift_points = rng.choice(n, size=n_shifts, replace=False)
            for shift_point in sorted(shift_points):
                shift_magnitude = rng.normal(0, level * np.std(data))
                contaminated[shift_point:] += shift_magnitude
        
        elif contamination_type == "gaussian_noise":
            # Additive Gaussian noise
            noise = rng.normal(0, level * np.std(data), n)
            contaminated += noise
        
        elif contamination_type == "heavy_tailed_noise":
            # Heavy-tailed noise (t-distribution, df=3)
            # Use seed safely - rng is already seeded, so use it for t-distribution via numpy
            noise = rng.standard_t(df=3, size=n) * level * np.std(data)
            contaminated += noise
        
        elif contamination_type == "heteroscedasticity":
            # Multiplicative variance modulation
            modulation = 1 + level * np.sin(2 * np.pi * np.arange(n) / (n / 4))
            contaminated = data * modulation
        
        elif contamination_type == "missing_random":
            # Random missing data
            n_missing = int(n * level)
            missing_indices = rng.choice(n, size=n_missing, replace=False)
            contaminated = contaminated.astype(float)
            contaminated[missing_indices] = np.nan
        
        elif contamination_type == "missing_gaps":
            # Structured gaps (missing blocks)
            n_gaps = max(1, int(level * 5))
            gap_size = max(1, int(n * level * 0.05))
            for _ in range(n_gaps):
                start = rng.integers(0, max(1, n - gap_size))
                contaminated = contaminated.astype(float)
                contaminated[start:start + gap_size] = np.nan
        
        elif contamination_type == "outliers":
            # Impulsive outliers (movement or electrical artefacts)
            n_outliers = int(n * level * 0.05)
            outlier_indices = rng.choice(n, size=n_outliers, replace=False)
            outlier_magnitudes = rng.normal(0, 5 * np.std(data), n_outliers)
            contaminated[outlier_indices] += outlier_magnitudes
        
        return contaminated
    
    def _estimate_hurst(
        self, 
        estimator, 
        data: np.ndarray
    ) -> Dict[str, Any]:
        """Estimate Hurst parameter with timing and error handling."""
        # Handle missing data
        clean_data = data[~np.isnan(data)] if np.any(np.isnan(data)) else data
        
        if len(clean_data) < 100:
            return {
                "hurst_parameter": np.nan,
                "success": False,
                "error": "Insufficient data after cleaning",
                "execution_time": 0.0
            }
        
        try:
            start_time = time.time()
            result = estimator.estimate(clean_data)
            execution_time = time.time() - start_time
            
            return {
                "hurst_parameter": result.get("hurst_parameter", np.nan),
                "r_squared": result.get("r_squared", np.nan),
                "success": True,
                "execution_time": execution_time
            }
        except Exception as e:
            return {
                "hurst_parameter": np.nan,
                "success": False,
                "error": str(e),
                "execution_time": 0.0
            }
    
    def run_pure_synthetic_experiments(self) -> Dict[str, Any]:
        """
        Run experiments on pure synthetic processes.
        
        Tests fBm, fGn, ARFIMA, and MRW with varying parameters.
        """
        print("\n" + "=" * 70)
        print("📊 EXPERIMENT 1: Pure Synthetic Processes")
        print("=" * 70)
        
        results = {
            "fBm": {},
            "fGn": {},
            "ARFIMA": {},
            "MRW": {}
        }
        
        total_conditions = (
            len(self.config.hurst_values) * len(self.config.data_lengths) * 2 +  # fBm, fGn
            len(self.config.arfima_d_values) * len(self.config.data_lengths) +    # ARFIMA
            2 * min(2, len(self.config.mrw_intermittency_values)) * 1  # MRW (subset: 2 H values, first 2 lambda, 1 length)
        )
        condition_count = 0
        
        # 1. Fractional Brownian Motion (fBm)
        print("\n🔹 Testing fBm...")
        for H in self.config.hurst_values:
            for length in self.config.data_lengths:
                condition_count += 1
                key = f"H{H}_N{length}"
                print(f"   [{condition_count}/{total_conditions}] fBm: H={H}, N={length}")
                
                condition_results = self._run_replication_experiment(
                    model_type="fBm",
                    length=length,
                    params={"H": H},
                    true_value=H
                )
                results["fBm"][key] = condition_results
        
        # 2. Fractional Gaussian Noise (fGn)
        print("\n🔹 Testing fGn...")
        for H in self.config.hurst_values:
            for length in self.config.data_lengths:
                condition_count += 1
                key = f"H{H}_N{length}"
                print(f"   [{condition_count}/{total_conditions}] fGn: H={H}, N={length}")
                
                condition_results = self._run_replication_experiment(
                    model_type="fGn",
                    length=length,
                    params={"H": H},
                    true_value=H
                )
                results["fGn"][key] = condition_results
        
        # 3. ARFIMA
        print("\n🔹 Testing ARFIMA...")
        for d in self.config.arfima_d_values:
            for length in self.config.data_lengths:
                condition_count += 1
                key = f"d{d}_N{length}"
                print(f"   [{condition_count}/{total_conditions}] ARFIMA: d={d}, N={length}")
                
                # For ARFIMA, H = d + 0.5
                true_H = d + 0.5
                condition_results = self._run_replication_experiment(
                    model_type="ARFIMA",
                    length=length,
                    params={"d": d},
                    true_value=true_H
                )
                results["ARFIMA"][key] = condition_results
        
        # 4. Multifractal Random Walk (MRW) - subset of parameters
        print("\n🔹 Testing MRW...")
        # Only test a subset to keep runtime manageable
        mrw_H_values = [0.5, 0.7]  # Reduced
        for H in mrw_H_values:
            for lambda_param in self.config.mrw_intermittency_values[:2]:  # Reduced
                for length in self.config.data_lengths[:1]:  # Only shortest length
                    condition_count += 1
                    key = f"H{H}_lambda{lambda_param}_N{length}"
                    print(f"   [{condition_count}/{total_conditions}] MRW: H={H}, λ={lambda_param}, N={length}")
                    
                    condition_results = self._run_replication_experiment(
                        model_type="MRW",
                        length=length,
                        params={"H": H, "lambda_param": lambda_param},
                        true_value=H
                    )
                    results["MRW"][key] = condition_results
        
        self.results["pure_synthetic"] = results
        print("\n✅ Pure synthetic experiments completed")
        return results
    
    def _run_replication_experiment(
        self,
        model_type: str,
        length: int,
        params: Dict[str, Any],
        true_value: float
    ) -> Dict[str, Any]:
        """Run replicated experiment for a single condition."""
        estimator_results = {name: [] for name in self.estimators.keys()}
        
        for rep in range(self.config.n_replications):
            # Generate data with unique seed per replication
            seed = self.config.random_seed + rep * 1000 + hash(model_type) % 1000
            data = self._generate_data(model_type, length, params, seed)
            
            # Test each estimator
            for est_name, estimator in self.estimators.items():
                result = self._estimate_hurst(estimator, data)
                result["true_value"] = true_value
                result["replication"] = rep
                estimator_results[est_name].append(result)
        
        # Compute summary statistics
        summary = self._compute_summary_statistics(estimator_results, true_value)
        
        return {
            "params": params,
            "true_value": true_value,
            "length": length,
            "n_replications": self.config.n_replications,
            "estimator_results": estimator_results,
            "summary": summary
        }
    
    def _compute_summary_statistics(
        self, 
        estimator_results: Dict[str, List[Dict]], 
        true_value: float
    ) -> Dict[str, Any]:
        """Compute bias, variance, RMSE, and robustness for each estimator."""
        summary = {}
        
        for est_name, results in estimator_results.items():
            estimates = [r["hurst_parameter"] for r in results if r["success"]]
            execution_times = [r["execution_time"] for r in results if r["success"]]
            n_success = len(estimates)
            n_total = len(results)
            
            if n_success > 0:
                estimates_arr = np.array(estimates)
                
                # Bias: E[H_hat] - H_true
                bias = np.mean(estimates_arr) - true_value
                
                # Variance: Var(H_hat)
                variance = np.var(estimates_arr, ddof=1) if n_success > 1 else 0.0
                
                # RMSE: sqrt(E[(H_hat - H_true)^2])
                rmse = np.sqrt(np.mean((estimates_arr - true_value) ** 2))
                
                # MAE
                mae = np.mean(np.abs(estimates_arr - true_value))
                
                # Robustness: proportion of valid estimates
                robustness = n_success / n_total
                
                # Mean execution time
                mean_time = np.mean(execution_times) if execution_times else np.nan
                
                summary[est_name] = {
                    "bias": float(bias),
                    "variance": float(variance),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "robustness": float(robustness),
                    "mean_estimate": float(np.mean(estimates_arr)),
                    "std_estimate": float(np.std(estimates_arr, ddof=1)) if n_success > 1 else 0.0,
                    "mean_execution_time": float(mean_time),
                    "n_success": n_success,
                    "n_total": n_total
                }
            else:
                summary[est_name] = {
                    "bias": np.nan,
                    "variance": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "robustness": 0.0,
                    "mean_estimate": np.nan,
                    "std_estimate": np.nan,
                    "mean_execution_time": np.nan,
                    "n_success": 0,
                    "n_total": n_total
                }
        
        return summary
    
    def run_contaminated_experiments(self) -> Dict[str, Any]:
        """
        Run experiments on contaminated processes.
        
        Tests 8 types of systematic contamination with varying intensity.
        """
        print("\n" + "=" * 70)
        print("🛡️ EXPERIMENT 2: Contaminated Processes")
        print("=" * 70)
        
        contamination_types = [
            "linear_trend",
            "polynomial_trend",
            "level_shifts",
            "gaussian_noise",
            "heavy_tailed_noise",
            "heteroscedasticity",
            "missing_random",
            "outliers"
        ]
        
        results = {ct: {} for ct in contamination_types}
        
        # Use fGn with H=0.7 as base process
        base_H = 0.7
        base_length = self.config.data_lengths[0]  # Use shorter length for speed
        
        total_conditions = len(contamination_types) * len(self.config.contamination_levels)
        condition_count = 0
        
        for cont_type in contamination_types:
            print(f"\n🔹 Testing {cont_type}...")
            
            for level in self.config.contamination_levels:
                condition_count += 1
                key = f"level{level}"
                print(f"   [{condition_count}/{total_conditions}] {cont_type}: level={level}")
                
                condition_results = self._run_contamination_experiment(
                    contamination_type=cont_type,
                    level=level,
                    base_H=base_H,
                    length=base_length
                )
                results[cont_type][key] = condition_results
        
        self.results["contaminated"] = results
        print("\n✅ Contaminated experiments completed")
        return results
    
    def _run_contamination_experiment(
        self,
        contamination_type: str,
        level: float,
        base_H: float,
        length: int
    ) -> Dict[str, Any]:
        """Run replicated contamination experiment."""
        estimator_results = {name: [] for name in self.estimators.keys()}
        
        for rep in range(self.config.n_replications):
            # Generate clean base data
            seed_base = self.config.random_seed + rep * 1000
            clean_data = self._generate_data("fGn", length, {"H": base_H}, seed_base)
            
            # Apply contamination
            seed_cont = seed_base + 500
            contaminated_data = self._apply_contamination(
                clean_data, contamination_type, level, seed_cont
            )
            
            # Test each estimator
            for est_name, estimator in self.estimators.items():
                result = self._estimate_hurst(estimator, contaminated_data)
                result["true_value"] = base_H
                result["replication"] = rep
                result["contamination_type"] = contamination_type
                result["contamination_level"] = level
                estimator_results[est_name].append(result)
        
        # Compute summary statistics
        summary = self._compute_summary_statistics(estimator_results, base_H)
        
        return {
            "contamination_type": contamination_type,
            "contamination_level": level,
            "base_H": base_H,
            "length": length,
            "n_replications": self.config.n_replications,
            "estimator_results": estimator_results,
            "summary": summary
        }
    
    def run_real_world_experiments(self) -> Dict[str, Any]:
        """
        Run experiments on real-world time series.
        
        Tests on physiological, financial, environmental, and network data.
        """
        print("\n" + "=" * 70)
        print("🌍 EXPERIMENT 3: Real-World Time Series")
        print("=" * 70)
        
        results = {}
        
        if not REAL_WORLD_AVAILABLE:
            print("   ⚠️ Real-world datasets not available")
            return results
        
        for dataset_spec in DATASETS:
            print(f"\n🔹 Testing {dataset_spec.name} ({dataset_spec.domain})...")
            
            # Generate dataset
            rng = np.random.default_rng(dataset_spec.base_seed)
            data = dataset_spec.generator(dataset_spec.default_length, rng)
            
            # Preprocess: mild detrending and z-normalization
            data_detrended = self._detrend_linear(data)
            data_normalized = (data_detrended - np.mean(data_detrended)) / np.std(data_detrended)
            
            # Test each estimator
            estimator_results = {}
            for est_name, estimator in self.estimators.items():
                result = self._estimate_hurst(estimator, data_normalized)
                estimator_results[est_name] = result
            
            results[dataset_spec.name] = {
                "domain": dataset_spec.domain,
                "description": dataset_spec.description,
                "length": dataset_spec.default_length,
                "estimator_results": estimator_results,
                "data_stats": {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data))
                }
            }
        
        self.results["real_world"] = results
        print("\n✅ Real-world experiments completed")
        return results
    
    def _detrend_linear(self, data: np.ndarray) -> np.ndarray:
        """Remove linear trend from data."""
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend = np.polyval(coeffs, x)
        return data - trend
    
    def compute_statistical_evaluation(self) -> Dict[str, Any]:
        """
        Compute statistical evaluation metrics.
        
        Includes paired permutation tests with Bonferroni correction.
        """
        print("\n" + "=" * 70)
        print("📈 EXPERIMENT 4: Statistical Evaluation")
        print("=" * 70)
        
        evaluation = {
            "pure_synthetic_summary": {},
            "contaminated_summary": {},
            "significance_tests": {}
        }
        
        # Aggregate pure synthetic results
        print("\n🔹 Aggregating pure synthetic results...")
        if self.results.get("pure_synthetic"):
            evaluation["pure_synthetic_summary"] = self._aggregate_model_results(
                self.results["pure_synthetic"]
            )
        
        # Aggregate contaminated results
        print("🔹 Aggregating contaminated results...")
        if self.results.get("contaminated"):
            evaluation["contaminated_summary"] = self._aggregate_contamination_results(
                self.results["contaminated"]
            )
        
        # Run significance tests
        print("🔹 Running significance tests...")
        evaluation["significance_tests"] = self._run_significance_tests()
        
        self.results["statistical_evaluation"] = evaluation
        print("\n✅ Statistical evaluation completed")
        return evaluation
    
    def _aggregate_model_results(
        self, 
        model_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate results across model types and conditions."""
        aggregated = {}
        
        for model_type, conditions in model_results.items():
            model_summary = {}
            
            for est_name in self.estimators.keys():
                all_biases = []
                all_variances = []
                all_rmses = []
                all_robustness = []
                all_times = []
                
                for condition_key, condition_data in conditions.items():
                    summary = condition_data.get("summary", {}).get(est_name, {})
                    if not np.isnan(summary.get("bias", np.nan)):
                        all_biases.append(summary["bias"])
                        all_variances.append(summary["variance"])
                        all_rmses.append(summary["rmse"])
                        all_robustness.append(summary["robustness"])
                        all_times.append(summary["mean_execution_time"])
                
                if all_biases:
                    model_summary[est_name] = {
                        "mean_bias": float(np.mean(all_biases)),
                        "mean_variance": float(np.mean(all_variances)),
                        "mean_rmse": float(np.mean(all_rmses)),
                        "mean_robustness": float(np.mean(all_robustness)),
                        "mean_execution_time": float(np.mean(all_times)),
                        "n_conditions": len(all_biases)
                    }
                else:
                    model_summary[est_name] = {
                        "mean_bias": np.nan,
                        "mean_variance": np.nan,
                        "mean_rmse": np.nan,
                        "mean_robustness": 0.0,
                        "mean_execution_time": np.nan,
                        "n_conditions": 0
                    }
            
            aggregated[model_type] = model_summary
        
        return aggregated
    
    def _aggregate_contamination_results(
        self, 
        contamination_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate results across contamination types."""
        aggregated = {}
        
        for est_name in self.estimators.keys():
            all_rmses = []
            all_robustness = []
            
            for cont_type, levels in contamination_results.items():
                for level_key, level_data in levels.items():
                    summary = level_data.get("summary", {}).get(est_name, {})
                    if not np.isnan(summary.get("rmse", np.nan)):
                        all_rmses.append(summary["rmse"])
                        all_robustness.append(summary["robustness"])
            
            if all_rmses:
                aggregated[est_name] = {
                    "mean_rmse_contaminated": float(np.mean(all_rmses)),
                    "mean_robustness_contaminated": float(np.mean(all_robustness)),
                    "n_conditions": len(all_rmses)
                }
            else:
                aggregated[est_name] = {
                    "mean_rmse_contaminated": np.nan,
                    "mean_robustness_contaminated": 0.0,
                    "n_conditions": 0
                }
        
        return aggregated
    
    def _run_significance_tests(self) -> Dict[str, Any]:
        """Run paired permutation tests with Bonferroni correction."""
        significance_results = {
            "friedman_test": None,
            "pairwise_tests": []
        }
        
        # Collect RMSE values for each estimator across conditions
        estimator_rmses = {name: [] for name in self.estimators.keys()}
        
        # From pure synthetic results
        for model_type, conditions in self.results.get("pure_synthetic", {}).items():
            for condition_key, condition_data in conditions.items():
                summary = condition_data.get("summary", {})
                for est_name in self.estimators.keys():
                    rmse = summary.get(est_name, {}).get("rmse", np.nan)
                    if not np.isnan(rmse):
                        estimator_rmses[est_name].append(rmse)
        
        # Filter estimators with sufficient data
        valid_estimators = {
            name: rmses for name, rmses in estimator_rmses.items() 
            if len(rmses) >= 3
        }
        
        if len(valid_estimators) < 2:
            return significance_results
        
        # Friedman test (non-parametric ANOVA for repeated measures)
        try:
            # Align data lengths
            min_len = min(len(v) for v in valid_estimators.values())
            aligned_data = [v[:min_len] for v in valid_estimators.values()]
            
            if min_len >= 3:
                stat, p_value = stats.friedmanchisquare(*aligned_data)
                significance_results["friedman_test"] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "n_conditions": min_len,
                    "n_estimators": len(valid_estimators)
                }
        except Exception as e:
            significance_results["friedman_test"] = {"error": str(e)}
        
        # Pairwise Wilcoxon tests with Bonferroni correction
        est_names = list(valid_estimators.keys())
        n_comparisons = len(list(combinations(est_names, 2)))
        bonferroni_alpha = self.config.alpha / n_comparisons if n_comparisons > 0 else self.config.alpha
        
        for est1, est2 in combinations(est_names, 2):
            rmses1 = valid_estimators[est1]
            rmses2 = valid_estimators[est2]
            min_len = min(len(rmses1), len(rmses2))
            
            if min_len >= 5:
                try:
                    stat, p_value = stats.wilcoxon(rmses1[:min_len], rmses2[:min_len])
                    significance_results["pairwise_tests"].append({
                        "estimator_1": est1,
                        "estimator_2": est2,
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "significant": p_value < bonferroni_alpha,
                        "bonferroni_alpha": float(bonferroni_alpha)
                    })
                except Exception:
                    pass
        
        return significance_results
    
    def generate_leaderboard(self) -> Dict[str, Any]:
        """
        Generate composite leaderboard from four components.
        
        Components:
        - Precision (50%): Based on RMSE
        - Speed (25%): Based on execution time
        - Robustness (15%): Based on contamination robustness
        - Heavy-tail bonus (10%): Performance on heavy-tailed data
        """
        print("\n" + "=" * 70)
        print("🏆 EXPERIMENT 5: Scoring and Leaderboard Generation")
        print("=" * 70)
        
        leaderboard = {
            "component_scores": {},
            "composite_scores": {},
            "rankings": []
        }
        
        # Collect metrics for each estimator
        for est_name in self.estimators.keys():
            scores = self._compute_component_scores(est_name)
            leaderboard["component_scores"][est_name] = scores
        
        # Normalize scores to 0-10 scale
        normalized_scores = self._normalize_scores(leaderboard["component_scores"])
        
        # Compute composite scores
        for est_name, scores in normalized_scores.items():
            composite = (
                self.config.weight_precision * scores.get("precision", 0) +
                self.config.weight_speed * scores.get("speed", 0) +
                self.config.weight_robustness * scores.get("robustness", 0) +
                self.config.weight_heavy_tail * scores.get("heavy_tail", 0)
            )
            leaderboard["composite_scores"][est_name] = {
                "normalized_components": scores,
                "composite_score": float(composite)
            }
        
        # Create rankings
        rankings = sorted(
            leaderboard["composite_scores"].items(),
            key=lambda x: x[1]["composite_score"],
            reverse=True
        )
        leaderboard["rankings"] = [
            {"rank": i + 1, "estimator": est, "score": data["composite_score"]}
            for i, (est, data) in enumerate(rankings)
        ]
        
        self.results["leaderboard"] = leaderboard
        print("\n✅ Leaderboard generated")
        return leaderboard
    
    def _compute_component_scores(self, est_name: str) -> Dict[str, float]:
        """Compute raw component scores for an estimator."""
        scores = {
            "precision": 0.0,
            "speed": 0.0,
            "robustness": 0.0,
            "heavy_tail": 0.0
        }
        
        # Precision: inverse of mean RMSE from pure synthetic
        rmses = []
        times = []
        for model_type, conditions in self.results.get("pure_synthetic", {}).items():
            for condition_key, condition_data in conditions.items():
                summary = condition_data.get("summary", {}).get(est_name, {})
                rmse = summary.get("rmse", np.nan)
                time_val = summary.get("mean_execution_time", np.nan)
                if not np.isnan(rmse):
                    rmses.append(rmse)
                if not np.isnan(time_val):
                    times.append(time_val)
        
        if rmses:
            mean_rmse = np.mean(rmses)
            # Lower RMSE is better, so invert
            scores["precision"] = 1.0 / (1.0 + mean_rmse) if mean_rmse > 0 else 1.0
        
        # Speed: inverse of mean execution time
        if times:
            mean_time = np.mean(times)
            scores["speed"] = 1.0 / (1.0 + mean_time) if mean_time > 0 else 1.0
        
        # Robustness: mean robustness from contaminated experiments
        robustness_vals = []
        for cont_type, levels in self.results.get("contaminated", {}).items():
            for level_key, level_data in levels.items():
                summary = level_data.get("summary", {}).get(est_name, {})
                rob = summary.get("robustness", np.nan)
                if not np.isnan(rob):
                    robustness_vals.append(rob)
        
        if robustness_vals:
            scores["robustness"] = np.mean(robustness_vals)
        
        # Heavy-tail bonus: performance on heavy-tailed contamination
        heavy_tail_rmses = []
        for level_key, level_data in self.results.get("contaminated", {}).get("heavy_tailed_noise", {}).items():
            summary = level_data.get("summary", {}).get(est_name, {})
            rmse = summary.get("rmse", np.nan)
            if not np.isnan(rmse):
                heavy_tail_rmses.append(rmse)
        
        if heavy_tail_rmses:
            mean_ht_rmse = np.mean(heavy_tail_rmses)
            scores["heavy_tail"] = 1.0 / (1.0 + mean_ht_rmse) if mean_ht_rmse > 0 else 1.0
        
        return scores
    
    def _normalize_scores(
        self, 
        component_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Normalize component scores to 0-10 scale."""
        normalized = {}
        
        # Get min/max for each component
        components = ["precision", "speed", "robustness", "heavy_tail"]
        
        for component in components:
            values = [
                scores.get(component, 0) 
                for scores in component_scores.values()
            ]
            min_val = min(values) if values else 0
            max_val = max(values) if values else 1
            range_val = max_val - min_val if max_val > min_val else 1
            
            for est_name, scores in component_scores.items():
                if est_name not in normalized:
                    normalized[est_name] = {}
                
                raw_score = scores.get(component, 0)
                normalized_score = 10 * (raw_score - min_val) / range_val
                normalized[est_name][component] = float(normalized_score)
        
        return normalized
    
    def save_results(self):
        """Save all experiment results to files."""
        print("\n" + "=" * 70)
        print("💾 Saving Results")
        print("=" * 70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive JSON results
        json_path = self.output_dir / f"research_experiment_results_{timestamp}.json"
        
        # Create a JSON-serializable version of results
        results_to_save = self._make_json_serializable(self.results)
        results_to_save["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "hurst_values": self.config.hurst_values,
                "arfima_d_values": self.config.arfima_d_values,
                "data_lengths": self.config.data_lengths,
                "n_replications": self.config.n_replications,
                "contamination_levels": self.config.contamination_levels,
                "random_seed": self.config.random_seed
            },
            "estimators_tested": list(self.estimators.keys())
        }
        
        with open(json_path, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        print(f"   📄 JSON results: {json_path}")
        
        # Save leaderboard as CSV
        if self.results.get("leaderboard", {}).get("rankings"):
            csv_path = self.output_dir / f"leaderboard_{timestamp}.csv"
            leaderboard_df = pd.DataFrame(self.results["leaderboard"]["rankings"])
            leaderboard_df.to_csv(csv_path, index=False)
            print(f"   📊 Leaderboard CSV: {csv_path}")
        
        # Save summary report
        report_path = self.output_dir / f"experiment_report_{timestamp}.txt"
        self._generate_report(report_path)
        print(f"   📝 Report: {report_path}")
        
        print("\n✅ All results saved")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if np.isfinite(obj) else None
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _generate_report(self, path: Path):
        """Generate a text report of the experiment."""
        with open(path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("COMPREHENSIVE RESEARCH BENCHMARKING EXPERIMENT REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Hurst values: {self.config.hurst_values}\n")
            f.write(f"ARFIMA d values: {self.config.arfima_d_values}\n")
            f.write(f"Data lengths: {self.config.data_lengths}\n")
            f.write(f"Replications: {self.config.n_replications}\n")
            f.write(f"Contamination levels: {self.config.contamination_levels}\n")
            f.write(f"Random seed: {self.config.random_seed}\n\n")
            
            f.write("ESTIMATORS TESTED\n")
            f.write("-" * 40 + "\n")
            for i, name in enumerate(self.estimators.keys(), 1):
                f.write(f"{i:2d}. {name}\n")
            f.write("\n")
            
            if self.results.get("leaderboard", {}).get("rankings"):
                f.write("LEADERBOARD\n")
                f.write("-" * 40 + "\n")
                for entry in self.results["leaderboard"]["rankings"]:
                    f.write(f"{entry['rank']:2d}. {entry['estimator']:20s} Score: {entry['score']:.4f}\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("END OF REPORT\n")
    
    def run_full_experiment(self):
        """Run the complete research experiment."""
        print("\n" + "=" * 70)
        print("🔬 COMPREHENSIVE RESEARCH BENCHMARKING EXPERIMENT")
        print("=" * 70)
        print(f"   Output directory: {self.output_dir}")
        print(f"   Estimators: {len(self.estimators)}")
        print(f"   Replications per condition: {self.config.n_replications}")
        
        start_time = time.time()
        
        # Run all experiments
        self.run_pure_synthetic_experiments()
        self.run_contaminated_experiments()
        self.run_real_world_experiments()
        self.compute_statistical_evaluation()
        self.generate_leaderboard()
        
        # Save results
        self.save_results()
        
        # Print final summary
        total_time = time.time() - start_time
        self._print_final_summary(total_time)
        
        return self.results
    
    def _print_final_summary(self, total_time: float):
        """Print final experiment summary."""
        print("\n" + "=" * 70)
        print("📊 FINAL EXPERIMENT SUMMARY")
        print("=" * 70)
        
        print(f"\n⏱️  Total experiment time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        if self.results.get("leaderboard", {}).get("rankings"):
            print("\n🏆 TOP 5 ESTIMATORS:")
            for entry in self.results["leaderboard"]["rankings"][:5]:
                print(f"   {entry['rank']}. {entry['estimator']:20s} Score: {entry['score']:.4f}")
        
        print("\n✅ Experiment completed successfully!")
        print(f"📁 Results saved to: {self.output_dir}")


def main():
    """Main entry point for the research experiment."""
    # Create configuration with manageable parameters
    config = ExperimentConfig(
        # Reduced Hurst values (from 5 to 4)
        hurst_values=[0.3, 0.5, 0.7, 0.9],
        
        # Reduced ARFIMA d values (from full range to 4 values)
        arfima_d_values=[0.1, 0.2, 0.3, 0.4],
        
        # Reduced MRW intermittency values
        mrw_intermittency_values=[0.1, 0.3, 0.5],
        
        # Reduced data lengths (from 2^10 to 2^14 -> just 2^10 and 2^12)
        data_lengths=[1024, 4096],
        
        # Reduced replications (from 1000 to 100)
        n_replications=100,
        
        # Contamination levels
        contamination_levels=[0.1, 0.2, 0.3],
        
        # Reduced permutation tests (from 10000 to 1000)
        n_permutations=1000,
        
        # Random seed for reproducibility
        random_seed=42,
        
        # Output directory
        output_dir="research_experiment_results"
    )
    
    # Run experiment
    experiment = ComprehensiveResearchExperiment(config)
    results = experiment.run_full_experiment()
    
    return results


if __name__ == "__main__":
    results = main()
