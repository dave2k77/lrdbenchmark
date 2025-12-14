#!/usr/bin/env python3
"""
ML and NN Benchmarking Experiment Script
========================================

Comprehensive benchmarking experiment for ML and NN based LRD estimators covering:
1. Pure Synthetic Processes (n=320): fBm, fGn, ARFIMA, MRW
2. Contaminated Processes (n=240): 8 contamination types
3. Realistic Scenarios (n=300): 15 domain-specific scenarios

Generates LaTeX tables, visualizations, and statistical analysis reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from scipy import stats

# Import the core benchmark class
from lrdbenchmark.analysis.benchmark import ComprehensiveBenchmark
from lrdbenchmark.real_world.datasets import dataset_map

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# =============================================================================
# CONTAMINATION CLASSES
# =============================================================================

class LinearTrendContamination:
    """Add linear trend to signal."""
    def __init__(self, rng: np.random.Generator, level: float = 0.1):
        self.level = level
        self.rng = rng
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        n = len(data)
        signal_std = np.std(data)
        trend = np.linspace(0, self.level * signal_std * np.sqrt(12), n)
        return data + trend


class PolynomialTrendContamination:
    """Add polynomial/baseline drift to signal."""
    def __init__(self, rng: np.random.Generator, level: float = 0.1):
        self.level = level
        self.rng = rng
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        n = len(data)
        signal_std = np.std(data)
        t = np.linspace(0, 1, n)
        # Quadratic + cubic components
        drift = self.level * signal_std * (2 * t**2 - t**3)
        return data + drift


class GaussianNoiseContamination:
    """Add Gaussian noise to signal."""
    def __init__(self, rng: np.random.Generator, level: float = 0.1):
        self.level = level
        self.rng = rng
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        signal_var = np.var(data)
        noise_std = np.sqrt(self.level * signal_var)
        noise = self.rng.normal(0, noise_std, len(data))
        return data + noise


class HeavyTailedNoiseContamination:
    """Add heavy-tailed (t-distributed) noise to signal."""
    def __init__(self, rng: np.random.Generator, level: float = 0.1, df: float = 3.0):
        self.level = level
        self.df = df
        self.rng = rng
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        signal_std = np.std(data)
        noise = self.rng.standard_t(self.df, len(data)) * self.level * signal_std
        return data + noise


class MissingDataContamination:
    """Introduce missing data (NaN values) to signal."""
    def __init__(self, rng: np.random.Generator, level: float = 0.1):
        self.level = level
        self.rng = rng
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        result = data.copy()
        n_missing = int(self.level * len(data))
        missing_idx = self.rng.choice(len(data), n_missing, replace=False)
        result[missing_idx] = np.nan
        # Interpolate to handle NaNs
        df = pd.DataFrame(result)
        result = df.interpolate(method='linear', limit_direction='both').values.flatten()
        return result


class OutlierContamination:
    """Add outliers to signal."""
    def __init__(self, rng: np.random.Generator, level: float = 0.1, magnitude: float = 5.0):
        self.level = level
        self.magnitude = magnitude
        self.rng = rng
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        result = data.copy()
        n_outliers = int(self.level * len(data))
        outlier_idx = self.rng.choice(len(data), n_outliers, replace=False)
        signal_std = np.std(data)
        outlier_values = self.rng.choice([-1, 1], n_outliers) * self.magnitude * signal_std
        result[outlier_idx] += outlier_values
        return result


class MotionArtifactContamination:
    """Add motion artifact (sudden shifts) to signal."""
    def __init__(self, rng: np.random.Generator, level: float = 0.1):
        self.level = level
        self.rng = rng
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        result = data.copy()
        n = len(data)
        signal_std = np.std(data)
        # Number of motion events
        n_events = max(1, int(self.level * 10))
        for _ in range(n_events):
            # Random start and duration
            start = self.rng.integers(0, n - 50)
            duration = self.rng.integers(10, 50)
            end = min(start + duration, n)
            # Sharp displacement followed by recovery
            displacement = self.rng.normal(0, self.level * signal_std * 3)
            artifact = displacement * np.exp(-np.arange(end - start) / (duration / 3))
            result[start:end] += artifact
        return result


class NeuralAvalancheContamination:
    """Add neural avalanche patterns (burst-like activity)."""
    def __init__(self, rng: np.random.Generator, level: float = 0.1):
        self.level = level
        self.rng = rng
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        result = data.copy()
        n = len(data)
        signal_std = np.std(data)
        # Power-law distributed avalanche sizes
        n_avalanches = max(1, int(self.level * 20))
        for _ in range(n_avalanches):
            # Pareto-distributed duration (power-law)
            duration = int(min(100, max(5, self.rng.pareto(2.0) * 10)))
            start = self.rng.integers(0, max(1, n - duration))
            end = min(start + duration, n)
            # Avalanche shape: rapid rise, exponential decay
            t = np.arange(end - start)
            amplitude = self.rng.exponential(self.level * signal_std * 2)
            avalanche = amplitude * np.exp(-t / (duration / 2)) * (1 - np.exp(-t * 3 / duration))
            result[start:end] += avalanche
        return result


# =============================================================================
# EXPERIMENT RUNNER CLASS
# =============================================================================

class ExperimentRunner:
    """
    Comprehensive experiment runner for ML and NN LRD estimator benchmarking.
    
    Generates 860 total samples:
    - Pure Synthetic: 320 (4 models × 4 H values × 2 lengths × 10 reps)
    - Contaminated: 240 (3 H values × 8 types × 10 reps)  
    - Realistic: 300 (15 scenarios × 2 lengths × 10 reps)
    """
    
    def __init__(self, output_dir: str = "benchmark_results_ml_nn_v2", n_realizations: int = 10, 
                 mode: str = "full", lengths: list = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_realizations = n_realizations
        self.mode = mode
        self.benchmark = ComprehensiveBenchmark(output_dir=output_dir, runtime_profile="quick")
        
        # Initialize storage
        self.pure_results = []
        self.contaminated_results = []
        self.realistic_results = []
        
        # Define experiment parameters based on mode
        if self.mode == "fast":
            self.hurst_values = [0.5, 0.7]
            self.contamination_hurst_values = [0.7]
            self.lengths = lengths if lengths else [2048, 4096]
            self.mrw_lambda_values = [0.1]
        else:  # full mode
            self.hurst_values = [0.3, 0.5, 0.7, 0.9]
            self.contamination_hurst_values = [0.5, 0.7, 0.9]
            self.lengths = lengths if lengths else [1024, 4096]
            self.mrw_lambda_values = [0.05, 0.15]
        
        # Define contamination types (8 total)
        self.contamination_types = {
            "linear_trend": LinearTrendContamination,
            "polynomial_trend": PolynomialTrendContamination,
            "gaussian_noise": GaussianNoiseContamination,
            "heavy_tailed_noise": HeavyTailedNoiseContamination,
            "missing_data": MissingDataContamination,
            "outliers": OutlierContamination,
            "motion_artifacts": MotionArtifactContamination,
            "neural_avalanche": NeuralAvalancheContamination,
        }
        
        # Get ML and NN Estimators
        try:
            ml_est = self.benchmark.get_estimators_by_type("ML")
        except Exception:
            ml_est = {}
        
        try:
            nn_est = self.benchmark.get_estimators_by_type("neural")
        except Exception:
            nn_est = {}

        self.estimators = {**ml_est, **nn_est}
        
        print(f"Loaded {len(self.estimators)} ML/NN estimators: {list(self.estimators.keys())}")
        print(f"Mode: {self.mode}, Replications: {self.n_realizations}")

    def _save_incremental(self, result: Dict, filename: str):
        """Save each result incrementally to CSV."""
        df = pd.DataFrame([result])
        file_path = self.output_dir / filename
        header = not file_path.exists()
        df.to_csv(file_path, mode='a', header=header, index=False)

    def run_pure_experiment(self):
        """
        Run pure synthetic process experiments.
        Models: fBm, fGn, ARFIMA, MRW
        """
        print("\n" + "="*60)
        print("Experiment 1: Pure Synthetic Processes")
        print("="*60)
        
        # 1. fBm and fGn
        for model_name in ["fBm", "fGn"]:
            print(f"\nProcessing {model_name}...")
            for H in tqdm(self.hurst_values, desc=f"{model_name} H values"):
                for N in self.lengths:
                    for i in range(self.n_realizations):
                        try:
                            data, params = self.benchmark.generate_test_data(
                                model_name, data_length=N, H=H, random_state=None
                            )
                            
                            for est_name, estimator in self.estimators.items():
                                try:
                                    start_time = time.time()
                                    res = estimator.estimate(data)
                                    duration = (time.time() - start_time) * 1000  # ms
                                    
                                    h_est = res.get("hurst_parameter", np.nan)
                                    
                                    result = {
                                        "Model": model_name,
                                        "True_H": H,
                                        "Length": N,
                                        "Lambda": None,
                                        "Replication": i,
                                        "Estimator": est_name,
                                        "Estimate": h_est,
                                        "Error": h_est - H,
                                        "AbsError": abs(h_est - H),
                                        "SqError": (h_est - H)**2,
                                        "Time_ms": duration,
                                        "Success": not np.isnan(h_est)
                                    }
                                    self.pure_results.append(result)
                                    self._save_incremental(result, "pure_results.csv")
                                    
                                except Exception as e:
                                    result = {
                                        "Model": model_name,
                                        "True_H": H,
                                        "Length": N,
                                        "Lambda": None,
                                        "Replication": i,
                                        "Estimator": est_name,
                                        "Estimate": np.nan,
                                        "Error": np.nan,
                                        "AbsError": np.nan,
                                        "SqError": np.nan,
                                        "Time_ms": np.nan,
                                        "Success": False
                                    }
                                    self.pure_results.append(result)
                        except Exception as e:
                            print(f"    Error generating {model_name} data: {e}")

        # 2. ARFIMA
        print("\nProcessing ARFIMA...")
        for H in tqdm(self.hurst_values, desc="ARFIMA H values"):
            d = H - 0.5
            for N in self.lengths:
                for i in range(self.n_realizations):
                    try:
                        data, params = self.benchmark.generate_test_data(
                            "ARFIMAModel", data_length=N, d=d, random_state=None
                        )
                        
                        for est_name, estimator in self.estimators.items():
                            try:
                                start_time = time.time()
                                res = estimator.estimate(data)
                                duration = (time.time() - start_time) * 1000
                                h_est = res.get("hurst_parameter", np.nan)
                                
                                result = {
                                    "Model": "ARFIMA",
                                    "True_H": H,
                                    "Length": N,
                                    "Lambda": None,
                                    "Replication": i,
                                    "Estimator": est_name,
                                    "Estimate": h_est,
                                    "Error": h_est - H,
                                    "AbsError": abs(h_est - H),
                                    "SqError": (h_est - H)**2,
                                    "Time_ms": duration,
                                    "Success": not np.isnan(h_est)
                                }
                                self.pure_results.append(result)
                                self._save_incremental(result, "pure_results.csv")
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"    Error: {e}")

        # 3. MRW (Multifractal Random Walk)
        print("\nProcessing MRW...")
        for H in tqdm(self.hurst_values, desc="MRW H values"):
            for lambda_param in self.mrw_lambda_values:
                for N in self.lengths:
                    for i in range(self.n_realizations):
                        try:
                            data, params = self.benchmark.generate_test_data(
                                "MRW", data_length=N, H=H, lambda_param=lambda_param, random_state=None
                            )
                            
                            for est_name, estimator in self.estimators.items():
                                try:
                                    start_time = time.time()
                                    res = estimator.estimate(data)
                                    duration = (time.time() - start_time) * 1000
                                    h_est = res.get("hurst_parameter", np.nan)
                                    
                                    result = {
                                        "Model": "MRW",
                                        "True_H": H,
                                        "Length": N,
                                        "Lambda": lambda_param,
                                        "Replication": i,
                                        "Estimator": est_name,
                                        "Estimate": h_est,
                                        "Error": h_est - H,
                                        "AbsError": abs(h_est - H),
                                        "SqError": (h_est - H)**2,
                                        "Time_ms": duration,
                                        "Success": not np.isnan(h_est)
                                    }
                                    self.pure_results.append(result)
                                    self._save_incremental(result, "pure_results.csv")
                                except Exception:
                                    pass
                        except Exception as e:
                            print(f"    Error: {e}")

        print(f"\n✓ Pure experiments complete: {len(self.pure_results)} results")

    def run_contaminated_experiment(self):
        """
        Run contaminated process experiments.
        """
        print("\n" + "="*60)
        print("Experiment 2: Contaminated Processes")
        print("="*60)
        
        contamination_level = 0.1  # 10% of signal variance
        base_N = 4096 if self.mode == "full" else 1024
        
        for base_H in tqdm(self.contamination_hurst_values, desc="Base H values"):
            for cont_name, ContClass in self.contamination_types.items():
                print(f"\n  Processing: H={base_H}, {cont_name}")
                for i in range(self.n_realizations):
                    try:
                        # Generate clean fBm data
                        clean_data, _ = self.benchmark.generate_test_data(
                            "fBm", data_length=base_N, H=base_H, random_state=None
                        )
                        
                        # Apply contamination
                        rng = np.random.default_rng()
                        contaminator = ContClass(rng, level=contamination_level)
                        contaminated_data = contaminator.apply(clean_data)
                        
                        # Handle any NaNs from contamination
                        if np.isnan(contaminated_data).any():
                            df = pd.DataFrame(contaminated_data)
                            contaminated_data = df.interpolate(method='linear', limit_direction='both').values.flatten()
                        
                        for est_name, estimator in self.estimators.items():
                            try:
                                start_time = time.time()
                                res = estimator.estimate(contaminated_data)
                                duration = (time.time() - start_time) * 1000
                                h_est = res.get("hurst_parameter", np.nan)
                                
                                result = {
                                    "Base_H": base_H,
                                    "Contamination": cont_name,
                                    "Level": contamination_level,
                                    "Length": base_N,
                                    "Replication": i,
                                    "Estimator": est_name,
                                    "Estimate": h_est,
                                    "Error": h_est - base_H,
                                    "AbsError": abs(h_est - base_H),
                                    "SqError": (h_est - base_H)**2,
                                    "Time_ms": duration,
                                    "Success": not np.isnan(h_est)
                                }
                                self.contaminated_results.append(result)
                                self._save_incremental(result, "contaminated_results.csv")
                            except Exception:
                                result = {
                                    "Base_H": base_H,
                                    "Contamination": cont_name,
                                    "Level": contamination_level,
                                    "Length": base_N,
                                    "Replication": i,
                                    "Estimator": est_name,
                                    "Estimate": np.nan,
                                    "Error": np.nan,
                                    "AbsError": np.nan,
                                    "SqError": np.nan,
                                    "Time_ms": np.nan,
                                    "Success": False
                                }
                                self.contaminated_results.append(result)
                    except Exception as e:
                        print(f"Error: {e}")

        print(f"\n✓ Contaminated experiments complete: {len(self.contaminated_results)} results")

    def run_realistic_experiment(self):
        """
        Run realistic scenario experiments.
        """
        print("\n" + "="*60)
        print("Experiment 3: Realistic Scenarios")
        print("="*60)
        
        datasets = dataset_map()
        
        for name, spec in tqdm(datasets.items(), desc="Datasets"):
            for N in self.lengths:
                for i in range(self.n_realizations):
                    rng = np.random.default_rng(spec.base_seed + i + N)
                    data = spec.generator(N, rng)
                    
                    for est_name, estimator in self.estimators.items():
                        try:
                            start_time = time.time()
                            res = estimator.estimate(data)
                            duration = (time.time() - start_time) * 1000
                            h_est = res.get("hurst_parameter", np.nan)
                            
                            result = {
                                "Dataset": name,
                                "Domain": spec.domain,
                                "Length": N,
                                "Replication": i,
                                "Estimator": est_name,
                                "Estimate": h_est,
                                "Time_ms": duration,
                                "Success": not np.isnan(h_est)
                            }
                            self.realistic_results.append(result)
                            self._save_incremental(result, "realistic_results.csv")
                        except Exception:
                            result = {
                                "Dataset": name,
                                "Domain": spec.domain,
                                "Length": N,
                                "Replication": i,
                                "Estimator": est_name,
                                "Estimate": np.nan,
                                "Time_ms": np.nan,
                                "Success": False
                            }
                            self.realistic_results.append(result)

        print(f"\n✓ Realistic experiments complete: {len(self.realistic_results)} results")

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics and statistical tests."""
        print("\n" + "="*60)
        print("Computing Statistics and Statistical Tests")
        print("="*60)
        
        stats_results = {}
        
        # Load results
        df_pure = pd.DataFrame(self.pure_results)
        df_cont = pd.DataFrame(self.contaminated_results)
        df_real = pd.DataFrame(self.realistic_results)
        
        # --- Pure Data Statistics ---
        if not df_pure.empty:
            pure_stats = df_pure.groupby("Estimator").agg({
                "AbsError": ["mean", "std"],
                "SqError": lambda x: np.sqrt(x.mean()),
                "Error": "mean",  # Bias
                "Time_ms": "mean",
                "Success": "mean"
            }).round(4)
            pure_stats.columns = ["MAE", "MAE_std", "RMSE", "Bias", "Time_ms", "Success_Rate"]
            stats_results["pure"] = pure_stats.to_dict()
        
        # --- Contaminated Data Statistics ---
        if not df_cont.empty:
            cont_stats = df_cont.groupby("Estimator").agg({
                "AbsError": "mean",
                "SqError": lambda x: np.sqrt(x.mean()),
                "Success": "mean"
            }).round(4)
            cont_stats.columns = ["MAE", "RMSE", "Success_Rate"]
            stats_results["contaminated"] = cont_stats.to_dict()
        
        # --- Friedman Test ---
        if not df_pure.empty:
            try:
                # Pivot table
                pivot = df_pure.pivot_table(
                    values="AbsError",
                    index=["Model", "True_H", "Length", "Replication"],
                    columns="Estimator",
                    aggfunc="first"
                ).dropna()
                
                if pivot.shape[0] >= 3 and pivot.shape[1] >= 3:
                    stat, p_value = stats.friedmanchisquare(*[pivot[col].values for col in pivot.columns])
                    stats_results["friedman_test"] = {
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "n_conditions": int(pivot.shape[0]),
                        "n_estimators": int(pivot.shape[1]),
                        "significant": p_value < 0.05
                    }
                    print(f"  Friedman Test: χ²={stat:.4f}, p={p_value:.4e}")
            except Exception as e:
                print(f"  Friedman test error: {e}")
        
        # --- Composite Scoring ---
        if not df_pure.empty and not df_cont.empty:
            try:
                pure_mae = df_pure.groupby("Estimator")["AbsError"].mean()
                pure_time = df_pure.groupby("Estimator")["Time_ms"].mean()
                pure_success = df_pure.groupby("Estimator")["Success"].mean()
                cont_mae = df_cont.groupby("Estimator")["AbsError"].mean()
                
                # Normalize 
                precision_score = np.exp(-pure_mae / pure_mae.median())
                speed_score = np.exp(-pure_time / pure_time.median())
                robustness_score = pure_success
                
                # Heavy-tail performance: relative degradation
                heavy_tail_score = 1 - (cont_mae / pure_mae).clip(0, 2) / 2
                heavy_tail_score = heavy_tail_score.fillna(0.5)
                
                # Composite score
                composite = (
                    0.50 * precision_score +
                    0.25 * speed_score +
                    0.15 * robustness_score +
                    0.10 * heavy_tail_score
                )
                
                leaderboard = pd.DataFrame({
                    "Precision": precision_score,
                    "Speed": speed_score,
                    "Robustness": robustness_score,
                    "HeavyTail": heavy_tail_score,
                    "Composite": composite
                }).sort_values("Composite", ascending=False).round(4)
                
                stats_results["composite_scores"] = leaderboard.to_dict()
                
                # Save leaderboard
                leaderboard.to_csv(self.output_dir / "leaderboard.csv")
                print(f"\n  Leaderboard saved to {self.output_dir / 'leaderboard.csv'}")
            except Exception as e:
                print(f"  Composite scoring error: {e}")
        
        # Save statistics
        with open(self.output_dir / "statistical_analysis.json", "w") as f:
            json.dump(stats_results, f, indent=2, default=str)
        
        return stats_results

    def generate_report(self):
        """Generate comprehensive report and visualizations."""
        print("\n" + "="*60)
        print("Generating Report and Visualizations")
        print("="*60)
        
        df_pure = pd.DataFrame(self.pure_results)
        df_cont = pd.DataFrame(self.contaminated_results)
        df_real = pd.DataFrame(self.realistic_results)
        
        # Save consolidated CSVs
        if not df_pure.empty:
            df_pure.to_csv(self.output_dir / "pure_results.csv", index=False)
        if not df_cont.empty:
            df_cont.to_csv(self.output_dir / "contaminated_results.csv", index=False)
        if not df_real.empty:
            df_real.to_csv(self.output_dir / "realistic_results.csv", index=False)
        
        # --- Visualization 1: RMSE by Estimator for Pure Data ---
        if not df_pure.empty:
            plt.figure(figsize=(12, 6))
            rmse_by_est = df_pure.groupby("Estimator").apply(
                lambda x: np.sqrt((x["SqError"]).mean())
            ).sort_values()
            
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(rmse_by_est)))
            bars = plt.bar(range(len(rmse_by_est)), rmse_by_est.values, color=colors)
            plt.xticks(range(len(rmse_by_est)), rmse_by_est.index, rotation=45, ha='right')
            plt.ylabel("RMSE")
            plt.title("ML/NN Estimator Performance on Pure Synthetic Data (RMSE)")
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "plot_pure_rmse.png", dpi=300)
            plt.close()
        
        # --- Visualization 2: Robustness Heatmap ---
        if not df_cont.empty:
            plt.figure(figsize=(12, 8))
            robust_pivot = df_cont.pivot_table(
                values="AbsError",
                index="Contamination",
                columns="Estimator",
                aggfunc="mean"
            )
            sns.heatmap(robust_pivot, annot=True, fmt=".3f", cmap="RdYlGn_r")
            plt.title("Robustness to Contamination (MAE)")
            plt.tight_layout()
            plt.savefig(self.output_dir / "plot_robustness_heatmap.png", dpi=300)
            plt.close()
        
        # --- Visualization 3: Realistic Domain Comparison ---
        if not df_real.empty:
            plt.figure(figsize=(14, 8))
            domain_stats = df_real.groupby(["Domain", "Estimator"])["Estimate"].agg(["mean", "std"])
            domain_pivot = domain_stats["mean"].unstack()
            
            domain_pivot.plot(kind="bar", figsize=(14, 8), width=0.8)
            plt.ylabel("Mean Hurst Estimate")
            plt.title("Hurst Estimates by Domain and Estimator")
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "plot_realistic_domains.png", dpi=300)
            plt.close()
        
        # --- LaTeX Tables ---
        if not df_pure.empty:
            # Table 1: RMSE by Model and H
            stats_table = df_pure.groupby(["Model", "True_H", "Estimator"]).agg({
                "AbsError": "mean",
                "SqError": lambda x: np.sqrt(x.mean())
            }).unstack()
            
            latex_table = stats_table.to_latex(float_format="%.3f", multirow=True)
            with open(self.output_dir / "table_pure_stats.tex", "w") as f:
                f.write(latex_table)
        
        print(f"✓ Report generation complete. Output: {self.output_dir}")

    def run_full_experiment(self):
        """Run the complete benchmarking experiment."""
        print("\n" + "="*60)
        print("ML/NN LRD ESTIMATOR BENCHMARKING EXPERIMENT")
        print("="*60)
        print(f"Mode: {self.mode}")
        print(f"Replications: {self.n_realizations}")
        print(f"Output directory: {self.output_dir}")
        print(f"Estimators: {list(self.estimators.keys())}")
        print("="*60)
        
        start_time = time.time()
        
        # Run all experiments
        self.run_pure_experiment()
        self.run_contaminated_experiment()
        self.run_realistic_experiment()
        
        # Compute statistics
        self.compute_statistics()
        
        # Generate report
        self.generate_report()
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE")
        print("="*60)
        print(f"Total samples processed: {len(self.pure_results) + len(self.contaminated_results) + len(self.realistic_results)}")
        print(f"Total execution time: {total_time/60:.2f} minutes")
        print(f"Output saved to: {self.output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ML/NN LRD benchmarking experiment")
    parser.add_argument("--n_realizations", "-n", type=int, default=10, 
                       help="Number of realizations per condition (default: 10)")
    parser.add_argument("--output", "-o", type=str, default="benchmark_results_ml_nn", 
                       help="Output directory")
    parser.add_argument("--mode", "-m", type=str, default="full", choices=["fast", "full"], 
                       help="Experiment mode: fast (reduced) or full")
    parser.add_argument("--lengths", "-l", type=int, nargs="+", default=None,
                       help="Data lengths to use (e.g., --lengths 2048 4096)")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        output_dir=args.output, 
        n_realizations=args.n_realizations, 
        mode=args.mode,
        lengths=args.lengths
    )
    runner.run_full_experiment()


if __name__ == "__main__":
    main()
