#!/usr/bin/env python3
"""
Comprehensive Benchmark: All Estimators (Classical, ML, NN) on Pure and EEG Contamination Scenarios

This script runs a complete benchmark comparing:
- Classical estimators (R/S, DFA, DMA, Higuchi, GPH, Whittle, Periodogram)
- Machine Learning estimators (SVR, Gradient Boosting, Random Forest)
- Neural Network estimators (8 architectures)

On both:
- Pure synthetic data (FBM, FGN)
- EEG contamination scenarios (4 realistic biological artifacts)

This provides a comprehensive evaluation grounded in plausible biological contexts.
"""

import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime
import signal

# Import data models
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion as FBMModel
from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise as FGNModel

# Import contamination factory
from lrdbenchmark.models.contamination.contamination_factory import ContaminationFactory

# Import classical estimators
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.temporal.dma.dma_estimator_unified import DMAEstimator
from lrdbenchmark.analysis.temporal.higuchi.higuchi_estimator_unified import HiguchiEstimator
from lrdbenchmark.analysis.spectral.gph.gph_estimator_unified import GPHEstimator
from lrdbenchmark.analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator
from lrdbenchmark.analysis.spectral.periodogram.periodogram_estimator_unified import PeriodogramEstimator

# Import ML estimators
from lrdbenchmark.analysis.machine_learning.svr_estimator import SVREstimator
from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator import GradientBoostingEstimator
from lrdbenchmark.analysis.machine_learning.random_forest_estimator import RandomForestEstimator

# Import Neural Network factory
from lrdbenchmark.analysis.machine_learning.neural_network_factory import (
    NeuralNetworkFactory, create_all_benchmark_networks
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Operation timed out")

class ComprehensivePureAndEEGBenchmark:
    """
    Comprehensive benchmark comparing all estimator types on pure and EEG contamination scenarios.
    """
    
    def __init__(self, output_dir: str = "comprehensive_pure_eeg_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data models (will be created per test with specific H values)
        self.fbm_model_class = FBMModel
        self.fgn_model_class = FGNModel
        
        # Initialize contamination factory
        self.contamination_factory = ContaminationFactory()
        
        # Initialize all estimators
        self.classical_estimators = self._initialize_classical_estimators()
        self.ml_estimators = self._initialize_ml_estimators()
        self.nn_estimators = self._initialize_nn_estimators()
        
        # EEG contamination scenarios
        self.eeg_scenarios = [
            "eeg_ocular_artifacts",
            "eeg_muscle_artifacts", 
            "eeg_movement_artifacts",
            "eeg_60hz_noise"
        ]
        
        # Test parameters (reduced for faster testing)
        self.hurst_values = [0.2, 0.4, 0.6, 0.8]
        self.data_lengths = [500, 1000]
        self.n_samples_per_condition = 5
        
        logger.info(f"Initialized comprehensive benchmark with {len(self.classical_estimators)} classical, "
                   f"{len(self.ml_estimators)} ML, and {len(self.nn_estimators)} neural network estimators")
    
    def _initialize_classical_estimators(self) -> Dict[str, Any]:
        """Initialize classical estimators."""
        return {
            "RS": RSEstimator(),
            "DFA": DFAEstimator(),
            "DMA": DMAEstimator(),
            "Higuchi": HiguchiEstimator(),
            "GPH": GPHEstimator(),
            "Whittle": WhittleEstimator(),
            "Periodogram": PeriodogramEstimator()
        }
    
    def _initialize_ml_estimators(self) -> Dict[str, Any]:
        """Initialize ML estimators."""
        return {
            "SVR": SVREstimator(),
            "GradientBoosting": GradientBoostingEstimator(),
            "RandomForest": RandomForestEstimator()
        }
    
    def _initialize_nn_estimators(self) -> Dict[str, Any]:
        """Initialize neural network estimators."""
        networks = create_all_benchmark_networks(input_length=500)
        return networks
    
    def generate_test_data(self, hurst: float, length: int, data_type: str = "fbm") -> np.ndarray:
        """Generate test data with specified Hurst parameter."""
        if data_type == "fbm":
            model = self.fbm_model_class(H=hurst)
            return model.generate(n=length)
        elif data_type == "fgn":
            model = self.fgn_model_class(H=hurst)
            return model.generate(n=length)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def apply_eeg_contamination(self, data: np.ndarray, scenario: str) -> np.ndarray:
        """Apply EEG contamination scenario to data."""
        from lrdbenchmark.models.contamination.contamination_factory import ConfoundingScenario
        
        # Map string scenario to enum
        scenario_map = {
            "eeg_ocular_artifacts": ConfoundingScenario.EEG_OCULAR_ARTIFACTS,
            "eeg_muscle_artifacts": ConfoundingScenario.EEG_MUSCLE_ARTIFACTS,
            "eeg_movement_artifacts": ConfoundingScenario.EEG_MOVEMENT_ARTIFACTS,
            "eeg_60hz_noise": ConfoundingScenario.EEG_60HZ_NOISE
        }
        
        if scenario not in scenario_map:
            raise ValueError(f"Unknown EEG scenario: {scenario}")
        
        confounding_profile = self.contamination_factory.create_confounding_profile(scenario_map[scenario])
        return self.contamination_factory.apply_confounding(data, confounding_profile)
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark on all estimators and scenarios."""
        logger.info("🚀 Starting comprehensive pure and EEG contamination benchmark")
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_classical": len(self.classical_estimators),
                "n_ml": len(self.ml_estimators),
                "n_neural_network": len(self.nn_estimators),
                "n_eeg_scenarios": len(self.eeg_scenarios),
                "hurst_values": self.hurst_values,
                "data_lengths": self.data_lengths,
                "n_samples_per_condition": self.n_samples_per_condition
            },
            "pure_data_results": {},
            "eeg_contamination_results": {},
            "summary": {}
        }
        
        # Benchmark on pure data
        logger.info("📊 Benchmarking on pure data...")
        results["pure_data_results"] = self._benchmark_scenario("pure", None)
        
        # Benchmark on each EEG contamination scenario
        for scenario in self.eeg_scenarios:
            logger.info(f"🧠 Benchmarking on EEG scenario: {scenario}")
            results["eeg_contamination_results"][scenario] = self._benchmark_scenario("eeg", scenario)
        
        # Generate summary statistics
        results["summary"] = self._generate_summary(results)
        
        # Save results
        self._save_results(results)
        
        logger.info("✅ Comprehensive benchmark completed!")
        return results
    
    def _benchmark_scenario(self, scenario_type: str, contamination_scenario: str = None) -> Dict[str, Any]:
        """Benchmark all estimators on a specific scenario."""
        scenario_results = {
            "classical": {},
            "ml": {},
            "neural_network": {}
        }
        
        total_tests = len(self.hurst_values) * len(self.data_lengths) * self.n_samples_per_condition
        logger.info(f"  Running {total_tests} tests per estimator...")
        
        # Test classical estimators
        logger.info("  Testing Classical Estimators...")
        for name, estimator in self.classical_estimators.items():
            logger.info(f"    Testing {name}...")
            scenario_results["classical"][name] = self._test_estimator(
                estimator, name, scenario_type, contamination_scenario
            )
        
        # Test ML estimators
        logger.info("  Testing ML Estimators...")
        for name, estimator in self.ml_estimators.items():
            logger.info(f"    Testing {name}...")
            scenario_results["ml"][name] = self._test_estimator(
                estimator, name, scenario_type, contamination_scenario
            )
        
        # Test Neural Network estimators
        logger.info("  Testing Neural Network Estimators...")
        for name, network in self.nn_estimators.items():
            logger.info(f"    Testing {name}...")
            scenario_results["neural_network"][name] = self._test_estimator(
                network, name, scenario_type, contamination_scenario
            )
        
        return scenario_results
    
    def _test_estimator(self, estimator: Any, name: str, scenario_type: str, contamination_scenario: str = None) -> Dict[str, Any]:
        """Test a single estimator on all conditions with timeout protection."""
        results = {
            "estimator_name": name,
            "scenario_type": scenario_type,
            "contamination_scenario": contamination_scenario,
            "test_results": [],
            "summary": {}
        }
        
        successful_tests = 0
        total_tests = 0
        errors = []
        execution_times = []
        mae_values = []
        timeout_count = 0
        
        total_conditions = len(self.hurst_values) * len(self.data_lengths) * self.n_samples_per_condition
        logger.info(f"    Running {total_conditions} tests for {name}...")
        
        for hurst_idx, hurst in enumerate(self.hurst_values):
            for length_idx, length in enumerate(self.data_lengths):
                for sample_idx in range(self.n_samples_per_condition):
                    total_tests += 1
                    test_num = total_tests
                    
                    # Progress reporting
                    if test_num % 5 == 0 or test_num == 1:
                        logger.info(f"      Test {test_num}/{total_conditions} (H={hurst}, L={length}, S={sample_idx})")
                    
                    try:
                        # Generate test data
                        data = self.generate_test_data(hurst, length, "fbm")
                        
                        # Apply contamination if specified
                        if contamination_scenario:
                            data = self.apply_eeg_contamination(data, contamination_scenario)
                        
                        # Test estimator with timeout protection
                        start_time = time.time()
                        
                        # Set timeout for individual estimator calls (30 seconds)
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(30)
                        
                        try:
                            if hasattr(estimator, 'estimate'):
                                # Classical estimators
                                result = estimator.estimate(data)
                                if isinstance(result, dict):
                                    estimated_hurst = result.get('hurst_parameter', 0.5)
                                else:
                                    estimated_hurst = float(result)
                            elif hasattr(estimator, 'predict'):
                                # ML and NN estimators
                                if len(data.shape) == 1:
                                    data = data.reshape(1, -1)
                                estimated_hurst = estimator.predict(data)[0]
                            else:
                                raise ValueError(f"Unknown estimator type: {type(estimator)}")
                            
                            signal.alarm(0)  # Cancel timeout
                            execution_time = time.time() - start_time
                            
                            # Calculate metrics
                            mae = abs(estimated_hurst - hurst)
                            
                            # Store results
                            test_result = {
                                "true_hurst": hurst,
                                "estimated_hurst": estimated_hurst,
                                "mae": mae,
                                "execution_time": execution_time,
                                "data_length": length,
                                "sample_idx": sample_idx
                            }
                            results["test_results"].append(test_result)
                            
                            successful_tests += 1
                            execution_times.append(execution_time)
                            mae_values.append(mae)
                            
                        except TimeoutError:
                            signal.alarm(0)  # Cancel timeout
                            timeout_count += 1
                            error_msg = f"Test timed out after 30s"
                            errors.append(error_msg)
                            logger.warning(f"      {error_msg}")
                            
                    except Exception as e:
                        signal.alarm(0)  # Cancel timeout
                        error_msg = f"Test failed: {str(e)}"
                        errors.append(error_msg)
                        logger.warning(f"      {error_msg}")
        
        # Final progress report
        logger.info(f"    {name} completed: {successful_tests}/{total_tests} successful, {timeout_count} timeouts")
        
        # Calculate summary statistics
        if mae_values:
            results["summary"] = {
                "success_rate": successful_tests / total_tests,
                "mean_mae": np.mean(mae_values),
                "std_mae": np.std(mae_values),
                "mean_execution_time": np.mean(execution_times),
                "std_execution_time": np.std(execution_times),
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "timeout_count": timeout_count,
                "error_count": len(errors),
                "errors": errors
            }
        else:
            results["summary"] = {
                "success_rate": 0.0,
                "mean_mae": float('inf'),
                "std_mae": 0.0,
                "mean_execution_time": float('inf'),
                "std_execution_time": 0.0,
                "total_tests": total_tests,
                "successful_tests": 0,
                "timeout_count": timeout_count,
                "error_count": len(errors),
                "errors": errors
            }
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary statistics."""
        summary = {
            "overall": {},
            "by_scenario": {},
            "by_estimator_type": {},
            "eeg_contamination_analysis": {}
        }
        
        # Overall statistics
        all_mae_values = []
        all_success_rates = []
        all_execution_times = []
        
        # Pure data summary
        pure_results = results["pure_data_results"]
        pure_summary = self._calculate_scenario_summary(pure_results)
        summary["by_scenario"]["pure_data"] = pure_summary
        
        # EEG contamination summaries
        for scenario, scenario_results in results["eeg_contamination_results"].items():
            scenario_summary = self._calculate_scenario_summary(scenario_results)
            summary["by_scenario"][scenario] = scenario_summary
        
        # EEG contamination analysis
        summary["eeg_contamination_analysis"] = self._analyze_eeg_contamination_effects(results)
        
        return summary
    
    def _calculate_scenario_summary(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for a scenario."""
        summary = {
            "classical": {},
            "ml": {},
            "neural_network": {},
            "overall": {}
        }
        
        all_mae_values = []
        all_success_rates = []
        all_execution_times = []
        
        for estimator_type in ["classical", "ml", "neural_network"]:
            if estimator_type in scenario_results:
                type_results = scenario_results[estimator_type]
                mae_values = []
                success_rates = []
                execution_times = []
                
                for estimator_name, estimator_results in type_results.items():
                    if estimator_results["summary"]["successful_tests"] > 0:
                        mae_values.append(estimator_results["summary"]["mean_mae"])
                        success_rates.append(estimator_results["summary"]["success_rate"])
                        execution_times.append(estimator_results["summary"]["mean_execution_time"])
                
                if mae_values:
                    summary[estimator_type] = {
                        "mean_mae": np.mean(mae_values),
                        "std_mae": np.std(mae_values),
                        "mean_success_rate": np.mean(success_rates),
                        "std_success_rate": np.std(success_rates),
                        "mean_execution_time": np.mean(execution_times),
                        "std_execution_time": np.std(execution_times),
                        "n_estimators": len(mae_values)
                    }
                    
                    all_mae_values.extend(mae_values)
                    all_success_rates.extend(success_rates)
                    all_execution_times.extend(execution_times)
        
        # Overall summary
        if all_mae_values:
            summary["overall"] = {
                "mean_mae": np.mean(all_mae_values),
                "std_mae": np.std(all_mae_values),
                "mean_success_rate": np.mean(all_success_rates),
                "std_success_rate": np.std(all_success_rates),
                "mean_execution_time": np.mean(all_execution_times),
                "std_execution_time": np.std(all_execution_times),
                "total_estimators": len(all_mae_values)
            }
        
        return summary
    
    def _analyze_eeg_contamination_effects(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the effects of EEG contamination on estimator performance."""
        analysis = {
            "contamination_impact": {},
            "robustness_ranking": {},
            "biological_context": {}
        }
        
        # Get pure data summary from the results structure
        pure_summary = None
        if "pure_data_results" in results:
            # Calculate pure data summary
            pure_results = results["pure_data_results"]
            pure_summary = self._calculate_scenario_summary(pure_results)
        
        # Analyze contamination impact
        for scenario in self.eeg_scenarios:
            if scenario in results["eeg_contamination_results"]:
                contaminated_results = results["eeg_contamination_results"][scenario]
                contaminated_summary = self._calculate_scenario_summary(contaminated_results)
                
                # Calculate performance degradation
                if (pure_summary and "overall" in pure_summary and 
                    "mean_mae" in pure_summary["overall"] and pure_summary["overall"]["mean_mae"] > 0):
                    mae_degradation = (contaminated_summary["overall"]["mean_mae"] - pure_summary["overall"]["mean_mae"]) / pure_summary["overall"]["mean_mae"] * 100
                else:
                    mae_degradation = 0
                
                if (pure_summary and "overall" in pure_summary and 
                    "mean_success_rate" in pure_summary["overall"]):
                    success_rate_change = contaminated_summary["overall"]["mean_success_rate"] - pure_summary["overall"]["mean_success_rate"]
                else:
                    success_rate_change = 0
                
                analysis["contamination_impact"][scenario] = {
                    "mae_degradation_percent": mae_degradation,
                    "success_rate_change": success_rate_change,
                    "pure_mae": pure_summary["overall"]["mean_mae"] if (pure_summary and "overall" in pure_summary and "mean_mae" in pure_summary["overall"]) else 0,
                    "contaminated_mae": contaminated_summary["overall"]["mean_mae"],
                    "pure_success_rate": pure_summary["overall"]["mean_success_rate"] if (pure_summary and "overall" in pure_summary and "mean_success_rate" in pure_summary["overall"]) else 0,
                    "contaminated_success_rate": contaminated_summary["overall"]["mean_success_rate"]
                }
        
        # Biological context descriptions
        analysis["biological_context"] = {
            "eeg_ocular_artifacts": {
                "description": "Eye movements and blinks that create large amplitude, low-frequency artifacts",
                "frequency_range": "0.5-4 Hz",
                "amplitude": "High (50-200 μV)",
                "clinical_relevance": "Common in awake EEG recordings, affects frontal channels"
            },
            "eeg_muscle_artifacts": {
                "description": "Muscle activity from jaw clenching, facial movements, and neck tension",
                "frequency_range": "20-100 Hz",
                "amplitude": "Variable (10-100 μV)",
                "clinical_relevance": "Affects temporal and occipital regions, common in clinical settings"
            },
            "eeg_movement_artifacts": {
                "description": "Head movements, body motion, and electrode displacement",
                "frequency_range": "0.1-10 Hz",
                "amplitude": "High (100-500 μV)",
                "clinical_relevance": "Particularly problematic in pediatric and movement disorder patients"
            },
            "eeg_60hz_noise": {
                "description": "Electrical line noise from power grid interference",
                "frequency_range": "60 Hz ± harmonics",
                "amplitude": "Low (5-20 μV)",
                "clinical_relevance": "Ubiquitous in clinical environments, affects all channels"
            }
        }
        
        return analysis
    
    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = self.output_dir / f"comprehensive_pure_eeg_benchmark_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV summary
        csv_path = self.output_dir / f"comprehensive_pure_eeg_benchmark_{timestamp}.csv"
        self._save_csv_summary(results, csv_path)
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
    
    def _save_csv_summary(self, results: Dict[str, Any], csv_path: Path):
        """Save summary results to CSV format."""
        rows = []
        
        # Pure data results
        pure_results = results["pure_data_results"]
        for estimator_type in ["classical", "ml", "neural_network"]:
            if estimator_type in pure_results:
                for estimator_name, estimator_results in pure_results[estimator_type].items():
                    summary = estimator_results["summary"]
                    rows.append({
                        "scenario": "pure_data",
                        "estimator_type": estimator_type,
                        "estimator_name": estimator_name,
                        "mean_mae": summary["mean_mae"],
                        "success_rate": summary["success_rate"],
                        "mean_execution_time": summary["mean_execution_time"],
                        "total_tests": summary["total_tests"],
                        "successful_tests": summary["successful_tests"]
                    })
        
        # EEG contamination results
        for scenario, scenario_results in results["eeg_contamination_results"].items():
            for estimator_type in ["classical", "ml", "neural_network"]:
                if estimator_type in scenario_results:
                    for estimator_name, estimator_results in scenario_results[estimator_type].items():
                        summary = estimator_results["summary"]
                        rows.append({
                            "scenario": scenario,
                            "estimator_type": estimator_type,
                            "estimator_name": estimator_name,
                            "mean_mae": summary["mean_mae"],
                            "success_rate": summary["success_rate"],
                            "mean_execution_time": summary["mean_execution_time"],
                            "total_tests": summary["total_tests"],
                            "successful_tests": summary["successful_tests"]
                        })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

def main():
    """Run the comprehensive pure and EEG contamination benchmark."""
    benchmark = ComprehensivePureAndEEGBenchmark()
    results = benchmark.run_benchmark()
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE PURE AND EEG CONTAMINATION BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\n📊 Overall Results:")
    print(f"  Total Estimators: {results['metadata']['n_classical'] + results['metadata']['n_ml'] + results['metadata']['n_neural_network']}")
    print(f"  Classical: {results['metadata']['n_classical']}")
    print(f"  ML: {results['metadata']['n_ml']}")
    print(f"  Neural Networks: {results['metadata']['n_neural_network']}")
    print(f"  EEG Scenarios: {results['metadata']['n_eeg_scenarios']}")
    
    print(f"\n🧠 EEG Contamination Analysis:")
    for scenario, impact in results['summary']['eeg_contamination_analysis']['contamination_impact'].items():
        print(f"  {scenario}: {impact['mae_degradation_percent']:.1f}% MAE degradation")
    
    print(f"\n✅ Benchmark completed! Results saved to comprehensive_pure_eeg_results/")

if __name__ == "__main__":
    main()
