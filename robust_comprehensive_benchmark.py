#!/usr/bin/env python3
"""
Robust Comprehensive Benchmark: All Working Estimators on Pure Data

This script runs a comprehensive benchmark comparing:
- Classical estimators (R/S, DFA, DMA, Higuchi, GPH, Whittle, Periodogram)
- Machine Learning estimators (SVR, Gradient Boosting, Random Forest)

On pure synthetic data (FBM, FGN) with robust error handling.
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Operation timed out")

class RobustComprehensiveBenchmark:
    """
    Robust comprehensive benchmark comparing all working estimator types on pure data.
    """
    
    def __init__(self, output_dir: str = "robust_comprehensive_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data models
        self.fbm_model_class = FBMModel
        self.fgn_model_class = FGNModel
        
        # Initialize all working estimators
        self.classical_estimators = self._initialize_classical_estimators()
        self.ml_estimators = self._initialize_ml_estimators()
        
        # Test parameters (balanced for comprehensive testing)
        self.hurst_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.data_lengths = [500, 1000, 2000]
        self.n_samples_per_condition = 10
        
        logger.info(f"Initialized robust benchmark with {len(self.classical_estimators)} classical and {len(self.ml_estimators)} ML estimators")
    
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
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark on pure data only."""
        logger.info("🚀 Starting robust comprehensive benchmark")
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_classical": len(self.classical_estimators),
                "n_ml": len(self.ml_estimators),
                "hurst_values": self.hurst_values,
                "data_lengths": self.data_lengths,
                "n_samples_per_condition": self.n_samples_per_condition
            },
            "classical_results": {},
            "ml_results": {},
            "summary": {}
        }
        
        # Test classical estimators
        logger.info("📊 Testing Classical Estimators...")
        for name, estimator in self.classical_estimators.items():
            logger.info(f"  Testing {name}...")
            results["classical_results"][name] = self.test_estimator(estimator, name, "classical")
        
        # Test ML estimators
        logger.info("🤖 Testing ML Estimators...")
        for name, estimator in self.ml_estimators.items():
            logger.info(f"  Testing {name}...")
            results["ml_results"][name] = self.test_estimator(estimator, name, "ml")
        
        # Generate summary
        results["summary"] = self.generate_summary(results)
        
        # Save results
        self.save_results(results)
        
        logger.info("✅ Robust comprehensive benchmark completed!")
        return results
    
    def test_estimator(self, estimator: Any, name: str, estimator_type: str) -> Dict[str, Any]:
        """Test a single estimator with robust error handling and timeout protection."""
        results = {
            "estimator_name": name,
            "estimator_type": estimator_type,
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
        
        for hurst in self.hurst_values:
            for length in self.data_lengths:
                for sample_idx in range(self.n_samples_per_condition):
                    total_tests += 1
                    test_num = total_tests
                    
                    # Progress reporting
                    if test_num % 20 == 0 or test_num == 1:
                        logger.info(f"      Test {test_num}/{total_conditions} (H={hurst}, L={length}, S={sample_idx})")
                    
                    try:
                        # Generate test data
                        data = self.generate_test_data(hurst, length, "fbm")
                        
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
                                # ML estimators
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
                            logger.warning(f"        {error_msg}")
                            
                    except Exception as e:
                        signal.alarm(0)  # Cancel timeout
                        error_msg = f"Test failed: {str(e)}"
                        errors.append(error_msg)
                        logger.warning(f"        {error_msg}")
        
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
                "errors": errors[:10]  # Limit to first 10 errors
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
                "errors": errors[:10]  # Limit to first 10 errors
            }
        
        # Final progress report
        logger.info(f"    {name} completed: {successful_tests}/{total_tests} successful, {timeout_count} timeouts")
        return results
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        summary = {
            "classical": {},
            "ml": {},
            "overall": {},
            "performance_ranking": {},
            "detailed_analysis": {}
        }
        
        all_mae_values = []
        all_success_rates = []
        all_execution_times = []
        
        # Classical summary
        classical_mae_values = []
        classical_success_rates = []
        classical_execution_times = []
        classical_estimators = []
        
        for name, result in results["classical_results"].items():
            if result["summary"]["successful_tests"] > 0:
                classical_mae_values.append(result["summary"]["mean_mae"])
                classical_success_rates.append(result["summary"]["success_rate"])
                classical_execution_times.append(result["summary"]["mean_execution_time"])
                classical_estimators.append(name)
        
        if classical_mae_values:
            summary["classical"] = {
                "mean_mae": np.mean(classical_mae_values),
                "std_mae": np.std(classical_mae_values),
                "mean_success_rate": np.mean(classical_success_rates),
                "mean_execution_time": np.mean(classical_execution_times),
                "n_estimators": len(classical_mae_values),
                "best_estimator": classical_estimators[np.argmin(classical_mae_values)],
                "worst_estimator": classical_estimators[np.argmax(classical_mae_values)]
            }
            all_mae_values.extend(classical_mae_values)
            all_success_rates.extend(classical_success_rates)
            all_execution_times.extend(classical_execution_times)
        
        # ML summary
        ml_mae_values = []
        ml_success_rates = []
        ml_execution_times = []
        ml_estimators = []
        
        for name, result in results["ml_results"].items():
            if result["summary"]["successful_tests"] > 0:
                ml_mae_values.append(result["summary"]["mean_mae"])
                ml_success_rates.append(result["summary"]["success_rate"])
                ml_execution_times.append(result["summary"]["mean_execution_time"])
                ml_estimators.append(name)
        
        if ml_mae_values:
            summary["ml"] = {
                "mean_mae": np.mean(ml_mae_values),
                "std_mae": np.std(ml_mae_values),
                "mean_success_rate": np.mean(ml_success_rates),
                "mean_execution_time": np.mean(ml_execution_times),
                "n_estimators": len(ml_mae_values),
                "best_estimator": ml_estimators[np.argmin(ml_mae_values)],
                "worst_estimator": ml_estimators[np.argmax(ml_mae_values)]
            }
            all_mae_values.extend(ml_mae_values)
            all_success_rates.extend(ml_success_rates)
            all_execution_times.extend(ml_execution_times)
        
        # Overall summary
        if all_mae_values:
            summary["overall"] = {
                "mean_mae": np.mean(all_mae_values),
                "std_mae": np.std(all_mae_values),
                "mean_success_rate": np.mean(all_success_rates),
                "mean_execution_time": np.mean(all_execution_times),
                "total_estimators": len(all_mae_values),
                "total_tests": sum(result["summary"]["total_tests"] for result in results["classical_results"].values()) + 
                              sum(result["summary"]["total_tests"] for result in results["ml_results"].values()),
                "total_successful_tests": sum(result["summary"]["successful_tests"] for result in results["classical_results"].values()) + 
                                         sum(result["summary"]["successful_tests"] for result in results["ml_results"].values())
            }
        
        # Performance ranking
        all_estimator_results = []
        for name, result in results["classical_results"].items():
            if result["summary"]["successful_tests"] > 0:
                all_estimator_results.append({
                    "name": name,
                    "type": "classical",
                    "mean_mae": result["summary"]["mean_mae"],
                    "success_rate": result["summary"]["success_rate"],
                    "mean_execution_time": result["summary"]["mean_execution_time"]
                })
        
        for name, result in results["ml_results"].items():
            if result["summary"]["successful_tests"] > 0:
                all_estimator_results.append({
                    "name": name,
                    "type": "ml",
                    "mean_mae": result["summary"]["mean_mae"],
                    "success_rate": result["summary"]["success_rate"],
                    "mean_execution_time": result["summary"]["mean_execution_time"]
                })
        
        # Sort by MAE (lower is better)
        all_estimator_results.sort(key=lambda x: x["mean_mae"])
        summary["performance_ranking"] = {
            "by_accuracy": all_estimator_results,
            "top_3": all_estimator_results[:3],
            "bottom_3": all_estimator_results[-3:]
        }
        
        # Detailed analysis
        summary["detailed_analysis"] = {
            "classical_vs_ml": {
                "classical_mean_mae": np.mean(classical_mae_values) if classical_mae_values else None,
                "ml_mean_mae": np.mean(ml_mae_values) if ml_mae_values else None,
                "classical_mean_time": np.mean(classical_execution_times) if classical_execution_times else None,
                "ml_mean_time": np.mean(ml_execution_times) if ml_execution_times else None
            },
            "data_length_analysis": self._analyze_by_data_length(results),
            "hurst_analysis": self._analyze_by_hurst_value(results)
        }
        
        return summary
    
    def _analyze_by_data_length(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by data length."""
        analysis = {}
        
        for length in self.data_lengths:
            mae_values = []
            for result_dict in [results["classical_results"], results["ml_results"]]:
                for result in result_dict.values():
                    length_results = [r for r in result["test_results"] if r["data_length"] == length]
                    if length_results:
                        mae_values.extend([r["mae"] for r in length_results])
            
            if mae_values:
                analysis[f"length_{length}"] = {
                    "mean_mae": np.mean(mae_values),
                    "std_mae": np.std(mae_values),
                    "n_tests": len(mae_values)
                }
        
        return analysis
    
    def _analyze_by_hurst_value(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by Hurst value."""
        analysis = {}
        
        for hurst in self.hurst_values:
            mae_values = []
            for result_dict in [results["classical_results"], results["ml_results"]]:
                for result in result_dict.values():
                    hurst_results = [r for r in result["test_results"] if r["true_hurst"] == hurst]
                    if hurst_results:
                        mae_values.extend([r["mae"] for r in hurst_results])
            
            if mae_values:
                analysis[f"hurst_{hurst}"] = {
                    "mean_mae": np.mean(mae_values),
                    "std_mae": np.std(mae_values),
                    "n_tests": len(mae_values)
                }
        
        return analysis
    
    def save_results(self, results: Dict[str, Any]):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = self.output_dir / f"robust_comprehensive_benchmark_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV summary
        csv_path = self.output_dir / f"robust_comprehensive_benchmark_{timestamp}.csv"
        self.save_csv_summary(results, csv_path)
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
    
    def save_csv_summary(self, results: Dict[str, Any], csv_path: Path):
        """Save summary results to CSV format."""
        rows = []
        
        # Classical results
        for name, result in results["classical_results"].items():
            summary = result["summary"]
            rows.append({
                "estimator_type": "classical",
                "estimator_name": name,
                "mean_mae": summary["mean_mae"],
                "std_mae": summary["std_mae"],
                "success_rate": summary["success_rate"],
                "mean_execution_time": summary["mean_execution_time"],
                "total_tests": summary["total_tests"],
                "successful_tests": summary["successful_tests"],
                "timeout_count": summary["timeout_count"],
                "error_count": summary["error_count"]
            })
        
        # ML results
        for name, result in results["ml_results"].items():
            summary = result["summary"]
            rows.append({
                "estimator_type": "ml",
                "estimator_name": name,
                "mean_mae": summary["mean_mae"],
                "std_mae": summary["std_mae"],
                "success_rate": summary["success_rate"],
                "mean_execution_time": summary["mean_execution_time"],
                "total_tests": summary["total_tests"],
                "successful_tests": summary["successful_tests"],
                "timeout_count": summary["timeout_count"],
                "error_count": summary["error_count"]
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

def main():
    """Run the robust comprehensive benchmark."""
    benchmark = RobustComprehensiveBenchmark()
    results = benchmark.run_benchmark()
    
    # Print summary
    print("\n" + "="*80)
    print("ROBUST COMPREHENSIVE BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\n📊 Overall Results:")
    print(f"  Total Estimators: {results['metadata']['n_classical'] + results['metadata']['n_ml']}")
    print(f"  Classical: {results['metadata']['n_classical']}")
    print(f"  ML: {results['metadata']['n_ml']}")
    print(f"  Total Tests: {results['summary']['overall']['total_tests']}")
    print(f"  Successful Tests: {results['summary']['overall']['total_successful_tests']}")
    print(f"  Overall Success Rate: {results['summary']['overall']['mean_success_rate']:.2%}")
    print(f"  Overall Mean MAE: {results['summary']['overall']['mean_mae']:.4f}")
    
    print(f"\n🏆 Top 3 Performers (by accuracy):")
    for i, estimator in enumerate(results['summary']['performance_ranking']['top_3'], 1):
        print(f"  {i}. {estimator['name']} ({estimator['type']}): MAE={estimator['mean_mae']:.4f}")
    
    print(f"\n📈 Classical vs ML:")
    if 'classical' in results['summary'] and 'ml' in results['summary']:
        classical_mae = results['summary']['classical']['mean_mae']
        ml_mae = results['summary']['ml']['mean_mae']
        print(f"  Classical Mean MAE: {classical_mae:.4f}")
        print(f"  ML Mean MAE: {ml_mae:.4f}")
        print(f"  Difference: {abs(classical_mae - ml_mae):.4f}")
    
    print(f"\n✅ Benchmark completed! Results saved to robust_comprehensive_results/")

if __name__ == "__main__":
    main()
