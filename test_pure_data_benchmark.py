#!/usr/bin/env python3
"""
Test Pure Data Benchmark - Simplified version to test basic functionality.
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

class TestPureDataBenchmark:
    """Simplified benchmark for testing pure data only."""
    
    def __init__(self, output_dir: str = "test_pure_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data models
        self.fbm_model_class = FBMModel
        self.fgn_model_class = FGNModel
        
        # Initialize estimators (only working ones)
        self.classical_estimators = {
            "RS": RSEstimator(),
            "DFA": DFAEstimator(),
            "DMA": DMAEstimator(),
            "Higuchi": HiguchiEstimator(),
            "GPH": GPHEstimator(),
            "Whittle": WhittleEstimator(),
            "Periodogram": PeriodogramEstimator()
        }
        
        self.ml_estimators = {
            "SVR": SVREstimator(),
            "GradientBoosting": GradientBoostingEstimator(),
            "RandomForest": RandomForestEstimator()
        }
        
        # Test parameters (very small for quick testing)
        self.hurst_values = [0.3, 0.5, 0.7]
        self.data_lengths = [500]
        self.n_samples_per_condition = 2
        
        logger.info(f"Initialized test benchmark with {len(self.classical_estimators)} classical and {len(self.ml_estimators)} ML estimators")
    
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
    
    def test_estimator(self, estimator: Any, name: str) -> Dict[str, Any]:
        """Test a single estimator with timeout protection."""
        results = {
            "estimator_name": name,
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
        logger.info(f"  Testing {name} with {total_conditions} tests...")
        
        for hurst in self.hurst_values:
            for length in self.data_lengths:
                for sample_idx in range(self.n_samples_per_condition):
                    total_tests += 1
                    
                    try:
                        # Generate test data
                        data = self.generate_test_data(hurst, length, "fbm")
                        
                        # Test estimator with timeout protection
                        start_time = time.time()
                        
                        # Set timeout for individual estimator calls (10 seconds)
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(10)
                        
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
                            error_msg = f"Test timed out after 10s"
                            errors.append(error_msg)
                            logger.warning(f"    {error_msg}")
                            
                    except Exception as e:
                        signal.alarm(0)  # Cancel timeout
                        error_msg = f"Test failed: {str(e)}"
                        errors.append(error_msg)
                        logger.warning(f"    {error_msg}")
        
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
        
        logger.info(f"  {name} completed: {successful_tests}/{total_tests} successful, {timeout_count} timeouts")
        return results
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run simplified benchmark on pure data only."""
        logger.info("🚀 Starting test pure data benchmark")
        
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
            results["classical_results"][name] = self.test_estimator(estimator, name)
        
        # Test ML estimators
        logger.info("🤖 Testing ML Estimators...")
        for name, estimator in self.ml_estimators.items():
            logger.info(f"  Testing {name}...")
            results["ml_results"][name] = self.test_estimator(estimator, name)
        
        # Generate summary
        results["summary"] = self.generate_summary(results)
        
        # Save results
        self.save_results(results)
        
        logger.info("✅ Test benchmark completed!")
        return results
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "classical": {},
            "ml": {},
            "overall": {}
        }
        
        all_mae_values = []
        all_success_rates = []
        all_execution_times = []
        
        # Classical summary
        classical_mae_values = []
        classical_success_rates = []
        classical_execution_times = []
        
        for name, result in results["classical_results"].items():
            if result["summary"]["successful_tests"] > 0:
                classical_mae_values.append(result["summary"]["mean_mae"])
                classical_success_rates.append(result["summary"]["success_rate"])
                classical_execution_times.append(result["summary"]["mean_execution_time"])
        
        if classical_mae_values:
            summary["classical"] = {
                "mean_mae": np.mean(classical_mae_values),
                "std_mae": np.std(classical_mae_values),
                "mean_success_rate": np.mean(classical_success_rates),
                "mean_execution_time": np.mean(classical_execution_times),
                "n_estimators": len(classical_mae_values)
            }
            all_mae_values.extend(classical_mae_values)
            all_success_rates.extend(classical_success_rates)
            all_execution_times.extend(classical_execution_times)
        
        # ML summary
        ml_mae_values = []
        ml_success_rates = []
        ml_execution_times = []
        
        for name, result in results["ml_results"].items():
            if result["summary"]["successful_tests"] > 0:
                ml_mae_values.append(result["summary"]["mean_mae"])
                ml_success_rates.append(result["summary"]["success_rate"])
                ml_execution_times.append(result["summary"]["mean_execution_time"])
        
        if ml_mae_values:
            summary["ml"] = {
                "mean_mae": np.mean(ml_mae_values),
                "std_mae": np.std(ml_mae_values),
                "mean_success_rate": np.mean(ml_success_rates),
                "mean_execution_time": np.mean(ml_execution_times),
                "n_estimators": len(ml_mae_values)
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
                "total_estimators": len(all_mae_values)
            }
        
        return summary
    
    def save_results(self, results: Dict[str, Any]):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = self.output_dir / f"test_pure_benchmark_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {json_path}")

def main():
    """Run the test pure data benchmark."""
    benchmark = TestPureDataBenchmark()
    results = benchmark.run_benchmark()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST PURE DATA BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\n📊 Overall Results:")
    print(f"  Classical Estimators: {results['metadata']['n_classical']}")
    print(f"  ML Estimators: {results['metadata']['n_ml']}")
    
    if "overall" in results["summary"] and "mean_mae" in results["summary"]["overall"]:
        print(f"  Overall Mean MAE: {results['summary']['overall']['mean_mae']:.4f}")
        print(f"  Overall Success Rate: {results['summary']['overall']['mean_success_rate']:.2%}")
    
    print(f"\n✅ Test completed! Results saved to test_pure_results/")

if __name__ == "__main__":
    main()
