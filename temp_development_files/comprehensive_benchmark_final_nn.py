#!/usr/bin/env python3
"""
Comprehensive Benchmark with All Neural Networks Working

This script runs a comprehensive benchmark comparing:
- Classical estimators (R/S, DFA, DMA, Higuchi, GPH, Whittle, Periodogram)
- Machine Learning estimators (SVR, Gradient Boosting, Random Forest)
- Neural Network estimators (CNN, LSTM, GRU, Transformer, ResNet, etc.)

On pure synthetic data (FBM, FGN) with all input shape issues resolved.
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
from lrdbenchmark.analysis.machine_learning.random_forest_estimator import RandomForestEstimator
from lrdbenchmark.analysis.machine_learning.svr_estimator import SVREstimator
from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator import GradientBoostingEstimator

# Import neural network factory
from lrdbenchmark.analysis.machine_learning.neural_network_factory import (
    NeuralNetworkFactory, NNArchitecture, NNConfig, create_all_benchmark_networks
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

class NeuralNetworkEstimator:
    """Wrapper for neural networks to handle input shape properly."""
    
    def __init__(self, network, input_length):
        self.network = network
        self.input_length = input_length
        self.is_trained = False
    
    def train_model(self, X, y, validation_split=0.2):
        """Train the neural network."""
        # Ensure all training data has the same length
        X_padded = []
        for x in X:
            if len(x) < self.input_length:
                # Pad with zeros
                x_padded = np.pad(x, (0, self.input_length - len(x)), 'constant')
            elif len(x) > self.input_length:
                # Truncate
                x_padded = x[:self.input_length]
            else:
                x_padded = x
            X_padded.append(x_padded)
        
        X_padded = np.array(X_padded)
        return self.network.train_model(X_padded, y, validation_split)
    
    def predict(self, x):
        """Make predictions on new data."""
        # Ensure input has correct length
        if len(x) < self.input_length:
            x_padded = np.pad(x, (0, self.input_length - len(x)), 'constant')
        elif len(x) > self.input_length:
            x_padded = x[:self.input_length]
        else:
            x_padded = x
        
        return self.network.predict(x_padded.reshape(1, -1))

class ComprehensiveBenchmarkFinalNN:
    """
    Comprehensive benchmark with all neural networks working for Hurst parameter estimation.
    """
    
    def __init__(self, output_dir: str = "comprehensive_final_nn_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test parameters (reduced for faster testing)
        self.hurst_values = [0.2, 0.4, 0.6, 0.8]
        self.data_lengths = [500, 1000]
        self.n_samples_per_condition = 3  # Reduced for faster testing
        
        # Initialize data models (will be created per test with specific Hurst parameter)
        self.fbm_model_class = FBMModel
        self.fgn_model_class = FGNModel
        
        # Initialize estimators
        self.estimators = self._initialize_estimators()
        
        # Results storage
        self.results = {}
        
        logger.info(f"Initialized comprehensive benchmark with {len(self.estimators)} estimators")
        logger.info(f"Test parameters: Hurst={self.hurst_values}, Lengths={self.data_lengths}, Samples={self.n_samples_per_condition}")
    
    def _initialize_estimators(self) -> Dict[str, Any]:
        """Initialize all estimators including neural networks with proper input handling."""
        estimators = {}
        
        # Classical estimators
        estimators.update({
            "R/S": RSEstimator(),
            "DFA": DFAEstimator(),
            "DMA": DMAEstimator(),
            "Higuchi": HiguchiEstimator(),
            "GPH": GPHEstimator(),
            "Whittle": WhittleEstimator(),
            "Periodogram": PeriodogramEstimator(),
        })
        
        # Machine Learning estimators
        estimators.update({
            "RandomForest": RandomForestEstimator(),
            "SVR": SVREstimator(),
            "GradientBoosting": GradientBoostingEstimator(),
        })
        
        # Neural Network estimators with proper input handling
        try:
            # Create neural networks for the largest data length
            max_length = max(self.data_lengths)
            
            # Create networks with proper configurations
            nn_configs = {
                "feedforward": NNConfig(
                    architecture=NNArchitecture.FFN,
                    input_length=max_length,
                    hidden_dims=[128, 64, 32],
                    dropout_rate=0.2,
                    learning_rate=0.001,
                    epochs=30
                ),
                "convolutional": NNConfig(
                    architecture=NNArchitecture.CNN,
                    input_length=max_length,
                    conv_filters=32,
                    hidden_dims=[64, 32],
                    dropout_rate=0.2,
                    learning_rate=0.001,
                    epochs=30
                ),
                "lstm": NNConfig(
                    architecture=NNArchitecture.LSTM,
                    input_length=max_length,
                    lstm_units=64,
                    hidden_dims=[32],
                    dropout_rate=0.1,  # Reduced dropout for single layer
                    learning_rate=0.001,
                    epochs=30
                ),
                "gru": NNConfig(
                    architecture=NNArchitecture.GRU,
                    input_length=max_length,
                    lstm_units=64,
                    hidden_dims=[32],
                    dropout_rate=0.1,  # Reduced dropout for single layer
                    learning_rate=0.001,
                    epochs=30
                ),
                "transformer": NNConfig(
                    architecture=NNArchitecture.TRANSFORMER,
                    input_length=max_length,
                    lstm_units=64,
                    transformer_heads=4,
                    transformer_layers=2,
                    hidden_dims=[32],
                    dropout_rate=0.2,
                    learning_rate=0.001,
                    epochs=30
                ),
                "resnet": NNConfig(
                    architecture=NNArchitecture.RESNET,
                    input_length=max_length,
                    resnet_blocks=2,
                    hidden_dims=[64, 32],
                    dropout_rate=0.2,
                    learning_rate=0.001,
                    epochs=30
                )
            }
            
            # Create and wrap neural networks
            for name, config in nn_configs.items():
                try:
                    network = NeuralNetworkFactory.create_network(config)
                    network.model_name = name
                    # Wrap with proper input handling
                    wrapped_network = NeuralNetworkEstimator(network, max_length)
                    estimators[f"NN_{name}"] = wrapped_network
                    logger.info(f"Successfully initialized NN_{name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize NN_{name}: {e}")
            
            logger.info(f"Successfully initialized {len([k for k in estimators.keys() if k.startswith('NN_')])} neural network architectures")
            
        except Exception as e:
            logger.warning(f"Failed to initialize neural networks: {e}")
            logger.info("Continuing with classical and ML estimators only")
        
        return estimators
    
    def _test_estimator(self, estimator: Any, name: str) -> Dict[str, Any]:
        """Test a single estimator on all conditions with timeout protection."""
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
        timeouts = 0
        
        logger.info(f"Testing {name}...")
        
        for hurst in self.hurst_values:
            for length in self.data_lengths:
                for sample_idx in range(self.n_samples_per_condition):
                    total_tests += 1
                    
                    try:
                        # Generate test data
                        if sample_idx % 2 == 0:
                            fbm_model = self.fbm_model_class(H=hurst)
                            data = fbm_model.generate(length)
                        else:
                            fgn_model = self.fgn_model_class(H=hurst)
                            data = fgn_model.generate(length)
                        
                        # Set timeout for this test (60 seconds for neural networks)
                        timeout_seconds = 60 if name.startswith("NN_") else 30
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(timeout_seconds)
                        
                        start_time = time.time()
                        
                        try:
                            # Test the estimator
                            if name.startswith("NN_"):
                                # Neural networks need training data first
                                if not estimator.is_trained:
                                    # Generate training data
                                    train_data = []
                                    train_labels = []
                                    for h in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                                        for _ in range(20):  # More training data
                                            if np.random.rand() < 0.5:
                                                fbm_model = self.fbm_model_class(H=h)
                                                train_data.append(fbm_model.generate(length))
                                            else:
                                                fgn_model = self.fgn_model_class(H=h)
                                                train_data.append(fgn_model.generate(length))
                                            train_labels.append(h)
                                    
                                    train_data = np.array(train_data)
                                    train_labels = np.array(train_labels)
                                    
                                    # Train the network
                                    estimator.train_model(train_data, train_labels)
                                    estimator.is_trained = True
                                
                                # Make prediction
                                result = estimator.predict(data)
                                hurst_estimate = float(result[0])
                                
                            else:
                                # Classical/ML estimators
                                result = estimator.estimate(data)
                                hurst_estimate = result["hurst_parameter"]
                            
                            execution_time = time.time() - start_time
                            signal.alarm(0)  # Cancel timeout
                            
                            # Calculate error
                            mae = abs(hurst_estimate - hurst)
                            
                            # Store result
                            test_result = {
                                "hurst_true": hurst,
                                "hurst_estimate": hurst_estimate,
                                "mae": mae,
                                "execution_time": execution_time,
                                "data_length": length,
                                "sample_idx": sample_idx,
                                "success": True
                            }
                            
                            results["test_results"].append(test_result)
                            successful_tests += 1
                            execution_times.append(execution_time)
                            mae_values.append(mae)
                            
                            if execution_time > 10:
                                logger.warning(f"  {name} took {execution_time:.2f}s for H={hurst}, L={length}")
                        
                        except TimeoutError:
                            signal.alarm(0)
                            execution_time = time.time() - start_time
                            logger.warning(f"  {name} timed out after {execution_time:.2f}s for H={hurst}, L={length}")
                            
                            test_result = {
                                "hurst_true": hurst,
                                "hurst_estimate": None,
                                "mae": float('inf'),
                                "execution_time": execution_time,
                                "data_length": length,
                                "sample_idx": sample_idx,
                                "success": False,
                                "error": "timeout"
                            }
                            
                            results["test_results"].append(test_result)
                            timeouts += 1
                            
                        except Exception as e:
                            signal.alarm(0)
                            execution_time = time.time() - start_time
                            error_msg = str(e)
                            logger.warning(f"  {name} failed: {error_msg}")
                            
                            test_result = {
                                "hurst_true": hurst,
                                "hurst_estimate": None,
                                "mae": float('inf'),
                                "execution_time": execution_time,
                                "data_length": length,
                                "sample_idx": sample_idx,
                                "success": False,
                                "error": error_msg
                            }
                            
                            results["test_results"].append(test_result)
                            errors.append(error_msg)
                    
                    except Exception as e:
                        logger.error(f"  {name} failed to generate data: {e}")
                        errors.append(f"data_generation: {e}")
        
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
                "timeouts": timeouts,
                "errors": errors[:10]  # Limit error messages
            }
        else:
            results["summary"] = {
                "success_rate": 0.0,
                "mean_mae": float('inf'),
                "std_mae": 0.0,
                "mean_execution_time": float('inf'),
                "std_execution_time": 0.0,
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "timeouts": timeouts,
                "errors": errors[:10]
            }
        
        logger.info(f"  {name} completed: {successful_tests}/{total_tests} successful, "
                   f"MAE={results['summary']['mean_mae']:.3f}, "
                   f"Time={results['summary']['mean_execution_time']:.2f}s")
        
        return results
    
    def run_benchmark(self):
        """Run the comprehensive benchmark."""
        logger.info("Starting comprehensive benchmark with all neural networks...")
        start_time = time.time()
        
        # Test all estimators
        for name, estimator in self.estimators.items():
            try:
                result = self._test_estimator(estimator, name)
                self.results[name] = result
            except Exception as e:
                logger.error(f"Failed to test {name}: {e}")
                self.results[name] = {
                    "estimator_name": name,
                    "test_results": [],
                    "summary": {"success_rate": 0.0, "error": str(e)}
                }
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        self._save_results()
        
        total_time = time.time() - start_time
        logger.info(f"Benchmark completed in {total_time:.2f} seconds")
    
    def _generate_summary(self):
        """Generate summary statistics."""
        summary_data = []
        
        for name, result in self.results.items():
            summary = result["summary"]
            summary_data.append({
                "estimator": name,
                "success_rate": summary["success_rate"],
                "mean_mae": summary["mean_mae"],
                "std_mae": summary["std_mae"],
                "mean_execution_time": summary["mean_execution_time"],
                "std_execution_time": summary["std_execution_time"],
                "total_tests": summary["total_tests"],
                "successful_tests": summary["successful_tests"],
                "timeouts": summary.get("timeouts", 0),
                "error_count": len(summary.get("errors", []))
            })
        
        self.summary_df = pd.DataFrame(summary_data)
        
        # Sort by mean MAE
        self.summary_df = self.summary_df.sort_values("mean_mae")
        
        logger.info("\n" + "="*80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*80)
        logger.info(self.summary_df.to_string(index=False, float_format='%.4f'))
        logger.info("="*80)
    
    def _save_results(self):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"comprehensive_final_nn_benchmark_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.output_dir / f"comprehensive_final_nn_benchmark_{timestamp}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.summary_df.to_dict('records'), f, indent=2, default=str)
        
        # Save CSV
        csv_file = self.output_dir / f"comprehensive_final_nn_benchmark_{timestamp}.csv"
        self.summary_df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"  Detailed results: {results_file}")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  CSV: {csv_file}")

def main():
    """Run the comprehensive benchmark."""
    benchmark = ComprehensiveBenchmarkFinalNN()
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()
