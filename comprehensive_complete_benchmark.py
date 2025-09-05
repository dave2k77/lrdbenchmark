#!/usr/bin/env python3
"""
Comprehensive Complete Benchmark for LRDBenchmark

This script runs a benchmark with ALL properly implemented estimators including:
- Classical temporal methods (DFA, RS, DMA, Higuchi)
- Classical spectral methods (Whittle, GPH, Periodogram)
- Wavelet methods (CWT, Wavelet Variance, Wavelet Log Variance, Wavelet Whittle)
- Multifractal methods (MFDFA, Wavelet Leaders)
- Machine learning methods (Random Forest, SVR, Gradient Boosting)
- Neural network methods (CNN only - excluding problematic LSTM, GRU, Transformer)
"""

import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import data models
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbenchmark.models.data_models.arfima.arfima_model import ARFIMAModel
from lrdbenchmark.models.data_models.mrw.mrw_model import MultifractalRandomWalk

# Import classical temporal estimators
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dma.dma_estimator_unified import DMAEstimator
from lrdbenchmark.analysis.temporal.higuchi.higuchi_estimator_unified import HiguchiEstimator

# Import classical spectral estimators
from lrdbenchmark.analysis.spectral.gph.gph_estimator_unified import GPHEstimator
from lrdbenchmark.analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator
from lrdbenchmark.analysis.spectral.periodogram.periodogram_estimator_unified import PeriodogramEstimator

# Import wavelet estimators
from lrdbenchmark.analysis.wavelet.cwt.cwt_estimator_unified import CWTEstimator
from lrdbenchmark.analysis.wavelet.variance.variance_estimator_unified import WaveletVarianceEstimator
from lrdbenchmark.analysis.wavelet.log_variance.log_variance_estimator_unified import WaveletLogVarianceEstimator
from lrdbenchmark.analysis.wavelet.whittle.whittle_estimator_unified import WaveletWhittleEstimator

# Import multifractal estimators
from lrdbenchmark.analysis.multifractal.mfdfa.mfdfa_estimator_unified import MFDFAEstimator
from lrdbenchmark.analysis.multifractal.wavelet_leaders.wavelet_leaders_estimator_unified import MultifractalWaveletLeadersEstimator

# Import ML estimators
from lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified import RandomForestEstimator
from lrdbenchmark.analysis.machine_learning.svr_estimator_unified import SVREstimator
from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator_unified import GradientBoostingEstimator

# Import neural network estimators (only working ones)
from lrdbenchmark.analysis.machine_learning.cnn_estimator_unified import CNNEstimator

# Import contamination models
from lrdbenchmark.models.contamination.contamination_models import ContaminationModel, ContaminationType


class ComprehensiveCompleteBenchmark:
    """
    Comprehensive benchmark including ALL properly implemented estimators.
    """
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        
        # Define ALL working estimators
        self.estimators = {
            # Classical - Temporal (4)
            'Classical_DFA': DFAEstimator(),
            'Classical_RS': RSEstimator(),
            'Classical_DMA': DMAEstimator(),
            'Classical_Higuchi': HiguchiEstimator(),
            
            # Classical - Spectral (3)
            'Classical_Whittle': WhittleEstimator(),
            'Classical_GPH': GPHEstimator(),
            'Classical_Periodogram': PeriodogramEstimator(),
            
            # Classical - Wavelet (4)
            'Classical_CWT': CWTEstimator(),
            'Classical_WaveletVariance': WaveletVarianceEstimator(),
            'Classical_WaveletLogVariance': WaveletLogVarianceEstimator(),
            'Classical_WaveletWhittle': WaveletWhittleEstimator(),
            
            # Classical - Multifractal (2)
            'Classical_MFDFA': MFDFAEstimator(),
            'Classical_WaveletLeaders': MultifractalWaveletLeadersEstimator(),
            
            # Machine Learning (3)
            'ML_RandomForest': RandomForestEstimator(),
            'ML_SVR': SVREstimator(),
            'ML_GradientBoosting': GradientBoostingEstimator(),
            
            # Neural Networks (1 - only working CNN)
            'Neural_CNN': CNNEstimator(),
        }
        
        # Define data models
        self.data_models = {
            'FBM': FractionalBrownianMotion,
            'FGN': FractionalGaussianNoise,
            'ARFIMA': ARFIMAModel,
            'MRW': MultifractalRandomWalk,
        }
        
        # Define contamination model
        self.contamination_model = ContaminationModel()
        
        # Test parameters
        self.hurst_values = [0.3, 0.5, 0.7, 0.9]
        self.data_lengths = [1000, 2000]
        self.contamination_levels = [0.0, 0.1, 0.2]
        
        print(f"Initialized comprehensive benchmark with {len(self.estimators)} estimators")
        print(f"Estimators: {list(self.estimators.keys())}")

    def generate_data(self, model_name: str, hurst: float, length: int) -> np.ndarray:
        """Generate data using the specified model."""
        try:
            if model_name == 'FBM':
                model = self.data_models[model_name](H=hurst, sigma=1.0)
                data = model.generate(length)
            elif model_name == 'FGN':
                model = self.data_models[model_name](H=hurst, sigma=1.0)
                data = model.generate(length)
            elif model_name == 'ARFIMA':
                # Convert Hurst to fractional differencing parameter d = H - 0.5
                d = hurst - 0.5
                model = self.data_models[model_name](d=d, ar_params=[0.3], ma_params=[0.1])
                data = model.generate(length)
            elif model_name == 'MRW':
                model = self.data_models[model_name](H=hurst, lambda_param=0.5)
                data = model.generate(n=length)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            return data
        except Exception as e:
            print(f"Error generating {model_name} data: {e}")
            return None

    def apply_contamination(self, data: np.ndarray, contamination_level: float) -> np.ndarray:
        """Apply contamination to the data."""
        if contamination_level == 0.0:
            return data
        
        try:
            # Use additive Gaussian noise contamination
            contaminated_data = self.contamination_model.apply_contamination(
                data, [ContaminationType.NOISE_GAUSSIAN], 
                noise_gaussian_std=contamination_level
            )
            return contaminated_data
        except Exception as e:
            print(f"Error applying contamination: {e}")
            # Fallback: simple additive noise
            noise = np.random.normal(0, contamination_level, len(data))
            return data + noise

    def run_single_test(self, estimator_name: str, model_name: str, 
                       hurst: float, length: int, contamination_level: float) -> Dict[str, Any]:
        """Run a single test case."""
        start_time = time.time()
        
        try:
            # Generate data
            data = self.generate_data(model_name, hurst, length)
            if data is None:
                return None
            
            # Apply contamination
            contaminated_data = self.apply_contamination(data, contamination_level)
            
            # Run estimation
            estimator = self.estimators[estimator_name]
            results = estimator.estimate(contaminated_data)
            
            # Extract Hurst estimate
            hurst_estimate = results.get('hurst', results.get('H', results.get('d', None)))
            if hurst_estimate is None:
                # Try to extract from nested results
                if 'results' in results:
                    hurst_estimate = results['results'].get('hurst', results['results'].get('H', None))
            
            if hurst_estimate is None:
                print(f"Warning: Could not extract Hurst estimate from {estimator_name}")
                return None
            
            # Calculate error
            error = abs(hurst_estimate - hurst)
            relative_error = error / hurst if hurst > 0 else 0
            
            execution_time = time.time() - start_time
            
            return {
                'model_name': model_name,
                'true_hurst': hurst,
                'data_length': length,
                'contamination_level': contamination_level,
                'estimator_name': estimator_name,
                'estimator_category': estimator_name.split('_')[0],
                'success': True,
                'estimated_hurst': hurst_estimate,
                'hurst_error': error,
                'relative_error': relative_error,
                'execution_time': execution_time,
                'optimization_framework': getattr(estimator, 'optimization_framework', 'numpy'),
                'error_message': ''
            }
            
        except Exception as e:
            print(f"Error in {estimator_name} with {model_name}: {e}")
            return {
                'model_name': model_name,
                'true_hurst': hurst,
                'data_length': length,
                'contamination_level': contamination_level,
                'estimator_name': estimator_name,
                'estimator_category': estimator_name.split('_')[0],
                'success': False,
                'estimated_hurst': None,
                'hurst_error': None,
                'relative_error': None,
                'execution_time': time.time() - start_time,
                'optimization_framework': 'numpy',
                'error_message': str(e)
            }

    def run_benchmark(self):
        """Run the comprehensive benchmark."""
        print("Starting comprehensive complete benchmark...")
        total_tests = len(self.estimators) * len(self.data_models) * len(self.hurst_values) * len(self.data_lengths) * len(self.contamination_levels)
        print(f"Total test cases: {total_tests}")
        
        test_count = 0
        successful_tests = 0
        
        for estimator_name in self.estimators.keys():
            print(f"\nTesting {estimator_name}...")
            
            for model_name in self.data_models.keys():
                for hurst in self.hurst_values:
                    for length in self.data_lengths:
                        for contamination_level in self.contamination_levels:
                            test_count += 1
                            
                            result = self.run_single_test(
                                estimator_name, model_name, hurst, length, contamination_level
                            )
                            
                            if result:
                                self.results.append(result)
                                if result['success']:
                                    successful_tests += 1
                            
                            if test_count % 100 == 0:
                                print(f"Completed {test_count} tests, {successful_tests} successful")
        
        print(f"\nBenchmark completed!")
        print(f"Total tests: {test_count}")
        print(f"Successful tests: {successful_tests}")
        print(f"Success rate: {successful_tests/test_count*100:.1f}%")

    def save_results(self, filename_prefix: str = None):
        """Save results to CSV and JSON files."""
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"comprehensive_complete_benchmark_{timestamp}"
        
        # Save CSV
        df = pd.DataFrame(self.results)
        csv_filename = f"{filename_prefix}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")
        
        # Save summary JSON
        summary = {
            "timestamp": self.start_time.isoformat(),
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r['success']),
            "success_rate": sum(1 for r in self.results if r['success']) / len(self.results) if self.results else 0,
            "estimators_tested": list(self.estimators.keys()),
            "estimator_categories": list(set([name.split('_')[0] for name in self.estimators.keys()])),
            "models_tested": list(self.data_models.keys()),
            "data_lengths": self.data_lengths,
            "hurst_values": self.hurst_values,
            "contamination_levels": self.contamination_levels
        }
        
        json_filename = f"{filename_prefix}_summary.json"
        with open(json_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {json_filename}")
        
        return csv_filename, json_filename


def main():
    """Main function to run the benchmark."""
    benchmark = ComprehensiveCompleteBenchmark()
    benchmark.run_benchmark()
    benchmark.save_results()


if __name__ == "__main__":
    main()
