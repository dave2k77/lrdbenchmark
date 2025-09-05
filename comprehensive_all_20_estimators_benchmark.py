#!/usr/bin/env python3
"""
Comprehensive All 20 Estimators Benchmark for LRDBenchmark

This script runs a complete benchmark including ALL 20 estimators:
- 13 Classical estimators (temporal, spectral, wavelet, multifractal)
- 3 Machine Learning estimators  
- 4 Neural Network estimators
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

# Import classical estimators - Temporal
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dma.dma_estimator_unified import DMAEstimator
from lrdbenchmark.analysis.temporal.higuchi.higuchi_estimator_unified import HiguchiEstimator
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator

# Import classical estimators - Spectral
from lrdbenchmark.analysis.spectral.gph.gph_estimator_unified import GPHEstimator
from lrdbenchmark.analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator
from lrdbenchmark.analysis.spectral.periodogram.periodogram_estimator_unified import PeriodogramEstimator

# Import classical estimators - Wavelet
from lrdbenchmark.analysis.wavelet.cwt.cwt_estimator_unified import CWTEstimator
from lrdbenchmark.analysis.wavelet.variance.variance_estimator_unified import WaveletVarianceEstimator
from lrdbenchmark.analysis.wavelet.log_variance.log_variance_estimator_unified import WaveletLogVarianceEstimator
from lrdbenchmark.analysis.wavelet.whittle.whittle_estimator_unified import WaveletWhittleEstimator

# Import classical estimators - Multifractal
from lrdbenchmark.analysis.multifractal.mfdfa.mfdfa_estimator_unified import MFDFAEstimator
from lrdbenchmark.analysis.multifractal.wavelet_leaders.wavelet_leaders_estimator_unified import WaveletLeadersEstimator

# Import ML estimators
from lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified import RandomForestEstimator
from lrdbenchmark.analysis.machine_learning.svr_estimator_unified import SVREstimator
from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator_unified import GradientBoostingEstimator

# Import neural network estimators
from lrdbenchmark.analysis.machine_learning.cnn_estimator_unified import CNNEstimator
from lrdbenchmark.analysis.machine_learning.lstm_estimator_unified import LSTMEstimator
from lrdbenchmark.analysis.machine_learning.gru_estimator_unified import GRUEstimator
from lrdbenchmark.analysis.machine_learning.transformer_estimator_unified import TransformerEstimator

# Import contamination models
from lrdbenchmark.models.contamination.contamination_models import ContaminationModel


class ComprehensiveAll20EstimatorsBenchmark:
    """
    Comprehensive benchmark including all 20 estimators.
    """
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        
        # Define all estimators
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
            'Classical_WaveletLeaders': WaveletLeadersEstimator(),
            
            # Machine Learning (3)
            'ML_RandomForest': RandomForestEstimator(),
            'ML_SVR': SVREstimator(),
            'ML_GradientBoosting': GradientBoostingEstimator(),
            
            # Neural Networks (4)
            'Neural_CNN': CNNEstimator(),
            'Neural_LSTM': LSTMEstimator(),
            'Neural_GRU': GRUEstimator(),
            'Neural_Transformer': TransformerEstimator(),
        }
        
        # Define data models
        self.data_models = {
            'FBM': FractionalBrownianMotion,
            'FGN': FractionalGaussianNoise,
            'ARFIMA': ARFIMAModel,
            'MRW': MultifractalRandomWalk
        }
        
        # Define experimental parameters
        self.hurst_values = [0.3, 0.5, 0.7, 0.9]
        self.data_lengths = [1000, 2000]
        self.contamination_levels = [0.0, 0.1, 0.2]
        self.replications = 5
        
        print(f"Initialized benchmark with {len(self.estimators)} estimators")
        print(f"Estimators: {list(self.estimators.keys())}")
    
    def generate_data(self, model_name: str, hurst: float, length: int) -> np.ndarray:
        """Generate data using the specified model."""
        try:
            if model_name == 'MRW':
                model = self.data_models[model_name](hurst=hurst, length=length, lambda_param=0.5)
            else:
                model = self.data_models[model_name](hurst=hurst, length=length)
            
            data = model.generate()
            return data
        except Exception as e:
            print(f"Error generating {model_name} data: {e}")
            return None
    
    def apply_contamination(self, data: np.ndarray, level: float) -> np.ndarray:
        """Apply contamination to the data."""
        if level == 0.0:
            return data
        
        contamination_model = ContaminationModel()
        contaminated_data = contamination_model.add_noise(data, noise_level=level)
        return contaminated_data
    
    def run_single_test(self, model_name: str, hurst: float, length: int, 
                       contamination: float, estimator_name: str, estimator) -> Dict[str, Any]:
        """Run a single test case."""
        try:
            # Generate data
            data = self.generate_data(model_name, hurst, length)
            if data is None:
                return self._create_failed_result(model_name, hurst, length, contamination, estimator_name, "Data generation failed")
            
            # Apply contamination
            contaminated_data = self.apply_contamination(data, contamination)
            
            # Run estimator
            start_time = time.time()
            estimated_hurst = estimator.estimate(contaminated_data)
            execution_time = time.time() - start_time
            
            # Calculate errors
            hurst_error = abs(estimated_hurst - hurst)
            relative_error = hurst_error / hurst if hurst != 0 else float('inf')
            
            return {
                'model_name': model_name,
                'true_hurst': hurst,
                'data_length': length,
                'contamination_level': contamination,
                'estimator_name': estimator_name,
                'estimator_category': self._get_category(estimator_name),
                'success': True,
                'estimated_hurst': estimated_hurst,
                'hurst_error': hurst_error,
                'relative_error': relative_error,
                'execution_time': execution_time,
                'optimization_framework': 'unified',
                'error_message': None
            }
            
        except Exception as e:
            return self._create_failed_result(model_name, hurst, length, contamination, estimator_name, str(e))
    
    def _create_failed_result(self, model_name: str, hurst: float, length: int, 
                             contamination: float, estimator_name: str, error_msg: str) -> Dict[str, Any]:
        """Create a result entry for a failed test."""
        return {
            'model_name': model_name,
            'true_hurst': hurst,
            'data_length': length,
            'contamination_level': contamination,
            'estimator_name': estimator_name,
            'estimator_category': self._get_category(estimator_name),
            'success': False,
            'estimated_hurst': None,
            'hurst_error': None,
            'relative_error': None,
            'execution_time': None,
            'optimization_framework': 'unified',
            'error_message': error_msg
        }
    
    def _get_category(self, estimator_name: str) -> str:
        """Get the category of an estimator."""
        if estimator_name.startswith('Classical_'):
            return 'Classical'
        elif estimator_name.startswith('ML_'):
            return 'ML'
        elif estimator_name.startswith('Neural_'):
            return 'Neural'
        else:
            return 'Unknown'
    
    def run_benchmark(self):
        """Run the complete benchmark."""
        print("Starting comprehensive benchmark with all 20 estimators...")
        print(f"Total test cases: {len(self.data_models) * len(self.hurst_values) * len(self.data_lengths) * len(self.contamination_levels) * len(self.estimators) * self.replications}")
        
        total_tests = 0
        successful_tests = 0
        
        for model_name in self.data_models.keys():
            print(f"\nTesting {model_name} model...")
            
            for hurst in self.hurst_values:
                for length in self.data_lengths:
                    for contamination in self.contamination_levels:
                        for estimator_name, estimator in self.estimators.items():
                            for rep in range(self.replications):
                                total_tests += 1
                                
                                if total_tests % 100 == 0:
                                    print(f"Progress: {total_tests} tests completed...")
                                
                                result = self.run_single_test(
                                    model_name, hurst, length, contamination, 
                                    estimator_name, estimator
                                )
                                
                                self.results.append(result)
                                
                                if result['success']:
                                    successful_tests += 1
        
        print(f"\nBenchmark completed!")
        print(f"Total tests: {total_tests}")
        print(f"Successful tests: {successful_tests}")
        print(f"Success rate: {successful_tests/total_tests:.2%}")
    
    def save_results(self, filename_prefix: str = "comprehensive_all_20_estimators_benchmark"):
        """Save results to CSV and JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        df = pd.DataFrame(self.results)
        csv_file = f"{filename_prefix}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Detailed results saved to: {csv_file}")
        
        # Save summary statistics
        summary = self._calculate_summary_statistics(df)
        json_file = f"{filename_prefix}_{timestamp}_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary statistics saved to: {json_file}")
        
        return csv_file, json_file
    
    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics."""
        successful_results = df[df['success'] == True]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': int(len(df)),
            'successful_tests': int(len(successful_results)),
            'success_rate': float(len(successful_results) / len(df)) if len(df) > 0 else 0.0,
            'estimators_tested': [str(x) for x in df['estimator_name'].unique()],
            'estimator_categories': [str(x) for x in df['estimator_category'].unique()],
            'models_tested': [str(x) for x in df['model_name'].unique()],
            'data_lengths': [int(x) for x in df['data_length'].unique()],
            'hurst_values': [float(x) for x in df['true_hurst'].unique()],
            'contamination_levels': [float(x) for x in df['contamination_level'].unique()],
        }
        
        if len(successful_results) > 0:
            summary.update({
                'mean_execution_time': float(successful_results['execution_time'].mean()),
                'mean_hurst_error': float(successful_results['hurst_error'].mean()),
                'mean_relative_error': float(successful_results['relative_error'].mean()),
                'std_hurst_error': float(successful_results['hurst_error'].std()),
                'std_relative_error': float(successful_results['relative_error'].std()),
            })
            
            # Category-wise statistics
            for category in df['estimator_category'].unique():
                cat_data = successful_results[successful_results['estimator_category'] == category]
                if len(cat_data) > 0:
                    summary[f'{category.lower()}_mean_error'] = float(cat_data['hurst_error'].mean())
                    summary[f'{category.lower()}_std_error'] = float(cat_data['hurst_error'].std())
                    summary[f'{category.lower()}_mean_time'] = float(cat_data['execution_time'].mean())
                    summary[f'{category.lower()}_success_rate'] = float(len(cat_data) / len(df[df['estimator_category'] == category]))
        
        return summary


def main():
    """Main function to run the benchmark."""
    print("="*60)
    print("LRDBenchmark - Comprehensive All 20 Estimators Benchmark")
    print("="*60)
    
    # Create and run benchmark
    benchmark = ComprehensiveAll20EstimatorsBenchmark()
    
    try:
        benchmark.run_benchmark()
        csv_file, json_file = benchmark.save_results()
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {csv_file}")
        print(f"Summary saved to: {json_file}")
        
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
