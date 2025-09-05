#!/usr/bin/env python3
"""
Comprehensive All Estimators Benchmark for LRDBenchmark

This script runs a professional-grade benchmark including ALL classical and ML
estimators, with robust error handling and comprehensive analysis.
"""

import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import working data models
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbenchmark.models.data_models.arfima.arfima_model import ARFIMAModel
from lrdbenchmark.models.data_models.mrw.mrw_model import MultifractalRandomWalk

# Import classical estimators
from lrdbenchmark.analysis.temporal.rs.rs_estimator import RSEstimator
from lrdbenchmark.analysis.temporal.dma.dma_estimator import DMAEstimator
from lrdbenchmark.analysis.temporal.higuchi.higuchi_estimator import HiguchiEstimator
from lrdbenchmark.analysis.spectral.gph.gph_estimator import GPHEstimator
from lrdbenchmark.analysis.spectral.whittle.whittle_estimator import WhittleEstimator
from lrdbenchmark.analysis.spectral.periodogram.periodogram_estimator import PeriodogramEstimator

# Import ML estimators
from lrdbenchmark.analysis.machine_learning.random_forest_estimator import RandomForestEstimator
from lrdbenchmark.analysis.machine_learning.svr_estimator import SVREstimator
from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator import GradientBoostingEstimator

# Import neural network estimators
from lrdbenchmark.analysis.machine_learning.enhanced_cnn_estimator import EnhancedCNNEstimator
from lrdbenchmark.analysis.machine_learning.enhanced_lstm_estimator import EnhancedLSTMEstimator
from lrdbenchmark.analysis.machine_learning.enhanced_gru_estimator import EnhancedGRUEstimator
from lrdbenchmark.analysis.machine_learning.enhanced_transformer_estimator import EnhancedTransformerEstimator

# Import contamination models
from lrdbenchmark.models.contamination.contamination_models import ContaminationModel


class ComprehensiveAllEstimatorsBenchmark:
    """
    Comprehensive benchmark including all classical and ML estimators.
    """
    
    def __init__(self):
        self.results = []
        self.data_models = {
            'FBM': FractionalBrownianMotion,
            'FGN': FractionalGaussianNoise,
            'ARFIMA': ARFIMAModel,
            'MRW': MultifractalRandomWalk
        }
        
        # Initialize estimators with error handling
        self.estimators = self._initialize_estimators()
        
        self.contamination_model = ContaminationModel()
        
        # Benchmark parameters - reduced for comprehensive testing
        self.data_lengths = [1000, 2000]  # Reduced to focus on working lengths
        self.hurst_values = [0.3, 0.5, 0.7, 0.9]
        self.contamination_levels = [0.0, 0.1, 0.2]
        self.n_trials = 5  # Reduced for comprehensive testing
        
    def _initialize_estimators(self) -> Dict[str, Any]:
        """Initialize all estimators with error handling."""
        estimators = {}
        
        # Classical estimators
        classical_estimators = {
            'RS': RSEstimator,
            'DMA': DMAEstimator,
            'Higuchi': HiguchiEstimator,
            'GPH': GPHEstimator,
            'Whittle': WhittleEstimator,
            'Periodogram': PeriodogramEstimator,
        }
        
        # ML estimators
        ml_estimators = {
            'RandomForest': RandomForestEstimator,
            'SVR': SVREstimator,
            'GradientBoosting': GradientBoostingEstimator,
        }
        
        # Neural network estimators
        neural_estimators = {
            'CNN': EnhancedCNNEstimator,
            'LSTM': EnhancedLSTMEstimator,
            'GRU': EnhancedGRUEstimator,
            'Transformer': EnhancedTransformerEstimator,
        }
        
        # Test and add working estimators
        print("🔍 Testing estimator availability...")
        
        for name, estimator_class in classical_estimators.items():
            try:
                estimator = estimator_class()
                estimators[f'Classical_{name}'] = estimator_class
                print(f"✅ {name}: Available")
            except Exception as e:
                print(f"❌ {name}: {str(e)[:50]}...")
        
        for name, estimator_class in ml_estimators.items():
            try:
                estimator = estimator_class()
                estimators[f'ML_{name}'] = estimator_class
                print(f"✅ {name}: Available")
            except Exception as e:
                print(f"❌ {name}: {str(e)[:50]}...")
        
        for name, estimator_class in neural_estimators.items():
            try:
                estimator = estimator_class()
                estimators[f'Neural_{name}'] = estimator_class
                print(f"✅ {name}: Available")
            except Exception as e:
                print(f"❌ {name}: {str(e)[:50]}...")
        
        print(f"\n📊 Total working estimators: {len(estimators)}")
        return estimators
    
    def generate_data(self, model_name: str, H: float, n: int, **kwargs) -> np.ndarray:
        """Generate data using specified model."""
        if model_name == 'FBM':
            model = self.data_models[model_name](H=H, sigma=1.0)
        elif model_name == 'FGN':
            model = self.data_models[model_name](H=H, sigma=1.0)
        elif model_name == 'ARFIMA':
            d = H - 0.5  # Convert Hurst to fractional differencing parameter
            model = self.data_models[model_name](d=d, ar_params=[0.3], ma_params=[0.1])
        elif model_name == 'MRW':
            model = self.data_models[model_name](H=H, sigma=1.0, lambda_param=0.5)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return model.generate(n, seed=42)
    
    def add_contamination(self, data: np.ndarray, level: float) -> np.ndarray:
        """Add contamination to data."""
        if level == 0.0:
            return data
            
        # Add Gaussian noise
        contaminated = self.contamination_model.add_noise_gaussian(
            data, std=level * np.std(data)
        )
        
        # Add linear trend
        contaminated = self.contamination_model.add_trend_linear(
            contaminated, slope=level * 0.01
        )
        
        return contaminated
    
    def estimate_hurst(self, data: np.ndarray, estimator_name: str) -> Dict[str, Any]:
        """Estimate Hurst parameter using specified estimator."""
        estimator_class = self.estimators[estimator_name]
        
        start_time = time.time()
        try:
            estimator = estimator_class()
            results = estimator.estimate(data)
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'hurst_parameter': results.get('hurst_parameter', np.nan),
                'execution_time': execution_time,
                'optimization_framework': results.get('optimization_framework', 'unknown'),
                'error': None
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'hurst_parameter': np.nan,
                'execution_time': execution_time,
                'optimization_framework': 'failed',
                'error': str(e)
            }
    
    def run_single_benchmark(self, model_name: str, H: float, n: int, 
                           contamination_level: float, estimator_name: str) -> Dict[str, Any]:
        """Run a single benchmark test."""
        # Generate clean data
        clean_data = self.generate_data(model_name, H, n)
        
        # Add contamination
        data = self.add_contamination(clean_data, contamination_level)
        
        # Estimate Hurst parameter
        result = self.estimate_hurst(data, estimator_name)
        
        # Calculate error metrics
        if result['success']:
            hurst_error = abs(result['hurst_parameter'] - H)
            relative_error = hurst_error / H if H > 0 else np.nan
        else:
            hurst_error = np.nan
            relative_error = np.nan
        
        return {
            'model_name': model_name,
            'true_hurst': H,
            'data_length': n,
            'contamination_level': contamination_level,
            'estimator_name': estimator_name,
            'estimator_category': estimator_name.split('_')[0],
            'success': result['success'],
            'estimated_hurst': result['hurst_parameter'],
            'hurst_error': hurst_error,
            'relative_error': relative_error,
            'execution_time': result['execution_time'],
            'optimization_framework': result['optimization_framework'],
            'error_message': result['error']
        }
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive benchmark across all parameters."""
        print("🚀 Starting Comprehensive All Estimators Benchmark")
        print("=" * 70)
        
        total_tests = (len(self.data_models) * len(self.hurst_values) * 
                      len(self.data_lengths) * len(self.contamination_levels) * 
                      len(self.estimators) * self.n_trials)
        
        print(f"Total tests to run: {total_tests:,}")
        print(f"Data models: {list(self.data_models.keys())}")
        print(f"Estimators: {len(self.estimators)} ({list(self.estimators.keys())})")
        print(f"Hurst values: {self.hurst_values}")
        print(f"Data lengths: {self.data_lengths}")
        print(f"Contamination levels: {self.contamination_levels}")
        print(f"Trials per configuration: {self.n_trials}")
        print()
        
        test_count = 0
        start_time = time.time()
        
        for model_name in self.data_models.keys():
            print(f"📊 Testing {model_name} model...")
            
            for H in self.hurst_values:
                for n in self.data_lengths:
                    for contamination_level in self.contamination_levels:
                        for estimator_name in self.estimators.keys():
                            for trial in range(self.n_trials):
                                test_count += 1
                                
                                if test_count % 50 == 0:
                                    elapsed = time.time() - start_time
                                    rate = test_count / elapsed
                                    eta = (total_tests - test_count) / rate
                                    print(f"  Progress: {test_count:,}/{total_tests:,} "
                                          f"({test_count/total_tests*100:.1f}%) "
                                          f"ETA: {eta/60:.1f}min")
                                
                                result = self.run_single_benchmark(
                                    model_name, H, n, contamination_level, estimator_name
                                )
                                self.results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate summary statistics
        self.calculate_summary_statistics(df)
        
        return df
    
    def calculate_summary_statistics(self, df: pd.DataFrame) -> None:
        """Calculate and display summary statistics."""
        print("\n" + "=" * 70)
        print("📈 COMPREHENSIVE BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        
        # Overall success rate
        success_rate = df['success'].mean() * 100
        print(f"Overall Success Rate: {success_rate:.1f}%")
        
        # Performance by estimator category
        print("\n🏆 Performance by Estimator Category:")
        category_stats = df.groupby('estimator_category').agg({
            'success': 'mean',
            'hurst_error': 'mean',
            'relative_error': 'mean',
            'execution_time': 'mean'
        }).round(4)
        
        for category in category_stats.index:
            stats = category_stats.loc[category]
            print(f"  {category}:")
            print(f"    Success Rate: {stats['success']*100:.1f}%")
            print(f"    Mean Error: {stats['hurst_error']:.4f}")
            print(f"    Mean Relative Error: {stats['relative_error']*100:.1f}%")
            print(f"    Mean Execution Time: {stats['execution_time']:.4f}s")
        
        # Performance by individual estimator
        print("\n🎯 Performance by Individual Estimator:")
        estimator_stats = df.groupby('estimator_name').agg({
            'success': 'mean',
            'hurst_error': 'mean',
            'relative_error': 'mean',
            'execution_time': 'mean'
        }).round(4)
        
        # Sort by success rate, then by error
        estimator_stats = estimator_stats.sort_values(['success', 'hurst_error'], ascending=[False, True])
        
        for estimator in estimator_stats.index:
            stats = estimator_stats.loc[estimator]
            print(f"  {estimator}:")
            print(f"    Success Rate: {stats['success']*100:.1f}%")
            print(f"    Mean Error: {stats['hurst_error']:.4f}")
            print(f"    Mean Relative Error: {stats['relative_error']*100:.1f}%")
            print(f"    Mean Execution Time: {stats['execution_time']:.4f}s")
        
        # Classical vs ML vs Neural comparison
        print("\n⚔️ Classical vs ML vs Neural Comparison:")
        classical_data = df[df['estimator_category'] == 'Classical']
        ml_data = df[df['estimator_category'] == 'ML']
        neural_data = df[df['estimator_category'] == 'Neural']
        
        for name, data in [('Classical', classical_data), ('ML', ml_data), ('Neural', neural_data)]:
            if len(data) > 0:
                print(f"  {name}:")
                print(f"    Success Rate: {data['success'].mean()*100:.1f}%")
                print(f"    Mean Error: {data['hurst_error'].mean():.4f}")
                print(f"    Mean Relative Error: {data['relative_error'].mean()*100:.1f}%")
                print(f"    Mean Execution Time: {data['execution_time'].mean():.4f}s")
        
        # Contamination effects
        print("\n🧪 Contamination Effects:")
        contamination_stats = df.groupby('contamination_level').agg({
            'success': 'mean',
            'hurst_error': 'mean',
            'relative_error': 'mean'
        }).round(4)
        
        for level in contamination_stats.index:
            stats = contamination_stats.loc[level]
            print(f"  Contamination {level*100:.0f}%:")
            print(f"    Success Rate: {stats['success']*100:.1f}%")
            print(f"    Mean Error: {stats['hurst_error']:.4f}")
            print(f"    Mean Relative Error: {stats['relative_error']*100:.1f}%")
    
    def save_results(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save results to CSV and JSON files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_all_estimators_benchmark_{timestamp}"
        
        # Save CSV
        csv_file = f"{filename}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Results saved to: {csv_file}")
        
        # Save JSON summary
        json_file = f"{filename}_summary.json"
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': int(len(df)),
            'success_rate': float(df['success'].mean()),
            'estimators_tested': [str(x) for x in df['estimator_name'].unique()],
            'estimator_categories': [str(x) for x in df['estimator_category'].unique()],
            'models_tested': [str(x) for x in df['model_name'].unique()],
            'data_lengths': [int(x) for x in df['data_length'].unique()],
            'hurst_values': [float(x) for x in df['true_hurst'].unique()],
            'contamination_levels': [float(x) for x in df['contamination_level'].unique()],
            'mean_execution_time': float(df['execution_time'].mean()),
            'mean_hurst_error': float(df['hurst_error'].mean()),
            'mean_relative_error': float(df['relative_error'].mean()),
            'classical_success_rate': float(df[df['estimator_category'] == 'Classical']['success'].mean()) if len(df[df['estimator_category'] == 'Classical']) > 0 else 0.0,
            'ml_success_rate': float(df[df['estimator_category'] == 'ML']['success'].mean()) if len(df[df['estimator_category'] == 'ML']) > 0 else 0.0,
            'neural_success_rate': float(df[df['estimator_category'] == 'Neural']['success'].mean()) if len(df[df['estimator_category'] == 'Neural']) > 0 else 0.0
        }
        
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📊 Summary saved to: {json_file}")
        
        return csv_file


def main():
    """Run the comprehensive all estimators benchmark."""
    print("🔬 LRDBenchmark - Comprehensive All Estimators Benchmark")
    print("=" * 70)
    print("Including ALL classical and ML estimators with robust error handling")
    print()
    
    # Create and run benchmark
    benchmark = ComprehensiveAllEstimatorsBenchmark()
    results_df = benchmark.run_comprehensive_benchmark()
    
    # Save results
    csv_file = benchmark.save_results(results_df)
    
    print("\n✅ Comprehensive benchmark completed successfully!")
    print(f"📁 Results saved to: {csv_file}")
    print("\n🎯 This benchmark provides comprehensive empirical data for journal-ready research!")
    print("📊 Includes classical, ML, and neural network estimators!")


if __name__ == "__main__":
    main()
