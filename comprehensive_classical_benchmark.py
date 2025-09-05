#!/usr/bin/env python3
"""
Comprehensive Classical Estimators Benchmark with Realistic Confounding

This script runs a comprehensive benchmark on classical LRD estimators using
the new adaptive optimization backend system and realistic confounding profiles
from the contamination factory. This provides a much more robust and realistic
evaluation of estimator performance under real-world conditions.
"""

import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our adaptive estimators and backend
from lrdbenchmark.analysis.optimization_backend import OptimizationBackend
from lrdbenchmark.analysis.adaptive_estimator import AdaptiveRS

# Import data models
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion as FBMModel
from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise as FGNModel
from lrdbenchmark.models.data_models.arfima.arfima_model import ARFIMAModel
from lrdbenchmark.models.data_models.mrw.mrw_model import MultifractalRandomWalk as MRWModel

# Import contamination factory
from lrdbenchmark.models.contamination.contamination_factory import (
    ContaminationFactory, 
    ConfoundingScenario
)


class ComprehensiveClassicalBenchmark:
    """
    Comprehensive benchmark for classical LRD estimators with realistic confounding.
    """
    
    def __init__(self, output_dir: str = "comprehensive_results"):
        """
        Initialize the benchmark.
        
        Parameters
        ----------
        output_dir : str
            Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize optimization backend
        self.backend = OptimizationBackend()
        
        # Initialize contamination factory
        self.contamination_factory = ContaminationFactory(random_seed=42)
        
        # Initialize estimators
        self.estimators = {
            'Adaptive_RS': AdaptiveRS(),
            # Add more adaptive estimators here as we create them
        }
        
        # Test parameters
        self.hurst_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.data_lengths = [500, 1000, 2000, 5000]
        self.num_trials = 3  # Number of trials per configuration
        
        # Realistic confounding scenarios
        self.confounding_scenarios = [
            ConfoundingScenario.MIXED_REALISTIC_LIGHT,
            ConfoundingScenario.MIXED_REALISTIC_MODERATE,
            ConfoundingScenario.MIXED_REALISTIC_SEVERE,
            ConfoundingScenario.FINANCIAL_CRASH,
            ConfoundingScenario.FINANCIAL_VOLATILITY_CLUSTERING,
            ConfoundingScenario.PHYSIOLOGICAL_SENSOR_DRIFT,
            ConfoundingScenario.PHYSIOLOGICAL_MOTION_ARTIFACTS,
            ConfoundingScenario.ENVIRONMENTAL_SEASONAL,
            ConfoundingScenario.ENVIRONMENTAL_EXTREME_EVENTS,
            ConfoundingScenario.NETWORK_BURSTS,
            ConfoundingScenario.INDUSTRIAL_CALIBRATION_DRIFT,
            # EEG scenarios
            ConfoundingScenario.EEG_OCULAR_ARTIFACTS,
            ConfoundingScenario.EEG_MUSCLE_ARTIFACTS,
            ConfoundingScenario.EEG_CARDIAC_ARTIFACTS,
            ConfoundingScenario.EEG_ELECTRODE_POPPING,
            ConfoundingScenario.EEG_ELECTRODE_DRIFT,
            ConfoundingScenario.EEG_60HZ_NOISE,
            ConfoundingScenario.EEG_SWEAT_ARTIFACTS,
            ConfoundingScenario.EEG_MOVEMENT_ARTIFACTS,
        ]
        
        # Results storage
        self.results = []
        self.performance_summary = {}
        
        print("Comprehensive Classical Benchmark initialized")
        print(f"Estimators: {list(self.estimators.keys())}")
        print(f"Confounding scenarios: {len(self.confounding_scenarios)}")
        print(f"Hardware: {self.backend.hardware_info.cpu_cores} cores, "
              f"{self.backend.hardware_info.memory_gb:.1f}GB RAM")
        print(f"GPU: {'Available' if self.backend.hardware_info.has_gpu else 'Not available'}")
    
    def generate_test_data(self, 
                          model_type: str, 
                          hurst: float, 
                          length: int) -> np.ndarray:
        """
        Generate test data for benchmarking.
        
        Parameters
        ----------
        model_type : str
            Type of data model ('fbm', 'fgn', 'arfima', 'mrw')
        hurst : float
            Hurst parameter value
        length : int
            Data length
            
        Returns
        -------
        np.ndarray
            Generated time series data
        """
        try:
            if model_type == 'fbm':
                model = FBMModel(H=hurst)
                data = model.generate(n=length)
            elif model_type == 'fgn':
                model = FGNModel(H=hurst)
                data = model.generate(n=length)
            elif model_type == 'arfima':
                d = hurst - 0.5  # Convert Hurst to fractional differencing parameter
                model = ARFIMAModel(d=d)
                data = model.generate(n=length)
            elif model_type == 'mrw':
                model = MRWModel(H=hurst, lambda_param=0.5)
                data = model.generate(n=length)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return data
            
        except Exception as e:
            warnings.warn(f"Failed to generate {model_type} data: {e}")
            # Fallback to simple random walk
            return np.cumsum(np.random.randn(length))
    
    def apply_confounding(self, 
                         data: np.ndarray, 
                         scenario: ConfoundingScenario,
                         intensity: float = 1.0) -> Tuple[np.ndarray, str]:
        """
        Apply realistic confounding to data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        scenario : ConfoundingScenario
            Confounding scenario to apply
        intensity : float
            Intensity of confounding (0.0 to 1.0)
            
        Returns
        -------
        tuple
            (contaminated_data, scenario_description)
        """
        try:
            # Create confounding profile
            profile = self.contamination_factory.create_confounding_profile(
                scenario=scenario,
                intensity=intensity
            )
            
            # Apply confounding
            contaminated_data = self.contamination_factory.apply_confounding(data, profile)
            
            return contaminated_data, profile.description
            
        except Exception as e:
            warnings.warn(f"Failed to apply confounding {scenario}: {e}")
            return data, f"Failed to apply {scenario.value}"
    
    def run_single_test(self, 
                       estimator_name: str, 
                       estimator, 
                       data: np.ndarray, 
                       true_hurst: float,
                       test_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test case.
        
        Parameters
        ----------
        estimator_name : str
            Name of the estimator
        estimator : AdaptiveEstimator
            Estimator instance
        data : np.ndarray
            Input data
        true_hurst : float
            True Hurst parameter
        test_config : dict
            Test configuration
            
        Returns
        -------
        dict
            Test results
        """
        start_time = time.time()
        
        try:
            # Run estimation
            result = estimator.estimate(data)
            execution_time = time.time() - start_time
            
            # Calculate error metrics
            estimated_hurst = result.get('hurst_parameter', np.nan)
            error = abs(estimated_hurst - true_hurst) if not np.isnan(estimated_hurst) else np.nan
            relative_error = error / true_hurst if true_hurst > 0 else np.nan
            
            # Determine success
            success = not np.isnan(estimated_hurst) and 0 < estimated_hurst < 1
            
            return {
                'estimator_name': estimator_name,
                'true_hurst': true_hurst,
                'estimated_hurst': estimated_hurst,
                'error': error,
                'relative_error': relative_error,
                'execution_time': execution_time,
                'success': success,
                'r_squared': result.get('r_squared', np.nan),
                'framework_used': result.get('framework_used', 'unknown'),
                'framework_reasoning': result.get('framework_reasoning', ''),
                'data_length': len(data),
                'data_model': test_config['data_model'],
                'confounding_scenario': test_config['confounding_scenario'],
                'confounding_intensity': test_config['confounding_intensity'],
                'confounding_description': test_config['confounding_description'],
                'trial': test_config['trial'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'estimator_name': estimator_name,
                'true_hurst': true_hurst,
                'estimated_hurst': np.nan,
                'error': np.nan,
                'relative_error': np.nan,
                'execution_time': execution_time,
                'success': False,
                'r_squared': np.nan,
                'framework_used': 'failed',
                'framework_reasoning': f'Error: {str(e)}',
                'data_length': len(data),
                'data_model': test_config['data_model'],
                'confounding_scenario': test_config['confounding_scenario'],
                'confounding_intensity': test_config['confounding_intensity'],
                'confounding_description': test_config['confounding_description'],
                'trial': test_config['trial'],
                'timestamp': datetime.now().isoformat()
            }
    
    def run_benchmark(self):
        """Run the complete benchmark."""
        print("Starting comprehensive classical benchmark...")
        
        total_tests = (len(self.hurst_values) * len(self.data_lengths) * 
                      len(self.confounding_scenarios) * self.num_trials * 
                      len(self.estimators) * 4)  # 4 data models
        
        print(f"Total test cases: {total_tests}")
        
        successful_tests = 0
        current_test = 0
        
        for hurst in self.hurst_values:
            for length in self.data_lengths:
                for data_model in ['fbm', 'fgn', 'arfima', 'mrw']:
                    for scenario in self.confounding_scenarios:
                        for trial in range(self.num_trials):
                            test_config = {
                                'data_model': data_model,
                                'confounding_scenario': scenario.value,
                                'confounding_intensity': 1.0,
                                'confounding_description': '',
                                'trial': trial
                            }
                            
                            # Generate test data
                            data = self.generate_test_data(data_model, hurst, length)
                            
                            # Apply confounding
                            contaminated_data, description = self.apply_confounding(
                                data, scenario, intensity=1.0
                            )
                            test_config['confounding_description'] = description
                            
                            # Test each estimator
                            for estimator_name, estimator in self.estimators.items():
                                result = self.run_single_test(
                                    estimator_name, estimator, contaminated_data, hurst, test_config
                                )
                                
                                self.results.append(result)
                                current_test += 1
                                
                                if result['success']:
                                    successful_tests += 1
                                
                                # Progress update
                                if current_test % 100 == 0:
                                    success_rate = (successful_tests / current_test) * 100
                                    print(f"Progress: {current_test}/{total_tests} tests completed, "
                                          f"success rate: {success_rate:.1f}%")
        
        print(f"Benchmark completed: {current_test} tests, "
              f"success rate: {(successful_tests/current_test)*100:.1f}%")
        
        # Save results
        self.save_results()
        
        # Generate performance summary
        self.generate_performance_summary()
    
    def save_results(self):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as CSV
        df = pd.DataFrame(self.results)
        csv_file = os.path.join(self.output_dir, f"comprehensive_classical_benchmark_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"Detailed results saved to: {csv_file}")
        
        # Save results as JSON
        json_file = os.path.join(self.output_dir, f"comprehensive_classical_benchmark_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to: {json_file}")
        
        # Save optimization backend performance data
        self.backend.save_performance_cache()
    
    def generate_performance_summary(self):
        """Generate performance summary statistics."""
        df = pd.DataFrame(self.results)
        
        # Overall performance
        total_tests = len(df)
        successful_tests = df['success'].sum()
        success_rate = (successful_tests / total_tests) * 100
        
        # Performance by estimator
        estimator_performance = df.groupby('estimator_name').agg({
            'error': ['mean', 'std', 'median'],
            'execution_time': ['mean', 'std', 'median'],
            'success': 'sum'
        }).round(4)
        
        # Convert to JSON-serializable format
        estimator_perf_dict = {}
        for col in estimator_performance.columns:
            if isinstance(col, tuple):
                col_name = '_'.join(str(x) for x in col)
            else:
                col_name = str(col)
            estimator_perf_dict[col_name] = estimator_performance[col].to_dict()
        
        # Performance by confounding scenario
        scenario_performance = df.groupby('confounding_scenario').agg({
            'error': ['mean', 'std'],
            'execution_time': ['mean', 'std'],
            'success': 'sum'
        }).round(4)
        
        # Convert to JSON-serializable format
        scenario_perf_dict = {}
        for col in scenario_performance.columns:
            if isinstance(col, tuple):
                col_name = '_'.join(str(x) for x in col)
            else:
                col_name = str(col)
            scenario_perf_dict[col_name] = scenario_performance[col].to_dict()
        
        # Performance by framework
        framework_performance = df.groupby('framework_used').agg({
            'error': ['mean', 'std'],
            'execution_time': ['mean', 'std'],
            'success': 'sum'
        }).round(4)
        
        # Convert to JSON-serializable format
        framework_perf_dict = {}
        for col in framework_performance.columns:
            if isinstance(col, tuple):
                col_name = '_'.join(str(x) for x in col)
            else:
                col_name = str(col)
            framework_perf_dict[col_name] = framework_performance[col].to_dict()
        
        # Performance by data characteristics
        data_performance = df.groupby(['data_length', 'confounding_scenario']).agg({
            'error': 'mean',
            'execution_time': 'mean',
            'success': 'sum'
        }).round(4)
        
        # Convert to JSON-serializable format
        data_perf_dict = {}
        for col in data_performance.columns:
            col_data = {}
            for idx, value in data_performance[col].items():
                # Convert tuple index to string
                if isinstance(idx, tuple):
                    key = f"{idx[0]}_{idx[1]}"
                else:
                    key = str(idx)
                col_data[key] = float(value) if not np.isnan(value) else None
            data_perf_dict[str(col)] = col_data
        
        summary = {
            'overall': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate,
                'mean_error': float(df['error'].mean()),
                'mean_execution_time': float(df['execution_time'].mean())
            },
            'by_estimator': estimator_perf_dict,
            'by_confounding_scenario': scenario_perf_dict,
            'by_framework': framework_perf_dict,
            'by_data_characteristics': data_perf_dict,
            'optimization_backend_summary': self.backend.get_performance_summary()
        }
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.output_dir, f"comprehensive_classical_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Performance summary saved to: {summary_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE CLASSICAL BENCHMARK SUMMARY")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Mean Error: {df['error'].mean():.4f}")
        print(f"Mean Execution Time: {df['execution_time'].mean():.4f}s")
        
        print("\nFramework Usage:")
        framework_counts = df['framework_used'].value_counts()
        for framework, count in framework_counts.items():
            percentage = (count / total_tests) * 100
            print(f"  {framework}: {count} ({percentage:.1f}%)")
        
        print("\nEstimator Performance:")
        for estimator in df['estimator_name'].unique():
            estimator_data = df[df['estimator_name'] == estimator]
            estimator_success = estimator_data['success'].sum()
            estimator_total = len(estimator_data)
            estimator_success_rate = (estimator_success / estimator_total) * 100
            estimator_error = estimator_data['error'].mean()
            print(f"  {estimator}: {estimator_success_rate:.1f}% success, "
                  f"mean error: {estimator_error:.4f}")
        
        print("\nConfounding Scenario Performance:")
        for scenario in df['confounding_scenario'].unique():
            scenario_data = df[df['confounding_scenario'] == scenario]
            scenario_success = scenario_data['success'].sum()
            scenario_total = len(scenario_data)
            scenario_success_rate = (scenario_success / scenario_total) * 100
            scenario_error = scenario_data['error'].mean()
            print(f"  {scenario}: {scenario_success_rate:.1f}% success, "
                  f"mean error: {scenario_error:.4f}")
        
        self.performance_summary = summary


def main():
    """Main function to run the benchmark."""
    print("Comprehensive Classical LRD Estimators Benchmark with Realistic Confounding")
    print("=" * 80)
    
    # Initialize and run benchmark
    benchmark = ComprehensiveClassicalBenchmark()
    benchmark.run_benchmark()
    
    print("\nBenchmark completed successfully!")
    print("Check the 'comprehensive_results' directory for detailed results.")


if __name__ == "__main__":
    main()
