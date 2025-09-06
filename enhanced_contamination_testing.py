#!/usr/bin/env python3
"""
Enhanced Contamination Testing Framework for LRDBenchmark

This module provides comprehensive contamination testing beyond additive Gaussian noise:
- Multiplicative noise
- Outliers (spikes, drops)
- Missing data (gaps, random missing)
- Domain-specific contamination scenarios
- Mixed contamination types
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import LRDBenchmark components
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.temporal.dma.dma_estimator_unified import DMAEstimator
from lrdbenchmark.analysis.temporal.higuchi.higuchi_estimator_unified import HiguchiEstimator
from lrdbenchmark.analysis.spectral.gph.gph_estimator_unified import GPHEstimator
from lrdbenchmark.analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator
from lrdbenchmark.analysis.spectral.periodogram.periodogram_estimator_unified import PeriodogramEstimator
from lrdbenchmark.analysis.machine_learning.random_forest_estimator import RandomForestEstimator
from lrdbenchmark.analysis.machine_learning.svr_estimator import SVREstimator
from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator import GradientBoostingEstimator
from lrdbenchmark.analysis.machine_learning.neural_network_factory import NeuralNetworkFactory, NNArchitecture, NNConfig

class ContaminationGenerator:
    """Generate various types of contamination for testing robustness."""
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
    
    def additive_gaussian_noise(self, data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add additive Gaussian noise (existing method)."""
        noise = np.random.normal(0, noise_level * np.std(data), len(data))
        return data + noise
    
    def multiplicative_noise(self, data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add multiplicative noise (proportional to signal amplitude)."""
        noise = np.random.normal(1, noise_level, len(data))
        return data * noise
    
    def outliers_spikes(self, data: np.ndarray, outlier_fraction: float = 0.05, 
                       spike_magnitude: float = 3.0) -> np.ndarray:
        """Add random spikes (outliers) to the data."""
        contaminated = data.copy()
        n_outliers = int(len(data) * outlier_fraction)
        outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
        
        for idx in outlier_indices:
            spike = np.random.choice([-1, 1]) * spike_magnitude * np.std(data)
            contaminated[idx] += spike
        
        return contaminated
    
    def outliers_drops(self, data: np.ndarray, outlier_fraction: float = 0.05, 
                      drop_magnitude: float = 0.5) -> np.ndarray:
        """Add random drops (negative outliers) to the data."""
        contaminated = data.copy()
        n_outliers = int(len(data) * outlier_fraction)
        outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
        
        for idx in outlier_indices:
            drop = -drop_magnitude * np.std(data)
            contaminated[idx] += drop
        
        return contaminated
    
    def missing_data_random(self, data: np.ndarray, missing_fraction: float = 0.1) -> np.ndarray:
        """Randomly remove data points (replace with NaN)."""
        contaminated = data.copy().astype(float)
        n_missing = int(len(data) * missing_fraction)
        missing_indices = np.random.choice(len(data), n_missing, replace=False)
        contaminated[missing_indices] = np.nan
        return contaminated
    
    def missing_data_gaps(self, data: np.ndarray, gap_fraction: float = 0.1, 
                         min_gap_length: int = 5) -> np.ndarray:
        """Create gaps of missing data (consecutive NaN values)."""
        contaminated = data.copy().astype(float)
        n_gaps = int(len(data) * gap_fraction / min_gap_length)
        
        for _ in range(n_gaps):
            gap_length = np.random.randint(min_gap_length, min_gap_length * 3)
            start_idx = np.random.randint(0, len(data) - gap_length)
            contaminated[start_idx:start_idx + gap_length] = np.nan
        
        return contaminated
    
    def domain_specific_finance(self, data: np.ndarray, contamination_level: float = 0.1) -> np.ndarray:
        """Finance-specific contamination: market crashes, flash crashes, volatility clustering."""
        contaminated = data.copy()
        
        # Market crash (sudden drop)
        if np.random.random() < 0.3:
            crash_idx = np.random.randint(len(data) // 4, 3 * len(data) // 4)
            crash_duration = np.random.randint(10, 50)
            crash_magnitude = -np.random.uniform(0.1, 0.3) * np.std(data)
            contaminated[crash_idx:crash_idx + crash_duration] += crash_magnitude
        
        # Flash crash (brief extreme drop)
        if np.random.random() < 0.2:
            flash_idx = np.random.randint(0, len(data) - 5)
            flash_magnitude = -np.random.uniform(0.2, 0.5) * np.std(data)
            contaminated[flash_idx:flash_idx + 5] += flash_magnitude
        
        # Volatility clustering (periods of high noise)
        if np.random.random() < 0.4:
            vol_start = np.random.randint(0, len(data) - 100)
            vol_duration = np.random.randint(20, 100)
            vol_noise = np.random.normal(0, contamination_level * 2 * np.std(data), vol_duration)
            contaminated[vol_start:vol_start + vol_duration] += vol_noise
        
        return contaminated
    
    def domain_specific_neuroscience(self, data: np.ndarray, contamination_level: float = 0.1) -> np.ndarray:
        """Neuroscience-specific contamination: artifacts, electrode pops, muscle artifacts."""
        contaminated = data.copy()
        
        # Electrode pop (sudden spike)
        if np.random.random() < 0.3:
            pop_idx = np.random.randint(0, len(data) - 10)
            pop_magnitude = np.random.uniform(2, 5) * np.std(data)
            contaminated[pop_idx:pop_idx + 10] += pop_magnitude
        
        # Muscle artifact (high-frequency noise)
        if np.random.random() < 0.4:
            muscle_start = np.random.randint(0, len(data) - 50)
            muscle_duration = np.random.randint(10, 50)
            muscle_noise = np.random.normal(0, contamination_level * 3 * np.std(data), muscle_duration)
            contaminated[muscle_start:muscle_start + muscle_duration] += muscle_noise
        
        # Eye movement artifact (slow drift)
        if np.random.random() < 0.2:
            eye_start = np.random.randint(0, len(data) - 100)
            eye_duration = np.random.randint(20, 100)
            eye_drift = np.linspace(0, np.random.uniform(-0.5, 0.5) * np.std(data), eye_duration)
            contaminated[eye_start:eye_start + eye_duration] += eye_drift
        
        return contaminated
    
    def domain_specific_climate(self, data: np.ndarray, contamination_level: float = 0.1) -> np.ndarray:
        """Climate-specific contamination: sensor failures, extreme weather, seasonal gaps."""
        contaminated = data.copy()
        
        # Sensor failure (constant value)
        if np.random.random() < 0.2:
            failure_start = np.random.randint(0, len(data) - 30)
            failure_duration = np.random.randint(10, 30)
            failure_value = np.mean(data) + np.random.normal(0, 0.1 * np.std(data))
            contaminated[failure_start:failure_start + failure_duration] = failure_value
        
        # Extreme weather event (sudden change)
        if np.random.random() < 0.3:
            extreme_idx = np.random.randint(0, len(data) - 20)
            extreme_magnitude = np.random.uniform(2, 4) * np.std(data)
            contaminated[extreme_idx:extreme_idx + 20] += extreme_magnitude
        
        # Seasonal gap (missing data during specific periods)
        if np.random.random() < 0.1:
            gap_start = np.random.randint(0, len(data) - 50)
            gap_duration = np.random.randint(20, 50)
            contaminated[gap_start:gap_start + gap_duration] = np.nan
        
        return contaminated
    
    def mixed_contamination(self, data: np.ndarray, contamination_level: float = 0.1) -> np.ndarray:
        """Apply multiple types of contamination simultaneously."""
        contaminated = data.copy()
        
        # Apply different contamination types with different probabilities
        if np.random.random() < 0.3:
            contaminated = self.additive_gaussian_noise(contaminated, contamination_level)
        
        if np.random.random() < 0.2:
            contaminated = self.multiplicative_noise(contaminated, contamination_level)
        
        if np.random.random() < 0.2:
            contaminated = self.outliers_spikes(contaminated, 0.05, 2.0)
        
        if np.random.random() < 0.1:
            contaminated = self.missing_data_random(contaminated, 0.05)
        
        return contaminated

class EnhancedContaminationTester:
    """Comprehensive contamination testing framework."""
    
    def __init__(self, data_dir: str = "contamination_testing"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize contamination generator
        self.contamination_generator = ContaminationGenerator()
        
        # Initialize estimators
        self.estimators = self._initialize_estimators()
        
        # Results storage
        self.results = {}
    
    def _initialize_estimators(self) -> Dict[str, Any]:
        """Initialize all estimators for contamination testing."""
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
        
        # Neural Network estimators (simplified for contamination testing)
        try:
            input_length = 1000
            nn_configs = {
                "CNN": NNConfig(
                    architecture=NNArchitecture.CNN,
                    input_length=input_length,
                    conv_filters=32,
                    hidden_dims=[64, 32],
                    dropout_rate=0.2,
                    learning_rate=0.001,
                    epochs=10  # Reduced for contamination testing
                ),
                "LSTM": NNConfig(
                    architecture=NNArchitecture.LSTM,
                    input_length=input_length,
                    lstm_units=64,
                    hidden_dims=[32],
                    dropout_rate=0.1,
                    learning_rate=0.001,
                    epochs=10
                )
            }
            
            for name, config in nn_configs.items():
                try:
                    network = NeuralNetworkFactory.create_network(config)
                    network.model_name = name
                    estimators[f"NN_{name}"] = network
                except Exception as e:
                    print(f"Warning: Failed to initialize NN_{name}: {e}")
        
        except Exception as e:
            print(f"Warning: Failed to initialize neural networks: {e}")
        
        return estimators
    
    def generate_test_data(self) -> Dict[str, np.ndarray]:
        """Generate test data with different Hurst parameters."""
        test_data = {}
        
        # Generate FBM data with different Hurst parameters
        hurst_values = [0.3, 0.5, 0.7]
        data_lengths = [500, 1000, 2000]
        
        for h in hurst_values:
            for length in data_lengths:
                from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
                fbm = FractionalBrownianMotion(H=h)
                data = fbm.generate(length)
                test_data[f"fbm_h{h}_len{length}"] = data
        
        return test_data
    
    def test_contamination_robustness(self, data: np.ndarray, data_name: str, 
                                    hurst_true: float) -> Dict[str, Any]:
        """Test robustness of all estimators on contaminated data."""
        results = {
            'data_name': data_name,
            'hurst_true': hurst_true,
            'data_length': len(data),
            'contamination_results': {}
        }
        
        # Define contamination scenarios
        contamination_scenarios = {
            'additive_gaussian_0.05': lambda x: self.contamination_generator.additive_gaussian_noise(x, 0.05),
            'additive_gaussian_0.1': lambda x: self.contamination_generator.additive_gaussian_noise(x, 0.1),
            'additive_gaussian_0.2': lambda x: self.contamination_generator.additive_gaussian_noise(x, 0.2),
            'multiplicative_0.05': lambda x: self.contamination_generator.multiplicative_noise(x, 0.05),
            'multiplicative_0.1': lambda x: self.contamination_generator.multiplicative_noise(x, 0.1),
            'multiplicative_0.2': lambda x: self.contamination_generator.multiplicative_noise(x, 0.2),
            'outliers_spikes_0.05': lambda x: self.contamination_generator.outliers_spikes(x, 0.05, 2.0),
            'outliers_spikes_0.1': lambda x: self.contamination_generator.outliers_spikes(x, 0.1, 3.0),
            'outliers_drops_0.05': lambda x: self.contamination_generator.outliers_drops(x, 0.05, 0.5),
            'outliers_drops_0.1': lambda x: self.contamination_generator.outliers_drops(x, 0.1, 0.8),
            'missing_random_0.05': lambda x: self.contamination_generator.missing_data_random(x, 0.05),
            'missing_random_0.1': lambda x: self.contamination_generator.missing_data_random(x, 0.1),
            'missing_gaps_0.05': lambda x: self.contamination_generator.missing_data_gaps(x, 0.05, 5),
            'missing_gaps_0.1': lambda x: self.contamination_generator.missing_data_gaps(x, 0.1, 10),
            'finance_contamination': lambda x: self.contamination_generator.domain_specific_finance(x, 0.1),
            'neuroscience_contamination': lambda x: self.contamination_generator.domain_specific_neuroscience(x, 0.1),
            'climate_contamination': lambda x: self.contamination_generator.domain_specific_climate(x, 0.1),
            'mixed_contamination': lambda x: self.contamination_generator.mixed_contamination(x, 0.1)
        }
        
        # Test each contamination scenario
        for scenario_name, contamination_func in contamination_scenarios.items():
            print(f"  Testing {scenario_name}...")
            
            try:
                # Apply contamination
                contaminated_data = contamination_func(data.copy())
                
                # Test each estimator
                scenario_results = {}
                for estimator_name, estimator in self.estimators.items():
                    result = self._test_estimator_on_contaminated_data(
                        estimator, estimator_name, contaminated_data, hurst_true
                    )
                    scenario_results[estimator_name] = result
                
                results['contamination_results'][scenario_name] = scenario_results
                
            except Exception as e:
                print(f"    Error in {scenario_name}: {e}")
                results['contamination_results'][scenario_name] = {'error': str(e)}
        
        return results
    
    def _test_estimator_on_contaminated_data(self, estimator: Any, estimator_name: str, 
                                           contaminated_data: np.ndarray, 
                                           hurst_true: float) -> Dict[str, Any]:
        """Test a single estimator on contaminated data."""
        result = {
            'estimator': estimator_name,
            'success': False,
            'hurst_estimate': None,
            'mae': None,
            'execution_time': None,
            'error': None
        }
        
        try:
            import time
            start_time = time.time()
            
            # Handle missing data (NaN values)
            if np.any(np.isnan(contaminated_data)):
                # For now, skip estimators that can't handle NaN
                if estimator_name.startswith('NN_'):
                    result['error'] = 'Cannot handle missing data'
                    return result
                
                # For classical/ML estimators, interpolate missing values
                valid_mask = ~np.isnan(contaminated_data)
                if np.sum(valid_mask) < 100:  # Need minimum data points
                    result['error'] = 'Too much missing data'
                    return result
                
                # Interpolate missing values
                contaminated_data = self._interpolate_missing_data(contaminated_data)
            
            # Handle different estimator types
            if estimator_name.startswith('NN_'):
                # Neural networks need training data first
                if not hasattr(estimator, 'is_trained') or not estimator.is_trained:
                    # Generate training data
                    train_data = []
                    train_labels = []
                    for h in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                        for _ in range(5):  # Reduced for contamination testing
                            from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
                            fbm = FractionalBrownianMotion(H=h)
                            train_data.append(fbm.generate(len(contaminated_data)))
                            train_labels.append(h)
                    
                    train_data = np.array(train_data)
                    train_labels = np.array(train_labels)
                    
                    # Train the network
                    estimator.train_model(train_data, train_labels)
                    estimator.is_trained = True
                
                # Make prediction
                if len(contaminated_data) < 1000:
                    padded_data = np.pad(contaminated_data, (0, 1000 - len(contaminated_data)), 'constant')
                else:
                    padded_data = contaminated_data[:1000]
                
                hurst_estimate = estimator.predict(padded_data.reshape(1, -1))[0]
                
            else:
                # Classical/ML estimators
                if len(contaminated_data) < 100:
                    result['error'] = 'Data too short'
                    return result
                
                # Ensure data is properly formatted
                if len(contaminated_data.shape) > 1:
                    contaminated_data = contaminated_data.flatten()
                
                # Estimate Hurst parameter
                estimation_result = estimator.estimate(contaminated_data)
                hurst_estimate = estimation_result['hurst_parameter']
            
            execution_time = time.time() - start_time
            mae = abs(hurst_estimate - hurst_true)
            
            result.update({
                'success': True,
                'hurst_estimate': float(hurst_estimate),
                'mae': float(mae),
                'execution_time': execution_time
            })
            
        except Exception as e:
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time if 'start_time' in locals() else None
        
        return result
    
    def _interpolate_missing_data(self, data: np.ndarray) -> np.ndarray:
        """Interpolate missing data using linear interpolation."""
        valid_mask = ~np.isnan(data)
        if np.sum(valid_mask) < 2:
            return data
        
        valid_indices = np.where(valid_mask)[0]
        valid_values = data[valid_mask]
        
        # Linear interpolation
        interpolated = np.interp(np.arange(len(data)), valid_indices, valid_values)
        return interpolated
    
    def run_comprehensive_contamination_testing(self) -> Dict[str, Any]:
        """Run comprehensive contamination testing."""
        print("Starting comprehensive contamination testing...")
        
        # Generate test data
        test_data = self.generate_test_data()
        
        # Initialize results
        self.results = {
            'contamination_scenarios': {},
            'estimator_robustness': {},
            'summary': {}
        }
        
        # Test each dataset
        all_results = []
        for data_name, data in test_data.items():
            print(f"\nTesting {data_name}...")
            
            # Extract Hurst parameter from data name
            hurst_true = float(data_name.split('_h')[1].split('_')[0])
            
            # Test contamination robustness
            results = self.test_contamination_robustness(data, data_name, hurst_true)
            all_results.append(results)
        
        # Analyze results
        self._analyze_contamination_results(all_results)
        
        # Generate summary
        self._generate_contamination_summary(all_results)
        
        # Save results
        self._save_results()
        
        print(f"\nContamination testing complete!")
        print(f"Tested {len(all_results)} datasets across multiple contamination scenarios")
        
        return self.results
    
    def _analyze_contamination_results(self, all_results: List[Dict]) -> None:
        """Analyze contamination testing results."""
        # Initialize analysis structures
        scenario_stats = {}
        estimator_stats = {}
        
        for result in all_results:
            data_name = result['data_name']
            hurst_true = result['hurst_true']
            
            for scenario_name, scenario_results in result['contamination_results'].items():
                if scenario_name not in scenario_stats:
                    scenario_stats[scenario_name] = {
                        'total_tests': 0,
                        'successful_tests': 0,
                        'mae_values': [],
                        'execution_times': []
                    }
                
                for estimator_name, estimator_result in scenario_results.items():
                    if estimator_name not in estimator_stats:
                        estimator_stats[estimator_name] = {
                            'total_tests': 0,
                            'successful_tests': 0,
                            'mae_values': [],
                            'execution_times': []
                        }
                    
                    # Update scenario stats
                    scenario_stats[scenario_name]['total_tests'] += 1
                    if estimator_result.get('success', False):
                        scenario_stats[scenario_name]['successful_tests'] += 1
                        scenario_stats[scenario_name]['mae_values'].append(estimator_result['mae'])
                        scenario_stats[scenario_name]['execution_times'].append(estimator_result['execution_time'])
                    
                    # Update estimator stats
                    estimator_stats[estimator_name]['total_tests'] += 1
                    if estimator_result.get('success', False):
                        estimator_stats[estimator_name]['successful_tests'] += 1
                        estimator_stats[estimator_name]['mae_values'].append(estimator_result['mae'])
                        estimator_stats[estimator_name]['execution_times'].append(estimator_result['execution_time'])
        
        # Calculate summary statistics
        for scenario, stats in scenario_stats.items():
            stats['success_rate'] = stats['successful_tests'] / stats['total_tests'] if stats['total_tests'] > 0 else 0
            stats['mean_mae'] = np.mean(stats['mae_values']) if stats['mae_values'] else None
            stats['std_mae'] = np.std(stats['mae_values']) if stats['mae_values'] else None
            stats['mean_execution_time'] = np.mean(stats['execution_times']) if stats['execution_times'] else None
        
        for estimator, stats in estimator_stats.items():
            stats['success_rate'] = stats['successful_tests'] / stats['total_tests'] if stats['total_tests'] > 0 else 0
            stats['mean_mae'] = np.mean(stats['mae_values']) if stats['mae_values'] else None
            stats['std_mae'] = np.std(stats['mae_values']) if stats['mae_values'] else None
            stats['mean_execution_time'] = np.mean(stats['execution_times']) if stats['execution_times'] else None
        
        self.results['contamination_scenarios'] = scenario_stats
        self.results['estimator_robustness'] = estimator_stats
    
    def _generate_contamination_summary(self, all_results: List[Dict]) -> None:
        """Generate contamination testing summary."""
        total_tests = sum(len(result['contamination_results']) * len(self.estimators) for result in all_results)
        successful_tests = sum(
            sum(1 for estimator_result in scenario_results.values() 
                if estimator_result.get('success', False))
            for result in all_results
            for scenario_results in result['contamination_results'].values()
        )
        
        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Scenario breakdown
        scenario_success = {}
        for scenario, stats in self.results['contamination_scenarios'].items():
            scenario_success[scenario] = stats['success_rate']
        
        # Estimator breakdown
        estimator_success = {}
        for estimator, stats in self.results['estimator_robustness'].items():
            estimator_success[estimator] = stats['success_rate']
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'overall_success_rate': overall_success_rate,
            'scenario_success_rates': scenario_success,
            'estimator_success_rates': estimator_success
        }
    
    def _save_results(self) -> None:
        """Save contamination testing results."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.data_dir / f"contamination_testing_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for estimator, stats in self.results['estimator_robustness'].items():
            summary_data.append({
                'estimator': estimator,
                'success_rate': stats['success_rate'],
                'total_tests': stats['total_tests'],
                'successful_tests': stats['successful_tests'],
                'mean_mae': stats['mean_mae'],
                'mean_execution_time': stats['mean_execution_time']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('success_rate', ascending=False)
        
        csv_file = self.data_dir / f"contamination_testing_summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"Results saved to {self.data_dir}")
        print(f"  Detailed results: {results_file}")
        print(f"  Summary: {csv_file}")
    
    def generate_contamination_plots(self) -> None:
        """Generate contamination testing visualization plots."""
        if not self.results:
            print("No results to plot. Run contamination testing first.")
            return
        
        # Create plots directory
        plots_dir = self.data_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Success rate by contamination scenario
        self._plot_scenario_success_rates(plots_dir)
        
        # 2. Success rate by estimator
        self._plot_estimator_robustness(plots_dir)
        
        # 3. MAE by contamination type
        self._plot_mae_by_contamination(plots_dir)
        
        # 4. Robustness heatmap
        self._plot_robustness_heatmap(plots_dir)
    
    def _plot_scenario_success_rates(self, plots_dir: Path) -> None:
        """Plot success rates by contamination scenario."""
        scenario_success = self.results['summary']['scenario_success_rates']
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        scenarios = list(scenario_success.keys())
        success_rates = list(scenario_success.values())
        
        # Color by contamination type
        colors = []
        for scenario in scenarios:
            if 'additive' in scenario:
                colors.append('#1f77b4')  # Blue
            elif 'multiplicative' in scenario:
                colors.append('#ff7f0e')  # Orange
            elif 'outliers' in scenario:
                colors.append('#2ca02c')  # Green
            elif 'missing' in scenario:
                colors.append('#d62728')  # Red
            elif 'domain' in scenario:
                colors.append('#9467bd')  # Purple
            else:
                colors.append('#8c564b')  # Brown
        
        bars = ax.bar(range(len(scenarios)), success_rates, color=colors)
        
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.set_ylabel('Success Rate')
        ax.set_title('Contamination Testing: Success Rate by Scenario')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{rate:.2f}', ha='center', va='bottom')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Additive Noise'),
            Patch(facecolor='#ff7f0e', label='Multiplicative Noise'),
            Patch(facecolor='#2ca02c', label='Outliers'),
            Patch(facecolor='#d62728', label='Missing Data'),
            Patch(facecolor='#9467bd', label='Domain-Specific')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'scenario_success_rates.png', dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / 'scenario_success_rates.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_estimator_robustness(self, plots_dir: Path) -> None:
        """Plot estimator robustness to contamination."""
        estimator_success = self.results['summary']['estimator_success_rates']
        
        # Sort by success rate
        sorted_estimators = sorted(estimator_success.items(), key=lambda x: x[1], reverse=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        estimators = [item[0] for item in sorted_estimators]
        success_rates = [item[1] for item in sorted_estimators]
        
        # Color by estimator type
        colors = []
        for estimator in estimators:
            if estimator.startswith('NN_'):
                colors.append('#ff7f0e')  # Orange for neural networks
            elif estimator in ['RandomForest', 'SVR', 'GradientBoosting']:
                colors.append('#2ca02c')  # Green for ML
            else:
                colors.append('#1f77b4')  # Blue for classical
        
        bars = ax.barh(range(len(estimators)), success_rates, color=colors)
        
        ax.set_yticks(range(len(estimators)))
        ax.set_yticklabels(estimators)
        ax.set_xlabel('Success Rate')
        ax.set_title('Contamination Testing: Estimator Robustness')
        ax.set_xlim(0, 1)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, success_rates)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{rate:.2f}', va='center', ha='left')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='Classical'),
                          Patch(facecolor='#2ca02c', label='Machine Learning'),
                          Patch(facecolor='#ff7f0e', label='Neural Networks')]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'estimator_robustness.png', dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / 'estimator_robustness.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_mae_by_contamination(self, plots_dir: Path) -> None:
        """Plot MAE by contamination type."""
        # Group scenarios by contamination type
        contamination_types = {
            'Additive Noise': [],
            'Multiplicative Noise': [],
            'Outliers': [],
            'Missing Data': [],
            'Domain-Specific': []
        }
        
        for scenario, stats in self.results['contamination_scenarios'].items():
            if stats['mean_mae'] is not None:
                if 'additive' in scenario:
                    contamination_types['Additive Noise'].append(stats['mean_mae'])
                elif 'multiplicative' in scenario:
                    contamination_types['Multiplicative Noise'].append(stats['mean_mae'])
                elif 'outliers' in scenario:
                    contamination_types['Outliers'].append(stats['mean_mae'])
                elif 'missing' in scenario:
                    contamination_types['Missing Data'].append(stats['mean_mae'])
                elif 'domain' in scenario:
                    contamination_types['Domain-Specific'].append(stats['mean_mae'])
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        box_data = [mae_values for mae_values in contamination_types.values() if mae_values]
        box_labels = [label for label, mae_values in contamination_types.items() if mae_values]
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Mean Absolute Error (MAE)')
        ax.set_title('Contamination Testing: MAE by Contamination Type')
        ax.set_yscale('log')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'mae_by_contamination.png', dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / 'mae_by_contamination.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_heatmap(self, plots_dir: Path) -> None:
        """Plot robustness heatmap (estimators vs contamination scenarios)."""
        # Create robustness matrix
        estimators = list(self.results['estimator_robustness'].keys())
        scenarios = list(self.results['contamination_scenarios'].keys())
        
        robustness_matrix = np.zeros((len(estimators), len(scenarios)))
        
        for i, estimator in enumerate(estimators):
            for j, scenario in enumerate(scenarios):
                # Find success rate for this estimator-scenario combination
                # This would require more detailed tracking in the results
                # For now, use overall estimator success rate
                robustness_matrix[i, j] = self.results['estimator_robustness'][estimator]['success_rate']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(robustness_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(scenarios)))
        ax.set_yticks(range(len(estimators)))
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.set_yticklabels(estimators)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Success Rate')
        
        ax.set_title('Contamination Testing: Robustness Heatmap')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'robustness_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / 'robustness_heatmap.pdf', bbox_inches='tight')
        plt.close()

def main():
    """Run comprehensive contamination testing."""
    print("LRDBenchmark Enhanced Contamination Testing Framework")
    print("=" * 60)
    
    # Initialize tester
    tester = EnhancedContaminationTester()
    
    # Run comprehensive contamination testing
    results = tester.run_comprehensive_contamination_testing()
    
    # Generate plots
    tester.generate_contamination_plots()
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONTAMINATION TESTING SUMMARY")
    print("=" * 60)
    
    summary = results['summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful Tests: {summary['successful_tests']}")
    print(f"Overall Success Rate: {summary['overall_success_rate']:.2%}")
    
    print("\nTop 5 Most Robust Estimators:")
    sorted_estimators = sorted(summary['estimator_success_rates'].items(), 
                             key=lambda x: x[1], reverse=True)
    for i, (estimator, rate) in enumerate(sorted_estimators[:5]):
        print(f"  {i+1}. {estimator}: {rate:.2%}")
    
    print("\nContamination Scenario Success Rates:")
    for scenario, rate in summary['scenario_success_rates'].items():
        print(f"  {scenario}: {rate:.2%}")
    
    print(f"\nResults saved to: {tester.data_dir}")
    print("Contamination testing complete!")

if __name__ == "__main__":
    main()
