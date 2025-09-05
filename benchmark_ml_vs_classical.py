#!/usr/bin/env python3
"""
Comprehensive Benchmark: ML Models vs Classical Models.

This script performs a thorough comparison between machine learning models
and classical LRD estimation methods across various data types and conditions.
"""

import numpy as np
import time
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import classical estimators
from lrdbenchmark.analysis.comprehensive_adaptive_estimators import (
    ComprehensiveAdaptiveRS, ComprehensiveAdaptiveDFA, ComprehensiveAdaptiveDMA,
    ComprehensiveAdaptiveHiguchi, ComprehensiveAdaptiveGPH, ComprehensiveAdaptiveWhittle,
    ComprehensiveAdaptivePeriodogram, ComprehensiveAdaptiveCWT, ComprehensiveAdaptiveWaveletVar,
    ComprehensiveAdaptiveWaveletLogVar, ComprehensiveAdaptiveWaveletWhittle,
    ComprehensiveAdaptiveWaveletLeaders, ComprehensiveAdaptiveMFDFA
)

# Import ML models
from lrdbenchmark.analysis.machine_learning.production_ml_system import (
    ProductionMLSystem, ProductionConfig
)
from lrdbenchmark.analysis.machine_learning.enhanced_ml_estimators import (
    EnhancedRandomForestEstimator, EnhancedSVREstimator, EnhancedGradientBoostingEstimator
)

# Import data models
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbenchmark.models.data_models.arfima.arfima_model import ARFIMAModel
from lrdbenchmark.models.data_models.mrw.mrw_model import MultifractalRandomWalk

# Import contamination factory
from lrdbenchmark.models.contamination.contamination_factory import (
    ContaminationFactory, ConfoundingScenario
)

class MLvsClassicalBenchmark:
    """Comprehensive benchmark comparing ML and classical models."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize estimators
        self.classical_estimators = self._initialize_classical_estimators()
        self.ml_estimators = self._initialize_ml_estimators()
        
        # Initialize data models
        self.data_models = {
            'FBM': FractionalBrownianMotion,
            'FGN': FractionalGaussianNoise,
            'ARFIMA': ARFIMAModel,
            'MRW': MultifractalRandomWalk
        }
        
        # Initialize contamination factory
        self.contamination_factory = ContaminationFactory()
        
        # Results storage
        self.results = []
        
    def _initialize_classical_estimators(self) -> Dict[str, Any]:
        """Initialize classical estimators."""
        return {
            'RS': ComprehensiveAdaptiveRS(),
            'DFA': ComprehensiveAdaptiveDFA(),
            'DMA': ComprehensiveAdaptiveDMA(),
            'Higuchi': ComprehensiveAdaptiveHiguchi(),
            'GPH': ComprehensiveAdaptiveGPH(),
            'Whittle': ComprehensiveAdaptiveWhittle(),
            'Periodogram': ComprehensiveAdaptivePeriodogram(),
            'CWT': ComprehensiveAdaptiveCWT(),
            'WaveletVar': ComprehensiveAdaptiveWaveletVar(),
            'WaveletLogVar': ComprehensiveAdaptiveWaveletLogVar(),
            'WaveletWhittle': ComprehensiveAdaptiveWaveletWhittle(),
            'WaveletLeaders': ComprehensiveAdaptiveWaveletLeaders(),
            'MFDFA': ComprehensiveAdaptiveMFDFA()
        }
    
    def _initialize_ml_estimators(self) -> Dict[str, Any]:
        """Initialize ML estimators."""
        return {
            'RandomForest': EnhancedRandomForestEstimator(),
            'SVR': EnhancedSVREstimator(),
            'GradientBoosting': EnhancedGradientBoostingEstimator(),
            'CNN': None,  # Will be initialized with production system
            'Transformer': None  # Will be initialized with production system
        }
    
    def generate_benchmark_data(self, n_samples: int = 100, sequence_lengths: List[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate comprehensive benchmark data."""
        if sequence_lengths is None:
            sequence_lengths = [100, 250, 500, 1000]
        
        logger.info(f"Generating benchmark data: {n_samples} samples")
        
        X_list = []
        y_list = []
        metadata = {
            'data_model': [],
            'hurst_parameter': [],
            'sequence_length': [],
            'contamination_type': [],
            'sample_id': []
        }
        
        # Generate data for each model type
        samples_per_combination = max(1, n_samples // (len(self.data_models) * len(sequence_lengths) * 8))
        
        for model_name, model_class in self.data_models.items():
            logger.info(f"  Generating {model_name} data...")
            
            for seq_len in sequence_lengths:
                for hurst in np.linspace(0.2, 0.8, 8):  # 8 different Hurst values
                    for _ in range(samples_per_combination):
                        
                        # Generate base data
                        if model_name == 'ARFIMA':
                            d = hurst - 0.5
                            model = model_class(d=d)
                        elif model_name == 'MRW':
                            model = model_class(H=hurst, lambda_param=0.5)
                        else:
                            model = model_class(H=hurst)
                        
                        # Generate data
                        try:
                            data = model.generate(n=seq_len)
                            data = np.asarray(data).flatten()
                            
                            # Ensure correct length and type
                            if len(data) != seq_len:
                                if len(data) > seq_len:
                                    data = data[:seq_len]
                                else:
                                    padded = np.zeros(seq_len)
                                    padded[:len(data)] = data
                                    data = padded
                            
                            # Ensure data is float64 and has correct shape
                            data = data.astype(np.float64)
                            
                        except Exception as e:
                            logger.warning(f"Failed to generate {model_name} data: {e}")
                            # Create dummy data
                            data = np.random.randn(seq_len).astype(np.float64)
                        
                        X_list.append(data)
                        y_list.append(hurst)
                        
                        # Update metadata
                        metadata['data_model'].append(model_name)
                        metadata['hurst_parameter'].append(hurst)
                        metadata['sequence_length'].append(seq_len)
                        metadata['contamination_type'].append('pure')
                        metadata['sample_id'].append(len(X_list) - 1)
        
        # Add contaminated data
        logger.info("  Adding contaminated data...")
        contamination_scenarios = [
            ConfoundingScenario.EEG_OCULAR_ARTIFACTS,
            ConfoundingScenario.EEG_MUSCLE_ARTIFACTS,
            ConfoundingScenario.EEG_60HZ_NOISE,
            ConfoundingScenario.EEG_MOVEMENT_ARTIFACTS
        ]
        
        # Add contaminated versions of some samples
        n_contaminated = min(50, len(X_list) // 4)
        contaminated_indices = np.random.choice(len(X_list), n_contaminated, replace=False)
        
        for i, idx in enumerate(contaminated_indices):
            original_data = X_list[idx].copy()
            scenario = contamination_scenarios[i % len(contamination_scenarios)]
            
            try:
                contaminated_data, _ = self.contamination_factory.apply_confounding(
                    original_data, scenario, intensity=0.1
                )
                
                # Ensure contaminated data has correct shape and type
                contaminated_data = np.asarray(contaminated_data).flatten().astype(np.float64)
                if len(contaminated_data) != len(original_data):
                    if len(contaminated_data) > len(original_data):
                        contaminated_data = contaminated_data[:len(original_data)]
                    else:
                        padded = np.zeros(len(original_data))
                        padded[:len(contaminated_data)] = contaminated_data
                        contaminated_data = padded
                
                X_list.append(contaminated_data)
                y_list.append(y_list[idx])  # Same Hurst parameter
                
                # Update metadata
                metadata['data_model'].append(metadata['data_model'][idx])
                metadata['hurst_parameter'].append(metadata['hurst_parameter'][idx])
                metadata['sequence_length'].append(metadata['sequence_length'][idx])
                metadata['contamination_type'].append(scenario.value)
                metadata['sample_id'].append(len(X_list) - 1)
                
            except Exception as e:
                logger.warning(f"Failed to apply contamination {scenario}: {e}")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Generated {len(X)} total samples")
        return X, y, metadata
    
    def train_ml_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train ML models on the benchmark data."""
        logger.info("Training ML models...")
        
        trained_models = {}
        
        # Train traditional ML models
        for name, estimator in self.ml_estimators.items():
            if estimator is not None:
                try:
                    logger.info(f"  Training {name}...")
                    start_time = time.time()
                    estimator.train(X, y)
                    training_time = time.time() - start_time
                    
                    trained_models[name] = {
                        'estimator': estimator,
                        'training_time': training_time,
                        'type': 'traditional_ml'
                    }
                    logger.info(f"    ✅ {name} trained in {training_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"    ❌ {name} training failed: {e}")
        
        # Train deep learning models
        for model_type in ['cnn', 'transformer']:
            try:
                logger.info(f"  Training {model_type.upper()}...")
                
                config = ProductionConfig(
                    model_type=model_type,
                    input_length=X.shape[1],
                    hidden_dims=[64, 32],
                    dropout_rate=0.2,
                    learning_rate=0.001,
                    batch_size=32,
                    epochs=50,
                    early_stopping_patience=10,
                    validation_split=0.2,
                    framework_priority=['torch', 'numba']
                )
                
                system = ProductionMLSystem(config)
                start_time = time.time()
                training_result = system.train(X, y)
                training_time = time.time() - start_time
                
                trained_models[model_type.upper()] = {
                    'system': system,
                    'training_time': training_time,
                    'training_result': training_result,
                    'type': 'deep_learning'
                }
                logger.info(f"    ✅ {model_type.upper()} trained in {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"    ❌ {model_type.upper()} training failed: {e}")
        
        return trained_models
    
    def run_benchmark(self, X: np.ndarray, y: np.ndarray, metadata: Dict[str, Any], 
                     trained_ml_models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run comprehensive benchmark."""
        logger.info("Running comprehensive benchmark...")
        
        results = []
        n_samples = len(X)
        
        # Benchmark classical estimators
        logger.info("Benchmarking classical estimators...")
        for name, estimator in self.classical_estimators.items():
            logger.info(f"  Testing {name}...")
            
            for i in range(n_samples):
                try:
                    start_time = time.time()
                    result = estimator.estimate(X[i])
                    execution_time = time.time() - start_time
                    
                    results.append({
                        'estimator': name,
                        'estimator_type': 'classical',
                        'sample_id': i,
                        'true_hurst': y[i],
                        'estimated_hurst': result.get('hurst_parameter', 0.5),
                        'execution_time': execution_time,
                        'r_squared': result.get('r_squared', 0.0),
                        'p_value': result.get('p_value', None),
                        'method': result.get('method', name),
                        'optimization_framework': result.get('optimization_framework', 'unknown'),
                        'data_model': metadata['data_model'][i],
                        'sequence_length': metadata['sequence_length'][i],
                        'contamination_type': metadata['contamination_type'][i],
                        'success': True,
                        'error': None
                    })
                    
                except Exception as e:
                    results.append({
                        'estimator': name,
                        'estimator_type': 'classical',
                        'sample_id': i,
                        'true_hurst': y[i],
                        'estimated_hurst': 0.5,
                        'execution_time': 0.0,
                        'r_squared': 0.0,
                        'p_value': None,
                        'method': name,
                        'optimization_framework': 'unknown',
                        'data_model': metadata['data_model'][i],
                        'sequence_length': metadata['sequence_length'][i],
                        'contamination_type': metadata['contamination_type'][i],
                        'success': False,
                        'error': str(e)
                    })
        
        # Benchmark ML models
        logger.info("Benchmarking ML models...")
        for name, model_info in trained_ml_models.items():
            logger.info(f"  Testing {name}...")
            
            for i in range(n_samples):
                try:
                    start_time = time.time()
                    
                    if model_info['type'] == 'traditional_ml':
                        result = model_info['estimator'].estimate(X[i])
                        estimated_hurst = result.get('hurst_parameter', 0.5)
                        execution_time = time.time() - start_time
                        
                    elif model_info['type'] == 'deep_learning':
                        prediction = model_info['system'].predict(X[i])
                        estimated_hurst = prediction.hurst_parameter
                        execution_time = prediction.execution_time
                    
                    results.append({
                        'estimator': name,
                        'estimator_type': 'ml',
                        'sample_id': i,
                        'true_hurst': y[i],
                        'estimated_hurst': estimated_hurst,
                        'execution_time': execution_time,
                        'r_squared': 0.0,  # Would need training result
                        'p_value': None,
                        'method': f"{name}_ML",
                        'optimization_framework': 'ml',
                        'data_model': metadata['data_model'][i],
                        'sequence_length': metadata['sequence_length'][i],
                        'contamination_type': metadata['contamination_type'][i],
                        'success': True,
                        'error': None
                    })
                    
                except Exception as e:
                    results.append({
                        'estimator': name,
                        'estimator_type': 'ml',
                        'sample_id': i,
                        'true_hurst': y[i],
                        'estimated_hurst': 0.5,
                        'execution_time': 0.0,
                        'r_squared': 0.0,
                        'p_value': None,
                        'method': f"{name}_ML",
                        'optimization_framework': 'ml',
                        'data_model': metadata['data_model'][i],
                        'sequence_length': metadata['sequence_length'][i],
                        'contamination_type': metadata['contamination_type'][i],
                        'success': False,
                        'error': str(e)
                    })
        
        logger.info(f"Benchmark completed: {len(results)} total evaluations")
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results."""
        logger.info("Analyzing benchmark results...")
        
        if not results:
            logger.warning("No results to analyze")
            return {}
        
        df = pd.DataFrame(results)
        
        # Calculate performance metrics
        analysis = {}
        
        # Overall performance
        analysis['overall'] = self._calculate_performance_metrics(df)
        
        # Performance by estimator type
        analysis['by_type'] = {}
        for estimator_type in df['estimator_type'].unique():
            type_df = df[df['estimator_type'] == estimator_type]
            analysis['by_type'][estimator_type] = self._calculate_performance_metrics(type_df)
        
        # Performance by individual estimator
        analysis['by_estimator'] = {}
        for estimator in df['estimator'].unique():
            estimator_df = df[df['estimator'] == estimator]
            analysis['by_estimator'][estimator] = self._calculate_performance_metrics(estimator_df)
        
        # Performance by data model
        analysis['by_data_model'] = {}
        for data_model in df['data_model'].unique():
            model_df = df[df['data_model'] == data_model]
            analysis['by_data_model'][data_model] = self._calculate_performance_metrics(model_df)
        
        # Performance by contamination type
        analysis['by_contamination'] = {}
        for contamination in df['contamination_type'].unique():
            cont_df = df[df['contamination_type'] == contamination]
            analysis['by_contamination'][contamination] = self._calculate_performance_metrics(cont_df)
        
        # Performance by sequence length
        analysis['by_sequence_length'] = {}
        for seq_len in df['sequence_length'].unique():
            len_df = df[df['sequence_length'] == seq_len]
            analysis['by_sequence_length'][seq_len] = self._calculate_performance_metrics(len_df)
        
        return analysis
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for a subset of results."""
        if len(df) == 0:
            return {}
        
        # Filter successful results
        successful_df = df[df['success'] == True]
        
        if len(successful_df) == 0:
            return {
                'success_rate': 0.0,
                'n_total': len(df),
                'n_successful': 0
            }
        
        # Calculate errors
        errors = np.abs(successful_df['estimated_hurst'] - successful_df['true_hurst'])
        
        return {
            'success_rate': len(successful_df) / len(df),
            'n_total': len(df),
            'n_successful': len(successful_df),
            'mean_absolute_error': float(np.mean(errors)),
            'median_absolute_error': float(np.median(errors)),
            'std_absolute_error': float(np.std(errors)),
            'mean_execution_time': float(np.mean(successful_df['execution_time'])),
            'median_execution_time': float(np.median(successful_df['execution_time'])),
            'std_execution_time': float(np.std(successful_df['execution_time'])),
            'correlation': float(np.corrcoef(successful_df['true_hurst'], successful_df['estimated_hurst'])[0, 1])
        }
    
    def create_visualizations(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations...")
        
        if not results:
            logger.warning("No results to visualize")
            return
        
        df = pd.DataFrame(results)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Overall performance comparison
        ax1 = plt.subplot(3, 3, 1)
        self._plot_overall_performance(ax1, analysis)
        
        # 2. Performance by estimator type
        ax2 = plt.subplot(3, 3, 2)
        self._plot_by_estimator_type(ax2, analysis)
        
        # 3. Individual estimator performance
        ax3 = plt.subplot(3, 3, 3)
        self._plot_individual_estimators(ax3, analysis)
        
        # 4. Error distribution
        ax4 = plt.subplot(3, 3, 4)
        self._plot_error_distribution(ax4, df)
        
        # 5. Execution time comparison
        ax5 = plt.subplot(3, 3, 5)
        self._plot_execution_times(ax5, df)
        
        # 6. Performance by data model
        ax6 = plt.subplot(3, 3, 6)
        self._plot_by_data_model(ax6, analysis)
        
        # 7. Performance by contamination
        ax7 = plt.subplot(3, 3, 7)
        self._plot_by_contamination(ax7, analysis)
        
        # 8. Performance by sequence length
        ax8 = plt.subplot(3, 3, 8)
        self._plot_by_sequence_length(ax8, analysis)
        
        # 9. Scatter plot: True vs Estimated
        ax9 = plt.subplot(3, 3, 9)
        self._plot_true_vs_estimated(ax9, df)
        
        plt.tight_layout()
        
        # Save the figure
        output_path = self.output_dir / "ml_vs_classical_comprehensive_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
    
    def _plot_overall_performance(self, ax, analysis):
        """Plot overall performance metrics."""
        overall = analysis.get('overall', {})
        
        metrics = ['Success Rate', 'Mean Absolute Error', 'Mean Execution Time (s)']
        values = [
            overall.get('success_rate', 0) * 100,
            overall.get('mean_absolute_error', 0),
            overall.get('mean_execution_time', 0)
        ]
        
        bars = ax.bar(metrics, values, color=['#2E8B57', '#DC143C', '#4169E1'])
        ax.set_title('Overall Performance Metrics', fontweight='bold')
        ax.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_by_estimator_type(self, ax, analysis):
        """Plot performance by estimator type."""
        by_type = analysis.get('by_type', {})
        
        types = list(by_type.keys())
        success_rates = [by_type[t].get('success_rate', 0) * 100 for t in types]
        maes = [by_type[t].get('mean_absolute_error', 0) for t in types]
        
        x = np.arange(len(types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, success_rates, width, label='Success Rate (%)', color='#2E8B57')
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, maes, width, label='Mean Absolute Error', color='#DC143C')
        
        ax.set_xlabel('Estimator Type')
        ax.set_ylabel('Success Rate (%)', color='#2E8B57')
        ax2.set_ylabel('Mean Absolute Error', color='#DC143C')
        ax.set_title('Performance by Estimator Type', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(types)
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    def _plot_individual_estimators(self, ax, analysis):
        """Plot individual estimator performance."""
        by_estimator = analysis.get('by_estimator', {})
        
        # Get top 10 estimators by success rate
        estimators = sorted(by_estimator.items(), 
                          key=lambda x: x[1].get('success_rate', 0), 
                          reverse=True)[:10]
        
        names = [e[0] for e in estimators]
        success_rates = [e[1].get('success_rate', 0) * 100 for e in estimators]
        
        bars = ax.barh(names, success_rates, color='#4169E1')
        ax.set_xlabel('Success Rate (%)')
        ax.set_title('Top 10 Estimators by Success Rate', fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, success_rates):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}%', ha='left', va='center')
    
    def _plot_error_distribution(self, ax, df):
        """Plot error distribution."""
        successful_df = df[df['success'] == True]
        if len(successful_df) == 0:
            ax.text(0.5, 0.5, 'No successful results', ha='center', va='center', transform=ax.transAxes)
            return
        
        errors = np.abs(successful_df['estimated_hurst'] - successful_df['true_hurst'])
        
        ax.hist(errors, bins=30, alpha=0.7, color='#FF6347', edgecolor='black')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Absolute Errors', fontweight='bold')
        ax.axvline(np.mean(errors), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(errors):.3f}')
        ax.legend()
    
    def _plot_execution_times(self, ax, df):
        """Plot execution time comparison."""
        successful_df = df[df['success'] == True]
        if len(successful_df) == 0:
            ax.text(0.5, 0.5, 'No successful results', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Group by estimator type
        classical_times = successful_df[successful_df['estimator_type'] == 'classical']['execution_time']
        ml_times = successful_df[successful_df['estimator_type'] == 'ml']['execution_time']
        
        data = [classical_times, ml_times]
        labels = ['Classical', 'ML']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('#2E8B57')
        bp['boxes'][1].set_facecolor('#DC143C')
        
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Execution Time Comparison', fontweight='bold')
        ax.set_yscale('log')
    
    def _plot_by_data_model(self, ax, analysis):
        """Plot performance by data model."""
        by_data_model = analysis.get('by_data_model', {})
        
        models = list(by_data_model.keys())
        success_rates = [by_data_model[m].get('success_rate', 0) * 100 for m in models]
        
        bars = ax.bar(models, success_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Performance by Data Model', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}%', ha='center', va='bottom')
    
    def _plot_by_contamination(self, ax, analysis):
        """Plot performance by contamination type."""
        by_contamination = analysis.get('by_contamination', {})
        
        contaminations = list(by_contamination.keys())
        success_rates = [by_contamination[c].get('success_rate', 0) * 100 for c in contaminations]
        
        bars = ax.bar(contaminations, success_rates, color='#9B59B6')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Performance by Contamination Type', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}%', ha='center', va='bottom')
    
    def _plot_by_sequence_length(self, ax, analysis):
        """Plot performance by sequence length."""
        by_sequence_length = analysis.get('by_sequence_length', {})
        
        lengths = sorted(by_sequence_length.keys())
        success_rates = [by_sequence_length[l].get('success_rate', 0) * 100 for l in lengths]
        
        ax.plot(lengths, success_rates, marker='o', linewidth=2, markersize=8, color='#E67E22')
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Performance by Sequence Length', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(lengths, success_rates):
            ax.text(x, y + y*0.01, f'{y:.1f}%', ha='center', va='bottom')
    
    def _plot_true_vs_estimated(self, ax, df):
        """Plot true vs estimated Hurst parameters."""
        successful_df = df[df['success'] == True]
        if len(successful_df) == 0:
            ax.text(0.5, 0.5, 'No successful results', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Separate classical and ML results
        classical_df = successful_df[successful_df['estimator_type'] == 'classical']
        ml_df = successful_df[successful_df['estimator_type'] == 'ml']
        
        if len(classical_df) > 0:
            ax.scatter(classical_df['true_hurst'], classical_df['estimated_hurst'], 
                      alpha=0.6, label='Classical', color='#2E8B57', s=20)
        
        if len(ml_df) > 0:
            ax.scatter(ml_df['true_hurst'], ml_df['estimated_hurst'], 
                      alpha=0.6, label='ML', color='#DC143C', s=20)
        
        # Perfect prediction line
        min_val = min(successful_df['true_hurst'].min(), successful_df['estimated_hurst'].min())
        max_val = max(successful_df['true_hurst'].max(), successful_df['estimated_hurst'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
        
        ax.set_xlabel('True Hurst Parameter')
        ax.set_ylabel('Estimated Hurst Parameter')
        ax.set_title('True vs Estimated Hurst Parameters', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save benchmark results."""
        logger.info("Saving benchmark results...")
        
        # Save raw results
        results_path = self.output_dir / "ml_vs_classical_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save analysis
        analysis_path = self.output_dir / "ml_vs_classical_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save CSV for easy analysis
        df = pd.DataFrame(results)
        csv_path = self.output_dir / "ml_vs_classical_results.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {self.output_dir}")

def main():
    """Main benchmark function."""
    logger.info("🚀 Starting ML vs Classical Models Benchmark")
    
    # Create benchmark instance
    benchmark = MLvsClassicalBenchmark()
    
    try:
        # Generate benchmark data
        logger.info("=" * 60)
        X, y, metadata = benchmark.generate_benchmark_data(n_samples=50, sequence_lengths=[100, 250, 500])
        
        # Train ML models
        logger.info("=" * 60)
        trained_ml_models = benchmark.train_ml_models(X, y)
        
        # Run benchmark
        logger.info("=" * 60)
        results = benchmark.run_benchmark(X, y, metadata, trained_ml_models)
        
        # Analyze results
        logger.info("=" * 60)
        analysis = benchmark.analyze_results(results)
        
        # Create visualizations
        logger.info("=" * 60)
        benchmark.create_visualizations(results, analysis)
        
        # Save results
        logger.info("=" * 60)
        benchmark.save_results(results, analysis)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("🎉 ML vs Classical Benchmark Completed!")
        
        # Print key findings
        overall = analysis.get('overall', {})
        by_type = analysis.get('by_type', {})
        
        logger.info("📊 Key Findings:")
        logger.info(f"  Total evaluations: {overall.get('n_total', 0)}")
        logger.info(f"  Overall success rate: {overall.get('success_rate', 0)*100:.1f}%")
        logger.info(f"  Overall MAE: {overall.get('mean_absolute_error', 0):.4f}")
        logger.info(f"  Overall execution time: {overall.get('mean_execution_time', 0):.4f}s")
        
        if 'classical' in by_type and 'ml' in by_type:
            classical_success = by_type['classical'].get('success_rate', 0) * 100
            ml_success = by_type['ml'].get('success_rate', 0) * 100
            classical_mae = by_type['classical'].get('mean_absolute_error', 0)
            ml_mae = by_type['ml'].get('mean_absolute_error', 0)
            
            logger.info(f"  Classical success rate: {classical_success:.1f}%")
            logger.info(f"  ML success rate: {ml_success:.1f}%")
            logger.info(f"  Classical MAE: {classical_mae:.4f}")
            logger.info(f"  ML MAE: {ml_mae:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
