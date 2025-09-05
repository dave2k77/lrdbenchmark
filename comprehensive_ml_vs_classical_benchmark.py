#!/usr/bin/env python3
"""
Comprehensive ML vs Classical Benchmark.

This script includes ALL available ML models and compares them against classical methods.
"""

import numpy as np
import time
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings
from datetime import datetime
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import classical estimators
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.spectral.gph.gph_estimator_unified import GPHEstimator
from lrdbenchmark.analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator

# Import ALL ML models
from lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified import RandomForestEstimator
from lrdbenchmark.analysis.machine_learning.svr_estimator_unified import SVREstimator
from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator_unified import GradientBoostingEstimator
from lrdbenchmark.analysis.machine_learning.cnn_estimator_unified import CNNEstimator
from lrdbenchmark.analysis.machine_learning.lstm_estimator_unified import LSTMEstimator
from lrdbenchmark.analysis.machine_learning.gru_estimator_unified import GRUEstimator
from lrdbenchmark.analysis.machine_learning.transformer_estimator_unified import TransformerEstimator

# Import production system for deep learning models
from lrdbenchmark.analysis.machine_learning.production_ml_system import (
    ProductionMLSystem, ProductionConfig
)

class ComprehensiveMLvsClassicalBenchmark:
    """Comprehensive benchmark including ALL ML models vs classical methods."""
    
    def __init__(self, output_dir: str = "comprehensive_ml_benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize classical estimators
        self.classical_estimators = {
            'RS': RSEstimator(use_optimization='numpy'),
            'DFA': DFAEstimator(use_optimization='numpy'),
            'GPH': GPHEstimator(use_optimization='numpy'),
            'Whittle': WhittleEstimator(use_optimization='numpy')
        }
        
        # Initialize ALL ML estimators
        self.ml_estimators = {
            'RandomForest': RandomForestEstimator(use_optimization='numpy'),
            'SVR': SVREstimator(use_optimization='numpy'),
            'GradientBoosting': GradientBoostingEstimator(use_optimization='numpy'),
            'CNN': CNNEstimator(use_optimization='numpy'),
            'LSTM': LSTMEstimator(use_optimization='numpy'),
            'GRU': GRUEstimator(use_optimization='numpy'),
            'Transformer': TransformerEstimator(use_optimization='numpy')
        }
        
        # Results storage
        self.results = []
        
    def generate_synthetic_data(self, n_samples: int = 100, seq_len: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic time series data with known Hurst parameters."""
        logger.info(f"Generating {n_samples} synthetic samples of length {seq_len}")
        
        X = []
        y = []
        
        # Generate data with different Hurst parameters
        hurst_values = np.linspace(0.2, 0.8, 10)
        
        for hurst in hurst_values:
            for _ in range(n_samples // len(hurst_values)):
                # Generate fractional Brownian motion-like data
                t = np.linspace(0, 1, seq_len)
                dt = t[1] - t[0]
                
                # Generate increments with Hurst scaling
                dB = np.random.normal(0, np.sqrt(dt), seq_len)
                fbm = np.cumsum(dB) * (dt ** hurst)
                
                # Add some noise
                noise = np.random.normal(0, 0.1, seq_len)
                data = fbm + noise
                
                X.append(data)
                y.append(hurst)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Generated {len(X)} samples with shape {X.shape}")
        return X, y
    
    def train_production_ml_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train production ML models (CNN, Transformer)."""
        logger.info("Training production ML models...")
        
        trained_models = {}
        
        # Train CNN
        try:
            logger.info("  Training CNN...")
            config = ProductionConfig(
                model_type="cnn",
                input_length=X.shape[1],
                hidden_dims=[64, 32],
                dropout_rate=0.2,
                learning_rate=0.001,
                batch_size=32,
                epochs=20,
                early_stopping_patience=5,
                validation_split=0.2,
                framework_priority=['torch']
            )
            
            system = ProductionMLSystem(config)
            start_time = time.time()
            training_result = system.train(X, y)
            training_time = time.time() - start_time
            
            trained_models['CNN_Production'] = {
                'system': system,
                'training_time': training_time,
                'training_result': training_result,
                'type': 'production'
            }
            logger.info(f"    ✅ CNN (Production) trained in {training_time:.2f}s")
            
        except Exception as e:
            logger.error(f"    ❌ CNN (Production) training failed: {e}")
        
        # Train Transformer
        try:
            logger.info("  Training Transformer...")
            config = ProductionConfig(
                model_type="transformer",
                input_length=X.shape[1],
                hidden_dims=[64, 32],
                dropout_rate=0.2,
                learning_rate=0.001,
                batch_size=32,
                epochs=20,
                early_stopping_patience=5,
                validation_split=0.2,
                framework_priority=['torch']
            )
            
            system = ProductionMLSystem(config)
            start_time = time.time()
            training_result = system.train(X, y)
            training_time = time.time() - start_time
            
            trained_models['Transformer_Production'] = {
                'system': system,
                'training_time': training_time,
                'training_result': training_result,
                'type': 'production'
            }
            logger.info(f"    ✅ Transformer (Production) trained in {training_time:.2f}s")
            
        except Exception as e:
            logger.error(f"    ❌ Transformer (Production) training failed: {e}")
        
        return trained_models
    
    def run_benchmark(self, X: np.ndarray, y: np.ndarray, trained_production_models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run comprehensive benchmark including ALL models."""
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
                        'method': result.get('method', name),
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
                        'method': name,
                        'success': False,
                        'error': str(e)
                    })
        
        # Benchmark traditional ML estimators
        logger.info("Benchmarking traditional ML estimators...")
        for name, estimator in self.ml_estimators.items():
            logger.info(f"  Testing {name}...")
            
            for i in range(n_samples):
                try:
                    start_time = time.time()
                    result = estimator.estimate(X[i])
                    execution_time = time.time() - start_time
                    
                    results.append({
                        'estimator': name,
                        'estimator_type': 'ml_traditional',
                        'sample_id': i,
                        'true_hurst': y[i],
                        'estimated_hurst': result.get('hurst_parameter', 0.5),
                        'execution_time': execution_time,
                        'r_squared': result.get('r_squared', 0.0),
                        'method': result.get('method', f"{name}_ML"),
                        'success': True,
                        'error': None
                    })
                    
                except Exception as e:
                    results.append({
                        'estimator': name,
                        'estimator_type': 'ml_traditional',
                        'sample_id': i,
                        'true_hurst': y[i],
                        'estimated_hurst': 0.5,
                        'execution_time': 0.0,
                        'r_squared': 0.0,
                        'method': f"{name}_ML",
                        'success': False,
                        'error': str(e)
                    })
        
        # Benchmark production ML models
        logger.info("Benchmarking production ML models...")
        for name, model_info in trained_production_models.items():
            logger.info(f"  Testing {name}...")
            
            for i in range(n_samples):
                try:
                    start_time = time.time()
                    prediction = model_info['system'].predict(X[i])
                    execution_time = time.time() - start_time
                    
                    results.append({
                        'estimator': name,
                        'estimator_type': 'ml_production',
                        'sample_id': i,
                        'true_hurst': y[i],
                        'estimated_hurst': prediction.hurst_parameter,
                        'execution_time': execution_time,
                        'r_squared': 0.0,
                        'method': f"{name}_Production",
                        'success': True,
                        'error': None
                    })
                    
                except Exception as e:
                    results.append({
                        'estimator': name,
                        'estimator_type': 'ml_production',
                        'sample_id': i,
                        'true_hurst': y[i],
                        'estimated_hurst': 0.5,
                        'execution_time': 0.0,
                        'r_squared': 0.0,
                        'method': f"{name}_Production",
                        'success': False,
                        'error': str(e)
                    })
        
        logger.info(f"Benchmark completed: {len(results)} total evaluations")
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comprehensive benchmark results."""
        logger.info("Analyzing comprehensive results...")
        
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        # Calculate performance metrics
        analysis = {}
        
        # Overall performance
        analysis['overall'] = self._calculate_metrics(df)
        
        # Performance by estimator type
        analysis['by_type'] = {}
        for estimator_type in df['estimator_type'].unique():
            type_df = df[df['estimator_type'] == estimator_type]
            analysis['by_type'][estimator_type] = self._calculate_metrics(type_df)
        
        # Performance by individual estimator
        analysis['by_estimator'] = {}
        for estimator in df['estimator'].unique():
            estimator_df = df[df['estimator'] == estimator]
            analysis['by_estimator'][estimator] = self._calculate_metrics(estimator_df)
        
        return analysis
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if len(df) == 0:
            return {}
        
        successful_df = df[df['success'] == True]
        
        if len(successful_df) == 0:
            return {
                'success_rate': 0.0,
                'n_total': len(df),
                'n_successful': 0
            }
        
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
            'correlation': float(np.corrcoef(successful_df['true_hurst'], successful_df['estimated_hurst'])[0, 1])
        }
    
    def create_visualizations(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Create comprehensive visualizations."""
        logger.info("Creating comprehensive visualizations...")
        
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Create figure with more subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Comprehensive ML vs Classical Models Comparison', fontsize=16, fontweight='bold')
        
        # 1. Success rate by estimator type
        ax1 = axes[0, 0]
        by_type = analysis.get('by_type', {})
        types = list(by_type.keys())
        success_rates = [by_type[t].get('success_rate', 0) * 100 for t in types]
        
        bars = ax1.bar(types, success_rates, color=['#2E8B57', '#FF6B6B', '#4169E1'])
        ax1.set_title('Success Rate by Estimator Type')
        ax1.set_ylabel('Success Rate (%)')
        ax1.tick_params(axis='x', rotation=45)
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 2. Mean Absolute Error by estimator type
        ax2 = axes[0, 1]
        maes = [by_type[t].get('mean_absolute_error', 0) for t in types]
        
        bars = ax2.bar(types, maes, color=['#2E8B57', '#FF6B6B', '#4169E1'])
        ax2.set_title('Mean Absolute Error by Estimator Type')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.tick_params(axis='x', rotation=45)
        for bar, mae in zip(bars, maes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{mae:.4f}', ha='center', va='bottom')
        
        # 3. Top 10 individual estimators by MAE
        ax3 = axes[0, 2]
        by_estimator = analysis.get('by_estimator', {})
        estimators = sorted(by_estimator.items(), 
                          key=lambda x: x[1].get('mean_absolute_error', 0))[:10]
        
        names = [e[0] for e in estimators]
        maes = [e[1].get('mean_absolute_error', 0) for e in estimators]
        
        bars = ax3.barh(names, maes, color='#FF6347')
        ax3.set_xlabel('Mean Absolute Error')
        ax3.set_title('Top 10 Estimators by Accuracy')
        
        # 4. Execution time comparison
        ax4 = axes[1, 0]
        successful_df = df[df['success'] == True]
        if len(successful_df) > 0:
            classical_times = successful_df[successful_df['estimator_type'] == 'classical']['execution_time']
            ml_traditional_times = successful_df[successful_df['estimator_type'] == 'ml_traditional']['execution_time']
            ml_production_times = successful_df[successful_df['estimator_type'] == 'ml_production']['execution_time']
            
            data = [classical_times, ml_traditional_times, ml_production_times]
            labels = ['Classical', 'ML Traditional', 'ML Production']
            
            bp = ax4.boxplot(data, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('#2E8B57')
            bp['boxes'][1].set_facecolor('#FF6B6B')
            bp['boxes'][2].set_facecolor('#4169E1')
            
            ax4.set_title('Execution Time Comparison')
            ax4.set_ylabel('Execution Time (seconds)')
            ax4.set_yscale('log')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. True vs Estimated scatter plot
        ax5 = axes[1, 1]
        if len(successful_df) > 0:
            classical_df = successful_df[successful_df['estimator_type'] == 'classical']
            ml_traditional_df = successful_df[successful_df['estimator_type'] == 'ml_traditional']
            ml_production_df = successful_df[successful_df['estimator_type'] == 'ml_production']
            
            if len(classical_df) > 0:
                ax5.scatter(classical_df['true_hurst'], classical_df['estimated_hurst'], 
                           alpha=0.6, label='Classical', color='#2E8B57', s=20)
            
            if len(ml_traditional_df) > 0:
                ax5.scatter(ml_traditional_df['true_hurst'], ml_traditional_df['estimated_hurst'], 
                           alpha=0.6, label='ML Traditional', color='#FF6B6B', s=20)
            
            if len(ml_production_df) > 0:
                ax5.scatter(ml_production_df['true_hurst'], ml_production_df['estimated_hurst'], 
                           alpha=0.6, label='ML Production', color='#4169E1', s=20)
            
            # Perfect prediction line
            min_val = min(successful_df['true_hurst'].min(), successful_df['estimated_hurst'].min())
            max_val = max(successful_df['true_hurst'].max(), successful_df['estimated_hurst'].max())
            ax5.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect')
            
            ax5.set_xlabel('True Hurst Parameter')
            ax5.set_ylabel('Estimated Hurst Parameter')
            ax5.set_title('True vs Estimated Hurst Parameters')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Error distribution
        ax6 = axes[1, 2]
        if len(successful_df) > 0:
            errors = np.abs(successful_df['estimated_hurst'] - successful_df['true_hurst'])
            ax6.hist(errors, bins=30, alpha=0.7, color='#FF6347', edgecolor='black')
            ax6.set_xlabel('Absolute Error')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Distribution of Absolute Errors')
            ax6.axvline(np.mean(errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(errors):.3f}')
            ax6.legend()
        
        # 7. Performance by ML model type
        ax7 = axes[2, 0]
        ml_estimators = [name for name in by_estimator.keys() if 'ML' in name or 'Production' in name]
        ml_maes = [by_estimator[name].get('mean_absolute_error', 0) for name in ml_estimators]
        
        if ml_estimators:
            bars = ax7.bar(ml_estimators, ml_maes, color='#9B59B6')
            ax7.set_title('ML Models Performance')
            ax7.set_ylabel('Mean Absolute Error')
            ax7.tick_params(axis='x', rotation=45)
            for bar, mae in zip(bars, ml_maes):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{mae:.4f}', ha='center', va='bottom')
        
        # 8. Success rate by individual estimator
        ax8 = axes[2, 1]
        success_rates = [by_estimator[e].get('success_rate', 0) * 100 for e in estimators]
        
        bars = ax8.barh(names, success_rates, color='#2ECC71')
        ax8.set_xlabel('Success Rate (%)')
        ax8.set_title('Success Rate by Estimator')
        
        # 9. Summary statistics
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # Create summary text
        overall = analysis.get('overall', {})
        summary_text = f"""
        COMPREHENSIVE BENCHMARK SUMMARY
        
        Total Evaluations: {overall.get('n_total', 0)}
        Overall Success Rate: {overall.get('success_rate', 0)*100:.1f}%
        Overall MAE: {overall.get('mean_absolute_error', 0):.4f}
        Overall Execution Time: {overall.get('mean_execution_time', 0):.4f}s
        
        Estimator Types:
        • Classical: {len([t for t in types if 'classical' in t])} methods
        • ML Traditional: {len([t for t in types if 'traditional' in t])} methods  
        • ML Production: {len([t for t in types if 'production' in t])} methods
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "comprehensive_ml_vs_classical_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive visualization saved to {output_path}")
    
    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save comprehensive results."""
        logger.info("Saving comprehensive results...")
        
        # Save raw results
        results_path = self.output_dir / "comprehensive_ml_vs_classical_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save analysis
        analysis_path = self.output_dir / "comprehensive_ml_vs_classical_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save CSV
        df = pd.DataFrame(results)
        csv_path = self.output_dir / "comprehensive_ml_vs_classical_results.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Comprehensive results saved to {self.output_dir}")

def main():
    """Main comprehensive benchmark function."""
    logger.info("🚀 Starting Comprehensive ML vs Classical Benchmark")
    
    # Create benchmark instance
    benchmark = ComprehensiveMLvsClassicalBenchmark()
    
    try:
        # Generate data
        logger.info("=" * 60)
        X, y = benchmark.generate_synthetic_data(n_samples=50, seq_len=500)
        
        # Train production ML models
        logger.info("=" * 60)
        trained_production_models = benchmark.train_production_ml_models(X, y)
        
        # Run comprehensive benchmark
        logger.info("=" * 60)
        results = benchmark.run_benchmark(X, y, trained_production_models)
        
        # Analyze results
        logger.info("=" * 60)
        analysis = benchmark.analyze_results(results)
        
        # Create visualizations
        logger.info("=" * 60)
        benchmark.create_visualizations(results, analysis)
        
        # Save results
        logger.info("=" * 60)
        benchmark.save_results(results, analysis)
        
        # Print comprehensive summary
        logger.info("=" * 60)
        logger.info("🎉 Comprehensive ML vs Classical Benchmark Completed!")
        
        # Print key findings
        overall = analysis.get('overall', {})
        by_type = analysis.get('by_type', {})
        
        logger.info("📊 Comprehensive Key Findings:")
        logger.info(f"  Total evaluations: {overall.get('n_total', 0)}")
        logger.info(f"  Overall success rate: {overall.get('success_rate', 0)*100:.1f}%")
        logger.info(f"  Overall MAE: {overall.get('mean_absolute_error', 0):.4f}")
        logger.info(f"  Overall execution time: {overall.get('mean_execution_time', 0):.4f}s")
        
        # Print by type results
        for estimator_type, metrics in by_type.items():
            logger.info(f"  {estimator_type}: {metrics.get('success_rate', 0)*100:.1f}% success, {metrics.get('mean_absolute_error', 0):.4f} MAE")
        
        # Find best performing estimator
        by_estimator = analysis.get('by_estimator', {})
        best_estimator = min(by_estimator.items(), 
                           key=lambda x: x[1].get('mean_absolute_error', float('inf')))
        
        logger.info(f"🏆 Best performing estimator: {best_estimator[0]} with {best_estimator[1].get('mean_absolute_error', 0):.4f} MAE")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
