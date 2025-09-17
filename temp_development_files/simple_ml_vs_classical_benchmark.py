#!/usr/bin/env python3
"""
Simple ML vs Classical Benchmark.

This script performs a focused comparison between ML and classical models
using synthetic data generation.
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

# Import ML models
from lrdbenchmark.analysis.machine_learning.production_ml_system import (
    ProductionMLSystem, ProductionConfig
)

class SimpleMLvsClassicalBenchmark:
    """Simple benchmark comparing ML and classical models."""
    
    def __init__(self, output_dir: str = "simple_benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize classical estimators
        self.classical_estimators = {
            'RS': RSEstimator(use_optimization='numpy'),
            'DFA': DFAEstimator(use_optimization='numpy'),
            'GPH': GPHEstimator(use_optimization='numpy'),
            'Whittle': WhittleEstimator(use_optimization='numpy')
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
    
    def train_ml_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train ML models."""
        logger.info("Training ML models...")
        
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
            
            trained_models['CNN'] = {
                'system': system,
                'training_time': training_time,
                'training_result': training_result
            }
            logger.info(f"    ✅ CNN trained in {training_time:.2f}s")
            
        except Exception as e:
            logger.error(f"    ❌ CNN training failed: {e}")
        
        return trained_models
    
    def run_benchmark(self, X: np.ndarray, y: np.ndarray, trained_ml_models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run benchmark comparison."""
        logger.info("Running benchmark...")
        
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
        
        # Benchmark ML models
        logger.info("Benchmarking ML models...")
        for name, model_info in trained_ml_models.items():
            logger.info(f"  Testing {name}...")
            
            for i in range(n_samples):
                try:
                    start_time = time.time()
                    prediction = model_info['system'].predict(X[i])
                    execution_time = time.time() - start_time
                    
                    results.append({
                        'estimator': name,
                        'estimator_type': 'ml',
                        'sample_id': i,
                        'true_hurst': y[i],
                        'estimated_hurst': prediction.hurst_parameter,
                        'execution_time': execution_time,
                        'r_squared': 0.0,
                        'method': f"{name}_ML",
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
                        'method': f"{name}_ML",
                        'success': False,
                        'error': str(e)
                    })
        
        logger.info(f"Benchmark completed: {len(results)} total evaluations")
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results."""
        logger.info("Analyzing results...")
        
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
        """Create visualizations."""
        logger.info("Creating visualizations...")
        
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ML vs Classical Models Comparison', fontsize=16, fontweight='bold')
        
        # 1. Success rate comparison
        ax1 = axes[0, 0]
        by_type = analysis.get('by_type', {})
        types = list(by_type.keys())
        success_rates = [by_type[t].get('success_rate', 0) * 100 for t in types]
        
        bars = ax1.bar(types, success_rates, color=['#2E8B57', '#DC143C'])
        ax1.set_title('Success Rate by Estimator Type')
        ax1.set_ylabel('Success Rate (%)')
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 2. Mean Absolute Error comparison
        ax2 = axes[0, 1]
        maes = [by_type[t].get('mean_absolute_error', 0) for t in types]
        
        bars = ax2.bar(types, maes, color=['#2E8B57', '#DC143C'])
        ax2.set_title('Mean Absolute Error by Estimator Type')
        ax2.set_ylabel('Mean Absolute Error')
        for bar, mae in zip(bars, maes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{mae:.4f}', ha='center', va='bottom')
        
        # 3. Execution time comparison
        ax3 = axes[1, 0]
        successful_df = df[df['success'] == True]
        if len(successful_df) > 0:
            classical_times = successful_df[successful_df['estimator_type'] == 'classical']['execution_time']
            ml_times = successful_df[successful_df['estimator_type'] == 'ml']['execution_time']
            
            data = [classical_times, ml_times]
            labels = ['Classical', 'ML']
            
            bp = ax3.boxplot(data, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('#2E8B57')
            bp['boxes'][1].set_facecolor('#DC143C')
            
            ax3.set_title('Execution Time Comparison')
            ax3.set_ylabel('Execution Time (seconds)')
            ax3.set_yscale('log')
        
        # 4. True vs Estimated scatter plot
        ax4 = axes[1, 1]
        if len(successful_df) > 0:
            classical_df = successful_df[successful_df['estimator_type'] == 'classical']
            ml_df = successful_df[successful_df['estimator_type'] == 'ml']
            
            if len(classical_df) > 0:
                ax4.scatter(classical_df['true_hurst'], classical_df['estimated_hurst'], 
                           alpha=0.6, label='Classical', color='#2E8B57', s=20)
            
            if len(ml_df) > 0:
                ax4.scatter(ml_df['true_hurst'], ml_df['estimated_hurst'], 
                           alpha=0.6, label='ML', color='#DC143C', s=20)
            
            # Perfect prediction line
            min_val = min(successful_df['true_hurst'].min(), successful_df['estimated_hurst'].min())
            max_val = max(successful_df['true_hurst'].max(), successful_df['estimated_hurst'].max())
            ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect')
            
            ax4.set_xlabel('True Hurst Parameter')
            ax4.set_ylabel('Estimated Hurst Parameter')
            ax4.set_title('True vs Estimated Hurst Parameters')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "ml_vs_classical_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
    
    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save results."""
        logger.info("Saving results...")
        
        # Save raw results
        results_path = self.output_dir / "ml_vs_classical_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save analysis
        analysis_path = self.output_dir / "ml_vs_classical_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save CSV
        df = pd.DataFrame(results)
        csv_path = self.output_dir / "ml_vs_classical_results.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {self.output_dir}")

def main():
    """Main benchmark function."""
    logger.info("🚀 Starting Simple ML vs Classical Benchmark")
    
    # Create benchmark instance
    benchmark = SimpleMLvsClassicalBenchmark()
    
    try:
        # Generate data
        logger.info("=" * 60)
        X, y = benchmark.generate_synthetic_data(n_samples=50, seq_len=500)
        
        # Train ML models
        logger.info("=" * 60)
        trained_ml_models = benchmark.train_ml_models(X, y)
        
        # Run benchmark
        logger.info("=" * 60)
        results = benchmark.run_benchmark(X, y, trained_ml_models)
        
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
        logger.info("🎉 Simple ML vs Classical Benchmark Completed!")
        
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
            
            # Determine winner
            if ml_success > classical_success and ml_mae < classical_mae:
                logger.info("🏆 ML models outperform classical models!")
            elif classical_success > ml_success and classical_mae < ml_mae:
                logger.info("🏆 Classical models outperform ML models!")
            else:
                logger.info("🤝 Mixed results - both approaches have strengths!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
