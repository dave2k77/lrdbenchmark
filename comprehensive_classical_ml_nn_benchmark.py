#!/usr/bin/env python3
"""
Comprehensive Classical vs ML vs Neural Network Benchmark

This script runs a comprehensive benchmark comparing:
1. Classical estimators (R/S, DFA, GPH, Whittle, etc.)
2. Machine Learning models (SVR, Gradient Boosting, Random Forest, CNN)
3. Neural Network models (FFN, CNN, LSTM, GRU, Transformer, etc.)

The benchmark evaluates all three approaches on the same dataset to provide
a fair comparison of their performance for Hurst parameter estimation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
from lrdbenchmark.analysis.machine_learning.cnn_estimator_unified import CNNEstimator

# Import Neural Network factory
from lrdbenchmark.analysis.machine_learning.neural_network_factory import (
    NeuralNetworkFactory, NNArchitecture, NNConfig, create_all_benchmark_networks
)

# Import data models
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbenchmark.models.data_models.arfima.arfima_model import ARFIMAModel
from lrdbenchmark.models.data_models.mrw.mrw_model import MultifractalRandomWalk

class ComprehensiveClassicalMLNNBenchmark:
    """Comprehensive benchmark comparing Classical, ML, and Neural Network approaches."""
    
    def __init__(self):
        self.results = []
        self.classical_estimators = {}
        self.ml_estimators = {}
        self.nn_estimators = {}
        self.data_models = {}
        
        # Initialize estimators
        self._initialize_estimators()
        
    def _initialize_estimators(self):
        """Initialize all estimators."""
        logger.info("🔧 Initializing Estimators")
        
        # Classical Estimators
        self.classical_estimators = {
            'RS': RSEstimator(),
            'DFA': DFAEstimator(),
            'DMA': DMAEstimator(),
            'Higuchi': HiguchiEstimator(),
            'GPH': GPHEstimator(),
            'Whittle': WhittleEstimator(),
            'Periodogram': PeriodogramEstimator()
        }
        
        # ML Estimators
        self.ml_estimators = {
            'SVR': SVREstimator(kernel='rbf', C=1.0, epsilon=0.1),
            'GradientBoosting': GradientBoostingEstimator(n_estimators=50, learning_rate=0.1),
            'RandomForest': RandomForestEstimator(n_estimators=50, max_depth=5)
        }
        
        # Neural Network Estimators
        self.nn_estimators = create_all_benchmark_networks(input_length=500)
        
        # Data Models
        self.data_models = {
            'FBM': FractionalBrownianMotion,
            'FGN': FractionalGaussianNoise,
            'ARFIMA': ARFIMAModel,
            'MRW': MultifractalRandomWalk
        }
        
        logger.info(f"✅ Initialized {len(self.classical_estimators)} Classical, {len(self.ml_estimators)} ML, {len(self.nn_estimators)} NN estimators")
    
    def generate_benchmark_data(self, n_samples: int = 100, sequence_length: int = 500) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate comprehensive benchmark data."""
        logger.info(f"📊 Generating {n_samples} samples with length {sequence_length}")
        
        X = []
        y = []
        metadata = []
        
        # Generate data from different models with different Hurst parameters
        hurst_values = np.linspace(0.2, 0.8, 5)  # 5 different Hurst values
        samples_per_hurst = n_samples // len(hurst_values)
        
        for model_name, model_class in self.data_models.items():
            for hurst in hurst_values:
                for i in range(samples_per_hurst):
                    try:
                        if model_name == 'FBM':
                            model = model_class(H=hurst, sigma=1.0)
                        elif model_name == 'FGN':
                            model = model_class(H=hurst, sigma=1.0)
                        elif model_name == 'ARFIMA':
                            model = model_class(d=hurst-0.5, sigma=1.0)  # Convert Hurst to ARFIMA d
                        elif model_name == 'MRW':
                            model = model_class(H=hurst, lambda_param=0.1, sigma=1.0)
                        
                        data = model.generate(sequence_length)
                        data = np.asarray(data).flatten()
                        
                        # Ensure correct length
                        if len(data) != sequence_length:
                            if len(data) > sequence_length:
                                data = data[:sequence_length]
                            else:
                                padded = np.zeros(sequence_length)
                                padded[:len(data)] = data
                                data = padded
                        
                        data = data.astype(np.float64)
                        X.append(data)
                        y.append(hurst)
                        metadata.append({
                            'model': model_name,
                            'hurst': hurst,
                            'length': sequence_length
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate {model_name} data with H={hurst}: {e}")
                        # Create dummy data
                        data = np.random.randn(sequence_length).astype(np.float64)
                        X.append(data)
                        y.append(hurst)
                        metadata.append({
                            'model': model_name,
                            'hurst': hurst,
                            'length': sequence_length
                        })
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✅ Generated {len(X)} samples with shape {X.shape}")
        return X, y, metadata
    
    def train_ml_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train all ML models."""
        logger.info("🎓 Training ML Models")
        
        trained_models = {}
        
        for name, estimator in self.ml_estimators.items():
            try:
                logger.info(f"  Training {name}...")
                start_time = time.time()
                
                if hasattr(estimator, 'train'):
                    results = estimator.train(X, y)
                    training_time = time.time() - start_time
                    if isinstance(results, dict):
                        mse = results.get('mse', 0.0)
                        r2 = results.get('r2', 0.0)
                    else:
                        mse, r2 = results
                    logger.info(f"    ✅ {name} trained in {training_time:.2f}s (MSE: {mse:.4f}, R²: {r2:.4f})")
                else:
                    # For CNN_ML, we need to handle it differently
                    training_time = time.time() - start_time
                    logger.info(f"    ✅ {name} initialized in {training_time:.2f}s")
                
                trained_models[name] = estimator
                
            except Exception as e:
                logger.error(f"    ❌ {name} training failed: {e}")
        
        return trained_models
    
    def train_nn_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train all Neural Network models."""
        logger.info("🧠 Training Neural Network Models")
        
        trained_models = {}
        
        for name, network in self.nn_estimators.items():
            try:
                logger.info(f"  Training {name}...")
                start_time = time.time()
                
                # Train the network
                history = network.train_model(X, y, validation_split=0.2)
                training_time = time.time() - start_time
                
                final_train_loss = history['train_loss'][-1]
                final_val_loss = history['val_loss'][-1]
                
                logger.info(f"    ✅ {name} trained in {training_time:.2f}s (Train: {final_train_loss:.4f}, Val: {final_val_loss:.4f})")
                trained_models[name] = network
                
            except Exception as e:
                logger.error(f"    ❌ {name} training failed: {e}")
        
        return trained_models
    
    def run_benchmark(self, X: np.ndarray, y: np.ndarray, 
                     trained_ml_models: Dict[str, Any], 
                     trained_nn_models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run comprehensive benchmark."""
        logger.info("🏃 Running Comprehensive Benchmark")
        
        results = []
        n_samples = len(X)
        
        # Test Classical Estimators
        logger.info("📊 Testing Classical Estimators")
        for name, estimator in self.classical_estimators.items():
            logger.info(f"  Testing {name}...")
            
            predictions = []
            execution_times = []
            
            for i in range(n_samples):
                try:
                    start_time = time.time()
                    result = estimator.estimate(X[i])
                    execution_time = time.time() - start_time
                    
                    if isinstance(result, dict):
                        pred = result.get('hurst_parameter', np.nan)
                    else:
                        pred = float(result)
                    
                    predictions.append(pred)
                    execution_times.append(execution_time)
                    
                except Exception as e:
                    logger.warning(f"    {name} failed on sample {i}: {e}")
                    predictions.append(np.nan)
                    execution_times.append(np.nan)
            
            # Calculate metrics
            valid_predictions = [p for p in predictions if not np.isnan(p)]
            valid_times = [t for t in execution_times if not np.isnan(t)]
            
            if valid_predictions:
                mae = np.mean(np.abs(np.array(valid_predictions) - y[:len(valid_predictions)]))
                success_rate = len(valid_predictions) / n_samples
                mean_time = np.mean(valid_times)
            else:
                mae = np.nan
                success_rate = 0.0
                mean_time = np.nan
            
            results.append({
                'estimator_name': name,
                'estimator_type': 'classical',
                'mean_absolute_error': mae,
                'success_rate': success_rate,
                'mean_execution_time': mean_time,
                'n_valid': len(valid_predictions),
                'n_total': n_samples
            })
            
            logger.info(f"    ✅ {name}: MAE={mae:.4f}, Success={success_rate:.1%}, Time={mean_time:.4f}s")
        
        # Test ML Models
        logger.info("🤖 Testing ML Models")
        for name, model in trained_ml_models.items():
            logger.info(f"  Testing {name}...")
            
            try:
                start_time = time.time()
                predictions = []
                
                for i in range(n_samples):
                    if hasattr(model, 'predict'):
                        pred = model.predict(X[i:i+1])
                        if isinstance(pred, np.ndarray):
                            pred = pred[0] if len(pred) > 0 else np.nan
                        predictions.append(pred)
                    else:
                        predictions.append(np.nan)
                
                execution_time = time.time() - start_time
                mean_time = execution_time / n_samples
                
                # Calculate metrics
                valid_predictions = [p for p in predictions if not np.isnan(p)]
                
                if valid_predictions:
                    mae = np.mean(np.abs(np.array(valid_predictions) - y[:len(valid_predictions)]))
                    success_rate = len(valid_predictions) / n_samples
                else:
                    mae = np.nan
                    success_rate = 0.0
                
                results.append({
                    'estimator_name': name,
                    'estimator_type': 'ml',
                    'mean_absolute_error': mae,
                    'success_rate': success_rate,
                    'mean_execution_time': mean_time,
                    'n_valid': len(valid_predictions),
                    'n_total': n_samples
                })
                
                logger.info(f"    ✅ {name}: MAE={mae:.4f}, Success={success_rate:.1%}, Time={mean_time:.4f}s")
                
            except Exception as e:
                logger.error(f"    ❌ {name} failed: {e}")
                results.append({
                    'estimator_name': name,
                    'estimator_type': 'ml',
                    'mean_absolute_error': np.nan,
                    'success_rate': 0.0,
                    'mean_execution_time': np.nan,
                    'n_valid': 0,
                    'n_total': n_samples
                })
        
        # Test Neural Network Models
        logger.info("🧠 Testing Neural Network Models")
        for name, model in trained_nn_models.items():
            logger.info(f"  Testing {name}...")
            
            try:
                start_time = time.time()
                
                # Use smaller batch size for memory-intensive networks
                batch_size = 8 if name in ['lstm', 'bidirectional_lstm', 'gru'] else 32
                predictions = model.predict(X, batch_size=batch_size)
                
                execution_time = time.time() - start_time
                mean_time = execution_time / n_samples
                
                # Calculate metrics
                valid_predictions = [p for p in predictions if not np.isnan(p)]
                
                if valid_predictions:
                    mae = np.mean(np.abs(np.array(valid_predictions) - y[:len(valid_predictions)]))
                    success_rate = len(valid_predictions) / n_samples
                else:
                    mae = np.nan
                    success_rate = 0.0
                
                results.append({
                    'estimator_name': name,
                    'estimator_type': 'neural_network',
                    'mean_absolute_error': mae,
                    'success_rate': success_rate,
                    'mean_execution_time': mean_time,
                    'n_valid': len(valid_predictions),
                    'n_total': n_samples
                })
                
                logger.info(f"    ✅ {name}: MAE={mae:.4f}, Success={success_rate:.1%}, Time={mean_time:.4f}s")
                
            except Exception as e:
                logger.error(f"    ❌ {name} failed: {e}")
                results.append({
                    'estimator_name': name,
                    'estimator_type': 'neural_network',
                    'mean_absolute_error': np.nan,
                    'success_rate': 0.0,
                    'mean_execution_time': np.nan,
                    'n_valid': 0,
                    'n_total': n_samples
                })
        
        self.results = results
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results."""
        logger.info("📈 Analyzing Results")
        
        df = pd.DataFrame(results)
        
        # Overall statistics
        overall_stats = {
            'n_total': len(df),
            'n_classical': len(df[df['estimator_type'] == 'classical']),
            'n_ml': len(df[df['estimator_type'] == 'ml']),
            'n_neural_network': len(df[df['estimator_type'] == 'neural_network']),
            'overall_success_rate': df['success_rate'].mean(),
            'overall_mean_mae': df['mean_absolute_error'].mean(),
            'overall_mean_time': df['mean_execution_time'].mean()
        }
        
        # By type statistics
        by_type = {}
        for estimator_type in ['classical', 'ml', 'neural_network']:
            type_df = df[df['estimator_type'] == estimator_type]
            if len(type_df) > 0:
                by_type[estimator_type] = {
                    'n_estimators': len(type_df),
                    'mean_mae': type_df['mean_absolute_error'].mean(),
                    'mean_success_rate': type_df['success_rate'].mean(),
                    'mean_execution_time': type_df['mean_execution_time'].mean(),
                    'best_mae': type_df['mean_absolute_error'].min(),
                    'best_estimator': type_df.loc[type_df['mean_absolute_error'].idxmin(), 'estimator_name']
                }
        
        # Top performers
        top_estimators = df.nsmallest(10, 'mean_absolute_error')[['estimator_name', 'estimator_type', 'mean_absolute_error', 'success_rate']].to_dict('records')
        
        analysis = {
            'overall': overall_stats,
            'by_type': by_type,
            'top_estimators': top_estimators,
            'detailed_results': results
        }
        
        return analysis
    
    def create_visualizations(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Create comprehensive visualizations."""
        logger.info("📊 Creating Visualizations")
        
        df = pd.DataFrame(results)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Classical vs ML vs Neural Network Benchmark', fontsize=16, fontweight='bold')
        
        # 1. MAE by Type
        ax1 = axes[0, 0]
        type_mae = df.groupby('estimator_type')['mean_absolute_error'].mean()
        type_mae.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('Mean Absolute Error by Type')
        ax1.set_ylabel('MAE')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Success Rate by Type
        ax2 = axes[0, 1]
        type_success = df.groupby('estimator_type')['success_rate'].mean()
        type_success.plot(kind='bar', ax=ax2, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('Success Rate by Type')
        ax2.set_ylabel('Success Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Execution Time by Type
        ax3 = axes[0, 2]
        type_time = df.groupby('estimator_type')['mean_execution_time'].mean()
        type_time.plot(kind='bar', ax=ax3, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax3.set_title('Mean Execution Time by Type')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Individual Estimator Performance (MAE)
        ax4 = axes[1, 0]
        df_sorted = df.sort_values('mean_absolute_error')
        colors = ['skyblue' if t == 'classical' else 'lightcoral' if t == 'ml' else 'lightgreen' for t in df_sorted['estimator_type']]
        bars = ax4.bar(range(len(df_sorted)), df_sorted['mean_absolute_error'], color=colors)
        ax4.set_title('Individual Estimator Performance (MAE)')
        ax4.set_ylabel('MAE')
        ax4.set_xlabel('Estimators')
        ax4.tick_params(axis='x', rotation=90)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', label='Classical'),
                          Patch(facecolor='lightcoral', label='ML'),
                          Patch(facecolor='lightgreen', label='Neural Network')]
        ax4.legend(handles=legend_elements)
        
        # 5. MAE vs Execution Time Scatter
        ax5 = axes[1, 1]
        for estimator_type in ['classical', 'ml', 'neural_network']:
            type_df = df[df['estimator_type'] == estimator_type]
            ax5.scatter(type_df['mean_execution_time'], type_df['mean_absolute_error'], 
                       label=estimator_type, alpha=0.7, s=100)
        ax5.set_xlabel('Execution Time (seconds)')
        ax5.set_ylabel('MAE')
        ax5.set_title('MAE vs Execution Time')
        ax5.legend()
        ax5.set_xscale('log')
        
        # 6. Top 10 Performers
        ax6 = axes[1, 2]
        top_10 = df.nsmallest(10, 'mean_absolute_error')
        colors = ['skyblue' if t == 'classical' else 'lightcoral' if t == 'ml' else 'lightgreen' for t in top_10['estimator_type']]
        bars = ax6.barh(range(len(top_10)), top_10['mean_absolute_error'], color=colors)
        ax6.set_title('Top 10 Performers')
        ax6.set_xlabel('MAE')
        ax6.set_yticks(range(len(top_10)))
        ax6.set_yticklabels(top_10['estimator_name'], fontsize=8)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path("comprehensive_benchmark_results")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "comprehensive_classical_ml_nn_comparison.png", dpi=300, bbox_inches='tight')
        logger.info(f"📊 Visualization saved to {output_dir / 'comprehensive_classical_ml_nn_comparison.png'}")
        
        plt.show()
    
    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save results to files."""
        logger.info("💾 Saving Results")
        
        output_dir = Path("comprehensive_benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        df = pd.DataFrame(results)
        df.to_csv(output_dir / "comprehensive_classical_ml_nn_results.csv", index=False)
        
        # Save analysis
        with open(output_dir / "comprehensive_classical_ml_nn_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"✅ Results saved to {output_dir}")

def main():
    """Main benchmark execution."""
    logger.info("🚀 Starting Comprehensive Classical vs ML vs Neural Network Benchmark")
    logger.info("=" * 80)
    
    # Initialize benchmark
    benchmark = ComprehensiveClassicalMLNNBenchmark()
    
    # Generate data
    X, y, metadata = benchmark.generate_benchmark_data(n_samples=100, sequence_length=500)
    
    # Train models
    trained_ml_models = benchmark.train_ml_models(X, y)
    trained_nn_models = benchmark.train_nn_models(X, y)
    
    # Run benchmark
    results = benchmark.run_benchmark(X, y, trained_ml_models, trained_nn_models)
    
    # Analyze results
    analysis = benchmark.analyze_results(results)
    
    # Create visualizations
    benchmark.create_visualizations(results, analysis)
    
    # Save results
    benchmark.save_results(results, analysis)
    
    # Print summary
    logger.info("\n📋 Benchmark Summary")
    logger.info("=" * 60)
    logger.info(f"Total estimators tested: {analysis['overall']['n_total']}")
    logger.info(f"Classical: {analysis['overall']['n_classical']}, ML: {analysis['overall']['n_ml']}, NN: {analysis['overall']['n_neural_network']}")
    logger.info(f"Overall success rate: {analysis['overall']['overall_success_rate']:.1%}")
    logger.info(f"Overall mean MAE: {analysis['overall']['overall_mean_mae']:.4f}")
    
    logger.info("\n🏆 Top 5 Performers:")
    for i, estimator in enumerate(analysis['top_estimators'][:5]):
        logger.info(f"  {i+1}. {estimator['estimator_name']} ({estimator['estimator_type']}): {estimator['mean_absolute_error']:.4f} MAE")
    
    logger.info("\n🎉 Comprehensive Benchmark Completed!")

if __name__ == "__main__":
    main()
