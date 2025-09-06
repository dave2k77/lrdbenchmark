#!/usr/bin/env python3
"""
Test script for the Neural Network Factory

This script demonstrates the creation and usage of various neural network
architectures for Hurst parameter estimation benchmarking.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path

# Add the project root to the path
import sys
sys.path.append(str(Path(__file__).parent))

from lrdbenchmark.analysis.machine_learning.neural_network_factory import (
    NeuralNetworkFactory, NNArchitecture, NNConfig,
    create_feedforward_network, create_cnn_network, create_lstm_network,
    create_all_benchmark_networks
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples: int = 100, sequence_length: int = 500) -> tuple:
    """Generate synthetic time series data with known Hurst parameters."""
    np.random.seed(42)
    
    X = []
    y = []
    
    for i in range(n_samples):
        # Generate random Hurst parameter
        hurst = np.random.uniform(0.2, 0.8)
        
        # Generate fractional Brownian motion-like data using a simpler approach
        # Create correlated noise with long-range dependence
        noise = np.random.randn(sequence_length)
        
        # Apply fractional differencing to create long-range dependence
        # This is a simplified version of fractional differencing
        alpha = 2 * hurst - 1  # Convert Hurst to fractional differencing parameter
        
        # Create fractional differenced series
        if abs(alpha) < 0.01:  # Near white noise
            data = noise
        else:
            # Simple fractional differencing approximation
            data = np.zeros(sequence_length)
            for t in range(sequence_length):
                data[t] = noise[t]
                for j in range(1, min(t + 1, 50)):  # Limit to 50 lags for efficiency
                    if t - j >= 0:
                        data[t] += alpha * data[t - j] / j
        
        # Normalize the data
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        
        X.append(data)
        y.append(hurst)
    
    return np.array(X), np.array(y)

def test_individual_networks():
    """Test individual network creation and basic functionality."""
    logger.info("🧠 Testing Individual Neural Network Creation")
    logger.info("=" * 60)
    
    input_length = 500
    n_samples = 50
    
    # Generate test data
    X, y = generate_synthetic_data(n_samples, input_length)
    logger.info(f"Generated {len(X)} samples with shape {X.shape}")
    
    # Test individual networks
    networks_to_test = [
        ("Feedforward", create_feedforward_network(input_length)),
        ("CNN", create_cnn_network(input_length)),
        ("LSTM", create_lstm_network(input_length))
    ]
    
    results = {}
    
    for name, network in networks_to_test:
        logger.info(f"\n🔧 Testing {name} Network")
        logger.info("-" * 40)
        
        try:
            # Test forward pass
            start_time = time.time()
            predictions = network.predict(X[:5])  # Test on first 5 samples
            prediction_time = time.time() - start_time
            
            logger.info(f"✅ {name} forward pass successful")
            logger.info(f"   Input shape: {X[:5].shape}")
            logger.info(f"   Output shape: {predictions.shape}")
            logger.info(f"   Prediction time: {prediction_time:.4f}s")
            logger.info(f"   Sample predictions: {predictions[:3]}")
            
            results[name] = {
                'status': 'success',
                'prediction_time': prediction_time,
                'sample_predictions': predictions[:3]
            }
            
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
            results[name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results

def test_factory_creation():
    """Test the neural network factory."""
    logger.info("\n🏭 Testing Neural Network Factory")
    logger.info("=" * 60)
    
    input_length = 500
    
    # Test factory creation
    try:
        networks = create_all_benchmark_networks(input_length)
        logger.info(f"✅ Factory created {len(networks)} networks")
        
        for name, network in networks.items():
            logger.info(f"   - {name}: {type(network).__name__}")
        
        return networks
        
    except Exception as e:
        logger.error(f"❌ Factory creation failed: {e}")
        return None

def test_network_training():
    """Test training a subset of networks."""
    logger.info("\n🎓 Testing Network Training")
    logger.info("=" * 60)
    
    input_length = 500
    n_samples = 100
    
    # Generate training data
    X, y = generate_synthetic_data(n_samples, input_length)
    logger.info(f"Generated training data: {X.shape}, {y.shape}")
    
    # Test training on a few networks
    test_architectures = [
        NNArchitecture.FFN,
        NNArchitecture.CNN,
        NNArchitecture.LSTM
    ]
    
    training_results = {}
    
    for arch in test_architectures:
        logger.info(f"\n🔧 Training {arch.value} Network")
        logger.info("-" * 40)
        
        try:
            # Create network
            config = NNConfig(
                architecture=arch,
                input_length=input_length,
                hidden_dims=[32, 16],  # Smaller for faster training
                dropout_rate=0.2,
                learning_rate=0.001,
                epochs=10,  # Fewer epochs for testing
                batch_size=16
            )
            
            network = NeuralNetworkFactory.create_network(config)
            
            # Train network
            start_time = time.time()
            history = network.train_model(X, y, validation_split=0.2)
            training_time = time.time() - start_time
            
            # Test predictions
            predictions = network.predict(X[:10])
            mae = np.mean(np.abs(y[:10] - predictions))
            
            logger.info(f"✅ {arch.value} training successful")
            logger.info(f"   Training time: {training_time:.2f}s")
            logger.info(f"   Final train loss: {history['train_loss'][-1]:.4f}")
            logger.info(f"   Final val loss: {history['val_loss'][-1]:.4f}")
            logger.info(f"   Test MAE: {mae:.4f}")
            
            training_results[arch.value] = {
                'status': 'success',
                'training_time': training_time,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'test_mae': mae
            }
            
        except Exception as e:
            logger.error(f"❌ {arch.value} training failed: {e}")
            training_results[arch.value] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return training_results

def test_architecture_comparison():
    """Compare different architectures on the same task."""
    logger.info("\n📊 Testing Architecture Comparison")
    logger.info("=" * 60)
    
    input_length = 500
    n_samples = 80
    
    # Generate data
    X, y = generate_synthetic_data(n_samples, input_length)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Test architectures
    architectures = [
        NNArchitecture.FFN,
        NNArchitecture.CNN,
        NNArchitecture.LSTM,
        NNArchitecture.GRU
    ]
    
    comparison_results = {}
    
    for arch in architectures:
        logger.info(f"\n🔧 Testing {arch.value}")
        logger.info("-" * 30)
        
        try:
            # Create and train network
            config = NNConfig(
                architecture=arch,
                input_length=input_length,
                hidden_dims=[32, 16],
                dropout_rate=0.2,
                learning_rate=0.001,
                epochs=15,
                batch_size=16
            )
            
            network = NeuralNetworkFactory.create_network(config)
            
            # Train
            start_time = time.time()
            history = network.train_model(X_train, y_train, validation_split=0.2)
            training_time = time.time() - start_time
            
            # Test
            predictions = network.predict(X_test)
            mae = np.mean(np.abs(y_test - predictions))
            mse = np.mean((y_test - predictions) ** 2)
            r2 = 1 - (mse / np.var(y_test))
            
            logger.info(f"✅ {arch.value} completed")
            logger.info(f"   Training time: {training_time:.2f}s")
            logger.info(f"   Test MAE: {mae:.4f}")
            logger.info(f"   Test R²: {r2:.4f}")
            
            comparison_results[arch.value] = {
                'training_time': training_time,
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1]
            }
            
        except Exception as e:
            logger.error(f"❌ {arch.value} failed: {e}")
            comparison_results[arch.value] = {
                'error': str(e)
            }
    
    return comparison_results

def create_comparison_plot(results: dict):
    """Create a comparison plot of the results."""
    logger.info("\n📈 Creating Comparison Plot")
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_results:
        logger.warning("No successful results to plot")
        return
    
    # Extract metrics
    architectures = list(successful_results.keys())
    mae_values = [successful_results[arch]['mae'] for arch in architectures]
    training_times = [successful_results[arch]['training_time'] for arch in architectures]
    r2_values = [successful_results[arch]['r2'] for arch in architectures]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MAE comparison
    axes[0].bar(architectures, mae_values, color='skyblue', alpha=0.7)
    axes[0].set_title('Mean Absolute Error (Lower is Better)')
    axes[0].set_ylabel('MAE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Training time comparison
    axes[1].bar(architectures, training_times, color='lightcoral', alpha=0.7)
    axes[1].set_title('Training Time')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # R² comparison
    axes[2].bar(architectures, r2_values, color='lightgreen', alpha=0.7)
    axes[2].set_title('R² Score (Higher is Better)')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("neural_network_test_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "neural_network_comparison.png", dpi=300, bbox_inches='tight')
    logger.info(f"📊 Comparison plot saved to {output_dir / 'neural_network_comparison.png'}")
    
    plt.show()

def main():
    """Main test function."""
    logger.info("🚀 Starting Neural Network Factory Tests")
    logger.info("=" * 80)
    
    # Test 1: Individual network creation
    individual_results = test_individual_networks()
    
    # Test 2: Factory creation
    factory_networks = test_factory_creation()
    
    # Test 3: Network training
    training_results = test_network_training()
    
    # Test 4: Architecture comparison
    comparison_results = test_architecture_comparison()
    
    # Create comparison plot
    if comparison_results:
        create_comparison_plot(comparison_results)
    
    # Summary
    logger.info("\n📋 Test Summary")
    logger.info("=" * 60)
    
    logger.info("Individual Network Tests:")
    for name, result in individual_results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        logger.info(f"  {status} {name}")
    
    logger.info(f"\nFactory Creation: {'✅' if factory_networks else '❌'}")
    if factory_networks:
        logger.info(f"  Created {len(factory_networks)} networks")
    
    logger.info("\nTraining Tests:")
    for name, result in training_results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        logger.info(f"  {status} {name}")
    
    logger.info("\nArchitecture Comparison:")
    for name, result in comparison_results.items():
        if 'error' not in result:
            logger.info(f"  ✅ {name}: MAE={result['mae']:.4f}, R²={result['r2']:.4f}")
        else:
            logger.info(f"  ❌ {name}: {result['error']}")
    
    logger.info("\n🎉 Neural Network Factory Tests Completed!")

if __name__ == "__main__":
    main()
