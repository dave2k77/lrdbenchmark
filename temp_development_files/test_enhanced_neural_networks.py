#!/usr/bin/env python3
"""
Test Enhanced Neural Network Factory

This script tests the enhanced neural network factory with attention mechanisms,
residual connections, proper regularization, and sequence preprocessing.

Author: LRDBenchmark Team
Date: 2025-01-05
"""

import numpy as np
import torch
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

# Import the enhanced neural network factory
from enhanced_neural_network_factory import (
    NeuralNetworkFactory, NNArchitecture, NNConfig, 
    SequencePreprocessor
)

# Import data generation
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion as FBMModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedNeuralNetworkTester:
    """Test the enhanced neural network factory."""
    
    def __init__(self):
        """Initialize the tester."""
        self.results = {}
        self.test_data = {}
        
        # Test parameters
        self.hurst_values = [0.2, 0.4, 0.6, 0.8]
        self.data_lengths = [1000]  # Use only one length for testing
        self.n_samples_per_condition = 10
        
        # Enhanced configurations
        self.enhanced_configs = {
            'enhanced_feedforward': NNConfig(
                architecture=NNArchitecture.FFN,
                input_length=1000,
                hidden_dims=[128, 64, 32],
                dropout_rate=0.3,
                learning_rate=0.001,
                batch_size=16,
                epochs=30,
                residual_connections=True,
                batch_normalization=True,
                gradient_clipping=1.0,
                early_stopping_patience=5,
                learning_rate_scheduler="cosine",
                normalize_input=True,
                add_positional_encoding=True
            ),
            'enhanced_cnn': NNConfig(
                architecture=NNArchitecture.CNN,
                input_length=1000,
                conv_filters=128,
                conv_kernel_size=5,
                resnet_blocks=3,
                attention_heads=8,
                attention_dropout=0.1,
                dropout_rate=0.3,
                learning_rate=0.001,
                batch_size=16,
                epochs=30,
                residual_connections=True,
                batch_normalization=True,
                gradient_clipping=1.0,
                early_stopping_patience=5,
                learning_rate_scheduler="cosine",
                normalize_input=True,
                add_positional_encoding=True
            ),
            'enhanced_lstm': NNConfig(
                architecture=NNArchitecture.LSTM,
                input_length=1000,
                lstm_units=128,
                attention_heads=8,
                attention_dropout=0.1,
                dropout_rate=0.3,
                learning_rate=0.001,
                batch_size=16,
                epochs=30,
                gradient_clipping=1.0,
                early_stopping_patience=5,
                learning_rate_scheduler="cosine",
                normalize_input=True,
                add_positional_encoding=True
            ),
            'enhanced_transformer': NNConfig(
                architecture=NNArchitecture.TRANSFORMER,
                input_length=1000,
                transformer_heads=8,
                transformer_layers=4,
                lstm_units=128,
                dropout_rate=0.3,
                learning_rate=0.001,
                batch_size=16,
                epochs=30,
                gradient_clipping=1.0,
                early_stopping_patience=5,
                learning_rate_scheduler="cosine",
                normalize_input=True,
                add_positional_encoding=True
            )
        }
    
    def generate_test_data(self):
        """Generate test data for neural network training."""
        print("Generating test data...")
        
        X_data = []
        y_data = []
        
        for hurst in self.hurst_values:
            for length in self.data_lengths:
                for sample in range(self.n_samples_per_condition):
                    # Generate FBM data
                    fbm_model = FBMModel(H=hurst)
                    fbm_data = fbm_model.generate(n=length)
                    
                    # Ensure consistent length
                    if len(fbm_data) != length:
                        fbm_data = fbm_data[:length] if len(fbm_data) > length else np.pad(fbm_data, (0, length - len(fbm_data)))
                    
                    # Debug: print lengths
                    if sample == 0:  # Only print for first sample of each condition
                        print(f"  Hurst {hurst}, Length {length}: Generated {len(fbm_data)} samples")
                    
                    X_data.append(fbm_data)
                    y_data.append(hurst)
        
        self.test_data['X'] = np.array(X_data)
        self.test_data['y'] = np.array(y_data)
        
        print(f"Generated {len(X_data)} samples")
        print(f"Input shape: {self.test_data['X'].shape}")
        print(f"Output shape: {self.test_data['y'].shape}")
    
    def test_sequence_preprocessor(self):
        """Test the sequence preprocessor."""
        print("\nTesting Sequence Preprocessor...")
        
        # Create test data
        X_test = np.random.randn(10, 1000)
        
        # Test different configurations
        configs = [
            NNConfig(NNArchitecture.FFN, 1000, normalize_input=True, add_positional_encoding=False),
            NNConfig(NNArchitecture.FFN, 1000, normalize_input=True, add_positional_encoding=True),
            NNConfig(NNArchitecture.FFN, 1000, normalize_input=False, add_positional_encoding=True),
        ]
        
        for i, config in enumerate(configs):
            print(f"\nTesting config {i+1}: normalize={config.normalize_input}, pos_encoding={config.add_positional_encoding}")
            
            preprocessor = SequencePreprocessor(config)
            preprocessor.fit(X_test)
            X_processed = preprocessor.transform(X_test)
            
            print(f"  Input shape: {X_test.shape}")
            print(f"  Output shape: {X_processed.shape}")
            print(f"  Mean: {np.mean(X_processed):.4f}")
            print(f"  Std: {np.std(X_processed):.4f}")
    
    def test_enhanced_networks(self):
        """Test all enhanced neural network architectures."""
        print("\nTesting Enhanced Neural Networks...")
        
        for config_name, config in self.enhanced_configs.items():
            print(f"\nTesting {config_name}...")
            
            try:
                # Create network
                network = NeuralNetworkFactory.create_network(config, config_name)
                print(f"  Network created: {network.model_name}")
                print(f"  Device: {network.device}")
                print(f"  Parameters: {sum(p.numel() for p in network.parameters()):,}")
                
                # Test forward pass
                test_input = torch.randn(2, config.input_length).to(network.device)
                if config.architecture == NNArchitecture.CNN:
                    test_input = test_input.unsqueeze(1)  # Add channel dimension
                elif config.architecture in [NNArchitecture.LSTM, NNArchitecture.TRANSFORMER]:
                    test_input = test_input.unsqueeze(-1)  # Add feature dimension
                
                with torch.no_grad():
                    output = network.forward(test_input)
                    print(f"  Forward pass successful: {test_input.shape} -> {output.shape}")
                
                # Test training
                print("  Training network...")
                start_time = time.time()
                
                # Use smaller subset for testing
                X_train = self.test_data['X'][:20]
                y_train = self.test_data['y'][:20]
                
                history = network.train_model(X_train, y_train, validation_split=0.2)
                
                training_time = time.time() - start_time
                print(f"  Training completed in {training_time:.2f}s")
                print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
                print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
                
                # Test prediction
                test_predictions = network.predict(X_train[:5])
                print(f"  Predictions: {test_predictions}")
                print(f"  True values: {y_train[:5]}")
                
                # Calculate MAE
                mae = np.mean(np.abs(test_predictions - y_train[:5]))
                print(f"  MAE: {mae:.4f}")
                
                # Store results
                self.results[config_name] = {
                    'network': network,
                    'config': config,
                    'training_time': training_time,
                    'final_train_loss': history['train_loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'mae': mae,
                    'parameters': sum(p.numel() for p in network.parameters()),
                    'success': True
                }
                
            except Exception as e:
                print(f"  Error testing {config_name}: {e}")
                self.results[config_name] = {
                    'success': False,
                    'error': str(e)
                }
    
    def test_attention_mechanisms(self):
        """Test attention mechanisms specifically."""
        print("\nTesting Attention Mechanisms...")
        
        # Create a simple attention test
        from enhanced_neural_network_factory import AttentionLayer
        
        batch_size, seq_len, d_model = 4, 100, 64
        x = torch.randn(batch_size, seq_len, d_model)
        
        attention = AttentionLayer(d_model, n_heads=8, dropout=0.1)
        output = attention(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Attention mechanism working: {output.shape == x.shape}")
    
    def test_residual_connections(self):
        """Test residual connections."""
        print("\nTesting Residual Connections...")
        
        from enhanced_neural_network_factory import ResidualBlock
        
        batch_size, in_channels, seq_len = 4, 32, 100
        x = torch.randn(batch_size, in_channels, seq_len)
        
        residual_block = ResidualBlock(in_channels, 64, kernel_size=3, dropout=0.1)
        output = residual_block(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Residual connection working: {output.shape[1] == 64}")
    
    def generate_performance_report(self):
        """Generate a performance report."""
        print("\n" + "="*60)
        print("ENHANCED NEURAL NETWORK PERFORMANCE REPORT")
        print("="*60)
        
        successful_networks = [name for name, result in self.results.items() if result.get('success', False)]
        
        if not successful_networks:
            print("No networks were successfully tested.")
            return
        
        print(f"\nSuccessfully tested {len(successful_networks)} networks:")
        
        # Performance metrics
        metrics = ['parameters', 'training_time', 'final_train_loss', 'final_val_loss', 'mae']
        
        for metric in metrics:
            print(f"\n{metric.upper()}:")
            sorted_networks = sorted(
                [(name, result[metric]) for name, result in self.results.items() 
                 if result.get('success', False) and metric in result],
                key=lambda x: x[1]
            )
            
            for name, value in sorted_networks:
                if metric == 'parameters':
                    print(f"  {name}: {value:,}")
                else:
                    print(f"  {name}: {value:.4f}")
        
        # Best performers
        print(f"\nBEST PERFORMERS:")
        
        best_mae = min([result['mae'] for result in self.results.values() if result.get('success', False)])
        best_mae_network = [name for name, result in self.results.items() 
                           if result.get('success', False) and result['mae'] == best_mae][0]
        print(f"  Best MAE: {best_mae_network} ({best_mae:.4f})")
        
        best_time = min([result['training_time'] for result in self.results.values() if result.get('success', False)])
        best_time_network = [name for name, result in self.results.items() 
                            if result.get('success', False) and result['training_time'] == best_time][0]
        print(f"  Fastest Training: {best_time_network} ({best_time:.2f}s)")
        
        smallest_model = min([result['parameters'] for result in self.results.values() if result.get('success', False)])
        smallest_model_network = [name for name, result in self.results.items() 
                                 if result.get('success', False) and result['parameters'] == smallest_model][0]
        print(f"  Smallest Model: {smallest_model_network} ({smallest_model:,} parameters)")
    
    def save_results(self):
        """Save test results."""
        results_path = "enhanced_neural_network_test_results.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for name, result in self.results.items():
            json_results[name] = {
                'success': result.get('success', False),
                'error': result.get('error', None),
                'training_time': result.get('training_time', None),
                'final_train_loss': result.get('final_train_loss', None),
                'final_val_loss': result.get('final_val_loss', None),
                'mae': result.get('mae', None),
                'parameters': result.get('parameters', None)
            }
        
        import json
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")

def main():
    """Main function to run the enhanced neural network tests."""
    print("Enhanced Neural Network Factory Test")
    print("="*50)
    
    # Initialize tester
    tester = EnhancedNeuralNetworkTester()
    
    try:
        # Generate test data
        tester.generate_test_data()
        
        # Test sequence preprocessor
        tester.test_sequence_preprocessor()
        
        # Test attention mechanisms
        tester.test_attention_mechanisms()
        
        # Test residual connections
        tester.test_residual_connections()
        
        # Test enhanced networks
        tester.test_enhanced_networks()
        
        # Generate performance report
        tester.generate_performance_report()
        
        # Save results
        tester.save_results()
        
        print("\nEnhanced neural network testing completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    main()
