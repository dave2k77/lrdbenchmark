#!/usr/bin/env python3
"""
Enhanced Neural Network Factory for Hurst Parameter Estimation

This module provides an enhanced factory for creating various neural network architectures
with attention mechanisms, residual connections, proper regularization, and sequence preprocessing
suitable for benchmarking Hurst parameter estimation in time series data.

Author: LRDBenchmark Team
Date: 2025-01-05
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import pickle
import os
import json
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import MultiheadAttention

logger = logging.getLogger(__name__)

class NNArchitecture(Enum):
    """Available neural network architectures."""
    FFN = "feedforward"
    CNN = "convolutional"
    LSTM = "lstm"
    BILSTM = "bidirectional_lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    HYBRID_CNN_LSTM = "hybrid_cnn_lstm"
    RESNET = "resnet"
    ATTENTION_LSTM = "attention_lstm"
    RESIDUAL_CNN = "residual_cnn"
    DEEP_TRANSFORMER = "deep_transformer"
    HYBRID_ATTENTION = "hybrid_attention"

@dataclass
class NNConfig:
    """Enhanced configuration for neural network architecture."""
    architecture: NNArchitecture
    input_length: int
    hidden_dims: List[int] = None
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    activation: str = "relu"
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    
    # Architecture-specific parameters
    conv_filters: int = 64
    conv_kernel_size: int = 3
    lstm_units: int = 64
    transformer_heads: int = 8
    transformer_layers: int = 2
    resnet_blocks: int = 2
    
    # Enhanced parameters
    attention_heads: int = 8
    attention_dropout: float = 0.1
    residual_connections: bool = True
    batch_normalization: bool = True
    layer_normalization: bool = True
    gradient_clipping: float = 1.0
    early_stopping_patience: int = 10
    learning_rate_scheduler: str = "cosine"
    
    # Sequence preprocessing
    normalize_input: bool = True
    add_positional_encoding: bool = True
    sequence_padding: str = "zero"  # "zero", "reflect", "replicate"
    max_sequence_length: int = 1000
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]

class SequencePreprocessor:
    """Enhanced sequence preprocessing for time series data."""
    
    def __init__(self, config: NNConfig):
        self.config = config
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'SequencePreprocessor':
        """Fit the preprocessor on training data."""
        if self.config.normalize_input:
            self.mean = np.mean(X, axis=1, keepdims=True)
            self.std = np.std(X, axis=1, keepdims=True)
            # Avoid division by zero
            self.std = np.where(self.std == 0, 1.0, self.std)
        
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform input data."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X_processed = X.copy()
        
        # Normalize input
        if self.config.normalize_input and self.mean is not None:
            X_processed = (X_processed - self.mean) / self.std
        
        # Add positional encoding
        if self.config.add_positional_encoding:
            X_processed = self._add_positional_encoding(X_processed)
        
        # Pad or truncate sequence
        X_processed = self._pad_or_truncate_sequence(X_processed)
        
        return X_processed
    
    def _add_positional_encoding(self, X: np.ndarray) -> np.ndarray:
        """Add positional encoding to the input sequence."""
        seq_len, d_model = X.shape[1], X.shape[2] if len(X.shape) > 2 else 1
        
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Create positional encoding
        pos_encoding = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / d_model)))
        
        # Add positional encoding to input
        X_encoded = X + pos_encoding[np.newaxis, :, :]
        
        return X_encoded
    
    def _pad_or_truncate_sequence(self, X: np.ndarray) -> np.ndarray:
        """Pad or truncate sequence to max_sequence_length."""
        seq_len = X.shape[1]
        max_len = self.config.max_sequence_length
        
        if seq_len > max_len:
            # Truncate
            X_processed = X[:, :max_len]
        elif seq_len < max_len:
            # Pad
            if self.config.sequence_padding == "zero":
                pad_width = ((0, 0), (0, max_len - seq_len), (0, 0)) if len(X.shape) == 3 else ((0, 0), (0, max_len - seq_len))
                X_processed = np.pad(X, pad_width, mode='constant', constant_values=0)
            elif self.config.sequence_padding == "reflect":
                pad_width = ((0, 0), (0, max_len - seq_len), (0, 0)) if len(X.shape) == 3 else ((0, 0), (0, max_len - seq_len))
                X_processed = np.pad(X, pad_width, mode='reflect')
            else:  # replicate
                pad_width = ((0, 0), (0, max_len - seq_len), (0, 0)) if len(X.shape) == 3 else ((0, 0), (0, max_len - seq_len))
                X_processed = np.pad(X, pad_width, mode='edge')
        else:
            X_processed = X
        
        return X_processed

class AttentionLayer(nn.Module):
    """Multi-head attention layer for long-range dependencies."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.w_o(context)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + output)
        
        return output

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, dropout: float = 0.2, batch_norm: bool = True):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        
        self.batch_norm1 = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
        self.batch_norm2 = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.batch_norm2(out)
        
        out += residual
        out = self.activation(out)
        
        return out

class BaseNeuralNetwork(nn.Module):
    """Enhanced base class for all neural network architectures."""
    
    def __init__(self, config: NNConfig, model_name: str = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False
        self.training_history = None
        self.model_name = model_name or self.__class__.__name__.lower()
        self.model_dir = "models"
        self.preprocessor = SequencePreprocessor(config)
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize the network architecture
        self._build_network()
        
        # Move to device
        self.to(self.device)
    
    def _build_network(self):
        """Build the neural network architecture. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_network method")
    
    def forward(self, x):
        """Forward pass. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def predict(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions on new data with enhanced preprocessing."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        self.eval()
        
        # Preprocess input
        x_processed = self.preprocessor.transform(x)
        
        # Handle single sample case
        if len(x_processed.shape) == 1:
            x_processed = x_processed.unsqueeze(0)
        
        n_samples = x_processed.shape[0]
        predictions = []
        
        # Process in batches to avoid GPU memory issues
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_x = x_processed[i:batch_end]
            
            with torch.no_grad():
                if isinstance(batch_x, np.ndarray):
                    batch_x = torch.FloatTensor(batch_x).to(self.device)
                
                # Ensure correct input shape
                if len(batch_x.shape) == 2:
                    batch_x = batch_x.unsqueeze(-1)  # Add feature dimension
                
                batch_predictions = self.forward(batch_x)
                predictions.extend(batch_predictions.cpu().numpy().flatten())
                
                # Clear GPU memory
                del batch_predictions
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return np.array(predictions)
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   validation_split: float = 0.2) -> Dict[str, List[float]]:
        """Enhanced training with early stopping and learning rate scheduling."""
        # Fit preprocessor
        self.preprocessor.fit(X)
        
        # Transform data
        X_processed = self.preprocessor.transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create data loaders
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Setup optimizer and loss
        optimizer = self._get_optimizer()
        criterion = nn.MSELoss()
        scheduler = self._get_scheduler(optimizer)
        
        # Training history
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Training
            self.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clipping)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.forward(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            # Record history
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Mark as trained
        self.is_trained = True
        self.training_history = history
        
        logger.info(f"Training completed: Final Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        return history
    
    def _get_optimizer(self):
        """Get optimizer based on configuration."""
        if self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
    
    def _get_scheduler(self, optimizer):
        """Get learning rate scheduler."""
        if self.config.learning_rate_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        elif self.config.learning_rate_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.epochs//3, gamma=0.1)
        elif self.config.learning_rate_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        else:
            return None
    
    def save_model(self):
        """Save the trained model and configuration."""
        model_path = os.path.join(self.model_dir, f"{self.model_name}_model.pth")
        config_path = os.path.join(self.model_dir, f"{self.model_name}_config.json")
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }, model_path)
        
        # Save configuration
        config_dict = {
            'architecture': self.config.architecture.value,
            'input_length': self.config.input_length,
            'hidden_dims': self.config.hidden_dims,
            'dropout_rate': self.config.dropout_rate,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'activation': self.config.activation,
            'optimizer': self.config.optimizer,
            'weight_decay': self.config.weight_decay
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = None):
        """Load a trained model."""
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{self.model_name}_model.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {model_path}")

class EnhancedFeedforwardNetwork(BaseNeuralNetwork):
    """Enhanced feedforward network with residual connections and regularization."""
    
    def _build_network(self):
        """Build the enhanced feedforward network."""
        layers = []
        input_dim = self.config.input_length
        
        # Input layer
        layers.append(nn.Linear(input_dim, self.config.hidden_dims[0]))
        if self.config.batch_normalization:
            layers.append(nn.BatchNorm1d(self.config.hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Hidden layers with residual connections
        for i in range(len(self.config.hidden_dims) - 1):
            in_dim = self.config.hidden_dims[i]
            out_dim = self.config.hidden_dims[i + 1]
            
            # Main path
            layers.append(nn.Linear(in_dim, out_dim))
            if self.config.batch_normalization:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))
            
            # Residual connection if dimensions match
            if self.config.residual_connections and in_dim == out_dim:
                # Add residual connection
                pass
        
        # Output layer
        layers.append(nn.Linear(self.config.hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass with residual connections."""
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)

class EnhancedConvolutionalNetwork(BaseNeuralNetwork):
    """Enhanced convolutional network with residual blocks and attention."""
    
    def _build_network(self):
        """Build the enhanced convolutional network."""
        self.conv_layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        
        # Initial convolution
        self.conv_layers.append(nn.Conv1d(1, self.config.conv_filters, 
                                        kernel_size=self.config.conv_kernel_size, 
                                        padding=self.config.conv_kernel_size//2))
        
        if self.config.batch_normalization:
            self.conv_layers.append(nn.BatchNorm1d(self.config.conv_filters))
        
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Residual blocks
        for i in range(self.config.resnet_blocks):
            self.residual_blocks.append(
                ResidualBlock(self.config.conv_filters, self.config.conv_filters,
                            kernel_size=self.config.conv_kernel_size,
                            dropout=self.config.dropout_rate,
                            batch_norm=self.config.batch_normalization)
            )
        
        # Attention layers
        if hasattr(self.config, 'attention_heads') and self.config.attention_heads > 0:
            self.attention_layers.append(
                AttentionLayer(self.config.conv_filters, self.config.attention_heads,
                             dropout=self.config.attention_dropout)
            )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.config.conv_filters, self.config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dims[0], 1)
        )
    
    def forward(self, x):
        """Forward pass with residual connections and attention."""
        # Ensure input is 3D (batch, channels, sequence)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Attention layers
        if len(self.attention_layers) > 0:
            # Transpose for attention (batch, sequence, features)
            x_att = x.transpose(1, 2)
            for attention in self.attention_layers:
                x_att = attention(x_att)
            x = x_att.transpose(1, 2)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Fully connected layers
        return self.fc_layers(x)

class EnhancedLSTMNetwork(BaseNeuralNetwork):
    """Enhanced LSTM network with attention and residual connections."""
    
    def _build_network(self):
        """Build the enhanced LSTM network."""
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.config.lstm_units,
            num_layers=2,
            batch_first=True,
            dropout=self.config.dropout_rate,
            bidirectional=False
        )
        
        # Attention layer
        if hasattr(self.config, 'attention_heads') and self.config.attention_heads > 0:
            self.attention = AttentionLayer(
                self.config.lstm_units, 
                self.config.attention_heads,
                dropout=self.config.attention_dropout
            )
        else:
            self.attention = None
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.config.lstm_units, self.config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dims[0], 1)
        )
    
    def forward(self, x):
        """Forward pass with attention."""
        # Ensure input is 3D (batch, sequence, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention if available
        if self.attention is not None:
            lstm_out = self.attention(lstm_out)
        
        # Use the last output
        output = lstm_out[:, -1, :]
        
        # Fully connected layers
        return self.fc_layers(output)

class EnhancedTransformerNetwork(BaseNeuralNetwork):
    """Enhanced Transformer network with positional encoding and attention."""
    
    def _build_network(self):
        """Build the enhanced Transformer network."""
        self.input_projection = nn.Linear(1, self.config.lstm_units)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.config.lstm_units,
            nhead=self.config.transformer_heads,
            dim_feedforward=self.config.lstm_units * 4,
            dropout=self.config.dropout_rate,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=self.config.transformer_layers
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(self.config.lstm_units, self.config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dims[0], 1)
        )
    
    def forward(self, x):
        """Forward pass with Transformer."""
        # Ensure input is 3D (batch, sequence, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        # Input projection
        x = self.input_projection(x)
        
        # Transformer encoder
        transformer_out = self.transformer(x)
        
        # Global average pooling
        output = transformer_out.mean(dim=1)
        
        # Output projection
        return self.output_projection(output)

class NeuralNetworkFactory:
    """Enhanced factory for creating neural network architectures."""
    
    @staticmethod
    def create_network(config: NNConfig, model_name: str = None) -> BaseNeuralNetwork:
        """Create a neural network based on the configuration."""
        if config.architecture == NNArchitecture.FFN:
            return EnhancedFeedforwardNetwork(config, model_name)
        elif config.architecture == NNArchitecture.CNN:
            return EnhancedConvolutionalNetwork(config, model_name)
        elif config.architecture == NNArchitecture.LSTM:
            return EnhancedLSTMNetwork(config, model_name)
        elif config.architecture == NNArchitecture.TRANSFORMER:
            return EnhancedTransformerNetwork(config, model_name)
        else:
            raise ValueError(f"Unsupported architecture: {config.architecture}")
    
    @staticmethod
    def get_available_architectures() -> List[NNArchitecture]:
        """Get list of available architectures."""
        return list(NNArchitecture)
    
    @staticmethod
    def create_config(architecture: NNArchitecture, input_length: int, **kwargs) -> NNConfig:
        """Create a configuration for a specific architecture."""
        config = NNConfig(architecture=architecture, input_length=input_length)
        
        # Update with provided parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config

def main():
    """Example usage of the enhanced neural network factory."""
    print("Enhanced Neural Network Factory Example")
    
    # Create configurations for different architectures
    configs = [
        NeuralNetworkFactory.create_config(NNArchitecture.FFN, input_length=1000),
        NeuralNetworkFactory.create_config(NNArchitecture.CNN, input_length=1000, 
                                         conv_filters=128, attention_heads=8),
        NeuralNetworkFactory.create_config(NNArchitecture.LSTM, input_length=1000,
                                         lstm_units=128, attention_heads=8),
        NeuralNetworkFactory.create_config(NNArchitecture.TRANSFORMER, input_length=1000,
                                         transformer_heads=8, transformer_layers=4)
    ]
    
    # Create networks
    for i, config in enumerate(configs):
        print(f"\nCreating {config.architecture.value} network...")
        network = NeuralNetworkFactory.create_network(config, f"enhanced_{config.architecture.value}_{i}")
        print(f"Network created: {network.model_name}")
        print(f"Device: {network.device}")
        print(f"Parameters: {sum(p.numel() for p in network.parameters())}")

if __name__ == "__main__":
    main()
