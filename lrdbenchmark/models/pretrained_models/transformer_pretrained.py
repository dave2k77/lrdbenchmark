"""
Pre-trained Transformer model for Hurst parameter estimation.

This model uses a simple transformer architecture and loads actual trained weights.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
from .base_pretrained_model import BasePretrainedModel


class SimpleTransformer(nn.Module):
    """Simple Transformer for time series analysis."""
    def __init__(self, input_length: int = 500, d_model: int = 64, nhead: int = 4):
        super(SimpleTransformer, self).__init__()
        self.input_length = input_length
        self.d_model = d_model
        
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(input_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x + self.pos_encoding.unsqueeze(0)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.output_head(x)
        return x

class TransformerPretrainedModel(BasePretrainedModel):
    def __init__(self, input_length: int = 500, use_gpu: bool = False):
        super().__init__()
        self.input_length = input_length
        self.use_gpu = use_gpu
        self.device = self._get_safe_device()
        self.model = None
        self.is_loaded = False
        self._create_model()

    def _get_safe_device(self):
        if not self.use_gpu:
            return torch.device('cpu')
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_model(self):
        # The model was trained with input_length=500, so we must load weights into a length 500 model
        self.model_input_length = 500
        self.model = SimpleTransformer(self.model_input_length).to(self.device)
        
        current_dir = Path(os.path.dirname(__file__))
        model_path = current_dir.parent.parent / "assets" / "models" / "transformer_pretrained.pth"
        
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.is_loaded = True
            except Exception as e:
                print(f"Warning: Failed to load Transformer weights from {model_path}: {e}")
        else:
            print(f"Warning: Pre-trained Transformer model not found at {model_path}")
            
        self.model.eval()

    def estimate(self, data: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Transformer Pretrained model weights not loaded.")

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] != self.model_input_length:
            if data.shape[1] > self.model_input_length:
                data = data[:, : self.model_input_length]
            else:
                padded_data = np.zeros((data.shape[0], self.model_input_length))
                padded_data[:, : data.shape[1]] = data
                data = padded_data

        data_normalized = (data - np.mean(data, axis=1, keepdims=True)) / (
            np.std(data, axis=1, keepdims=True) + 1e-8
        )

        data_tensor = torch.from_numpy(data_normalized).float().unsqueeze(-1).to(self.device)

        with torch.no_grad():
            predictions = self.model(data_tensor)
            predictions = predictions.cpu().numpy().flatten()

        mean_hurst = np.mean(predictions)
        std_error = (np.std(predictions) / np.sqrt(len(predictions))) if len(predictions) > 1 else 0.1
        ci = (max(0, mean_hurst - 1.96 * std_error), min(1, mean_hurst + 1.96 * std_error))

        return {
            "hurst_parameter": float(mean_hurst),
            "confidence_interval": ci,
            "std_error": float(std_error),
            "method": "Transformer (Pre-trained Neural Network)",
            "model_info": self.get_model_info(),
        }
