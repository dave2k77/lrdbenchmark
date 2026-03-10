"""
Pre-trained CNN model for Hurst parameter estimation.

This model uses a 1D CNN architecture and loads actual trained weights.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
from .base_pretrained_model import BasePretrainedModel


class SimpleCNN1D(nn.Module):
    """Simple 1D CNN for time series analysis."""
    def __init__(self, input_length: int = 500):
        super(SimpleCNN1D, self).__init__()
        self.input_length = input_length
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # Calculate output size dynamically based on input length
        x = torch.randn(1, 1, input_length)
        x = self.features(x)
        conv_output_size = x.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNNPretrainedModel(BasePretrainedModel):
    def __init__(self, input_length: int = 500):
        super().__init__()
        self.input_length = input_length
        self.is_loaded = False
        self._create_model()

    def _get_safe_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_model(self):
        device = self._get_safe_device()
        self.model = SimpleCNN1D(self.input_length).to(device)
        
        current_dir = Path(os.path.dirname(__file__))
        model_path = current_dir.parent.parent / "assets" / "models" / "cnn_pretrained.pth"
        
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.is_loaded = True
            except Exception as e:
                print(f"Warning: Failed to load CNN weights from {model_path}: {e}")
        else:
            print(f"Warning: Pre-trained CNN model not found at {model_path}")
            
        self.model.eval()

    def estimate(self, data: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("CNN Pretrained model weights not loaded.")

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] != self.input_length:
            if data.shape[1] > self.input_length:
                data = data[:, : self.input_length]
            else:
                padded_data = np.zeros((data.shape[0], self.input_length))
                padded_data[:, : data.shape[1]] = data
                data = padded_data

        data_normalized = (data - np.mean(data, axis=1, keepdims=True)) / (
            np.std(data, axis=1, keepdims=True) + 1e-8
        )

        device = next(self.model.parameters()).device
        data_tensor = torch.from_numpy(data_normalized).float().to(device)

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
            "method": "CNN (Pre-trained Neural Network)",
            "model_info": self.get_model_info(),
        }
