
import numpy as np
import torch
import torch.nn as nn
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .base_pretrained_model import BasePretrainedModel

class LSTMModel(nn.Module):
    """LSTM model for Hurst parameter estimation."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class LSTMPretrainedModel(BasePretrainedModel):
    def __init__(self, model_path: Optional[str] = None, input_length: int = 1024):
        # We need to initialize the default model path BEFORE calling super().__init__()
        # Or alternatively just let it be None and handle it later. We'll explicitly handle it.
        if model_path is None:
            current_dir = Path(os.path.dirname(__file__))
            default_path = current_dir.parent.parent / "assets" / "models" / "lstm_model.pth"
            model_path = str(default_path)

        super().__init__()
        self.model_path = model_path
        self.input_length = input_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.model = self._load_model()
    
    def _load_model(self) -> nn.Module:
        model = LSTMModel().to(self.device)
        if self.model_path and os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                self.is_loaded = True
            except Exception as e:
                print(f"Warning: Failed to load model weights from {self.model_path}: {e}")
        else:
            print(f"Warning: Pre-trained model not found at {self.model_path}.")
        return model
        
    def estimate(self, data: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("LSTM Pretrained model weights not loaded.")

        data = self._preprocess(data)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        data_tensor = torch.FloatTensor(data).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(data_tensor).cpu().numpy().flatten()
            
        mean_hurst = np.mean(predictions)
        std_error = (np.std(predictions) / np.sqrt(len(predictions))) if len(predictions) > 1 else 0.1
        ci = (max(0, mean_hurst - 1.96 * std_error), min(1, mean_hurst + 1.96 * std_error))
        
        return {
            "hurst_parameter": float(mean_hurst),
            "confidence_interval": ci,
            "std_error": float(std_error),
            "method": "LSTM (Pre-trained Neural Network)",
            "model_info": "PyTorch LSTM"
        }

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        if data.shape[1] > self.input_length:
            data = data[:, :self.input_length]
        elif data.shape[1] < self.input_length:
            pad_width = self.input_length - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
            
        data_normalized = (data - np.mean(data, axis=1, keepdims=True)) / (
            np.std(data, axis=1, keepdims=True) + 1e-8
        )
        return data_normalized
