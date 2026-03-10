"""
Pre-trained ML models for Hurst parameter estimation.

This module provides pre-trained versions of machine learning estimators
that use genuine scikit-learn models.
"""

import numpy as np
import os
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

from .base_pretrained_model import BasePretrainedModel


def extract_ml_features(data: np.ndarray) -> np.ndarray:
    """Extract standard features for ML models: variance ratio, spectral slope, autocorrelation."""
    features = []
    if data.ndim == 1:
        data = data.reshape(1, -1)
        
    for i in range(data.shape[0]):
        series = data[i].copy()
        
        # 1. Variance ratio across segments
        segment_size = max(10, len(series) // 4)
        segments = [
            series[j : j + segment_size]
            for j in range(0, len(series), segment_size)
            if j + segment_size <= len(series)
        ]
        
        var_ratio = 1.0
        if segments:
            variances = [np.var(seg) for seg in segments]
            var_ratio = np.std(variances) / (np.mean(variances) + 1e-8)
            
        # 2. Spectral slope
        fft_vals = np.abs(np.fft.fft(series))
        freqs = np.fft.fftfreq(len(series))
        positive_freqs = freqs > 0
        spectral_slope = -1.0
        if np.sum(positive_freqs) > 1:
            log_freqs = np.log(freqs[positive_freqs] + 1e-8)
            log_fft = np.log(fft_vals[positive_freqs] + 1e-8)
            if len(log_freqs) > 1:
                spectral_slope = np.polyfit(log_freqs, log_fft, 1)[0]
                
        # 3. Autocorrelation
        autocorr = 0.0
        if len(series) > 1:
            cc = np.corrcoef(series[:-1], series[1:])[0, 1]
            if not np.isnan(cc):
                autocorr = cc
                
        features.append([var_ratio, spectral_slope, autocorr])
        
    return np.array(features)

class ScikitLearnPretrainedModel(BasePretrainedModel):
    """Base class for pre-trained Scikit-Learn based models."""

    def __init__(self, model_filename: str, method_name: str, **kwargs):
        super().__init__()
        self.method_name = method_name
        self.is_loaded = False
        self.model = None
        
        # Resolve path to assets/models
        current_dir = Path(os.path.dirname(__file__))
        self.model_path = current_dir.parent.parent / "assets" / "models" / model_filename
        
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
            except Exception as e:
                print(f"Warning: Failed to load {model_filename} from {self.model_path}: {e}")
        else:
            print(f"Warning: Pre-trained model {model_filename} not found at {self.model_path}")

    def estimate(self, data: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"Model {self.method_name} not properly loaded.")

        if data.ndim == 1:
            data = data.reshape(1, -1)

        features = extract_ml_features(data)
        predictions = self.model.predict(features)
        
        # Calculate confidence interval (simplified standard error)
        mean_hurst = np.mean(predictions)
        std_error = (
            np.std(predictions) / np.sqrt(len(predictions))
            if len(predictions) > 1
            else 0.1
        )
        # bound to [0,1]
        confidence_interval = (
            max(0.01, mean_hurst - 1.96 * std_error),
            min(0.99, mean_hurst + 1.96 * std_error),
        )

        return {
            "hurst_parameter": float(mean_hurst),
            "confidence_interval": confidence_interval,
            "std_error": float(std_error),
            "method": f"{self.method_name} (Pre-trained ML)",
            "model_info": self.get_model_info(),
        }

class RandomForestPretrainedModel(ScikitLearnPretrainedModel):
    def __init__(self, **kwargs):
        super().__init__("rf_pretrained.joblib", "Random Forest", **kwargs)

class SVREstimatorPretrainedModel(ScikitLearnPretrainedModel):
    def __init__(self, **kwargs):
        super().__init__("svr_pretrained.joblib", "SVR", **kwargs)

class GradientBoostingPretrainedModel(ScikitLearnPretrainedModel):
    def __init__(self, **kwargs):
        super().__init__("gb_pretrained.joblib", "Gradient Boosting", **kwargs)
