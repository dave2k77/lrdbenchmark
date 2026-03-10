import numpy as np
import sys
import os

# Add the parent directory to sys.path so we can import lrdbenchmark
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lrdbenchmark.generation.time_series_generator import TimeSeriesGenerator
from lrdbenchmark.models.pretrained_models.ml_pretrained import (
    RandomForestPretrainedModel,
    SVREstimatorPretrainedModel,
    GradientBoostingPretrainedModel
)
from lrdbenchmark.models.pretrained_models.cnn_pretrained import CNNPretrainedModel
from lrdbenchmark.models.pretrained_models.transformer_pretrained import TransformerPretrainedModel
from lrdbenchmark.models.pretrained_models.lstm_pretrained import LSTMPretrainedModel

def main():
    print("Generating simulated Fractional Brownian Motion (H=0.7)...")
    generator = TimeSeriesGenerator(random_state=42)
    length = 1024
    true_h = 0.7
    
    result = generator.generate(
        model='fbm',
        length=length,
        params={'H': true_h},
        preprocess=True
    )
    
    data = result['signal']
    print(f"Data generated. Shape: {data.shape}")
    
    models = {
        "Random Forest": RandomForestPretrainedModel(),
        "SVR": SVREstimatorPretrainedModel(),
        "Gradient Boosting": GradientBoostingPretrainedModel(),
        "CNN": CNNPretrainedModel(input_length=length),
        "Transformer": TransformerPretrainedModel(input_length=length),
        "LSTM": LSTMPretrainedModel(input_length=length, model_path=None)
    }
    
    print("\n--- Model Accuracy Benchmark ---")
    print(f"True Hurst Parameter: {true_h:.4f}")
    print("-" * 40)
    
    for name, model in models.items():
        try:
            estimate = model.estimate(data)
            h_est = estimate['hurst_parameter']
            error = abs(h_est - true_h)
            print(f"{name:18s} | Est: {h_est:.4f} | Error: {error:.4f} | CI: {estimate.get('confidence_interval', [])}")
        except Exception as e:
            print(f"{name:18s} | Est: FAILED | Error: {e}")

if __name__ == "__main__":
    main()
