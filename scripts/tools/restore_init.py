
content = """\"\"\"
Pre-trained Models Package for LRDBench

This package contains pre-trained neural network models for
Hurst estimation. Using pre-trained models allows faster
evaluation and benchmarking by avoiding training each
model during runtime.
\"\"\"

from .base_pretrained_model import BasePretrainedModel
from .cnn_pretrained import CNNPretrainedModel
from .transformer_pretrained import TransformerPretrainedModel
from .lstm_pretrained import LSTMPretrainedModel
from .gru_pretrained import GRUPretrainedModel
from .ml_pretrained import (
    RandomForestPretrainedModel,
    SVREstimatorPretrainedModel,
    GradientBoostingPretrainedModel,
)

__all__ = [
    "BasePretrainedModel",
    "CNNPretrainedModel",
    "TransformerPretrainedModel",
    "LSTMPretrainedModel",
    "GRUPretrainedModel",
    "RandomForestPretrainedModel",
    "SVREstimatorPretrainedModel",
    "GradientBoostingPretrainedModel",
]
"""
path = r"c:\Users\davia\git_clone_projects\lrdbenchmark\lrdbenchmark\models\pretrained_models\__init__.py"
with open(path, "w") as f:
    f.write(content)
print("Restored __init__.py")
