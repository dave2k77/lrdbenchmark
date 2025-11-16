# ML & Neural Estimator Alignment Notes

This note captures the current assumptions that align the machine-learning and neural estimators with the classical/statistical pipeline.

## Unified feature extractor

- `UnifiedFeatureExtractor` exposes 76 canonical features that cover:
  - Descriptive statistics (mean, variance, quantiles, skew/kurtosis)
  - Autocorrelation at lags 1–10
  - DFA/DMA/R/S slopes on logarithmic scales
  - Spectral slopes from the periodogram and Whittle likelihood
  - Wavelet variance/log-variance slopes across octaves
  - Multifractal ratios (structure functions for \\( q \in \{1,2,3,4,5\} \\))
- Subsets of the master vector feed legacy checkpoints:
  - 29 features → SVR
  - 54 features → GradientBoosting
  - 76 features → RandomForest + neural networks
- All subsets are simple prefixes of the 76-vector; tests (`tests/test_unified_feature_extractor.py`) guarantee deterministic extraction and NaN/Inf handling.

## Pretrained classical ML models

- Checkpoints now live outside the repo and are downloaded via `tools/fetch_pretrained_models.py`.
- Each estimator automatically calls `ensure_model_artifact(<key>)`, which:
  1. Looks for `LRDBENCHMARK_MODELS_DIR` (default `~/.cache/lrdbenchmark/models`).
  2. Verifies SHA256 checksums before use.
  3. Falls back to downloading from the GitHub release channel.
- Estimator classes (`RandomForestEstimator`, `SVREstimator`, `GradientBoostingEstimator`) now gracefully handle missing artefacts by returning informative errors instead of trying to open `models/*.joblib`.

## Neural estimators

- Neural-network configs are packaged under `lrdbenchmark/model_configs/*.json` and resolved with `get_model_config_path`.
- The `NeuralNetworkFactory` still trains architectures on demand, but `get_neural_network_model_path` now defers to the asset cache when pretrained `.pth` files exist (currently the reference feedforward checkpoint).
- High-level estimators (CNN/LSTM/GRU/Transformer) emit clearer log messages when configurations/weights are missing and default to inference with freshly initialised networks.

## Next steps

- Expand the asset manifest with additional neural checkpoints as they become available (ensure SHA256 entries are added to `lrdbenchmark/assets.py`).
- Once GPU CI is in place, schedule nightly smoke tests that:
  - Download artefacts (`tools/fetch_pretrained_models.py --models …`).
  - Run a short benchmark script (`scripts/benchmarks/neural_estimators_benchmark.py --smoke`).
- Document best practices for retraining ML/NN models (datasets, RNG seeds, evaluation metrics) to keep checkpoints reproducible.

