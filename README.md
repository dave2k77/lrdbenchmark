# LRDBenchmark

A comprehensive, reproducible framework for Long‑Range Dependence (LRD) estimation and benchmarking across Classical, Machine Learning, and Neural Network methods. It provides:

- Unified estimator API (8+ classical, 3 ML, 4 NN; includes the Generalised Hurst Exponent).
- Heavy‑tailed robustness evaluation using α‑stable models with adaptive preprocessing (standardisation/winsorisation/log‑winsorisation/detrending).
- Intelligent optimisation back‑end with graceful fallbacks (JAX/Numba/NumPy) for reliable execution on diverse hardware.
- End‑to‑end benchmarking scripts, statistical analysis (CIs, significance tests, effect sizes), and comprehensive leaderboards (including heavy‑tail performance).

## Quick start

```python
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion

# Generate synthetic data
fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
x = fbm.generate(n=1000, seed=42)

# Estimate H
est = RSEstimator()
result = est.estimate(x)
print(result["hurst_parameter"])  # ~0.7
```

## Documentation

- ReadTheDocs: https://lrdbenchmark.readthedocs.io/
- Examples: see `docs/quickstart.rst` and `docs/examples/`

## Installation

```bash
pip install lrdbenchmark
```

Optional extras for docs/development are available in `pyproject.toml`.

## Citation

If you use LRDBenchmark in your research, please cite the accompanying manuscript (see `research/`).

## Licence

MIT Licence. See `LICENSE`.









