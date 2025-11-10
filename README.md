# lrdbenchmark

Modern, reproducible benchmarking for long-range dependence (LRD) estimation across classical statistics, machine learning, and neural approaches.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8–3.12](https://img.shields.io/badge/python-3.8%E2%80%933.12-blue.svg)](https://www.python.org/downloads/)
[![Version 2.3.1](https://img.shields.io/badge/version-2.3.1-green.svg)](https://pypi.org/project/lrdbenchmark/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17534599.svg)](https://doi.org/10.5281/zenodo.17534599)

---

## Why lrdbenchmark?

- **One interface, twenty estimators** – 13 classical, 3 machine learning, and 4 neural estimators share a unified API with consistent metadata.
- **Deterministic by construction** – global RNG coordination, stratified summaries, significance testing, and provenance capture are built in.
- **Runtime profiles** – choose `quick` for smoke tests or CI, or `full` for exhaustive diagnostics, bootstraps, and robustness panels.
- **Production-aware workflows** – supports CPU-only deployments by default with optional JAX/Numba/Torch acceleration.
- **Documentation-first tutorials** – the tutorial series now ships directly in `docs/tutorials/`, mirrored by lightweight Markdown notebooks for interactive sessions.

---

## Getting Started

### Installation

```bash
pip install lrdbenchmark                          # CPU-only
pip install lrdbenchmark[accel-jax]              # + JAX acceleration
pip install lrdbenchmark[accel-numba]            # + Numba acceleration
pip install lrdbenchmark[accel-pytorch]          # + PyTorch acceleration
# or everything: pip install lrdbenchmark[accel-all]
```

### First Benchmark

```python
from lrdbenchmark import ComprehensiveBenchmark

# Quick profile skips heavy diagnostics – perfect for tests and CI
benchmark = ComprehensiveBenchmark(runtime_profile="quick")
summary = benchmark.run_comprehensive_benchmark(
    data_length=256,
    benchmark_type="classical",
    save_results=False,
)

print(summary["random_state"])
print(summary["stratified_metrics"]["hurst_bands"])
```

Want the full analysis (bootstrap confidence intervals, robustness panels, influence diagnostics)? Simply drop the profile override:

```python
benchmark = ComprehensiveBenchmark()   # runtime_profile defaults to "auto"/"full"
```

### Runtime Profiles at a Glance

| Profile | How to enable | Designed for | What is disabled |
|---------|---------------|--------------|------------------|
| `quick` | `ComprehensiveBenchmark(runtime_profile="quick")` or `export LRDBENCHMARK_RUNTIME_PROFILE=quick` | Unit tests, CI, exploratory work | Advanced metrics, bootstraps, robustness panels, heavy diagnostics |
| `full`  | Default when running outside pytest/quick mode | End-to-end studies, publications | Nothing – full diagnostics and provenance |

---

## Core Capabilities

- **Estimator families** – temporal (R/S, DFA, DMA, Higuchi), spectral (Periodogram, GPH, Whittle), wavelet (CWT, variance, log-variance, wavelet Whittle), multifractal (MFDFA, wavelet leaders), machine-learning (Random Forest, SVR, Gradient Boosting), and neural (CNN, LSTM, GRU, Transformer).
- **Robust benchmarking** – contamination models, adaptive preprocessing, stratified reporting, non-parametric significance tests, and provenance bundles per result.
- **Analytics tooling** – convergence analysis, bias estimation, stress panels, uncertainty calibration, scale influence diagnostics.
- **GPU-aware execution** – intelligent fallbacks (JAX ▶ Numba ▶ NumPy) with automatic CPU mode unless the user explicitly opts into GPU acceleration.

For the full catalogue see the [API reference](https://lrdbenchmark.readthedocs.io/en/latest/api/).

---

## Documentation & Learning Path

- **Full documentation**: <https://lrdbenchmark.readthedocs.io/>
- **Tutorial sequence**: `docs/tutorials/` (rendered on Read the Docs, aligned with the original notebook curriculum)
- **Interactive notebooks**: Markdown sources in `notebooks/markdown/`, easily opened via [Jupytext](https://jupytext.readthedocs.io/) or any Markdown-friendly notebook environment
- **Examples & scripts**: runnable patterns in `examples/` and `scripts/`

### Working with the Markdown notebooks

```bash
pip install jupytext
jupytext --to notebook notebooks/markdown/02_estimation_and_validation.md
jupyter notebook notebooks/markdown/
```

This keeps the repository light while preserving the original interactive walkthroughs.

---

## Project Layout

```
lrdbenchmark/
├── lrdbenchmark/            # Package modules
│   ├── analysis/            # Estimators, benchmarking, diagnostics
│   ├── analytics/           # Provenance, reporting, dashboards
│   ├── models/              # Data generators
│   └── robustness/          # Adaptive preprocessing & stress tests
├── docs/                    # Sphinx documentation & tutorials
├── notebooks/               # Markdown notebooks + supporting artefacts
├── examples/                # Minimal usage examples
├── scripts/                 # Reproducible benchmarking pipelines
└── tests/                   # Pytest suite (quick profile by default)
```

---

## Testing

```bash
python -m pytest                       # quick profile exercises
python -m pytest --cov=lrdbenchmark    # add coverage
```

---

## Contributing

We welcome improvements to estimators, diagnostics, documentation, and tutorials.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/improvement`
3. Run the test suite (see above)
4. Submit a pull request describing the change and relevant use-cases

Please consult `CONTRIBUTING.md` for coding standards and review expectations.

---

## Citation

```bibtex
@software{chin2024lrdbenchmark,
  author  = {Chin, Davian R.},
  title   = {lrdbenchmark: A Comprehensive Framework for Long-Range Dependence Estimation},
  version = {2.3.1},
  year    = {2024},
  doi     = {10.5281/zenodo.17534599},
  url     = {https://github.com/dave2k77/lrdbenchmark}
}
```

---

## Licence & Support

- **Licence**: MIT (see [`LICENSE`](LICENSE))
- **Issues & feature requests**: <https://github.com/dave2k77/lrdbenchmark/issues>
- **Discussions**: <https://github.com/dave2k77/lrdbenchmark/discussions>
- **Documentation**: <https://lrdbenchmark.readthedocs.io/>

Made with care for the time-series community. If you publish results using lrdbenchmark, please share them – the benchmarking suite evolves with real-world feedback.









