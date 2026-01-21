# lrdbenchmark v2.4.0 Release Notes

## 🚀 New Features

### Phase 1: Nonstationarity Testing

**Time-varying H generators** (`generation/nonstationary_generator.py`)
- `RegimeSwitchingProcess`: Abrupt H transitions at specified change points
- `ContinuousDriftProcess`: Smooth H(t) evolution (linear, sinusoidal, logistic)
- `StructuralBreakProcess`: Level shifts and variance changes
- `EnsembleTimeAverageProcess`: Ergodicity testing for aging phenomena

**Structural Break Detection** (`analysis/diagnostics.py`)
- New `StructuralBreakDetector` class with CUSUM, Recursive CUSUM, Chow test, and ICSS algorithm

### Phase 2: Advanced Robustness

**Critical regime models** (`generation/critical_regime_generator.py`)
- `OrnsteinUhlenbeckProcess`: Time-varying friction for transient criticality
- `SubordinatedProcess`: Subordinated Brownian motion (nonequilibrium)
- `FractionalLevyMotion`: Heavy-tailed LRD (α<2 stable regimes)
- `SOCAvalancheModel`: Bak-Tang-Wiesenfeld self-organized criticality

**Enhanced uncertainty quantification**
- Studentized bootstrap intervals with bias correction
- `CoverageAnalyzer` for Monte Carlo CI coverage probability analysis

**Surrogate data testing** (`generation/surrogate_generator.py`)
- `IAFFTSurrogate`: Preserve power spectrum and amplitude distribution
- `PhaseRandomizedSurrogate`: Phase randomization
- `ARSurrogate`: Autoregressive surrogates for linear null hypothesis

### CLI Benchmarks

- New `run_classical_failure_benchmark.py` script with `--profile quick|standard|full`
- Checkpointing for cloud/HPC compatibility
- Progress tracking with ETA estimation

---

## 📦 Containerization

- Added `Dockerfile` for reproducible environments
- Added `.dockerignore` for optimized build context

---

## 📚 Documentation

- Updated README with simplified installation (accelerators are optional/auto-detected)
- Added Command-Line Benchmarks section with CLI options table
- New API docs: `generation.rst`, `diagnostics.rst`, `uncertainty.rst`
- Updated `quickstart.rst` with nonstationarity and surrogate testing examples
- Added `experimental_protocol.md` publication template

---

## 🔧 Fixes

- Fixed `publish-robust.yml` duplicate YAML document error
- Updated GitHub Actions `setup-python` from v4 to v5
- Removed deprecated `publish.yml` and `hello-world.yml` workflows

---

## Installation

```bash
pip install lrdbenchmark==2.4.0
```

## Quick Start

```python
from lrdbenchmark.generation import RegimeSwitchingProcess, IAFFTSurrogate
from lrdbenchmark.analysis.diagnostics import StructuralBreakDetector

# Generate nonstationary data
gen = RegimeSwitchingProcess(h_regimes=[0.3, 0.8])
result = gen.generate(1000)

# Detect structural breaks
detector = StructuralBreakDetector()
breaks = detector.detect_all(result['signal'])
```

## CLI Usage

```bash
python scripts/benchmarks/run_classical_failure_benchmark.py --profile quick
```
