# Benchmarking Methodology: MSE Estimator vs Classical Hurst Exponent Estimators

## 1. Overview

This document describes the methodology used to benchmark the Multiscale Entropy (MSE) estimator — specifically, the Refined Composite Multiscale Sample Entropy (RCMSE) variant — against six classical Hurst exponent estimators. The benchmarking framework evaluates estimator accuracy, robustness, and reliability across three progressively challenging tiers:

1. **Tier 1 — Clean Synthetic Data**: Baseline accuracy on uncontaminated fractional Gaussian noise (fGn) with known Hurst parameters.
2. **Tier 2 — Controlled Contamination**: Robustness against isolated, parameterised confounding effects (noise, artifacts, trends).
3. **Tier 3 — Realistic Domain-Specific Scenarios**: Robustness against complex, domain-calibrated contamination profiles mimicking real-world data acquisition conditions.

---

## 2. Estimators Under Evaluation

### 2.1 Entropy-Based Estimator (Proposed)

| Estimator | Method | Input Type | Primary Output |
|-----------|--------|------------|----------------|
| **MSE (RCMSE)** | Refined Composite Multiscale Sample Entropy | fBm (cumulative sum of fGn) | Complexity Index (CI) |

The MSE estimator computes Sample Entropy at multiple coarse-grained temporal scales (1 to `max_scale`, default 15 in benchmarks). The primary native metric is the **Complexity Index** — the area under the entropy-versus-scale curve. An approximate Hurst parameter is derived from the MSE curve slope using an empirical calibration:

```
H_approx = 0.5 + 0.5 × (1 − |slope| / reference_slope)
```

where `reference_slope = 0.5` (the typical entropy decay rate for white noise). A flat MSE curve (|slope| ≈ 0, characteristic of 1/f noise) maps to H ≈ 1.0 (strong long-range dependence), while a steeply decaying curve maps to H ≈ 0.5 (no LRD). The result is clamped to [0, 1].

**Parameters used in benchmarks:**
- Embedding dimension: `m = 2`
- Similarity tolerance: `r = 0.15` (fraction of standard deviation)
- Maximum scale: `max_scale = 15`
- Entropy computation backend: `EntropyHub` (RCMSE variant with `Refined=True`)

### 2.2 Classical Estimators (Baselines)

| Estimator | Category | Method | Input Type |
|-----------|----------|--------|------------|
| **DFA** | Temporal | Detrended Fluctuation Analysis | fBm or fGn (varies by benchmark) |
| **R/S** | Temporal | Rescaled Range Analysis | fBm or fGn (varies by benchmark) |
| **Higuchi** | Temporal | Higuchi Fractal Dimension → H mapping | fBm or fGn (varies by benchmark) |
| **GPH** | Spectral | Geweke–Porter-Hudak log-periodogram regression | fGn |
| **Periodogram** | Spectral | Power spectral density slope estimation | fGn |
| **Whittle MLE** | Spectral | Whittle maximum likelihood estimation | fGn |

> **Note on input types:** Temporal estimators (DFA, R/S, Higuchi) operate on either fBm or fGn depending on the benchmark tier. When applied to fBm, α = H + 1; when applied to fGn, α = H directly. In the contaminated and realistic benchmarks, the input type is standardised for fair comparison (fGn for temporal estimators in the realistic benchmark; fBm in the contaminated benchmark). Spectral estimators (GPH, Periodogram, Whittle) consistently operate on fGn. MSE consistently operates on fBm (cumulative sum of fGn) to measure persistence across scales.

---

## 3. Synthetic Data Generation

### 3.1 Fractional Gaussian Noise (fGn) via Spectral Synthesis

All benchmarks use approximate fGn generated via the **Davies-Harte spectral method**:

1. Compute the one-sided power spectrum of fGn at frequency *f*:

   ```
   S(f) ∝ |f|^{−(2H − 1)}
   ```

2. Assign random uniform phases in [0, 2π) in the frequency domain.
3. Compute the inverse FFT to produce a time-domain realisation.
4. Standardise (zero-mean, unit variance).

**Parameters:**
- Series length: `N = 2048`
- True Hurst parameter values: `H ∈ {0.3, 0.5, 0.7, 0.9}` (Tier 1) or `H = 0.7` (Tiers 2–3)
- Random seed: `seed = 42` (reproducibility)

### 3.2 Fractional Brownian Motion (fBm)

Where fBm is required (e.g., for the MSE estimator or temporal methods operating on the cumulative process), it is produced as the cumulative sum of the fGn series:

```
fBm[t] = Σ_{i=1}^{t} fGn[i]
```

---

## 4. Tier 1: Baseline Accuracy on Clean Data

### 4.1 Protocol

For each true Hurst value H ∈ {0.3, 0.5, 0.7, 0.9}:

1. Generate an fGn series of length N = 2048 with the target H.
2. Derive fBm by cumulative summation (for estimators requiring it).
3. Apply each estimator to its appropriate input type.
4. Record the estimated H and execution time.

### 4.2 Metrics

- **Estimated H** (`H_est`): The Hurst parameter returned by each estimator.
- **Absolute Error**: `|H_est − H_true|`
- **Execution Time**: Wall-clock time per estimation (measured via `time.perf_counter()`).

### 4.3 Purpose

This tier establishes the fundamental accuracy of each estimator under idealised conditions, providing a baseline against which contaminated-data performance is compared.

---

## 5. Tier 2: Controlled Contamination Robustness

### 5.1 Contamination Model

The `ContaminationModel` class applies isolated, parameterised confounding effects to the clean signals. Each contamination is applied independently (i.e., one at a time) to both the fBm and fGn versions of the clean signal with `H_true = 0.7`.

### 5.2 Contamination Scenarios

| # | Scenario | Method | Parameters |
|---|----------|--------|------------|
| 1 | **Clean (baseline)** | No contamination | — |
| 2 | **Additive Gaussian noise** | `add_noise_gaussian()` | std = 0.3 × std(signal), SNR ≈ 10 dB |
| 3 | **Outlier spikes** | `add_artifact_spikes()` | probability = 0.02, amplitude = 5.0× |
| 4 | **Level shifts** | `add_artifact_level_shifts()` | probability = 0.003, amplitude = 3.0× |
| 5 | **Linear trend** | `add_trend_linear()` | slope = 0.02 |
| 6 | **Seasonal trend** | `add_trend_seasonal()` | period = 100, amplitude = 1.0 |
| 7 | **Impulsive noise** | `add_noise_impulsive()` | probability = 0.01, amplitude = 8.0× |
| 8 | **Coloured noise (1/f²)** | `add_noise_colored()` | spectral exponent = 2.0, std = 0.5 |

### 5.3 Protocol

For each contamination scenario:

1. Apply the contamination to the clean fBm and clean fGn signals.
2. Run each estimator on its appropriate contaminated input.
3. Handle NaN values via linear interpolation (if the fraction of valid samples ≥ 100) and clip infinities to ±10¹⁰.
4. Record `H_est` and compute the signed error (`H_est − H_true`).

### 5.4 Metrics

- **Signed Error**: `H_est − H_true` (positive = overestimate)
- **Mean Absolute Error (MAE)**: Averaged across all contamination scenarios per estimator
- **Maximum Absolute Error** (`Max|err|`): Worst-case deviation per estimator
- **Robustness Summary Table**: MAE and Max|err| per estimator across all scenarios

---

## 6. Tier 3: Realistic Domain-Specific Scenarios

### 6.1 Contamination Factory

The `ContaminationFactory` generates complex, multi-component confounding profiles that mimic realistic data acquisition conditions for specific application domains. Each scenario applies a combination of domain-calibrated artifacts at default intensity via the `apply_confounding()` method with `random_seed = 42`.

### 6.2 Domain Groups and Scenarios

The 27 scenarios are organised into 7 domain groups:

#### Financial (3 scenarios)
| Scenario | Description |
|----------|-------------|
| Financial Crash | Sudden price drops with volatility spikes |
| Volatility Clustering | GARCH-like heteroscedastic variance dynamics |
| Regime Change | Abrupt shifts in mean/variance properties |

#### Physiological (3 scenarios)
| Scenario | Description |
|----------|-------------|
| Sensor Drift | Gradual baseline wander from sensor degradation |
| Motion Artifacts | Transient high-amplitude disturbances from subject movement |
| Equipment Failure | Intermittent signal dropout and saturation |

#### Environmental (3 scenarios)
| Scenario | Description |
|----------|-------------|
| Seasonal | Periodic oscillations from climate/diurnal cycles |
| Extreme Events | Rare, high-magnitude outlier events |
| Measurement Drift | Calibration drift in environmental sensors |

#### Network (3 scenarios)
| Scenario | Description |
|----------|-------------|
| Bursts | Packet-burst traffic patterns |
| Congestion | Sustained throughput degradation |
| Equipment Failure | Router/switch failures causing signal loss and recovery transients |

#### Industrial (3 scenarios)
| Scenario | Description |
|----------|-------------|
| Calibration Drift | Slow sensor de-calibration over time |
| Sensor Aging | Progressive degradation of sensor sensitivity |
| Environmental Interference | External electromagnetic or thermal interference |

#### EEG (8 scenarios)
| Scenario | Description |
|----------|-------------|
| Ocular Artifacts | Blink and saccade artifacts |
| Muscle Artifacts | EMG contamination from facial/jaw muscle activity |
| Cardiac Artifacts | QRS complex leakage into EEG channels |
| Electrode Popping | Sudden impedance changes at electrode contacts |
| Electrode Drift | Slow electrode impedance changes |
| 60 Hz Noise | Power-line interference |
| Sweat Artifacts | Galvanic skin response / slow potential drifts |
| Movement Artifacts | Gross head/body movement |

#### Mixed (3 scenarios)
| Scenario | Description |
|----------|-------------|
| Mixed Light | Low-intensity composition of multiple confounds |
| Mixed Moderate | Medium-intensity multi-confound composition |
| Mixed Severe | High-intensity multi-confound composition |

### 6.3 Protocol

1. Generate a single clean fGn series with `H_true = 0.7`, `N = 2048`, `seed = 42`.
2. Derive the clean fBm by cumulative summation.
3. Establish baseline: run all estimators on the clean signals.
4. For each scenario within each domain group:
   a. Apply domain-specific contamination via `ContaminationFactory.apply_confounding()` at default intensity.
   b. Run each estimator on the contaminated signal appropriate to its input type.
   c. Record `H_est`, compute signed error and absolute error.
5. NaN-handling: interpolate if ≥ 100 valid samples remain; otherwise return NaN.

### 6.4 Metrics

#### Per-Scenario Metrics
- **Estimated H** and **signed error** for each estimator on each scenario.

#### Per-Domain Summary
- **Domain MAE**: Mean absolute error averaged across scenarios within each domain group, per estimator.

#### Overall Summary
- **Overall MAE**: Mean absolute error across all 27 scenarios per estimator.
- **Maximum Absolute Error**: Worst-case deviation per estimator.
- **Robustness Ranking**: Estimators ranked by overall MAE (ascending), with Max|err| reported.

---

## 7. Error Handling and Data Preprocessing

All benchmarks apply consistent error handling:

| Condition | Action |
|-----------|--------|
| NaN values in data | Linear interpolation if ≥ 100 valid samples; otherwise return NaN |
| Infinite values in data | Clip to ±10¹⁰ |
| Estimator exception | Return NaN for `H_est` |
| Fewer than 3 finite entropy values (MSE) | Return `H = 0.5` (uninformative prior) |

---

## 8. Evaluation Framework Summary

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        BENCHMARKING FRAMEWORK                          │
├────────────┬─────────────────────┬──────────────────────────────────────┤
│   Tier     │     Data Regime     │          Purpose                    │
├────────────┼─────────────────────┼──────────────────────────────────────┤
│ Tier 1     │ Clean fGn/fBm       │ Baseline accuracy across H values  │
│ (4 runs)   │ H ∈ {0.3,0.5,0.7,0.9} │                                 │
├────────────┼─────────────────────┼──────────────────────────────────────┤
│ Tier 2     │ Controlled          │ Robustness to isolated confounds   │
│ (8 runs)   │ contamination       │                                    │
├────────────┼─────────────────────┼──────────────────────────────────────┤
│ Tier 3     │ Domain-specific     │ Robustness under realistic,        │
│ (27 runs)  │ realistic scenarios │ multi-component contamination      │
└────────────┴─────────────────────┴──────────────────────────────────────┘
```

### Key Design Decisions

1. **Fixed ground truth (H = 0.7)** for Tiers 2–3 places the true value in the persistent LRD regime, which is the most relevant operating range for biomedical, financial, and environmental applications.

2. **Input-type matching:** Each estimator receives data in the form (fBm vs fGn) appropriate to its theoretical derivation, ensuring fair comparison rather than penalising estimators for receiving incorrectly preprocessed input.

3. **Single realisation with fixed seed** (`seed = 42`) ensures exact reproducibility across runs and enables meaningful comparison of estimator responses to identical perturbations.

4. **MSE Hurst mapping is approximate:** The Complexity Index is the primary MSE output. The derived `hurst_parameter` is a heuristic mapping provided solely for cross-estimator comparison. This is explicitly noted in all benchmark outputs.

---

## 9. Reproducibility

All experiments are fully reproducible using the scripts in the `scripts/` directory:

| Script | Tier | Output |
|--------|------|--------|
| `benchmark_entropy_vs_classical.py` | 1 | `benchmark_results.txt` |
| `benchmark_contaminated.py` | 2 | `benchmark_contaminated_results.txt` |
| `benchmark_realistic.py` | 3 | `benchmark_realistic_results.txt` |

**Environment requirements:**
- Python ≥ 3.8
- `numpy`, `scipy`, `EntropyHub`
- `lrdbenchmark` library (installed in development mode)

**Execution:**
```bash
python scripts/benchmark_entropy_vs_classical.py
python scripts/benchmark_contaminated.py
python scripts/benchmark_realistic.py
```

---

## 10. References

1. Costa, M., Goldberger, A. L., & Peng, C.-K. (2005). Multiscale entropy analysis of biological signals. *Physical Review E*, 71(2), 021906.
2. Peng, C.-K., Havlin, S., Stanley, H. E., & Goldberger, A. L. (1995). Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series. *Chaos*, 5(1), 82–87. (DFA)
3. Hurst, H. E. (1951). Long-term storage capacity of reservoirs. *Transactions of the American Society of Civil Engineers*, 116, 770–808. (R/S)
4. Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory. *Physica D*, 31(2), 277–283. (Higuchi)
5. Geweke, J., & Porter-Hudak, S. (1983). The estimation and application of long memory time series models. *Journal of Time Series Analysis*, 4(4), 221–238. (GPH)
6. Whittle, P. (1953). Estimation and information in stationary time series. *Arkiv för Matematik*, 2(5), 423–434. (Whittle MLE)
