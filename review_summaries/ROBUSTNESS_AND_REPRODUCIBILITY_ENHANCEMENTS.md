# Robustness Stress Panels and Reproducibility Scaffolding Enhancements

**Date:** 2025-01-08  
**Status:** Implemented  
**Related Issues:** Based on `lrdbenchmark_evaluation_review.md` recommendations

## Overview

This document describes the implementation of two major enhancements to the LRDBenchmark framework:

1. **Robustness Stress Panels**: Standard missingness/burst/regime-shift/oscillation suites with before/after H comparisons, wired into the benchmark workflow
2. **Reproducibility Scaffolding**: A unified YAML/JSON experiment schema for data generation, estimator knobs, preprocessing, analytics, and emitted provenance artefacts per run

These enhancements address critical gaps identified in the evaluation review document, specifically:

- **Robustness panes**: Standardized stress panels for missingness patterns (MCAR/MAR/MNAR), intermittent bursts, regime shifts, and additive oscillations, with before/after H to quantify preprocessing-induced bias and variance
- **Reproducible configs**: A single experiment schema (YAML/JSON) for dataset generation, estimator knobs, preprocessing, and analytics, with provenance bundles per result row to preclude hidden differences across runs

---

## 1. Robustness Stress Panels

### 1.1 Enhanced RobustnessStressTester

**File:** `lrdbenchmark/analysis/advanced_metrics.py`

#### New Missingness Patterns

The `RobustnessStressTester` class was enhanced to support three types of missingness patterns:

1. **MCAR (Missing Completely At Random)**: Missingness is independent of observed and unobserved data
   - Method: `_apply_missing_mcar()`
   - Random uniform probability of missingness

2. **MAR (Missing At Random)**: Missingness depends on observed values
   - Method: `_apply_missing_mar()`
   - Higher values have higher probability of being missing
   - Uses normalized data values to compute missingness probabilities

3. **MNAR (Missing Not At Random)**: Missingness depends on the unobserved values themselves
   - Method: `_apply_missing_mnar()`
   - Extreme values (based on absolute values) are more likely to be missing
   - Uses percentile-based thresholding

#### New Oscillation Scenario

Added `_apply_oscillation()` method to introduce additive periodic components:
- Configurable amplitude (relative to data std)
- Configurable frequency (as fraction of Nyquist)
- Converts frequency to period and applies sinusoidal oscillation

#### Enhanced Before/After H Comparisons

Each scenario result now includes comprehensive before/after metrics:

```python
{
    "status": "ok",
    "before_h": float,           # H before stress test
    "after_h": float,            # H after stress test
    "delta_h": float,             # Absolute change
    "abs_delta_h": float,         # Absolute magnitude of change
    "relative_delta_h": float,    # Relative change (%)
    "execution_time": float,
    "true_value": float,           # If available
    "bias": float                  # Bias relative to true value
}
```

#### Configuration

The stress tester is now configurable via protocol config:

```yaml
benchmark:
  robustness:
    enabled: true
    random_state: null  # Auto-generated if null
    config:
      missing_rate: 0.1
      block_fraction: 0.2
      regime_shift: 0.5
      burst_rate: 0.05
      burst_magnitude: 3.0
      oscillation_amplitude: 0.3
      oscillation_frequency: 0.1
```

### 1.2 Integration into Benchmark Workflow

**File:** `lrdbenchmark/analysis/benchmark.py`

#### Initialization

The `ComprehensiveBenchmark` class now initializes a `RobustnessStressTester` instance:

```python
robustness_cfg = benchmark_cfg.get("robustness", {})
robustness_enabled = robustness_cfg.get("enabled", True)
if robustness_enabled:
    robustness_config = robustness_cfg.get("config", {})
    self.robustness_tester = RobustnessStressTester(
        random_state=robustness_cfg.get("random_state"),
        config=robustness_config
    )
```

#### Execution

Robustness panels are automatically run for each successful estimator test:

```python
# Robustness stress panels: before/after H comparisons
robustness_panel = None
if self.robustness_tester is not None and hurst_est is not None:
    robustness_panel = self.robustness_tester.run_panels(
        estimator=estimator,
        data=data,  # Original data before preprocessing
        baseline_result=result,  # Baseline result with H estimate
        true_value=true_params.get("H"),
    )
```

#### Result Structure

Each test result includes a `robustness_panel` field:

```python
{
    "baseline_estimate": float,
    "scenarios": {
        "missing_mcar": {...},
        "missing_mar": {...},
        "missing_mnar": {...},
        "missing_block": {...},
        "regime_shift": {...},
        "burst_noise": {...},
        "oscillation": {...}
    },
    "summary": {
        "n_scenarios": int,
        "successful_scenarios": int,
        "mean_abs_delta": float,
        "max_abs_delta": float,
        "baseline_estimate": float
    }
}
```

---

## 2. Reproducibility Scaffolding

### 2.1 Unified YAML Experiment Schema

**File:** `config/benchmark_protocol.yaml`

Converted the protocol configuration from JSON to YAML and created a comprehensive unified schema covering:

#### Data Generation Parameters

```yaml
data_generation:
  default_length: 1000
  models:
    fBm:
      H: 0.7
      sigma: 1.0
      enabled: true
    # ... other models
  random_seed: null  # Auto-generated if null
  contamination:
    enabled: false
    type: null
    level: 0.1
```

#### Preprocessing Configuration

```yaml
preprocessing:
  outlier_threshold: 3.0
  winsorize_limits: [0.01, 0.99]
  detrend: true
  apply_winsorize: true
  detrend_order: 1
  tapering: null  # Options: hann, hamming, blackman, null
```

#### Scale Selection Parameters

```yaml
scale_selection:
  spectral:
    min_freq_ratio: 0.01
    max_freq_ratio: 0.1
    low_freq_trim: 0.0
    selection_method: "auto"
  temporal:
    min_window: 16
    max_window: 256
    window_density: "log"
    selection_method: "auto"
  wavelet:
    min_level: 1
    max_level: 8
    wavelet: "db4"
    selection_method: "auto"
```

#### Estimator Configuration

```yaml
estimators:
  overrides:
    GPH:
      min_freq_ratio: 0.01
      max_freq_ratio: 0.08
    # ... other overrides
  categories:
    classical: true
    ML: true
    neural: true
  include: null  # null = all
  exclude: null  # null = none
```

#### Benchmark Settings

```yaml
benchmark:
  confidence_level: 0.95
  uncertainty:
    enabled: true
    n_block_bootstrap: 64
    # ... other settings
  robustness:
    enabled: true
    config: {...}
  advanced_metrics:
    enabled: true
    convergence_analysis: true
    bias_analysis: true
  significance_testing:
    enabled: true
    alpha: 0.05
    methods:
      - friedman
      - nemenyi
      - wilcoxon
```

#### Analytics Configuration

```yaml
analytics:
  error_analysis:
    enabled: true
    metrics: [mae, mse, rmse, bias, variance]
  stratified_reporting:
    enabled: true
    stratify_by:
      - true_hurst
      - data_length
      - tail_index
      - estimator_category
  visualization:
    enabled: true
    formats: [png, pdf]
    dpi: 300
```

#### Output Configuration

```yaml
output:
  results_dir: "benchmark_results"
  save_results: true
  save_provenance: true
  provenance:
    format: "json"  # Options: json, yaml
    include_full_config: true
    include_environment: true
    include_git_info: true
```

### 2.2 YAML Support in Config Loader

**File:** `lrdbenchmark/analysis/benchmark.py`

Enhanced `_load_protocol_config()` to support both YAML and JSON:

```python
def _load_protocol_config(self, path: Path) -> Dict[str, Any]:
    # Try YAML first, then JSON
    if path.suffix.lower() in ['.yaml', '.yml']:
        try:
            import yaml
            config_data = yaml.safe_load(f)
        except ImportError:
            # Fallback to JSON parsing
            config_data = json.load(f)
    else:
        config_data = json.load(f)
    
    # Map new unified schema to legacy format for backward compatibility
    if "data_generation" in merged:
        if "models" in merged["data_generation"]:
            merged["data_models"] = merged["data_generation"]["models"]
    
    # ... other mappings
```

### 2.3 Enhanced Provenance Bundle System

#### Experiment-Level Provenance

**Method:** `_build_provenance_bundle()`

The provenance bundle now captures:

1. **Experiment Metadata**
   - Experiment ID (auto-generated if not provided)
   - Timestamp
   - Protocol version and path
   - Description

2. **Data Generation Parameters**
   - Default length
   - Model configurations
   - Random seed

3. **Preprocessing Configuration**
   - All preprocessing settings

4. **Scale Selection Parameters**
   - Spectral, temporal, and wavelet settings

5. **Estimator Configuration**
   - Overrides
   - Categories tested
   - Estimator listing

6. **Benchmark Settings**
   - Confidence level
   - Uncertainty quantification settings
   - Robustness configuration
   - Advanced metrics settings
   - Significance testing configuration

7. **Analytics Configuration**
   - Error analysis settings
   - Stratified reporting settings
   - Visualization settings

8. **Results Summary**
   - Total tests
   - Success rate
   - Data models tested
   - Estimators tested

9. **Environment Information** (optional)
   - Python version
   - Platform information
   - Processor and machine details

10. **Git Information** (optional, if available)
    - Commit hash
    - Branch name

11. **Full Config** (optional)
    - Complete protocol configuration

#### Per-Result-Row Provenance

**Method:** `_build_result_row_provenance()`

Each result row includes a lightweight provenance artifact:

```python
{
    "experiment_id": str,
    "timestamp": str,
    "estimator": str,
    "data_model": str,
    "data_params": dict,
    "preprocessing": dict,
    "estimated_hurst": float,
    "true_hurst": float,
    "error": float,
    "robustness_panel": dict,
    "uncertainty": dict,
    "protocol_version": str,
    "protocol_path": str
}
```

This ensures that each result row can be traced back to its exact experimental conditions, enabling full reproducibility.

---

## 3. Additional Enhancements (User Contributions)

### 3.1 Diagnostic Systems

#### Power-Law Diagnostics

Added `PowerLawDiagnostics` class for automated log-log scaling diagnostics:
- Goodness-of-fit checks (R² validation)
- Minimum points validation
- Slope and intercept validation

#### Scale Window Sensitivity Analysis

Added `ScaleWindowSensitivityAnalyser` for:
- Perturbation analysis (testing sensitivity to scale window changes)
- Leave-one-out analysis
- Influence analysis for scale cutoffs

#### Scaling Influence Analysis

Integrated `ScalingInfluenceAnalyzer` for:
- Breakpoint detection
- Residual tests to catch non-power-law segments
- Influence analysis for scale cutoffs

### 3.2 Enhanced Significance Testing

#### Nemenyi Test

Added Nemenyi post-hoc test for multiple comparisons:
- Critical difference calculation
- Pairwise rank difference comparisons
- Better estimator identification

#### Sign Test

Added paired sign test:
- Binomial test on paired differences
- Better estimator identification
- Holm correction support

#### Enhanced Test Results

Significance test results now include:
- `global_null_rejected`: Boolean indicating if Friedman test rejected null
- `estimator_markers`: Dictionary tracking which estimators beat others in each test
- `better_estimator`: Identified for each pairwise comparison
- `alpha`: Significance level used

### 3.3 Enhanced Uncertainty Calibration

#### Multi-Method Tracking

Enhanced `_record_uncertainty_event()` to track all uncertainty methods:
- Block bootstrap
- Wavelet bootstrap
- Parametric Monte Carlo

#### Calibration Summary

Added `_attach_uncertainty_calibration_summary()` to:
- Summarise calibration across methods
- Generate calibration plots
- Track coverage by method and estimator family

### 3.4 Stratified Reporting

Integrated `StratifiedReportGenerator` for:
- Enhanced stratified reports by true H, data length, tail index, estimator category
- Multiple output formats (markdown, JSON, CSV)
- Comprehensive performance breakdowns

### 3.5 Provenance Tracker

Added `ProvenanceTracker` class for:
- Centralised provenance management
- Consistent provenance format across experiments
- Integration with output configuration

---

## 4. Usage Examples

### 4.1 Running Benchmarks with Robustness Panels

```python
from lrdbenchmark.analysis.benchmark import ComprehensiveBenchmark

benchmark = ComprehensiveBenchmark()

# Robustness panels are automatically enabled by default
results = benchmark.run_comprehensive_benchmark(
    data_length=1000,
    benchmark_type="comprehensive"
)

# Access robustness panel for each result
for model_name, model_data in results["results"].items():
    for result in model_data["estimator_results"]:
        robustness = result.get("robustness_panel")
        if robustness:
            print(f"Estimator: {result['estimator']}")
            print(f"Baseline H: {robustness['baseline_estimate']}")
            for scenario, scenario_result in robustness["scenarios"].items():
                if scenario_result.get("status") == "ok":
                    print(f"  {scenario}: {scenario_result['before_h']} -> {scenario_result['after_h']}")
```

### 4.2 Configuring Robustness Panels

```yaml
# config/benchmark_protocol.yaml
benchmark:
  robustness:
    enabled: true
    random_state: 42
    config:
      missing_rate: 0.15
      block_fraction: 0.25
      regime_shift: 0.6
      burst_rate: 0.08
      burst_magnitude: 4.0
      oscillation_amplitude: 0.4
      oscillation_frequency: 0.15
```

### 4.3 Accessing Provenance Information

```python
# Experiment-level provenance
provenance = results["provenance"]
print(f"Experiment ID: {provenance['experiment_id']}")
print(f"Protocol Version: {provenance['protocol_version']}")
print(f"Git Commit: {provenance.get('git', {}).get('commit_hash')}")

# Per-row provenance
for model_name, model_data in results["results"].items():
    for result in model_data["estimator_results"]:
        row_provenance = result.get("provenance", {})
        print(f"Estimator: {row_provenance['estimator']}")
        print(f"Data Model: {row_provenance['data_model']}")
        print(f"Protocol Version: {row_provenance['protocol_version']}")
```

---

## 5. Benefits

### 5.1 Robustness Stress Panels

1. **Standardised Testing**: Consistent stress tests across all estimators
2. **Quantified Sensitivity**: Before/after H comparisons quantify estimator sensitivity
3. **Missingness Patterns**: Comprehensive coverage of MCAR/MAR/MNAR scenarios
4. **Real-World Scenarios**: Oscillations, bursts, and regime shifts reflect real data challenges
5. **Automated Integration**: Seamlessly integrated into benchmark workflow

### 5.2 Reproducibility Scaffolding

1. **Unified Schema**: Single configuration file for all experiment parameters
2. **Complete Provenance**: Every result row includes full provenance information
3. **Environment Tracking**: Captures Python version, platform, and git information
4. **Backward Compatible**: Supports both YAML and JSON formats
5. **Reproducible Experiments**: All settings needed for reproduction are captured

---

## 6. Files Modified

### Core Implementation

- `lrdbenchmark/analysis/advanced_metrics.py`: Enhanced `RobustnessStressTester` class
- `lrdbenchmark/analysis/benchmark.py`: Integrated robustness panels and enhanced provenance
- `config/benchmark_protocol.yaml`: Unified YAML experiment schema

### New Dependencies

- `yaml` (PyYAML): For YAML config file support (optional, falls back to JSON if not available)

---

## 7. Testing Recommendations

### 7.1 Robustness Panels

1. **Test Missingness Patterns**: Verify MCAR/MAR/MNAR produce expected missingness distributions
2. **Test Oscillation**: Verify oscillation amplitude and frequency parameters work correctly
3. **Test Before/After Comparisons**: Verify delta calculations are correct
4. **Test Integration**: Verify robustness panels run for all estimator types

### 7.2 Reproducibility

1. **Test YAML Loading**: Verify YAML configs load correctly
2. **Test JSON Compatibility**: Verify JSON configs still work
3. **Test Provenance**: Verify provenance bundles contain all expected fields
4. **Test Per-Row Provenance**: Verify each result row has provenance information
5. **Test Git Integration**: Verify git information is captured when available

---

## 8. Future Enhancements

### 8.1 Robustness Panels

- [ ] Add more missingness patterns (e.g., monotone missingness)
- [ ] Add trend contamination scenarios
- [ ] Add multiplicative noise scenarios
- [ ] Add configurable scenario severity levels
- [ ] Add visualisation of robustness panel results

### 8.2 Reproducibility

- [ ] Add provenance validation
- [ ] Add provenance comparison tools
- [ ] Add experiment replay functionality
- [ ] Add provenance database for tracking experiments over time
- [ ] Add provenance-based result filtering and querying

---

## 9. References

- `lrdbenchmark_evaluation_review.md`: Original review document with recommendations
- `config/benchmark_protocol.yaml`: Unified experiment schema
- `lrdbenchmark/analysis/advanced_metrics.py`: Robustness stress tester implementation
- `lrdbenchmark/analysis/benchmark.py`: Benchmark workflow integration

---

## 10. Changelog

### 2025-01-08

- **Added**: MCAR/MAR/MNAR missingness patterns to `RobustnessStressTester`
- **Added**: Oscillation scenario to robustness stress panels
- **Added**: Enhanced before/after H comparison metrics
- **Added**: Unified YAML experiment schema
- **Added**: Enhanced provenance bundle system
- **Added**: Per-result-row provenance artifacts
- **Added**: YAML support in config loader
- **Added**: Diagnostic systems integration (user contribution)
- **Added**: Enhanced significance testing (user contribution)
- **Added**: Enhanced uncertainty calibration (user contribution)
- **Added**: Stratified reporting integration (user contribution)

---

**Document Status**: Complete  
**Last Updated**: 2025-01-08

