# Protocol Standardisation & Enhanced Diagnostics Implementation

**Date**: 2025-11-08  
**Version**: 2.0

## Summary

This document summarises the comprehensive enhancements made to the LRDBenchmark framework to address the recommendations from the evaluation review document, focusing on:

1. Protocol standardisation and provenance tracking
2. Richer stratified reporting
3. Comprehensive diagnostics and scale-window analysis

---

## 1. Protocol Standardisation & Provenance

### Enhanced Protocol Schema (`config/benchmark_protocol.yaml`)

**Version**: 2.0

The protocol configuration has been significantly expanded to centrally capture all critical settings:

#### Key Additions:

- **Protocol Metadata**: Name, description, authors, version tracking
  
- **Comprehensive Preprocessing Configuration**:
  - Detrending: method, order, edge handling
  - Tapering: method (Tukey), alpha parameter, options
  - Outlier removal: MAD-based method, thresholds, iterations
  - Winsorising: limits, options
  - Filtering: lowpass/highpass options

- **Detailed Scale Selection Heuristics**:
  - **Spectral domain**: min/max frequency ratios, low/high-frequency trimming, trimming method, min scales, log spacing, Nyquist exclusion
  - **Temporal domain**: min/max windows, window density (log/linear), step type (geometric), min/max scales
  - **Wavelet domain**: min/max levels, wavelet type, boundary conditions, edge coefficient handling

- **Diagnostics Configuration**:
  - Log-log checks: R² thresholds, min points, breakpoint detection, influence analysis
  - Residual tests: normality (Shapiro), autocorrelation lags, heteroscedasticity
  - Goodness-of-fit: AIC/BIC/adjusted R², confidence bands
  - Scale-window sensitivity: perturbation levels, leave-one-out analysis

- **Stratification Settings**:
  - Hurst bands: configurable ranges for anti-persistent to ultra-persistent regimes
  - Length bands: very short to ultra-long data
  - Tail classes: Gaussian, linear-LRD, multifractal, alpha-stable
  - Contamination tracking: by type and level

### New Provenance Tracking System

**File**: `lrdbenchmark/analytics/provenance.py`

#### Features:

- **Comprehensive Capture**:
  - Full protocol configuration
  - Python environment (version, implementation, compiler)
  - System information (platform, architecture, processor)
  - Package versions (numpy, scipy, pandas, sklearn, pywt, jax, numba, torch, etc.)
  - Hardware (CPU count, memory, GPU information)
  - Runtime metadata (tests, success rates, data models)
  - Git information (commit hash, branch, uncommitted changes, remote URL)

- **Reproducibility Verification**:
  - `verify_reproducibility()` method compares current environment against stored provenance
  - Identifies mismatches in Python version, package versions, protocol version
  - Provides status assessment: "fully_reproducible", "likely_reproducible", "possibly_not_reproducible"

- **Provenance Bundle Structure**:
  ```json
  {
    "provenance_version": "2.0",
    "timestamp": "...",
    "benchmark_metadata": {...},
    "protocol": {
      "version": "2.0",
      "preprocessing": {...},
      "scale_selection": {...},
      "diagnostics": {...},
      "stratification": {...},
      ...
    },
    "environment": {...},
    "package_versions": {...},
    "hardware": {...},
    "runtime": {...},
    "git_info": {...}
  }
  ```

#### Integration:

- Provenance bundles are automatically generated for each benchmark run
- Saved as `benchmark_provenance_<timestamp>.json` in the results directory
- Embedded in result rows for fine-grained tracking

---

## 2. Richer Stratified Reporting

**File**: `lrdbenchmark/analytics/stratified_report_generator.py`

### Enhanced Stratification Dimensions

The new `StratifiedReportGenerator` class breaks out results by:

1. **True Hurst Parameter Bands** (for synthetic data):
   - Anti-persistent (H < 0.3)
   - Short-range (0.3 ≤ H < 0.5)
   - Borderline (0.5 ≤ H < 0.55)
   - Moderate persistence (0.55 ≤ H < 0.7)
   - Persistent (0.7 ≤ H < 0.85)
   - Ultra-persistent (H ≥ 0.85)

2. **Estimated Hurst Parameter Bands** (for all data):
   - Same bands as above, but based on estimates
   - Reveals regime-dependent biases where estimators systematically over/under-estimate

3. **Tail Classes**:
   - Gaussian (fBm, fGn)
   - Linear-LRD (ARFIMA)
   - Multifractal-heavy-tail (MRW)
   - Alpha-stable
   - Neural-fSDE

4. **Data Length Bands**:
   - Very short (≤256)
   - Short (257-512)
   - Medium (513-2048)
   - Long (2049-8192)
   - Ultra-long (>8192)

5. **Contamination Scenarios**:
   - Clean data
   - By contamination type and level

6. **Estimator Families**:
   - Classical (spectral, temporal, wavelet, multifractal)
   - Machine learning (RandomForest, GradientBoosting, SVR)
   - Neural networks (CNN, LSTM, GRU, Transformer)

7. **Cross-Stratifications**:
   - Hurst band × tail class
   - (Extensible to other combinations)

### Metrics Reported per Stratum

For each stratification dimension and band:

- **n**: Number of observations
- **Success rate**: Percentage of successful estimations
- **Mean/median/std error**: Error statistics
- **Min/max error**: Error range
- **Mean CI width**: Average confidence interval width
- **Coverage rate**: Empirical coverage of true values
- **Mean/std estimated H**: Distribution of estimates
- **Mean true H**: Average true Hurst parameter in band
- **Mean execution time**: Computational cost
- **Mean R²**: Fit quality
- **Mean convergence rate**: Convergence characteristics
- **Mean bias percentage**: Systematic bias
- **Data models**: Which data models contribute to the band
- **Estimators**: Which estimators were tested
- **Number of estimators**: Count of tested estimators

### Output Formats

Stratified reports are generated in multiple formats:

1. **Markdown** (`stratified_report_<timestamp>.md`):
   - Human-readable tables
   - Section headers for each stratification
   - Interpretative text

2. **JSON** (`stratified_report_<timestamp>.json`):
   - Machine-readable
   - Full nested structure with all metadata

3. **CSV** (multiple files):
   - `stratified_true_h_<timestamp>.csv`
   - `stratified_estimated_h_<timestamp>.csv`
   - `stratified_tail_class_<timestamp>.csv`
   - `stratified_length_<timestamp>.csv`
   - `stratified_contamination_<timestamp>.csv`
   - `stratified_estimator_family_<timestamp>.csv`

### Key Improvements Over Original

- **Estimated H stratification**: Previously only stratified by true H; now also by estimated H to reveal where methods place data
- **Richer metrics**: Convergence rates, bias percentages, R² values added
- **Cross-tabulations**: Multi-dimensional stratifications (e.g., H × tail class)
- **Automated generation**: Integrated into benchmark workflow, no manual post-processing needed

---

## 3. Diagnostics & Scale-Window Analysis

**File**: `lrdbenchmark/analysis/diagnostics.py`

### PowerLawDiagnostics Class

Comprehensive diagnostics for power-law fits in LRD estimation.

#### Features:

1. **Linearity Check**:
   - R² threshold validation (default: 0.5)
   - Runs test for residual randomness
   - Null hypothesis: residuals are random
   - Reports z-statistic and p-value

2. **Residual Analysis**:
   - **Normality test** (Shapiro-Wilk):
     - H₀: residuals are normally distributed
     - Statistic, p-value, pass/fail status
   - **Autocorrelation test** (Ljung-Box):
     - H₀: no autocorrelation in residuals up to lag k
     - Tests for serial correlation that would violate OLS assumptions
   - **Heteroscedasticity test** (Breusch-Pagan):
     - H₀: homoscedastic errors
     - Regresses squared residuals on log-scales
     - Chi-square test statistic and p-value
   - **Residual statistics**: mean, std, min, max, skewness, kurtosis

3. **Goodness-of-Fit Assessment**:
   - R², adjusted R²
   - **AIC** (Akaike Information Criterion): penalises complexity
   - **BIC** (Bayesian Information Criterion): stronger complexity penalty
   - MAE and RMSE in log-log space
   - Interpretations provided

4. **Breakpoint Detection**:
   - Two-segment regression via exhaustive search
   - F-test for significance of breakpoint
   - Reports:
     - Break scale location
     - Left and right segment slopes
     - Slope difference
     - R² for each segment
     - RSS improvement ratio

5. **Overall Quality Score** (0-1):
   - Weighted combination of:
     - R² (40%)
     - Normality (20%)
     - No autocorrelation (20%)
     - Homoscedasticity (20%)

6. **Warnings Generation**:
   - Human-readable warnings for:
     - Low R²
     - Non-normal residuals
     - Autocorrelated residuals
     - Heteroscedastic residuals
     - Significant breakpoints

### ScaleWindowSensitivityAnalyser Class

Analyses sensitivity of H estimates to scale window selection.

#### Features:

1. **Perturbation Analysis**:
   - Applies multiplicative perturbations to scale bounds (default: [0.9, 0.95, 1.05, 1.1])
   - Re-estimates with perturbed scale windows
   - Computes ΔH for each perturbation
   - Reports:
     - H estimate for each perturbation
     - ΔH relative to baseline
     - Success status

2. **Summary Statistics**:
   - Mean absolute ΔH
   - Maximum absolute ΔH
   - Standard deviation of ΔH
   - Number of successful perturbations

3. **Interpretation**:
   - **Low sensitivity** (|ΔH|_max < 0.05): Robust to scale window changes
   - **Moderate sensitivity** (0.05 ≤ |ΔH|_max < 0.1): Some variability
   - **High sensitivity** (|ΔH|_max ≥ 0.1): Highly sensitive, caution advised

4. **Leave-One-Out Analysis**:
   - (Framework in place for future enhancement)
   - Would systematically remove each scale point and re-estimate
   - Identify influential scale points

### ScalingInfluenceAnalyser Class

From `advanced_metrics.py`, now integrated into diagnostics workflow.

#### Features:

1. **Log-Log Fit**:
   - Linear regression in log₂-log₂ space
   - Reports slope, intercept, R², p-value, standard error

2. **Leave-One-Out Influence**:
   - Computes slope when each scale point is removed
   - Reports:
     - Removed scale
     - New slope
     - Slope change (ΔSlope)
     - Relative change (%)
   - Sorted by influence magnitude

3. **Breakpoint Detection**:
   - Two-segment regression
   - Reports break scale, left/right slopes, RSS

### Integration into Benchmark Workflow

#### In `run_single_estimator_test()`:

After estimator execution and before result assembly:

1. **Extract scale data** (`_extract_scale_data()` helper):
   - Tries various common keys: `scales`, `log_scales`, `windows`, `frequencies`, `wavelet_scales`
   - Tries statistic keys: `statistics`, `values`, `fluctuations`, `psd`, `power`, `variance`
   - Falls back to nested structures and estimator methods

2. **Run power-law diagnostics** (if enabled):
   - Calls `PowerLawDiagnostics.diagnose(scales, statistics, slope, intercept)`
   - Stores in `test_result["diagnostics"]["power_law"]`

3. **Run scaling influence analysis** (if enabled):
   - Calls `ScalingInfluenceAnalyzer.analyse(scales, statistics)`
   - Stores in `test_result["diagnostics"]["scaling_influence"]`

4. **Run scale-window sensitivity** (if enabled):
   - Calls `ScaleWindowSensitivityAnalyser.analyse(estimator, data, result, scales)`
   - Stores in `test_result["diagnostics"]["scale_sensitivity"]`

5. **Diagnostics included in result**:
   ```python
   test_result = {
       ...
       "diagnostics": {
           "power_law": {...},
           "scaling_influence": {...},
           "scale_sensitivity": {...}
       },
       ...
   }
   ```

#### Configuration Control:

All diagnostics are controlled via `config/benchmark_protocol.yaml`:

```yaml
diagnostics:
  log_log_checks:
    enabled: true
    min_r_squared: 0.5
    min_points: 6
    detect_breakpoints: true
    influence_analysis: true
  residual_tests:
    enabled: true
    normality_test: "shapiro"
    autocorrelation_lags: 10
    heteroscedasticity_test: true
  goodness_of_fit:
    enabled: true
    methods: ["aic", "bic", "adjusted_r_squared"]
    confidence_bands: true
  scale_window_sensitivity:
    enabled: true
    perturbation_levels: [0.9, 0.95, 1.05, 1.1]
    leave_one_out: true
```

---

## 4. Helper Methods and Utilities

### New Helper Methods in `benchmark.py`:

1. **`_extract_scale_data(result, estimator)`**:
   - Intelligently extracts scale/statistic data from diverse estimator result structures
   - Handles various naming conventions
   - Falls back to estimator methods if direct extraction fails

2. **`_infer_estimator_family(estimator_name)`**:
   - Categorises estimators into classical, ML, or neural families
   - Used for family-level stratification

3. **Enhanced `_build_provenance_bundle(summary)`**:
   - Now delegates to `ProvenanceTracker.capture_provenance()`
   - Captures full environment, packages, hardware, git info

---

## 5. Usage Example

### Running a Benchmark with Full Diagnostics:

```python
from lrdbenchmark.analysis.benchmark import ComprehensiveBenchmark

# Initialise benchmark (automatically loads enhanced protocol)
benchmark = ComprehensiveBenchmark(output_dir="benchmark_results")

# Run comprehensive benchmark
results = benchmark.run_comprehensive_benchmark(
    data_length=1000,
    benchmark_type="classical",  # or "ML", "neural", "comprehensive"
    contamination_type=None,     # or "additive_gaussian", etc.
    contamination_level=0.1,
    save_results=True
)

# Results automatically include:
# - Diagnostics for each estimator run (power-law, scaling influence, sensitivity)
# - Enhanced stratified reports (by true H, estimated H, tail class, length, contamination, family)
# - Comprehensive provenance bundle (environment, packages, protocol, git)
```

### Output Files:

After running, the following files are generated in `benchmark_results/`:

1. **Core Results**:
   - `comprehensive_benchmark_<timestamp>.json`: Full results with all diagnostics
   - `benchmark_summary_<timestamp>.csv`: Flattened summary for analysis

2. **Provenance**:
   - `benchmark_provenance_<timestamp>.json`: Full provenance bundle with environment, protocol, git info

3. **Stratified Reports**:
   - `stratified_report_<timestamp>.md`: Human-readable Markdown report
   - `stratified_report_<timestamp>.json`: Machine-readable JSON
   - `stratified_true_h_<timestamp>.csv`: True H stratification
   - `stratified_estimated_h_<timestamp>.csv`: Estimated H stratification
   - `stratified_tail_class_<timestamp>.csv`: Tail class stratification
   - `stratified_length_<timestamp>.csv`: Data length stratification
   - `stratified_contamination_<timestamp>.csv`: Contamination stratification
   - `stratified_estimator_family_<timestamp>.csv`: Estimator family stratification

4. **Legacy Stratified Metrics**:
   - `stratified_metrics_<timestamp>.json`: Original stratified metrics (still generated for backward compatibility)

5. **Calibration** (if available):
   - `calibration/uncertainty_calibration_<timestamp>.json`

---

## 6. Key Benefits

### For Researchers:

1. **Full Reproducibility**:
   - Every run captures complete environment and protocol
   - Can verify if current setup matches historical run
   - Git commit tracking ensures code version traceability

2. **Regime-Specific Conclusions**:
   - No more averaging across disparate H regimes
   - Can see where estimators excel or fail
   - Tail class and contamination effects isolated

3. **Diagnostic Confidence**:
   - Power-law fit quality automatically assessed
   - Breakpoints detected automatically
   - Scale window robustness quantified
   - Residual assumptions tested

4. **Protocol Standardisation**:
   - All critical settings (detrending, trimming, tapering, scale selection) centralised
   - No hidden differences between runs
   - Ablation studies facilitated

### For Method Developers:

1. **Comprehensive Feedback**:
   - See where your estimator works well (H range, tail class, length)
   - Diagnostics pinpoint issues (poor log-log fit, breakpoints, sensitivity)

2. **Fair Comparisons**:
   - All methods tested under identical, documented protocol
   - Provenance ensures reproducibility

3. **Rich Stratification**:
   - Understand performance across regimes
   - Target improvements to weak areas

---

## 7. Configuration Reference

### Key Protocol Settings:

| Setting | Location | Description |
|---------|----------|-------------|
| Protocol version | `version` | Current: "2.0" |
| Detrending order | `preprocessing.detrend.order` | Polynomial degree for detrending |
| Tapering method | `preprocessing.tapering.method` | e.g., "tukey" |
| Outlier threshold | `preprocessing.outlier_removal.threshold` | MAD multiplier |
| Spectral min freq | `scale_selection.spectral.min_freq_ratio` | Low-frequency cutoff |
| Spectral max freq | `scale_selection.spectral.max_freq_ratio` | High-frequency cutoff |
| Temporal min window | `scale_selection.temporal.min_window` | Minimum DFA/DMA/R/S window |
| Temporal max window | `scale_selection.temporal.max_window` | Maximum DFA/DMA/R/S window |
| Wavelet type | `scale_selection.wavelet.wavelet` | e.g., "db4" |
| Wavelet min level | `scale_selection.wavelet.min_level` | Minimum decomposition level |
| Wavelet max level | `scale_selection.wavelet.max_level` | Maximum decomposition level |
| Diagnostics enabled | `diagnostics.log_log_checks.enabled` | Master switch for diagnostics |
| Min R² threshold | `diagnostics.log_log_checks.min_r_squared` | Acceptable fit quality |
| Sensitivity perturbations | `diagnostics.scale_window_sensitivity.perturbation_levels` | Scale window perturbation factors |
| Hurst bands | `stratification.hurst_bands.bands` | List of H range definitions |
| Track environment | `provenance.track_environment` | Capture Python/system info |
| Track packages | `provenance.track_package_versions` | Capture package versions |
| Track hardware | `provenance.track_hardware` | Capture CPU/GPU info |

---

## 8. Next Steps and Future Enhancements

### Potential Extensions:

1. **Advanced Diagnostics**:
   - Wavelet-domain residual tests
   - Cross-validation for scale window selection
   - Influence diagnostics for spectral tapering parameters

2. **Richer Stratifications**:
   - By convergence achieved (yes/no)
   - By R² bands (excellent/good/acceptable/poor)
   - By computational time (fast/medium/slow)
   - Multi-dimensional cross-tabs (e.g., H × length × contamination)

3. **Automated Protocol Tuning**:
   - Adaptive scale window selection based on diagnostics
   - Automatic detrending order selection
   - Data-driven tapering parameter optimisation

4. **Enhanced Provenance**:
   - Track data checksums for exact reproducibility
   - Capture estimator internal state
   - Link to external experiment tracking systems (MLflow, Weights & Biases)

5. **Interactive Reports**:
   - HTML reports with interactive plots
   - Drill-down capabilities in stratifications
   - Real-time diagnostic visualisations

---

## 9. Backward Compatibility

All enhancements are designed to maintain backward compatibility:

- Old protocol configuration files (version 1.0) still work; new fields use sensible defaults
- Existing stratified metrics generation unchanged; enhanced reports are additional
- Old provenance structure still available; new tracker adds more detail
- Diagnostics are optional (can be disabled via config)

---

## 10. Testing and Validation

### Recommended Testing:

1. **Protocol Migration**:
   ```bash
   # Verify old config still works
   python -c "from lrdbenchmark.analysis.benchmark import ComprehensiveBenchmark; b = ComprehensiveBenchmark()"
   ```

2. **Diagnostics**:
   ```python
   from lrdbenchmark.analysis.diagnostics import PowerLawDiagnostics
   import numpy as np
   
   diag = PowerLawDiagnostics(min_r_squared=0.5, min_points=6)
   scales = np.logspace(0, 2, 20)
   stats = scales ** 0.7  # Perfect power law
   results = diag.diagnose(scales, stats)
   print(results["overall_assessment"]["quality_score"])  # Should be high
   ```

3. **Stratified Reporting**:
   ```python
   # Run small benchmark and check reports are generated
   results = benchmark.run_comprehensive_benchmark(data_length=500, save_results=True)
   # Check benchmark_results/ for new stratified_report_*.md, *.json, *.csv files
   ```

4. **Provenance**:
   ```python
   from lrdbenchmark.analytics.provenance import ProvenanceTracker, verify_provenance
   
   tracker = ProvenanceTracker()
   bundle = tracker.capture_provenance({"timestamp": "..."})
   print(bundle.keys())  # Should include environment, package_versions, hardware, git_info
   ```

---

## Conclusion

These enhancements address all three key areas identified in the review:

1. ✅ **Protocol standardisation & provenance**: Comprehensive config schema, full provenance tracking, reproducibility verification
2. ✅ **Richer stratified reporting**: Breakouts by true H, estimated H, tail class, length, contamination, family; cross-stratifications
3. ✅ **Diagnostics & scale-window analysis**: Power-law fit diagnostics, residual tests, goodness-of-fit, breakpoint detection, scale window sensitivity

The framework now provides the infrastructure for **statistically defensible, fully reproducible, regime-specific LRD benchmarking** as advocated in the review document.

