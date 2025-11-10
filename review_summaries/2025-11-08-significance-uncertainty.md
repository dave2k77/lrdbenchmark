# Statistical Significance & Uncertainty Calibration Enhancements (2025-11-08)

## Overview
- Added a full non-parametric significance testing pipeline (Friedman, Nemenyi, Holm-adjusted Wilcoxon, paired sign) to benchmarking so leaderboard standings now carry explicit statistical evidence.
- Extended uncertainty calibration collection to log per-method coverage across all estimator families, save canonical artefacts, and render nominal-versus-empirical coverage plots.
- Surfaced the new statistics and calibration signals in leaderboard generation, reports, and console summaries for immediate interpretation.

## Key Updates
- `lrdbenchmark/analysis/benchmark.py`
  - Augmented `_compute_significance_tests` with Friedman omnibus tests, Nemenyi critical-difference checks, Holm-adjusted Wilcoxon signed-rank comparisons, and paired sign tests. Captures per-estimator win markers for later use (`estimator_markers`).
  - Recorded uncertainty intervals for each resampling method (block bootstrap, wavelet bootstrap, parametric Monte Carlo) with estimator family, primary flag, and method metadata.
  - Attached calibration summaries and plots to benchmark results; saved both JSON and PNG artefacts alongside provenance bundles.
  - Printed aggregated calibration diagnostics (method/family coverage rates) during benchmark summaries.
- `lrdbenchmark/analytics/error_analyzer.py`
  - Added `summarise_uncertainty_calibration` to aggregate empirical coverage per estimator/method and `plot_uncertainty_calibration` for nominal vs empirical coverage visualisation.
- `scripts/analysis/comprehensive_leaderboard.py`
  - Ingested latest significance metadata, appended significance columns (win counts, evidence labels, mean ranks), and propagated them through reports and console output.

## Usage Notes
1. Re-run comprehensive benchmarks to populate the enlarged significance analysis and uncertainty calibration artefacts.
2. New calibration outputs are stored under `benchmark_results/calibration/` with plots in `benchmark_results/figures/`.
3. Re-running `scripts/analysis/comprehensive_leaderboard.py` will now highlight statistically validated leaders directly in the leaderboard tables and summaries.

## Follow-Up Suggestions
- Mirror the calibration plots and significance tables into the published documentation/notebooks.
- Incorporate the new empirical coverage metrics into automated regression tests for estimator calibration.
- Consider exporting significance markers with the public leaderboard CSV to support downstream consumers.

