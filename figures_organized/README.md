# Manuscript Figures

This folder contains all the figures referenced in the manuscript, organized by figure number and properly named.

## Figure Files

### Figure 1: Comprehensive Performance Analysis
- **File**: `Figure1_Comprehensive_Performance.png`
- **Label**: `fig:comprehensive_performance`
- **Description**: Comprehensive adaptive estimator performance showing (a) success rates, (b) mean absolute errors, (c) execution times, and (d) performance trade-offs. The intelligent backend system ensures robust performance across all estimators.

### Figure 2: EEG Contamination Robustness
- **File**: `Figure2_EEG_Robustness.png`
- **Label**: `fig:eeg_robustness`
- **Description**: EEG contamination robustness analysis showing performance across 4 realistic artifact scenarios. The comprehensive adaptive estimators maintain high success rates and consistent accuracy under contamination.

### Figure 3: Contamination Effects Analysis
- **File**: `Figure3_Contamination_Effects.png`
- **Label**: `fig:contamination_effects`
- **Description**: Performance degradation with contamination showing the robustness of different estimator categories. ML methods show minimal degradation while classical methods suffer significant performance loss.

### Figure 4: Speed-Accuracy Tradeoff
- **File**: `Figure4_Speed_Accuracy_Tradeoff.png`
- **Label**: `fig:speed_accuracy`
- **Description**: Speed vs accuracy trade-off showing the relationship between execution time and estimation accuracy. Classical methods offer faster execution while ML methods provide superior accuracy.

### Figure 5: Three-Way Comparison
- **File**: `Figure5_Three_Way_Comparison.png`
- **Label**: `fig:three_way_comparison`
- **Description**: Comprehensive three-way comparison showing (a) mean absolute error by type, (b) success rate by type, (c) execution time by type, (d) individual estimator performance, (e) MAE vs execution time scatter plot, and (f) top 10 performers. Neural networks demonstrate excellent speed-accuracy trade-offs with consistent performance.

## Usage in Manuscript

All figures are referenced in the manuscript using the `\includegraphics` command with the `figures_organized/` path prefix. The figures are properly labeled and captioned for LaTeX compilation.

## Source Files

- **Figures 1-4**: Originally from `final_results/figures/`
- **Figure 5**: Originally from `comprehensive_benchmark_results/`

## Figure Generation

These figures were generated from comprehensive benchmark results comparing:
- Classical estimators (R/S, DFA, DMA, Higuchi, GPH, Whittle, Periodogram)
- Machine Learning estimators (SVR, Gradient Boosting, Random Forest)
- Neural Network estimators (8 architectures: FFN, CNN, LSTM, GRU, Transformer, ResNet, etc.)
