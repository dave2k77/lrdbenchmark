# LaTeX Tables Summary

This document provides an overview of all LaTeX tables generated from the LRDBenchmark results.

## Available Tables

### 1. Individual Estimator Performance (`latex_tables.tex`)

**Table: Individual Estimator Performance**
- **Label**: `tab:individual_estimators`
- **Content**: Mean error, standard deviation, and execution time for each estimator
- **Key Finding**: Classical Whittle achieves the best individual performance (0.1332 mean error)

### 2. Category Comparison (`latex_tables.tex`)

**Table: Performance Summary by Estimator Category**
- **Label**: `tab:category_comparison`
- **Content**: Mean error, execution time, and success rate by category
- **Key Finding**: ML methods achieve 54.5% better accuracy than classical methods

### 3. Contamination Effects (`latex_tables.tex`)

**Table: Contamination Effects on Estimator Performance**
- **Label**: `tab:contamination_effects`
- **Content**: Performance at 0% vs 20% contamination with degradation percentages
- **Key Finding**: ML methods show only 6.5% degradation vs 203.9% for classical methods

### 4. Data Model Performance (`latex_tables.tex`)

**Table: Performance by Data Model**
- **Label**: `tab:data_model_performance`
- **Content**: Mean error by data model (FBM, FGN, ARFIMA, MRW) and estimator category
- **Key Finding**: ML methods consistently outperform across all data models

### 5. Hurst Value Performance (`latex_tables.tex`)

**Table: Performance by True Hurst Value**
- **Label**: `tab:hurst_value_performance`
- **Content**: Mean error by true Hurst value (0.3, 0.5, 0.7, 0.9) and estimator category
- **Key Finding**: Performance varies significantly with Hurst value, especially for classical methods

### 6. Comprehensive Analysis (`comprehensive_latex_tables.tex`)

**Table: Comprehensive Estimator Performance Analysis**
- **Label**: `tab:comprehensive_analysis`
- **Content**: Detailed statistics including mean error, std dev, coefficient of variation, execution times, and efficiency scores
- **Key Finding**: Classical Whittle has the highest efficiency score (3128.13), followed by Classical GPH (2421.25)

### 7. Statistical Summary (`comprehensive_latex_tables.tex`)

**Table: Statistical Summary by Category**
- **Label**: `tab:statistical_summary`
- **Content**: Category-level aggregated statistics
- **Key Finding**: ML methods have the lowest coefficient of variation (65.9%), indicating most consistent performance

## Usage Instructions

### In LaTeX Document

1. **Include the tables in your manuscript**:
   ```latex
   \input{final_results/latex_tables.tex}
   \input{final_results/comprehensive_latex_tables.tex}
   ```

2. **Reference tables in text**:
   ```latex
   As shown in Table \ref{tab:individual_estimators}, the Classical Whittle estimator achieves the best individual performance.
   ```

3. **Required packages**:
   ```latex
   \usepackage{booktabs}  % For \toprule, \midrule, \bottomrule
   \usepackage{adjustbox} % For \resizebox (used in comprehensive table)
   ```

### Key Statistics Highlighted

- **Best Individual Performance**: Classical Whittle (0.1332 mean error)
- **Best Category Performance**: ML methods (0.2032 mean error)
- **Most Robust**: ML methods (6.5% contamination degradation)
- **Most Efficient**: Classical Whittle (3128.13 efficiency score)
- **Most Consistent**: ML methods (65.9% coefficient of variation)

## File Locations

- `final_results/latex_tables.tex` - Basic performance tables
- `final_results/comprehensive_latex_tables.tex` - Detailed analysis tables

## Data Source

All tables are generated from:
- **Source**: `comprehensive_all_estimators_benchmark_20250905_074313.csv`
- **Tests**: 6,240 total test cases
- **Estimators**: 13 estimators across 3 categories
- **Data Models**: 4 models (FBM, FGN, ARFIMA, MRW)
- **Hurst Values**: 4 values (0.3, 0.5, 0.7, 0.9)
- **Contamination**: 3 levels (0%, 10%, 20%)

These tables provide comprehensive statistical evidence for the manuscript's conclusions about estimator performance, robustness, and efficiency.
