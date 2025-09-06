#!/usr/bin/env python3
"""
Statistical Analysis Framework for LRDBenchmark Results

This module provides comprehensive statistical analysis including:
- Confidence intervals for performance metrics
- Effect sizes and statistical significance testing
- Multiple comparison correction
- Power analysis
- Bootstrap resampling for robust statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, kruskal, friedmanchisquare
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
# from statsmodels.stats.contingency import mcnemar  # Not needed for this analysis
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """Comprehensive statistical analysis for benchmark results."""
    
    def __init__(self, results_data: Dict, alpha: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            results_data: Dictionary containing benchmark results
            alpha: Significance level for statistical tests
        """
        self.results_data = results_data
        self.alpha = alpha
        self.estimator_names = list(results_data.keys())
        
        # Extract performance metrics
        self._extract_metrics()
        
    def _extract_metrics(self):
        """Extract performance metrics from results data."""
        self.metrics = {}
        
        for estimator, data in self.results_data.items():
            if 'test_results' in data and data['test_results']:
                # Extract individual test results
                test_results = data['test_results']
                
                # Filter successful tests only
                successful_tests = [t for t in test_results if t.get('success', False)]
                
                if successful_tests:
                    mae_values = [t['mae'] for t in successful_tests]
                    exec_times = [t['execution_time'] for t in successful_tests]
                    hurst_true = [t['hurst_true'] for t in successful_tests]
                    hurst_est = [t['hurst_estimate'] for t in successful_tests]
                    
                    self.metrics[estimator] = {
                        'mae': np.array(mae_values),
                        'execution_time': np.array(exec_times),
                        'hurst_true': np.array(hurst_true),
                        'hurst_estimate': np.array(hurst_est),
                        'n_tests': len(successful_tests),
                        'success_rate': len(successful_tests) / len(test_results)
                    }
    
    def calculate_confidence_intervals(self, metric: str = 'mae', confidence: float = 0.95) -> Dict:
        """
        Calculate confidence intervals for specified metric.
        
        Args:
            metric: Metric to analyze ('mae', 'execution_time', 'hurst_estimate')
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Dictionary with confidence intervals for each estimator
        """
        ci_results = {}
        
        for estimator, data in self.metrics.items():
            if metric in data:
                values = data[metric]
                n = len(values)
                
                if n > 1:
                    # Calculate confidence interval
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)
                    se = std_val / np.sqrt(n)
                    
                    # t-distribution critical value
                    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
                    
                    # Confidence interval
                    margin_error = t_crit * se
                    ci_lower = mean_val - margin_error
                    ci_upper = mean_val + margin_error
                    
                    ci_results[estimator] = {
                        'mean': mean_val,
                        'std': std_val,
                        'se': se,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'margin_error': margin_error,
                        'n': n
                    }
                else:
                    ci_results[estimator] = {
                        'mean': values[0] if len(values) > 0 else np.nan,
                        'std': 0,
                        'se': 0,
                        'ci_lower': values[0] if len(values) > 0 else np.nan,
                        'ci_upper': values[0] if len(values) > 0 else np.nan,
                        'margin_error': 0,
                        'n': n
                    }
        
        return ci_results
    
    def calculate_effect_sizes(self, metric: str = 'mae') -> Dict:
        """
        Calculate effect sizes (Cohen's d) between estimator pairs.
        
        Args:
            metric: Metric to analyze
            
        Returns:
            Dictionary with effect sizes between all pairs
        """
        effect_sizes = {}
        
        estimators = list(self.metrics.keys())
        
        for i, est1 in enumerate(estimators):
            for j, est2 in enumerate(estimators[i+1:], i+1):
                if metric in self.metrics[est1] and metric in self.metrics[est2]:
                    values1 = self.metrics[est1][metric]
                    values2 = self.metrics[est2][metric]
                    
                    if len(values1) > 1 and len(values2) > 1:
                        # Cohen's d
                        mean1, mean2 = np.mean(values1), np.mean(values2)
                        std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
                        
                        # Pooled standard deviation
                        n1, n2 = len(values1), len(values2)
                        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                        
                        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                        
                        effect_sizes[f"{est1}_vs_{est2}"] = {
                            'cohens_d': cohens_d,
                            'interpretation': self._interpret_effect_size(abs(cohens_d)),
                            'mean1': mean1,
                            'mean2': mean2,
                            'n1': n1,
                            'n2': n2
                        }
        
        return effect_sizes
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def perform_statistical_tests(self, metric: str = 'mae') -> Dict:
        """
        Perform statistical significance tests.
        
        Args:
            metric: Metric to analyze
            
        Returns:
            Dictionary with test results
        """
        test_results = {}
        
        # Extract data for all estimators
        estimator_data = {}
        for estimator, data in self.metrics.items():
            if metric in data and len(data[metric]) > 1:
                estimator_data[estimator] = data[metric]
        
        if len(estimator_data) < 2:
            return test_results
        
        estimators = list(estimator_data.keys())
        
        # 1. Kruskal-Wallis test (non-parametric ANOVA)
        try:
            groups = [estimator_data[est] for est in estimators]
            h_stat, p_value = kruskal(*groups)
            
            test_results['kruskal_wallis'] = {
                'h_statistic': h_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'interpretation': 'Significant difference between groups' if p_value < self.alpha else 'No significant difference between groups'
            }
        except Exception as e:
            test_results['kruskal_wallis'] = {'error': str(e)}
        
        # 2. Pairwise comparisons (Mann-Whitney U tests)
        pairwise_results = {}
        p_values = []
        
        for i, est1 in enumerate(estimators):
            for j, est2 in enumerate(estimators[i+1:], i+1):
                try:
                    u_stat, p_value = mannwhitneyu(
                        estimator_data[est1], 
                        estimator_data[est2], 
                        alternative='two-sided'
                    )
                    
                    pairwise_results[f"{est1}_vs_{est2}"] = {
                        'u_statistic': u_stat,
                        'p_value': p_value,
                        'significant': p_value < self.alpha
                    }
                    p_values.append(p_value)
                except Exception as e:
                    pairwise_results[f"{est1}_vs_{est2}"] = {'error': str(e)}
        
        test_results['pairwise_comparisons'] = pairwise_results
        
        # 3. Multiple comparison correction
        if p_values:
            try:
                # Bonferroni correction
                bonferroni_corrected = multipletests(p_values, method='bonferroni')
                # FDR correction
                fdr_corrected = multipletests(p_values, method='fdr_bh')
                
                test_results['multiple_comparison_correction'] = {
                    'bonferroni': {
                        'rejected': bonferroni_corrected[0].tolist(),
                        'p_values_corrected': bonferroni_corrected[1].tolist()
                    },
                    'fdr_bh': {
                        'rejected': fdr_corrected[0].tolist(),
                        'p_values_corrected': fdr_corrected[1].tolist()
                    }
                }
            except Exception as e:
                test_results['multiple_comparison_correction'] = {'error': str(e)}
        
        return test_results
    
    def calculate_power_analysis(self, metric: str = 'mae', effect_size: float = 0.5) -> Dict:
        """
        Calculate statistical power for the analysis.
        
        Args:
            metric: Metric to analyze
            effect_size: Expected effect size (Cohen's d)
            
        Returns:
            Dictionary with power analysis results
        """
        power_results = {}
        
        for estimator, data in self.metrics.items():
            if metric in data and len(data[metric]) > 1:
                n = len(data[metric])
                
                # Calculate power for t-test
                try:
                    power = ttest_power(effect_size, n, alpha=self.alpha)
                    
                    power_results[estimator] = {
                        'n': n,
                        'effect_size': effect_size,
                        'power': power,
                        'adequate_power': power >= 0.8,
                        'interpretation': 'Adequate power' if power >= 0.8 else 'Insufficient power'
                    }
                except Exception as e:
                    power_results[estimator] = {'error': str(e)}
        
        return power_results
    
    def bootstrap_analysis(self, metric: str = 'mae', n_bootstrap: int = 1000) -> Dict:
        """
        Perform bootstrap resampling for robust statistics.
        
        Args:
            metric: Metric to analyze
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with bootstrap results
        """
        bootstrap_results = {}
        
        for estimator, data in self.metrics.items():
            if metric in data and len(data[metric]) > 1:
                values = data[metric]
                n = len(values)
                
                # Bootstrap resampling
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(values, size=n, replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                
                bootstrap_means = np.array(bootstrap_means)
                
                # Calculate bootstrap statistics
                bootstrap_results[estimator] = {
                    'original_mean': np.mean(values),
                    'bootstrap_mean': np.mean(bootstrap_means),
                    'bootstrap_std': np.std(bootstrap_means),
                    'bootstrap_ci_lower': np.percentile(bootstrap_means, 2.5),
                    'bootstrap_ci_upper': np.percentile(bootstrap_means, 97.5),
                    'bias': np.mean(bootstrap_means) - np.mean(values),
                    'n_bootstrap': n_bootstrap
                }
        
        return bootstrap_results
    
    def generate_statistical_report(self, output_dir: str = "statistical_analysis") -> Dict:
        """
        Generate comprehensive statistical report.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary with complete statistical analysis
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report = {
            'confidence_intervals_mae': self.calculate_confidence_intervals('mae'),
            'confidence_intervals_execution_time': self.calculate_confidence_intervals('execution_time'),
            'effect_sizes_mae': self.calculate_effect_sizes('mae'),
            'statistical_tests_mae': self.perform_statistical_tests('mae'),
            'power_analysis_mae': self.calculate_power_analysis('mae'),
            'bootstrap_analysis_mae': self.bootstrap_analysis('mae'),
            'summary_statistics': self._generate_summary_statistics()
        }
        
        # Save detailed report
        with open(output_path / 'statistical_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary table
        self._generate_summary_table(report, output_path)
        
        # Generate visualizations
        self._generate_statistical_plots(report, output_path)
        
        return report
    
    def _generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for all estimators."""
        summary = {}
        
        for estimator, data in self.metrics.items():
            summary[estimator] = {
                'n_tests': data['n_tests'],
                'success_rate': data['success_rate'],
                'mae_mean': np.mean(data['mae']),
                'mae_std': np.std(data['mae'], ddof=1),
                'execution_time_mean': np.mean(data['execution_time']),
                'execution_time_std': np.std(data['execution_time'], ddof=1)
            }
        
        return summary
    
    def _generate_summary_table(self, report: Dict, output_path: Path):
        """Generate summary table with confidence intervals."""
        # Create summary DataFrame
        summary_data = []
        
        for estimator, stats in report['summary_statistics'].items():
            ci_data = report['confidence_intervals_mae'].get(estimator, {})
            
            summary_data.append({
                'Estimator': estimator,
                'N_Tests': stats['n_tests'],
                'Success_Rate': f"{stats['success_rate']:.3f}",
                'MAE_Mean': f"{stats['mae_mean']:.4f}",
                'MAE_Std': f"{stats['mae_std']:.4f}",
                'MAE_CI_Lower': f"{ci_data.get('ci_lower', 0):.4f}",
                'MAE_CI_Upper': f"{ci_data.get('ci_upper', 0):.4f}",
                'Exec_Time_Mean': f"{stats['execution_time_mean']:.4f}",
                'Exec_Time_Std': f"{stats['execution_time_std']:.4f}"
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('MAE_Mean')
        
        # Save as CSV
        df.to_csv(output_path / 'statistical_summary_table.csv', index=False)
        
        # Save as LaTeX table
        latex_table = df.to_latex(index=False, escape=False, float_format='%.4f')
        with open(output_path / 'statistical_summary_table.tex', 'w') as f:
            f.write(latex_table)
    
    def _generate_statistical_plots(self, report: Dict, output_path: Path):
        """Generate statistical visualization plots."""
        # 1. Confidence intervals plot
        self._plot_confidence_intervals(report, output_path)
        
        # 2. Effect sizes heatmap
        self._plot_effect_sizes(report, output_path)
        
        # 3. Statistical significance plot
        self._plot_statistical_significance(report, output_path)
    
    def _plot_confidence_intervals(self, report: Dict, output_path: Path):
        """Plot confidence intervals for MAE."""
        ci_data = report['confidence_intervals_mae']
        
        estimators = list(ci_data.keys())
        means = [ci_data[est]['mean'] for est in estimators]
        ci_lowers = [ci_data[est]['ci_lower'] for est in estimators]
        ci_uppers = [ci_data[est]['ci_upper'] for est in estimators]
        
        # Sort by mean MAE
        sorted_indices = np.argsort(means)
        estimators = [estimators[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        ci_lowers = [ci_lowers[i] for i in sorted_indices]
        ci_uppers = [ci_uppers[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(estimators))
        errors = [[m - l for m, l in zip(means, ci_lowers)],
                 [u - m for m, u in zip(means, ci_uppers)]]
        
        bars = ax.barh(y_pos, means, xerr=errors, capsize=5, alpha=0.7)
        
        # Color bars by estimator type
        colors = []
        for estimator in estimators:
            if estimator.startswith('NN_'):
                colors.append('#ff7f0e')  # Orange for neural networks
            elif estimator in ['RandomForest', 'SVR', 'GradientBoosting']:
                colors.append('#2ca02c')  # Green for ML
            else:
                colors.append('#1f77b4')  # Blue for classical
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(estimators)
        ax.set_xlabel('Mean Absolute Error (MAE)')
        ax.set_title('MAE with 95% Confidence Intervals')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='Classical'),
                          Patch(facecolor='#2ca02c', label='Machine Learning'),
                          Patch(facecolor='#ff7f0e', label='Neural Networks')]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_intervals_mae.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'confidence_intervals_mae.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_effect_sizes(self, report: Dict, output_path: Path):
        """Plot effect sizes heatmap."""
        effect_sizes = report['effect_sizes_mae']
        
        if not effect_sizes:
            return
        
        # Create effect size matrix
        estimators = list(report['summary_statistics'].keys())
        n_estimators = len(estimators)
        
        effect_matrix = np.zeros((n_estimators, n_estimators))
        
        for pair, data in effect_sizes.items():
            est1, est2 = pair.split('_vs_')
            i1, i2 = estimators.index(est1), estimators.index(est2)
            effect_matrix[i1, i2] = data['cohens_d']
            effect_matrix[i2, i1] = -data['cohens_d']  # Symmetric
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(effect_matrix, cmap='RdBu_r', vmin=-2, vmax=2)
        
        ax.set_xticks(range(n_estimators))
        ax.set_yticks(range(n_estimators))
        ax.set_xticklabels(estimators, rotation=45, ha='right')
        ax.set_yticklabels(estimators)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Cohen's d Effect Size")
        
        ax.set_title('Effect Sizes Between Estimators (Cohen\'s d)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'effect_sizes_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'effect_sizes_heatmap.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_significance(self, report: Dict, output_path: Path):
        """Plot statistical significance results."""
        test_results = report['statistical_tests_mae']
        
        if 'pairwise_comparisons' not in test_results:
            return
        
        pairwise = test_results['pairwise_comparisons']
        
        # Extract significant comparisons
        significant_pairs = []
        p_values = []
        
        for pair, data in pairwise.items():
            if 'p_value' in data and not np.isnan(data['p_value']):
                significant_pairs.append(pair)
                p_values.append(data['p_value'])
        
        if not significant_pairs:
            return
        
        # Create significance plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        y_pos = np.arange(len(significant_pairs))
        colors = ['red' if p < 0.05 else 'blue' for p in p_values]
        
        bars = ax.barh(y_pos, -np.log10(p_values), color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(significant_pairs)
        ax.set_xlabel('-log10(p-value)')
        ax.set_title('Statistical Significance of Pairwise Comparisons')
        ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'statistical_significance.pdf', bbox_inches='tight')
        plt.close()

def main():
    """Run comprehensive statistical analysis."""
    # Load latest benchmark results
    import glob
    
    # Look for detailed results files (not summary)
    result_files = glob.glob("comprehensive_final_nn_results/comprehensive_final_nn_benchmark_*.json")
    # Filter out summary files
    result_files = [f for f in result_files if not f.endswith('_summary.json')]
    
    if not result_files:
        print("No detailed results files found!")
        return
    
    latest_file = max(result_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Initialize statistical analyzer
    analyzer = StatisticalAnalyzer(results)
    
    # Generate comprehensive statistical report
    print("Generating statistical analysis...")
    report = analyzer.generate_statistical_report()
    
    print("Statistical analysis complete!")
    print(f"Results saved to: statistical_analysis/")
    
    # Print summary
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*80)
    
    # Print confidence intervals for top 5 estimators
    ci_data = report['confidence_intervals_mae']
    sorted_estimators = sorted(ci_data.keys(), key=lambda x: ci_data[x]['mean'])
    
    print("\nTop 5 Estimators by MAE (with 95% CI):")
    for i, estimator in enumerate(sorted_estimators[:5]):
        ci = ci_data[estimator]
        print(f"{i+1}. {estimator}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    
    # Print statistical test results
    test_results = report['statistical_tests_mae']
    if 'kruskal_wallis' in test_results:
        kw = test_results['kruskal_wallis']
        print(f"\nKruskal-Wallis Test: H = {kw['h_statistic']:.4f}, p = {kw['p_value']:.4f}")
        print(f"Result: {kw['interpretation']}")
    
    # Print effect sizes
    effect_sizes = report['effect_sizes_mae']
    large_effects = [(pair, data) for pair, data in effect_sizes.items() 
                    if data['interpretation'] == 'large']
    
    if large_effects:
        print(f"\nLarge Effect Sizes (|d| > 0.8):")
        for pair, data in large_effects[:5]:  # Show top 5
            print(f"  {pair}: d = {data['cohens_d']:.3f}")

if __name__ == "__main__":
    main()
