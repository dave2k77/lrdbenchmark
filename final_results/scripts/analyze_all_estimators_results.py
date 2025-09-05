#!/usr/bin/env python3
"""
Comprehensive Analysis of All Estimators Benchmark Results

This script analyzes the comprehensive benchmark results including ALL classical,
ML, and neural network estimators for journal-ready research.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_results(filename: str) -> pd.DataFrame:
    """Load benchmark results from CSV file."""
    df = pd.read_csv(filename)
    print(f"Loaded {len(df):,} benchmark results")
    return df


def analyze_estimator_categories(df: pd.DataFrame) -> None:
    """Analyze performance by estimator categories."""
    print("\n" + "="*70)
    print("📊 ESTIMATOR CATEGORY ANALYSIS")
    print("="*70)
    
    # Performance by category
    category_stats = df.groupby('estimator_category').agg({
        'success': 'mean',
        'hurst_error': ['mean', 'std', 'median'],
        'relative_error': ['mean', 'std', 'median'],
        'execution_time': ['mean', 'std', 'median']
    }).round(4)
    
    print("\n🏆 Performance by Category:")
    for category in df['estimator_category'].unique():
        subset = df[df['estimator_category'] == category]
        print(f"\n{category} Estimators:")
        print(f"  Success Rate: {subset['success'].mean()*100:.1f}%")
        print(f"  Mean Absolute Error: {subset['hurst_error'].mean():.4f}")
        print(f"  Median Absolute Error: {subset['hurst_error'].median():.4f}")
        print(f"  Mean Relative Error: {subset['relative_error'].mean()*100:.1f}%")
        print(f"  Median Relative Error: {subset['relative_error'].median()*100:.1f}%")
        print(f"  Mean Execution Time: {subset['execution_time'].mean():.4f}s")
        print(f"  Median Execution Time: {subset['execution_time'].median():.4f}s")
    
    # Statistical comparison between categories
    classical_data = df[df['estimator_category'] == 'Classical']['hurst_error']
    ml_data = df[df['estimator_category'] == 'ML']['hurst_error']
    neural_data = df[df['estimator_category'] == 'Neural']['hurst_error']
    
    print(f"\n🔬 Statistical Comparisons:")
    
    # Classical vs ML
    t_stat, p_value = stats.ttest_ind(classical_data, ml_data)
    print(f"  Classical vs ML: t={t_stat:.4f}, p={p_value:.2e}")
    print(f"    {'ML significantly better' if p_value < 0.05 and ml_data.mean() < classical_data.mean() else 'No significant difference'}")
    
    # Classical vs Neural
    t_stat, p_value = stats.ttest_ind(classical_data, neural_data)
    print(f"  Classical vs Neural: t={t_stat:.4f}, p={p_value:.2e}")
    print(f"    {'Neural significantly better' if p_value < 0.05 and neural_data.mean() < classical_data.mean() else 'No significant difference'}")
    
    # ML vs Neural
    t_stat, p_value = stats.ttest_ind(ml_data, neural_data)
    print(f"  ML vs Neural: t={t_stat:.4f}, p={p_value:.2e}")
    print(f"    {'ML significantly better' if p_value < 0.05 and ml_data.mean() < neural_data.mean() else 'No significant difference'}")
    
    # Create category comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Estimator Category Performance Comparison', fontsize=16, fontweight='bold')
    
    # Error comparison
    sns.boxplot(data=df, x='estimator_category', y='hurst_error', ax=axes[0,0])
    axes[0,0].set_title('Absolute Hurst Parameter Error by Category')
    axes[0,0].set_ylabel('Absolute Error')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Relative error comparison
    sns.boxplot(data=df, x='estimator_category', y='relative_error', ax=axes[0,1])
    axes[0,1].set_title('Relative Hurst Parameter Error by Category')
    axes[0,1].set_ylabel('Relative Error')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Execution time comparison
    sns.boxplot(data=df, x='estimator_category', y='execution_time', ax=axes[1,0])
    axes[1,0].set_title('Execution Time by Category')
    axes[1,0].set_ylabel('Time (seconds)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Success rate comparison
    success_rates = df.groupby('estimator_category')['success'].mean()
    axes[1,1].bar(success_rates.index, success_rates.values)
    axes[1,1].set_title('Success Rate by Category')
    axes[1,1].set_ylabel('Success Rate')
    axes[1,1].set_ylim(0, 1.1)
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('estimator_category_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_individual_estimators(df: pd.DataFrame) -> None:
    """Analyze performance of individual estimators."""
    print("\n" + "="*70)
    print("🎯 INDIVIDUAL ESTIMATOR ANALYSIS")
    print("="*70)
    
    # Calculate performance metrics by estimator
    estimator_stats = df.groupby('estimator_name').agg({
        'success': 'mean',
        'hurst_error': ['mean', 'std', 'median'],
        'relative_error': ['mean', 'std', 'median'],
        'execution_time': ['mean', 'std', 'median']
    }).round(4)
    
    # Sort by mean error (best first)
    estimator_stats = estimator_stats.sort_values(('hurst_error', 'mean'))
    
    print("\n🏆 Individual Estimator Performance (Ranked by Accuracy):")
    for estimator in estimator_stats.index:
        stats = estimator_stats.loc[estimator]
        print(f"\n{estimator}:")
        print(f"  Success Rate: {stats[('success', 'mean')]*100:.1f}%")
        print(f"  Mean Absolute Error: {stats[('hurst_error', 'mean')]:.4f}")
        print(f"  Median Absolute Error: {stats[('hurst_error', 'median')]:.4f}")
        print(f"  Mean Relative Error: {stats[('relative_error', 'mean')]*100:.1f}%")
        print(f"  Median Relative Error: {stats[('relative_error', 'median')]*100:.1f}%")
        print(f"  Mean Execution Time: {stats[('execution_time', 'mean')]:.4f}s")
        print(f"  Median Execution Time: {stats[('execution_time', 'median')]:.4f}s")
    
    # Create individual estimator plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Individual Estimator Performance Comparison', fontsize=16, fontweight='bold')
    
    # Error comparison
    sns.boxplot(data=df, x='estimator_name', y='hurst_error', ax=axes[0,0])
    axes[0,0].set_title('Absolute Hurst Parameter Error by Estimator')
    axes[0,0].set_ylabel('Absolute Error')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Relative error comparison
    sns.boxplot(data=df, x='estimator_name', y='relative_error', ax=axes[0,1])
    axes[0,1].set_title('Relative Hurst Parameter Error by Estimator')
    axes[0,1].set_ylabel('Relative Error')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Execution time comparison
    sns.boxplot(data=df, x='estimator_name', y='execution_time', ax=axes[1,0])
    axes[1,0].set_title('Execution Time by Estimator')
    axes[1,0].set_ylabel('Time (seconds)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Performance ranking
    mean_errors = df.groupby('estimator_name')['hurst_error'].mean().sort_values()
    axes[1,1].bar(range(len(mean_errors)), mean_errors.values)
    axes[1,1].set_title('Estimator Performance Ranking (Lower is Better)')
    axes[1,1].set_ylabel('Mean Absolute Error')
    axes[1,1].set_xticks(range(len(mean_errors)))
    axes[1,1].set_xticklabels(mean_errors.index, rotation=45)
    
    plt.tight_layout()
    plt.savefig('individual_estimator_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_contamination_effects(df: pd.DataFrame) -> None:
    """Analyze the impact of data contamination."""
    print("\n" + "="*70)
    print("🧪 CONTAMINATION EFFECTS ANALYSIS")
    print("="*70)
    
    # Calculate contamination effects by category
    contamination_stats = df.groupby(['contamination_level', 'estimator_category']).agg({
        'success': 'mean',
        'hurst_error': ['mean', 'std'],
        'relative_error': ['mean', 'std']
    }).round(4)
    
    print("\n📈 Performance vs Contamination Level by Category:")
    for level in sorted(df['contamination_level'].unique()):
        print(f"\nContamination {level*100:.0f}%:")
        level_data = df[df['contamination_level'] == level]
        
        for category in level_data['estimator_category'].unique():
            subset = level_data[level_data['estimator_category'] == category]
            print(f"  {category}:")
            print(f"    Success Rate: {subset['success'].mean()*100:.1f}%")
            print(f"    Mean Absolute Error: {subset['hurst_error'].mean():.4f}")
            print(f"    Mean Relative Error: {subset['relative_error'].mean()*100:.1f}%")
    
    # Statistical test for contamination effects
    clean_data = df[df['contamination_level'] == 0.0]['hurst_error']
    contaminated_data = df[df['contamination_level'] > 0.0]['hurst_error']
    
    t_stat, p_value = stats.ttest_ind(clean_data, contaminated_data)
    print(f"\n🔬 Overall Contamination Effect:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Effect: {'Significant' if p_value < 0.05 else 'Not significant'}")
    
    # Create contamination effect plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Contamination Effects on Estimator Performance', fontsize=16, fontweight='bold')
    
    # Error vs contamination level
    sns.boxplot(data=df, x='contamination_level', y='hurst_error', ax=axes[0,0])
    axes[0,0].set_title('Absolute Error vs Contamination Level')
    axes[0,0].set_xlabel('Contamination Level')
    axes[0,0].set_ylabel('Absolute Error')
    
    # Error by category and contamination
    sns.boxplot(data=df, x='contamination_level', y='hurst_error', hue='estimator_category', ax=axes[0,1])
    axes[0,1].set_title('Error by Category and Contamination')
    axes[0,1].set_xlabel('Contamination Level')
    axes[0,1].set_ylabel('Absolute Error')
    
    # Success rate vs contamination
    success_by_contamination = df.groupby(['contamination_level', 'estimator_category'])['success'].mean().reset_index()
    sns.lineplot(data=success_by_contamination, x='contamination_level', y='success', 
                 hue='estimator_category', marker='o', ax=axes[1,0])
    axes[1,0].set_title('Success Rate vs Contamination Level')
    axes[1,0].set_xlabel('Contamination Level')
    axes[1,0].set_ylabel('Success Rate')
    axes[1,0].set_ylim(0, 1.1)
    
    # Performance degradation
    clean_performance = df[df['contamination_level'] == 0.0].groupby('estimator_category')['hurst_error'].mean()
    contaminated_performance = df[df['contamination_level'] > 0.0].groupby('estimator_category')['hurst_error'].mean()
    degradation = (contaminated_performance - clean_performance) / clean_performance * 100
    
    axes[1,1].bar(degradation.index, degradation.values)
    axes[1,1].set_title('Performance Degradation with Contamination')
    axes[1,1].set_xlabel('Estimator Category')
    axes[1,1].set_ylabel('Performance Degradation (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('contamination_effects_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_data_length_effects(df: pd.DataFrame) -> None:
    """Analyze the impact of data length on performance."""
    print("\n" + "="*70)
    print("📏 DATA LENGTH EFFECTS ANALYSIS")
    print("="*70)
    
    # Calculate length effects by category
    length_stats = df.groupby(['data_length', 'estimator_category']).agg({
        'success': 'mean',
        'hurst_error': ['mean', 'std'],
        'relative_error': ['mean', 'std'],
        'execution_time': ['mean', 'std']
    }).round(4)
    
    print("\n📊 Performance vs Data Length by Category:")
    for length in sorted(df['data_length'].unique()):
        print(f"\nLength {length}:")
        length_data = df[df['data_length'] == length]
        
        for category in length_data['estimator_category'].unique():
            subset = length_data[length_data['estimator_category'] == category]
            print(f"  {category}:")
            print(f"    Success Rate: {subset['success'].mean()*100:.1f}%")
            print(f"    Mean Absolute Error: {subset['hurst_error'].mean():.4f}")
            print(f"    Mean Relative Error: {subset['relative_error'].mean()*100:.1f}%")
            print(f"    Mean Execution Time: {subset['execution_time'].mean():.4f}s")
    
    # Create data length effect plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Length Effects on Estimator Performance', fontsize=16, fontweight='bold')
    
    # Error vs data length
    sns.boxplot(data=df, x='data_length', y='hurst_error', ax=axes[0,0])
    axes[0,0].set_title('Absolute Error vs Data Length')
    axes[0,0].set_xlabel('Data Length')
    axes[0,0].set_ylabel('Absolute Error')
    
    # Execution time vs data length
    sns.boxplot(data=df, x='data_length', y='execution_time', ax=axes[0,1])
    axes[0,1].set_title('Execution Time vs Data Length')
    axes[0,1].set_xlabel('Data Length')
    axes[0,1].set_ylabel('Execution Time (seconds)')
    
    # Error by category and length
    sns.boxplot(data=df, x='data_length', y='hurst_error', hue='estimator_category', ax=axes[1,0])
    axes[1,0].set_title('Error by Category and Data Length')
    axes[1,0].set_xlabel('Data Length')
    axes[1,0].set_ylabel('Absolute Error')
    
    # Performance trends
    mean_errors = df.groupby(['data_length', 'estimator_category'])['hurst_error'].mean().reset_index()
    sns.lineplot(data=mean_errors, x='data_length', y='hurst_error', 
                 hue='estimator_category', marker='o', ax=axes[1,1])
    axes[1,1].set_title('Mean Error Trends by Data Length')
    axes[1,1].set_xlabel('Data Length')
    axes[1,1].set_ylabel('Mean Absolute Error')
    
    plt.tight_layout()
    plt.savefig('data_length_effects_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_publication_summary(df: pd.DataFrame) -> None:
    """Generate a publication-ready summary of results."""
    print("\n" + "="*70)
    print("📝 PUBLICATION-READY SUMMARY")
    print("="*70)
    
    # Overall statistics
    total_tests = len(df)
    success_rate = df['success'].mean() * 100
    mean_error = df['hurst_error'].mean()
    mean_relative_error = df['relative_error'].mean() * 100
    
    print(f"\n🎯 Overall Performance:")
    print(f"  Total Tests: {total_tests:,}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Mean Absolute Error: {mean_error:.4f}")
    print(f"  Mean Relative Error: {mean_relative_error:.1f}%")
    
    # Best performing estimator
    estimator_performance = df.groupby('estimator_name').agg({
        'hurst_error': 'mean',
        'relative_error': 'mean',
        'execution_time': 'mean'
    })
    
    best_estimator = estimator_performance['hurst_error'].idxmin()
    fastest_estimator = estimator_performance['execution_time'].idxmin()
    
    print(f"\n🏆 Best Performing Estimator: {best_estimator}")
    print(f"  Mean Error: {estimator_performance.loc[best_estimator, 'hurst_error']:.4f}")
    print(f"  Mean Relative Error: {estimator_performance.loc[best_estimator, 'relative_error']*100:.1f}%")
    
    print(f"\n⚡ Fastest Estimator: {fastest_estimator}")
    print(f"  Mean Execution Time: {estimator_performance.loc[fastest_estimator, 'execution_time']:.4f}s")
    
    # Category comparison
    classical_performance = df[df['estimator_category'] == 'Classical']['hurst_error'].mean()
    ml_performance = df[df['estimator_category'] == 'ML']['hurst_error'].mean()
    neural_performance = df[df['estimator_category'] == 'Neural']['hurst_error'].mean()
    
    print(f"\n⚔️ Category Performance:")
    print(f"  Classical: {classical_performance:.4f} mean error")
    print(f"  ML: {ml_performance:.4f} mean error")
    print(f"  Neural: {neural_performance:.4f} mean error")
    
    # ML vs Classical comparison
    ml_advantage = (classical_performance - ml_performance) / classical_performance * 100
    print(f"  ML Advantage over Classical: {ml_advantage:.1f}%")
    
    # Contamination robustness
    clean_performance = df[df['contamination_level'] == 0.0]['hurst_error'].mean()
    contaminated_performance = df[df['contamination_level'] > 0.0]['hurst_error'].mean()
    robustness_degradation = (contaminated_performance - clean_performance) / clean_performance * 100
    
    print(f"\n🧪 Contamination Robustness:")
    print(f"  Clean Data Error: {clean_performance:.4f}")
    print(f"  Contaminated Data Error: {contaminated_performance:.4f}")
    print(f"  Performance Degradation: {robustness_degradation:.1f}%")
    
    # Generate summary statistics for publication
    summary_stats = {
        'total_tests': int(total_tests),
        'success_rate': float(success_rate),
        'mean_absolute_error': float(mean_error),
        'mean_relative_error': float(mean_relative_error),
        'best_estimator': str(best_estimator),
        'fastest_estimator': str(fastest_estimator),
        'classical_mean_error': float(classical_performance),
        'ml_mean_error': float(ml_performance),
        'neural_mean_error': float(neural_performance),
        'ml_advantage_over_classical': float(ml_advantage),
        'contamination_robustness_degradation': float(robustness_degradation)
    }
    
    # Save summary statistics
    import json
    with open('all_estimators_publication_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\n💾 Summary statistics saved to: all_estimators_publication_summary.json")


def main():
    """Run comprehensive analysis of all estimators benchmark results."""
    print("🔬 LRDBenchmark - Comprehensive All Estimators Results Analysis")
    print("=" * 70)
    print("Analyzing comprehensive benchmark results for journal-ready research")
    print("Including classical, ML, and neural network estimators")
    print()
    
    # Load results
    results_file = "../data/comprehensive_all_estimators_benchmark_20250905_074313.csv"
    df = load_results(results_file)
    
    # Run analyses
    analyze_estimator_categories(df)
    analyze_individual_estimators(df)
    analyze_contamination_effects(df)
    analyze_data_length_effects(df)
    generate_publication_summary(df)
    
    print("\n✅ Comprehensive analysis completed successfully!")
    print("📊 Publication-ready visualizations and statistics generated!")
    print("📁 Files created:")
    print("  - estimator_category_comparison.png")
    print("  - individual_estimator_comparison.png")
    print("  - contamination_effects_analysis.png")
    print("  - data_length_effects_analysis.png")
    print("  - all_estimators_publication_summary.json")


if __name__ == "__main__":
    main()
