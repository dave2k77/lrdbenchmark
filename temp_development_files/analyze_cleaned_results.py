#!/usr/bin/env python3
"""
Analyze cleaned results from the comprehensive benchmark.

This script processes the existing results and removes problematic estimators,
focusing on properly implemented methods for the research paper.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_clean_results(csv_file: str) -> pd.DataFrame:
    """Load results and remove problematic estimators."""
    df = pd.read_csv(csv_file)
    
    # Remove problematic neural network estimators
    problematic_estimators = [
        'Neural_LSTM', 'Neural_GRU', 'Neural_Transformer', 'Neural_CNN'
    ]
    
    # Filter out problematic estimators
    df_cleaned = df[~df['estimator_name'].isin(problematic_estimators)].copy()
    
    print(f"Original results: {len(df)} tests")
    print(f"Cleaned results: {len(df_cleaned)} tests")
    print(f"Removed estimators: {problematic_estimators}")
    
    return df_cleaned

def analyze_category_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance by estimator category."""
    # Extract category from estimator name
    df['category'] = df['estimator_name'].str.split('_').str[0]
    
    # Calculate metrics by category
    category_stats = df.groupby('category').agg({
        'hurst_error': ['mean', 'std', 'median'],
        'execution_time': ['mean', 'median'],
        'success': 'sum'
    }).round(4)
    
    # Flatten column names
    category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
    category_stats = category_stats.reset_index()
    
    return category_stats

def analyze_top_performers(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Identify top performing estimators."""
    # Calculate mean error by estimator
    estimator_performance = df.groupby('estimator_name').agg({
        'hurst_error': ['mean', 'std', 'count'],
        'execution_time': 'mean',
        'success': 'sum'
    }).round(4)
    
    # Flatten column names
    estimator_performance.columns = ['_'.join(col).strip() for col in estimator_performance.columns]
    estimator_performance = estimator_performance.reset_index()
    
    # Sort by mean error (lower is better)
    top_performers = estimator_performance.nsmallest(top_n, 'hurst_error_mean')
    
    return top_performers

def analyze_contamination_robustness(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze robustness to contamination."""
    # Extract category
    df['category'] = df['estimator_name'].str.split('_').str[0]
    
    # Calculate performance by contamination level and category
    contamination_analysis = df.groupby(['category', 'contamination_level']).agg({
        'hurst_error': 'mean',
        'success': 'sum'
    }).reset_index()
    
    # Calculate degradation for each category
    degradation_analysis = []
    for category in contamination_analysis['category'].unique():
        cat_data = contamination_analysis[contamination_analysis['category'] == category]
        
        # Get performance at 0% contamination
        clean_performance = cat_data[cat_data['contamination_level'] == 0.0]['hurst_error'].iloc[0]
        
        # Calculate degradation at each contamination level
        for _, row in cat_data.iterrows():
            if row['contamination_level'] > 0:
                degradation = ((row['hurst_error'] - clean_performance) / clean_performance) * 100
                degradation_analysis.append({
                    'category': category,
                    'contamination_level': row['contamination_level'],
                    'degradation_percent': degradation,
                    'error': row['hurst_error']
                })
    
    return pd.DataFrame(degradation_analysis)

def create_publication_figures(df: pd.DataFrame, output_dir: str = "figures"):
    """Create publication-quality figures."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract category
    df['category'] = df['estimator_name'].str.split('_').str[0]
    
    # 1. Category comparison
    plt.figure(figsize=(10, 6))
    category_errors = df.groupby('category')['hurst_error'].mean().sort_values()
    bars = plt.bar(range(len(category_errors)), category_errors.values, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Estimator Category')
    plt.ylabel('Mean Absolute Error')
    plt.title('Performance Comparison by Category')
    plt.xticks(range(len(category_errors)), category_errors.index, rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/category_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Individual estimator performance
    plt.figure(figsize=(12, 8))
    estimator_errors = df.groupby('estimator_name')['hurst_error'].mean().sort_values()
    top_10 = estimator_errors.head(10)
    
    bars = plt.barh(range(len(top_10)), top_10.values)
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Estimator')
    plt.title('Top 10 Performing Estimators')
    plt.yticks(range(len(top_10)), [name.replace('_', ' ') for name in top_10.index])
    
    # Color bars by category
    colors = {'Classical': '#1f77b4', 'ML': '#ff7f0e', 'Neural': '#2ca02c'}
    for i, (estimator, error) in enumerate(top_10.items()):
        category = estimator.split('_')[0]
        bars[i].set_color(colors.get(category, '#666666'))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/individual_estimator_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Contamination effects
    plt.figure(figsize=(10, 6))
    contamination_data = df.groupby(['category', 'contamination_level'])['hurst_error'].mean().unstack()
    
    for category in contamination_data.index:
        plt.plot(contamination_data.columns, contamination_data.loc[category], 
                marker='o', label=category, linewidth=2, markersize=8)
    
    plt.xlabel('Contamination Level')
    plt.ylabel('Mean Absolute Error')
    plt.title('Robustness to Contamination')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/contamination_effects_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Data length effects
    plt.figure(figsize=(10, 6))
    length_data = df.groupby(['category', 'data_length'])['hurst_error'].mean().unstack()
    
    for category in length_data.index:
        plt.plot(length_data.columns, length_data.loc[category], 
                marker='s', label=category, linewidth=2, markersize=8)
    
    plt.xlabel('Data Length')
    plt.ylabel('Mean Absolute Error')
    plt.title('Performance vs Data Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_length_effects_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive summary statistics."""
    df['category'] = df['estimator_name'].str.split('_').str[0]
    
    summary = {
        'total_tests': int(len(df)),
        'successful_tests': int(df['success'].sum()),
        'success_rate': float(df['success'].mean()),
        'estimators_tested': int(df['estimator_name'].nunique()),
        'categories_tested': int(df['category'].nunique()),
        'models_tested': int(df['model_name'].nunique()),
        'data_lengths': sorted(df['data_length'].unique().tolist()),
        'hurst_values': sorted(df['true_hurst'].unique().tolist()),
        'contamination_levels': sorted(df['contamination_level'].unique().tolist()),
    }
    
    # Category performance
    category_performance = df.groupby('category').agg({
        'hurst_error': ['mean', 'std', 'median'],
        'execution_time': ['mean', 'median'],
        'success': 'sum'
    }).round(4)
    
    # Convert to JSON-serializable format
    category_perf_dict = {}
    for col in category_performance.columns:
        if isinstance(col, tuple):
            col_name = '_'.join(str(x) for x in col)
        else:
            col_name = str(col)
        category_perf_dict[col_name] = category_performance[col].to_dict()
    summary['category_performance'] = category_perf_dict
    
    # Top performers
    top_performers = analyze_top_performers(df, top_n=5)
    summary['top_5_performers'] = top_performers.to_dict('records')
    
    # Contamination robustness
    contamination_robustness = analyze_contamination_robustness(df)
    if not contamination_robustness.empty:
        summary['contamination_robustness'] = contamination_robustness.to_dict('records')
    
    return summary

def main():
    """Main analysis function."""
    # Load and clean results
    results_file = "final_results/data/comprehensive_all_estimators_benchmark_20250905_074313.csv"
    df = load_and_clean_results(results_file)
    
    # Generate analysis
    print("\n=== ANALYSIS RESULTS ===")
    
    # Category performance
    category_stats = analyze_category_performance(df)
    print("\nCategory Performance:")
    print(category_stats)
    
    # Top performers
    top_performers = analyze_top_performers(df, top_n=10)
    print("\nTop 10 Performers:")
    print(top_performers[['estimator_name', 'hurst_error_mean', 'hurst_error_std', 'execution_time_mean']])
    
    # Contamination robustness
    contamination_robustness = analyze_contamination_robustness(df)
    if not contamination_robustness.empty:
        print("\nContamination Robustness:")
        print(contamination_robustness)
    
    # Create figures
    print("\nCreating publication figures...")
    create_publication_figures(df, "final_results/figures")
    
    # Generate summary
    summary = generate_summary_statistics(df)
    
    # Save summary
    with open("final_results/cleaned_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Total tests analyzed: {summary['total_tests']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Estimators tested: {summary['estimators_tested']}")
    print(f"Categories: {summary['categories_tested']}")
    
    # Print key findings
    print("\n=== KEY FINDINGS ===")
    print("Top 3 performing estimators:")
    for i, row in top_performers.head(3).iterrows():
        print(f"{i+1}. {row['estimator_name']}: {row['hurst_error_mean']:.4f} ± {row['hurst_error_std']:.4f}")
    
    print("\nCategory performance (mean error):")
    for _, row in category_stats.iterrows():
        print(f"{row['category']}: {row['hurst_error_mean']:.4f}")

if __name__ == "__main__":
    main()
