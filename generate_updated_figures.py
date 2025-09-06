#!/usr/bin/env python3
"""
Generate Updated Figures for Comprehensive Benchmark Results

This script creates updated figures showing the latest comprehensive benchmark results
including classical, ML, and neural network estimators.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results():
    """Load the latest benchmark results."""
    results_file = "comprehensive_final_nn_results/comprehensive_final_nn_benchmark_20250905_200517.csv"
    df = pd.read_csv(results_file)
    
    # Filter out failed estimators for main analysis
    df_working = df[df['success_rate'] > 0].copy()
    
    # Categorize estimators
    df_working['category'] = 'Classical'
    df_working.loc[df_working['estimator'].isin(['RandomForest', 'SVR', 'GradientBoosting']), 'category'] = 'Machine Learning'
    df_working.loc[df_working['estimator'].str.startswith('NN_'), 'category'] = 'Neural Network'
    
    return df, df_working

def create_performance_comparison(df_working):
    """Create comprehensive performance comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Benchmark Results: Classical vs ML vs Neural Networks', fontsize=16, fontweight='bold')
    
    # 1. Mean Absolute Error by Category
    ax1 = axes[0, 0]
    category_mae = df_working.groupby('category')['mean_mae'].agg(['mean', 'std']).reset_index()
    bars1 = ax1.bar(category_mae['category'], category_mae['mean'], 
                    yerr=category_mae['std'], capsize=5, alpha=0.8)
    ax1.set_title('Mean Absolute Error by Category', fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_ylim(0, 0.8)
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars1, category_mae['mean'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Execution Time by Category
    ax2 = axes[0, 1]
    category_time = df_working.groupby('category')['mean_execution_time'].agg(['mean', 'std']).reset_index()
    bars2 = ax2.bar(category_mae['category'], category_time['mean'], 
                    yerr=category_time['std'], capsize=5, alpha=0.8)
    ax2.set_title('Mean Execution Time by Category', fontweight='bold')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_yscale('log')
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars2, category_time['mean'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{mean_val:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # 3. Individual Estimator Performance (Top 10)
    ax3 = axes[1, 0]
    top_10 = df_working.nsmallest(10, 'mean_mae')
    colors = ['red' if cat == 'Classical' else 'blue' if cat == 'Machine Learning' else 'green' 
              for cat in top_10['category']]
    bars3 = ax3.barh(range(len(top_10)), top_10['mean_mae'], color=colors, alpha=0.8)
    ax3.set_yticks(range(len(top_10)))
    ax3.set_yticklabels(top_10['estimator'], fontsize=10)
    ax3.set_title('Top 10 Estimators by Accuracy', fontweight='bold')
    ax3.set_xlabel('Mean Absolute Error')
    ax3.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars3, top_10['mean_mae'])):
        ax3.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontweight='bold')
    
    # 4. Speed vs Accuracy Trade-off
    ax4 = axes[1, 1]
    categories = df_working['category'].unique()
    colors = {'Classical': 'red', 'Machine Learning': 'blue', 'Neural Network': 'green'}
    
    for category in categories:
        data = df_working[df_working['category'] == category]
        ax4.scatter(data['mean_execution_time'], data['mean_mae'], 
                   c=colors[category], label=category, s=100, alpha=0.7, edgecolors='black')
    
    ax4.set_xlabel('Execution Time (seconds)')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Speed vs Accuracy Trade-off', fontweight='bold')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add estimator labels for neural networks
    nn_data = df_working[df_working['category'] == 'Neural Network']
    for _, row in nn_data.iterrows():
        ax4.annotate(row['estimator'].replace('NN_', ''), 
                    (row['mean_execution_time'], row['mean_mae']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('figures_organized/Figure_Updated_Comprehensive_Performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_neural_network_analysis(df_working):
    """Create detailed neural network analysis figure."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Neural Network Performance Analysis', fontsize=16, fontweight='bold')
    
    # Filter neural network data
    nn_data = df_working[df_working['category'] == 'Neural Network'].copy()
    nn_data['network_type'] = nn_data['estimator'].str.replace('NN_', '')
    
    # 1. Neural Network Accuracy Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(nn_data['network_type'], nn_data['mean_mae'], alpha=0.8, color='green')
    ax1.set_title('Neural Network Accuracy Comparison', fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars1, nn_data['mean_mae']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Neural Network Speed Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(nn_data['network_type'], nn_data['mean_execution_time'], alpha=0.8, color='orange')
    ax2.set_title('Neural Network Speed Comparison', fontweight='bold')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(bars2, nn_data['mean_execution_time']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{val:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # 3. Success Rate Analysis
    ax3 = axes[1, 0]
    success_data = df_working.groupby('category')['success_rate'].mean()
    colors = ['red', 'blue', 'green']
    bars3 = ax3.bar(success_data.index, success_data.values, color=colors, alpha=0.8)
    ax3.set_title('Success Rate by Category', fontweight='bold')
    ax3.set_ylabel('Success Rate')
    ax3.set_ylim(0, 1.1)
    
    # Add value labels
    for bar, val in zip(bars3, success_data.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Performance Distribution
    ax4 = axes[1, 1]
    categories = df_working['category'].unique()
    mae_data = [df_working[df_working['category'] == cat]['mean_mae'].values for cat in categories]
    
    bp = ax4.boxplot(mae_data, labels=categories, patch_artist=True)
    colors = ['red', 'blue', 'green']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_title('Performance Distribution by Category', fontweight='bold')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures_organized/Figure_Neural_Network_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_results_table(df_working):
    """Create a detailed results table."""
    # Sort by mean_mae
    df_sorted = df_working.sort_values('mean_mae').copy()
    
    # Create ranking
    df_sorted['rank'] = range(1, len(df_sorted) + 1)
    
    # Select key columns
    results_table = df_sorted[['rank', 'estimator', 'category', 'mean_mae', 'std_mae', 
                              'mean_execution_time', 'success_rate']].copy()
    
    # Round numerical columns
    results_table['mean_mae'] = results_table['mean_mae'].round(4)
    results_table['std_mae'] = results_table['std_mae'].round(4)
    results_table['mean_execution_time'] = results_table['mean_execution_time'].round(4)
    results_table['success_rate'] = results_table['success_rate'].round(3)
    
    # Save to CSV
    results_table.to_csv('figures_organized/Detailed_Results_Table.csv', index=False)
    
    print("Detailed Results Table:")
    print("=" * 80)
    print(results_table.to_string(index=False))
    print("=" * 80)
    
    return results_table

def create_summary_statistics(df_working):
    """Create summary statistics."""
    print("\nSUMMARY STATISTICS")
    print("=" * 50)
    
    # Overall statistics
    total_estimators = len(df_working)
    total_tests = df_working['total_tests'].sum()
    successful_tests = df_working['successful_tests'].sum()
    overall_success_rate = successful_tests / total_tests
    
    print(f"Total Working Estimators: {total_estimators}")
    print(f"Total Tests: {total_tests}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Overall Success Rate: {overall_success_rate:.1%}")
    print(f"Mean MAE (all estimators): {df_working['mean_mae'].mean():.4f}")
    print(f"Mean Execution Time (all estimators): {df_working['mean_execution_time'].mean():.4f}s")
    
    # Category statistics
    print("\nBY CATEGORY:")
    for category in df_working['category'].unique():
        cat_data = df_working[df_working['category'] == category]
        print(f"\n{category}:")
        print(f"  Estimators: {len(cat_data)}")
        print(f"  Mean MAE: {cat_data['mean_mae'].mean():.4f} ± {cat_data['mean_mae'].std():.4f}")
        print(f"  Mean Execution Time: {cat_data['mean_execution_time'].mean():.4f}s ± {cat_data['mean_execution_time'].std():.4f}s")
        print(f"  Success Rate: {cat_data['success_rate'].mean():.1%}")
    
    # Top performers
    print("\nTOP 5 PERFORMERS:")
    top_5 = df_working.nsmallest(5, 'mean_mae')
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i}. {row['estimator']} ({row['category']}): MAE={row['mean_mae']:.4f}, Time={row['mean_execution_time']:.4f}s")

def main():
    """Generate all updated figures and analysis."""
    print("Loading benchmark results...")
    df, df_working = load_results()
    
    print("Creating performance comparison figure...")
    create_performance_comparison(df_working)
    
    print("Creating neural network analysis figure...")
    create_neural_network_analysis(df_working)
    
    print("Creating detailed results table...")
    results_table = create_detailed_results_table(df_working)
    
    print("Generating summary statistics...")
    create_summary_statistics(df_working)
    
    print("\nAll figures and analysis completed!")
    print("Figures saved to figures_organized/")

if __name__ == "__main__":
    main()
