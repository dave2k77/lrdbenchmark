#!/usr/bin/env python3
"""
Generate updated figures with neural network results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob

def load_latest_results():
    """Load the latest benchmark results."""
    # Find the most recent results file
    result_files = glob.glob("comprehensive_final_nn_results/comprehensive_final_nn_benchmark_*_summary.json")
    if not result_files:
        raise FileNotFoundError("No results files found")
    
    latest_file = max(result_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return pd.DataFrame(results)

def create_performance_comparison_figure(df):
    """Create a comprehensive performance comparison figure."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sort by MAE for better visualization
    df_sorted = df.sort_values('mean_mae')
    
    # 1. MAE Comparison
    colors = []
    for estimator in df_sorted['estimator']:
        if estimator.startswith('NN_'):
            colors.append('#ff7f0e')  # Orange for neural networks
        elif estimator in ['RandomForest', 'SVR', 'GradientBoosting']:
            colors.append('#2ca02c')  # Green for ML
        else:
            colors.append('#1f77b4')  # Blue for classical
    
    bars1 = ax1.barh(range(len(df_sorted)), df_sorted['mean_mae'], color=colors)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['estimator'], fontsize=10)
    ax1.set_xlabel('Mean Absolute Error (MAE)')
    ax1.set_title('Performance Comparison: Mean Absolute Error')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, df_sorted['mean_mae'])):
        ax1.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=8)
    
    # 2. Execution Time Comparison
    bars2 = ax2.barh(range(len(df_sorted)), df_sorted['mean_execution_time'], color=colors)
    ax2.set_yticks(range(len(df_sorted)))
    ax2.set_yticklabels(df_sorted['estimator'], fontsize=10)
    ax2.set_xlabel('Mean Execution Time (seconds)')
    ax2.set_title('Performance Comparison: Execution Time')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars2, df_sorted['mean_execution_time'])):
        ax2.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.2f}s', va='center', fontsize=8)
    
    # 3. Success Rate
    bars3 = ax3.barh(range(len(df_sorted)), df_sorted['success_rate'], color=colors)
    ax3.set_yticks(range(len(df_sorted)))
    ax3.set_yticklabels(df_sorted['estimator'], fontsize=10)
    ax3.set_xlabel('Success Rate')
    ax3.set_title('Performance Comparison: Success Rate')
    ax3.set_xlim(0, 1.1)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars3, df_sorted['success_rate'])):
        ax3.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.2f}', va='center', fontsize=8)
    
    # 4. MAE vs Execution Time Scatter
    scatter_colors = []
    for estimator in df_sorted['estimator']:
        if estimator.startswith('NN_'):
            scatter_colors.append('#ff7f0e')  # Orange for neural networks
        elif estimator in ['RandomForest', 'SVR', 'GradientBoosting']:
            scatter_colors.append('#2ca02c')  # Green for ML
        else:
            scatter_colors.append('#1f77b4')  # Blue for classical
    
    scatter = ax4.scatter(df_sorted['mean_execution_time'], df_sorted['mean_mae'], 
                         c=scatter_colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Mean Execution Time (seconds)')
    ax4.set_ylabel('Mean Absolute Error (MAE)')
    ax4.set_title('MAE vs Execution Time Trade-off')
    ax4.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, estimator in enumerate(df_sorted['estimator']):
        ax4.annotate(estimator, (df_sorted.iloc[i]['mean_execution_time'], 
                                df_sorted.iloc[i]['mean_mae']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='Classical'),
                      Patch(facecolor='#2ca02c', label='Machine Learning'),
                      Patch(facecolor='#ff7f0e', label='Neural Networks')]
    ax4.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

def create_neural_network_analysis_figure(df):
    """Create a detailed analysis of neural network performance."""
    # Filter neural network results
    nn_df = df[df['estimator'].str.startswith('NN_')].copy()
    nn_df['architecture'] = nn_df['estimator'].str.replace('NN_', '')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Neural Network MAE Comparison
    bars1 = ax1.bar(range(len(nn_df)), nn_df['mean_mae'], 
                   color=['#ff7f0e', '#ff9500', '#ffaa00', '#ffbf00', '#ffd400', '#ffe900'])
    ax1.set_xticks(range(len(nn_df)))
    ax1.set_xticklabels(nn_df['architecture'], rotation=45, ha='right')
    ax1.set_ylabel('Mean Absolute Error (MAE)')
    ax1.set_title('Neural Network Performance: MAE Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, nn_df['mean_mae'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Neural Network Execution Time
    bars2 = ax2.bar(range(len(nn_df)), nn_df['mean_execution_time'], 
                   color=['#ff7f0e', '#ff9500', '#ffaa00', '#ffbf00', '#ffd400', '#ffe900'])
    ax2.set_xticks(range(len(nn_df)))
    ax2.set_xticklabels(nn_df['architecture'], rotation=45, ha='right')
    ax2.set_ylabel('Mean Execution Time (seconds)')
    ax2.set_title('Neural Network Performance: Execution Time')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars2, nn_df['mean_execution_time'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # 3. MAE vs Execution Time for Neural Networks
    scatter = ax3.scatter(nn_df['mean_execution_time'], nn_df['mean_mae'], 
                         c=['#ff7f0e', '#ff9500', '#ffaa00', '#ffbf00', '#ffd400', '#ffe900'], 
                         s=200, alpha=0.7, edgecolors='black', linewidth=1)
    ax3.set_xlabel('Mean Execution Time (seconds)')
    ax3.set_ylabel('Mean Absolute Error (MAE)')
    ax3.set_title('Neural Network MAE vs Execution Time Trade-off')
    ax3.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, architecture in enumerate(nn_df['architecture']):
        ax3.annotate(architecture, (nn_df.iloc[i]['mean_execution_time'], 
                                   nn_df.iloc[i]['mean_mae']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # 4. Success Rate for Neural Networks
    bars4 = ax4.bar(range(len(nn_df)), nn_df['success_rate'], 
                   color=['#ff7f0e', '#ff9500', '#ffaa00', '#ffbf00', '#ffd400', '#ffe900'])
    ax4.set_xticks(range(len(nn_df)))
    ax4.set_xticklabels(nn_df['architecture'], rotation=45, ha='right')
    ax4.set_ylabel('Success Rate')
    ax4.set_title('Neural Network Performance: Success Rate')
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars4, nn_df['success_rate'])):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_detailed_results_table(df):
    """Create a detailed results table figure."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Prepare data for table
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['estimator'],
            f"{row['mean_mae']:.4f}",
            f"{row['std_mae']:.4f}",
            f"{row['mean_execution_time']:.3f}",
            f"{row['success_rate']:.2f}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Estimator', 'Mean MAE', 'Std MAE', 'Mean Time (s)', 'Success Rate'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code rows by estimator type
    for i, estimator in enumerate(df['estimator']):
        if estimator.startswith('NN_'):
            color = '#fff2e6'  # Light orange
        elif estimator in ['RandomForest', 'SVR', 'GradientBoosting']:
            color = '#e6f7e6'  # Light green
        else:
            color = '#e6f2ff'  # Light blue
        
        for j in range(5):  # 5 columns
            table[(i+1, j)].set_facecolor(color)
    
    # Style header
    for j in range(5):
        table[(0, j)].set_facecolor('#4a4a4a')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax.axis('off')
    ax.set_title('Comprehensive Benchmark Results with Neural Networks', 
                fontsize=16, fontweight='bold', pad=20)
    
    return fig

def main():
    """Generate all updated figures."""
    print("Loading latest benchmark results...")
    df = load_latest_results()
    
    print(f"Loaded results for {len(df)} estimators")
    print("\nTop 5 performers by MAE:")
    print(df.nsmallest(5, 'mean_mae')[['estimator', 'mean_mae', 'mean_execution_time']].to_string(index=False))
    
    # Create output directory
    output_dir = Path("updated_figures_with_nn")
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating figures...")
    
    # 1. Performance comparison figure
    print("Creating performance comparison figure...")
    fig1 = create_performance_comparison_figure(df)
    fig1.savefig(output_dir / "comprehensive_performance_with_nn.png", dpi=300, bbox_inches='tight')
    fig1.savefig(output_dir / "comprehensive_performance_with_nn.pdf", bbox_inches='tight')
    
    # 2. Neural network analysis figure
    print("Creating neural network analysis figure...")
    fig2 = create_neural_network_analysis_figure(df)
    fig2.savefig(output_dir / "neural_network_analysis.png", dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / "neural_network_analysis.pdf", bbox_inches='tight')
    
    # 3. Detailed results table
    print("Creating detailed results table...")
    fig3 = create_detailed_results_table(df)
    fig3.savefig(output_dir / "detailed_results_table_with_nn.png", dpi=300, bbox_inches='tight')
    fig3.savefig(output_dir / "detailed_results_table_with_nn.pdf", bbox_inches='tight')
    
    print(f"\nAll figures saved to {output_dir}/")
    print("Generated files:")
    for file in output_dir.glob("*"):
        print(f"  - {file.name}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("NEURAL NETWORK PERFORMANCE SUMMARY")
    print("="*80)
    
    nn_df = df[df['estimator'].str.startswith('NN_')]
    classical_df = df[~df['estimator'].str.startswith('NN_') & ~df['estimator'].isin(['RandomForest', 'SVR', 'GradientBoosting'])]
    ml_df = df[df['estimator'].isin(['RandomForest', 'SVR', 'GradientBoosting'])]
    
    print(f"Neural Networks (n={len(nn_df)}):")
    print(f"  - Mean MAE: {nn_df['mean_mae'].mean():.4f} ± {nn_df['mean_mae'].std():.4f}")
    print(f"  - Mean Time: {nn_df['mean_execution_time'].mean():.3f}s ± {nn_df['mean_execution_time'].std():.3f}s")
    print(f"  - Success Rate: {nn_df['success_rate'].mean():.2f}")
    
    print(f"\nMachine Learning (n={len(ml_df)}):")
    print(f"  - Mean MAE: {ml_df['mean_mae'].mean():.4f} ± {ml_df['mean_mae'].std():.4f}")
    print(f"  - Mean Time: {ml_df['mean_execution_time'].mean():.3f}s ± {ml_df['mean_execution_time'].std():.3f}s")
    print(f"  - Success Rate: {ml_df['success_rate'].mean():.2f}")
    
    print(f"\nClassical (n={len(classical_df)}):")
    print(f"  - Mean MAE: {classical_df['mean_mae'].mean():.4f} ± {classical_df['mean_mae'].std():.4f}")
    print(f"  - Mean Time: {classical_df['mean_execution_time'].mean():.3f}s ± {classical_df['mean_execution_time'].std():.3f}s")
    print(f"  - Success Rate: {classical_df['success_rate'].mean():.2f}")

if __name__ == "__main__":
    main()
