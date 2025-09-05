#!/usr/bin/env python3
"""
Create Simple, Clean Figures for LRDBenchmark Manuscript

This script creates minimal, highly readable figures with perfect spacing
and no overlapping text.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set minimal, clean style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.5,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Simple, high-contrast colors
colors = {
    'Classical': '#1f77b4',      # Blue
    'ML': '#ff7f0e',            # Orange  
    'Neural': '#2ca02c',        # Green
}

def load_results(filename: str) -> pd.DataFrame:
    """Load benchmark results from CSV file."""
    df = pd.read_csv(filename)
    print(f"Loaded {len(df):,} benchmark results")
    return df

def create_figure_1_main_comparison(df: pd.DataFrame):
    """Create Figure 1: Main category comparison - single figure."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate mean errors by category
    classical_error = df[df['estimator_category'] == 'Classical']['hurst_error'].mean()
    ml_error = df[df['estimator_category'] == 'ML']['hurst_error'].mean()
    neural_error = df[df['estimator_category'] == 'Neural']['hurst_error'].mean()
    
    categories = ['Classical', 'ML', 'Neural']
    mean_errors = [classical_error, ml_error, neural_error]
    
    # Create bar chart
    bars = ax.bar(categories, mean_errors, 
                  color=[colors['Classical'], colors['ML'], colors['Neural']],
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, error in zip(bars, mean_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{error:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add ML advantage annotation
    ml_advantage = (classical_error - ml_error) / classical_error * 100
    ax.text(0.5, 0.95, f'ML Advantage: {ml_advantage:.1f}%', 
            transform=ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax.set_title('Mean Absolute Error by Estimator Category', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Estimator Category', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(mean_errors) * 1.2)
    
    plt.tight_layout()
    plt.savefig('Figure1_Main_Comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_figure_2_individual_performance(df: pd.DataFrame):
    """Create Figure 2: Individual estimator performance."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Calculate performance metrics by estimator
    estimator_stats = df.groupby('estimator_name').agg({
        'hurst_error': 'mean',
        'execution_time': 'mean'
    }).round(4)
    
    # Sort by mean error (best first)
    estimator_stats = estimator_stats.sort_values('hurst_error')
    
    # Take top 10 performers
    top_10 = estimator_stats.head(10)
    
    # Color by category
    colors_list = []
    for estimator in top_10.index:
        if 'Classical' in estimator:
            colors_list.append(colors['Classical'])
        elif 'ML' in estimator:
            colors_list.append(colors['ML'])
        else:
            colors_list.append(colors['Neural'])
    
    # Create horizontal bar chart
    y_pos = np.arange(len(top_10))
    bars = ax.barh(y_pos, top_10['hurst_error'], color=colors_list, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, error) in enumerate(zip(bars, top_10['hurst_error'])):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{error:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.replace('_', ' ') for name in top_10.index], fontsize=10)
    ax.set_xlabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Top 10 Estimators by Accuracy (Lower is Better)', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['Classical'], label='Classical'),
                      Patch(facecolor=colors['ML'], label='ML'),
                      Patch(facecolor=colors['Neural'], label='Neural')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('Figure2_Individual_Performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_figure_3_contamination_robustness(df: pd.DataFrame):
    """Create Figure 3: Contamination robustness."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate performance degradation
    clean_performance = df[df['contamination_level'] == 0.0].groupby('estimator_category')['hurst_error'].mean()
    contaminated_performance = df[df['contamination_level'] > 0.0].groupby('estimator_category')['hurst_error'].mean()
    degradation = (contaminated_performance - clean_performance) / clean_performance * 100
    
    categories = degradation.index
    degradation_values = degradation.values
    
    # Create bar chart
    bars = ax.bar(categories, degradation_values, 
                  color=[colors['Classical'], colors['ML'], colors['Neural']],
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, deg in zip(bars, degradation_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{deg:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_title('Performance Degradation with Contamination', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Estimator Category', fontsize=12)
    ax.set_ylabel('Performance Degradation (%)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(degradation_values) * 1.2)
    
    plt.tight_layout()
    plt.savefig('Figure3_Contamination_Robustness.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_figure_4_speed_accuracy_tradeoff(df: pd.DataFrame):
    """Create Figure 4: Speed vs accuracy trade-off."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Calculate category averages
    category_stats = df.groupby('estimator_category').agg({
        'hurst_error': 'mean',
        'execution_time': 'mean'
    })
    
    # Create scatter plot
    for category in category_stats.index:
        error = category_stats.loc[category, 'hurst_error']
        time = category_stats.loc[category, 'execution_time']
        color = colors[category]
        
        ax.scatter(time, error, s=200, c=color, alpha=0.8, edgecolors='black', linewidth=2, label=category)
        
        # Add category label
        ax.annotate(category, (time, error), 
                   xytext=(10, 10), textcoords='offset points', 
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlabel('Mean Execution Time (seconds)', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Speed vs Accuracy Trade-off', fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('Figure4_Speed_Accuracy_Tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_figure_5_summary_table(df: pd.DataFrame):
    """Create Figure 5: Summary table."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.axis('off')
    
    # Calculate summary statistics
    summary_data = []
    for category in df['estimator_category'].unique():
        subset = df[df['estimator_category'] == category]
        summary_data.append([
            category,
            f"{subset['hurst_error'].mean():.4f}",
            f"{subset['relative_error'].mean()*100:.1f}%",
            f"{subset['execution_time'].mean():.4f}s",
            f"{subset['success'].mean()*100:.1f}%"
        ])
    
    # Create table
    table = ax.table(cellText=summary_data,
                    colLabels=['Category', 'Mean Abs Error', 'Mean Rel Error', 'Mean Time', 'Success Rate'],
                    cellLoc='center',
                    loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Color the header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color the data rows
    for i, category in enumerate(df['estimator_category'].unique()):
        color = colors[category]
        for j in range(5):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
    
    ax.set_title('Performance Summary by Category', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('Figure5_Summary_Table.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all simple, clean figures."""
    print("🎨 Creating Simple, Clean Figures for LRDBenchmark Manuscript")
    print("=" * 70)
    
    # Load results
    results_file = "comprehensive_all_estimators_benchmark_20250905_074313.csv"
    df = load_results(results_file)
    
    print("\n📊 Creating Figure 1: Main Comparison...")
    create_figure_1_main_comparison(df)
    
    print("📊 Creating Figure 2: Individual Performance...")
    create_figure_2_individual_performance(df)
    
    print("📊 Creating Figure 3: Contamination Robustness...")
    create_figure_3_contamination_robustness(df)
    
    print("📊 Creating Figure 4: Speed vs Accuracy Trade-off...")
    create_figure_4_speed_accuracy_tradeoff(df)
    
    print("📊 Creating Figure 5: Summary Table...")
    create_figure_5_summary_table(df)
    
    print("\n✅ All simple figures generated successfully!")
    print("📁 Files created:")
    print("  - Figure1_Main_Comparison.png")
    print("  - Figure2_Individual_Performance.png")
    print("  - Figure3_Contamination_Robustness.png")
    print("  - Figure4_Speed_Accuracy_Tradeoff.png")
    print("  - Figure5_Summary_Table.png")
    print("\n🎯 These figures are simple, clean, and highly readable!")

if __name__ == "__main__":
    main()
