#!/usr/bin/env python3
"""
Create Clean Error Quantification Figure for LRDBenchmark Manuscript

This script creates a simple, clean figure showing key error metrics
without overwhelming detail.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set clean style
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

# Colors
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

def create_clean_error_figure(df: pd.DataFrame):
    """Create clean error quantification figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Error Quantification Analysis', fontsize=16, fontweight='bold', y=0.95)
    
    # Panel A: Mean Absolute Error Comparison
    classical_error = df[df['estimator_category'] == 'Classical']['hurst_error'].mean()
    ml_error = df[df['estimator_category'] == 'ML']['hurst_error'].mean()
    neural_error = df[df['estimator_category'] == 'Neural']['hurst_error'].mean()
    
    categories = ['Classical', 'ML', 'Neural']
    mean_errors = [classical_error, ml_error, neural_error]
    
    bars = axes[0].bar(categories, mean_errors, 
                      color=[colors['Classical'], colors['ML'], colors['Neural']],
                      alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, error in zip(bars, mean_errors):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{error:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    axes[0].set_title('Mean Absolute Error by Category', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Estimator Category', fontsize=12)
    axes[0].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, max(mean_errors) * 1.2)
    
    # Panel B: Error Distribution (Box Plot)
    sns.boxplot(data=df, x='estimator_category', y='hurst_error', 
                palette=[colors['Classical'], colors['ML'], colors['Neural']], ax=axes[1])
    axes[1].set_title('Error Distribution by Category', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Estimator Category', fontsize=12)
    axes[1].set_ylabel('Absolute Error', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('Figure_Clean_Error_Quantification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print key statistics
    print("\n📊 KEY ERROR QUANTIFICATION STATISTICS")
    print("=" * 50)
    
    print(f"\nMean Absolute Error:")
    print(f"  Classical: {classical_error:.4f}")
    print(f"  ML:        {ml_error:.4f}")
    print(f"  Neural:    {neural_error:.4f}")
    
    print(f"\nPerformance Advantages:")
    ml_advantage_classical = (classical_error - ml_error) / classical_error * 100
    ml_advantage_neural = (neural_error - ml_error) / neural_error * 100
    print(f"  ML vs Classical: {ml_advantage_classical:.1f}% better")
    print(f"  ML vs Neural:    {ml_advantage_neural:.1f}% better")
    
    # Calculate standard deviations
    classical_std = df[df['estimator_category'] == 'Classical']['hurst_error'].std()
    ml_std = df[df['estimator_category'] == 'ML']['hurst_error'].std()
    neural_std = df[df['estimator_category'] == 'Neural']['hurst_error'].std()
    
    print(f"\nError Variability (Standard Deviation):")
    print(f"  Classical: {classical_std:.4f}")
    print(f"  ML:        {ml_std:.4f}")
    print(f"  Neural:    {neural_std:.4f}")

def main():
    """Generate clean error quantification figure."""
    print("📊 Creating Clean Error Quantification Figure")
    print("=" * 50)
    
    # Load results
    results_file = "comprehensive_all_estimators_benchmark_20250905_074313.csv"
    df = load_results(results_file)
    
    print("\n📊 Creating Clean Error Analysis...")
    create_clean_error_figure(df)
    
    print("\n✅ Clean error quantification figure generated successfully!")
    print("📁 File created: Figure_Clean_Error_Quantification.png")
    print("\n🎯 This figure provides clear, simple error analysis!")

if __name__ == "__main__":
    main()
