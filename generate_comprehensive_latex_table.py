#!/usr/bin/env python3
"""
Generate a comprehensive LaTeX table with all key statistics
"""

import pandas as pd
import numpy as np

def load_and_process_data():
    """Load and process the benchmark data"""
    df = pd.read_csv('final_results/data/comprehensive_all_estimators_benchmark_20250905_074313.csv')
    
    # Calculate comprehensive statistics by estimator
    stats = df.groupby(['estimator_name', 'estimator_category']).agg({
        'hurst_error': ['mean', 'std', 'min', 'max'],
        'relative_error': ['mean', 'std'],
        'execution_time': ['mean', 'std', 'min', 'max'],
        'success': 'mean'
    }).round(4)
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns]
    stats = stats.reset_index()
    
    # Calculate additional metrics
    stats['coefficient_variation'] = (stats['hurst_error_std'] / stats['hurst_error_mean'] * 100).round(1)
    stats['efficiency_score'] = (1 / (stats['hurst_error_mean'] * stats['execution_time_mean'])).round(2)
    
    return stats

def generate_comprehensive_table(stats):
    """Generate comprehensive LaTeX table"""
    
    # Sort by mean absolute error
    stats = stats.sort_values('hurst_error_mean')
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Comprehensive Estimator Performance Analysis}
\\label{tab:comprehensive_analysis}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llccccccc}
\\toprule
\\textbf{Estimator} & \\textbf{Category} & \\textbf{Mean Error} & \\textbf{Std Dev} & \\textbf{CV (\\%)} & \\textbf{Mean Time (s)} & \\textbf{Min Time (s)} & \\textbf{Max Time (s)} & \\textbf{Efficiency} \\\\
\\midrule
"""
    
    for _, row in stats.iterrows():
        estimator_name = row['estimator_name'].replace('_', ' ')
        category = row['estimator_category']
        mean_error = f"{row['hurst_error_mean']:.4f}"
        std_error = f"{row['hurst_error_std']:.4f}"
        cv = f"{row['coefficient_variation']:.1f}"
        mean_time = f"{row['execution_time_mean']:.4f}"
        min_time = f"{row['execution_time_min']:.4f}"
        max_time = f"{row['execution_time_max']:.4f}"
        efficiency = f"{row['efficiency_score']:.2f}"
        
        latex_table += f"{estimator_name} & {category} & {mean_error} & {std_error} & {cv} & {mean_time} & {min_time} & {max_time} & {efficiency} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}%
}
\\end{table}"""
    
    return latex_table

def generate_statistical_summary_table(stats):
    """Generate statistical summary table"""
    
    # Calculate category-level statistics
    category_stats = stats.groupby('estimator_category').agg({
        'hurst_error_mean': ['mean', 'std', 'min', 'max'],
        'hurst_error_std': 'mean',
        'execution_time_mean': ['mean', 'std', 'min', 'max'],
        'coefficient_variation': 'mean',
        'efficiency_score': 'mean'
    }).round(4)
    
    # Flatten column names
    category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
    category_stats = category_stats.reset_index()
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Statistical Summary by Category}
\\label{tab:statistical_summary}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Category} & \\textbf{Mean Error} & \\textbf{Avg Std Dev} & \\textbf{Mean Time (s)} & \\textbf{Avg CV (\\%)} \\\\
\\midrule
"""
    
    for _, row in category_stats.iterrows():
        category = row['estimator_category']
        mean_error = f"{row['hurst_error_mean_mean']:.4f}"
        avg_std = f"{row['hurst_error_std_mean']:.4f}"
        mean_time = f"{row['execution_time_mean_mean']:.4f}"
        avg_cv = f"{row['coefficient_variation_mean']:.1f}"
        
        latex_table += f"{category} & {mean_error} & {avg_std} & {mean_time} & {avg_cv} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex_table

def main():
    """Generate comprehensive LaTeX tables"""
    print("Loading and processing benchmark data...")
    stats = load_and_process_data()
    
    print("Generating comprehensive LaTeX tables...")
    
    # Generate tables
    comprehensive_table = generate_comprehensive_table(stats)
    summary_table = generate_statistical_summary_table(stats)
    
    # Write to file
    with open('comprehensive_latex_tables.tex', 'w') as f:
        f.write("% Comprehensive LaTeX Tables Generated from LRDBenchmark Results\n")
        f.write("% Generated automatically from comprehensive_all_estimators_benchmark_20250905_074313.csv\n\n")
        
        f.write("% Comprehensive Analysis Table\n")
        f.write(comprehensive_table)
        f.write("\n\n")
        
        f.write("% Statistical Summary Table\n")
        f.write(summary_table)
        f.write("\n\n")
    
    print("Comprehensive LaTeX tables written to comprehensive_latex_tables.tex")
    
    # Print to console
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS TABLE")
    print('='*80)
    print(comprehensive_table)
    
    print(f"\n{'='*80}")
    print("STATISTICAL SUMMARY TABLE")
    print('='*80)
    print(summary_table)

if __name__ == "__main__":
    main()
