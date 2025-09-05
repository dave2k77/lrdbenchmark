#!/usr/bin/env python3
"""
Generate LaTeX tables from benchmark results
"""

import pandas as pd
import numpy as np

def load_and_process_data():
    """Load and process the benchmark data"""
    # Load the CSV data
    df = pd.read_csv('final_results/data/comprehensive_all_estimators_benchmark_20250905_074313.csv')
    
    # Calculate summary statistics by estimator
    summary_stats = df.groupby(['estimator_name', 'estimator_category']).agg({
        'hurst_error': ['mean', 'std'],
        'relative_error': ['mean', 'std'],
        'execution_time': ['mean', 'std'],
        'success': 'mean'
    }).round(4)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()
    
    # Calculate category-level statistics
    category_stats = df.groupby('estimator_category').agg({
        'hurst_error': ['mean', 'std'],
        'relative_error': ['mean', 'std'],
        'execution_time': ['mean', 'std'],
        'success': 'mean'
    }).round(4)
    
    # Flatten column names
    category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
    category_stats = category_stats.reset_index()
    
    return summary_stats, category_stats, df

def generate_individual_estimator_table(summary_stats):
    """Generate LaTeX table for individual estimator performance"""
    
    # Sort by mean absolute error
    summary_stats = summary_stats.sort_values('hurst_error_mean')
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Individual Estimator Performance}
\\label{tab:individual_estimators}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Estimator} & \\textbf{Mean Error} & \\textbf{Std Dev} & \\textbf{Mean Time (s)} \\\\
\\midrule
"""
    
    for _, row in summary_stats.iterrows():
        estimator_name = row['estimator_name'].replace('_', ' ')
        mean_error = f"{row['hurst_error_mean']:.4f}"
        std_error = f"{row['hurst_error_std']:.4f}"
        mean_time = f"{row['execution_time_mean']:.4f}"
        
        latex_table += f"{estimator_name} & {mean_error} & {std_error} & {mean_time} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex_table

def generate_category_comparison_table(category_stats):
    """Generate LaTeX table for category comparison"""
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Performance Summary by Estimator Category}
\\label{tab:category_comparison}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Category} & \\textbf{Mean Error} & \\textbf{Mean Time (s)} & \\textbf{Success Rate} \\\\
\\midrule
"""
    
    for _, row in category_stats.iterrows():
        category = row['estimator_category']
        mean_error = f"{row['hurst_error_mean']:.4f}"
        mean_time = f"{row['execution_time_mean']:.4f}"
        success_rate = f"{row['success_mean']:.3f}"
        
        latex_table += f"{category} & {mean_error} & {mean_time} & {success_rate} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex_table

def generate_contamination_analysis_table(df):
    """Generate LaTeX table for contamination effects"""
    
    # Calculate contamination effects
    contamination_stats = df.groupby(['estimator_category', 'contamination_level']).agg({
        'hurst_error': 'mean'
    }).reset_index()
    
    # Pivot to get contamination levels as columns
    contamination_pivot = contamination_stats.pivot(
        index='estimator_category', 
        columns='contamination_level', 
        values='hurst_error'
    ).round(4)
    
    # Calculate degradation percentages
    degradation = ((contamination_pivot[0.2] - contamination_pivot[0.0]) / contamination_pivot[0.0] * 100).round(1)
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Contamination Effects on Estimator Performance}
\\label{tab:contamination_effects}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Category} & \\textbf{0\\% Contamination} & \\textbf{20\\% Contamination} & \\textbf{Degradation (\\%)} \\\\
\\midrule
"""
    
    for category in contamination_pivot.index:
        clean_error = f"{contamination_pivot.loc[category, 0.0]:.4f}"
        contaminated_error = f"{contamination_pivot.loc[category, 0.2]:.4f}"
        degradation_pct = f"{degradation[category]:.1f}"
        
        latex_table += f"{category} & {clean_error} & {contaminated_error} & {degradation_pct} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex_table

def generate_data_model_analysis_table(df):
    """Generate LaTeX table for data model performance"""
    
    # Calculate performance by data model
    model_stats = df.groupby(['model_name', 'estimator_category']).agg({
        'hurst_error': 'mean'
    }).reset_index()
    
    # Pivot to get categories as columns
    model_pivot = model_stats.pivot(
        index='model_name', 
        columns='estimator_category', 
        values='hurst_error'
    ).round(4)
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Performance by Data Model}
\\label{tab:data_model_performance}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Data Model} & \\textbf{Classical} & \\textbf{ML} & \\textbf{Neural} \\\\
\\midrule
"""
    
    for model in model_pivot.index:
        classical = f"{model_pivot.loc[model, 'Classical']:.4f}"
        ml = f"{model_pivot.loc[model, 'ML']:.4f}"
        neural = f"{model_pivot.loc[model, 'Neural']:.4f}"
        
        latex_table += f"{model} & {classical} & {ml} & {neural} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex_table

def generate_hurst_value_analysis_table(df):
    """Generate LaTeX table for performance by Hurst value"""
    
    # Calculate performance by Hurst value
    hurst_stats = df.groupby(['true_hurst', 'estimator_category']).agg({
        'hurst_error': 'mean'
    }).reset_index()
    
    # Pivot to get categories as columns
    hurst_pivot = hurst_stats.pivot(
        index='true_hurst', 
        columns='estimator_category', 
        values='hurst_error'
    ).round(4)
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Performance by True Hurst Value}
\\label{tab:hurst_value_performance}
\\begin{tabular}{lccc}
\\toprule
\\textbf{True Hurst} & \\textbf{Classical} & \\textbf{ML} & \\textbf{Neural} \\\\
\\midrule
"""
    
    for hurst in sorted(hurst_pivot.index):
        classical = f"{hurst_pivot.loc[hurst, 'Classical']:.4f}"
        ml = f"{hurst_pivot.loc[hurst, 'ML']:.4f}"
        neural = f"{hurst_pivot.loc[hurst, 'Neural']:.4f}"
        
        latex_table += f"{hurst:.1f} & {classical} & {ml} & {neural} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex_table

def main():
    """Generate all LaTeX tables"""
    print("Loading and processing benchmark data...")
    summary_stats, category_stats, df = load_and_process_data()
    
    print("Generating LaTeX tables...")
    
    # Generate all tables
    tables = {
        'individual_estimators': generate_individual_estimator_table(summary_stats),
        'category_comparison': generate_category_comparison_table(category_stats),
        'contamination_effects': generate_contamination_analysis_table(df),
        'data_model_performance': generate_data_model_analysis_table(df),
        'hurst_value_performance': generate_hurst_value_analysis_table(df)
    }
    
    # Write tables to file
    with open('latex_tables.tex', 'w') as f:
        f.write("% LaTeX Tables Generated from LRDBenchmark Results\n")
        f.write("% Generated automatically from comprehensive_all_estimators_benchmark_20250905_074313.csv\n\n")
        
        for table_name, table_content in tables.items():
            f.write(f"% {table_name.replace('_', ' ').title()} Table\n")
            f.write(table_content)
            f.write("\n\n")
    
    print("LaTeX tables written to latex_tables.tex")
    
    # Also print individual tables to console
    for table_name, table_content in tables.items():
        print(f"\n{'='*60}")
        print(f"TABLE: {table_name.replace('_', ' ').title()}")
        print('='*60)
        print(table_content)

if __name__ == "__main__":
    main()
