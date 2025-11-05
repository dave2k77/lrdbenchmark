#!/usr/bin/env python3
"""
Script to update the research folder with the latest benchmark results.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

def load_latest_benchmark():
    """Load the most recent benchmark JSON file."""
    benchmark_dir = Path("benchmark_results")
    json_files = sorted(benchmark_dir.glob("comprehensive_benchmark_*.json"), reverse=True)
    
    if not json_files:
        print("No benchmark JSON files found")
        return None
    
    latest_file = json_files[0]
    print(f"Loading latest benchmark: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def load_latest_csv():
    """Load the most recent benchmark CSV file."""
    benchmark_dir = Path("benchmark_results")
    csv_files = sorted(benchmark_dir.glob("benchmark_summary_*.csv"), reverse=True)
    
    if not csv_files:
        print("No benchmark CSV files found")
        return None
    
    latest_file = csv_files[0]
    print(f"Loading latest CSV: {latest_file.name}")
    
    return pd.read_csv(latest_file), latest_file

def process_results_for_research(benchmark_data, csv_df):
    """Process benchmark results into research-ready format."""
    
    # Extract summary statistics
    summary = {
        "timestamp": benchmark_data.get("timestamp", datetime.now().isoformat()),
        "total_tests": benchmark_data.get("total_tests", 0),
        "successful_tests": benchmark_data.get("successful_tests", 0),
        "success_rate": benchmark_data.get("success_rate", 0.0),
        "data_models_tested": benchmark_data.get("data_models_tested", 0),
        "estimators_tested": benchmark_data.get("estimators_tested", 0),
    }
    
    # Create comprehensive leaderboard CSV
    leaderboard_data = []
    
    for model_name, model_results in benchmark_data.get("results", {}).items():
        true_hurst = model_results.get("data_params", {}).get("H", None)
        
        for est_result in model_results.get("estimator_results", []):
            if est_result.get("success", False):
                leaderboard_data.append({
                    "estimator": est_result["estimator"],
                    "category": _get_category(est_result["estimator"]),
                    "data_model": model_name,
                    "error": est_result.get("error", None),
                    "time": est_result.get("execution_time", None),
                    "hurst_estimated": est_result.get("estimated_hurst", None),
                    "hurst_true": true_hurst,
                    "r_squared": est_result.get("r_squared", None),
                    "pretrained": _is_pretrained(est_result["estimator"]),
                })
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    
    # Sort by error (ascending)
    leaderboard_df = leaderboard_df.sort_values("error", ascending=True, na_position='last')
    
    return summary, leaderboard_df

def _get_category(estimator_name):
    """Determine estimator category."""
    classical = ["R/S", "DFA", "DMA", "Higuchi", "GPH", "Whittle", "Periodogram"]
    wavelet = ["CWT", "WaveletVar", "WaveletLogVar", "WaveletWhittle", "WaveletLeaders"]
    multifractal = ["MFDFA"]
    ml = ["RandomForest", "SVR", "GradientBoosting"]
    neural = ["CNN", "LSTM", "GRU", "Transformer"]
    
    if estimator_name in classical:
        return "Classical"
    elif estimator_name in wavelet:
        return "Wavelet"
    elif estimator_name in multifractal:
        return "Multifractal"
    elif estimator_name in ml:
        return "ML"
    elif estimator_name in neural:
        return "Neural"
    else:
        return "Unknown"

def _is_pretrained(estimator_name):
    """Check if estimator uses pretrained models."""
    pretrained = ["RandomForest", "SVR", "GradientBoosting", "CNN", "LSTM", "GRU", "Transformer"]
    return estimator_name in pretrained

def create_category_performance_summary(leaderboard_df):
    """Create category-wise performance summary."""
    category_summary = []
    
    for category in leaderboard_df["category"].unique():
        cat_data = leaderboard_df[leaderboard_df["category"] == category]
        valid_errors = cat_data["error"].dropna()
        
        if len(valid_errors) > 0:
            category_summary.append({
                "category": category,
                "average_error": float(valid_errors.mean()),
                "error_range_min": float(valid_errors.min()),
                "error_range_max": float(valid_errors.max()),
                "average_time": float(cat_data["time"].mean()),
                "number_of_tests": len(cat_data),
            })
    
    return pd.DataFrame(category_summary)

def update_research_folder():
    """Update research folder with latest benchmark results."""
    print("=" * 80)
    print("Updating Research Folder with Latest Benchmark Results")
    print("=" * 80)
    
    # Load latest data
    benchmark_data = load_latest_benchmark()
    if benchmark_data is None:
        print("Failed to load benchmark data")
        return
    
    csv_df, csv_file = load_latest_csv()
    if csv_df is None:
        print("Failed to load CSV data")
        return
    
    # Process results
    summary, leaderboard_df = process_results_for_research(benchmark_data, csv_df)
    category_summary = create_category_performance_summary(leaderboard_df)
    
    # Update research folder
    research_dir = Path("research")
    tables_dir = research_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    # Save comprehensive leaderboard
    leaderboard_path = tables_dir / "comprehensive_leaderboard_latest.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)
    print(f"✅ Saved comprehensive leaderboard to {leaderboard_path}")
    
    # Save as JSON too
    leaderboard_json_path = tables_dir / "comprehensive_leaderboard_latest.json"
    leaderboard_df.to_json(leaderboard_json_path, orient="records", indent=2)
    print(f"✅ Saved comprehensive leaderboard JSON to {leaderboard_json_path}")
    
    # Save category performance summary
    category_path = tables_dir / "category_performance_summary.csv"
    category_summary.to_csv(category_path, index=False)
    print(f"✅ Saved category performance summary to {category_path}")
    
    # Update latest benchmark results markdown
    md_content = f"""# Latest Benchmark Results Summary

**Generated**: {summary['timestamp']}

## Executive Summary

- **Total Tests**: {summary['total_tests']}
- **Success Rate**: {summary['success_rate']*100:.1f}%
- **Data Models**: {summary['data_models_tested']}
- **Estimators**: {summary['estimators_tested']}
- **Pretrained Models**: {len(leaderboard_df[leaderboard_df['pretrained'] == True])}

## Top 10 Performers

| Rank | Estimator | Category | Data Model | Error | Time (s) |
|------|-----------|----------|------------|-------|----------|
"""
    
    # Add top 10 performers
    top_10 = leaderboard_df.head(10)
    for idx, row in enumerate(top_10.itertuples(), 1):
        md_content += f"| {idx} | {row.estimator} | {row.category} | {row.data_model} | {row.error:.4f} | {row.time:.3f} |\n"
    
    md_content += "\n## Category Performance\n\n"
    
    # Add category summaries
    for _, cat_row in category_summary.iterrows():
        md_content += f"### {cat_row['category']}\n"
        md_content += f"- Average Error: {cat_row['average_error']:.4f}\n"
        md_content += f"- Error Range: {cat_row['error_range_min']:.4f} - {cat_row['error_range_max']:.4f}\n"
        md_content += f"- Average Time: {cat_row['average_time']:.3f}s\n"
        md_content += f"- Number of Tests: {int(cat_row['number_of_tests'])}\n\n"
    
    # Save markdown
    md_path = research_dir / "latest_benchmark_results.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"✅ Updated latest benchmark results markdown: {md_path}")
    
    print("\n" + "=" * 80)
    print("Research folder update complete!")
    print("=" * 80)

if __name__ == "__main__":
    update_research_folder()

