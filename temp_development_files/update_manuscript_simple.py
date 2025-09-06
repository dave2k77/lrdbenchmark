#!/usr/bin/env python3
"""
Simple script to update the manuscript with the latest comprehensive adaptive estimator results.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_latest_results():
    """Load the most recent comprehensive results."""
    results_files = [
        "comprehensive_results/comprehensive_classical_summary_20250905_111607.json",
        "demo_results/quick_classical_eeg_summary_20250905_113108.json"
    ]
    
    results = {}
    for file_path in results_files:
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                results[file_path] = json.load(f)
    
    return results

def create_comprehensive_figures(results):
    """Create comprehensive figures for the manuscript."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure directory
    fig_dir = Path("final_results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Comprehensive Adaptive Estimator Performance
    demo_results = results.get("demo_results/quick_classical_eeg_summary_20250905_113108.json", {})
    if demo_results:
        estimator_perf = demo_results.get("estimator_performance", {})
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        estimators = list(estimator_perf.keys())
        success_rates = [estimator_perf[est]["success_rate"] for est in estimators]
        mean_errors = [estimator_perf[est]["mean_error"] for est in estimators]
        exec_times = [estimator_perf[est]["mean_execution_time"] for est in estimators]
        
        # Clean estimator names
        clean_names = [est.replace("Adaptive_", "") for est in estimators]
        
        # Success rates
        bars1 = ax1.bar(clean_names, success_rates, color='skyblue', alpha=0.7)
        ax1.set_title('Success Rate by Estimator', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Mean errors
        bars2 = ax2.bar(clean_names, mean_errors, color='lightcoral', alpha=0.7)
        ax2.set_title('Mean Absolute Error by Estimator', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, mean_errors):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Execution times (log scale)
        bars3 = ax3.bar(clean_names, exec_times, color='lightgreen', alpha=0.7)
        ax3.set_title('Mean Execution Time by Estimator', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Execution Time (seconds)')
        ax3.set_yscale('log')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars3, exec_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                    f'{value:.3f}s', ha='center', va='bottom', fontsize=10)
        
        # Performance scatter plot
        scatter = ax4.scatter(exec_times, mean_errors, s=[sr*3 for sr in success_rates], 
                             c=success_rates, cmap='viridis', alpha=0.7, edgecolors='black')
        ax4.set_xlabel('Execution Time (seconds)')
        ax4.set_ylabel('Mean Absolute Error')
        ax4.set_title('Performance Trade-offs\n(Size = Success Rate)', fontsize=14, fontweight='bold')
        ax4.set_xscale('log')
        
        # Add estimator labels
        for i, name in enumerate(clean_names):
            ax4.annotate(name, (exec_times[i], mean_errors[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Success Rate (%)')
        
        plt.tight_layout()
        plt.savefig(fig_dir / "comprehensive_adaptive_estimator_performance.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Created comprehensive_adaptive_estimator_performance.png")
    
    # Figure 2: EEG Contamination Robustness
    if demo_results:
        eeg_perf = demo_results.get("eeg_scenario_performance", {})
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract EEG scenario data
        scenarios = list(eeg_perf.keys())
        eeg_success_rates = [eeg_perf[scenario]["success_rate"] for scenario in scenarios]
        eeg_mean_errors = [eeg_perf[scenario]["mean_error"] for scenario in scenarios]
        
        # Clean scenario names
        clean_scenarios = [scenario.replace("eeg_", "").replace("_", " ").title() for scenario in scenarios]
        
        # Success rates
        bars1 = ax1.bar(clean_scenarios, eeg_success_rates, color='orange', alpha=0.7)
        ax1.set_title('Success Rate by EEG Contamination Scenario', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars1, eeg_success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Mean errors
        bars2 = ax2.bar(clean_scenarios, eeg_mean_errors, color='purple', alpha=0.7)
        ax2.set_title('Mean Absolute Error by EEG Contamination Scenario', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars2, eeg_mean_errors):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(fig_dir / "eeg_contamination_robustness.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Created eeg_contamination_robustness.png")

def update_manuscript_sections(results):
    """Update specific sections of the manuscript with latest results."""
    
    demo_results = results.get("demo_results/quick_classical_eeg_summary_20250905_113108.json", {})
    comprehensive_results = results.get("comprehensive_results/comprehensive_classical_summary_20250905_111607.json", {})
    
    if not demo_results:
        print("No demo results found!")
        return
    
    # Extract key statistics
    total_tests = demo_results.get("overall", {}).get("total_tests", 0)
    success_rate = demo_results.get("overall", {}).get("success_rate", 0)
    mean_error = demo_results.get("overall", {}).get("mean_error", 0)
    mean_exec_time = demo_results.get("overall", {}).get("mean_execution_time", 0)
    
    estimator_perf = demo_results.get("estimator_performance", {})
    framework_usage = demo_results.get("framework_usage", {})
    eeg_perf = demo_results.get("eeg_scenario_performance", {})
    
    print(f"\n=== LATEST BENCHMARK RESULTS ===")
    print(f"Total test cases: {total_tests}")
    print(f"Overall success rate: {success_rate:.1f}%")
    print(f"Mean absolute error: {mean_error:.3f}")
    print(f"Mean execution time: {mean_exec_time:.3f}s")
    
    print(f"\n=== ESTIMATOR PERFORMANCE ===")
    for estimator, perf in estimator_perf.items():
        clean_name = estimator.replace("Adaptive_", "")
        print(f"{clean_name:15s}: Success={perf['success_rate']:5.1f}%, Error={perf['mean_error']:.3f}, Time={perf['mean_execution_time']:.3f}s")
    
    print(f"\n=== FRAMEWORK USAGE ===")
    total_framework_uses = sum(framework_usage.values())
    for framework, count in framework_usage.items():
        percentage = count / total_framework_uses * 100
        print(f"{framework.upper():10s}: {count:3d} uses ({percentage:5.1f}%)")
    
    print(f"\n=== EEG CONTAMINATION ROBUSTNESS ===")
    for scenario, perf in eeg_perf.items():
        clean_scenario = scenario.replace("eeg_", "").replace("_", " ").title()
        print(f"{clean_scenario:25s}: Success={perf['success_rate']:5.1f}%, Error={perf['mean_error']:.3f}")
    
    # Create summary for manuscript update
    print(f"\n=== MANUSCRIPT UPDATE SUMMARY ===")
    print("Key statistics to update in manuscript:")
    print(f"- Total test cases: {total_tests}")
    print(f"- Success rate: {success_rate:.1f}%")
    print(f"- Mean absolute error: {mean_error:.3f}")
    print(f"- Best estimator: {min(estimator_perf.keys(), key=lambda k: estimator_perf[k]['mean_error']).replace('Adaptive_', '')}")
    print(f"- Most used framework: {max(framework_usage.keys(), key=lambda k: framework_usage[k]).upper()}")
    print(f"- EEG robustness: {min(eeg_perf.values(), key=lambda v: v['success_rate'])['success_rate']:.1f}% minimum success rate")

def main():
    """Main function to update manuscript with latest results."""
    print("Loading latest results...")
    results = load_latest_results()
    
    print("Creating comprehensive figures...")
    create_comprehensive_figures(results)
    
    print("Analyzing results for manuscript update...")
    update_manuscript_sections(results)
    
    print("\nManuscript update analysis complete!")
    print("Please manually update the manuscript.tex file with the statistics shown above.")

if __name__ == "__main__":
    main()
