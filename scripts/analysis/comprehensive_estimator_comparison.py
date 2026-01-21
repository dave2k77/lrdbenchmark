#!/usr/bin/env python3
"""
Comprehensive Comparison of All Estimator Categories

This script compares performance across:
1. Classical Estimators (R/S, DFA, GPH, etc.)
2. Machine Learning Estimators (RandomForest, SVR, GradientBoosting)
3. Neural Network Estimators (CNN, LSTM, GRU, Transformer)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class EstimatorCategoryComparison:
    """Comprehensive comparison of all estimator categories."""
    
    def __init__(self):
        self.results_dir = Path(".")
        self.comparison_data = {}
        
    def load_benchmark_results(self):
        """Load results from all benchmark categories."""
        print("📊 Loading Benchmark Results from All Categories...")
        
        # Load Classical Estimators results
        classical_file = self.results_dir / "benchmark_results" / "classical_estimators_benchmark_summary.csv"
        if classical_file.exists():
            classical_df = pd.read_csv(classical_file)
            self.comparison_data["Classical"] = classical_df
            print(f"✅ Loaded Classical results: {len(classical_df)} estimators")
        else:
            print("⚠️ Classical benchmark results not found")
        
        # Load ML Estimators results
        ml_file = self.results_dir / "ml_benchmark_results" / "ml_estimators_benchmark_summary.csv"
        if ml_file.exists():
            ml_df = pd.read_csv(ml_file)
            self.comparison_data["ML"] = ml_df
            print(f"✅ Loaded ML results: {len(ml_df)} estimators")
        else:
            print("⚠️ ML benchmark results not found")
        
        # Load Neural Network results
        nn_file = self.results_dir / "neural_benchmark_results" / "neural_estimators_benchmark_summary.csv"
        if nn_file.exists():
            nn_df = pd.read_csv(nn_file)
            self.comparison_data["Neural Networks"] = nn_df
            print(f"✅ Loaded Neural Network results: {len(nn_df)} estimators")
        else:
            print("⚠️ Neural Network benchmark results not found")
    
    def analyze_performance_by_category(self):
        """Analyze performance metrics by category."""
        print("\n📈 Analyzing Performance by Category...")
        
        category_analysis = {}
        
        for category, df in self.comparison_data.items():
            if df.empty:
                continue
            
            # Handle different column naming conventions
            mae_col = None
            time_col = None
            robustness_col = None
            realistic_col = None
            overall_col = None
            
            # Check for different column name patterns
            for col in df.columns:
                if "Mean_Absolute_Error" in col and ("Pure" in col or "MAE" in col):
                    mae_col = col
                elif "Execution_Time" in col and ("Pure" in col or "Time" in col):
                    time_col = col
                elif "Robustness" in col:
                    robustness_col = col
                elif "Realistic" in col or "Context" in col:
                    realistic_col = col
                elif "Overall" in col and "Score" in col:
                    overall_col = col
            
            analysis = {
                "count": len(df),
                "mean_mae": df[mae_col].mean() if mae_col else np.nan,
                "std_mae": df[mae_col].std() if mae_col else np.nan,
                "mean_execution_time": df[time_col].mean() if time_col else np.nan,
                "std_execution_time": df[time_col].std() if time_col else np.nan,
                "mean_robustness": df[robustness_col].mean() if robustness_col else np.nan,
                "mean_realistic": df[realistic_col].mean() if realistic_col else np.nan,
                "mean_overall_score": df[overall_col].mean() if overall_col else np.nan,
                "best_mae": df[mae_col].min() if mae_col else np.nan,
                "best_execution_time": df[time_col].min() if time_col else np.nan,
                "best_overall_score": df[overall_col].max() if overall_col else np.nan
            }
            
            category_analysis[category] = analysis
        
        return category_analysis
    
    def create_comprehensive_comparison_table(self, category_analysis):
        """Create a comprehensive comparison table."""
        print("\n📋 Creating Comprehensive Comparison Table...")
        
        comparison_data = []
        for category, analysis in category_analysis.items():
            comparison_data.append({
                "Category": category,
                "Estimators": analysis["count"],
                "Mean MAE": f"{analysis['mean_mae']:.3f}" if not np.isnan(analysis['mean_mae']) else "N/A",
                "Best MAE": f"{analysis['best_mae']:.3f}" if not np.isnan(analysis['best_mae']) else "N/A",
                "Mean Execution Time (s)": f"{analysis['mean_execution_time']:.4f}" if not np.isnan(analysis['mean_execution_time']) else "N/A",
                "Fastest (s)": f"{analysis['best_execution_time']:.4f}" if not np.isnan(analysis['best_execution_time']) else "N/A",
                "Mean Robustness": f"{analysis['mean_robustness']:.1f}/10" if not np.isnan(analysis['mean_robustness']) else "N/A",
                "Mean Realistic": f"{analysis['mean_realistic']:.1f}/10" if not np.isnan(analysis['mean_realistic']) else "N/A",
                "Mean Overall Score": f"{analysis['mean_overall_score']:.2f}/10" if not np.isnan(analysis['mean_overall_score']) else "N/A",
                "Best Overall Score": f"{analysis['best_overall_score']:.2f}/10" if not np.isnan(analysis['best_overall_score']) else "N/A"
            })
        
        return pd.DataFrame(comparison_data)
    
    def identify_top_performers(self):
        """Identify top performers across all categories."""
        print("\n🏆 Identifying Top Performers...")
        
        top_performers = {}
        
        # Combine all results
        all_results = []
        for category, df in self.comparison_data.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy["Category"] = category
                all_results.append(df_copy)
        
        if not all_results:
            return top_performers
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Find the correct column names for metrics
        mae_col = None
        time_col = None
        robustness_col = None
        realistic_col = None
        overall_col = None
        
        for col in combined_df.columns:
            if "Mean_Absolute_Error" in col and ("Pure" in col or "MAE" in col):
                mae_col = col
            elif "Execution_Time" in col and ("Pure" in col or "Time" in col):
                time_col = col
            elif "Robustness" in col:
                robustness_col = col
            elif "Realistic" in col or "Context" in col:
                realistic_col = col
            elif "Overall" in col and "Score" in col:
                overall_col = col
        
        # Top performers by different metrics
        top_performers = {
            "Best Accuracy (Lowest MAE)": combined_df.loc[combined_df[mae_col].idxmin()] if mae_col else None,
            "Fastest Execution": combined_df.loc[combined_df[time_col].idxmin()] if time_col else None,
            "Most Robust": combined_df.loc[combined_df[robustness_col].idxmax()] if robustness_col else None,
            "Best Realistic Performance": combined_df.loc[combined_df[realistic_col].idxmax()] if realistic_col else None,
            "Highest Overall Score": combined_df.loc[combined_df[overall_col].idxmax()] if overall_col else None
        }
        
        return top_performers
    
    def create_comprehensive_visualizations(self, category_analysis, top_performers):
        """Create comprehensive visualizations."""
        print("\n📊 Creating Comprehensive Visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        categories = list(category_analysis.keys())
        
        # 1. Mean Absolute Error Comparison
        ax1 = plt.subplot(3, 4, 1)
        mean_mae = [category_analysis[cat]["mean_mae"] for cat in categories]
        std_mae = [category_analysis[cat]["std_mae"] for cat in categories]
        
        bars1 = ax1.bar(categories, mean_mae, yerr=std_mae, alpha=0.7, capsize=5)
        ax1.set_title("Mean Absolute Error by Category", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Mean Absolute Error")
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Execution Time Comparison
        ax2 = plt.subplot(3, 4, 2)
        mean_time = [category_analysis[cat]["mean_execution_time"] for cat in categories]
        
        bars2 = ax2.bar(categories, mean_time, alpha=0.7, color='orange')
        ax2.set_title("Mean Execution Time by Category", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Execution Time (seconds)")
        ax2.set_yscale('log')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Overall Score Comparison
        ax3 = plt.subplot(3, 4, 3)
        mean_score = [category_analysis[cat]["mean_overall_score"] for cat in categories]
        
        bars3 = ax3.bar(categories, mean_score, alpha=0.7, color='green')
        ax3.set_title("Mean Overall Score by Category", fontsize=12, fontweight='bold')
        ax3.set_ylabel("Overall Score (/10)")
        ax3.set_ylim(0, 10)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Robustness Comparison
        ax4 = plt.subplot(3, 4, 4)
        mean_robust = [category_analysis[cat]["mean_robustness"] for cat in categories]
        
        bars4 = ax4.bar(categories, mean_robust, alpha=0.7, color='purple')
        ax4.set_title("Mean Robustness by Category", fontsize=12, fontweight='bold')
        ax4.set_ylabel("Robustness Score (/10)")
        ax4.set_ylim(0, 10)
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Performance vs Speed Scatter
        ax5 = plt.subplot(3, 4, 5)
        colors = ['blue', 'orange', 'green']
        for i, category in enumerate(categories):
            if category in self.comparison_data:
                df = self.comparison_data[category]
                if not df.empty:
                    # Find the correct column names
                    mae_col = None
                    time_col = None
                    for col in df.columns:
                        if "Mean_Absolute_Error" in col and ("Pure" in col or "MAE" in col):
                            mae_col = col
                        elif "Execution_Time" in col and ("Pure" in col or "Time" in col):
                            time_col = col
                    
                    if mae_col and time_col:
                        ax5.scatter(df[time_col], df[mae_col], 
                                  label=category, s=100, alpha=0.7, color=colors[i % len(colors)])
        
        ax5.set_title("Accuracy vs Speed", fontsize=12, fontweight='bold')
        ax5.set_xlabel("Execution Time (seconds)")
        ax5.set_ylabel("Mean Absolute Error")
        ax5.set_xscale('log')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Category Performance Radar Chart
        ax6 = plt.subplot(3, 4, 6, projection='polar')
        
        # Only use metrics that have valid data for all categories
        metrics = ['Accuracy', 'Speed', 'Robustness', 'Overall']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, category in enumerate(categories):
            if category in category_analysis:
                analysis = category_analysis[category]
                
                # Handle NaN values and normalize properly
                mean_mae = analysis["mean_mae"] if not np.isnan(analysis["mean_mae"]) else 1.0
                mean_time = analysis["mean_execution_time"] if not np.isnan(analysis["mean_execution_time"]) else 1.0
                mean_robustness = analysis["mean_robustness"] if not np.isnan(analysis["mean_robustness"]) else 0.0
                mean_overall = analysis["mean_overall_score"] if not np.isnan(analysis["mean_overall_score"]) else 0.0
                
                # Normalize values (lower MAE and execution time are better)
                # For accuracy: convert MAE to score (lower MAE = higher score)
                accuracy_score = max(0, 10 - mean_mae * 20)  # Scale MAE to 0-10 range
                
                # For speed: convert time to score (lower time = higher score)
                speed_score = max(0, 10 - np.log10(mean_time + 1e-6) * 3)  # Scale log(time) to 0-10 range
                
                values = [
                    accuracy_score,
                    speed_score,
                    mean_robustness,
                    mean_overall
                ]
                values += values[:1]  # Complete the circle
                
                # Ensure all values are within valid range
                values = [max(0, min(10, v)) for v in values]
                
                ax6.plot(angles, values, 'o-', linewidth=2, label=category)
                ax6.fill(angles, values, alpha=0.25)
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(metrics)
        ax6.set_ylim(0, 10)
        ax6.set_title("Performance Radar Chart", fontsize=12, fontweight='bold', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 7. Estimator Count by Category
        ax7 = plt.subplot(3, 4, 7)
        counts = [category_analysis[cat]["count"] for cat in categories]
        
        bars7 = ax7.bar(categories, counts, alpha=0.7, color='teal')
        ax7.set_title("Number of Estimators by Category", fontsize=12, fontweight='bold')
        ax7.set_ylabel("Count")
        ax7.tick_params(axis='x', rotation=45)
        
        # 8. Best Individual Performers
        ax8 = plt.subplot(3, 4, 8)
        if top_performers and "Highest Overall Score" in top_performers and top_performers["Highest Overall Score"] is not None:
            best = top_performers["Highest Overall Score"]
            ax8.text(0.1, 0.8, f"🏆 Best Overall: {best['Estimator']} ({best['Category']})", 
                    fontsize=12, fontweight='bold', transform=ax8.transAxes)
            ax8.text(0.1, 0.6, f"Score: {best['Overall_Score']:.2f}/10", 
                    fontsize=10, transform=ax8.transAxes)
            ax8.text(0.1, 0.4, f"MAE: {best['Pure_Mean_Absolute_Error']:.3f}", 
                    fontsize=10, transform=ax8.transAxes)
            ax8.text(0.1, 0.2, f"Time: {best['Pure_Mean_Execution_Time']:.4f}s", 
                    fontsize=10, transform=ax8.transAxes)
        
        ax8.set_title("Best Performer", fontsize=12, fontweight='bold')
        ax8.axis('off')
        
        # 9-12. Individual category breakdowns
        for i, category in enumerate(categories[:4]):  # Limit to 4 categories
            ax = plt.subplot(3, 4, 9 + i)
            
            if category in self.comparison_data:
                df = self.comparison_data[category]
                if not df.empty:
                    # Plot MAE distribution
                    if "Pure_Mean_Absolute_Error" in df.columns:
                        ax.hist(df["Pure_Mean_Absolute_Error"], bins=10, alpha=0.7, color=colors[i % len(colors)])
                        ax.set_title(f"{category}: MAE Distribution", fontsize=10, fontweight='bold')
                        ax.set_xlabel("Mean Absolute Error")
                        ax.set_ylabel("Count")
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.results_dir / "comprehensive_estimator_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comprehensive comparison visualization saved to {plot_path}")
    
    def generate_comprehensive_report(self, comparison_table, category_analysis, top_performers):
        """Generate comprehensive comparison report."""
        print("\n📋 Generating Comprehensive Comparison Report...")
        
        report = {
            "metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "categories_compared": list(category_analysis.keys()),
                "total_estimators": sum(analysis["count"] for analysis in category_analysis.values())
            },
            "category_analysis": category_analysis,
            "comparison_table": comparison_table.to_dict('records'),
            "top_performers": {}
        }
        
        # Add top performers info
        for metric, performer in top_performers.items():
            if performer is not None:
                report["top_performers"][metric] = {
                    "estimator": performer.get("Estimator", "Unknown"),
                    "category": performer.get("Category", "Unknown"),
                    "value": performer.get(metric.split()[1].lower().replace("(", "").replace(")", ""), "N/A")
                }
        
        # Save JSON report
        json_path = self.results_dir / "comprehensive_estimator_comparison.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Comprehensive report saved to {json_path}")
        
        return report
    
    def print_comprehensive_summary(self, comparison_table, category_analysis, top_performers):
        """Print comprehensive summary."""
        print("\n" + "="*80)
        print("🏆 COMPREHENSIVE ESTIMATOR CATEGORY COMPARISON")
        print("="*80)
        
        print(f"\n📊 COMPARISON TABLE:")
        print(comparison_table.to_string(index=False))
        
        print(f"\n🏆 TOP PERFORMERS ACROSS ALL CATEGORIES:")
        for metric, performer in top_performers.items():
            if performer is not None:
                print(f"   • {metric}: {performer['Estimator']} ({performer['Category']})")
        
        print(f"\n📈 KEY INSIGHTS:")
        
        # Find best category for each metric (filtering out NaN values)
        valid_accuracy = {k: v for k, v in category_analysis.items() if not np.isnan(v["mean_mae"])}
        valid_speed = {k: v for k, v in category_analysis.items() if not np.isnan(v["mean_execution_time"])}
        valid_overall = {k: v for k, v in category_analysis.items() if not np.isnan(v["mean_overall_score"])}
        
        if valid_accuracy:
            best_accuracy_cat = min(valid_accuracy.items(), key=lambda x: x[1]["mean_mae"])[0]
            print(f"   • Best Accuracy: {best_accuracy_cat} (MAE: {category_analysis[best_accuracy_cat]['mean_mae']:.3f})")
        
        if valid_speed:
            best_speed_cat = min(valid_speed.items(), key=lambda x: x[1]["mean_execution_time"])[0]
            print(f"   • Fastest Execution: {best_speed_cat} ({category_analysis[best_speed_cat]['mean_execution_time']:.4f}s)")
        
        if valid_overall:
            best_overall_cat = max(valid_overall.items(), key=lambda x: x[1]["mean_overall_score"])[0]
            print(f"   • Highest Overall Score: {best_overall_cat} ({category_analysis[best_overall_cat]['mean_overall_score']:.2f}/10)")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if valid_accuracy:
            print(f"   • For High Accuracy: Use {best_accuracy_cat} estimators")
        if valid_speed:
            print(f"   • For Real-time Applications: Use {best_speed_cat} estimators")
        if valid_overall:
            print(f"   • For General Purpose: Use {best_overall_cat} estimators")
        
        print("\n" + "="*80)
    
    def run_comprehensive_comparison(self):
        """Run the complete comparison analysis."""
        print("🔍 Starting Comprehensive Estimator Category Comparison")
        print("="*80)
        
        # Load results
        self.load_benchmark_results()
        
        if not self.comparison_data:
            print("❌ No benchmark results found. Please run benchmarks first.")
            return
        
        # Analyze performance
        category_analysis = self.analyze_performance_by_category()
        
        # Create comparison table
        comparison_table = self.create_comprehensive_comparison_table(category_analysis)
        
        # Identify top performers
        top_performers = self.identify_top_performers()
        
        # Create visualizations
        self.create_comprehensive_visualizations(category_analysis, top_performers)
        
        # Generate report
        report = self.generate_comprehensive_report(comparison_table, category_analysis, top_performers)
        
        # Print summary
        self.print_comprehensive_summary(comparison_table, category_analysis, top_performers)
        
        return report

def main():
    """Run comprehensive estimator comparison."""
    comparison = EstimatorCategoryComparison()
    results = comparison.run_comprehensive_comparison()
    return results

if __name__ == "__main__":
    results = main()
