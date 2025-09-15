#!/usr/bin/env python3
"""
Comprehensive Estimator Leaderboard

This script creates detailed leaderboards for:
1. Overall Performance Rankings
2. Category-wise Rankings (Classical, ML, Neural Networks)
3. Contamination Robustness Rankings
4. Dataset Quality Assessment
5. Cross-contamination Performance Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

class ComprehensiveLeaderboard:
    """Create comprehensive leaderboards for all estimators."""
    
    def __init__(self):
        self.results_dir = Path(".")
        self.leaderboard_data = {}
        self.quality_scores = {}
        
    def load_all_benchmark_data(self):
        """Load data from all benchmark categories."""
        print("📊 Loading All Benchmark Data...")
        
        # Load Classical results
        classical_file = self.results_dir / "benchmark_results" / "classical_estimators_benchmark_summary.csv"
        if classical_file.exists():
            classical_df = pd.read_csv(classical_file)
            classical_df['Category'] = 'Classical'
            self.leaderboard_data['Classical'] = classical_df
            print(f"✅ Loaded Classical: {len(classical_df)} estimators")
        
        # Load ML results
        ml_file = self.results_dir / "ml_benchmark_results" / "ml_estimators_benchmark_summary.csv"
        if ml_file.exists():
            ml_df = pd.read_csv(ml_file)
            ml_df['Category'] = 'ML'
            self.leaderboard_data['ML'] = ml_df
            print(f"✅ Loaded ML: {len(ml_df)} estimators")
        
        # Load Neural Network results
        nn_file = self.results_dir / "neural_benchmark_results" / "neural_estimators_benchmark_summary.csv"
        if nn_file.exists():
            nn_df = pd.read_csv(nn_file)
            nn_df['Category'] = 'Neural Networks'
            self.leaderboard_data['Neural Networks'] = nn_df
            print(f"✅ Loaded Neural Networks: {len(nn_df)} estimators")
    
    def standardize_column_names(self, df, category):
        """Standardize column names across different benchmark results."""
        standardized_df = df.copy()
        
        # Map different column naming conventions to standard names
        column_mapping = {}
        
        for col in df.columns:
            if "Mean_Absolute_Error" in col and "Pure" in col:
                column_mapping[col] = "Pure_MAE"
            elif "Mean_Absolute_Error" in col and "Contaminated" in col:
                column_mapping[col] = "Contaminated_MAE"
            elif "Execution_Time" in col and "Pure" in col:
                column_mapping[col] = "Pure_Execution_Time"
            elif "Execution_Time" in col and "Contaminated" in col:
                column_mapping[col] = "Contaminated_Execution_Time"
            elif "Robustness_Score" in col:
                column_mapping[col] = "Robustness_Score"
            elif "Realistic_Success_Rate" in col:
                column_mapping[col] = "Realistic_Performance"
            elif "Overall_Score" in col:
                column_mapping[col] = "Overall_Score"
            elif col == "Estimator":
                column_mapping[col] = "Estimator"
        
        # Rename columns
        standardized_df = standardized_df.rename(columns=column_mapping)
        
        return standardized_df
    
    def calculate_composite_scores(self, df):
        """Calculate composite scores for ranking."""
        scores_df = df.copy()
        
        # Normalize MAE (lower is better) - convert to score
        if 'Pure_MAE' in scores_df.columns:
            max_mae = scores_df['Pure_MAE'].max()
            scores_df['MAE_Score'] = 10 * (1 - scores_df['Pure_MAE'] / max_mae)
        else:
            scores_df['MAE_Score'] = 0
        
        # Normalize execution time (lower is better) - convert to score
        if 'Pure_Execution_Time' in scores_df.columns:
            max_time = scores_df['Pure_Execution_Time'].max()
            scores_df['Speed_Score'] = 10 * (1 - scores_df['Pure_Execution_Time'] / max_time)
        else:
            scores_df['Speed_Score'] = 0
        
        # Robustness score (already 0-10)
        if 'Robustness_Score' in scores_df.columns:
            scores_df['Robustness_Score_Normalized'] = scores_df['Robustness_Score'].values
        else:
            scores_df['Robustness_Score_Normalized'] = 0
        
        # Realistic performance score
        if 'Realistic_Performance' in scores_df.columns:
            scores_df['Realistic_Score'] = scores_df['Realistic_Performance'].values
        else:
            scores_df['Realistic_Score'] = 0
        
        # Calculate composite scores
        # Overall composite (weighted average)
        weights = {
            'accuracy': 0.35,
            'speed': 0.20,
            'robustness': 0.25,
            'realistic': 0.20
        }
        
        scores_df['Composite_Score'] = (
            weights['accuracy'] * scores_df['MAE_Score'] +
            weights['speed'] * scores_df['Speed_Score'] +
            weights['robustness'] * scores_df['Robustness_Score_Normalized'] +
            weights['realistic'] * scores_df['Realistic_Score']
        )
        
        # Pure performance composite (accuracy + speed)
        scores_df['Pure_Performance_Score'] = 0.6 * scores_df['MAE_Score'] + 0.4 * scores_df['Speed_Score']
        
        # Robustness composite (robustness + realistic)
        scores_df['Robustness_Composite'] = 0.6 * scores_df['Robustness_Score_Normalized'] + 0.4 * scores_df['Realistic_Score']
        
        return scores_df
    
    def create_overall_leaderboard(self):
        """Create overall leaderboard across all estimators."""
        print("\n🏆 Creating Overall Leaderboard...")
        
        # Combine all data
        all_estimators = []
        for category, df in self.leaderboard_data.items():
            standardized_df = self.standardize_column_names(df, category)
            scored_df = self.calculate_composite_scores(standardized_df)
            # Ensure unique column names by adding category prefix if needed
            scored_df.columns = [f"{col}_{category}" if col in ['Robustness_Score', 'Overall_Score'] else col 
                               for col in scored_df.columns]
            all_estimators.append(scored_df)
        
        if not all_estimators:
            return pd.DataFrame()
        
        # Combine with proper handling of duplicate columns
        combined_df = pd.concat(all_estimators, ignore_index=True, sort=False)
        
        # Clean up duplicate columns by keeping the first occurrence
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        
        # Create overall rankings
        leaderboard = combined_df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
        leaderboard['Overall_Rank'] = range(1, len(leaderboard) + 1)
        
        # Create performance category rankings
        leaderboard = leaderboard.sort_values('Pure_Performance_Score', ascending=False).reset_index(drop=True)
        leaderboard['Performance_Rank'] = range(1, len(leaderboard) + 1)
        
        # Create robustness rankings
        leaderboard = leaderboard.sort_values('Robustness_Composite', ascending=False).reset_index(drop=True)
        leaderboard['Robustness_Rank'] = range(1, len(leaderboard) + 1)
        
        # Re-sort by overall rank
        leaderboard = leaderboard.sort_values('Overall_Rank').reset_index(drop=True)
        
        return leaderboard
    
    def create_category_leaderboards(self):
        """Create leaderboards for each category."""
        print("\n📊 Creating Category-Specific Leaderboards...")
        
        category_leaderboards = {}
        
        for category, df in self.leaderboard_data.items():
            standardized_df = self.standardize_column_names(df, category)
            scored_df = self.calculate_composite_scores(standardized_df)
            
            # Sort by composite score
            leaderboard = scored_df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
            leaderboard['Category_Rank'] = range(1, len(leaderboard) + 1)
            
            category_leaderboards[category] = leaderboard
        
        return category_leaderboards
    
    def create_contamination_robustness_leaderboard(self):
        """Create leaderboard based on contamination robustness."""
        print("\n🛡️ Creating Contamination Robustness Leaderboard...")
        
        # Combine all data
        all_estimators = []
        for category, df in self.leaderboard_data.items():
            standardized_df = self.standardize_column_names(df, category)
            scored_df = self.calculate_composite_scores(standardized_df)
            all_estimators.append(scored_df)
        
        if not all_estimators:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_estimators, ignore_index=True)
        
        # Focus on robustness metrics
        robustness_cols = ['Robustness_Score_Normalized', 'Realistic_Score', 'Robustness_Composite']
        available_cols = [col for col in robustness_cols if col in combined_df.columns]
        
        if not available_cols:
            return pd.DataFrame()
        
        # Create robustness leaderboard
        robustness_df = combined_df[['Estimator', 'Category'] + available_cols].copy()
        
        # Calculate contamination resistance score
        if 'Robustness_Score_Normalized' in available_cols and 'Realistic_Score' in available_cols:
            robustness_df['Contamination_Resistance'] = (
                0.6 * robustness_df['Robustness_Score_Normalized'] + 
                0.4 * robustness_df['Realistic_Score']
            )
        elif 'Robustness_Score_Normalized' in available_cols:
            robustness_df['Contamination_Resistance'] = robustness_df['Robustness_Score_Normalized']
        else:
            robustness_df['Contamination_Resistance'] = robustness_df[available_cols[0]]
        
        # Sort by contamination resistance
        robustness_leaderboard = robustness_df.sort_values('Contamination_Resistance', ascending=False).reset_index(drop=True)
        robustness_leaderboard['Robustness_Rank'] = range(1, len(robustness_leaderboard) + 1)
        
        return robustness_leaderboard
    
    def assess_dataset_quality(self):
        """Assess dataset quality across different contamination settings."""
        print("\n📈 Assessing Dataset Quality...")
        
        quality_assessment = {}
        
        # Define quality criteria
        quality_criteria = {
            'Pure_Data': {
                'description': 'Clean synthetic data with known Hurst parameters',
                'expected_performance': 'High accuracy, low variance',
                'quality_score': 10
            },
            'Contaminated_Data': {
                'description': 'Data with systematic contamination (trends, noise, artifacts)',
                'expected_performance': 'Moderate accuracy, higher variance',
                'quality_score': 6
            },
            'Realistic_Contexts': {
                'description': 'Real-world scenarios with complex contamination patterns',
                'expected_performance': 'Variable accuracy, high robustness requirement',
                'quality_score': 7
            }
        }
        
        # Analyze performance by dataset type
        for dataset_type, criteria in quality_criteria.items():
            quality_assessment[dataset_type] = {
                'criteria': criteria,
                'estimator_performance': {}
            }
            
            # Analyze each estimator's performance on this dataset type
            for category, df in self.leaderboard_data.items():
                standardized_df = self.standardize_column_names(df, category)
                
                for _, row in standardized_df.iterrows():
                    estimator = row['Estimator']
                    
                    if dataset_type == 'Pure_Data':
                        if 'Pure_MAE' in row:
                            quality_assessment[dataset_type]['estimator_performance'][estimator] = {
                                'mae': row['Pure_MAE'],
                                'execution_time': row.get('Pure_Execution_Time', np.nan),
                                'quality_score': criteria['quality_score']
                            }
                    elif dataset_type == 'Contaminated_Data':
                        if 'Contaminated_MAE' in row:
                            quality_assessment[dataset_type]['estimator_performance'][estimator] = {
                                'mae': row['Contaminated_MAE'],
                                'execution_time': row.get('Contaminated_Execution_Time', np.nan),
                                'quality_score': criteria['quality_score']
                            }
                    elif dataset_type == 'Realistic_Contexts':
                        if 'Realistic_Performance' in row:
                            quality_assessment[dataset_type]['estimator_performance'][estimator] = {
                                'success_rate': row['Realistic_Performance'],
                                'quality_score': criteria['quality_score']
                            }
        
        return quality_assessment
    
    def create_leaderboard_visualizations(self, overall_leaderboard, category_leaderboards, robustness_leaderboard):
        """Create comprehensive leaderboard visualizations."""
        print("\n📊 Creating Leaderboard Visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(24, 18))
        
        # 1. Overall Top 10 Leaderboard
        ax1 = plt.subplot(3, 4, 1)
        top_10 = overall_leaderboard.head(10)
        bars1 = ax1.barh(range(len(top_10)), top_10['Composite_Score'], 
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_10))))
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels([f"{row['Estimator']} ({row['Category']})" for _, row in top_10.iterrows()], fontsize=8)
        ax1.set_xlabel('Composite Score')
        ax1.set_title('Top 10 Overall Performers', fontweight='bold')
        ax1.invert_yaxis()
        
        # 2. Category Performance Comparison
        ax2 = plt.subplot(3, 4, 2)
        category_scores = []
        category_names = []
        for category, leaderboard in category_leaderboards.items():
            category_scores.append(leaderboard['Composite_Score'].mean())
            category_names.append(category)
        
        bars2 = ax2.bar(category_names, category_scores, alpha=0.7, color=['blue', 'orange', 'green'])
        ax2.set_title('Average Performance by Category', fontweight='bold')
        ax2.set_ylabel('Average Composite Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Robustness Top 10
        ax3 = plt.subplot(3, 4, 3)
        if not robustness_leaderboard.empty:
            top_robust = robustness_leaderboard.head(10)
            bars3 = ax3.barh(range(len(top_robust)), top_robust['Contamination_Resistance'], 
                            color=plt.cm.plasma(np.linspace(0, 1, len(top_robust))))
            ax3.set_yticks(range(len(top_robust)))
            ax3.set_yticklabels([f"{row['Estimator']} ({row['Category']})" for _, row in top_robust.iterrows()], fontsize=8)
            ax3.set_xlabel('Contamination Resistance')
            ax3.set_title('Top 10 Most Robust', fontweight='bold')
            ax3.invert_yaxis()
        
        # 4. Performance vs Robustness Scatter
        ax4 = plt.subplot(3, 4, 4)
        colors = {'Classical': 'blue', 'ML': 'orange', 'Neural Networks': 'green'}
        for category in colors.keys():
            category_data = overall_leaderboard[overall_leaderboard['Category'] == category]
            if not category_data.empty:
                ax4.scatter(category_data['Pure_Performance_Score'], 
                           category_data['Robustness_Composite'],
                           label=category, s=100, alpha=0.7, color=colors[category])
        
        ax4.set_xlabel('Performance Score')
        ax4.set_ylabel('Robustness Score')
        ax4.set_title('Performance vs Robustness', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5-8. Individual Category Leaderboards
        categories = list(category_leaderboards.keys())
        for i, category in enumerate(categories[:4]):
            ax = plt.subplot(3, 4, 5 + i)
            leaderboard = category_leaderboards[category]
            
            bars = ax.barh(range(len(leaderboard)), leaderboard['Composite_Score'], 
                          alpha=0.7, color=colors.get(category, 'gray'))
            ax.set_yticks(range(len(leaderboard)))
            ax.set_yticklabels(leaderboard['Estimator'], fontsize=8)
            ax.set_xlabel('Composite Score')
            ax.set_title(f'{category} Rankings', fontweight='bold')
            ax.invert_yaxis()
        
        # 9. Speed vs Accuracy Trade-off
        ax9 = plt.subplot(3, 4, 9)
        for category in colors.keys():
            category_data = overall_leaderboard[overall_leaderboard['Category'] == category]
            if not category_data.empty and 'Pure_MAE' in category_data.columns and 'Pure_Execution_Time' in category_data.columns:
                ax9.scatter(category_data['Pure_Execution_Time'], 
                           category_data['Pure_MAE'],
                           label=category, s=100, alpha=0.7, color=colors[category])
        
        ax9.set_xlabel('Execution Time (seconds)')
        ax9.set_ylabel('Mean Absolute Error')
        ax9.set_title('Speed vs Accuracy Trade-off', fontweight='bold')
        ax9.set_xscale('log')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. Score Distribution
        ax10 = plt.subplot(3, 4, 10)
        score_columns = ['MAE_Score', 'Speed_Score', 'Robustness_Score_Normalized', 'Realistic_Score']
        available_scores = [col for col in score_columns if col in overall_leaderboard.columns]
        
        if available_scores:
            score_data = overall_leaderboard[available_scores]
            ax10.boxplot([score_data[col].dropna() for col in available_scores], 
                        labels=[col.replace('_Score', '').replace('_Normalized', '') for col in available_scores])
            ax10.set_title('Score Distributions', fontweight='bold')
            ax10.set_ylabel('Score')
            ax10.tick_params(axis='x', rotation=45)
        
        # 11. Ranking Stability
        ax11 = plt.subplot(3, 4, 11)
        if 'Overall_Rank' in overall_leaderboard.columns and 'Performance_Rank' in overall_leaderboard.columns:
            rank_diff = overall_leaderboard['Overall_Rank'] - overall_leaderboard['Performance_Rank']
            ax11.scatter(overall_leaderboard['Overall_Rank'], rank_diff, alpha=0.7)
            ax11.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax11.set_xlabel('Overall Rank')
            ax11.set_ylabel('Rank Difference (Overall - Performance)')
            ax11.set_title('Ranking Stability', fontweight='bold')
            ax11.grid(True, alpha=0.3)
        
        # 12. Summary Statistics
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # Calculate summary statistics
        total_estimators = len(overall_leaderboard)
        categories_count = overall_leaderboard['Category'].value_counts()
        
        summary_text = f"""
        LEADERBOARD SUMMARY
        ===================
        
        Total Estimators: {total_estimators}
        
        By Category:
        """
        for category, count in categories_count.items():
            summary_text += f"• {category}: {count}\n"
        
        if not overall_leaderboard.empty:
            summary_text += f"""
        Best Overall: {overall_leaderboard.iloc[0]['Estimator']} ({overall_leaderboard.iloc[0]['Category']})
        Score: {overall_leaderboard.iloc[0]['Composite_Score']:.2f}/10
        
        Best Performance: {overall_leaderboard.loc[overall_leaderboard['Performance_Rank'].idxmin(), 'Estimator']} ({overall_leaderboard.loc[overall_leaderboard['Performance_Rank'].idxmin(), 'Category']})
        
        Most Robust: {overall_leaderboard.loc[overall_leaderboard['Robustness_Rank'].idxmin(), 'Estimator']} ({overall_leaderboard.loc[overall_leaderboard['Robustness_Rank'].idxmin(), 'Category']})
        """
        
        ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.results_dir / "comprehensive_leaderboard.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Leaderboard visualization saved to {plot_path}")
    
    def generate_leaderboard_reports(self, overall_leaderboard, category_leaderboards, robustness_leaderboard, quality_assessment):
        """Generate detailed leaderboard reports."""
        print("\n📋 Generating Leaderboard Reports...")
        
        # Create summary tables
        reports = {}
        
        # Overall leaderboard report
        reports['overall'] = {
            'top_10': overall_leaderboard.head(10)[['Estimator', 'Category', 'Composite_Score', 'Overall_Rank']].to_dict('records'),
            'summary_stats': {
                'total_estimators': len(overall_leaderboard),
                'categories': overall_leaderboard['Category'].value_counts().to_dict(),
                'best_overall': overall_leaderboard.iloc[0].to_dict() if not overall_leaderboard.empty else None
            }
        }
        
        # Category reports
        reports['categories'] = {}
        for category, leaderboard in category_leaderboards.items():
            reports['categories'][category] = {
                'top_5': leaderboard.head(5)[['Estimator', 'Composite_Score', 'Category_Rank']].to_dict('records'),
                'summary': {
                    'count': len(leaderboard),
                    'avg_score': leaderboard['Composite_Score'].mean(),
                    'best': leaderboard.iloc[0].to_dict() if not leaderboard.empty else None
                }
            }
        
        # Robustness report
        if not robustness_leaderboard.empty:
            reports['robustness'] = {
                'top_10': robustness_leaderboard.head(10)[['Estimator', 'Category', 'Contamination_Resistance', 'Robustness_Rank']].to_dict('records'),
                'summary': {
                    'total_estimators': len(robustness_leaderboard),
                    'best_robust': robustness_leaderboard.iloc[0].to_dict()
                }
            }
        
        # Quality assessment report
        reports['quality_assessment'] = quality_assessment
        
        # Save reports
        json_path = self.results_dir / "comprehensive_leaderboard_reports.json"
        with open(json_path, 'w') as f:
            json.dump(reports, f, indent=2, default=str)
        
        print(f"✅ Leaderboard reports saved to {json_path}")
        
        return reports
    
    def print_leaderboard_summary(self, overall_leaderboard, category_leaderboards, robustness_leaderboard):
        """Print comprehensive leaderboard summary."""
        print("\n" + "="*100)
        print("🏆 COMPREHENSIVE ESTIMATOR LEADERBOARD")
        print("="*100)
        
        # Overall Top 10
        print(f"\n🥇 TOP 10 OVERALL PERFORMERS:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Estimator':<20} {'Category':<15} {'Composite Score':<15} {'Performance':<12} {'Robustness':<10}")
        print("-" * 80)
        
        for _, row in overall_leaderboard.head(10).iterrows():
            print(f"{row['Overall_Rank']:<4} {row['Estimator']:<20} {row['Category']:<15} "
                  f"{row['Composite_Score']:<15.2f} {row.get('Pure_Performance_Score', 0):<12.2f} "
                  f"{row.get('Robustness_Composite', 0):<10.2f}")
        
        # Category Champions
        print(f"\n🏆 CATEGORY CHAMPIONS:")
        print("-" * 50)
        for category, leaderboard in category_leaderboards.items():
            if not leaderboard.empty:
                champ = leaderboard.iloc[0]
                print(f"{category:<15}: {champ['Estimator']:<20} (Score: {champ['Composite_Score']:.2f})")
        
        # Robustness Champions
        if not robustness_leaderboard.empty:
            print(f"\n🛡️ TOP 5 MOST ROBUST ESTIMATORS:")
            print("-" * 60)
            print(f"{'Rank':<4} {'Estimator':<20} {'Category':<15} {'Resistance Score':<15}")
            print("-" * 60)
            
            for _, row in robustness_leaderboard.head(5).iterrows():
                print(f"{row['Robustness_Rank']:<4} {row['Estimator']:<20} {row['Category']:<15} "
                      f"{row['Contamination_Resistance']:<15.2f}")
        
        # Key Insights
        print(f"\n📈 KEY INSIGHTS:")
        print("-" * 50)
        
        if not overall_leaderboard.empty:
            # Best in each metric
            best_accuracy = overall_leaderboard.loc[overall_leaderboard['MAE_Score'].idxmax()]
            best_speed = overall_leaderboard.loc[overall_leaderboard['Speed_Score'].idxmax()]
            best_robustness = overall_leaderboard.loc[overall_leaderboard['Robustness_Composite'].idxmax()]
            
            print(f"• Best Accuracy: {best_accuracy['Estimator']} ({best_accuracy['Category']})")
            print(f"• Fastest Execution: {best_speed['Estimator']} ({best_speed['Category']})")
            print(f"• Most Robust: {best_robustness['Estimator']} ({best_robustness['Category']})")
            
            # Category performance
            category_performance = {}
            for category in overall_leaderboard['Category'].unique():
                cat_data = overall_leaderboard[overall_leaderboard['Category'] == category]
                category_performance[category] = cat_data['Composite_Score'].mean()
            
            best_category = max(category_performance.items(), key=lambda x: x[1])
            print(f"• Best Performing Category: {best_category[0]} (Avg Score: {best_category[1]:.2f})")
        
        print("\n" + "="*100)
    
    def run_comprehensive_leaderboard(self):
        """Run the complete leaderboard analysis."""
        print("🏆 Starting Comprehensive Leaderboard Analysis")
        print("="*100)
        
        # Load data
        self.load_all_benchmark_data()
        
        if not self.leaderboard_data:
            print("❌ No benchmark data found. Please run benchmarks first.")
            return
        
        # Create leaderboards
        overall_leaderboard = self.create_overall_leaderboard()
        category_leaderboards = self.create_category_leaderboards()
        robustness_leaderboard = self.create_contamination_robustness_leaderboard()
        quality_assessment = self.assess_dataset_quality()
        
        # Create visualizations
        self.create_leaderboard_visualizations(overall_leaderboard, category_leaderboards, robustness_leaderboard)
        
        # Generate reports
        reports = self.generate_leaderboard_reports(overall_leaderboard, category_leaderboards, robustness_leaderboard, quality_assessment)
        
        # Print summary
        self.print_leaderboard_summary(overall_leaderboard, category_leaderboards, robustness_leaderboard)
        
        return {
            'overall_leaderboard': overall_leaderboard,
            'category_leaderboards': category_leaderboards,
            'robustness_leaderboard': robustness_leaderboard,
            'quality_assessment': quality_assessment,
            'reports': reports
        }

def main():
    """Run comprehensive leaderboard analysis."""
    leaderboard = ComprehensiveLeaderboard()
    results = leaderboard.run_comprehensive_leaderboard()
    return results

if __name__ == "__main__":
    results = main()
