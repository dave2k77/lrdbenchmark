#!/usr/bin/env python3
"""
Simple Comprehensive Leaderboard

A more robust approach to creating leaderboards across all estimator categories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

class SimpleLeaderboard:
    """Simple and robust leaderboard creation."""
    
    def __init__(self):
        self.results_dir = Path(".")
        
    def load_and_standardize_data(self):
        """Load and standardize data from all benchmarks."""
        print("📊 Loading and Standardizing Data...")
        
        all_data = []
        
        # Load Classical results
        classical_file = self.results_dir / "benchmark_results" / "classical_estimators_benchmark_summary.csv"
        if classical_file.exists():
            df = pd.read_csv(classical_file)
            df['Category'] = 'Classical'
            # Standardize column names
            df = df.rename(columns={
                'Mean_Absolute_Error_Pure': 'Pure_MAE',
                'Mean_Execution_Time_Pure': 'Pure_Execution_Time',
                'Mean_Absolute_Error_Contaminated': 'Contaminated_MAE'
            })
            all_data.append(df)
            print(f"✅ Loaded Classical: {len(df)} estimators")
        
        # Load ML results
        ml_file = self.results_dir / "ml_benchmark_results" / "ml_estimators_benchmark_summary.csv"
        if ml_file.exists():
            df = pd.read_csv(ml_file)
            df['Category'] = 'ML'
            # Standardize column names
            df = df.rename(columns={
                'Pure_Mean_Absolute_Error': 'Pure_MAE',
                'Pure_Mean_Execution_Time': 'Pure_Execution_Time',
                'Realistic_Success_Rate': 'Realistic_Performance'
            })
            all_data.append(df)
            print(f"✅ Loaded ML: {len(df)} estimators")
        
        # Load Neural Network results
        nn_file = self.results_dir / "neural_benchmark_results" / "neural_estimators_benchmark_summary.csv"
        if nn_file.exists():
            df = pd.read_csv(nn_file)
            df['Category'] = 'Neural Networks'
            # Standardize column names
            df = df.rename(columns={
                'Pure_Mean_Absolute_Error': 'Pure_MAE',
                'Pure_Mean_Execution_Time': 'Pure_Execution_Time',
                'Realistic_Success_Rate': 'Realistic_Performance'
            })
            all_data.append(df)
            print(f"✅ Loaded Neural Networks: {len(df)} estimators")
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True, sort=False)
        
        # Handle missing columns by filling with appropriate defaults
        required_columns = ['Pure_MAE', 'Pure_Execution_Time', 'Robustness_Score', 'Realistic_Performance', 'Overall_Score']
        for col in required_columns:
            if col not in combined_df.columns:
                combined_df[col] = 0.0
        
        # Fill NaN values
        combined_df = combined_df.fillna(0.0)
        
        return combined_df
    
    def calculate_scores(self, df):
        """Calculate normalized scores for ranking."""
        scores_df = df.copy()
        
        # Accuracy Score (lower MAE is better)
        if 'Pure_MAE' in scores_df.columns:
            max_mae = scores_df['Pure_MAE'].max()
            if max_mae > 0:
                scores_df['Accuracy_Score'] = 10 * (1 - scores_df['Pure_MAE'] / max_mae)
            else:
                scores_df['Accuracy_Score'] = 10.0
        else:
            scores_df['Accuracy_Score'] = 0.0
        
        # Speed Score (lower execution time is better)
        if 'Pure_Execution_Time' in scores_df.columns:
            max_time = scores_df['Pure_Execution_Time'].max()
            if max_time > 0:
                scores_df['Speed_Score'] = 10 * (1 - scores_df['Pure_Execution_Time'] / max_time)
            else:
                scores_df['Speed_Score'] = 10.0
        else:
            scores_df['Speed_Score'] = 0.0
        
        # Robustness Score (already normalized)
        if 'Robustness_Score' in scores_df.columns:
            scores_df['Robustness_Score_Norm'] = scores_df['Robustness_Score']
        else:
            scores_df['Robustness_Score_Norm'] = 0.0
        
        # Realistic Performance Score
        if 'Realistic_Performance' in scores_df.columns:
            scores_df['Realistic_Score'] = scores_df['Realistic_Performance']
        else:
            scores_df['Realistic_Score'] = 0.0
        
        # Overall Score (if available)
        if 'Overall_Score' in scores_df.columns:
            scores_df['Overall_Score_Norm'] = scores_df['Overall_Score']
        else:
            scores_df['Overall_Score_Norm'] = 0.0
        
        # Calculate composite scores
        # Performance Composite (Accuracy + Speed)
        scores_df['Performance_Composite'] = 0.6 * scores_df['Accuracy_Score'] + 0.4 * scores_df['Speed_Score']
        
        # Robustness Composite (Robustness + Realistic)
        scores_df['Robustness_Composite'] = 0.6 * scores_df['Robustness_Score_Norm'] + 0.4 * scores_df['Realistic_Score']
        
        # Final Composite Score (Performance + Robustness + Overall)
        scores_df['Final_Composite'] = (
            0.4 * scores_df['Performance_Composite'] +
            0.3 * scores_df['Robustness_Composite'] +
            0.3 * scores_df['Overall_Score_Norm']
        )
        
        return scores_df
    
    def create_rankings(self, df):
        """Create various rankings."""
        rankings_df = df.copy()
        
        # Overall ranking by final composite score
        rankings_df = rankings_df.sort_values('Final_Composite', ascending=False).reset_index(drop=True)
        rankings_df['Overall_Rank'] = range(1, len(rankings_df) + 1)
        
        # Performance ranking
        rankings_df = rankings_df.sort_values('Performance_Composite', ascending=False).reset_index(drop=True)
        rankings_df['Performance_Rank'] = range(1, len(rankings_df) + 1)
        
        # Robustness ranking
        rankings_df = rankings_df.sort_values('Robustness_Composite', ascending=False).reset_index(drop=True)
        rankings_df['Robustness_Rank'] = range(1, len(rankings_df) + 1)
        
        # Accuracy ranking
        rankings_df = rankings_df.sort_values('Accuracy_Score', ascending=False).reset_index(drop=True)
        rankings_df['Accuracy_Rank'] = range(1, len(rankings_df) + 1)
        
        # Speed ranking
        rankings_df = rankings_df.sort_values('Speed_Score', ascending=False).reset_index(drop=True)
        rankings_df['Speed_Rank'] = range(1, len(rankings_df) + 1)
        
        # Re-sort by overall rank
        rankings_df = rankings_df.sort_values('Overall_Rank').reset_index(drop=True)
        
        return rankings_df
    
    def create_leaderboard_visualization(self, df):
        """Create leaderboard visualization."""
        print("\n📊 Creating Leaderboard Visualization...")
        
        plt.style.use('default')
        sns.set_palette("Set2")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Estimator Leaderboard', fontsize=16, fontweight='bold')
        
        # 1. Top 15 Overall Performers
        ax1 = axes[0, 0]
        top_15 = df.head(15)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_15)))
        bars1 = ax1.barh(range(len(top_15)), top_15['Final_Composite'], color=colors)
        ax1.set_yticks(range(len(top_15)))
        ax1.set_yticklabels([f"{row['Estimator']} ({row['Category']})" for _, row in top_15.iterrows()], fontsize=8)
        ax1.set_xlabel('Final Composite Score')
        ax1.set_title('Top 15 Overall Performers')
        ax1.invert_yaxis()
        
        # 2. Performance by Category
        ax2 = axes[0, 1]
        category_avg = df.groupby('Category')['Final_Composite'].mean().sort_values(ascending=False)
        bars2 = ax2.bar(category_avg.index, category_avg.values, alpha=0.7, color=['blue', 'orange', 'green'])
        ax2.set_title('Average Performance by Category')
        ax2.set_ylabel('Average Final Composite Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Performance vs Robustness Scatter
        ax3 = axes[0, 2]
        categories = df['Category'].unique()
        colors_map = {'Classical': 'blue', 'ML': 'orange', 'Neural Networks': 'green'}
        for category in categories:
            cat_data = df[df['Category'] == category]
            ax3.scatter(cat_data['Performance_Composite'], cat_data['Robustness_Composite'],
                       label=category, s=100, alpha=0.7, color=colors_map.get(category, 'gray'))
        ax3.set_xlabel('Performance Composite')
        ax3.set_ylabel('Robustness Composite')
        ax3.set_title('Performance vs Robustness')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Accuracy vs Speed Trade-off
        ax4 = axes[1, 0]
        for category in categories:
            cat_data = df[df['Category'] == category]
            ax4.scatter(cat_data['Speed_Score'], cat_data['Accuracy_Score'],
                       label=category, s=100, alpha=0.7, color=colors_map.get(category, 'gray'))
        ax4.set_xlabel('Speed Score')
        ax4.set_ylabel('Accuracy Score')
        ax4.set_title('Accuracy vs Speed Trade-off')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Score Distribution
        ax5 = axes[1, 1]
        score_columns = ['Accuracy_Score', 'Speed_Score', 'Robustness_Score_Norm', 'Realistic_Score']
        score_data = [df[col].values for col in score_columns if col in df.columns]
        score_labels = [col.replace('_Score', '').replace('_Norm', '') for col in score_columns if col in df.columns]
        
        # Create boxplot with better formatting
        box_plot = ax5.boxplot(score_data, labels=score_labels, patch_artist=True)
        
        # Color the boxes
        colors_box = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(box_plot['boxes'], colors_box[:len(box_plot['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax5.set_title('Score Distributions by Metric', fontweight='bold')
        ax5.set_ylabel('Score (0-10 scale)')
        ax5.set_xlabel('Performance Metrics')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 10)
        
        # Add legend for score distribution
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=colors_box[i], alpha=0.7, label=score_labels[i])
            for i in range(len(score_labels))
        ]
        ax5.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        # 6. Ranking Stability
        ax6 = axes[1, 2]
        rank_diff = df['Overall_Rank'] - df['Performance_Rank']
        
        # Color points by category
        colors_map = {'Classical': 'blue', 'ML': 'orange', 'Neural Networks': 'green'}
        for category in categories:
            cat_data = df[df['Category'] == category]
            cat_rank_diff = cat_data['Overall_Rank'] - cat_data['Performance_Rank']
            ax6.scatter(cat_data['Overall_Rank'], cat_rank_diff, 
                       label=category, s=100, alpha=0.7, color=colors_map.get(category, 'gray'))
        
        # Add reference lines
        ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Perfect Stability')
        ax6.axhline(y=5, color='orange', linestyle=':', alpha=0.5, label='Moderate Instability')
        ax6.axhline(y=-5, color='orange', linestyle=':', alpha=0.5)
        ax6.axhline(y=10, color='red', linestyle=':', alpha=0.5, label='High Instability')
        ax6.axhline(y=-10, color='red', linestyle=':', alpha=0.5)
        
        ax6.set_xlabel('Overall Rank (1=Best)')
        ax6.set_ylabel('Rank Difference (Overall - Performance)')
        ax6.set_title('Ranking Stability Analysis', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='upper right', fontsize=8)
        
        # Add annotations for interpretation
        ax6.text(0.02, 0.98, 'Negative: Performance > Overall\nPositive: Overall > Performance\nZero: Perfect Stability', 
                transform=ax6.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.results_dir / "simple_leaderboard.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Leaderboard visualization saved to {plot_path}")
    
    def print_detailed_leaderboard(self, df):
        """Print detailed leaderboard."""
        print("\n" + "="*120)
        print("🏆 COMPREHENSIVE ESTIMATOR LEADERBOARD")
        print("="*120)
        
        # Top 20 Overall
        print(f"\n🥇 TOP 20 OVERALL PERFORMERS:")
        print("-" * 120)
        print(f"{'Rank':<4} {'Estimator':<20} {'Category':<15} {'Final Score':<12} {'Performance':<12} {'Robustness':<12} {'Accuracy':<10} {'Speed':<10}")
        print("-" * 120)
        
        for _, row in df.head(20).iterrows():
            print(f"{row['Overall_Rank']:<4} {row['Estimator']:<20} {row['Category']:<15} "
                  f"{row['Final_Composite']:<12.2f} {row['Performance_Composite']:<12.2f} "
                  f"{row['Robustness_Composite']:<12.2f} {row['Accuracy_Score']:<10.2f} {row['Speed_Score']:<10.2f}")
        
        # Category Champions
        print(f"\n🏆 CATEGORY CHAMPIONS:")
        print("-" * 60)
        for category in df['Category'].unique():
            champ = df[df['Category'] == category].iloc[0]
            print(f"{category:<15}: {champ['Estimator']:<20} (Final Score: {champ['Final_Composite']:.2f})")
        
        # Best in each metric
        print(f"\n🏅 BEST IN EACH METRIC:")
        print("-" * 60)
        
        best_accuracy = df.loc[df['Accuracy_Rank'].idxmin()]
        best_speed = df.loc[df['Speed_Rank'].idxmin()]
        best_performance = df.loc[df['Performance_Rank'].idxmin()]
        best_robustness = df.loc[df['Robustness_Rank'].idxmin()]
        
        print(f"Best Accuracy: {best_accuracy['Estimator']} ({best_accuracy['Category']}) - Score: {best_accuracy['Accuracy_Score']:.2f}")
        print(f"Fastest Speed: {best_speed['Estimator']} ({best_speed['Category']}) - Score: {best_speed['Speed_Score']:.2f}")
        print(f"Best Performance: {best_performance['Estimator']} ({best_performance['Category']}) - Score: {best_performance['Performance_Composite']:.2f}")
        print(f"Most Robust: {best_robustness['Estimator']} ({best_robustness['Category']}) - Score: {best_robustness['Robustness_Composite']:.2f}")
        
        # Summary Statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print("-" * 60)
        print(f"Total Estimators: {len(df)}")
        print(f"Categories: {', '.join(df['Category'].unique())}")
        
        for category in df['Category'].unique():
            cat_data = df[df['Category'] == category]
            print(f"{category}: {len(cat_data)} estimators, avg score: {cat_data['Final_Composite'].mean():.2f}")
        
        print("\n" + "="*120)
    
    def save_leaderboard_data(self, df):
        """Save leaderboard data to files."""
        print("\n💾 Saving Leaderboard Data...")
        
        # Save CSV
        csv_path = self.results_dir / "comprehensive_leaderboard.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Leaderboard CSV saved to {csv_path}")
        
        # Save JSON
        json_path = self.results_dir / "comprehensive_leaderboard.json"
        leaderboard_dict = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_estimators': len(df),
                'categories': df['Category'].value_counts().to_dict()
            },
            'top_10': df.head(10)[['Estimator', 'Category', 'Final_Composite', 'Overall_Rank']].to_dict('records'),
            'category_champions': {},
            'best_metrics': {
                'accuracy': df.loc[df['Accuracy_Rank'].idxmin()].to_dict(),
                'speed': df.loc[df['Speed_Rank'].idxmin()].to_dict(),
                'performance': df.loc[df['Performance_Rank'].idxmin()].to_dict(),
                'robustness': df.loc[df['Robustness_Rank'].idxmin()].to_dict()
            }
        }
        
        # Add category champions
        for category in df['Category'].unique():
            champ = df[df['Category'] == category].iloc[0]
            leaderboard_dict['category_champions'][category] = champ.to_dict()
        
        with open(json_path, 'w') as f:
            json.dump(leaderboard_dict, f, indent=2, default=str)
        
        print(f"✅ Leaderboard JSON saved to {json_path}")
    
    def run_leaderboard_analysis(self):
        """Run the complete leaderboard analysis."""
        print("🏆 Starting Comprehensive Leaderboard Analysis")
        print("="*100)
        
        # Load and standardize data
        df = self.load_and_standardize_data()
        
        if df.empty:
            print("❌ No data found. Please run benchmarks first.")
            return None
        
        # Calculate scores
        scored_df = self.calculate_scores(df)
        
        # Create rankings
        rankings_df = self.create_rankings(scored_df)
        
        # Create visualization
        self.create_leaderboard_visualization(rankings_df)
        
        # Print detailed leaderboard
        self.print_detailed_leaderboard(rankings_df)
        
        # Save data
        self.save_leaderboard_data(rankings_df)
        
        return rankings_df

def main():
    """Run the leaderboard analysis."""
    leaderboard = SimpleLeaderboard()
    results = leaderboard.run_leaderboard_analysis()
    return results

if __name__ == "__main__":
    results = main()
