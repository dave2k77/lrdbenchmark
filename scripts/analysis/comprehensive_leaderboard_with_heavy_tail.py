#!/usr/bin/env python3
"""
Comprehensive Leaderboard with Heavy-Tail Performance

This script creates an updated leaderboard that incorporates heavy-tail performance
into the overall scoring system, providing a more complete picture of estimator performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

def create_comprehensive_leaderboard_with_heavy_tail():
    """Create comprehensive leaderboard incorporating heavy-tail performance."""
    
    # Standard benchmark results (from previous analysis)
    standard_results = {
        'LSTM': {'mae': 0.097, 'time': 0.0012, 'robustness': 1.00, 'category': 'Neural Network'},
        'CNN': {'mae': 0.103, 'time': 0.0064, 'robustness': 1.00, 'category': 'Neural Network'},
        'Transformer': {'mae': 0.106, 'time': 0.0026, 'robustness': 1.00, 'category': 'Neural Network'},
        'GRU': {'mae': 0.108, 'time': 0.0007, 'robustness': 1.00, 'category': 'Neural Network'},
        'R/S': {'mae': 0.099, 'time': 0.348, 'robustness': 1.00, 'category': 'Classical'},
        'GradientBoosting': {'mae': 0.193, 'time': 0.013, 'robustness': 1.00, 'category': 'ML'},
        'SVR': {'mae': 0.202, 'time': 0.009, 'robustness': 1.00, 'category': 'ML'},
        'Whittle': {'mae': 0.200, 'time': 0.0002, 'robustness': 1.00, 'category': 'Classical'},
        'Periodogram': {'mae': 0.205, 'time': 0.0005, 'robustness': 1.00, 'category': 'Classical'},
        'CWT': {'mae': 0.269, 'time': 0.063, 'robustness': 1.00, 'category': 'Classical'},
        'GPH': {'mae': 0.274, 'time': 0.032, 'robustness': 1.00, 'category': 'Classical'},
        'RandomForest': {'mae': 0.202, 'time': 2.099, 'robustness': 1.00, 'category': 'ML'},
        'DFA': {'mae': 0.465, 'time': 0.009, 'robustness': 1.00, 'category': 'Classical'},
        'Higuchi': {'mae': 0.509, 'time': 0.004, 'robustness': 1.00, 'category': 'Classical'},
        'DMA': {'mae': 0.527, 'time': 0.0005, 'robustness': 1.00, 'category': 'Classical'}
    }
    
    # Heavy-tail performance results (from comprehensive analysis)
    heavy_tail_results = {
        'GradientBoosting': {'mae': 0.201, 'category': 'ML'},
        'RandomForest': {'mae': 0.211, 'category': 'ML'},
        'SVR': {'mae': 0.308, 'category': 'ML'},
        'LSTM': {'mae': 0.245, 'category': 'Neural Network'},
        'GRU': {'mae': 0.247, 'category': 'Neural Network'},
        'Transformer': {'mae': 0.249, 'category': 'Neural Network'},
        'CNN': {'mae': 0.300, 'category': 'Neural Network'},
        'DFA': {'mae': 0.346, 'category': 'Classical'},
        'DMA': {'mae': 0.346, 'category': 'Classical'},
        'R/S': {'mae': 0.409, 'category': 'Classical'},
        'Higuchi': {'mae': 0.539, 'category': 'Classical'}
    }
    
    # Create comprehensive scoring system
    def calculate_comprehensive_score(estimator_name: str, standard_data: Dict, heavy_tail_data: Dict = None) -> Dict[str, float]:
        """Calculate comprehensive score incorporating both standard and heavy-tail performance."""
        
        # Standard performance metrics
        mae_standard = standard_data['mae']
        time_standard = standard_data['time']
        robustness_standard = standard_data['robustness']
        
        # Heavy-tail performance (if available)
        if heavy_tail_data and estimator_name in heavy_tail_data:
            mae_heavy_tail = heavy_tail_data[estimator_name]['mae']
            # Weighted average: 60% standard, 40% heavy-tail
            mae_combined = 0.6 * mae_standard + 0.4 * mae_heavy_tail
            heavy_tail_available = True
        else:
            mae_heavy_tail = None
            mae_combined = mae_standard
            heavy_tail_available = False
        
        # Normalize metrics (lower is better for MAE and time)
        mae_score = max(0, 10 - (mae_combined * 20))  # Scale MAE to 0-10
        time_score = max(0, 10 - (time_standard * 10))  # Scale time to 0-10
        robustness_score = robustness_standard * 10  # Scale robustness to 0-10
        
        # Heavy-tail bonus (if available)
        heavy_tail_bonus = 1.0 if heavy_tail_available else 0.0
        
        # Comprehensive score calculation
        # Weighted: 40% accuracy, 20% speed, 20% robustness, 20% heavy-tail capability
        comprehensive_score = (
            0.4 * mae_score +
            0.2 * time_score +
            0.2 * robustness_score +
            0.2 * heavy_tail_bonus * 10
        )
        
        return {
            'mae_standard': mae_standard,
            'mae_heavy_tail': mae_heavy_tail,
            'mae_combined': mae_combined,
            'time': time_standard,
            'robustness': robustness_standard,
            'mae_score': mae_score,
            'time_score': time_score,
            'robustness_score': robustness_score,
            'heavy_tail_bonus': heavy_tail_bonus,
            'comprehensive_score': comprehensive_score,
            'heavy_tail_available': heavy_tail_available
        }
    
    # Calculate comprehensive scores for all estimators
    comprehensive_scores = {}
    for estimator_name, standard_data in standard_results.items():
        scores = calculate_comprehensive_score(estimator_name, standard_data, heavy_tail_results)
        comprehensive_scores[estimator_name] = {
            **scores,
            'category': standard_data['category']
        }
    
    # Create DataFrame for analysis
    df_data = []
    for estimator, scores in comprehensive_scores.items():
        df_data.append({
            'Estimator': estimator,
            'Category': scores['category'],
            'MAE_Standard': scores['mae_standard'],
            'MAE_HeavyTail': scores['mae_heavy_tail'],
            'MAE_Combined': scores['mae_combined'],
            'Execution_Time': scores['time'],
            'Robustness': scores['robustness'],
            'MAE_Score': scores['mae_score'],
            'Time_Score': scores['time_score'],
            'Robustness_Score': scores['robustness_score'],
            'HeavyTail_Bonus': scores['heavy_tail_bonus'],
            'Comprehensive_Score': scores['comprehensive_score'],
            'HeavyTail_Available': scores['heavy_tail_available']
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('Comprehensive_Score', ascending=False).reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)
    
    return df, comprehensive_scores

def create_comprehensive_visualization(df: pd.DataFrame, comprehensive_scores: Dict):
    """Create comprehensive visualization of the updated leaderboard."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Comprehensive Leaderboard
    ax1 = fig.add_subplot(gs[0, :])
    top_10 = df.head(10)
    
    colors = ['gold' if i < 3 else 'silver' if i < 6 else 'lightcoral' for i in range(len(top_10))]
    bars = ax1.barh(range(len(top_10)), top_10['Comprehensive_Score'], color=colors, alpha=0.8)
    
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels([f"{row['Rank']}. {row['Estimator']} ({row['Category']})" 
                        for _, row in top_10.iterrows()], fontsize=10)
    ax1.set_xlabel('Comprehensive Score', fontsize=12, fontweight='bold')
    ax1.set_title('Comprehensive Leaderboard: Standard + Heavy-Tail Performance', 
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, top_10['Comprehensive_Score'])):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', va='center', fontweight='bold')
    
    # 2. Performance by Category
    ax2 = fig.add_subplot(gs[1, 0])
    category_scores = df.groupby('Category')['Comprehensive_Score'].agg(['mean', 'std', 'count']).reset_index()
    
    bars2 = ax2.bar(category_scores['Category'], category_scores['mean'], 
                   yerr=category_scores['std'], capsize=5, alpha=0.7,
                   color=['skyblue', 'lightcoral', 'lightgreen'])
    ax2.set_title('Average Comprehensive Score by Category', fontweight='bold')
    ax2.set_ylabel('Average Comprehensive Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, mean_score in zip(bars2, category_scores['mean']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mean_score:.2f}', ha='center', fontweight='bold')
    
    # 3. MAE Comparison: Standard vs Heavy-Tail
    ax3 = fig.add_subplot(gs[1, 1])
    heavy_tail_estimators = df[df['HeavyTail_Available'] == True]
    
    if len(heavy_tail_estimators) > 0:
        x = np.arange(len(heavy_tail_estimators))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, heavy_tail_estimators['MAE_Standard'], width, 
                       label='Standard Data', alpha=0.8, color='skyblue')
        bars2 = ax3.bar(x + width/2, heavy_tail_estimators['MAE_HeavyTail'], width, 
                       label='Heavy-Tail Data', alpha=0.8, color='lightcoral')
        
        ax3.set_xlabel('Estimators')
        ax3.set_ylabel('Mean Absolute Error')
        ax3.set_title('MAE: Standard vs Heavy-Tail Data', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(heavy_tail_estimators['Estimator'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Score Components Breakdown
    ax4 = fig.add_subplot(gs[1, 2])
    top_5 = df.head(5)
    
    components = ['MAE_Score', 'Time_Score', 'Robustness_Score', 'HeavyTail_Bonus']
    component_labels = ['Accuracy', 'Speed', 'Robustness', 'Heavy-Tail Capability']
    
    x = np.arange(len(top_5))
    width = 0.2
    
    for i, (component, label) in enumerate(zip(components, component_labels)):
        offset = (i - 1.5) * width
        ax4.bar(x + offset, top_5[component], width, label=label, alpha=0.8)
    
    ax4.set_xlabel('Top 5 Estimators')
    ax4.set_ylabel('Score Component')
    ax4.set_title('Score Components Breakdown', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_5['Estimator'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Heavy-Tail Impact Analysis
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Calculate impact of heavy-tail inclusion
    heavy_tail_impact = []
    for _, row in df.iterrows():
        if row['HeavyTail_Available']:
            # Calculate score without heavy-tail bonus
            score_without_ht = (row['MAE_Score'] * 0.5 + row['Time_Score'] * 0.2 + 
                              row['Robustness_Score'] * 0.3)
            impact = row['Comprehensive_Score'] - score_without_ht
            heavy_tail_impact.append(impact)
        else:
            heavy_tail_impact.append(0)
    
    df['HeavyTail_Impact'] = heavy_tail_impact
    heavy_tail_estimators = df[df['HeavyTail_Available'] == True].sort_values('HeavyTail_Impact', ascending=True)
    
    if len(heavy_tail_estimators) > 0:
        bars = ax5.barh(range(len(heavy_tail_estimators)), heavy_tail_estimators['HeavyTail_Impact'], 
                       color='orange', alpha=0.7)
        ax5.set_yticks(range(len(heavy_tail_estimators)))
        ax5.set_yticklabels(heavy_tail_estimators['Estimator'])
        ax5.set_xlabel('Score Impact from Heavy-Tail Capability')
        ax5.set_title('Heavy-Tail Capability Impact on Overall Score', fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)
    
    # 6. Performance vs Heavy-Tail Capability
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Create scatter plot
    colors = {'Neural Network': 'blue', 'ML': 'red', 'Classical': 'green'}
    for category in df['Category'].unique():
        category_data = df[df['Category'] == category]
        ax6.scatter(category_data['HeavyTail_Bonus'], category_data['Comprehensive_Score'], 
                   c=colors[category], label=category, s=100, alpha=0.7)
    
    ax6.set_xlabel('Heavy-Tail Capability Bonus')
    ax6.set_ylabel('Comprehensive Score')
    ax6.set_title('Performance vs Heavy-Tail Capability', fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Category Performance Summary
    ax7 = fig.add_subplot(gs[2, 2])
    
    category_summary = df.groupby('Category').agg({
        'Comprehensive_Score': ['mean', 'std', 'count'],
        'HeavyTail_Available': 'sum'
    }).round(2)
    
    category_summary.columns = ['Mean_Score', 'Std_Score', 'Count', 'HeavyTail_Count']
    category_summary['HeavyTail_Ratio'] = category_summary['HeavyTail_Count'] / category_summary['Count']
    
    # Create table
    ax7.axis('tight')
    ax7.axis('off')
    
    table_data = []
    for category in category_summary.index:
        table_data.append([
            category,
            f"{category_summary.loc[category, 'Mean_Score']:.2f}",
            f"{category_summary.loc[category, 'Std_Score']:.2f}",
            f"{category_summary.loc[category, 'HeavyTail_Ratio']:.1%}"
        ])
    
    table = ax7.table(cellText=table_data,
                     colLabels=['Category', 'Mean Score', 'Std Score', 'Heavy-Tail Ratio'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax7.set_title('Category Performance Summary', fontweight='bold', pad=20)
    
    plt.suptitle('Comprehensive Leaderboard: Incorporating Heavy-Tail Performance', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('comprehensive_leaderboard_with_heavy_tail.png', dpi=300, bbox_inches='tight')
    print("📊 Comprehensive leaderboard visualization saved as 'comprehensive_leaderboard_with_heavy_tail.png'")
    
    return fig

def analyze_heavy_tail_impact(df: pd.DataFrame):
    """Analyze the impact of heavy-tail performance on overall scoring."""
    
    print("\n🔍 Heavy-Tail Impact Analysis")
    print("=" * 60)
    
    # Overall impact
    heavy_tail_estimators = df[df['HeavyTail_Available'] == True]
    no_heavy_tail_estimators = df[df['HeavyTail_Available'] == False]
    
    print(f"\n📊 Estimators with Heavy-Tail Data: {len(heavy_tail_estimators)}")
    print(f"📊 Estimators without Heavy-Tail Data: {len(no_heavy_tail_estimators)}")
    
    # Average scores
    avg_score_with_ht = heavy_tail_estimators['Comprehensive_Score'].mean()
    avg_score_without_ht = no_heavy_tail_estimators['Comprehensive_Score'].mean()
    
    print(f"\n📈 Average Comprehensive Score:")
    print(f"   With Heavy-Tail Data: {avg_score_with_ht:.2f}")
    print(f"   Without Heavy-Tail Data: {avg_score_without_ht:.2f}")
    print(f"   Difference: {avg_score_with_ht - avg_score_without_ht:.2f}")
    
    # Top performers analysis
    print(f"\n🏆 Top 5 Performers (with Heavy-Tail Capability):")
    top_5_with_ht = heavy_tail_estimators.head(5)
    for _, row in top_5_with_ht.iterrows():
        print(f"   {row['Rank']}. {row['Estimator']} ({row['Category']}): {row['Comprehensive_Score']:.2f}")
    
    # Category analysis
    print(f"\n📊 Category Performance (with Heavy-Tail Capability):")
    category_performance = heavy_tail_estimators.groupby('Category')['Comprehensive_Score'].agg(['mean', 'std', 'count'])
    for category in category_performance.index:
        mean_score = category_performance.loc[category, 'mean']
        std_score = category_performance.loc[category, 'std']
        count = category_performance.loc[category, 'count']
        print(f"   {category}: {mean_score:.2f} ± {std_score:.2f} (n={count})")
    
    # Heavy-tail impact on rankings
    print(f"\n📈 Heavy-Tail Impact on Rankings:")
    heavy_tail_impact = heavy_tail_estimators['HeavyTail_Impact'].sort_values(ascending=False)
    for estimator, impact in heavy_tail_impact.head(5).items():
        print(f"   {estimator}: +{impact:.2f} points from heavy-tail capability")
    
    return heavy_tail_estimators, no_heavy_tail_estimators

def main():
    """Main function to create comprehensive leaderboard with heavy-tail performance."""
    
    print("🚀 Creating Comprehensive Leaderboard with Heavy-Tail Performance")
    print("=" * 70)
    
    # Create comprehensive leaderboard
    df, comprehensive_scores = create_comprehensive_leaderboard_with_heavy_tail()
    
    # Display results
    print(f"\n📊 Comprehensive Leaderboard (Top 15)")
    print("=" * 80)
    print(f"{'Rank':<4} {'Estimator':<15} {'Category':<12} {'Score':<8} {'MAE_Combined':<12} {'Heavy-Tail':<10}")
    print("-" * 80)
    
    for _, row in df.head(15).iterrows():
        heavy_tail_status = "✅" if row['HeavyTail_Available'] else "❌"
        print(f"{row['Rank']:<4} {row['Estimator']:<15} {row['Category']:<12} "
              f"{row['Comprehensive_Score']:<8.2f} {row['MAE_Combined']:<12.3f} {heavy_tail_status:<10}")
    
    # Create visualization
    print(f"\n📈 Creating comprehensive visualization...")
    fig = create_comprehensive_visualization(df, comprehensive_scores)
    
    # Analyze heavy-tail impact
    heavy_tail_estimators, no_heavy_tail_estimators = analyze_heavy_tail_impact(df)
    
    # Save detailed results
    df.to_csv('comprehensive_leaderboard_with_heavy_tail.csv', index=False)
    print(f"\n💾 Detailed results saved as 'comprehensive_leaderboard_with_heavy_tail.csv'")
    
    print(f"\n✅ Comprehensive leaderboard with heavy-tail performance completed!")
    print(f"   Total estimators: {len(df)}")
    print(f"   With heavy-tail data: {len(heavy_tail_estimators)}")
    print(f"   Without heavy-tail data: {len(no_heavy_tail_estimators)}")
    
    return df, comprehensive_scores

if __name__ == "__main__":
    df, scores = main()
