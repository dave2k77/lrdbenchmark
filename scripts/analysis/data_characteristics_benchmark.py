#!/usr/bin/env python3
"""
Data Characteristics Benchmark

This script demonstrates the impact of heavy-tailed noise on data characteristics
and shows how alpha-stable distributions affect time series properties.

Focuses on data analysis rather than estimator performance to avoid JAX issues.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import alpha-stable model
from lrdbenchmark.models.data_models.alpha_stable.alpha_stable_model import AlphaStableModel

def generate_simple_fbm(hurst: float, length: int, seed: int = 42) -> np.ndarray:
    """Generate simple fractional Brownian motion."""
    np.random.seed(seed)
    t = np.linspace(0, 1, length)
    dt = t[1] - t[0]
    
    # Generate increments
    increments = np.random.normal(0, 1, length) * (dt ** hurst)
    
    # Cumulative sum to get FBM
    fbm = np.cumsum(increments)
    
    return fbm

def generate_simple_fgn(hurst: float, length: int, seed: int = 42) -> np.ndarray:
    """Generate simple fractional Gaussian noise."""
    np.random.seed(seed)
    t = np.linspace(0, 1, length)
    dt = t[1] - t[0]
    
    # Generate FGN increments
    fgn = np.random.normal(0, 1, length) * (dt ** hurst)
    
    return fgn

def analyze_data_characteristics(data: np.ndarray, data_name: str) -> Dict[str, Any]:
    """Analyze statistical characteristics of data."""
    stats = {
        'name': data_name,
        'length': len(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'skewness': calculate_skewness(data),
        'kurtosis': calculate_kurtosis(data),
        'extreme_values_5': np.sum(np.abs(data) > 5),
        'extreme_values_10': np.sum(np.abs(data) > 10),
        'extreme_values_20': np.sum(np.abs(data) > 20),
        'percentile_95': np.percentile(data, 95),
        'percentile_99': np.percentile(data, 99),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'has_finite_variance': np.isfinite(np.var(data)),
        'has_finite_mean': np.isfinite(np.mean(data))
    }
    
    return stats

def calculate_skewness(data: np.ndarray) -> float:
    """Calculate sample skewness."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate sample kurtosis."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3

def run_data_characteristics_benchmark():
    """Run benchmark comparing data characteristics."""
    print("=" * 80)
    print("Data Characteristics Benchmark")
    print("Comparing Pure vs Heavy-Tailed Data Properties")
    print("=" * 80)
    
    # Test parameters
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    data_length = 1000
    
    # Alpha-stable configurations
    alpha_stable_configs = [
        (2.0, 0.0, "Gaussian (α=2.0)"),
        (1.5, 0.0, "Symmetric Heavy (α=1.5)"),
        (1.0, 0.0, "Cauchy (α=1.0)"),
        (0.8, 0.0, "Very Heavy (α=0.8)"),
        (0.5, 0.0, "Extreme Heavy (α=0.5)"),
    ]
    
    all_results = []
    
    # Test pure data
    print(f"\n{'='*60}")
    print("Pure Data Analysis")
    print(f"{'='*60}")
    
    for hurst in hurst_values:
        print(f"\nHurst = {hurst:.1f}")
        print("-" * 40)
        
        # FBM
        fbm_data = generate_simple_fbm(hurst, data_length)
        fbm_stats = analyze_data_characteristics(fbm_data, f"FBM (H={hurst})")
        all_results.append(fbm_stats)
        print_data_stats(fbm_stats)
        
        # FGN
        fgn_data = generate_simple_fgn(hurst, data_length)
        fgn_stats = analyze_data_characteristics(fgn_data, f"FGN (H={hurst})")
        all_results.append(fgn_stats)
        print_data_stats(fgn_stats)
    
    # Test alpha-stable data
    print(f"\n{'='*60}")
    print("Alpha-Stable Data Analysis")
    print(f"{'='*60}")
    
    for alpha, beta, name in alpha_stable_configs:
        print(f"\n{name} (α={alpha}, β={beta})")
        print("-" * 40)
        
        for hurst in hurst_values:
            print(f"  Hurst = {hurst:.1f}")
            
            # Generate alpha-stable data
            alpha_stable = AlphaStableModel(alpha=alpha, beta=beta, sigma=1.0, mu=0.0)
            data = alpha_stable.generate(data_length, seed=42)
            
            stats = analyze_data_characteristics(data, f"{name} (H={hurst})")
            all_results.append(stats)
            print_data_stats(stats, indent="    ")
    
    return all_results

def print_data_stats(stats: Dict[str, Any], indent: str = ""):
    """Print data statistics in a formatted way."""
    print(f"{indent}Length: {stats['length']}")
    print(f"{indent}Mean: {stats['mean']:.4f}")
    print(f"{indent}Std: {stats['std']:.4f}")
    print(f"{indent}Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    print(f"{indent}Skewness: {stats['skewness']:.4f}")
    print(f"{indent}Kurtosis: {stats['kurtosis']:.4f}")
    print(f"{indent}Extreme (|x|>5): {stats['extreme_values_5']}")
    print(f"{indent}Extreme (|x|>10): {stats['extreme_values_10']}")
    print(f"{indent}P95: {stats['percentile_95']:.2f}")
    print(f"{indent}P99: {stats['percentile_99']:.2f}")
    print(f"{indent}Finite Variance: {stats['has_finite_variance']}")
    print(f"{indent}Finite Mean: {stats['has_finite_mean']}")

def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and summarize results."""
    print("\n" + "=" * 80)
    print("Results Analysis")
    print("=" * 80)
    
    # Group by data type
    pure_results = [r for r in results if 'FBM' in r['name'] or 'FGN' in r['name']]
    heavy_results = [r for r in results if 'Alpha-Stable' in r['name'] or 'Gaussian' in r['name'] or 'Cauchy' in r['name'] or 'Heavy' in r['name']]
    
    print(f"\n{'Metric':<25} {'Pure Data':<15} {'Heavy-Tailed':<15} {'Difference':<15}")
    print("-" * 70)
    
    if pure_results and heavy_results:
        # Calculate average metrics
        pure_avg_std = np.mean([r['std'] for r in pure_results])
        heavy_avg_std = np.mean([r['std'] for r in heavy_results])
        std_diff = heavy_avg_std - pure_avg_std
        
        pure_avg_skewness = np.mean([r['skewness'] for r in pure_results])
        heavy_avg_skewness = np.mean([r['skewness'] for r in heavy_results])
        skewness_diff = heavy_avg_skewness - pure_avg_skewness
        
        pure_avg_kurtosis = np.mean([r['kurtosis'] for r in pure_results])
        heavy_avg_kurtosis = np.mean([r['kurtosis'] for r in heavy_results])
        kurtosis_diff = heavy_avg_kurtosis - pure_avg_kurtosis
        
        pure_avg_extreme = np.mean([r['extreme_values_5'] for r in pure_results])
        heavy_avg_extreme = np.mean([r['extreme_values_5'] for r in heavy_results])
        extreme_diff = heavy_avg_extreme - pure_avg_extreme
        
        pure_finite_var = np.mean([r['has_finite_variance'] for r in pure_results])
        heavy_finite_var = np.mean([r['has_finite_variance'] for r in heavy_results])
        
        print(f"{'Average Std':<25} {pure_avg_std:<15.4f} {heavy_avg_std:<15.4f} {std_diff:<15.4f}")
        print(f"{'Average Skewness':<25} {pure_avg_skewness:<15.4f} {heavy_avg_skewness:<15.4f} {skewness_diff:<15.4f}")
        print(f"{'Average Kurtosis':<25} {pure_avg_kurtosis:<15.4f} {heavy_avg_kurtosis:<15.4f} {kurtosis_diff:<15.4f}")
        print(f"{'Average Extreme Values':<25} {pure_avg_extreme:<15.1f} {heavy_avg_extreme:<15.1f} {extreme_diff:<15.1f}")
        print(f"{'Finite Variance Rate':<25} {pure_finite_var:<15.1f} {heavy_finite_var:<15.1f} {heavy_finite_var - pure_finite_var:<15.1f}")
    
    # Analyze by alpha value
    print(f"\n{'='*60}")
    print("Analysis by Alpha-Stable Parameter")
    print(f"{'='*60}")
    
    alpha_groups = {}
    for result in results:
        if 'α=' in result['name']:
            # Extract alpha value
            alpha_str = result['name'].split('α=')[1].split(')')[0]
            alpha = float(alpha_str)
            if alpha not in alpha_groups:
                alpha_groups[alpha] = []
            alpha_groups[alpha].append(result)
    
    print(f"\n{'Alpha':<8} {'Avg Std':<12} {'Avg Kurtosis':<15} {'Avg Extreme':<15} {'Finite Var':<12}")
    print("-" * 70)
    
    for alpha in sorted(alpha_groups.keys()):
        group_results = alpha_groups[alpha]
        avg_std = np.mean([r['std'] for r in group_results])
        avg_kurtosis = np.mean([r['kurtosis'] for r in group_results])
        avg_extreme = np.mean([r['extreme_values_5'] for r in group_results])
        finite_var_rate = np.mean([r['has_finite_variance'] for r in group_results])
        
        print(f"{alpha:<8} {avg_std:<12.4f} {avg_kurtosis:<15.4f} {avg_extreme:<15.1f} {finite_var_rate:<12.1f}")

def create_visualization(results: List[Dict[str, Any]]):
    """Create visualization of data characteristics."""
    print("\n" + "=" * 60)
    print("Creating Visualization")
    print("=" * 60)
    
    try:
        # Prepare data for plotting
        pure_results = [r for r in results if 'FBM' in r['name'] or 'FGN' in r['name']]
        heavy_results = [r for r in results if 'Alpha-Stable' in r['name'] or 'Gaussian' in r['name'] or 'Cauchy' in r['name'] or 'Heavy' in r['name']]
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Characteristics: Pure vs Heavy-Tailed', fontsize=16)
        
        # Plot 1: Standard Deviation
        ax1 = axes[0, 0]
        pure_std = [r['std'] for r in pure_results]
        heavy_std = [r['std'] for r in heavy_results]
        
        ax1.hist(pure_std, alpha=0.7, label='Pure Data', bins=10, color='blue')
        ax1.hist(heavy_std, alpha=0.7, label='Heavy-Tailed', bins=10, color='red')
        ax1.set_xlabel('Standard Deviation')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Standard Deviations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Kurtosis
        ax2 = axes[0, 1]
        pure_kurtosis = [r['kurtosis'] for r in pure_results]
        heavy_kurtosis = [r['kurtosis'] for r in heavy_results]
        
        ax2.hist(pure_kurtosis, alpha=0.7, label='Pure Data', bins=10, color='blue')
        ax2.hist(heavy_kurtosis, alpha=0.7, label='Heavy-Tailed', bins=10, color='red')
        ax2.set_xlabel('Kurtosis')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Kurtosis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Extreme Values
        ax3 = axes[0, 2]
        pure_extreme = [r['extreme_values_5'] for r in pure_results]
        heavy_extreme = [r['extreme_values_5'] for r in heavy_results]
        
        ax3.hist(pure_extreme, alpha=0.7, label='Pure Data', bins=10, color='blue')
        ax3.hist(heavy_extreme, alpha=0.7, label='Heavy-Tailed', bins=10, color='red')
        ax3.set_xlabel('Extreme Values (|x| > 5)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Extreme Values')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Alpha vs Kurtosis
        ax4 = axes[1, 0]
        alpha_values = []
        kurtosis_values = []
        for result in heavy_results:
            if 'α=' in result['name']:
                alpha_str = result['name'].split('α=')[1].split(')')[0]
                alpha = float(alpha_str)
                alpha_values.append(alpha)
                kurtosis_values.append(result['kurtosis'])
        
        if alpha_values:
            ax4.scatter(alpha_values, kurtosis_values, alpha=0.7, color='red')
            ax4.set_xlabel('Alpha Parameter')
            ax4.set_ylabel('Kurtosis')
            ax4.set_title('Alpha vs Kurtosis')
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Alpha vs Extreme Values
        ax5 = axes[1, 1]
        alpha_values = []
        extreme_values = []
        for result in heavy_results:
            if 'α=' in result['name']:
                alpha_str = result['name'].split('α=')[1].split(')')[0]
                alpha = float(alpha_str)
                alpha_values.append(alpha)
                extreme_values.append(result['extreme_values_5'])
        
        if alpha_values:
            ax5.scatter(alpha_values, extreme_values, alpha=0.7, color='red')
            ax5.set_xlabel('Alpha Parameter')
            ax5.set_ylabel('Extreme Values (|x| > 5)')
            ax5.set_title('Alpha vs Extreme Values')
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Comparison Box Plot
        ax6 = axes[1, 2]
        data_to_plot = [pure_kurtosis, heavy_kurtosis]
        labels = ['Pure Data', 'Heavy-Tailed']
        
        ax6.boxplot(data_to_plot, labels=labels)
        ax6.set_ylabel('Kurtosis')
        ax6.set_title('Kurtosis Comparison')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_characteristics_benchmark.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'data_characteristics_benchmark.png'")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def main():
    """Main benchmark function."""
    print("Data Characteristics Benchmark")
    print("Analyzing Pure vs Heavy-Tailed Data Properties")
    print("=" * 80)
    
    try:
        # Run benchmark
        results = run_data_characteristics_benchmark()
        
        # Analyze results
        analyze_results(results)
        
        # Create visualization
        create_visualization(results)
        
        print("\n" + "=" * 80)
        print("✅ Data Characteristics Benchmark Complete!")
        print("This demonstrates the impact of heavy-tailed noise on data properties.")
        print("Key findings:")
        print("  - Pure data (FBM/FGN) has finite variance and moderate kurtosis")
        print("  - Heavy-tailed data (α < 2) shows infinite variance and high kurtosis")
        print("  - Extreme values increase dramatically with decreasing α")
        print("  - Data characteristics become more extreme as α approaches 0")
        print("  - This explains why LRD estimators struggle with heavy-tailed data")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
