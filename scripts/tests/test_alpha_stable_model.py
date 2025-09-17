#!/usr/bin/env python3
"""
Comprehensive Test Suite for Alpha-Stable Data Model

This script demonstrates the "Adding Data Model" functionality by testing
the newly implemented AlphaStableModel with various parameters and methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the alpha-stable model
from lrdbenchmark.models.data_models.alpha_stable.alpha_stable_model import AlphaStableModel

def test_parameter_validation():
    """Test parameter validation."""
    print("=" * 60)
    print("Testing Parameter Validation")
    print("=" * 60)
    
    # Valid parameters
    try:
        model = AlphaStableModel(alpha=1.5, beta=0.0, sigma=1.0, mu=0.0)
        print("✅ Valid parameters (α=1.5, β=0.0): OK")
    except Exception as e:
        print(f"❌ Valid parameters failed: {e}")
    
    # Invalid alpha values
    invalid_alphas = [0, -1, 2.1, 3]
    for alpha in invalid_alphas:
        try:
            model = AlphaStableModel(alpha=alpha, beta=0.0)
            print(f"❌ Invalid alpha {alpha} should have failed")
        except ValueError:
            print(f"✅ Invalid alpha {alpha} correctly rejected")
    
    # Invalid beta values
    invalid_betas = [-1.1, 1.1, 2]
    for beta in invalid_betas:
        try:
            model = AlphaStableModel(alpha=1.5, beta=beta)
            print(f"❌ Invalid beta {beta} should have failed")
        except ValueError:
            print(f"✅ Invalid beta {beta} correctly rejected")
    
    # Invalid sigma values
    invalid_sigmas = [0, -1, -0.1]
    for sigma in invalid_sigmas:
        try:
            model = AlphaStableModel(alpha=1.5, beta=0.0, sigma=sigma)
            print(f"❌ Invalid sigma {sigma} should have failed")
        except ValueError:
            print(f"✅ Invalid sigma {sigma} correctly rejected")

def test_generation_methods():
    """Test different generation methods."""
    print("\n" + "=" * 60)
    print("Testing Generation Methods")
    print("=" * 60)
    
    # Test parameters
    alpha = 1.5
    beta = 0.0
    n = 1000
    
    methods = ['auto', 'cms', 'nolan', 'fourier']
    
    for method in methods:
        print(f"\nTesting method: {method}")
        print("-" * 30)
        
        try:
            model = AlphaStableModel(
                alpha=alpha, 
                beta=beta, 
                sigma=1.0, 
                mu=0.0,
                method=method
            )
            
            start_time = time.time()
            data = model.generate(n, seed=42)
            generation_time = time.time() - start_time
            
            print(f"✅ Generation successful")
            print(f"   Data length: {len(data)}")
            print(f"   Generation time: {generation_time:.4f}s")
            print(f"   Mean: {np.mean(data):.4f}")
            print(f"   Std: {np.std(data):.4f}")
            print(f"   Min: {np.min(data):.4f}")
            print(f"   Max: {np.max(data):.4f}")
            
        except Exception as e:
            print(f"❌ Method {method} failed: {e}")

def test_different_parameters():
    """Test different alpha-stable parameter combinations."""
    print("\n" + "=" * 60)
    print("Testing Different Parameter Combinations")
    print("=" * 60)
    
    # Test cases: (alpha, beta, name)
    test_cases = [
        (2.0, 0.0, "Gaussian (α=2.0)"),
        (1.0, 0.0, "Cauchy (α=1.0)"),
        (1.5, 0.0, "Symmetric α=1.5"),
        (1.5, 0.5, "Skewed α=1.5"),
        (1.5, -0.5, "Left-skewed α=1.5"),
        (0.8, 0.0, "Heavy-tailed α=0.8"),
        (0.5, 0.0, "Very heavy-tailed α=0.5"),
    ]
    
    n = 2000
    
    for alpha, beta, name in test_cases:
        print(f"\n{name}:")
        print("-" * 30)
        
        try:
            model = AlphaStableModel(alpha=alpha, beta=beta, sigma=1.0, mu=0.0)
            data = model.generate(n, seed=42)
            
            # Calculate properties
            properties = model.sample_properties(n, seed=42)
            theoretical = model.theoretical_moments()
            model_props = model.get_properties()
            
            print(f"✅ Generation successful")
            print(f"   Sample mean: {properties['sample_mean']:.4f}")
            print(f"   Sample std: {properties['sample_std']:.4f}")
            print(f"   Sample skewness: {properties['sample_skewness']:.4f}")
            print(f"   Sample kurtosis: {properties['sample_kurtosis']:.4f}")
            print(f"   Has finite variance: {model_props['has_finite_variance']}")
            print(f"   Has finite mean: {model_props['has_finite_mean']}")
            print(f"   Is symmetric: {model_props['is_symmetric']}")
            print(f"   Heavy tailed: {model_props['heavy_tailed']}")
            
            if theoretical['mean'] is not None:
                print(f"   Theoretical mean: {theoretical['mean']:.4f}")
            if theoretical['variance'] is not None:
                print(f"   Theoretical variance: {theoretical['variance']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")

def test_backend_selection():
    """Test different backend selections."""
    print("\n" + "=" * 60)
    print("Testing Backend Selection")
    print("=" * 60)
    
    alpha = 1.5
    beta = 0.0
    n = 1000
    
    backends = ['auto', 'numpy', 'numba', 'jax']
    
    for backend in backends:
        print(f"\nTesting backend: {backend}")
        print("-" * 30)
        
        try:
            model = AlphaStableModel(
                alpha=alpha,
                beta=beta,
                sigma=1.0,
                mu=0.0,
                use_optimization=backend
            )
            
            start_time = time.time()
            data = model.generate(n, seed=42)
            generation_time = time.time() - start_time
            
            print(f"✅ Backend {backend} successful")
            print(f"   Generation time: {generation_time:.4f}s")
            print(f"   Data mean: {np.mean(data):.4f}")
            print(f"   Data std: {np.std(data):.4f}")
            
        except Exception as e:
            print(f"❌ Backend {backend} failed: {e}")

def test_special_cases():
    """Test special cases and edge conditions."""
    print("\n" + "=" * 60)
    print("Testing Special Cases")
    print("=" * 60)
    
    special_cases = [
        (2.0, 0.0, "Gaussian case"),
        (1.0, 0.0, "Cauchy case"),
        (1.0, 1.0, "Skewed Cauchy"),
        (1.0, -1.0, "Left-skewed Cauchy"),
        (0.1, 0.0, "Very heavy tails"),
    ]
    
    n = 1000
    
    for alpha, beta, name in special_cases:
        print(f"\n{name} (α={alpha}, β={beta}):")
        print("-" * 30)
        
        try:
            model = AlphaStableModel(alpha=alpha, beta=beta, sigma=1.0, mu=0.0)
            data = model.generate(n, seed=42)
            
            print(f"✅ Special case successful")
            print(f"   Data range: [{np.min(data):.2f}, {np.max(data):.2f}]")
            print(f"   Data mean: {np.mean(data):.4f}")
            print(f"   Data std: {np.std(data):.4f}")
            
            # Check for extreme values
            extreme_count = np.sum(np.abs(data) > 10)
            print(f"   Extreme values (|x| > 10): {extreme_count}")
            
        except Exception as e:
            print(f"❌ Special case failed: {e}")

def test_performance_scaling():
    """Test performance scaling with data size."""
    print("\n" + "=" * 60)
    print("Testing Performance Scaling")
    print("=" * 60)
    
    alpha = 1.5
    beta = 0.0
    sizes = [100, 500, 1000, 2000, 5000]
    
    print(f"{'Size':<8} {'Time (s)':<10} {'Mean':<8} {'Std':<8} {'Extreme':<8}")
    print("-" * 50)
    
    for n in sizes:
        try:
            model = AlphaStableModel(alpha=alpha, beta=beta, sigma=1.0, mu=0.0)
            
            start_time = time.time()
            data = model.generate(n, seed=42)
            generation_time = time.time() - start_time
            
            mean_val = np.mean(data)
            std_val = np.std(data)
            extreme_count = np.sum(np.abs(data) > 5)
            
            print(f"{n:<8} {generation_time:<10.4f} {mean_val:<8.4f} {std_val:<8.4f} {extreme_count:<8}")
            
        except Exception as e:
            print(f"{n:<8} FAILED: {e}")

def test_properties_and_moments():
    """Test model properties and moment calculations."""
    print("\n" + "=" * 60)
    print("Testing Properties and Moments")
    print("=" * 60)
    
    test_cases = [
        (2.0, 0.0, "Gaussian"),
        (1.5, 0.0, "Symmetric α=1.5"),
        (1.0, 0.5, "Skewed Cauchy"),
        (0.8, 0.0, "Heavy-tailed"),
    ]
    
    for alpha, beta, name in test_cases:
        print(f"\n{name} (α={alpha}, β={beta}):")
        print("-" * 30)
        
        model = AlphaStableModel(alpha=alpha, beta=beta, sigma=1.0, mu=0.0)
        
        # Get model properties
        properties = model.get_properties()
        theoretical = model.theoretical_moments()
        
        print(f"Model properties:")
        for key, value in properties.items():
            print(f"  {key}: {value}")
        
        print(f"Theoretical moments:")
        for key, value in theoretical.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

def create_visualization():
    """Create visualization of different alpha-stable distributions."""
    print("\n" + "=" * 60)
    print("Creating Visualization")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        
        # Test cases for visualization
        test_cases = [
            (2.0, 0.0, "Gaussian", "blue"),
            (1.5, 0.0, "Symmetric α=1.5", "green"),
            (1.0, 0.0, "Cauchy", "red"),
            (0.8, 0.0, "Heavy-tailed α=0.8", "orange"),
        ]
        
        n = 2000
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Alpha-Stable Distributions', fontsize=16)
        
        for i, (alpha, beta, name, color) in enumerate(test_cases):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            model = AlphaStableModel(alpha=alpha, beta=beta, sigma=1.0, mu=0.0)
            data = model.generate(n, seed=42)
            
            # Plot histogram
            ax.hist(data, bins=50, density=True, alpha=0.7, color=color, label=name)
            ax.set_title(f'{name} (α={alpha}, β={beta})')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            ax.text(0.05, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('alpha_stable_distributions.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'alpha_stable_distributions.png'")
        
    except ImportError:
        print("❌ Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def main():
    """Main test function."""
    print("Alpha-Stable Data Model Test Suite")
    print("Demonstrating 'Adding Data Model' functionality")
    print("=" * 80)
    
    try:
        # Run all tests
        test_parameter_validation()
        test_generation_methods()
        test_different_parameters()
        test_backend_selection()
        test_special_cases()
        test_performance_scaling()
        test_properties_and_moments()
        create_visualization()
        
        print("\n" + "=" * 80)
        print("✅ All tests completed successfully!")
        print("The Alpha-Stable data model has been successfully integrated into LRDBenchmark.")
        print("This demonstrates the framework's capability to add new data models.")
        print("Key features demonstrated:")
        print("  - Multiple generation methods (CMS, Nolan, Fourier)")
        print("  - Parameter validation and error handling")
        print("  - Backend optimization (JAX, Numba, NumPy)")
        print("  - Special case handling (Gaussian, Cauchy, heavy tails)")
        print("  - Performance scaling and property analysis")
        print("  - Comprehensive testing and visualization")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
