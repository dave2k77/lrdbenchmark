#!/usr/bin/env python3
"""
Comprehensive Analysis of Classical LRD Estimators

This script analyzes the mathematical foundations, algorithmic logic, and implementations
of all classical estimators to identify issues and ensure proper backend integration.
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Any, Tuple
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all estimators
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.temporal.dma.dma_estimator_unified import DMAEstimator
from lrdbenchmark.analysis.temporal.higuchi.higuchi_estimator_unified import HiguchiEstimator
from lrdbenchmark.analysis.spectral.gph.gph_estimator_unified import GPHEstimator
from lrdbenchmark.analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator
from lrdbenchmark.analysis.spectral.periodogram.periodogram_estimator_unified import PeriodogramEstimator
from lrdbenchmark.analysis.wavelet.cwt.cwt_estimator_unified import CWTEstimator
from lrdbenchmark.analysis.wavelet.variance.variance_estimator_unified import WaveletVarianceEstimator
from lrdbenchmark.analysis.wavelet.log_variance.log_variance_estimator_unified import WaveletLogVarianceEstimator
from lrdbenchmark.analysis.wavelet.whittle.whittle_estimator_unified import WaveletWhittleEstimator
from lrdbenchmark.analysis.multifractal.mfdfa.mfdfa_estimator_unified import MFDFAEstimator
from lrdbenchmark.analysis.multifractal.wavelet_leaders.wavelet_leaders_estimator_unified import MultifractalWaveletLeadersEstimator


class ClassicalEstimatorAnalyzer:
    """
    Comprehensive analyzer for classical LRD estimators.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.estimators = {
            # Temporal estimators
            'RS': RSEstimator(),
            'DFA': DFAEstimator(),
            'DMA': DMAEstimator(),
            'Higuchi': HiguchiEstimator(),
            
            # Spectral estimators
            'GPH': GPHEstimator(),
            'Whittle': WhittleEstimator(),
            'Periodogram': PeriodogramEstimator(),
            
            # Wavelet estimators
            'CWT': CWTEstimator(),
            'WaveletVar': WaveletVarianceEstimator(),
            'WaveletLogVar': WaveletLogVarianceEstimator(),
            'WaveletWhittle': WaveletWhittleEstimator(),
            
            # Multifractal estimators
            'MFDFA': MFDFAEstimator(),
            'WaveletLeaders': MultifractalWaveletLeadersEstimator(),
        }
        
        self.analysis_results = {}
        
    def analyze_mathematical_foundations(self) -> Dict[str, Any]:
        """Analyze the mathematical foundations of each estimator."""
        print("🔬 Analyzing Mathematical Foundations...")
        
        foundations = {
            'RS': {
                'theory': 'R/S analysis measures the rescaled range of cumulative deviations',
                'formula': 'R/S = (max(cumsum(X - mean(X))) - min(cumsum(X - mean(X)))) / std(X)',
                'hurst_relation': 'E[R/S] ∝ n^H',
                'validity': 'Valid for 0 < H < 1, works well for H > 0.5',
                'limitations': 'Biased for small samples, sensitive to trends'
            },
            'DFA': {
                'theory': 'Detrended Fluctuation Analysis measures RMS of detrended fluctuations',
                'formula': 'F(n) = sqrt(mean((Y - Y_trend)^2))',
                'hurst_relation': 'F(n) ∝ n^H',
                'validity': 'Valid for 0 < H < 1, robust to trends',
                'limitations': 'Requires polynomial detrending, sensitive to detrending order'
            },
            'DMA': {
                'theory': 'Detrended Moving Average uses moving average detrending',
                'formula': 'F(n) = sqrt(mean((Y - MA(Y))^2))',
                'hurst_relation': 'F(n) ∝ n^H',
                'validity': 'Valid for 0 < H < 1, good for non-stationary data',
                'limitations': 'Less robust than DFA, sensitive to window size'
            },
            'Higuchi': {
                'theory': 'Higuchi method estimates fractal dimension from curve length',
                'formula': 'L(k) = (N-1)/(k^2) * sum(|X(i+k) - X(i)|)',
                'hurst_relation': 'H = 2 - D, where D is fractal dimension',
                'validity': 'Valid for 0 < H < 1, good for short series',
                'limitations': 'Sensitive to noise, requires careful k selection'
            },
            'GPH': {
                'theory': 'Geweke-Porter-Hudak uses log-periodogram regression',
                'formula': 'log(I(ω)) = c - d*log(4*sin²(ω/2)) + error',
                'hurst_relation': 'H = d + 0.5',
                'validity': 'Valid for 0 < H < 1, good for long series',
                'limitations': 'Sensitive to frequency range, requires bias correction'
            },
            'Whittle': {
                'theory': 'Whittle likelihood maximizes spectral likelihood',
                'formula': 'L = -0.5 * sum(log(f(ω)) + I(ω)/f(ω))',
                'hurst_relation': 'H estimated from spectral density f(ω)',
                'validity': 'Valid for 0 < H < 1, statistically efficient',
                'limitations': 'Computationally intensive, requires model specification'
            },
            'Periodogram': {
                'theory': 'Periodogram-based estimation using power spectral density',
                'formula': 'I(ω) = |FFT(X)|²',
                'hurst_relation': 'I(ω) ∝ ω^(-β), β = 2H - 1',
                'validity': 'Valid for 0 < H < 1, simple implementation',
                'limitations': 'Noisy for short series, requires smoothing'
            },
            'CWT': {
                'theory': 'Continuous Wavelet Transform analyzes scaling behavior',
                'formula': 'W(a,b) = (1/√a) ∫ X(t) ψ*((t-b)/a) dt',
                'hurst_relation': '|W(a,b)|² ∝ a^(2H+1)',
                'validity': 'Valid for 0 < H < 1, good time-frequency resolution',
                'limitations': 'Computationally intensive, requires wavelet selection'
            },
            'WaveletVar': {
                'theory': 'Wavelet variance estimates Hurst from wavelet coefficients',
                'formula': 'Var(W_j) = 2^(j(2H-1)) * σ²',
                'hurst_relation': 'log(Var(W_j)) = j(2H-1) + log(σ²)',
                'validity': 'Valid for 0 < H < 1, robust to trends',
                'limitations': 'Requires appropriate wavelet, sensitive to boundary effects'
            },
            'WaveletLogVar': {
                'theory': 'Log-wavelet variance for better linearity',
                'formula': 'log(Var(W_j)) = j(2H-1) + log(σ²)',
                'hurst_relation': 'Same as WaveletVar but with log transformation',
                'validity': 'Valid for 0 < H < 1, improved linearity',
                'limitations': 'Same as WaveletVar, additional log transformation'
            },
            'WaveletWhittle': {
                'theory': 'Wavelet-based Whittle likelihood estimation',
                'formula': 'L = -0.5 * sum(log(f_w(ω)) + |W(ω)|²/f_w(ω))',
                'hurst_relation': 'H estimated from wavelet spectral density',
                'validity': 'Valid for 0 < H < 1, combines wavelet and Whittle',
                'limitations': 'Complex implementation, computationally intensive'
            },
            'MFDFA': {
                'theory': 'Multifractal DFA extends DFA to multifractal analysis',
                'formula': 'F_q(n) = (mean(|F(n)|^q))^(1/q)',
                'hurst_relation': 'F_q(n) ∝ n^h(q), h(2) ≈ H',
                'validity': 'Valid for multifractal processes, H = h(2)',
                'limitations': 'Computationally intensive, requires q range selection'
            },
            'WaveletLeaders': {
                'theory': 'Wavelet leaders for multifractal analysis',
                'formula': 'L(j,k) = sup_{λ⊂3λ_j,k} |W(λ)|',
                'hurst_relation': 'S_q(j) ∝ 2^(jζ(q)), ζ(2) ≈ 2H-1',
                'validity': 'Valid for multifractal processes, robust method',
                'limitations': 'Complex implementation, requires leader selection'
            }
        }
        
        return foundations
    
    def test_estimator_functionality(self) -> Dict[str, Any]:
        """Test the functionality of each estimator."""
        print("🧪 Testing Estimator Functionality...")
        
        # Generate test data
        np.random.seed(42)
        test_data = {
            'fbm': np.cumsum(np.random.randn(1000)),  # Random walk (H ≈ 0.5)
            'fgn': np.random.randn(1000),  # White noise (H ≈ 0.5)
            'short': np.cumsum(np.random.randn(100)),  # Short series
            'long': np.cumsum(np.random.randn(5000)),  # Long series
        }
        
        results = {}
        
        for name, estimator in self.estimators.items():
            print(f"  Testing {name}...")
            estimator_results = {}
            
            for data_name, data in test_data.items():
                try:
                    start_time = time.time()
                    result = estimator.estimate(data)
                    execution_time = time.time() - start_time
                    
                    # Extract key metrics
                    hurst = result.get('hurst_parameter', np.nan)
                    r_squared = result.get('r_squared', np.nan)
                    success = not np.isnan(hurst) and 0 < hurst < 1
                    
                    estimator_results[data_name] = {
                        'success': success,
                        'hurst': hurst,
                        'r_squared': r_squared,
                        'execution_time': execution_time,
                        'error': None
                    }
                    
                except Exception as e:
                    estimator_results[data_name] = {
                        'success': False,
                        'hurst': np.nan,
                        'r_squared': np.nan,
                        'execution_time': 0,
                        'error': str(e)
                    }
            
            results[name] = estimator_results
        
        return results
    
    def analyze_optimization_integration(self) -> Dict[str, Any]:
        """Analyze the optimization framework integration."""
        print("⚡ Analyzing Optimization Integration...")
        
        integration_results = {}
        
        for name, estimator in self.estimators.items():
            try:
                # Check if estimator has optimization framework selection
                has_optimization = hasattr(estimator, 'optimization_framework')
                has_jax = hasattr(estimator, '_estimate_jax')
                has_numba = hasattr(estimator, '_estimate_numba')
                has_numpy = hasattr(estimator, '_estimate_numpy')
                
                # Check parameter structure
                has_parameters = hasattr(estimator, 'parameters')
                has_validation = hasattr(estimator, '_validate_parameters')
                
                integration_results[name] = {
                    'has_optimization_framework': has_optimization,
                    'has_jax_implementation': has_jax,
                    'has_numba_implementation': has_numba,
                    'has_numpy_implementation': has_numpy,
                    'has_parameters': has_parameters,
                    'has_validation': has_validation,
                    'optimization_framework': getattr(estimator, 'optimization_framework', 'unknown'),
                    'parameters': getattr(estimator, 'parameters', {}),
                }
                
            except Exception as e:
                integration_results[name] = {
                    'error': str(e),
                    'has_optimization_framework': False,
                    'has_jax_implementation': False,
                    'has_numba_implementation': False,
                    'has_numpy_implementation': False,
                    'has_parameters': False,
                    'has_validation': False,
                }
        
        return integration_results
    
    def identify_issues(self) -> Dict[str, List[str]]:
        """Identify issues with the estimators."""
        print("🔍 Identifying Issues...")
        
        issues = {}
        
        for name, estimator in self.estimators.items():
            estimator_issues = []
            
            try:
                # Check for common issues
                if not hasattr(estimator, 'optimization_framework'):
                    estimator_issues.append("Missing optimization framework selection")
                
                if not hasattr(estimator, '_estimate_numpy'):
                    estimator_issues.append("Missing NumPy implementation")
                
                if not hasattr(estimator, '_estimate_numba'):
                    estimator_issues.append("Missing Numba implementation")
                
                if not hasattr(estimator, '_estimate_jax'):
                    estimator_issues.append("Missing JAX implementation")
                
                if not hasattr(estimator, 'parameters'):
                    estimator_issues.append("Missing parameters structure")
                
                if not hasattr(estimator, '_validate_parameters'):
                    estimator_issues.append("Missing parameter validation")
                
                # Test with small data
                try:
                    small_data = np.random.randn(50)
                    result = estimator.estimate(small_data)
                    if np.isnan(result.get('hurst_parameter', np.nan)):
                        estimator_issues.append("Returns NaN for small data")
                except Exception as e:
                    estimator_issues.append(f"Fails with small data: {str(e)}")
                
                # Test with large data
                try:
                    large_data = np.random.randn(10000)
                    result = estimator.estimate(large_data)
                    if np.isnan(result.get('hurst_parameter', np.nan)):
                        estimator_issues.append("Returns NaN for large data")
                except Exception as e:
                    estimator_issues.append(f"Fails with large data: {str(e)}")
                
            except Exception as e:
                estimator_issues.append(f"Critical error: {str(e)}")
            
            issues[name] = estimator_issues
        
        return issues
    
    def generate_recommendations(self) -> Dict[str, List[str]]:
        """Generate recommendations for improving the estimators."""
        print("💡 Generating Recommendations...")
        
        recommendations = {}
        
        for name, estimator in self.estimators.items():
            estimator_recommendations = []
            
            # General recommendations
            estimator_recommendations.append("Implement comprehensive error handling")
            estimator_recommendations.append("Add input validation for data types and sizes")
            estimator_recommendations.append("Implement proper logging for debugging")
            estimator_recommendations.append("Add confidence intervals for estimates")
            estimator_recommendations.append("Implement adaptive parameter selection")
            
            # Specific recommendations based on estimator type
            if name in ['RS', 'DFA', 'DMA']:
                estimator_recommendations.append("Implement adaptive scale selection")
                estimator_recommendations.append("Add trend detection and handling")
                estimator_recommendations.append("Implement robust regression methods")
            
            elif name in ['GPH', 'Whittle', 'Periodogram']:
                estimator_recommendations.append("Implement adaptive frequency range selection")
                estimator_recommendations.append("Add bias correction methods")
                estimator_recommendations.append("Implement robust spectral estimation")
            
            elif name in ['CWT', 'WaveletVar', 'WaveletLogVar', 'WaveletWhittle']:
                estimator_recommendations.append("Implement adaptive wavelet selection")
                estimator_recommendations.append("Add boundary effect handling")
                estimator_recommendations.append("Implement robust wavelet estimation")
            
            elif name in ['MFDFA', 'WaveletLeaders']:
                estimator_recommendations.append("Implement adaptive q-value selection")
                estimator_recommendations.append("Add multifractal spectrum analysis")
                estimator_recommendations.append("Implement robust multifractal estimation")
            
            recommendations[name] = estimator_recommendations
        
        return recommendations
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis of all estimators."""
        print("🚀 Running Comprehensive Analysis of Classical Estimators")
        print("=" * 70)
        
        # Run all analyses
        foundations = self.analyze_mathematical_foundations()
        functionality = self.test_estimator_functionality()
        integration = self.analyze_optimization_integration()
        issues = self.identify_issues()
        recommendations = self.generate_recommendations()
        
        # Compile results
        results = {
            'mathematical_foundations': foundations,
            'functionality_tests': functionality,
            'optimization_integration': integration,
            'identified_issues': issues,
            'recommendations': recommendations,
            'summary': self._generate_summary(foundations, functionality, integration, issues)
        }
        
        return results
    
    def _generate_summary(self, foundations, functionality, integration, issues) -> Dict[str, Any]:
        """Generate a summary of the analysis."""
        total_estimators = len(self.estimators)
        
        # Count estimators with issues
        estimators_with_issues = sum(1 for issues_list in issues.values() if issues_list)
        
        # Count estimators with optimization integration
        estimators_with_optimization = sum(1 for result in integration.values() 
                                         if result.get('has_optimization_framework', False))
        
        # Count estimators with all implementations
        estimators_with_all_impl = sum(1 for result in integration.values() 
                                     if (result.get('has_numpy_implementation', False) and
                                         result.get('has_numba_implementation', False) and
                                         result.get('has_jax_implementation', False)))
        
        # Calculate overall success rate
        total_tests = 0
        successful_tests = 0
        
        for estimator_results in functionality.values():
            for test_result in estimator_results.values():
                total_tests += 1
                if test_result['success']:
                    successful_tests += 1
        
        overall_success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'total_estimators': total_estimators,
            'estimators_with_issues': estimators_with_issues,
            'estimators_with_optimization': estimators_with_optimization,
            'estimators_with_all_implementations': estimators_with_all_impl,
            'overall_success_rate': overall_success_rate,
            'total_functionality_tests': total_tests,
            'successful_functionality_tests': successful_tests
        }


def main():
    """Main function to run the analysis."""
    analyzer = ClassicalEstimatorAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    # Print summary
    summary = results['summary']
    print(f"\n📊 ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total Estimators: {summary['total_estimators']}")
    print(f"Estimators with Issues: {summary['estimators_with_issues']}")
    print(f"Estimators with Optimization: {summary['estimators_with_optimization']}")
    print(f"Estimators with All Implementations: {summary['estimators_with_all_implementations']}")
    print(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
    print(f"Total Functionality Tests: {summary['total_functionality_tests']}")
    print(f"Successful Tests: {summary['successful_functionality_tests']}")
    
    # Print detailed issues
    print(f"\n🔍 DETAILED ISSUES")
    print("=" * 70)
    for name, issues in results['identified_issues'].items():
        if issues:
            print(f"\n{name}:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n{name}: No issues found ✅")
    
    # Save results
    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"classical_estimators_analysis_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("🎉 Analysis completed!")


if __name__ == "__main__":
    main()
