#!/usr/bin/env python3
"""
Enhanced Evaluation Metrics Framework for LRDBenchmark

This module provides comprehensive evaluation metrics beyond basic MAE and execution time,
including bias, variance, confidence interval coverage, scaling behavior accuracy,
and domain-specific evaluation criteria.

Author: LRDBenchmark Team
Date: 2025-01-05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any, Optional, Union
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EnhancedEvaluationMetrics:
    """
    Comprehensive evaluation metrics framework for LRD estimation methods.
    
    Provides evaluation metrics including:
    - Bias and variance analysis
    - Confidence interval coverage
    - Scaling behavior accuracy
    - Domain-specific evaluation criteria
    - Robustness metrics
    - Computational efficiency metrics
    """
    
    def __init__(self, results_dir: str = "."):
        """Initialize the enhanced evaluation metrics framework."""
        self.results_dir = Path(results_dir)
        self.evaluation_results = {}
        
        # Domain-specific evaluation criteria
        self.domain_criteria = {
            'finance': {
                'accuracy_threshold': 0.05,
                'speed_requirement': 1.0,  # seconds
                'robustness_requirement': 0.8,
                'scaling_importance': 'high'
            },
            'neuroscience': {
                'accuracy_threshold': 0.1,
                'speed_requirement': 5.0,  # seconds
                'robustness_requirement': 0.9,
                'scaling_importance': 'medium'
            },
            'climate': {
                'accuracy_threshold': 0.15,
                'speed_requirement': 10.0,  # seconds
                'robustness_requirement': 0.85,
                'scaling_importance': 'high'
            },
            'economics': {
                'accuracy_threshold': 0.08,
                'speed_requirement': 2.0,  # seconds
                'robustness_requirement': 0.8,
                'scaling_importance': 'medium'
            },
            'physics': {
                'accuracy_threshold': 0.12,
                'speed_requirement': 3.0,  # seconds
                'robustness_requirement': 0.85,
                'scaling_importance': 'high'
            }
        }
    
    def calculate_bias_metrics(self, true_values: np.ndarray, estimated_values: np.ndarray) -> Dict:
        """
        Calculate comprehensive bias metrics.
        
        Args:
            true_values: True Hurst parameter values
            estimated_values: Estimated Hurst parameter values
            
        Returns:
            Dictionary with bias metrics
        """
        errors = estimated_values - true_values
        
        # Basic bias metrics
        mean_bias = np.mean(errors)
        median_bias = np.median(errors)
        std_bias = np.std(errors)
        
        # Bias magnitude metrics
        mean_absolute_bias = np.mean(np.abs(errors))
        max_bias = np.max(np.abs(errors))
        
        # Bias direction analysis
        positive_bias = np.sum(errors > 0) / len(errors)
        negative_bias = np.sum(errors < 0) / len(errors)
        
        # Systematic bias detection
        # Test if bias is significantly different from zero
        t_stat, p_value = stats.ttest_1samp(errors, 0)
        significant_bias = p_value < 0.05
        
        # Bias stability (consistency across different true values)
        bias_stability = self._calculate_bias_stability(true_values, errors)
        
        return {
            'mean_bias': float(mean_bias),
            'median_bias': float(median_bias),
            'std_bias': float(std_bias),
            'mean_absolute_bias': float(mean_absolute_bias),
            'max_bias': float(max_bias),
            'positive_bias_ratio': float(positive_bias),
            'negative_bias_ratio': float(negative_bias),
            'significant_bias': bool(significant_bias),
            'bias_p_value': float(p_value),
            'bias_stability': float(bias_stability)
        }
    
    def calculate_variance_metrics(self, estimated_values: np.ndarray) -> Dict:
        """
        Calculate comprehensive variance metrics.
        
        Args:
            estimated_values: Estimated Hurst parameter values
            
        Returns:
            Dictionary with variance metrics
        """
        # Basic variance metrics
        variance = np.var(estimated_values)
        std_dev = np.std(estimated_values)
        coefficient_of_variation = std_dev / np.mean(estimated_values) if np.mean(estimated_values) != 0 else np.inf
        
        # Variance stability
        variance_stability = self._calculate_variance_stability(estimated_values)
        
        # Outlier detection
        q1, q3 = np.percentile(estimated_values, [25, 75])
        iqr = q3 - q1
        outlier_threshold = 1.5 * iqr
        outliers = np.sum((estimated_values < q1 - outlier_threshold) | 
                         (estimated_values > q3 + outlier_threshold))
        outlier_ratio = outliers / len(estimated_values)
        
        return {
            'variance': float(variance),
            'std_dev': float(std_dev),
            'coefficient_of_variation': float(coefficient_of_variation),
            'variance_stability': float(variance_stability),
            'outlier_count': int(outliers),
            'outlier_ratio': float(outlier_ratio)
        }
    
    def calculate_confidence_interval_metrics(self, true_values: np.ndarray, 
                                           estimated_values: np.ndarray, 
                                           confidence_level: float = 0.95) -> Dict:
        """
        Calculate confidence interval coverage and related metrics.
        
        Args:
            true_values: True Hurst parameter values
            estimated_values: Estimated Hurst parameter values
            confidence_level: Confidence level for intervals (default 0.95)
            
        Returns:
            Dictionary with confidence interval metrics
        """
        # Calculate confidence intervals
        n = len(estimated_values)
        mean_est = np.mean(estimated_values)
        std_est = np.std(estimated_values)
        
        # Standard error
        se = std_est / np.sqrt(n)
        
        # Critical value for confidence level
        alpha = 1 - confidence_level
        critical_value = stats.t.ppf(1 - alpha/2, n - 1)
        
        # Confidence interval
        margin_of_error = critical_value * se
        ci_lower = mean_est - margin_of_error
        ci_upper = mean_est + margin_of_error
        
        # Coverage analysis
        true_mean = np.mean(true_values)
        coverage = ci_lower <= true_mean <= ci_upper
        
        # Individual prediction intervals
        prediction_intervals = []
        for i in range(len(estimated_values)):
            pi_lower = estimated_values[i] - critical_value * std_est
            pi_upper = estimated_values[i] + critical_value * std_est
            prediction_intervals.append((pi_lower, pi_upper))
        
        # Coverage of individual predictions
        individual_coverage = np.sum([
            pi_lower <= true_val <= pi_upper 
            for (pi_lower, pi_upper), true_val in zip(prediction_intervals, true_values)
        ]) / len(true_values)
        
        # Interval width
        interval_width = ci_upper - ci_lower
        
        return {
            'confidence_level': confidence_level,
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'interval_width': float(interval_width),
            'coverage': coverage,
            'individual_coverage': float(individual_coverage),
            'margin_of_error': float(margin_of_error),
            'critical_value': float(critical_value)
        }
    
    def calculate_scaling_behavior_metrics(self, true_values: np.ndarray, 
                                         estimated_values: np.ndarray,
                                         data_lengths: np.ndarray) -> Dict:
        """
        Calculate scaling behavior accuracy metrics.
        
        Args:
            true_values: True Hurst parameter values
            estimated_values: Estimated Hurst parameter values
            data_lengths: Length of time series data
            
        Returns:
            Dictionary with scaling behavior metrics
        """
        # Group by data length
        unique_lengths = np.unique(data_lengths)
        scaling_metrics = {}
        
        for length in unique_lengths:
            mask = data_lengths == length
            true_subset = true_values[mask]
            est_subset = estimated_values[mask]
            
            if len(true_subset) > 1:
                # Calculate MAE for this length
                mae = np.mean(np.abs(est_subset - true_subset))
                scaling_metrics[f'mae_length_{length}'] = float(mae)
        
        # Scaling law analysis: MAE = a * n^b
        if len(unique_lengths) > 2:
            log_lengths = np.log(unique_lengths)
            log_maes = [scaling_metrics.get(f'mae_length_{length}', 0) for length in unique_lengths]
            log_maes = np.log(log_maes)
            
            # Remove any infinite or NaN values
            valid_mask = np.isfinite(log_lengths) & np.isfinite(log_maes)
            if np.sum(valid_mask) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_lengths[valid_mask], log_maes[valid_mask]
                )
                
                scaling_metrics.update({
                    'scaling_slope': float(slope),
                    'scaling_intercept': float(intercept),
                    'scaling_r_squared': float(r_value**2),
                    'scaling_p_value': float(p_value),
                    'scaling_std_error': float(std_err)
                })
        
        return scaling_metrics
    
    def calculate_domain_specific_metrics(self, estimator_name: str, 
                                        performance_metrics: Dict,
                                        domain: str) -> Dict:
        """
        Calculate domain-specific evaluation metrics.
        
        Args:
            estimator_name: Name of the estimator
            performance_metrics: Basic performance metrics
            domain: Domain of application
            
        Returns:
            Dictionary with domain-specific metrics
        """
        if domain not in self.domain_criteria:
            return {}
        
        criteria = self.domain_criteria[domain]
        mae = performance_metrics.get('mean_mae', 0)
        execution_time = performance_metrics.get('mean_execution_time', 0)
        success_rate = performance_metrics.get('success_rate', 0)
        
        # Domain-specific scoring
        accuracy_score = 1.0 if mae <= criteria['accuracy_threshold'] else 0.0
        speed_score = 1.0 if execution_time <= criteria['speed_requirement'] else 0.0
        robustness_score = 1.0 if success_rate >= criteria['robustness_requirement'] else 0.0
        
        # Overall domain score
        domain_score = (accuracy_score + speed_score + robustness_score) / 3.0
        
        # Scaling importance factor
        scaling_importance = criteria['scaling_importance']
        scaling_weight = {'high': 0.4, 'medium': 0.2, 'low': 0.1}[scaling_importance]
        
        return {
            'domain': domain,
            'accuracy_score': float(accuracy_score),
            'speed_score': float(speed_score),
            'robustness_score': float(robustness_score),
            'domain_score': float(domain_score),
            'scaling_importance': scaling_importance,
            'scaling_weight': float(scaling_weight),
            'meets_accuracy_threshold': bool(mae <= criteria['accuracy_threshold']),
            'meets_speed_requirement': bool(execution_time <= criteria['speed_requirement']),
            'meets_robustness_requirement': bool(success_rate >= criteria['robustness_requirement'])
        }
    
    def calculate_robustness_metrics(self, performance_data: Dict) -> Dict:
        """
        Calculate robustness metrics beyond contamination testing.
        
        Args:
            performance_data: Performance data across different conditions
            
        Returns:
            Dictionary with robustness metrics
        """
        # Extract performance across different conditions
        mae_values = []
        success_rates = []
        
        for condition, data in performance_data.items():
            if 'mae' in data:
                mae_values.append(data['mae'])
            if 'success_rate' in data:
                success_rates.append(data['success_rate'])
        
        if not mae_values:
            return {}
        
        # Robustness metrics
        mae_std = np.std(mae_values)
        mae_cv = mae_std / np.mean(mae_values) if np.mean(mae_values) != 0 else np.inf
        
        # Performance stability
        performance_stability = 1.0 - mae_cv if mae_cv < 1.0 else 0.0
        
        # Success rate stability
        success_rate_stability = 1.0 - np.std(success_rates) if success_rates else 0.0
        
        # Worst-case performance
        worst_mae = np.max(mae_values)
        worst_success_rate = np.min(success_rates) if success_rates else 0.0
        
        return {
            'mae_std': float(mae_std),
            'mae_coefficient_of_variation': float(mae_cv),
            'performance_stability': float(performance_stability),
            'success_rate_stability': float(success_rate_stability),
            'worst_mae': float(worst_mae),
            'worst_success_rate': float(worst_success_rate)
        }
    
    def calculate_computational_efficiency_metrics(self, execution_times: np.ndarray,
                                                data_lengths: np.ndarray) -> Dict:
        """
        Calculate computational efficiency metrics beyond execution time.
        
        Args:
            execution_times: Execution times for different runs
            data_lengths: Length of time series data
            
        Returns:
            Dictionary with computational efficiency metrics
        """
        # Basic efficiency metrics
        mean_time = np.mean(execution_times)
        std_time = np.std(execution_times)
        median_time = np.median(execution_times)
        
        # Time complexity analysis
        if len(np.unique(data_lengths)) > 2:
            log_lengths = np.log(data_lengths)
            log_times = np.log(execution_times)
            
            # Remove any infinite or NaN values
            valid_mask = np.isfinite(log_lengths) & np.isfinite(log_times)
            if np.sum(valid_mask) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_lengths[valid_mask], log_times[valid_mask]
                )
                
                time_complexity = {
                    'time_slope': float(slope),
                    'time_intercept': float(intercept),
                    'time_r_squared': float(r_value**2),
                    'time_p_value': float(p_value),
                    'time_std_error': float(std_err)
                }
            else:
                time_complexity = {}
        else:
            time_complexity = {}
        
        # Efficiency per data point
        efficiency_per_point = execution_times / data_lengths
        mean_efficiency = np.mean(efficiency_per_point)
        
        # Scalability metrics
        max_time = np.max(execution_times)
        min_time = np.min(execution_times)
        time_range = max_time - min_time
        
        return {
            'mean_execution_time': float(mean_time),
            'std_execution_time': float(std_time),
            'median_execution_time': float(median_time),
            'max_execution_time': float(max_time),
            'min_execution_time': float(min_time),
            'time_range': float(time_range),
            'mean_efficiency_per_point': float(mean_efficiency),
            'time_complexity': time_complexity
        }
    
    def _calculate_bias_stability(self, true_values: np.ndarray, errors: np.ndarray) -> float:
        """Calculate bias stability across different true values."""
        # Group errors by true value ranges
        true_ranges = np.digitize(true_values, bins=np.linspace(0, 1, 6))
        bias_by_range = []
        
        for i in range(1, 6):
            mask = true_ranges == i
            if np.sum(mask) > 0:
                bias_by_range.append(np.mean(errors[mask]))
        
        if len(bias_by_range) > 1:
            return 1.0 - np.std(bias_by_range) / np.mean(np.abs(bias_by_range))
        else:
            return 1.0
    
    def _calculate_variance_stability(self, values: np.ndarray) -> float:
        """Calculate variance stability across different subsets."""
        if len(values) < 10:
            return 1.0
        
        # Split into subsets and calculate variance for each
        n_subsets = min(5, len(values) // 2)
        subset_size = len(values) // n_subsets
        variances = []
        
        for i in range(n_subsets):
            start_idx = i * subset_size
            end_idx = start_idx + subset_size
            subset = values[start_idx:end_idx]
            if len(subset) > 1:
                variances.append(np.var(subset))
        
        if len(variances) > 1:
            return 1.0 - np.std(variances) / np.mean(variances)
        else:
            return 1.0
    
    def evaluate_estimator_comprehensive(self, estimator_name: str, 
                                       test_results: List[Dict]) -> Dict:
        """
        Perform comprehensive evaluation of an estimator.
        
        Args:
            estimator_name: Name of the estimator
            test_results: List of test results
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        print(f"Evaluating {estimator_name} with enhanced metrics...")
        
        # Extract data
        true_values = np.array([r['hurst_true'] for r in test_results if r.get('success', False)])
        estimated_values = np.array([r['hurst_estimate'] for r in test_results if r.get('success', False)])
        execution_times = np.array([r['execution_time'] for r in test_results if r.get('success', False)])
        data_lengths = np.array([r['data_length'] for r in test_results if r.get('success', False)])
        
        if len(true_values) == 0:
            return {}
        
        # Calculate all metrics
        bias_metrics = self.calculate_bias_metrics(true_values, estimated_values)
        variance_metrics = self.calculate_variance_metrics(estimated_values)
        ci_metrics = self.calculate_confidence_interval_metrics(true_values, estimated_values)
        scaling_metrics = self.calculate_scaling_behavior_metrics(true_values, estimated_values, data_lengths)
        robustness_metrics = self.calculate_robustness_metrics({'default': {'mae': np.mean(np.abs(estimated_values - true_values))}})
        efficiency_metrics = self.calculate_computational_efficiency_metrics(execution_times, data_lengths)
        
        # Domain-specific metrics for all domains
        domain_metrics = {}
        basic_metrics = {
            'mean_mae': np.mean(np.abs(estimated_values - true_values)),
            'mean_execution_time': np.mean(execution_times),
            'success_rate': len(true_values) / len(test_results)
        }
        
        for domain in self.domain_criteria.keys():
            domain_metrics[domain] = self.calculate_domain_specific_metrics(
                estimator_name, basic_metrics, domain
            )
        
        # Compile comprehensive results
        comprehensive_results = {
            'estimator_name': estimator_name,
            'basic_metrics': basic_metrics,
            'bias_metrics': bias_metrics,
            'variance_metrics': variance_metrics,
            'confidence_interval_metrics': ci_metrics,
            'scaling_metrics': scaling_metrics,
            'robustness_metrics': robustness_metrics,
            'efficiency_metrics': efficiency_metrics,
            'domain_metrics': domain_metrics,
            'sample_size': len(true_values)
        }
        
        return comprehensive_results
    
    def evaluate_all_estimators(self, results_data: Dict) -> Dict:
        """
        Evaluate all estimators with enhanced metrics.
        
        Args:
            results_data: Dictionary containing benchmark results
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        print("Performing comprehensive evaluation of all estimators...")
        
        evaluation_results = {}
        
        for estimator_name, estimator_data in results_data.items():
            if 'test_results' not in estimator_data:
                continue
            
            test_results = estimator_data['test_results']
            comprehensive_evaluation = self.evaluate_estimator_comprehensive(
                estimator_name, test_results
            )
            
            if comprehensive_evaluation:
                evaluation_results[estimator_name] = comprehensive_evaluation
        
        return evaluation_results
    
    def generate_evaluation_summary(self, evaluation_results: Dict) -> Dict:
        """
        Generate summary of evaluation results.
        
        Args:
            evaluation_results: Comprehensive evaluation results
            
        Returns:
            Dictionary with evaluation summary
        """
        summary = {
            'overall_statistics': {},
            'method_categories': {
                'classical': {'count': 0, 'avg_metrics': {}},
                'machine_learning': {'count': 0, 'avg_metrics': {}},
                'neural_networks': {'count': 0, 'avg_metrics': {}}
            },
            'domain_analysis': {},
            'recommendations': []
        }
        
        # Analyze by method category
        for estimator_name, results in evaluation_results.items():
            category = self._get_estimator_category(estimator_name)
            if category in summary['method_categories']:
                summary['method_categories'][category]['count'] += 1
                
                # Aggregate metrics
                basic_metrics = results['basic_metrics']
                for metric, value in basic_metrics.items():
                    if metric not in summary['method_categories'][category]['avg_metrics']:
                        summary['method_categories'][category]['avg_metrics'][metric] = []
                    summary['method_categories'][category]['avg_metrics'][metric].append(value)
        
        # Calculate averages
        for category in summary['method_categories']:
            count = summary['method_categories'][category]['count']
            if count > 0:
                for metric in summary['method_categories'][category]['avg_metrics']:
                    values = summary['method_categories'][category]['avg_metrics'][metric]
                    summary['method_categories'][category]['avg_metrics'][metric] = np.mean(values)
        
        # Domain analysis
        for domain in self.domain_criteria.keys():
            domain_scores = []
            for estimator_name, results in evaluation_results.items():
                if domain in results['domain_metrics']:
                    domain_scores.append(results['domain_metrics'][domain]['domain_score'])
            
            if domain_scores:
                summary['domain_analysis'][domain] = {
                    'avg_domain_score': np.mean(domain_scores),
                    'best_performers': self._get_best_performers_by_domain(evaluation_results, domain)
                }
        
        # Generate recommendations
        summary['recommendations'] = self._generate_evaluation_recommendations(evaluation_results)
        
        return summary
    
    def _get_estimator_category(self, estimator_name: str) -> str:
        """Get the category of an estimator."""
        classical_methods = ['R/S', 'DFA', 'DMA', 'Higuchi', 'GPH', 'Whittle', 'Periodogram']
        ml_methods = ['RandomForest', 'SVR', 'GradientBoosting']
        nn_methods = ['LSTM', 'GRU', 'Transformer', 'CNN', 'Feedforward', 'ResNet']
        
        if estimator_name in classical_methods:
            return 'classical'
        elif estimator_name in ml_methods:
            return 'machine_learning'
        elif estimator_name in nn_methods:
            return 'neural_networks'
        else:
            return 'unknown'
    
    def _get_best_performers_by_domain(self, evaluation_results: Dict, domain: str) -> List[str]:
        """Get best performing estimators for a specific domain."""
        domain_scores = []
        for estimator_name, results in evaluation_results.items():
            if domain in results['domain_metrics']:
                score = results['domain_metrics'][domain]['domain_score']
                domain_scores.append((estimator_name, score))
        
        # Sort by score and return top 3
        domain_scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, score in domain_scores[:3]]
    
    def _generate_evaluation_recommendations(self, evaluation_results: Dict) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Find best overall performers
        mae_scores = [(name, results['basic_metrics']['mean_mae']) 
                      for name, results in evaluation_results.items()]
        mae_scores.sort(key=lambda x: x[1])
        
        if mae_scores:
            best_mae = mae_scores[0]
            recommendations.append(f"Best accuracy: {best_mae[0]} (MAE: {best_mae[1]:.4f})")
        
        # Find most robust performers
        robustness_scores = [(name, results['robustness_metrics'].get('performance_stability', 0))
                            for name, results in evaluation_results.items()]
        robustness_scores.sort(key=lambda x: x[1], reverse=True)
        
        if robustness_scores:
            best_robustness = robustness_scores[0]
            recommendations.append(f"Most robust: {best_robustness[0]} (stability: {best_robustness[1]:.4f})")
        
        # Find most efficient performers
        efficiency_scores = [(name, results['efficiency_metrics']['mean_execution_time'])
                            for name, results in evaluation_results.items()]
        efficiency_scores.sort(key=lambda x: x[1])
        
        if efficiency_scores:
            best_efficiency = efficiency_scores[0]
            recommendations.append(f"Most efficient: {best_efficiency[0]} (time: {best_efficiency[1]:.4f}s)")
        
        return recommendations
    
    def save_evaluation_results(self, evaluation_results: Dict, filename: str = "enhanced_evaluation_results.json"):
        """Save evaluation results to file."""
        output_path = self.results_dir / filename
        
        # Convert numpy types and booleans to JSON-serializable types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return bool(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # Handle numpy scalars
                return obj.item()
            else:
                return obj
        
        converted_results = convert_types(evaluation_results)
        
        with open(output_path, 'w') as f:
            json.dump(converted_results, f, indent=2)
        print(f"Enhanced evaluation results saved to {output_path}")
    
    def load_benchmark_results(self, filename: str = "comprehensive_final_nn_results/comprehensive_final_nn_benchmark_*.json") -> Dict:
        """Load benchmark results from file."""
        results_files = list(self.results_dir.glob(filename))
        if not results_files:
            raise FileNotFoundError(f"No results files found matching {filename}")
        
        # Load the most recent detailed results file (not summary)
        detailed_files = [f for f in results_files if not f.name.endswith('_summary.json')]
        if not detailed_files:
            raise FileNotFoundError("No detailed results files found")
        
        latest_file = max(detailed_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading results from {latest_file}")
        
        with open(latest_file, 'r') as f:
            return json.load(f)

def main():
    """Main function to run enhanced evaluation metrics analysis."""
    print("Starting Enhanced Evaluation Metrics Framework...")
    
    # Initialize evaluator
    evaluator = EnhancedEvaluationMetrics()
    
    try:
        # Load benchmark results
        results_data = evaluator.load_benchmark_results()
        
        # Perform comprehensive evaluation
        evaluation_results = evaluator.evaluate_all_estimators(results_data)
        
        # Generate summary
        summary = evaluator.generate_evaluation_summary(evaluation_results)
        
        # Save results
        evaluator.save_evaluation_results(evaluation_results)
        
        # Print summary
        print("\n" + "="*60)
        print("ENHANCED EVALUATION METRICS SUMMARY")
        print("="*60)
        
        print(f"\nMethod Categories Analysis:")
        for category, stats in summary['method_categories'].items():
            if stats['count'] > 0:
                print(f"  {category.title()}: {stats['count']} methods")
                for metric, value in stats['avg_metrics'].items():
                    print(f"    {metric}: {value:.4f}")
        
        print(f"\nDomain Analysis:")
        for domain, analysis in summary['domain_analysis'].items():
            print(f"  {domain.title()}: avg score {analysis['avg_domain_score']:.4f}")
            print(f"    Best performers: {', '.join(analysis['best_performers'])}")
        
        print(f"\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
        
        print("\nEnhanced evaluation metrics analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during enhanced evaluation analysis: {e}")
        raise

if __name__ == "__main__":
    main()
