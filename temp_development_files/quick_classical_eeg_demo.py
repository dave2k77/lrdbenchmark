#!/usr/bin/env python3
"""
Quick Classical EEG Benchmark Demo

This script runs a quick demonstration of the classical estimators
with EEG contamination for immediate results.
"""

import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our adaptive estimators and backend
from lrdbenchmark.analysis.optimization_backend import OptimizationBackend
from lrdbenchmark.analysis.adaptive_classical_estimators import get_all_adaptive_classical_estimators

# Import data models
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion as FBMModel
from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise as FGNModel

# Import contamination factory
from lrdbenchmark.models.contamination.contamination_factory import (
    ContaminationFactory, 
    ConfoundingScenario
)


def run_quick_demo():
    """Run a quick demonstration of classical estimators with EEG contamination."""
    
    print("🧠 Classical LRD Estimators with EEG Contamination Demo")
    print("=" * 60)
    
    # Initialize components
    backend = OptimizationBackend()
    contamination_factory = ContaminationFactory(random_seed=42)
    estimators = get_all_adaptive_classical_estimators()
    
    print(f"✅ Initialized {len(estimators)} adaptive classical estimators")
    print(f"✅ Hardware: {backend.hardware_info.cpu_cores} cores, "
          f"{backend.hardware_info.memory_gb:.1f}GB RAM")
    print(f"✅ GPU: {'Available' if backend.hardware_info.has_gpu else 'Not available'}")
    
    # Test parameters
    hurst_values = [0.5, 0.7, 0.9]
    data_lengths = [1000, 2000]
    eeg_scenarios = [
        ConfoundingScenario.EEG_OCULAR_ARTIFACTS,
        ConfoundingScenario.EEG_MUSCLE_ARTIFACTS,
        ConfoundingScenario.EEG_60HZ_NOISE,
        ConfoundingScenario.EEG_MOVEMENT_ARTIFACTS,
    ]
    
    results = []
    
    print(f"\n🔬 Testing {len(hurst_values)} Hurst values × {len(data_lengths)} lengths × "
          f"{len(eeg_scenarios)} EEG scenarios × {len(estimators)} estimators")
    
    total_tests = len(hurst_values) * len(data_lengths) * (len(eeg_scenarios) + 1) * len(estimators) * 2  # +1 for pure, 2 data models
    current_test = 0
    
    for hurst in hurst_values:
        for length in data_lengths:
            for data_model in ['fbm', 'fgn']:
                # Generate test data
                if data_model == 'fbm':
                    model = FBMModel(H=hurst)
                    data = model.generate(n=length)
                else:
                    model = FGNModel(H=hurst)
                    data = model.generate(n=length)
                
                # Test pure data
                for estimator_name, estimator in estimators.items():
                    try:
                        start_time = time.time()
                        result = estimator.estimate(data)
                        execution_time = time.time() - start_time
                        
                        estimated_hurst = result.get('hurst_parameter', np.nan)
                        error = abs(estimated_hurst - hurst) if not np.isnan(estimated_hurst) else np.nan
                        success = not np.isnan(estimated_hurst) and 0 < estimated_hurst < 1
                        
                        results.append({
                            'estimator': estimator_name,
                            'data_model': data_model,
                            'hurst': hurst,
                            'length': length,
                            'scenario': 'pure',
                            'estimated_hurst': estimated_hurst,
                            'error': error,
                            'success': success,
                            'execution_time': execution_time,
                            'framework': result.get('framework_used', 'unknown')
                        })
                        
                    except Exception as e:
                        results.append({
                            'estimator': estimator_name,
                            'data_model': data_model,
                            'hurst': hurst,
                            'length': length,
                            'scenario': 'pure',
                            'estimated_hurst': np.nan,
                            'error': np.nan,
                            'success': False,
                            'execution_time': 0,
                            'framework': 'failed'
                        })
                    
                    current_test += 1
                    if current_test % 10 == 0:
                        print(f"  Progress: {current_test}/{total_tests} tests")
                
                # Test contaminated data
                for scenario in eeg_scenarios:
                    try:
                        # Apply EEG contamination
                        contaminated_data, description = contamination_factory.apply_confounding(
                            data, scenario, intensity=1.0
                        )
                        
                        for estimator_name, estimator in estimators.items():
                            try:
                                start_time = time.time()
                                result = estimator.estimate(contaminated_data)
                                execution_time = time.time() - start_time
                                
                                estimated_hurst = result.get('hurst_parameter', np.nan)
                                error = abs(estimated_hurst - hurst) if not np.isnan(estimated_hurst) else np.nan
                                success = not np.isnan(estimated_hurst) and 0 < estimated_hurst < 1
                                
                                results.append({
                                    'estimator': estimator_name,
                                    'data_model': data_model,
                                    'hurst': hurst,
                                    'length': length,
                                    'scenario': scenario.value,
                                    'estimated_hurst': estimated_hurst,
                                    'error': error,
                                    'success': success,
                                    'execution_time': execution_time,
                                    'framework': result.get('framework_used', 'unknown')
                                })
                                
                            except Exception as e:
                                results.append({
                                    'estimator': estimator_name,
                                    'data_model': data_model,
                                    'hurst': hurst,
                                    'length': length,
                                    'scenario': scenario.value,
                                    'estimated_hurst': np.nan,
                                    'error': np.nan,
                                    'success': False,
                                    'execution_time': 0,
                                    'framework': 'failed'
                                })
                            
                            current_test += 1
                            if current_test % 10 == 0:
                                print(f"  Progress: {current_test}/{total_tests} tests")
                                
                    except Exception as e:
                        print(f"  Warning: Failed to apply {scenario.value}: {e}")
    
    # Analyze results
    df = pd.DataFrame(results)
    
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 60)
    
    # Overall performance
    total_tests = len(df)
    successful_tests = df['success'].sum()
    success_rate = (successful_tests / total_tests) * 100
    
    print(f"Total Tests: {total_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Mean Error: {df['error'].mean():.4f}")
    print(f"Mean Execution Time: {df['execution_time'].mean():.4f}s")
    
    # Pure vs Contaminated
    pure_data = df[df['scenario'] == 'pure']
    contaminated_data = df[df['scenario'] != 'pure']
    
    print(f"\n🔬 Pure Data Performance:")
    print(f"  Success Rate: {(pure_data['success'].sum() / len(pure_data) * 100):.1f}%")
    print(f"  Mean Error: {pure_data['error'].mean():.4f}")
    print(f"  Mean Execution Time: {pure_data['execution_time'].mean():.4f}s")
    
    print(f"\n🧠 Contaminated Data Performance:")
    print(f"  Success Rate: {(contaminated_data['success'].sum() / len(contaminated_data) * 100):.1f}%")
    print(f"  Mean Error: {contaminated_data['error'].mean():.4f}")
    print(f"  Mean Execution Time: {contaminated_data['execution_time'].mean():.4f}s")
    
    # Framework usage
    print(f"\n⚡ Framework Usage:")
    framework_counts = df['framework'].value_counts()
    for framework, count in framework_counts.items():
        percentage = (count / total_tests) * 100
        print(f"  {framework}: {count} ({percentage:.1f}%)")
    
    # Estimator performance
    print(f"\n🎯 Estimator Performance:")
    for estimator in df['estimator'].unique():
        estimator_data = df[df['estimator'] == estimator]
        estimator_success = estimator_data['success'].sum()
        estimator_total = len(estimator_data)
        estimator_success_rate = (estimator_success / estimator_total) * 100
        estimator_error = estimator_data['error'].mean()
        print(f"  {estimator}: {estimator_success_rate:.1f}% success, "
              f"mean error: {estimator_error:.4f}")
    
    # EEG scenario performance
    print(f"\n🧠 EEG Scenario Performance:")
    for scenario in df['scenario'].unique():
        if scenario != 'pure':
            scenario_data = df[df['scenario'] == scenario]
            scenario_success = scenario_data['success'].sum()
            scenario_total = len(scenario_data)
            scenario_success_rate = (scenario_success / scenario_total) * 100
            scenario_error = scenario_data['error'].mean()
            print(f"  {scenario}: {scenario_success_rate:.1f}% success, "
                  f"mean error: {scenario_error:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("demo_results", exist_ok=True)
    
    csv_file = f"demo_results/quick_classical_eeg_demo_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Results saved to: {csv_file}")
    
    # Create summary
    summary = {
        'overall': {
            'total_tests': total_tests,
            'success_rate': success_rate,
            'mean_error': float(df['error'].mean()),
            'mean_execution_time': float(df['execution_time'].mean())
        },
        'pure_data': {
            'success_rate': float(pure_data['success'].sum() / len(pure_data) * 100),
            'mean_error': float(pure_data['error'].mean()),
            'mean_execution_time': float(pure_data['execution_time'].mean())
        },
        'contaminated_data': {
            'success_rate': float(contaminated_data['success'].sum() / len(contaminated_data) * 100),
            'mean_error': float(contaminated_data['error'].mean()),
            'mean_execution_time': float(contaminated_data['execution_time'].mean())
        },
        'framework_usage': framework_counts.to_dict(),
        'estimator_performance': {},
        'eeg_scenario_performance': {}
    }
    
    # Add estimator performance
    for estimator in df['estimator'].unique():
        estimator_data = df[df['estimator'] == estimator]
        summary['estimator_performance'][estimator] = {
            'success_rate': float(estimator_data['success'].sum() / len(estimator_data) * 100),
            'mean_error': float(estimator_data['error'].mean()),
            'mean_execution_time': float(estimator_data['execution_time'].mean())
        }
    
    # Add EEG scenario performance
    for scenario in df['scenario'].unique():
        if scenario != 'pure':
            scenario_data = df[df['scenario'] == scenario]
            summary['eeg_scenario_performance'][scenario] = {
                'success_rate': float(scenario_data['success'].sum() / len(scenario_data) * 100),
                'mean_error': float(scenario_data['error'].mean()),
                'mean_execution_time': float(scenario_data['execution_time'].mean())
            }
    
    summary_file = f"demo_results/quick_classical_eeg_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"💾 Summary saved to: {summary_file}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📈 This demonstrates the adaptive classical estimators working with EEG contamination")
    print(f"🔬 The full benchmark is running in the background for comprehensive results")


if __name__ == "__main__":
    run_quick_demo()
