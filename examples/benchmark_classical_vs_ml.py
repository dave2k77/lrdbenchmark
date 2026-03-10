#!/usr/bin/env python3
"""
Benchmark Script: Classical vs Machine Learning Estimators

Compares established Classical LRD Estimators against newly trained
Machine Learning Estimators (RF, CNN, LSTM) on simulated Fractional Brownian Motion.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lrdbenchmark.generation.time_series_generator import TimeSeriesGenerator

# Import Classical Estimators
from lrdbenchmark.analysis.spectral.whittle_estimator import WhittleEstimator
from lrdbenchmark.analysis.temporal.dfa_estimator import DFAEstimator
from lrdbenchmark.analysis.wavelet.cwt_estimator import CWTEstimator

# Import ML Estimators
from lrdbenchmark.models.pretrained_models.ml_pretrained import RandomForestPretrainedModel
from lrdbenchmark.models.pretrained_models.cnn_pretrained import CNNPretrainedModel
from lrdbenchmark.models.pretrained_models.lstm_pretrained import LSTMPretrainedModel

def run_benchmark():
    n_series = 200
    length = 1024
    
    print(f"Initializing Benchmark: Classical vs. ML Estimators over {n_series} clean and {n_series} contaminated fGn simulations...")
    
    generator = TimeSeriesGenerator(random_state=42)
    rng = np.random.default_rng(42)
    true_hs = rng.uniform(0.1, 0.9, size=n_series)
    
    estimators = {
        "Whittle (Classical Spectral)": WhittleEstimator(),
        "DFA (Classical Temporal)": DFAEstimator(),
        "CWT (Classical Wavelet)": CWTEstimator(),
        "Random Forest (ML)": RandomForestPretrainedModel(),
        "CNN (Neural)": CNNPretrainedModel(input_length=length),
        "LSTM (Neural)": LSTMPretrainedModel(input_length=length)
    }
    
    results = {
        'clean': {name: [] for name in estimators.keys()},
        'contam': {name: [] for name in estimators.keys()}
    }
    computation_times = {
        'clean': {name: 0.0 for name in estimators.keys()},
        'contam': {name: 0.0 for name in estimators.keys()}
    }
    
    # Generate data
    dataset_clean = []
    dataset_contam = []
    print("Generating pure fGn datasets (200 Clean, 200 Contaminated)...")
    for i, h in enumerate(tqdm(true_hs, desc="Generating")):
        
        # Clean Generation
        res_clean = generator.generate(
            model='fgn', 
            length=length, 
            params={'H': h}, 
            preprocess=True
        )
        signal_clean = res_clean['signal']
        signal_clean = (signal_clean - np.mean(signal_clean)) / (np.std(signal_clean) + 1e-8)
        dataset_clean.append((h, signal_clean))
        
        # Contaminated Generation
        contam_type = rng.choice([
            'physiological_sensor_drift', 
            'eeg_electrode_popping', 
            'mixed_realistic_moderate'
        ])
        contamination = [{'scenario': contam_type, 'intensity': 0.15}]
        res_contam = generator.generate(
            model='fgn', 
            length=length, 
            params={'H': h}, 
            contamination=contamination,
            preprocess=True
        )
        signal_contam = res_contam['signal']
        signal_contam = (signal_contam - np.mean(signal_contam)) / (np.std(signal_contam) + 1e-8)
        dataset_contam.append((h, signal_contam))
    
    # Execute Benchmark
    print("\nRunning Estimators on CLEAN Data...")
    for name, estimator in estimators.items():
        print(f"  Evaluating {name} (Clean)...")
        start_time = time.time()
        for h, signal in tqdm(dataset_clean, desc=f"{name} (Clean)", leave=False):
            try:
                estimate = estimator.estimate(signal)
                pred_h = estimate.get('hurst_parameter', np.nan)
                err = abs(pred_h - h)
                ci = estimate.get('confidence_interval')
                ci_hit = False
                if ci is not None and len(ci) == 2:
                    if ci[0] <= h <= ci[1]:
                        ci_hit = True
                results['clean'][name].append({'true_h': h, 'pred_h': pred_h, 'abs_err': err, 'ci_hit': ci_hit})
            except Exception as e:
                results['clean'][name].append({'true_h': h, 'pred_h': np.nan, 'abs_err': np.nan, 'ci_hit': False})
        computation_times['clean'][name] = time.time() - start_time
        
    print("\nRunning Estimators on CONTAMINATED Data...")
    for name, estimator in estimators.items():
        print(f"  Evaluating {name} (Contam)...")
        start_time = time.time()
        for h, signal in tqdm(dataset_contam, desc=f"{name} (Contam)", leave=False):
            try:
                estimate = estimator.estimate(signal)
                pred_h = estimate.get('hurst_parameter', np.nan)
                err = abs(pred_h - h)
                ci = estimate.get('confidence_interval')
                ci_hit = False
                if ci is not None and len(ci) == 2:
                    if ci[0] <= h <= ci[1]:
                        ci_hit = True
                results['contam'][name].append({'true_h': h, 'pred_h': pred_h, 'abs_err': err, 'ci_hit': ci_hit})
            except Exception as e:
                results['contam'][name].append({'true_h': h, 'pred_h': np.nan, 'abs_err': np.nan, 'ci_hit': False})
        computation_times['contam'][name] = time.time() - start_time
        
    def print_metrics(setting_name, output_dict, times_dict):
        print(f"\n\n Output Metrics: {setting_name.upper()} DATA")
        print("=" * 110)
        print(f"{'Estimator':<30} | {'MAE':<10} | {'MSE':<10} | {'Failures':<10} | {'Success Rate':<15} | {'CI Coverage':<15} | {'Time (s)':<10}")
        print("-" * 110)
        
        for name in estimators.keys():
            df = pd.DataFrame(output_dict[name])
            total_runs = len(df)
            failures = df['pred_h'].isna().sum()
            valid = df.dropna(subset=['pred_h', 'abs_err'])
            
            success_rate = ((total_runs - failures) / total_runs) * 100 if total_runs > 0 else 0
            
            if len(valid) > 0:
                mae = valid['abs_err'].mean()
                mse = (valid['abs_err'] ** 2).mean()
                ci_coverage = (valid['ci_hit'].sum() / len(valid)) * 100
            else:
                mae = np.nan
                mse = np.nan
                ci_coverage = 0.0
                
            t = times_dict[name]
            print(f"{name:<30} | {mae:<10.4f} | {mse:<10.4f} | {failures:<10} | {success_rate:>13.1f}% | {ci_coverage:>13.1f}% | {t:<10.2f}")
        print("=" * 110)

    print_metrics('clean', results['clean'], computation_times['clean'])
    print_metrics('contaminated', results['contam'], computation_times['contam'])
    print("Benchmark completed.")

if __name__ == "__main__":
    run_benchmark()
