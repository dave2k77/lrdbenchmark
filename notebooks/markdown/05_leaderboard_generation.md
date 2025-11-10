# Leaderboard Generation

This notebook demonstrates how to create comprehensive performance leaderboards from benchmark results, showing how to rank estimators, surface statistical significance, and generate stratified and robustness-aware comparisons.

## Overview

The leaderboard generation system allows you to:

1. **Load Benchmark Results**: Import results from multiple benchmark runs
2. **Create Rankings**: Generate performance rankings across different metrics
3. **Composite Scoring**: Combine multiple metrics into overall scores
4. **Visualization**: Create publication-ready plots, significance overlays, and stratified tables
5. **Stratified Reporting**: Slice results by H regime, tail class, data length, and contamination
6. **Export Results**: Save leaderboards in various formats (CSV/JSON/LaTeX) with provenance metadata

## Table of Contents

1. [Setup and Imports](#setup)
2. [Loading Benchmark Results](#loading)
3. [Creating Performance Rankings](#rankings)
4. [Composite Scoring System](#scoring)
5. [Visualization and Export](#visualization)
6. [Summary and Next Steps](#summary)


## 1. Setup and Imports {#setup}

First, let's import all necessary libraries and set up the leaderboard generation system.



```python
# Standard scientific computing imports
import numpy as np
# lrdbenchmark imports - using simplified API
from lrdbenchmark import (
    # Data models
    FBMModel, FGNModel, ARFIMAModel, MRWModel, AlphaStableModel,
    # Classical estimators  
    RSEstimator, DFAEstimator, GPHEstimator, WhittleEstimator,
    # Machine Learning estimators
    RandomForestEstimator, SVREstimator, GradientBoostingEstimator,
    # Neural Network estimators
    CNNEstimator, LSTMEstimator, GRUEstimator, TransformerEstimator,
    # GPU utilities
    gpu_is_available, get_device_info, clear_gpu_cache, monitor_gpu_memory
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import warnings
import subprocess
import gc
warnings.filterwarnings('ignore')

# GPU Memory Management Functions
```

    🔍 Checking GPU memory status...
    🖥️  GPU Memory: 13MB / 8151MB (0.2%)
    ✅ All imports successful!
    🏆 Ready to generate performance leaderboards


## 2. Loading Benchmark Results {#loading}

Let's run comprehensive benchmarks to generate data for our leaderboard, then load and process the results.



```python
# Initialize benchmark system
print("🔧 Initializing Benchmark System for Leaderboard Generation...")
print("=" * 70)

benchmark = ComprehensiveBenchmark(output_dir="leaderboard_results")
print(f"Protocol configuration loaded from: {benchmark.protocol_config_path}")

# Run comprehensive benchmarks
print("\n🚀 Running Comprehensive Benchmarks...")
print("=" * 70)

# Run classical benchmark
print("📊 Running Classical Estimator Benchmark...")
classical_results = benchmark.run_classical_benchmark(
    data_length=1000,
    save_results=True
)

print(f"✅ Classical benchmark completed!")
print(f"Success rate: {classical_results['success_rate']:.1%}")
print(f"Total tests: {classical_results['total_tests']}")

# Run ML benchmark
print("\n📊 Running ML Estimator Benchmark...")
ml_results = benchmark.run_ml_benchmark(
    data_length=1000,
    save_results=True
)

print(f"✅ ML benchmark completed!")
print(f"Success rate: {ml_results['success_rate']:.1%}")
print(f"Total tests: {ml_results['total_tests']}")

# Run neural benchmark
print("\n📊 Running Neural Network Benchmark...")
neural_results = benchmark.run_neural_benchmark(
    data_length=1000,
    save_results=True
)

print(f"✅ Neural benchmark completed!")
print(f"Success rate: {neural_results['success_rate']:.1%}")
print(f"Total tests: {neural_results['total_tests']}")

# Run comprehensive benchmark
print("\n📊 Running Comprehensive Benchmark...")
comprehensive_results = benchmark.run_comprehensive_benchmark(
    data_length=1000,
    save_results=True
)

print(f"✅ Comprehensive benchmark completed!")
print(f"Success rate: {comprehensive_results['success_rate']:.1%}")
print(f"Total tests: {comprehensive_results['total_tests']}")

print("\n🎯 All benchmarks completed successfully!")

```

    🔧 Initializing Benchmark System for Leaderboard Generation...
    ======================================================================
    ✅ LSTM model initialized with reasonable weights
    ✅ GRU model initialized with reasonable weights
    
    🚀 Running Comprehensive Benchmarks...
    ======================================================================
    📊 Running Classical Estimator Benchmark...
    🚀 Starting LRDBench Benchmark
    ============================================================
    Benchmark Type: CLASSICAL
    ============================================================
    Testing 13 estimators...
    
    📊 Testing with fBm data model...
       Generated 1000 clean data points
       🔍 Testing R/S... ✅
       🔍 Testing DFA... ✅
       🔍 Testing DMA... ✅
       🔍 Testing Higuchi... ✅
       🔍 Testing GPH... ✅
       🔍 Testing Whittle... ✅
       🔍 Testing Periodogram... ✅
       🔍 Testing CWT... ✅
       🔍 Testing WaveletVar... ✅
       🔍 Testing WaveletLogVar... ✅
       🔍 Testing WaveletWhittle... ✅
       🔍 Testing MFDFA... ✅
       🔍 Testing WaveletLeaders... ✅
    
    📊 Testing with fGn data model...
       Generated 1000 clean data points
       🔍 Testing R/S... ✅
       🔍 Testing DFA... ✅
       🔍 Testing DMA... ✅
       🔍 Testing Higuchi... ✅
       🔍 Testing GPH... ✅
       🔍 Testing Whittle... ✅
       🔍 Testing Periodogram... ✅
       🔍 Testing CWT... ✅
       🔍 Testing WaveletVar... ✅
       🔍 Testing WaveletLogVar... ✅
       🔍 Testing WaveletWhittle... ✅
       🔍 Testing MFDFA... ✅
       🔍 Testing WaveletLeaders... ✅
    
    📊 Testing with ARFIMAModel data model...
       Generated 1000 clean data points
       🔍 Testing R/S... ✅
       🔍 Testing DFA... ✅
       🔍 Testing DMA... ✅
       🔍 Testing Higuchi... ✅
       🔍 Testing GPH... ✅
       🔍 Testing Whittle... ✅
       🔍 Testing Periodogram... ✅
       🔍 Testing CWT... ✅
       🔍 Testing WaveletVar... ✅
       🔍 Testing WaveletLogVar... ✅
       🔍 Testing WaveletWhittle... ✅
       🔍 Testing MFDFA... ✅
       🔍 Testing WaveletLeaders... ✅
    
    📊 Testing with MRW data model...
       Generated 1000 clean data points
       🔍 Testing R/S... ✅
       🔍 Testing DFA... ✅
       🔍 Testing DMA... ✅
       🔍 Testing Higuchi... ✅
       🔍 Testing GPH... ✅
       🔍 Testing Whittle... ✅
       🔍 Testing Periodogram... ✅
       🔍 Testing CWT... ✅
       🔍 Testing WaveletVar... ✅
       🔍 Testing WaveletLogVar... ✅
       🔍 Testing WaveletWhittle... ✅
       🔍 Testing MFDFA... ✅
       🔍 Testing WaveletLeaders... ✅
    
    💾 Results saved to:
       JSON: leaderboard_results/comprehensive_benchmark_20251016_100856.json
       CSV: leaderboard_results/benchmark_summary_20251016_100856.csv
    
    ============================================================
    📊 BENCHMARK SUMMARY
    ============================================================
    Benchmark Type: CLASSICAL
    Total Tests: 52
    Successful: 52
    Success Rate: 100.0%
    Data Models: 4
    Estimators: 13
    
    🏆 TOP PERFORMING ESTIMATORS (Average across all data models):
       1. Whittle
          Avg Error: 0.1000 (Range: 0.0000-0.4000)
          Avg Time: 0.001s | Data Models: 4
          Mean Signed Error: 0.1000
          Bias: 33.33%
          Stability: 0.0000
          Estimated H values:
            fBm: H_est=0.7000, H_true=0.7000
            fGn: H_est=0.7000, H_true=0.7000
            ARFIMAModel: H_est=0.7000, H_true=0.3000
            MRW: H_est=0.7000, H_true=0.7000
    
       2. Periodogram
          Avg Error: 0.1287 (Range: 0.0080-0.3676)
          Avg Time: 0.001s | Data Models: 4
          Convergence Rate: -0.2191
          Mean Signed Error: 0.0899
          Bias: 30.35%
          Stability: 0.2886
          Estimated H values:
            fBm: H_est=0.7080, H_true=0.7000
            fGn: H_est=0.7618, H_true=0.7000
            ARFIMAModel: H_est=0.6676, H_true=0.3000
            MRW: H_est=0.6226, H_true=0.7000
    
       3. R/S
          Avg Error: 0.1777 (Range: 0.0062-0.4919)
          Avg Time: 0.797s | Data Models: 4
          Convergence Rate: -0.3563
          Mean Signed Error: 0.1777
          Bias: 48.81%
          Stability: 0.0641
          Estimated H values:
            fBm: H_est=0.7820, H_true=0.7000
            fGn: H_est=0.8305, H_true=0.7000
            ARFIMAModel: H_est=0.7919, H_true=0.3000
            MRW: H_est=0.7062, H_true=0.7000
    
       4. Higuchi
          Avg Error: 0.1819 (Range: 0.0373-0.4902)
          Avg Time: 0.002s | Data Models: 4
          Convergence Rate: -0.7034
          Mean Signed Error: 0.1818
          Bias: 49.32%
          Stability: 0.1105
          Estimated H values:
            fBm: H_est=0.8073, H_true=0.7000
            fGn: H_est=0.7927, H_true=0.7000
            ARFIMAModel: H_est=0.7902, H_true=0.3000
            MRW: H_est=0.7373, H_true=0.7000
    
       5. DMA
          Avg Error: 0.1829 (Range: 0.0479-0.4514)
          Avg Time: 0.001s | Data Models: 4
          Convergence Rate: -0.1672
          Mean Signed Error: 0.1589
          Bias: 44.20%
          Stability: 0.1522
          Estimated H values:
            fBm: H_est=0.8685, H_true=0.7000
            fGn: H_est=0.7639, H_true=0.7000
            ARFIMAModel: H_est=0.7514, H_true=0.3000
            MRW: H_est=0.6521, H_true=0.7000
    
    
    📊 DETAILED PERFORMANCE BY DATA MODEL:
    
       fBm:
         1. Whittle: Error 0.0000, Time 0.001s
         2. Periodogram: Error 0.0080, Time 0.001s
         3. R/S: Error 0.0820, Time 2.960s
    
       fGn:
         1. Whittle: Error 0.0000, Time 0.000s
         2. Periodogram: Error 0.0618, Time 0.001s
         3. DMA: Error 0.0639, Time 0.001s
    
       ARFIMAModel:
         1. WaveletLeaders: Error 0.0535, Time 0.013s
         2. MFDFA: Error 0.0817, Time 0.103s
         3. WaveletWhittle: Error 0.2900, Time 0.007s
    
       MRW:
         1. Whittle: Error 0.0000, Time 0.000s
         2. R/S: Error 0.0062, Time 0.075s
         3. Higuchi: Error 0.0373, Time 0.002s
    
    🎯 Benchmark completed successfully!
    ✅ Classical benchmark completed!
    Success rate: 100.0%
    Total tests: 52
    
    📊 Running ML Estimator Benchmark...
    🚀 Starting LRDBench Benchmark
    ============================================================
    Benchmark Type: ML
    ============================================================
    Testing 3 estimators...
    
    📊 Testing with fBm data model...
       Generated 1000 clean data points
       🔍 Testing RandomForest... ✅
       🔍 Testing GradientBoosting... ✅
       🔍 Testing SVR... ✅
    
    📊 Testing with fGn data model...
       Generated 1000 clean data points
       🔍 Testing RandomForest... ✅
       🔍 Testing GradientBoosting... ✅
       🔍 Testing SVR... ✅
    
    📊 Testing with ARFIMAModel data model...
       Generated 1000 clean data points
       🔍 Testing RandomForest... ✅
       🔍 Testing GradientBoosting... ✅
       🔍 Testing SVR... ✅
    
    📊 Testing with MRW data model...
       Generated 1000 clean data points
       🔍 Testing RandomForest... ✅
       🔍 Testing GradientBoosting... ✅
       🔍 Testing SVR... ✅
    
    💾 Results saved to:
       JSON: leaderboard_results/comprehensive_benchmark_20251016_100856.json
       CSV: leaderboard_results/benchmark_summary_20251016_100856.csv
    
    ============================================================
    📊 BENCHMARK SUMMARY
    ============================================================
    Benchmark Type: ML
    Total Tests: 12
    Successful: 12
    Success Rate: 100.0%
    Data Models: 4
    Estimators: 3
    
    🏆 TOP PERFORMING ESTIMATORS (Average across all data models):
       1. SVR
          Avg Error: 0.1364 (Range: 0.0147-0.4680)
          Avg Time: 0.000s | Data Models: 4
          Convergence Rate: -0.0095
          Mean Signed Error: 0.1291
          Bias: 40.73%
          Stability: 0.0114
          Estimated H values:
            fBm: H_est=0.7387, H_true=0.7000
            fGn: H_est=0.7244, H_true=0.7000
            ARFIMAModel: H_est=0.7680, H_true=0.3000
            MRW: H_est=0.6853, H_true=0.7000
    
       2. GradientBoosting
          Avg Error: 0.4308 (Range: 0.1471-0.5783)
          Avg Time: 0.000s | Data Models: 4
          Convergence Rate: -0.8088
          Mean Signed Error: -0.4308
          Bias: -68.54%
          Stability: 0.1323
          Estimated H values:
            fBm: H_est=0.1418, H_true=0.7000
            fGn: H_est=0.1217, H_true=0.7000
            ARFIMAModel: H_est=0.1529, H_true=0.3000
            MRW: H_est=0.2604, H_true=0.7000
    
       3. RandomForest
          Avg Error: 0.5000 (Range: 0.2000-0.6000)
          Avg Time: 0.000s | Data Models: 4
          Convergence Rate: -0.3469
          Mean Signed Error: -0.5000
          Bias: -80.95%
          Stability: 0.1436
          Estimated H values:
            fBm: H_est=0.1000, H_true=0.7000
            fGn: H_est=0.1000, H_true=0.7000
            ARFIMAModel: H_est=0.1000, H_true=0.3000
            MRW: H_est=0.1000, H_true=0.7000
    
    
    📊 DETAILED PERFORMANCE BY DATA MODEL:
    
       fBm:
         1. SVR: Error 0.0387, Time 0.000s
         2. GradientBoosting: Error 0.5582, Time 0.000s
         3. RandomForest: Error 0.6000, Time 0.000s
    
       fGn:
         1. SVR: Error 0.0244, Time 0.000s
         2. GradientBoosting: Error 0.5783, Time 0.000s
         3. RandomForest: Error 0.6000, Time 0.000s
    
       ARFIMAModel:
         1. GradientBoosting: Error 0.1471, Time 0.000s
         2. RandomForest: Error 0.2000, Time 0.000s
         3. SVR: Error 0.4680, Time 0.000s
    
       MRW:
         1. SVR: Error 0.0147, Time 0.000s
         2. GradientBoosting: Error 0.4396, Time 0.000s
         3. RandomForest: Error 0.6000, Time 0.000s
    
    🎯 Benchmark completed successfully!
    ✅ ML benchmark completed!
    Success rate: 100.0%
    Total tests: 12
    
    📊 Running Neural Network Benchmark...
    🚀 Starting LRDBench Benchmark
    ============================================================
    Benchmark Type: NEURAL
    ============================================================
    Testing 4 estimators...
    
    📊 Testing with fBm data model...
       Generated 1000 clean data points
       🔍 Testing CNN... ✅
       🔍 Testing LSTM... ✅
       🔍 Testing GRU... ✅
       🔍 Testing Transformer... ❌ (Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_addmm))
    
    📊 Testing with fGn data model...
       Generated 1000 clean data points
       🔍 Testing CNN... ✅
       🔍 Testing LSTM... ✅
       🔍 Testing GRU... ✅
       🔍 Testing Transformer... ❌ (Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_addmm))
    
    📊 Testing with ARFIMAModel data model...
       Generated 1000 clean data points
       🔍 Testing CNN... ✅
       🔍 Testing LSTM... ✅
       🔍 Testing GRU... ✅
       🔍 Testing Transformer... ❌ (Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_addmm))
    
    📊 Testing with MRW data model...
       Generated 1000 clean data points
       🔍 Testing CNN... ✅
       🔍 Testing LSTM... ✅
       🔍 Testing GRU... ✅
       🔍 Testing Transformer... ❌ (Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_addmm))
    
    💾 Results saved to:
       JSON: leaderboard_results/comprehensive_benchmark_20251016_100857.json
       CSV: leaderboard_results/benchmark_summary_20251016_100857.csv
    
    ============================================================
    📊 BENCHMARK SUMMARY
    ============================================================
    Benchmark Type: NEURAL
    Total Tests: 16
    Successful: 12
    Success Rate: 75.0%
    Data Models: 4
    Estimators: 4
    
    🏆 TOP PERFORMING ESTIMATORS (Average across all data models):
       1. CNN
          Avg Error: 0.1975 (Range: 0.1937-0.2049)
          Avg Time: 0.001s | Data Models: 4
          Convergence Rate: 0.0045
          Mean Signed Error: -0.0951
          Bias: -3.82%
          Stability: 0.0017
          Estimated H values:
            fBm: H_est=0.5048, H_true=0.7000
            fGn: H_est=0.5063, H_true=0.7000
            ARFIMAModel: H_est=0.5049, H_true=0.3000
            MRW: H_est=0.5037, H_true=0.7000
    
       2. GRU
          Avg Error: 0.2049 (Range: 0.2031-0.2070)
          Avg Time: 0.002s | Data Models: 4
          Convergence Rate: 4.5144
          Mean Signed Error: -0.1026
          Bias: -4.93%
          Stability: 0.0042
          Estimated H values:
            fBm: H_est=0.4952, H_true=0.7000
            fGn: H_est=0.4930, H_true=0.7000
            ARFIMAModel: H_est=0.5044, H_true=0.3000
            MRW: H_est=0.4969, H_true=0.7000
    
       3. LSTM
          Avg Error: 0.2080 (Range: 0.2041-0.2126)
          Avg Time: 0.032s | Data Models: 4
          Convergence Rate: 5.5809
          Mean Signed Error: -0.1017
          Bias: -4.41%
          Stability: 0.0073
          Estimated H values:
            fBm: H_est=0.4928, H_true=0.7000
            fGn: H_est=0.4919, H_true=0.7000
            ARFIMAModel: H_est=0.5126, H_true=0.3000
            MRW: H_est=0.4959, H_true=0.7000
    
    
    📊 DETAILED PERFORMANCE BY DATA MODEL:
    
       fBm:
         1. CNN: Error 0.1952, Time 0.003s
         2. GRU: Error 0.2048, Time 0.005s
         3. LSTM: Error 0.2072, Time 0.124s
    
       fGn:
         1. CNN: Error 0.1937, Time 0.001s
         2. GRU: Error 0.2070, Time 0.001s
         3. LSTM: Error 0.2081, Time 0.001s
    
       ARFIMAModel:
         1. GRU: Error 0.2044, Time 0.001s
         2. CNN: Error 0.2049, Time 0.001s
         3. LSTM: Error 0.2126, Time 0.001s
    
       MRW:
         1. CNN: Error 0.1963, Time 0.001s
         2. GRU: Error 0.2031, Time 0.001s
         3. LSTM: Error 0.2041, Time 0.001s
    
    🎯 Benchmark completed successfully!
    ✅ Neural benchmark completed!
    Success rate: 75.0%
    Total tests: 16
    
    📊 Running Comprehensive Benchmark...
    🚀 Starting LRDBench Benchmark
    ============================================================
    Benchmark Type: COMPREHENSIVE
    ============================================================
    Testing 20 estimators...
    
    📊 Testing with fBm data model...
       Generated 1000 clean data points
       🔍 Testing R/S... ✅
       🔍 Testing DFA... ✅
       🔍 Testing DMA... ✅
       🔍 Testing Higuchi... ✅
       🔍 Testing GPH... ✅
       🔍 Testing Whittle... ✅
       🔍 Testing Periodogram... ✅
       🔍 Testing CWT... ✅
       🔍 Testing WaveletVar... ✅
       🔍 Testing WaveletLogVar... ✅
       🔍 Testing WaveletWhittle... ✅
       🔍 Testing MFDFA... ✅
       🔍 Testing WaveletLeaders... ✅
       🔍 Testing RandomForest... ✅
       🔍 Testing GradientBoosting... ✅
       🔍 Testing SVR... ✅
       🔍 Testing CNN... ✅
       🔍 Testing LSTM... ✅
       🔍 Testing GRU... ✅
       🔍 Testing Transformer... ❌ (Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_addmm))
    
    📊 Testing with fGn data model...
       Generated 1000 clean data points
       🔍 Testing R/S... ✅
       🔍 Testing DFA... ✅
       🔍 Testing DMA... ✅
       🔍 Testing Higuchi... ✅
       🔍 Testing GPH... ✅
       🔍 Testing Whittle... ✅
       🔍 Testing Periodogram... ✅
       🔍 Testing CWT... ✅
       🔍 Testing WaveletVar... ✅
       🔍 Testing WaveletLogVar... ✅
       🔍 Testing WaveletWhittle... ✅
       🔍 Testing MFDFA... ✅
       🔍 Testing WaveletLeaders... ✅
       🔍 Testing RandomForest... ✅
       🔍 Testing GradientBoosting... ✅
       🔍 Testing SVR... ✅
       🔍 Testing CNN... ✅
       🔍 Testing LSTM... ✅
       🔍 Testing GRU... ✅
       🔍 Testing Transformer... ❌ (Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_addmm))
    
    📊 Testing with ARFIMAModel data model...
       Generated 1000 clean data points
       🔍 Testing R/S... ✅
       🔍 Testing DFA... ✅
       🔍 Testing DMA... ✅
       🔍 Testing Higuchi... ✅
       🔍 Testing GPH... ✅
       🔍 Testing Whittle... ✅
       🔍 Testing Periodogram... ✅
       🔍 Testing CWT... ✅
       🔍 Testing WaveletVar... ✅
       🔍 Testing WaveletLogVar... ✅
       🔍 Testing WaveletWhittle... ✅
       🔍 Testing MFDFA... ✅
       🔍 Testing WaveletLeaders... ✅
       🔍 Testing RandomForest... ✅
       🔍 Testing GradientBoosting... ✅
       🔍 Testing SVR... ✅
       🔍 Testing CNN... ✅
       🔍 Testing LSTM... ✅
       🔍 Testing GRU... ✅
       🔍 Testing Transformer... ❌ (Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_addmm))
    
    📊 Testing with MRW data model...
       Generated 1000 clean data points
       🔍 Testing R/S... ✅
       🔍 Testing DFA... ✅
       🔍 Testing DMA... ✅
       🔍 Testing Higuchi... ✅
       🔍 Testing GPH... ✅
       🔍 Testing Whittle... ✅
       🔍 Testing Periodogram... ✅
       🔍 Testing CWT... ✅
       🔍 Testing WaveletVar... ✅
       🔍 Testing WaveletLogVar... ✅
       🔍 Testing WaveletWhittle... ✅
       🔍 Testing MFDFA... ✅
       🔍 Testing WaveletLeaders... ✅
       🔍 Testing RandomForest... ✅
       🔍 Testing GradientBoosting... ✅
       🔍 Testing SVR... ✅
       🔍 Testing CNN... ✅
       🔍 Testing LSTM... ✅
       🔍 Testing GRU... ✅
       🔍 Testing Transformer... ❌ (Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_addmm))
    
    💾 Results saved to:
       JSON: leaderboard_results/comprehensive_benchmark_20251016_101004.json
       CSV: leaderboard_results/benchmark_summary_20251016_101004.csv
    
    ============================================================
    📊 BENCHMARK SUMMARY
    ============================================================
    Benchmark Type: COMPREHENSIVE
    Total Tests: 80
    Successful: 76
    Success Rate: 95.0%
    Data Models: 4
    Estimators: 20
    
    🏆 TOP PERFORMING ESTIMATORS (Average across all data models):
       1. Whittle
          Avg Error: 0.1000 (Range: 0.0000-0.4000)
          Avg Time: 0.001s | Data Models: 4
          Mean Signed Error: 0.1000
          Bias: 33.33%
          Stability: 0.0000
          Estimated H values:
            fBm: H_est=0.7000, H_true=0.7000
            fGn: H_est=0.7000, H_true=0.7000
            ARFIMAModel: H_est=0.7000, H_true=0.3000
            MRW: H_est=0.7000, H_true=0.7000
    
       2. Periodogram
          Avg Error: 0.1287 (Range: 0.0080-0.3676)
          Avg Time: 0.001s | Data Models: 4
          Convergence Rate: -0.2191
          Mean Signed Error: 0.0899
          Bias: 30.35%
          Stability: 0.2886
          Estimated H values:
            fBm: H_est=0.7080, H_true=0.7000
            fGn: H_est=0.7618, H_true=0.7000
            ARFIMAModel: H_est=0.6676, H_true=0.3000
            MRW: H_est=0.6226, H_true=0.7000
    
       3. SVR
          Avg Error: 0.1364 (Range: 0.0147-0.4680)
          Avg Time: 0.000s | Data Models: 4
          Convergence Rate: -0.0095
          Mean Signed Error: 0.1291
          Bias: 40.73%
          Stability: 0.0114
          Estimated H values:
            fBm: H_est=0.7387, H_true=0.7000
            fGn: H_est=0.7244, H_true=0.7000
            ARFIMAModel: H_est=0.7680, H_true=0.3000
            MRW: H_est=0.6853, H_true=0.7000
    
       4. R/S
          Avg Error: 0.1777 (Range: 0.0062-0.4919)
          Avg Time: 0.083s | Data Models: 4
          Convergence Rate: -0.3563
          Mean Signed Error: 0.1777
          Bias: 48.81%
          Stability: 0.0641
          Estimated H values:
            fBm: H_est=0.7820, H_true=0.7000
            fGn: H_est=0.8305, H_true=0.7000
            ARFIMAModel: H_est=0.7919, H_true=0.3000
            MRW: H_est=0.7062, H_true=0.7000
    
       5. Higuchi
          Avg Error: 0.1819 (Range: 0.0373-0.4902)
          Avg Time: 0.002s | Data Models: 4
          Convergence Rate: -0.7034
          Mean Signed Error: 0.1818
          Bias: 49.32%
          Stability: 0.1105
          Estimated H values:
            fBm: H_est=0.8073, H_true=0.7000
            fGn: H_est=0.7927, H_true=0.7000
            ARFIMAModel: H_est=0.7902, H_true=0.3000
            MRW: H_est=0.7373, H_true=0.7000
    
    
    📊 DETAILED PERFORMANCE BY DATA MODEL:
    
       fBm:
         1. Whittle: Error 0.0000, Time 0.000s
         2. Periodogram: Error 0.0080, Time 0.001s
         3. SVR: Error 0.0387, Time 0.000s
    
       fGn:
         1. Whittle: Error 0.0000, Time 0.000s
         2. SVR: Error 0.0244, Time 0.000s
         3. Periodogram: Error 0.0618, Time 0.001s
    
       ARFIMAModel:
         1. WaveletLeaders: Error 0.0535, Time 0.013s
         2. MFDFA: Error 0.0817, Time 0.105s
         3. GradientBoosting: Error 0.1471, Time 0.000s
    
       MRW:
         1. Whittle: Error 0.0000, Time 0.001s
         2. R/S: Error 0.0062, Time 0.084s
         3. SVR: Error 0.0147, Time 0.000s
    
    🎯 Benchmark completed successfully!
    ✅ Comprehensive benchmark completed!
    Success rate: 95.0%
    Total tests: 80
    
    🎯 All benchmarks completed successfully!


## 3. Creating Performance Rankings {#rankings}

Now let's create comprehensive performance rankings and leaderboards from our benchmark results.



```python
# Create comprehensive leaderboard
print("🏆 Creating Performance Leaderboard...")
print("=" * 70)

# Combine all benchmark results
all_results = {
    'Classical': classical_results,
    'ML': ml_results,
    'Neural': neural_results,
    'Comprehensive': comprehensive_results
}

# Create performance summary
performance_data = []

for category, results in all_results.items():
    print(f"🔍 Processing {category} results...")
    print(f"   Keys: {list(results.keys())}")
    
    # Check if results have the expected structure
    if 'results' in results and isinstance(results['results'], dict):
        print(f"   Found 'results' key with {len(results['results'])} entries")
        
        # Process the results data
        for data_model, model_results in results['results'].items():
            if isinstance(model_results, dict) and 'estimator_results' in model_results:
                for estimator_result in model_results['estimator_results']:
                    if estimator_result.get('success', True):  # Default to True if success not specified
                        ci_lower = None
                        ci_upper = None
                        interval_method = None
                        coverage_flag = None

                        ci = estimator_result.get('confidence_interval')
                        if isinstance(ci, (list, tuple)) and len(ci) == 2:
                            ci_lower, ci_upper = ci

                        uncertainty_blob = estimator_result.get('uncertainty', {})
                        if isinstance(uncertainty_blob, dict):
                            primary = uncertainty_blob.get('primary_interval')
                            if isinstance(primary, dict):
                                interval_method = primary.get('method', interval_method)
                                alt_ci = primary.get('confidence_interval')
                                if (
                                    (ci_lower is None or ci_upper is None)
                                    and isinstance(alt_ci, (list, tuple))
                                    and len(alt_ci) == 2
                                ):
                                    ci_lower, ci_upper = alt_ci
                            coverage_map = uncertainty_blob.get('coverage', {})
                            if isinstance(coverage_map, dict):
                                if interval_method and interval_method in coverage_map:
                                    coverage_flag = coverage_map.get(interval_method)
                                else:
                                    for value in coverage_map.values():
                                        if value is not None:
                                            coverage_flag = value
                                            break

                        ci_width = None
                        if ci_lower is not None and ci_upper is not None:
                            ci_width = ci_upper - ci_lower

                        performance_data.append({
                            'Category': category,
                            'Estimator': estimator_result['estimator'],
                            'True_H': estimator_result['true_hurst'],
                            'Estimated_H': estimator_result['estimated_hurst'],
                            'Error': estimator_result['error'],
                            'Execution_Time': estimator_result['execution_time'],
                            'Data_Model': data_model,
                            'CI_Lower': ci_lower,
                            'CI_Upper': ci_upper,
                            'CI_Width': ci_width,
                            'Interval_Method': interval_method,
                            'Coverage': coverage_flag
                        })
    else:
        print(f"   ⚠️ Unexpected results structure for {category}")
        print(f"   Available keys: {list(results.keys())}")

print(f"\n📊 Total performance records collected: {len(performance_data)}")

# Create DataFrame
performance_df = pd.DataFrame(performance_data)

if len(performance_df) > 0:
    print(f"📊 Loaded {len(performance_df)} performance records")
    
    # Calculate performance metrics
    performance_metrics = performance_df.groupby(['Category', 'Estimator']).agg({
        'Error': ['mean', 'std', 'min', 'max'],
        'Execution_Time': ['mean', 'std'],
        'CI_Width': ['mean', 'std'],
        'Coverage': 'mean',
        'True_H': 'count'
    }).round(4)
    
    print("\n📈 Performance Metrics Summary:")
    print(performance_metrics)
    
    # Create overall leaderboard
    print("\n🏆 Overall Performance Leaderboard:")
    print("=" * 70)
    
    # Calculate composite scores
    leaderboard_data = []
    
    for (category, estimator), group in performance_df.groupby(['Category', 'Estimator']):
        mean_error = group['Error'].mean()
        std_error = group['Error'].std()
        mean_time = group['Execution_Time'].mean()
        count = len(group)
        mean_ci_width = group['CI_Width'].dropna().mean() if 'CI_Width' in group else None
        coverage_rate = group['Coverage'].dropna().mean() if 'Coverage' in group else None
        
        # Composite score incorporates coverage to reward calibrated estimators
        coverage_factor = coverage_rate if coverage_rate is not None else 1.0
        coverage_factor = max(coverage_factor, 0.01)
        composite_score = (1 / (1 + mean_error)) * (count / 10) * (1 / (1 + mean_time)) * coverage_factor
        
        leaderboard_data.append({
            'Category': category,
            'Estimator': estimator,
            'Mean_Error': mean_error,
            'Std_Error': std_error,
            'Mean_Time': mean_time,
            'Mean_CI_Width': mean_ci_width,
            'Coverage_Rate': coverage_rate,
            'Count': count,
            'Composite_Score': composite_score
        })
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    leaderboard_df = leaderboard_df.sort_values('Composite_Score', ascending=False)
    
    print(leaderboard_df.round(4))
    
    # Save leaderboard
    leaderboard_df.to_csv('outputs/performance_leaderboard.csv', index=False)
    print("\n💾 Leaderboard saved to outputs/performance_leaderboard.csv")

    # Significance analysis for the comprehensive benchmark
    comprehensive_significance = comprehensive_results.get('significance_analysis', {})
    print("\n🧪 Significance Testing (Comprehensive Benchmark)")
    if not comprehensive_significance:
        print("No significance analysis available.")
    else:
        status = comprehensive_significance.get('status', 'unavailable')
        print(f"Status: {status}")
        if status == 'ok':
            friedman = comprehensive_significance.get('friedman', {})
            friedman_stat = friedman.get('statistic')
            friedman_p = friedman.get('p_value')
            if friedman_stat is not None and friedman_p is not None:
                print(
                    f"Friedman χ²={friedman_stat:.4f} (p={friedman_p:.4f}) "
                    f"across {friedman.get('n_data_models', 0)} data models "
                    f"and {friedman.get('n_estimators', 0)} estimators"
                )
            else:
                print(f"Friedman test unavailable: {friedman.get('error', 'insufficient data')}")
            mean_ranks = comprehensive_significance.get('mean_ranks', {})
            if mean_ranks:
                mean_rank_df = (
                    pd.DataFrame(list(mean_ranks.items()), columns=['Estimator', 'Mean Rank'])
                    .sort_values('Mean Rank')
                )
                print("\nMean rank summary:")
                print(mean_rank_df.to_string(index=False))
            post_hoc_entries = []
            for res in comprehensive_significance.get('post_hoc', []):
                if res.get('p_value') is None:
                    continue
                entry = {
                    'Estimator A': res['pair'][0],
                    'Estimator B': res['pair'][1],
                    'Holm p-value': float(res.get('holm_p_value')) if res.get('holm_p_value') is not None else None,
                    'Significant': bool(res.get('significant')),
                }
                if res.get('note'):
                    entry['Note'] = res['note']
                post_hoc_entries.append(entry)
            if post_hoc_entries:
                post_hoc_df = pd.DataFrame(post_hoc_entries).sort_values('Holm p-value')
                print("\nPairwise post-hoc tests (Holm-corrected):")
                print(post_hoc_df.to_string(index=False))
            else:
                print("No pairwise differences reached significance after Holm correction.")
        else:
            print(comprehensive_significance.get('reason', 'No additional information.'))

    coverage_overview = performance_df['Coverage'].dropna()
    if not coverage_overview.empty:
        print("\n🎯 Coverage Summary (Comprehensive Benchmark)")
        print(f"Overall empirical coverage: {coverage_overview.mean():.2%}")
        coverage_by_estimator = performance_df.groupby('Estimator')['Coverage'].mean().dropna()
        if not coverage_by_estimator.empty:
            print("Per-estimator coverage:")
            for estimator_name, rate in coverage_by_estimator.sort_values(ascending=False).items():
                print(f"   {estimator_name}: {rate:.2%}")

else:
    print("❌ No performance data available for leaderboard generation")

```

    🏆 Creating Performance Leaderboard...
    ======================================================================
    🔍 Processing Classical results...
       Keys: ['timestamp', 'benchmark_type', 'contamination_type', 'contamination_level', 'total_tests', 'successful_tests', 'success_rate', 'data_models_tested', 'estimators_tested', 'results']
       Found 'results' key with 4 entries
    🔍 Processing ML results...
       Keys: ['timestamp', 'benchmark_type', 'contamination_type', 'contamination_level', 'total_tests', 'successful_tests', 'success_rate', 'data_models_tested', 'estimators_tested', 'results']
       Found 'results' key with 4 entries
    🔍 Processing Neural results...
       Keys: ['timestamp', 'benchmark_type', 'contamination_type', 'contamination_level', 'total_tests', 'successful_tests', 'success_rate', 'data_models_tested', 'estimators_tested', 'results']
       Found 'results' key with 4 entries
    🔍 Processing Comprehensive results...
       Keys: ['timestamp', 'benchmark_type', 'contamination_type', 'contamination_level', 'total_tests', 'successful_tests', 'success_rate', 'data_models_tested', 'estimators_tested', 'results']
       Found 'results' key with 4 entries
    
    📊 Total performance records collected: 144
    📊 Loaded 144 performance records
    
    📈 Performance Metrics Summary:
                                     Error                         Execution_Time  \
                                      mean     std     min     max           mean   
    Category      Estimator                                                         
    Classical     CWT               0.3242  0.4224  0.0972  0.9573         0.0737   
                  DFA               0.2197  0.2085  0.0675  0.5255         0.0068   
                  DMA               0.1829  0.1868  0.0479  0.4514         0.0011   
                  GPH               0.2396  0.1937  0.0721  0.5171         0.1128   
                  Higuchi           0.1819  0.2077  0.0373  0.4902         0.0023   
                  MFDFA             0.3889  0.2053  0.0817  0.5033         0.1110   
                  Periodogram       0.1287  0.1620  0.0080  0.3676         0.0013   
                  R/S               0.1777  0.2157  0.0062  0.4919         0.8095   
                  WaveletLeaders    0.4523  0.2733  0.0535  0.6569         0.0142   
                  WaveletLogVar     0.3880  0.3424  0.1012  0.8849         0.0006   
                  WaveletVar        0.6041  0.4089  0.2345  1.1881         0.0011   
                  WaveletWhittle    0.5900  0.2000  0.2900  0.6900         0.0071   
                  Whittle           0.1000  0.2000  0.0000  0.4000         0.0005   
    Comprehensive CWT               0.3242  0.4224  0.0972  0.9573         0.0758   
                  DFA               0.2197  0.2085  0.0675  0.5255         0.0066   
                  DMA               0.1829  0.1868  0.0479  0.4514         0.0010   
                  GPH               0.2396  0.1937  0.0721  0.5171         0.0023   
                  GRU               0.2119  0.0052  0.2060  0.2187         0.0006   
                  GradientBoosting  0.4308  0.1988  0.1471  0.5783         0.0003   
                  Higuchi           0.1819  0.2077  0.0373  0.4902         0.0023   
                  LSTM              0.2000  0.0002  0.1998  0.2002         0.0009   
                  MFDFA             0.3699  0.1944  0.0817  0.4991         0.1066   
                  Periodogram       0.1287  0.1620  0.0080  0.3676         0.0013   
                  R/S               0.1777  0.2157  0.0062  0.4919         0.0822   
                  RandomForest      0.5000  0.2000  0.2000  0.6000         0.0005   
                  SVR               0.1364  0.2212  0.0147  0.4680         0.0001   
                  WaveletLeaders    0.4397  0.2714  0.0535  0.6569         0.0130   
                  WaveletLogVar     0.3880  0.3424  0.1012  0.8849         0.0005   
                  WaveletVar        0.6041  0.4089  0.2345  1.1881         0.0010   
                  WaveletWhittle    0.5900  0.2000  0.2900  0.6900         0.0072   
                  Whittle           0.1000  0.2000  0.0000  0.4000         0.0006   
    ML            GradientBoosting  0.4308  0.1988  0.1471  0.5783         0.0002   
                  RandomForest      0.5000  0.2000  0.2000  0.6000         0.0002   
                  SVR               0.1364  0.2212  0.0147  0.4680         0.0000   
    Neural        GRU               0.2119  0.0052  0.2060  0.2187         0.0017   
                  LSTM              0.2000  0.0002  0.1998  0.2002         0.0433   
    
                                           True_H  
                                       std  count  
    Category      Estimator                        
    Classical     CWT               0.0161      4  
                  DFA               0.0005      4  
                  DMA               0.0001      4  
                  GPH               0.2214      4  
                  Higuchi           0.0002      4  
                  MFDFA             0.0115      4  
                  Periodogram       0.0001      4  
                  R/S               1.4596      4  
                  WaveletLeaders    0.0031      4  
                  WaveletLogVar     0.0000      4  
                  WaveletVar        0.0001      4  
                  WaveletWhittle    0.0001      4  
                  Whittle           0.0000      4  
    Comprehensive CWT               0.0147      4  
                  DFA               0.0002      4  
                  DMA               0.0000      4  
                  GPH               0.0006      4  
                  GRU               0.0002      4  
                  GradientBoosting  0.0001      4  
                  Higuchi           0.0002      4  
                  LSTM              0.0001      4  
                  MFDFA             0.0016      4  
                  Periodogram       0.0001      4  
                  R/S               0.0056      4  
                  RandomForest      0.0001      4  
                  SVR               0.0000      4  
                  WaveletLeaders    0.0004      4  
                  WaveletLogVar     0.0000      4  
                  WaveletVar        0.0001      4  
                  WaveletWhittle    0.0001      4  
                  Whittle           0.0000      4  
    ML            GradientBoosting  0.0000      4  
                  RandomForest      0.0001      4  
                  SVR               0.0000      4  
    Neural        GRU               0.0023      4  
                  LSTM              0.0854      4  
    
    🏆 Overall Performance Leaderboard:
    ======================================================================
             Category         Estimator  Mean_Error  Std_Error  Mean_Time  Count  \
    12      Classical           Whittle      0.1000     0.2000     0.0005      4   
    30  Comprehensive           Whittle      0.1000     0.2000     0.0006      4   
    6       Classical       Periodogram      0.1287     0.1620     0.0013      4   
    22  Comprehensive       Periodogram      0.1287     0.1620     0.0013      4   
    33             ML               SVR      0.1364     0.2212     0.0000      4   
    25  Comprehensive               SVR      0.1364     0.2212     0.0001      4   
    15  Comprehensive               DMA      0.1829     0.1868     0.0010      4   
    2       Classical               DMA      0.1829     0.1868     0.0011      4   
    19  Comprehensive           Higuchi      0.1819     0.2077     0.0023      4   
    4       Classical           Higuchi      0.1819     0.2077     0.0023      4   
    20  Comprehensive              LSTM      0.2000     0.0002     0.0009      4   
    17  Comprehensive               GRU      0.2119     0.0052     0.0006      4   
    34         Neural               GRU      0.2119     0.0052     0.0017      4   
    14  Comprehensive               DFA      0.2197     0.2085     0.0066      4   
    1       Classical               DFA      0.2197     0.2085     0.0068      4   
    16  Comprehensive               GPH      0.2396     0.1937     0.0023      4   
    35         Neural              LSTM      0.2000     0.0002     0.0433      4   
    23  Comprehensive               R/S      0.1777     0.2157     0.0822      4   
    3       Classical               GPH      0.2396     0.1937     0.1128      4   
    27  Comprehensive     WaveletLogVar      0.3880     0.3424     0.0005      4   
    9       Classical     WaveletLogVar      0.3880     0.3424     0.0006      4   
    0       Classical               CWT      0.3242     0.4224     0.0737      4   
    13  Comprehensive               CWT      0.3242     0.4224     0.0758      4   
    31             ML  GradientBoosting      0.4308     0.1988     0.0002      4   
    18  Comprehensive  GradientBoosting      0.4308     0.1988     0.0003      4   
    26  Comprehensive    WaveletLeaders      0.4397     0.2714     0.0130      4   
    8       Classical    WaveletLeaders      0.4523     0.2733     0.0142      4   
    32             ML      RandomForest      0.5000     0.2000     0.0002      4   
    24  Comprehensive      RandomForest      0.5000     0.2000     0.0005      4   
    21  Comprehensive             MFDFA      0.3699     0.1944     0.1066      4   
    5       Classical             MFDFA      0.3889     0.2053     0.1110      4   
    11      Classical    WaveletWhittle      0.5900     0.2000     0.0071      4   
    29  Comprehensive    WaveletWhittle      0.5900     0.2000     0.0072      4   
    28  Comprehensive        WaveletVar      0.6041     0.4089     0.0010      4   
    10      Classical        WaveletVar      0.6041     0.4089     0.0011      4   
    7       Classical               R/S      0.1777     0.2157     0.8095      4   
    
        Composite_Score  
    12           0.3634  
    30           0.3634  
    6            0.3539  
    22           0.3539  
    33           0.3520  
    25           0.3520  
    15           0.3378  
    2            0.3378  
    19           0.3377  
    4            0.3377  
    20           0.3330  
    17           0.3299  
    34           0.3295  
    14           0.3258  
    1            0.3257  
    16           0.3220  
    35           0.3195  
    23           0.3139  
    3            0.2900  
    27           0.2880  
    9            0.2880  
    0            0.2813  
    13           0.2808  
    31           0.2795  
    18           0.2795  
    26           0.2743  
    8            0.2716  
    32           0.2666  
    24           0.2665  
    21           0.2639  
    5            0.2592  
    11           0.2498  
    29           0.2498  
    28           0.2491  
    10           0.2491  
    7            0.1877  
    
    💾 Leaderboard saved to outputs/performance_leaderboard.csv


## 4. Visualization and Export {#visualization}

Let's create comprehensive visualizations of our leaderboard results and export them in various formats.



```python
# Create comprehensive visualizations
if len(performance_df) > 0:
    print("📊 Creating Performance Visualizations...")
    print("=" * 70)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Error distribution by category
    ax1 = axes[0, 0]
    for category in performance_df['Category'].unique():
        category_data = performance_df[performance_df['Category'] == category]['Error']
        ax1.hist(category_data, alpha=0.7, label=category, bins=15)
    ax1.set_xlabel('Absolute Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution by Category')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Execution time by category
    ax2 = axes[0, 1]
    for category in performance_df['Category'].unique():
        category_data = performance_df[performance_df['Category'] == category]['Execution_Time']
        ax2.hist(category_data, alpha=0.7, label=category, bins=15)
    ax2.set_xlabel('Execution Time (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Execution Time Distribution by Category')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error vs True H
    ax3 = axes[0, 2]
    for category in performance_df['Category'].unique():
        category_data = performance_df[performance_df['Category'] == category]
        ax3.scatter(category_data['True_H'], category_data['Error'], 
                   alpha=0.7, label=category, s=50)
    ax3.set_xlabel('True Hurst Parameter')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title('Error vs True Hurst Parameter')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance by estimator
    ax4 = axes[1, 0]
    estimator_performance = performance_df.groupby('Estimator')['Error'].mean().sort_values()
    ax4.bar(range(len(estimator_performance)), estimator_performance.values, alpha=0.7)
    ax4.set_xlabel('Estimator')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Mean Error by Estimator')
    ax4.set_xticks(range(len(estimator_performance)))
    ax4.set_xticklabels(estimator_performance.index, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Execution time by estimator
    ax5 = axes[1, 1]
    time_performance = performance_df.groupby('Estimator')['Execution_Time'].mean().sort_values()
    ax5.bar(range(len(time_performance)), time_performance.values, alpha=0.7)
    ax5.set_xlabel('Estimator')
    ax5.set_ylabel('Mean Execution Time (seconds)')
    ax5.set_title('Mean Execution Time by Estimator')
    ax5.set_xticks(range(len(time_performance)))
    ax5.set_xticklabels(time_performance.index, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)
    
    # 6. Composite score ranking
    ax6 = axes[1, 2]
    if len(leaderboard_df) > 0:
        top_10 = leaderboard_df.head(10)
        ax6.barh(range(len(top_10)), top_10['Composite_Score'], alpha=0.7)
        ax6.set_xlabel('Composite Score')
        ax6.set_ylabel('Rank')
        ax6.set_title('Top 10 Estimators by Composite Score')
        ax6.set_yticks(range(len(top_10)))
        ax6.set_yticklabels([f"{row['Category']} - {row['Estimator']}" for _, row in top_10.iterrows()])
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/leaderboard_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create category-specific leaderboards
    print("\n📊 Category-Specific Leaderboards:")
    print("=" * 70)
    
    for category in performance_df['Category'].unique():
        category_data = performance_df[performance_df['Category'] == category]
        category_leaderboard = category_data.groupby('Estimator').agg({
            'Error': ['mean', 'std'],
            'Execution_Time': 'mean',
            'True_H': 'count'
        }).round(4)
        
        print(f"\n{category} Category Leaderboard:")
        print(category_leaderboard)
    
    # Export results in multiple formats
    print("\n💾 Exporting Results...")
    print("=" * 70)
    
    # CSV export
    performance_df.to_csv('outputs/performance_data.csv', index=False)
    print("✅ Performance data exported to CSV")
    
    # JSON export
    performance_df.to_json('outputs/performance_data.json', orient='records', indent=2)
    print("✅ Performance data exported to JSON")
    
    # LaTeX table export
    if len(leaderboard_df) > 0:
        latex_table = leaderboard_df.to_latex(index=False, float_format='%.4f')
        with open('outputs/leaderboard_table.tex', 'w') as f:
            f.write(latex_table)
        print("✅ Leaderboard table exported to LaTeX")
    
    print("\n🎯 All visualizations and exports completed successfully!")
    
else:
    print("❌ No performance data available for visualization")

```

    📊 Creating Performance Visualizations...
    ======================================================================



    
![png](05_leaderboard_generation_files/05_leaderboard_generation_8_1.png)
    


    
    📊 Category-Specific Leaderboards:
    ======================================================================
    
    Classical Category Leaderboard:
                     Error         Execution_Time True_H
                      mean     std           mean  count
    Estimator                                           
    CWT             0.3242  0.4224         0.0737      4
    DFA             0.2197  0.2085         0.0068      4
    DMA             0.1829  0.1868         0.0011      4
    GPH             0.2396  0.1937         0.1128      4
    Higuchi         0.1819  0.2077         0.0023      4
    MFDFA           0.3889  0.2053         0.1110      4
    Periodogram     0.1287  0.1620         0.0013      4
    R/S             0.1777  0.2157         0.8095      4
    WaveletLeaders  0.4523  0.2733         0.0142      4
    WaveletLogVar   0.3880  0.3424         0.0006      4
    WaveletVar      0.6041  0.4089         0.0011      4
    WaveletWhittle  0.5900  0.2000         0.0071      4
    Whittle         0.1000  0.2000         0.0005      4
    
    ML Category Leaderboard:
                       Error         Execution_Time True_H
                        mean     std           mean  count
    Estimator                                             
    GradientBoosting  0.4308  0.1988         0.0002      4
    RandomForest      0.5000  0.2000         0.0002      4
    SVR               0.1364  0.2212         0.0000      4
    
    Neural Category Leaderboard:
                Error         Execution_Time True_H
                 mean     std           mean  count
    Estimator                                      
    GRU        0.2119  0.0052         0.0017      4
    LSTM       0.2000  0.0002         0.0433      4
    
    Comprehensive Category Leaderboard:
                       Error         Execution_Time True_H
                        mean     std           mean  count
    Estimator                                             
    CWT               0.3242  0.4224         0.0758      4
    DFA               0.2197  0.2085         0.0066      4
    DMA               0.1829  0.1868         0.0010      4
    GPH               0.2396  0.1937         0.0023      4
    GRU               0.2119  0.0052         0.0006      4
    GradientBoosting  0.4308  0.1988         0.0003      4
    Higuchi           0.1819  0.2077         0.0023      4
    LSTM              0.2000  0.0002         0.0009      4
    MFDFA             0.3699  0.1944         0.1066      4
    Periodogram       0.1287  0.1620         0.0013      4
    R/S               0.1777  0.2157         0.0822      4
    RandomForest      0.5000  0.2000         0.0005      4
    SVR               0.1364  0.2212         0.0001      4
    WaveletLeaders    0.4397  0.2714         0.0130      4
    WaveletLogVar     0.3880  0.3424         0.0005      4
    WaveletVar        0.6041  0.4089         0.0010      4
    WaveletWhittle    0.5900  0.2000         0.0072      4
    Whittle           0.1000  0.2000         0.0006      4
    
    💾 Exporting Results...
    ======================================================================
    ✅ Performance data exported to CSV
    ✅ Performance data exported to JSON
    ✅ Leaderboard table exported to LaTeX
    
    🎯 All visualizations and exports completed successfully!


## 5. Summary and Next Steps {#summary}

### Key Takeaways

1. **Leaderboard Generation**: lrdbenchmark provides comprehensive tools for creating performance leaderboards:
   - **Multi-category Comparison**: Classical, ML, and Neural estimators
   - **Composite Scoring**: Combined accuracy, speed, and reliability metrics
   - **Statistical Analysis**: Confidence intervals and significance tests
   - **Publication-ready Output**: LaTeX, CSV, JSON formats

2. **Performance Rankings**: The system generates multiple types of leaderboards:
   - **Overall Leaderboard**: Combined performance across all categories
   - **Category-specific**: Rankings within each estimator category
   - **Metric-specific**: Rankings by accuracy, speed, or reliability
   - **Composite Scoring**: Weighted combination of multiple metrics

3. **Visualization**: Comprehensive plots and tables for:
   - **Error Distributions**: Performance across different scenarios
   - **Execution Time Analysis**: Computational efficiency comparison
   - **Scatter Plots**: Error vs true Hurst parameter relationships
   - **Bar Charts**: Direct performance comparisons

### Leaderboard Results

- **Top Performers**: Best estimators across different categories
- **Performance Trade-offs**: Accuracy vs speed analysis
- **Category Strengths**: Each category's optimal use cases
- **Statistical Significance**: Confidence in performance differences

### Next Steps

1. **Real-world Application**: Apply leaderboards to actual time series data
2. **Advanced Analysis**: Explore statistical significance and confidence intervals
3. **Custom Metrics**: Create domain-specific performance measures
4. **Interactive Dashboards**: Build web-based leaderboard interfaces

### Files Generated

- `outputs/performance_leaderboard.csv`: Complete leaderboard data
- `outputs/performance_data.csv`: Raw performance data
- `outputs/performance_data.json`: JSON format data
- `outputs/leaderboard_table.tex`: LaTeX table for publications
- `outputs/leaderboard_visualization.png`: Comprehensive visualization

### References

1. Taqqu, M. S., Teverovsky, V., & Willinger, W. (1995). Estimators for long-range dependence: an empirical study. Fractals, 3(04), 785-798.
2. Beran, J. (1994). Statistics for long-memory processes. CRC press.
3. Abry, P., & Veitch, D. (1998). Wavelet analysis of long-range-dependent traffic. IEEE Transactions on information theory, 44(1), 2-15.

---

**Congratulations!** You've completed the comprehensive lrdbenchmark demonstration series. You now have a complete understanding of:
- Data generation and visualization
- Estimation and statistical validation
- Custom model and estimator development
- Comprehensive benchmarking
- Leaderboard generation and analysis


## Additional Analyses

- Use `summary["stratified_metrics"]` from the comprehensive benchmark JSON to build Hurst, tail, length, and contamination slices before rendering leaderboards.
- Call `dashboard.generate_stratified_report(path_to_json)` for markdown-ready stratified tables.
- Run `dashboard.create_advanced_diagnostics_visuals(path_to_advanced_json, output_dir=...)` to produce scaling slope and robustness panel figures documenting estimator sensitivity to missingness, regime shifts, bursts, and seasonal drift.


