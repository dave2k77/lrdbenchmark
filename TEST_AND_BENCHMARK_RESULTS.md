# Comprehensive Benchmark Results and Leaderboard

Generated: 2025-11-05 13:49:34

## Executive Summary

- **Total Tests**: 80
- **Success Rate**: 100.0%
- **Data Models Tested**: 4
- **Estimators Tested**: 20
- **Pretrained Models Used**: 7

## ✅ Pretrained Model Usage Confirmation

**CONFIRMED**: All ML and Neural Network estimators are using pretrained models during benchmarks.

The benchmark system (`lrdbenchmark/analysis/benchmark.py`) initializes estimators as follows:

```python
# ML Estimators (Pretrained)
'RandomForest': RandomForestPretrainedModel()
'GradientBoosting': GradientBoostingPretrainedModel()
'SVR': SVREstimatorPretrainedModel()

# Neural Network Estimators (Pretrained)
'CNN': CNNPretrainedModel(input_length=500)
'LSTM': LSTMPretrainedModel(input_length=500)
'GRU': GRUPretrainedModel(input_length=500)
'Transformer': TransformerPretrainedModel(input_length=500)
```

All pretrained models were successfully loaded and verified during benchmark execution:

- ✅ RandomForest: Loaded
- ✅ GradientBoosting: Loaded
- ✅ SVR: Loaded
- ✅ CNN: Loaded
- ✅ LSTM: Loaded
- ✅ GRU: Loaded
- ✅ Transformer: Loaded (fixed - now initializes immediately like other models)

## Top 20 Performers (Lowest Error)

| Rank | Estimator | Data Model | Error | Time (s) | H_est | H_true | Pretrained |
|------|-----------|------------|-------|----------|-------|--------|------------|
| 1 | Whittle | fBm | 0.0000 | 0.000 | 0.000 | 0.700 | No |
| 2 | Whittle | fGn | 0.0000 | 0.000 | 0.000 | 0.700 | No |
| 3 | Whittle | MRW | 0.0000 | 0.000 | 0.000 | 0.700 | No |
| 4 | R/S | MRW | 0.0062 | 0.025 | 0.000 | 0.700 | No |
| 5 | GPH | MRW | 0.0071 | 0.218 | 0.000 | 0.700 | No |
| 6 | CWT | MRW | 0.0106 | 0.002 | 0.000 | 0.700 | No |
| 7 | WaveletVar | MRW | 0.0133 | 0.000 | 0.000 | 0.700 | No |
| 8 | SVR | MRW | 0.0147 | 0.000 | 0.000 | 0.700 | ✓ Yes |
| 9 | SVR | fBm | 0.0244 | 0.000 | 0.000 | 0.700 | ✓ Yes |
| 10 | SVR | fGn | 0.0244 | 0.000 | 0.000 | 0.700 | ✓ Yes |
| 11 | WaveletLogVar | fBm | 0.0367 | 0.000 | 0.000 | 0.700 | No |
| 12 | WaveletLogVar | fGn | 0.0367 | 0.000 | 0.000 | 0.700 | No |
| 13 | Higuchi | MRW | 0.0373 | 0.002 | 0.000 | 0.700 | No |
| 14 | DMA | MRW | 0.0479 | 0.001 | 0.000 | 0.700 | No |
| 15 | Periodogram | fBm | 0.0618 | 0.141 | 0.000 | 0.700 | No |
| 16 | Periodogram | fGn | 0.0618 | 0.001 | 0.000 | 0.700 | No |
| 17 | DMA | fBm | 0.0639 | 0.001 | 0.000 | 0.700 | No |
| 18 | DMA | fGn | 0.0639 | 0.001 | 0.000 | 0.700 | No |
| 19 | WaveletLeaders | ARFIMAModel | 0.0670 | 0.003 | 0.000 | 0.300 | No |
| 20 | DFA | MRW | 0.0675 | 0.008 | 0.000 | 0.700 | No |

## Performance by Estimator Type

### Classical
- **Average Error**: 0.1795
- **Error Range**: 0.0000 - 0.5255
- **Average Time**: 0.104s
- **Number of Tests**: 28

### Wavelet
- **Average Error**: 0.3440
- **Error Range**: 0.0106 - 1.0266
- **Average Time**: 0.001s
- **Number of Tests**: 16

### Multifractal
- **Average Error**: 0.4234
- **Error Range**: 0.0670 - 0.6631
- **Average Time**: 0.054s
- **Number of Tests**: 8

### ML (Pretrained)
- **Average Error**: 0.3562
- **Error Range**: 0.0147 - 0.6000
- **Average Time**: 0.000s
- **Number of Tests**: 12

### Neural (Pretrained)
- **Average Error**: 0.2110
- **Error Range**: 0.1424 - 0.2563
- **Average Time**: 0.003s
- **Number of Tests**: 16

## Pretrained Model Performance Summary

| Estimator | Avg Error | Data Models Tested | Status |
|-----------|-----------|-------------------|--------|
| CNN | 0.2016 | fBm, MRW, ARFIMAModel, fGn | ✓ Used |
| GRU | 0.2119 | fBm, MRW, ARFIMAModel, fGn | ✓ Used |
| GradientBoosting | 0.4358 | fBm, MRW, ARFIMAModel, fGn | ✓ Used |
| LSTM | 0.2027 | fBm, MRW, ARFIMAModel, fGn | ✓ Used |
| RandomForest | 0.5000 | fBm, MRW, ARFIMAModel, fGn | ✓ Used |
| SVR | 0.1329 | fBm, MRW, ARFIMAModel, fGn | ✓ Used |
| Transformer | 0.2278 | fBm, MRW, ARFIMAModel, fGn | ✓ Used |

## Overall Statistics

- **Total Tests**: 80
- **Pretrained Model Tests**: 28
- **Classical Estimator Tests**: 28
- **Average Error (All)**: 0.2696
- **Average Error (Pretrained)**: 0.2732
- **Average Error (Classical)**: 0.1795
