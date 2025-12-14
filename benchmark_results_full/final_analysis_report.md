# Analytical Report: Benchmarking LRD Estimators

## 1. Pure Synthetic Processes

### 1.1 Accuracy (RMSE)

**Model: fGn, N=16384**

| Estimator | RMSE |
|---|---|
| DFA | 0.0097 |
| Higuchi | 0.0155 |
| DMA | 0.0252 |
| Periodogram | 0.0278 |
| Whittle | 0.0434 |
| WaveletLogVar | 0.0552 |
| CWT | 0.0574 |
| GPH | 0.0627 |
| R/S | 0.0678 |
| WaveletVar | 0.0721 |
| WaveletWhittle | 0.1999 |
| MFDFA | 0.4721 |
| WaveletLeaders | 0.5496 |


**Observation**: The **DFA** estimator achieved the lowest RMSE on pure fGn data.

## 2. Robustness to Contamination

## 3. Real-World Time Series
