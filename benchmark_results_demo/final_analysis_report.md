# Analytical Report: Benchmarking LRD Estimators

## 1. Pure Synthetic Processes

### 1.1 Accuracy (RMSE)

**Model: fGn, N=1024**

| Estimator | RMSE |
|---|---|
| Whittle | 0.0048 |
| DFA | 0.0067 |
| Higuchi | 0.0090 |
| GPH | 0.0094 |
| Periodogram | 0.0116 |
| WaveletLogVar | 0.0263 |
| DMA | 0.0436 |
| WaveletVar | 0.0527 |
| CWT | 0.0632 |
| R/S | 0.0665 |
| WaveletWhittle | 0.0865 |
| MFDFA | 0.5123 |
| WaveletLeaders | 0.5925 |


**Observation**: The **Whittle** estimator achieved the lowest RMSE on pure fGn data.

## 2. Robustness to Contamination

### 2.1 Performance under Severe Contamination (Level 0.5)

| Estimator | Mean Abs Error |
|---|---|
| WaveletLogVar | 0.0529 |
| Higuchi | 0.0601 |
| Periodogram | 0.0651 |
| DFA | 0.0671 |
| R/S | 0.0701 |
| Whittle | 0.0741 |
| DMA | 0.0886 |
| GPH | 0.0992 |
| WaveletVar | 0.1036 |
| CWT | 0.1048 |
| WaveletWhittle | 0.1643 |
| MFDFA | 0.5656 |
| WaveletLeaders | 0.6746 |


**Observation**: The **WaveletLogVar** estimator demonstrated the highest robustness.

## 3. Real-World Time Series

### 3.1 Mean Hurst Estimates

| Dataset | CWT | DFA | DMA | GPH | Higuchi | MFDFA | Periodogram | R/S | WaveletLeaders | WaveletLogVar | WaveletVar | WaveletWhittle | Whittle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| biophysics_protein | 0.827 | 1.710 | 0.741 | 0.990 | 0.998 | 0.108 | 0.990 | 1.124 | 0.120 | 1.107 | 1.375 | 0.517 | 0.677 |
| climate_temperature | 1.244 | 1.653 | 0.855 | 0.452 | 0.971 | 0.073 | 0.990 | 1.087 | 0.075 | 0.775 | 0.896 | 0.292 | 0.990 |
| financial_sp500 | 1.482 | 1.987 | 1.007 | 0.990 | 0.991 | 2.027 | 0.990 | 1.009 | 0.148 | 0.768 | 0.886 | 0.346 | 0.990 |
| network_traffic | 0.490 | 1.112 | 1.002 | 0.990 | 0.987 | 0.073 | 0.934 | 1.044 | 0.071 | 0.554 | 0.577 | 1.990 | 0.475 |
| physiological_eeg | 2.175 | 1.309 | 0.632 | 0.990 | 0.985 | 0.508 | 0.990 | 0.803 | 0.099 | 1.326 | 1.692 | 0.424 | 0.990 |
| physiological_hrv | 0.753 | 1.231 | 1.041 | 0.990 | 0.993 | 0.070 | 0.971 | 1.006 | 0.047 | 0.661 | 0.732 | 0.010 | 0.605 |

