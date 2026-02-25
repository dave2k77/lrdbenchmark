The Multiscale Entropy (MSE) algorithm is designed to quantify the complexity of a time series by evaluating it across various temporal resolutions. Unlike traditional methods that look at data on a single scale, this process involves two primary steps: **coarse-graining** the data and calculating **Sample Entropy (SampEn)** for each resulting series.

---

## The Two-Step MSE Algorithm

### 1. Coarse-Graining the Time Series

The algorithm begins by transforming the original one-dimensional discrete time series  into a series of "coarse-grained" versions.

* 
**Scale Factor ():** For a given scale factor , the original data is divided into non-overlapping windows of length .


* 
**Averaging:** Each element of the new coarse-grained time series  is calculated by taking the average of the data points within those windows.


* **The Equation:** This is represented mathematically as:



.


* 
**Result:** At Scale 1, the series is identical to the original. As  increases, the length of the time series decreases by a factor of .



### 2. Calculating Sample Entropy (SampEn)

Once the coarse-grained series is created for a specific scale, **Sample Entropy (SampEn)** is calculated for that sequence.

* 
**Metric Goal:** SampEn quantifies the regularity (orderliness) of the series.


* 
**Predictability:** It reflects the probability that two sequences which are similar for  points remain similar at the next () point.


* 
**Parameters:** It typically uses a pattern length () and a similarity criterion ().


* 
**Interpretation:** Higher SampEn values indicate more disorder or unpredictability; lower values indicate more regularity.



---

## Summary of the Complexity Profile

By plotting the SampEn values as a function of the scale factor , the algorithm creates a **complexity profile**.

| Signal Type | Behavior Across Scales | Complexity Interpretation |
| --- | --- | --- |
| **White Noise** | High entropy at Scale 1; decreases significantly as scales increase.

 | High randomness, but low structural complexity.

 |
| ** Noise** | Entropy remains nearly constant across all scales.

 | High complexity with structure across multiple scales.

 |
| **Healthy Heartbeat** | Entropy increases or stays high across multiple scales.

 | Highly complex and adaptive system.

 |
| **Pathologic Heartbeat** | Entropy drops off rapidly as scales increase (AF or CHF).

 | Reduced complexity and loss of adaptive capacity.

 ## PSEUDOCODE

```python
# Multiscale Entropy (MSE) Analysis
# Input: Original time series {X}, Max Scale Factor (tau_max), 
#        Pattern length (m), Similarity criterion (r)

FOR each scale_factor (tau) from 1 to tau_max:
    
    # STEP 1: Coarse-Graining [cite: 64]
    Initialize coarse_grained_series {Y} as empty
    series_length = length(X)
    
    # Divide the series into non-overlapping windows of size tau [cite: 64]
    FOR j from 1 to (series_length / tau):
        # Calculate the average of the data points in the window [cite: 64, 91]
        sum_window = 0
        FOR i from ((j-1) * tau + 1) to (j * tau):
            sum_window = sum_window + X[i]
        
        Y[j] = sum_window / tau
    
    # STEP 2: Calculate Sample Entropy (SampEn) for the current scale [cite: 67, 68]
    # SampEn reflects the probability that sequences similar for m points 
    # remain similar for m + 1 points [cite: 56, 224]
    
    entropy_value = Calculate_SampEn(series=Y, pattern_length=m, tolerance=r)
    
    # STEP 3: Store and Plot [cite: 67]
    Store (tau, entropy_value)

END FOR

# The resulting plot of entropy_value vs. tau represents the complexity profile.