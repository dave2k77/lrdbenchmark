Since EEG is a spatial network, analyzing a single channel only gives us half the story. To truly measure **Long-Range Dependence (LRD)** in the brain, we need to see how information flows *between* channels (e.g., Frontal vs. Occipital) across different time scales.

This brings us to **Multivariate Multiscale Entropy (mvMSE)**.

---

## 1. The Mathematical Shift: From Sample to Multivariate

In standard MSE, we look at patterns within one vector. In **mvMSE**, we define a composite delay vector that includes data from  different channels simultaneously.

If we have two EEG channels,  and , the embedding vector at scale  looks like this:


Instead of measuring the predictability of one line, we are measuring the **joint predictability** of the entire system. If the brain is healthy and exhibiting LRD, the multivariate entropy will stay high across scales, meaning the channels remain "coupled" in a complex, non-random way.

---

## 2. Python Implementation: Frontal-Occipital Coupling

We’ll simulate two EEG channels. In a healthy "resting state," these channels often show a specific  correlation. We will use `EntropyHub.MvMSEn`.

```python
import numpy as np
import EntropyHub as EH
import matplotlib.pyplot as plt

# 1. Generate two correlated EEG-like signals (N=3000)
N = 3000
t = np.linspace(0, 10, N)

# Channel 1: Frontal (Stronger Alpha 10Hz + Pink Noise)
sig1 = np.cumsum(np.random.randn(N)) + 0.5 * np.sin(2 * np.pi * 10 * t)
# Channel 2: Occipital (Correlated Pink Noise + slight lag)
sig2 = 0.7 * sig1 + 0.3 * np.cumsum(np.random.randn(N))

# Normalize
data = np.vstack([sig1, sig2]).T
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# 2. Compute Multivariate MSE (mvMSE)
# We define the Multivariate Object
# m=[2, 2] means we look for patterns of length 2 in both channels
Mobj = EH.MSobject('MvSampEn', m=[2, 2], r=0.15)

# Calculate mvMSE across 20 scales
mv_entropy, ci = EH.MvMSEn(data, Mobj, Scales=20)

# 3. Plotting the Global Complexity
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), mv_entropy, 's-', color='darkorchid', label='Multivariate (Fp1-O1)')
plt.title(f'Multivariate MSE: Brain Network Complexity (CI: {ci:.2f})')
plt.xlabel('Scale Factor (τ)')
plt.ylabel('Multivariate Sample Entropy')
plt.grid(alpha=0.3)
plt.legend()
plt.show()

```

---

## 3. Interpreting the Multivariate Curve

When you analyze EEG with mvMSE, the curve reveals the **Functional Connectivity** of the LRD:

* **Broadband LRD (Healthy):** The curve starts high and remains high. This suggests that the "memory" of the system isn't just local to one electrode; the frontal and occipital lobes are communicating in a fractal, self-similar way across time.
* **Systemic Breakdown (Pathological):** If the mvMSE curve drops faster than the individual channel MSE curves, it implies that while individual regions might still be active, the **coordination** between them has lost its complexity.

> **Pro Tip:** In EEG, the choice of  is critical here. Since multivariate data has different variances across channels, `EntropyHub` typically normalizes the joint variance to ensure the  threshold is meaningful for the entire system.

---