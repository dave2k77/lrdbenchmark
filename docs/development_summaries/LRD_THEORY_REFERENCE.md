# Long-Range Dependence (LRD) Theory Reference

This document records the mathematical background assumed by `lrdbenchmark`. It is intended as a quick reference when validating estimator implementations, writing tutorials, or extending the model zoo.

## Core definitions

- **Autocovariance decay**: A (weakly) stationary process \\( \{X_t\} \\) exhibits long-range dependence if its autocovariance \\( \gamma(k) = \mathrm{Cov}(X_t, X_{t+k}) \\) decays like
  \\[
    \gamma(k) \sim c_\gamma k^{2H-2} \quad \text{as } k \to \infty,
  \\]
  with Hurst exponent \\( H \in (0,1) \\) and constant \\( c_\gamma > 0 \\). Positive dependence corresponds to \\( H > 0.5 \\), while \\( H < 0.5 \\) indicates anti-persistence.

- **Spectral density behaviour**: Equivalently, the spectral density near the origin satisfies
  \\[
    f(\lambda) \sim c_f |\lambda|^{1-2H}, \qquad \lambda \to 0,
  \\]
  with \\( c_f > 0 \\). This links spectral estimators (GPH, Whittle, Periodogram) directly to \\( H \\).

- **ARFIMA differencing parameter**: For ARFIMA(\\(p,d,q\\)) processes, the fractional differencing power \\( d \\) relates to the Hurst exponent via \\( H = d + 0.5 \\) provided \\( d \in (-0.5, 0.5) \\). `lrdbenchmark` enforces this mapping in `lrdbenchmark.models.data_models.arfima`.

- **Fractional Brownian motion (fBm)**: fBm with parameter \\( H \\) satisfies \\( \mathbb{E}[(B_H(t+s) - B_H(s))^2] = \sigma^2 |t|^{2H} \\). Fractional Gaussian noise (fGn) is its increment process; both share the same \\( H \\).

- **Multifractal scaling**: Multifractional processes such as MRW or multifractal wavelet leaders (MWL) use the log-cumulant \\( c_1 = H \\) and higher-order log-cumulants \\( c_q \\) to describe scaling of higher-order moments \\( \mathbb{E}[|X_t - X_{t+\tau}|^q] \propto \tau^{qH + c_q} \\).

## Estimator taxonomy

| Category | Estimators | Primary statistic | Expected slope/intercept |
|----------|------------|-------------------|--------------------------|
| Temporal (time-domain) | R/S, DFA, DMA, Higuchi, GHE | Log-log slope of rescaled range, fluctuation function, moving average residual, fractal dimension | \\( \hat{H} = \text{slope} \\) (R/S, DFA); \\( H = 2 - D \\) for Higuchi |
| Spectral | Periodogram, GPH, Whittle (frequency-domain) | Low-frequency spectral density | \\( \text{slope} = 1 - 2H \\) (GPH); Whittle maximises likelihood under \\( f(\lambda) \propto |\lambda|^{1-2H} \\) |
| Wavelet / Multiresolution | Wavelet variance, log-variance, wavelet Whittle, CWT | Energy across octaves \\( j \\) | \\( \log_2 \sigma_j^2 = (2H-1) j + \beta \\) |
| Multifractal | MFDFA, Wavelet Leaders | Structure functions of order \\( q \\) | \\( \zeta(q) = qH - \frac{c_2}{2} q (q-1) + \cdots \\) |
| Machine learning | Random Forest, SVR, Gradient Boosting | Unified features derived from classical estimators | Models trained to regress \\( H \\) in \\([0.1, 0.9]\\) |
| Neural networks | CNN, LSTM, GRU, Transformer | Raw windowed series | Supervised regression on \\( H \\); invariances enforced via scaling/normalisation in `unified_feature_extractor` |

## Feature extraction alignment

The unified feature extractor maps each time series to the following sets of statistics:

- **76-feature master vector**: Combines temporal (R/S, DFA, DMA), spectral (periodogram slopes, Whittle score), wavelet (variance across octaves), and multifractal descriptors, ensuring ML models observe the same quantities used by analytical estimators.
- **Subsets (29, 54, 8 features)** map to legacy checkpoints. `RandomForestEstimator`, `GradientBoostingEstimator`, and `SVREstimator` detect the required feature count to stay consistent with the pretrained artefact.

## Consistency checks when modifying estimators

1. **Scaling windows**: For log-log regressions, ensure the range of window sizes \\( n \\) or wavelet octaves \\( j \\) covers at least 1.5 decades to avoid bias. Default selections follow published guidance (e.g., DFA box sizes 10–\\(N/4\\)).
2. **Detrending order**: DFA uses polynomial order 1 (linear) by default; higher orders require matching `order` parameter and documentation updates.
3. **Spectral tapering**: Whittle/GPH estimators employ tapering and trimming of the lowest Fourier frequencies; trimming proportion maps to \\( m = N^{0.5} \\) by default.
4. **Multifractal regressions**: MFDFA/regression of \\( \log F_q(s) \\) vs \\( \log s \\) uses \\( q \\) in [-5,5], excluding 0. Implementation must handle q=0 limit via logarithmic averaging.

## Data model summary

| Model | Parameters | True H mapping |
|-------|------------|----------------|
| FBM (`lrdbenchmark.models.data_models.fbm`) | \\(H, \sigma | | H (input) |
| FGN (`...fgn`) | \\(H, \sigma | | H (input) |
| ARFIMA (`...arfima`) | \\(d, \phi, \theta | | H = d + 0.5 |
| MRW (`...mrw`) | Intermittency \\( \lambda^2 \\), cascade depth | Effective \\( H = H_0 \\) from base fBm component |
| Alpha-stable (`...alpha_stable`) | Stability \\( \alpha \\), skew \\( \beta \\), scale \\( \sigma \\), location \\( \mu \\) | LRD controlled via injected fractional integration; documentation calls out when \\( H \\) is undefined |

When adding new models, document how the chosen parameters map to \\( H \\) or to the broader multifractal spectrum to maintain comparability across estimators.

## Validation checklist

1. **Synthetic calibration**: For each estimator, simulate FBM/FGN with \\( H \in \{0.2, 0.4, 0.6, 0.8\} \\) and confirm mean absolute error is within published tolerances (≤ 0.03 for classical methods on \\( N=4096 \\)).
2. **Heavy-tail robustness**: Use alpha-stable generators with \\( \alpha \in \{2.0, 1.5, 1.2, 0.8\} \\). Record bias vs \\( \alpha \\) to ensure robustness features behave as described.
3. **Contamination scenarios**: Apply contamination models (spikes, level shifts, variance shocks) and verify adaptive preprocessing reduces bias relative to vanilla estimators.
4. **Acceleration parity**: JAX/Numba/Torch implementations must numerically match NumPy baselines within 1e-6 for deterministic smoke tests.

Maintaining this reference alongside the code ensures future contributors can reason about estimator changes, validate new features, and keep documentation consistent with the underlying mathematics.

