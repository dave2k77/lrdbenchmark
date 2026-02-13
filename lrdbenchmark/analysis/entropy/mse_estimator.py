#!/usr/bin/env python3
"""
Multiscale Entropy (MSE) Estimator for Long-Range Dependence.

This module implements Refined Composite Multiscale Sample Entropy (RCMSE)
as a regularity-based measure of persistence / long-range dependence (LRD).

Notes
-----
**MSE is not a direct Hurst exponent estimator.** It produces a Complexity
Index (CI) — the area under the entropy-vs-scale curve — rather than a Hurst
parameter.  To enable comparison with the library's classical H estimators,
this class derives an *approximate* ``hurst_parameter`` from the normalised
slope of the MSE curve:

* A **flat** MSE curve (high CI, characteristic of 1/f noise) signals strong
  LRD and maps to H ≈ 1.0.
* A **steeply decaying** MSE curve (characteristic of white noise) signals no
  LRD and maps to H ≈ 0.5.

The ``complexity_index`` field is the primary, native output of this estimator.
The ``hurst_parameter`` field should be interpreted as a derived convenience
metric for cross-estimator comparison only.

References
----------
Costa, M., Goldberger, A. L., & Peng, C.-K. (2005). Multiscale entropy
analysis of biological signals. *Physical Review E*, 71(2), 021906.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import stats

from ..base_estimator import BaseEstimator

try:
    import EntropyHub as EH
except ImportError:
    EH = None  # type: ignore[assignment]


class MSEEstimator(BaseEstimator):
    """Refined Composite Multiscale Sample Entropy (RCMSE) estimator.

    Measures the complexity of a univariate time series across multiple
    temporal scales using Sample Entropy as the base metric.  The Complexity
    Index (area under the MSE curve) serves as the primary LRD indicator.

    Parameters
    ----------
    m : int, optional
        Pattern (embedding) length for Sample Entropy (default 2).
    r : float, optional
        Similarity tolerance as a fraction of the time-series standard
        deviation (default 0.15).
    max_scale : int, optional
        Maximum coarse-graining scale factor (default 20).

    Notes
    -----
    MSE produces a **Complexity Index**, not a direct Hurst parameter.
    The ``hurst_parameter`` returned by :meth:`estimate` is an approximate
    mapping derived from the MSE curve slope for comparability with the
    library's classical estimators.  The mapping interpretation:

    * Flat MSE curve (high CI, 1/f-like) → strong LRD → H ≈ 1.0
    * Steep decay (white noise) → no LRD → H ≈ 0.5

    See Also
    --------
    MultivariateMSEEstimator : Multivariate extension for multichannel data.
    """

    def __init__(
        self,
        m: int = 2,
        r: float = 0.15,
        max_scale: int = 20,
    ) -> None:
        super().__init__(m=m, r=r, max_scale=max_scale)
        self._validate_parameters()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_parameters(self) -> None:
        """Validate estimator parameters."""
        m = self.parameters["m"]
        r = self.parameters["r"]
        max_scale = self.parameters["max_scale"]

        if not isinstance(m, int) or m < 1:
            raise ValueError(f"m must be a positive integer, got {m}")
        if r <= 0:
            raise ValueError(f"r must be positive, got {r}")
        if not isinstance(max_scale, int) or max_scale < 2:
            raise ValueError(f"max_scale must be an integer >= 2, got {max_scale}")

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    def estimate(self, data: Union[np.ndarray, list]) -> Dict[str, Any]:
        """Compute Refined Composite Multiscale Sample Entropy.

        Parameters
        ----------
        data : array-like, shape (N,)
            Univariate time series.

        Returns
        -------
        dict
            Results dictionary containing:

            - ``complexity_index`` (float): Area under the MSE curve — the
              primary native metric.
            - ``hurst_parameter`` (float): Approximate Hurst parameter
              derived from the MSE curve slope (see class Notes).
            - ``entropy_values`` (list[float]): Sample Entropy at each scale.
            - ``scales`` (list[int]): Scale factors ``1 .. max_scale``.
            - ``slope`` (float): Linear regression slope of entropy vs
              log-scale (used for the H mapping).
            - ``r_squared`` (float): Goodness of fit of the slope regression.
            - ``method`` (str): ``"RCMSE"`` (Refined Composite MSE).

        Raises
        ------
        ImportError
            If ``EntropyHub`` is not installed.
        ValueError
            If data is too short for the requested parameters.
        """
        if EH is None:
            raise ImportError(
                "EntropyHub is required for MSEEstimator. "
                "Install it with: pip install EntropyHub"
            )

        data = np.asarray(data, dtype=float).ravel()
        n = len(data)

        if n < 50:
            raise ValueError(
                f"Data length ({n}) is too short for MSE analysis; "
                f"need at least 50 data points."
            )

        max_scale = self.parameters["max_scale"]
        # Ensure we don't request more scales than the data can support
        effective_max_scale = min(max_scale, n // 10)
        if effective_max_scale < 2:
            raise ValueError(
                f"Data length ({n}) is too short for the requested "
                f"max_scale ({max_scale}). Need at least 20 data points."
            )

        m = self.parameters["m"]
        r = self.parameters["r"]

        if n < 200:
            warnings.warn(
                f"Data length ({n}) is small for MSE analysis; "
                f"results at coarse scales may be unreliable.",
                stacklevel=2,
            )

        # ----- Compute RCMSE via EntropyHub -----
        Mobj = EH.MSobject("SampEn", m=m, r=r)
        msx, ci = EH.cMSEn(data, Mobj, Scales=effective_max_scale, Refined=True)

        msx = np.asarray(msx, dtype=float)
        scales = np.arange(1, effective_max_scale + 1)

        # Handle inf/nan values at coarse scales
        valid_mask = np.isfinite(msx)
        valid_entropy = msx.copy()
        valid_entropy[~valid_mask] = np.nan

        # Compute CI from valid scales only (sum of finite entropy values)
        if np.any(valid_mask):
            ci_clean = float(np.nansum(valid_entropy))
        else:
            ci_clean = 0.0

        # ----- Derive approximate Hurst parameter from slope -----
        hurst_approx, slope, r_squared = self._entropy_slope_to_hurst(
            scales, msx
        )

        self.results = {
            "complexity_index": ci_clean,
            "hurst_parameter": hurst_approx,
            "entropy_values": valid_entropy.tolist(),
            "scales": scales.tolist(),
            "slope": slope,
            "r_squared": r_squared,
            "method": "RCMSE",
            "m": m,
            "r": r,
            "max_scale_used": effective_max_scale,
            "n_valid_scales": int(np.sum(valid_mask)),
        }
        return self.results

    # ------------------------------------------------------------------
    # Hurst mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _entropy_slope_to_hurst(
        scales: np.ndarray,
        entropy_values: np.ndarray,
    ) -> tuple:
        """Map MSE curve slope to an approximate Hurst parameter.

        Strategy
        --------
        1. Regress entropy values against log(scale) using only finite values.
        2. Normalise the slope so that:
           - white-noise slope (steep negative) → H ≈ 0.5
           - 1/f-noise slope (≈ 0)               → H ≈ 1.0

        The mapping uses the empirical calibration:
            H_approx = 0.5 + 0.5 * (1 - |slope| / reference_slope)
        clamped to [0, 1].

        Returns
        -------
        (hurst_approx, slope, r_squared)
        """
        mask = np.isfinite(entropy_values)
        if np.sum(mask) < 3:
            return 0.5, 0.0, 0.0

        log_scales = np.log(scales[mask])
        ent_vals = entropy_values[mask]

        slope_val, _, r_value, _, _ = stats.linregress(log_scales, ent_vals)
        r_sq = float(r_value ** 2)

        # Reference slope: white noise typically decays at ~-0.5 per log-scale
        reference_slope = 0.5
        normalised = min(abs(slope_val) / reference_slope, 1.0)
        hurst_approx = 0.5 + 0.5 * (1.0 - normalised)

        # Clamp to valid range
        hurst_approx = float(np.clip(hurst_approx, 0.0, 1.0))
        return hurst_approx, float(slope_val), r_sq
