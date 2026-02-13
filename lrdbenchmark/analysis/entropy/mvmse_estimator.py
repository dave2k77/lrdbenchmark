#!/usr/bin/env python3
"""
Multivariate Multiscale Entropy (mvMSE) Estimator for Long-Range Dependence.

This module implements Multivariate Multiscale Sample Entropy (MvMSEn) to
measure the joint regularity / complexity of multichannel time series across
temporal scales.

Notes
-----
**mvMSE is not a direct Hurst exponent estimator.** It produces a Complexity
Index (CI) — the area under the multivariate entropy-vs-scale curve — rather
than a Hurst parameter.  To enable comparison with the library's classical H
estimators, this class derives an *approximate* ``hurst_parameter`` from the
normalised slope of the mvMSE curve:

* A **flat** mvMSE curve (high CI) signals strong inter-channel LRD and maps
  to H ≈ 1.0.
* A **steeply decaying** mvMSE curve signals loss of cross-channel persistence
  and maps to H ≈ 0.5.

The ``complexity_index`` field is the primary, native output of this estimator.
The ``hurst_parameter`` field should be interpreted as a derived convenience
metric for cross-estimator comparison only.

References
----------
Ahmed, M. U., & Mandic, D. P. (2011). Multivariate multiscale entropy
analysis. *IEEE Signal Processing Letters*, 19(2), 91–94.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy import stats

from ..base_estimator import BaseEstimator

try:
    import EntropyHub as EH
except ImportError:
    EH = None  # type: ignore[assignment]


class MultivariateMSEEstimator(BaseEstimator):
    """Multivariate Multiscale Sample Entropy (mvMSE) estimator.

    Measures the joint complexity of a multivariate time series by computing
    Multivariate Sample Entropy (MvSampEn) across multiple temporal scales.

    Parameters
    ----------
    m : list of int, optional
        Pattern (embedding) length for each channel (default ``[2, 2]``).
        Length must match the number of channels in the input data.
    r : float, optional
        Similarity tolerance as a fraction of the pooled standard deviation
        (default 0.15).
    max_scale : int, optional
        Maximum coarse-graining scale factor (default 20).

    Notes
    -----
    mvMSE produces a **Complexity Index**, not a direct Hurst parameter.
    The ``hurst_parameter`` returned by :meth:`estimate` is an approximate
    mapping derived from the mvMSE curve slope for comparability with the
    library's classical estimators.  The mapping interpretation:

    * Flat mvMSE curve (high CI) → strong inter-channel LRD → H ≈ 1.0
    * Steep decay → loss of cross-channel persistence → H ≈ 0.5

    See Also
    --------
    MSEEstimator : Univariate MSE for single-channel data.
    """

    def __init__(
        self,
        m: Optional[List[int]] = None,
        r: float = 0.15,
        max_scale: int = 20,
    ) -> None:
        if m is None:
            m = [2, 2]
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

        if not isinstance(m, (list, tuple)) or len(m) < 2:
            raise ValueError(
                f"m must be a list of at least 2 positive integers "
                f"(one per channel), got {m}"
            )
        if any(mi < 1 for mi in m):
            raise ValueError(f"All elements of m must be >= 1, got {m}")
        if r <= 0:
            raise ValueError(f"r must be positive, got {r}")
        if not isinstance(max_scale, int) or max_scale < 2:
            raise ValueError(f"max_scale must be an integer >= 2, got {max_scale}")

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    def estimate(self, data: Union[np.ndarray, list]) -> Dict[str, Any]:
        """Compute Multivariate Multiscale Sample Entropy.

        Parameters
        ----------
        data : array-like, shape (N, C)
            Multivariate time series with *N* time points and *C* channels.
            The number of channels must match the length of ``m``.

        Returns
        -------
        dict
            Results dictionary containing:

            - ``complexity_index`` (float): Area under the mvMSE curve —
              the primary native metric.
            - ``hurst_parameter`` (float): Approximate Hurst parameter
              derived from the mvMSE curve slope (see class Notes).
            - ``entropy_values`` (list[float]): Multivariate Sample Entropy
              at each scale.
            - ``scales`` (list[int]): Scale factors ``1 .. max_scale``.
            - ``slope`` (float): Linear regression slope of entropy vs
              log-scale (used for the H mapping).
            - ``r_squared`` (float): Goodness of fit of the slope regression.
            - ``n_channels`` (int): Number of input channels.
            - ``method`` (str): ``"MvMSEn"`` (Multivariate MSE).

        Raises
        ------
        ImportError
            If ``EntropyHub`` is not installed.
        ValueError
            If data shape is incompatible with parameters.
        """
        if EH is None:
            raise ImportError(
                "EntropyHub is required for MultivariateMSEEstimator. "
                "Install it with: pip install EntropyHub"
            )

        data = np.asarray(data, dtype=float)

        # Validate shape
        if data.ndim == 1:
            raise ValueError(
                "Multivariate MSE requires 2-D input (N, C) with C >= 2 "
                "channels. For univariate analysis use MSEEstimator."
            )
        if data.ndim != 2:
            raise ValueError(
                f"Expected 2-D array (N, C), got shape {data.shape}"
            )

        n, n_channels = data.shape
        m = self.parameters["m"]

        if n_channels < 2:
            raise ValueError(
                f"Need at least 2 channels, got {n_channels}. "
                f"For univariate analysis use MSEEstimator."
            )

        if len(m) != n_channels:
            raise ValueError(
                f"Length of m ({len(m)}) must match the number of channels "
                f"({n_channels})."
            )

        if n < 50:
            raise ValueError(
                f"Data length ({n}) is too short for mvMSE analysis; "
                f"need at least 50 data points per channel."
            )

        max_scale = self.parameters["max_scale"]
        effective_max_scale = min(max_scale, n // 10)
        if effective_max_scale < 2:
            raise ValueError(
                f"Data length ({n}) is too short for the requested "
                f"max_scale ({max_scale}). Need at least 20 data points."
            )

        r = self.parameters["r"]

        if n < 200:
            warnings.warn(
                f"Data length ({n}) is small for mvMSE analysis; "
                f"results at coarse scales may be unreliable.",
                stacklevel=2,
            )

        # Normalise each channel to zero mean, unit variance
        data_norm = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-12)

        # ----- Compute MvMSEn via EntropyHub -----
        Mobj = EH.MSobject("MvSampEn", m=np.array(m), r=r)
        msx, ci = EH.MvMSEn(data_norm, Mobj, Scales=effective_max_scale)

        msx = np.asarray(msx, dtype=float)
        scales = np.arange(1, effective_max_scale + 1)

        # Handle inf/nan values
        valid_mask = np.isfinite(msx)
        valid_entropy = msx.copy()
        valid_entropy[~valid_mask] = np.nan

        # Compute clean CI from valid scales only
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
            "n_channels": n_channels,
            "method": "MvMSEn",
            "m": m,
            "r": r,
            "max_scale_used": effective_max_scale,
            "n_valid_scales": int(np.sum(valid_mask)),
        }
        return self.results

    # ------------------------------------------------------------------
    # Hurst mapping  (identical logic to MSEEstimator)
    # ------------------------------------------------------------------
    @staticmethod
    def _entropy_slope_to_hurst(
        scales: np.ndarray,
        entropy_values: np.ndarray,
    ) -> tuple:
        """Map mvMSE curve slope to an approximate Hurst parameter.

        See :meth:`MSEEstimator._entropy_slope_to_hurst` for full details.
        """
        mask = np.isfinite(entropy_values)
        if np.sum(mask) < 3:
            return 0.5, 0.0, 0.0

        log_scales = np.log(scales[mask])
        ent_vals = entropy_values[mask]

        slope_val, _, r_value, _, _ = stats.linregress(log_scales, ent_vals)
        r_sq = float(r_value ** 2)

        reference_slope = 0.5
        normalised = min(abs(slope_val) / reference_slope, 1.0)
        hurst_approx = 0.5 + 0.5 * (1.0 - normalised)

        hurst_approx = float(np.clip(hurst_approx, 0.0, 1.0))
        return hurst_approx, float(slope_val), r_sq
