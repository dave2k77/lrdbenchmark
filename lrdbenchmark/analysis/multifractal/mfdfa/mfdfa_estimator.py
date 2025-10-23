"""
Multifractal Detrended Fluctuation Analysis (MFDFA) Estimator

Canonical MFDFA implementation with options to compute h(q) and report
monofractal H = h(2). NumPy/SciPy baseline; robust and well-tested defaults.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from lrdbenchmark.analysis.base_estimator import BaseEstimator

class MFDFAEstimator(BaseEstimator):
    """
    Multifractal Detrended Fluctuation Analysis estimator.

    Returns h(q) curve and H ≡ h(2) for monofractal signals.
    """
    
    def __init__(
        self,
        qs: Optional[np.ndarray] = None,
        m: int = 1,
        window_scales: Optional[np.ndarray] = None,
        two_sided: bool = True,
    ) -> None:
        super().__init__()
        if qs is None:
            qs = np.linspace(-4, 4, 17)
        self.parameters = {
            "qs": np.asarray(qs, float),
            "m": int(max(0, m)),
            "two_sided": bool(two_sided),
            "window_scales": window_scales,  # set in estimate if None
        }
    
    def _validate_parameters(self) -> None:
        if self.parameters["m"] < 0:
            raise ValueError("Polynomial order m must be >= 0")
        if self.parameters["qs"].ndim != 1:
            raise ValueError("qs must be a 1D array")
    
    def estimate(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Estimate h(q) and monofractal H = h(2) using MFDFA.
        """
        x = np.asarray(data, float)
        n = x.size
        if n < 32:
            raise ValueError("Data length must be at least 32 for MFDFA")

        qs = self.parameters["qs"]
        m = self.parameters["m"]
        two_sided = self.parameters["two_sided"]

        # Profile (cumulative sum of mean-centered signal)
        y = np.cumsum(x - np.nanmean(x))

        # Window scales
        if self.parameters["window_scales"] is None:
            s = np.unique((np.geomspace(8, max(16, n // 8), 24)).astype(int))
        else:
            s = np.asarray(self.parameters["window_scales"], int)

        Fq = np.zeros((qs.size, s.size), dtype=float)
        t = None
        for si, seg in enumerate(s):
            if seg < m + 2:
                continue
            if two_sided:
                Ns = n // seg
                y1 = y[: Ns * seg]
                y2 = y[n - Ns * seg :]
                Ys = [y1.reshape(Ns, seg), y2.reshape(Ns, seg)]
                F2s_list: List[float] = []
                t = np.arange(seg, dtype=float)
                for Yblk in Ys:
                    for row in Yblk:
                        coeff = np.polyfit(t, row, m)
                        trend = np.polyval(coeff, t)
                        F2s_list.append(float(np.mean((row - trend) ** 2)))
                F2s = np.asarray(F2s_list, float)
            else:
                Ns = n // seg
                Y = y[: Ns * seg].reshape(Ns, seg)
                F2s_list = []
                t = np.arange(seg, dtype=float)
                for row in Y:
                    coeff = np.polyfit(t, row, m)
                    trend = np.polyval(coeff, t)
                    F2s_list.append(float(np.mean((row - trend) ** 2)))
                F2s = np.asarray(F2s_list, float)

            for qi, q in enumerate(qs):
                if np.isclose(q, 0.0):
                    Fq[qi, si] = float(np.exp(0.5 * np.mean(np.log(F2s + 1e-300))))
                else:
                    Fq[qi, si] = float((np.mean(F2s ** (q / 2.0))) ** (1.0 / q))

        log_s = np.log2(s.astype(float))
        hq = np.zeros_like(qs, dtype=float)
        for qi, q in enumerate(qs):
            yv = np.log2(Fq[qi] + 1e-300)
            slope, intercept = np.polyfit(log_s, yv, 1)
            hq[qi] = float(slope)

        # Monofractal H as h(2)
        idx_q2 = int(np.argmin(np.abs(qs - 2.0)))
        H_hat = float(hq[idx_q2])

        self.results = {
            "hurst_parameter": H_hat,
            "method": "mfdfa",
            "q": qs.tolist(),
            "h": hq.tolist(),
            "s": s.tolist(),
            "Fq": Fq.tolist(),
        }
        return self.results
