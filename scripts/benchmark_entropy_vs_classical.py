#!/usr/bin/env python3
"""
Quick benchmark: Entropy-based estimators vs Classical estimators.

Generates synthetic time series with known LRD properties and compares
MSE-derived Hurst estimates against classical methods.
"""

import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── Estimators ──────────────────────────────────────────────────────
from lrdbenchmark.analysis.entropy.mse_estimator import MSEEstimator
from lrdbenchmark.analysis.temporal.dfa_estimator import DFAEstimator
from lrdbenchmark.analysis.temporal.rs_estimator import RSEstimator
from lrdbenchmark.analysis.temporal.higuchi_estimator import HiguchiEstimator
from lrdbenchmark.analysis.spectral.gph_estimator import GPHEstimator
from lrdbenchmark.analysis.spectral.periodogram_estimator import PeriodogramEstimator
from lrdbenchmark.analysis.spectral.whittle_estimator import WhittleEstimator

# ── Synthetic data generators ──────────────────────────────────────
def generate_fbm(n, H, seed=42):
    """Generate fractional Brownian motion (approximate via Cholesky)."""
    rng = np.random.RandomState(seed)
    # Use simple cumulative sum of fractional Gaussian noise approximation
    # For a quick benchmark, random walk (H≈0.5 increments) scaled works
    increments = rng.randn(n)
    # Apply fractional differencing approximation
    fbm = np.cumsum(increments)
    return fbm


def generate_fgn_approx(n, H, seed=42):
    """Generate approximate fractional Gaussian noise via spectral method."""
    rng = np.random.RandomState(seed)
    # Spectral synthesis (Davies-Harte approximation)
    fft_len = 2 * n
    freqs = np.fft.rfftfreq(fft_len)[1:]  # skip DC
    # Power spectrum of fGn: S(f) ~ |f|^{-(2H-1)}
    power = np.zeros(len(freqs) + 1)
    power[1:] = freqs ** (-(2 * H - 1))
    power[0] = 0
    # Generate in frequency domain
    phases = rng.uniform(0, 2 * np.pi, len(power))
    fft_coeff = np.sqrt(power) * np.exp(1j * phases)
    signal = np.fft.irfft(fft_coeff, n=fft_len)[:n]
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
    return signal


# ── Benchmark ──────────────────────────────────────────────────────
def run_benchmark():
    N = 2048
    H_values = [0.3, 0.5, 0.7, 0.9]

    estimators = {
        "MSE (RCMSE)":    lambda: MSEEstimator(max_scale=15),
        "DFA":            lambda: DFAEstimator(),
        "R/S":            lambda: RSEstimator(),
        "Higuchi":        lambda: HiguchiEstimator(),
        "GPH":            lambda: GPHEstimator(),
        "Periodogram":    lambda: PeriodogramEstimator(),
        "Whittle MLE":    lambda: WhittleEstimator(),
    }

    lines = []

    # Header
    lines.append("=" * 90)
    lines.append(f"{'BENCHMARK: Entropy vs Classical Estimators':^90}")
    lines.append(f"{'N=' + str(N) + ', fGn spectral synthesis':^90}")
    lines.append("=" * 90)
    lines.append("")

    # Column header
    header = f"{'Estimator':<16}"
    for H in H_values:
        header += f"  H={H:.1f} (H_est / time)"
        header += " "
    lines.append(header)
    lines.append("-" * 90)

    for name, factory in estimators.items():
        row = f"{name:<16}"
        for H in H_values:
            # Generate fGn with target H then cumsum for fBm
            if name in ("DFA", "R/S", "Higuchi"):
                # These expect fBm (cumulative process)
                data = np.cumsum(generate_fgn_approx(N, H, seed=42))
            else:
                # Spectral / entropy methods work on fGn or fBm
                # Use fGn for spectral; fBm for MSE (it measures persistence)
                if "MSE" in name:
                    data = np.cumsum(generate_fgn_approx(N, H, seed=42))
                else:
                    data = generate_fgn_approx(N, H, seed=42)

            est = factory()
            t0 = time.perf_counter()
            try:
                results = est.estimate(data)
                h_est = results["hurst_parameter"]
                elapsed = time.perf_counter() - t0
                row += f"  {h_est:5.3f} / {elapsed:5.3f}s  "
            except Exception as e:
                elapsed = time.perf_counter() - t0
                row += f"  ERROR / {elapsed:5.3f}s  "

        lines.append(row)

    lines.append("-" * 90)
    lines.append("")
    lines.append("Notes:")
    lines.append("  - MSE returns an *approximate* Hurst parameter derived from the")
    lines.append("    entropy curve slope (Complexity Index is the native metric).")
    lines.append("  - DFA, R/S, Higuchi operate on fBm (cumulative sum of fGn).")
    lines.append("  - GPH, Periodogram, Whittle operate on fGn directly.")
    lines.append("  - MSE operates on fBm to measure persistence across scales.")

    output = "\n".join(lines)
    print(output)

    # Also write to file for reliable capture
    import os
    out_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)


if __name__ == "__main__":
    run_benchmark()
