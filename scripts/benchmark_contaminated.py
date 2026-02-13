#!/usr/bin/env python3
"""
Benchmark: Entropy vs Classical Estimators on Contaminated Data.

Tests robustness of MSE against classical H estimators under various
real-world contamination scenarios using the library's ContaminationModel.
"""

import os
import sys
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

# ── Contamination ──────────────────────────────────────────────────
from lrdbenchmark.models.contamination.contamination_models import (
    ContaminationModel,
    ContaminationConfig,
)


# ── Data generation ────────────────────────────────────────────────
def generate_fgn_approx(n, H, seed=42):
    """Approximate fGn via spectral synthesis."""
    rng = np.random.RandomState(seed)
    fft_len = 2 * n
    freqs = np.fft.rfftfreq(fft_len)[1:]
    power = np.zeros(len(freqs) + 1)
    power[1:] = freqs ** (-(2 * H - 1))
    power[0] = 0
    phases = rng.uniform(0, 2 * np.pi, len(power))
    fft_coeff = np.sqrt(power) * np.exp(1j * phases)
    signal = np.fft.irfft(fft_coeff, n=fft_len)[:n]
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
    return signal


def make_fbm(n, H, seed=42):
    """Generate fBm as cumulative sum of fGn."""
    return np.cumsum(generate_fgn_approx(n, H, seed))


# ── Contamination scenarios ────────────────────────────────────────
def build_scenarios(clean_fbm, clean_fgn):
    """Build dict of (label -> (fbm_data, fgn_data)) for each scenario."""
    np.random.seed(42)
    cm = ContaminationModel()

    scenarios = {}

    # 1. Clean baseline
    scenarios["Clean (baseline)"] = (clean_fbm.copy(), clean_fgn.copy())

    # 2. Additive Gaussian noise (SNR ~10dB)
    std = 0.3 * np.std(clean_fbm)
    scenarios["Gaussian noise"] = (
        cm.add_noise_gaussian(clean_fbm, std=std),
        cm.add_noise_gaussian(clean_fgn, std=0.3 * np.std(clean_fgn)),
    )

    # 3. Outlier spikes (2% probability, 5x amplitude)
    scenarios["Spikes (2%)"] = (
        cm.add_artifact_spikes(clean_fbm, probability=0.02, amplitude=5.0),
        cm.add_artifact_spikes(clean_fgn, probability=0.02, amplitude=5.0),
    )

    # 4. Level shifts
    scenarios["Level shifts"] = (
        cm.add_artifact_level_shifts(clean_fbm, probability=0.003, amplitude=3.0),
        cm.add_artifact_level_shifts(clean_fgn, probability=0.003, amplitude=3.0),
    )

    # 5. Linear trend
    scenarios["Linear trend"] = (
        cm.add_trend_linear(clean_fbm, slope=0.02),
        cm.add_trend_linear(clean_fgn, slope=0.02),
    )

    # 6. Seasonal trend
    scenarios["Seasonal trend"] = (
        cm.add_trend_seasonal(clean_fbm, period=100, amplitude=1.0),
        cm.add_trend_seasonal(clean_fgn, period=100, amplitude=1.0),
    )

    # 7. Impulsive noise (heavy-tailed)
    scenarios["Impulsive noise"] = (
        cm.add_noise_impulsive(clean_fbm, probability=0.01, amplitude=8.0),
        cm.add_noise_impulsive(clean_fgn, probability=0.01, amplitude=8.0),
    )

    # 8. Colored noise (1/f^2 contamination)
    scenarios["Colored noise (1/f^2)"] = (
        cm.add_noise_colored(clean_fbm, power=2.0, std=0.5),
        cm.add_noise_colored(clean_fgn, power=2.0, std=0.5),
    )

    return scenarios


# ── Estimator config ───────────────────────────────────────────────
ESTIMATORS = {
    "MSE":         ("fbm", lambda: MSEEstimator(max_scale=15)),
    "DFA":         ("fbm", lambda: DFAEstimator()),
    "R/S":         ("fbm", lambda: RSEstimator()),
    "Higuchi":     ("fbm", lambda: HiguchiEstimator()),
    "GPH":         ("fgn", lambda: GPHEstimator()),
    "Periodogram": ("fgn", lambda: PeriodogramEstimator()),
    "Whittle":     ("fgn", lambda: WhittleEstimator()),
}


# ── Main benchmark ─────────────────────────────────────────────────
def run_benchmark():
    N = 2048
    H_true = 0.7  # known ground truth

    clean_fgn = generate_fgn_approx(N, H_true, seed=42)
    clean_fbm = make_fbm(N, H_true, seed=42)

    scenarios = build_scenarios(clean_fbm, clean_fgn)

    lines = []
    lines.append("=" * 120)
    lines.append(f"{'BENCHMARK: Estimator Robustness Under Contamination':^120}")
    lines.append(f"{'N=' + str(N) + ', True H=' + str(H_true) + ', fGn spectral synthesis':^120}")
    lines.append("=" * 120)
    lines.append("")

    # Build header row
    est_names = list(ESTIMATORS.keys())
    header = f"{'Scenario':<24}"
    for name in est_names:
        header += f" {name:>12}"
    lines.append(header)

    sub_header = f"{'':<24}"
    for _ in est_names:
        sub_header += f" {'H_est (err)':>12}"
    lines.append(sub_header)
    lines.append("-" * 120)

    for scenario_name, (fbm_data, fgn_data) in scenarios.items():
        row = f"{scenario_name:<24}"

        for est_name, (data_type, factory) in ESTIMATORS.items():
            data = fbm_data if data_type == "fbm" else fgn_data

            # Handle NaN in data (replace with interpolation for estimators that can't handle it)
            if np.any(np.isnan(data)):
                mask = ~np.isnan(data)
                data_clean = np.interp(
                    np.arange(len(data)),
                    np.arange(len(data))[mask],
                    data[mask]
                )
            else:
                data_clean = data

            est = factory()
            try:
                results = est.estimate(data_clean)
                h_est = results["hurst_parameter"]
                err = h_est - H_true
                sign = "+" if err >= 0 else ""
                row += f" {h_est:5.3f}({sign}{err:.2f})"
            except Exception:
                row += f" {'ERROR':>12}"

        lines.append(row)

    lines.append("-" * 120)
    lines.append("")

    # ── Summary statistics ──────────────────────────────────────────
    lines.append("ROBUSTNESS SUMMARY (Mean Absolute Error across contamination scenarios):")
    lines.append("-" * 80)

    for est_name, (data_type, factory) in ESTIMATORS.items():
        errors = []
        for scenario_name, (fbm_data, fgn_data) in scenarios.items():
            data = fbm_data if data_type == "fbm" else fgn_data
            if np.any(np.isnan(data)):
                mask = ~np.isnan(data)
                data = np.interp(np.arange(len(data)), np.arange(len(data))[mask], data[mask])
            est = factory()
            try:
                results = est.estimate(data)
                errors.append(abs(results["hurst_parameter"] - H_true))
            except Exception:
                errors.append(float("nan"))

        mae = np.nanmean(errors)
        max_err = np.nanmax(errors)
        lines.append(f"  {est_name:<14}  MAE = {mae:.4f}   Max|err| = {max_err:.4f}")

    lines.append("")
    lines.append("Notes:")
    lines.append("  - Errors are H_est - H_true (positive = overestimate)")
    lines.append("  - MSE H_est is an approximate mapping from Complexity Index")
    lines.append("  - DFA/R/S/Higuchi operate on fBm;  GPH/Periodogram/Whittle on fGn")

    output = "\n".join(lines)
    print(output)

    # Write to file
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "benchmark_contaminated_results.txt"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_benchmark()
