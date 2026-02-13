#!/usr/bin/env python3
"""
Benchmark: Estimator Robustness Across Realistic Scenarios.

Uses the library's ContaminationFactory to apply domain-specific confounding
profiles (financial, physiological, environmental, network, industrial, EEG)
and measures how well each estimator recovers the true Hurst parameter.
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

# ── Contamination Factory ──────────────────────────────────────────
from lrdbenchmark.models.contamination.contamination_factory import (
    ContaminationFactory,
    ConfoundingScenario,
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


# ── Estimator config ───────────────────────────────────────────────
# We use fGn for all estimators here for a fairer comparison.
# Spectral methods naturally work on fGn.
# For temporal methods (DFA/R/S/Higuchi), we use fGn too — DFA on fGn
# gives alpha = H directly (rather than alpha = H+1 on fBm).
# For MSE, we use fBm (cumsum of fGn) since it measures persistence.
ESTIMATORS = {
    "MSE":         ("fbm",  lambda: MSEEstimator(max_scale=15)),
    "DFA":         ("fgn",  lambda: DFAEstimator()),
    "R/S":         ("fgn",  lambda: RSEstimator()),
    "Higuchi":     ("fgn",  lambda: HiguchiEstimator()),
    "GPH":         ("fgn",  lambda: GPHEstimator()),
    "Periodogram": ("fgn",  lambda: PeriodogramEstimator()),
    "Whittle":     ("fgn",  lambda: WhittleEstimator()),
}


# ── Scenario groups ────────────────────────────────────────────────
SCENARIO_GROUPS = {
    "FINANCIAL": [
        ("Fin: Crash",       ConfoundingScenario.FINANCIAL_CRASH),
        ("Fin: Vol.Cluster", ConfoundingScenario.FINANCIAL_VOLATILITY_CLUSTERING),
        ("Fin: Regime Chg",  ConfoundingScenario.FINANCIAL_REGIME_CHANGE),
    ],
    "PHYSIOLOGICAL": [
        ("Phys: Sensor Drft",  ConfoundingScenario.PHYSIOLOGICAL_SENSOR_DRIFT),
        ("Phys: Motion Art.",   ConfoundingScenario.PHYSIOLOGICAL_MOTION_ARTIFACTS),
        ("Phys: Equip Fail",   ConfoundingScenario.PHYSIOLOGICAL_EQUIPMENT_FAILURE),
    ],
    "ENVIRONMENTAL": [
        ("Env: Seasonal",       ConfoundingScenario.ENVIRONMENTAL_SEASONAL),
        ("Env: Extreme Evt",    ConfoundingScenario.ENVIRONMENTAL_EXTREME_EVENTS),
        ("Env: Meas. Drift",    ConfoundingScenario.ENVIRONMENTAL_MEASUREMENT_DRIFT),
    ],
    "NETWORK": [
        ("Net: Bursts",         ConfoundingScenario.NETWORK_BURSTS),
        ("Net: Congestion",     ConfoundingScenario.NETWORK_CONGESTION),
        ("Net: Equip Fail",     ConfoundingScenario.NETWORK_EQUIPMENT_FAILURE),
    ],
    "INDUSTRIAL": [
        ("Ind: Calib Drift",    ConfoundingScenario.INDUSTRIAL_CALIBRATION_DRIFT),
        ("Ind: Sensor Aging",   ConfoundingScenario.INDUSTRIAL_SENSOR_AGING),
        ("Ind: Env Interf.",    ConfoundingScenario.INDUSTRIAL_ENVIRONMENTAL_INTERFERENCE),
    ],
    "EEG": [
        ("EEG: Ocular",        ConfoundingScenario.EEG_OCULAR_ARTIFACTS),
        ("EEG: Muscle",        ConfoundingScenario.EEG_MUSCLE_ARTIFACTS),
        ("EEG: Cardiac",       ConfoundingScenario.EEG_CARDIAC_ARTIFACTS),
        ("EEG: Elec Pop",      ConfoundingScenario.EEG_ELECTRODE_POPPING),
        ("EEG: Elec Drift",    ConfoundingScenario.EEG_ELECTRODE_DRIFT),
        ("EEG: 60Hz Noise",    ConfoundingScenario.EEG_60HZ_NOISE),
        ("EEG: Sweat",         ConfoundingScenario.EEG_SWEAT_ARTIFACTS),
        ("EEG: Movement",      ConfoundingScenario.EEG_MOVEMENT_ARTIFACTS),
    ],
    "MIXED": [
        ("Mixed: Light",    ConfoundingScenario.MIXED_REALISTIC_LIGHT),
        ("Mixed: Moderate", ConfoundingScenario.MIXED_REALISTIC_MODERATE),
        ("Mixed: Severe",   ConfoundingScenario.MIXED_REALISTIC_SEVERE),
    ],
}


def safe_estimate(est, data):
    """Run estimate, returning H or NaN on error."""
    # Handle NaN in data
    if np.any(np.isnan(data)):
        mask = ~np.isnan(data)
        if mask.sum() < 100:
            return float("nan")
        data = np.interp(np.arange(len(data)), np.arange(len(data))[mask], data[mask])

    # Handle inf in data
    if np.any(np.isinf(data)):
        data = np.clip(data, -1e10, 1e10)

    try:
        results = est.estimate(data)
        return results["hurst_parameter"]
    except Exception:
        return float("nan")


def run_benchmark():
    N = 2048
    H_true = 0.7
    factory = ContaminationFactory(random_seed=42)

    clean_fgn = generate_fgn_approx(N, H_true, seed=42)
    clean_fbm = np.cumsum(clean_fgn)

    lines = []
    lines.append("=" * 130)
    lines.append(f"{'BENCHMARK: Estimator Robustness Across Realistic Scenarios':^130}")
    lines.append(f"{'N=' + str(N) + ', True H=' + str(H_true) + ', fGn spectral synthesis, ContaminationFactory':^130}")
    lines.append("=" * 130)
    lines.append("")

    est_names = list(ESTIMATORS.keys())
    header = f"{'Scenario':<22}"
    for name in est_names:
        header += f" {name:>12}"
    lines.append(header)

    sub_header = f"{'':<22}"
    for _ in est_names:
        sub_header += f" {'H_est (err)':>12}"
    lines.append(sub_header)
    lines.append("-" * 130)

    # Baseline (clean)
    row = f"{'Clean (baseline)':<22}"
    for est_name, (data_type, factory_fn) in ESTIMATORS.items():
        data = clean_fbm if data_type == "fbm" else clean_fgn
        h = safe_estimate(factory_fn(), data)
        err = h - H_true
        sign = "+" if err >= 0 else ""
        row += f" {h:5.3f}({sign}{err:.2f})"
    lines.append(row)
    lines.append("-" * 130)

    # Per-domain results + per-domain MAE tracking
    domain_mae = {}  # domain -> {est_name: [errors]}
    all_errors = {name: [] for name in est_names}

    for group_name, scenarios in SCENARIO_GROUPS.items():
        lines.append(f"  -- {group_name} --")
        domain_mae[group_name] = {name: [] for name in est_names}

        for label, scenario in scenarios:
            row = f"{label:<22}"
            for est_name, (data_type, factory_fn) in ESTIMATORS.items():
                # Apply contamination to the appropriate signal
                data_src = clean_fbm if data_type == "fbm" else clean_fgn
                contaminated, _ = factory.apply_confounding(data_src, scenario)

                h = safe_estimate(factory_fn(), contaminated)
                err = h - H_true
                abs_err = abs(err)
                sign = "+" if err >= 0 else ""

                if np.isnan(h):
                    row += f" {'NaN':>12}"
                else:
                    row += f" {h:5.3f}({sign}{err:.2f})"

                domain_mae[group_name][est_name].append(abs_err)
                all_errors[est_name].append(abs_err)

            lines.append(row)

        lines.append("")

    lines.append("=" * 130)
    lines.append("")

    # ── Per-domain MAE summary ──────────────────────────────────
    lines.append("PER-DOMAIN MAE SUMMARY:")
    lines.append("-" * 130)
    header2 = f"{'Domain':<18}"
    for name in est_names:
        header2 += f" {name:>12}"
    lines.append(header2)
    lines.append("-" * 130)

    for group_name in SCENARIO_GROUPS:
        row = f"{group_name:<18}"
        for est_name in est_names:
            errs = domain_mae[group_name][est_name]
            mae = np.nanmean(errs) if errs else float("nan")
            row += f" {mae:12.4f}"
        lines.append(row)

    lines.append("-" * 130)

    # Overall MAE
    row = f"{'OVERALL':<18}"
    for est_name in est_names:
        mae = np.nanmean(all_errors[est_name])
        row += f" {mae:12.4f}"
    lines.append(row)
    lines.append("")

    # ── Ranking ──────────────────────────────────────────────────
    lines.append("OVERALL ROBUSTNESS RANKING:")
    lines.append("-" * 50)
    ranking = []
    for est_name in est_names:
        mae = np.nanmean(all_errors[est_name])
        max_err = np.nanmax(all_errors[est_name])
        ranking.append((est_name, mae, max_err))
    ranking.sort(key=lambda x: x[1])

    for rank, (name, mae, max_err) in enumerate(ranking, 1):
        lines.append(f"  {rank}. {name:<14}  MAE = {mae:.4f}   Max|err| = {max_err:.4f}")

    lines.append("")
    lines.append("Notes:")
    lines.append("  - ContaminationFactory applies domain-specific confounding at default intensity")
    lines.append("  - DFA/R/S/Higuchi operate on fGn directly (alpha = H for fGn)")
    lines.append("  - GPH/Periodogram/Whittle operate on fGn")
    lines.append("  - MSE operates on fBm (cumsum of fGn) to measure persistence across scales")
    lines.append("  - MSE H_est is an approximate mapping from Complexity Index (not a direct H estimate)")

    output = "\n".join(lines)
    print(output)

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "benchmark_realistic_results.txt"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_benchmark()
