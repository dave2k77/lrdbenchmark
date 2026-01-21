#!/usr/bin/env python3
"""
Baseline bias/variance/coverage benchmarking for classical LRD estimators.

This script measures, per estimator and data model, across seeds:
- bias: mean(H_hat - H_true)
- variance: var(H_hat)
- rmse: sqrt(mean((H_hat - H_true)^2))
- coverage_95: fraction of runs where the reported 95% CI contains H_true (if available)
- ci_width_mean: mean CI width (if available)

Outputs:
- CSV summary at benchmark_results/classical_baseline_summary.csv
- JSON with raw replicate results at benchmark_results/classical_baseline_raw.json

Quick start:
  python scripts/benchmarks/classical_baseline_biasvariance.py --H 0.7 --length 1000 --seeds 10

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import models and estimators from the package public API
from lrdbenchmark import (
    FBMModel,
    FGNModel,
    ARFIMAModel,
    RSEstimator,
    DFAEstimator,
    GPHEstimator,
    WhittleEstimator,
)


@dataclass
class RunConfig:
    hurst_values: List[float]
    lengths: List[int]
    seeds: int
    output_dir: Path
    include_models: List[str]
    include_estimators: List[str]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classical estimator baseline for bias/variance/coverage")
    parser.add_argument("--H", dest="hurst", type=float, nargs="*", default=[0.3, 0.5, 0.7, 0.9], help="Hurst values")
    parser.add_argument("--length", dest="length", type=int, nargs="*", default=[500, 1000, 4000], help="Series lengths")
    parser.add_argument("--seeds", dest="seeds", type=int, default=20, help="Number of seeds per configuration")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--models", dest="models", type=str, nargs="*", default=["fbm", "fgn"], help="Data models: fbm fgn arfima")
    parser.add_argument(
        "--estimators",
        dest="estimators",
        type=str,
        nargs="*",
        default=["rs", "dfa", "gph", "whittle"],
        help="Estimators: rs dfa gph whittle",
    )
    parser.add_argument("--min-window", type=int, default=16, help="R/S min window size")
    parser.add_argument("--num-windows", type=int, default=20, help="R/S number of windows")
    parser.add_argument("--min-scale", type=int, default=16, help="DFA min scale")
    parser.add_argument("--num-scales", type=int, default=20, help="DFA number of scales")
    parser.add_argument("--gph-min", type=float, default=0.005, help="GPH min freq ratio")
    parser.add_argument("--gph-max", type=float, default=0.05, help="GPH max freq ratio")
    return parser


def make_estimators(args: argparse.Namespace) -> Dict[str, Any]:
    estimators: Dict[str, Any] = {}

    if "rs" in args.estimators:
        estimators["R/S"] = RSEstimator(
            min_window_size=args.min_window,
            num_windows=args.num_windows,
            overlap=True,
        )

    if "dfa" in args.estimators:
        estimators["DFA"] = DFAEstimator(
            min_scale=args.min_scale,
            num_scales=args.num_scales,
            order=1,
        )

    if "gph" in args.estimators:
        estimators["GPH"] = GPHEstimator(
            min_freq_ratio=args.gph_min,
            max_freq_ratio=args.gph_max,
            use_welch=True,
        )

    if "whittle" in args.estimators:
        try:
            estimators["Whittle"] = WhittleEstimator()
        except Exception:
            pass

    return estimators


def make_models(models: List[str], H: float) -> Dict[str, Any]:
    model_map: Dict[str, Any] = {}
    for m in models:
        if m == "fbm":
            model_map["fbm"] = FBMModel(H=H, sigma=1.0)
        elif m == "fgn":
            model_map["fgn"] = FGNModel(H=H, sigma=1.0)
        elif m == "arfima":
            model_map["arfima"] = ARFIMAModel(d=H - 0.5, sigma=1.0)
    return model_map


def compute_metrics(values: List[float], true_h: float, cis: Optional[List[Tuple[float, float]]] = None) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    bias = float(np.mean(arr) - true_h)
    var = float(np.var(arr, ddof=1)) if len(arr) > 1 else float("nan")
    rmse = float(np.sqrt(np.mean((arr - true_h) ** 2)))

    metrics: Dict[str, float] = {"bias": bias, "variance": var, "rmse": rmse}

    if cis and len(cis) == len(arr):
        covered = 0
        widths: List[float] = []
        for lo, hi in cis:
            if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
                continue
            widths.append(hi - lo)
            if lo <= true_h <= hi:
                covered += 1
        if widths:
            metrics["coverage_95"] = covered / len(widths)
            metrics["ci_width_mean"] = float(np.mean(widths))
        else:
            metrics["coverage_95"] = float("nan")
            metrics["ci_width_mean"] = float("nan")
    else:
        metrics["coverage_95"] = float("nan")
        metrics["ci_width_mean"] = float("nan")

    return metrics


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    cfg = RunConfig(
        hurst_values=list(args.hurst),
        lengths=list(args.length),
        seeds=args.seeds,
        output_dir=Path(args.output_dir),
        include_models=list(args.models),
        include_estimators=list(args.estimators),
    )

    cfg.output_dir.mkdir(exist_ok=True)

    estimators = make_estimators(args)

    raw_records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for H in cfg.hurst_values:
        for n in cfg.lengths:
            models = make_models(cfg.include_models, H)
            for model_name, model in models.items():
                # Collect replicate estimates per estimator
                est_to_values: Dict[str, List[float]] = {name: [] for name in estimators.keys()}
                est_to_ci: Dict[str, List[Tuple[Optional[float], Optional[float]]]] = {name: [] for name in estimators.keys()}

                for seed in range(cfg.seeds):
                    try:
                        x = model.generate(length=n, seed=seed)
                    except TypeError:
                        # Some models use n instead of length
                        x = model.generate(n=n, seed=seed)

                    for est_name, est in estimators.items():
                        try:
                            res = est.estimate(x)
                            h_hat = float(res.get("hurst_parameter", np.nan))
                            ci = res.get("confidence_interval", [None, None])
                            lo = float(ci[0]) if ci and ci[0] is not None else None
                            hi = float(ci[1]) if ci and ci[1] is not None else None
                        except Exception as e:
                            h_hat = float("nan")
                            lo, hi = None, None

                        est_to_values[est_name].append(h_hat)
                        est_to_ci[est_name].append((lo, hi))

                        raw_records.append(
                            {
                                "estimator": est_name,
                                "model": model_name,
                                "H_true": H,
                                "length": n,
                                "seed": seed,
                                "H_hat": h_hat,
                                "ci_low": lo,
                                "ci_high": hi,
                            }
                        )

                # Summarise per estimator
                for est_name in estimators.keys():
                    values = [v for v in est_to_values[est_name] if np.isfinite(v)]
                    cis = est_to_ci[est_name]
                    metrics = compute_metrics(values, H, cis)

                    summary_row = {
                        "estimator": est_name,
                        "model": model_name,
                        "H_true": H,
                        "length": n,
                        **metrics,
                        "n_replicates": len(values),
                    }
                    summary_rows.append(summary_row)

    # Save outputs
    summary_df = pd.DataFrame(summary_rows)
    summary_path = cfg.output_dir / "classical_baseline_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    raw_path = cfg.output_dir / "classical_baseline_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw_records, f)

    print(f"Saved summary to: {summary_path}")
    print(f"Saved raw results to: {raw_path}")


if __name__ == "__main__":
    main()



