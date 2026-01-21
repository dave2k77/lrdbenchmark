#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for Classical Estimator Failure Analysis

This script systematically tests classical LRD estimators under various
nonstationarity and contamination conditions to demonstrate their failure modes.

Usage:
    python run_classical_failure_benchmark.py --profile quick
    python run_classical_failure_benchmark.py --profile standard
    python run_classical_failure_benchmark.py --profile full --realizations 1000
    python run_classical_failure_benchmark.py --dry-run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

import numpy as np
import pandas as pd

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lrdbenchmark.generation import (
    TimeSeriesGenerator,
    RegimeSwitchingProcess,
    ContinuousDriftProcess,
    StructuralBreakProcess,
    EnsembleTimeAverageProcess
)
from lrdbenchmark.analysis.diagnostics import StructuralBreakDetector
from lrdbenchmark.benchmarks.classical_benchmark import ClassicalBenchmark


# ============================================================================
# Configuration Profiles
# ============================================================================

PROFILES = {
    "quick": {
        "H_values": [0.3, 0.5, 0.7],
        "lengths": [512, 1024],
        "realizations": 10,
        "nonstationarity": ["none", "regime_switching"],
        "description": "Quick screening (~5 minutes)"
    },
    "standard": {
        "H_values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "lengths": [512, 1024, 2048],
        "realizations": 100,
        "nonstationarity": ["none", "regime_switching", "continuous_drift", "structural_break"],
        "description": "Standard analysis (~1 hour)"
    },
    "full": {
        "H_values": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "lengths": [256, 512, 1024, 2048, 4096, 8192, 16384],
        "realizations": 500,
        "nonstationarity": ["none", "regime_switching", "continuous_drift", 
                           "structural_break", "ensemble_time_average"],
        "description": "Full publication-quality (~8-10 hours)"
    }
}

ESTIMATORS = [
    "RS", "DFA", "DMA", "Higuchi", "GHE",
    "GPH", "Periodogram", "Whittle",
    "CWT", "WaveletVar", "WaveletLogVar",
    "MFDFA"
]


# ============================================================================
# Nonstationarity Scenario Generators
# ============================================================================

def create_nonstationarity_scenario(
    scenario_type: str,
    H: float,
    length: int,
    seed: int
) -> Dict[str, Any]:
    """
    Create a nonstationary time series for a given scenario.
    
    Returns dict with 'signal', 'h_trajectory', 'metadata'.
    """
    if scenario_type == "none":
        # Stationary baseline using regular FGN
        from lrdbenchmark.models.data_models.fgn_model import FractionalGaussianNoise
        model = FractionalGaussianNoise(H=H)
        signal = model.generate(length=length, seed=seed)
        return {
            'signal': signal,
            'h_trajectory': np.full(length, H),
            'metadata': {'scenario': 'stationary', 'true_H': H}
        }
    
    elif scenario_type == "regime_switching":
        # H switches from low to high at midpoint
        h_low = max(0.15, H - 0.25)
        h_high = min(0.85, H + 0.25)
        gen = RegimeSwitchingProcess(
            h_regimes=[h_low, h_high],
            change_points=[0.5],
            random_state=seed
        )
        return gen.generate(length)
    
    elif scenario_type == "continuous_drift":
        # Linear drift from H-0.2 to H+0.2
        h_start = max(0.15, H - 0.2)
        h_end = min(0.85, H + 0.2)
        gen = ContinuousDriftProcess(
            h_start=h_start,
            h_end=h_end,
            drift_type='linear',
            random_state=seed
        )
        return gen.generate(length)
    
    elif scenario_type == "structural_break":
        # Single structural break with level shift
        h_before = max(0.15, H - 0.15)
        h_after = min(0.85, H + 0.15)
        gen = StructuralBreakProcess(
            h_before=h_before,
            h_after=h_after,
            break_position=0.5,
            break_severity=0.5,
            n_breaks=1,
            random_state=seed
        )
        return gen.generate(length)
    
    elif scenario_type == "ensemble_time_average":
        # Aging dynamics
        gen = EnsembleTimeAverageProcess(
            H=H,
            aging_exponent=0.5,
            aging_type='power_law',
            random_state=seed
        )
        return gen.generate(length)
    
    else:
        raise ValueError(f"Unknown scenario: {scenario_type}")


# ============================================================================
# Benchmark Runner
# ============================================================================

class ClassicalFailureBenchmark:
    """
    Comprehensive benchmark for classical estimator failure analysis.
    """
    
    def __init__(
        self,
        profile: str = "standard",
        output_dir: Optional[str] = None,
        seed: int = 42,
        checkpoint_every: int = 100
    ):
        self.profile = profile
        self.config = PROFILES[profile]
        self.seed = seed
        self.checkpoint_every = checkpoint_every
        
        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            output_dir = f"results/classical_failure_{profile}_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.break_detector = StructuralBreakDetector()
        self.results = []
        self.start_time = None
        
        # Initialize base benchmark for estimators
        self.base_benchmark = ClassicalBenchmark(seed=seed)
        
    def _get_total_runs(self) -> int:
        """Calculate total number of experimental runs."""
        return (
            len(self.config["H_values"]) *
            len(self.config["lengths"]) *
            self.config["realizations"] *
            len(self.config["nonstationarity"]) *
            len(ESTIMATORS)
        )
    
    def _format_eta(self, seconds: float) -> str:
        """Format seconds to human-readable ETA."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _save_checkpoint(self, run_index: int):
        """Save checkpoint of current results."""
        checkpoint_path = self.output_dir / f"checkpoint_{run_index}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(checkpoint_path, index=False)
        print(f"  💾 Checkpoint saved ({len(self.results)} results)")
    
    def _load_checkpoint(self) -> int:
        """Load most recent checkpoint if exists. Returns last run index."""
        checkpoints = list(self.output_dir.glob("checkpoint_*.csv"))
        if not checkpoints:
            return 0
        
        # Find latest checkpoint
        latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
        df = pd.read_csv(latest)
        self.results = df.to_dict('records')
        last_index = int(latest.stem.split("_")[1])
        print(f"  📂 Loaded checkpoint with {len(self.results)} results (run {last_index})")
        return last_index
    
    def run(self, resume: bool = True) -> pd.DataFrame:
        """
        Run the full benchmark.
        
        Parameters
        ----------
        resume : bool
            Whether to resume from checkpoint if available
            
        Returns
        -------
        pd.DataFrame
            Complete benchmark results
        """
        self.start_time = time.time()
        total_runs = self._get_total_runs()
        
        print(f"\n{'='*60}")
        print(f"Classical Estimator Failure Benchmark")
        print(f"{'='*60}")
        print(f"Profile: {self.profile}")
        print(f"Total estimations: {total_runs:,}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Resume from checkpoint
        start_index = 0
        if resume:
            start_index = self._load_checkpoint()
        
        # Create run iterator
        rng = np.random.default_rng(self.seed)
        run_index = 0
        
        for scenario in self.config["nonstationarity"]:
            for H in self.config["H_values"]:
                for length in self.config["lengths"]:
                    for real_idx in range(self.config["realizations"]):
                        
                        # Skip already completed runs
                        run_index += len(ESTIMATORS)
                        if run_index <= start_index:
                            continue
                        
                        # Generate data
                        data_seed = rng.integers(0, 2**32)
                        try:
                            data_result = create_nonstationarity_scenario(
                                scenario, H, length, data_seed
                            )
                        except Exception as e:
                            warnings.warn(f"Generation failed: {e}")
                            continue
                        
                        signal = data_result['signal']
                        h_trajectory = data_result['h_trajectory']
                        
                        # Run structural break detection (once per signal)
                        break_result = self.break_detector.detect_all(signal)
                        
                        # Run all estimators
                        for est_name in ESTIMATORS:
                            try:
                                estimator = self.base_benchmark.estimators[est_name]
                                est_result = self.base_benchmark._evaluate_estimator(
                                    estimator, signal, {'H': H}
                                )
                                
                                # Calculate true mean H (for nonstationary cases)
                                true_mean_H = np.mean(h_trajectory)
                                
                                # Store result
                                self.results.append({
                                    'scenario': scenario,
                                    'true_H': H,
                                    'true_mean_H': true_mean_H,
                                    'length': length,
                                    'realization': real_idx,
                                    'estimator': est_name,
                                    'estimated_H': est_result.get('hurst_parameter'),
                                    'execution_time': est_result.get('execution_time'),
                                    'success': est_result.get('success', False),
                                    'absolute_error': est_result.get('absolute_error'),
                                    'squared_error': est_result.get('squared_error'),
                                    'break_detected': break_result.get('any_break_detected'),
                                    'seed': data_seed
                                })
                                
                            except Exception as e:
                                self.results.append({
                                    'scenario': scenario,
                                    'true_H': H,
                                    'length': length,
                                    'realization': real_idx,
                                    'estimator': est_name,
                                    'estimated_H': None,
                                    'success': False,
                                    'error': str(e)
                                })
                        
                        # Progress update
                        if run_index % 100 == 0:
                            elapsed = time.time() - self.start_time
                            progress = run_index / total_runs
                            if progress > 0:
                                eta = elapsed / progress * (1 - progress)
                                print(f"  Progress: {run_index:,}/{total_runs:,} "
                                      f"({progress*100:.1f}%) - ETA: {self._format_eta(eta)}")
                        
                        # Checkpoint
                        if run_index % self.checkpoint_every == 0:
                            self._save_checkpoint(run_index)
        
        # Final save
        self._save_results()
        
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Benchmark complete! Total time: {self._format_eta(elapsed)}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return pd.DataFrame(self.results)
    
    def _save_results(self):
        """Save final results and summary."""
        df = pd.DataFrame(self.results)
        
        # Raw results
        df.to_csv(self.output_dir / "results.csv", index=False)
        
        # Summary statistics
        summary = self._compute_summary(df)
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Configuration
        config = {
            'profile': self.profile,
            'config': self.config,
            'seed': self.seed,
            'timestamp': datetime.now().isoformat(),
            'n_results': len(df)
        }
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def _compute_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics."""
        summary = {
            'total_runs': len(df),
            'success_rate': df['success'].mean() if 'success' in df else None,
            'by_estimator': {},
            'by_scenario': {},
            'failure_modes': []
        }
        
        # Per-estimator summaries
        for est in ESTIMATORS:
            est_df = df[df['estimator'] == est]
            if len(est_df) > 0 and 'absolute_error' in est_df:
                summary['by_estimator'][est] = {
                    'mae': est_df['absolute_error'].mean(),
                    'rmse': np.sqrt(est_df['squared_error'].mean()) if 'squared_error' in est_df else None,
                    'success_rate': est_df['success'].mean() if 'success' in est_df else None
                }
        
        # Per-scenario summaries
        for scenario in self.config["nonstationarity"]:
            scen_df = df[df['scenario'] == scenario]
            if len(scen_df) > 0 and 'absolute_error' in scen_df:
                summary['by_scenario'][scenario] = {
                    'mae': scen_df['absolute_error'].mean(),
                    'success_rate': scen_df['success'].mean() if 'success' in scen_df else None
                }
        
        return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classical Estimator Failure Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_classical_failure_benchmark.py --profile quick
  python run_classical_failure_benchmark.py --profile standard --output results/my_run
  python run_classical_failure_benchmark.py --profile full --realizations 1000
  python run_classical_failure_benchmark.py --dry-run
        """
    )
    
    parser.add_argument(
        '--profile', 
        choices=['quick', 'standard', 'full'],
        default='standard',
        help='Experiment profile (default: standard)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: auto-generated)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--realizations',
        type=int,
        default=None,
        help='Override number of realizations'
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=100,
        help='Checkpoint frequency (default: 100)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from checkpoint'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration without running'
    )
    
    args = parser.parse_args()
    
    # Override realizations if specified
    if args.realizations is not None:
        PROFILES[args.profile]['realizations'] = args.realizations
    
    if args.dry_run:
        config = PROFILES[args.profile]
        total_runs = (
            len(config["H_values"]) *
            len(config["lengths"]) *
            config["realizations"] *
            len(config["nonstationarity"]) *
            len(ESTIMATORS)
        )
        
        print(f"\n{'='*60}")
        print(f"DRY RUN - Configuration Summary")
        print(f"{'='*60}")
        print(f"Profile: {args.profile}")
        print(f"Description: {config['description']}")
        print(f"\nParameters:")
        print(f"  H values: {len(config['H_values'])} ({config['H_values'][0]} to {config['H_values'][-1]})")
        print(f"  Lengths: {config['lengths']}")
        print(f"  Realizations: {config['realizations']}")
        print(f"  Scenarios: {config['nonstationarity']}")
        print(f"  Estimators: {len(ESTIMATORS)}")
        print(f"\nTotal estimations: {total_runs:,}")
        print(f"{'='*60}\n")
        return
    
    # Run benchmark
    benchmark = ClassicalFailureBenchmark(
        profile=args.profile,
        output_dir=args.output,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every
    )
    
    results = benchmark.run(resume=not args.no_resume)
    print(f"Results shape: {results.shape}")


if __name__ == "__main__":
    main()
