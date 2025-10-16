#!/usr/bin/env python3
"""
Cleanup script to remove duplicate files and temporary development files.

This script identifies and removes:
- Duplicate files with similar functionality
- Temporary development files
- Old/obsolete implementations
- Unused test files
"""

import os
import shutil
from pathlib import Path
from typing import List, Set

def find_duplicate_files() -> List[str]:
    """Find potential duplicate files."""
    duplicates = []
    
    # Files to remove (duplicates or obsolete)
    files_to_remove = [
        # Duplicate JAX GPU setup files
        "config/jax_gpu_setup.py",  # Duplicate of tools/jax_gpu_setup.py
        
        # Temporary development files
        "temp_development_files/",
        
        # Debug files
        "debug_dfa_dma_scaling.py",
        "test_dfa_dma_investigation.py", 
        "test_fixed_dfa_dma.py",
        
        # Duplicate upload scripts
        "upload_to_pypi.py",  # Duplicate of upload_pypi.py
        
        # Old estimator implementations (keep only unified versions)
        "lrdbenchmark/analysis/temporal/rs/rs_estimator.py",  # Keep rs_estimator_unified.py
        "lrdbenchmark/analysis/temporal/dfa/dfa_estimator.py",  # Keep dfa_estimator_unified.py
        "lrdbenchmark/analysis/temporal/dfa/dfa_estimator_optimized.py",
        "lrdbenchmark/analysis/temporal/dfa/dfa_estimator_scipy_optimized.py",
        "lrdbenchmark/analysis/temporal/dfa/dfa_estimator_ultra_optimized.py",
        "lrdbenchmark/analysis/temporal/dfa/dfa_estimator_jax_optimized.py",
        "lrdbenchmark/analysis/temporal/dfa/dfa_estimator_numba_optimized.py",
        
        # Old Higuchi implementations
        "lrdbenchmark/analysis/temporal/higuchi/higuchi_estimator_old.py",
        
        # Old DMA implementations
        "lrdbenchmark/analysis/temporal/dma/dma_estimator_optimized.py",
        
        # Old GPH implementations
        "lrdbenchmark/analysis/spectral/gph/gph_estimator.py",
        "lrdbenchmark/analysis/spectral/gph/gph_estimator_improved.py",
        "lrdbenchmark/analysis/spectral/gph/gph_estimator_numba_optimized.py",
        
        # Old periodogram implementations
        "lrdbenchmark/analysis/spectral/periodogram/periodogram_estimator.py",
        "lrdbenchmark/analysis/spectral/periodogram/periodogram_estimator_improved.py",
        "lrdbenchmark/analysis/spectral/periodogram/periodogram_estimator_numba_optimized.py",
        
        # Old Whittle implementations
        "lrdbenchmark/analysis/spectral/whittle/whittle_estimator.py",
        "lrdbenchmark/analysis/spectral/whittle/whittle_estimator_numba_optimized.py",
        
        # Old CWT implementations
        "lrdbenchmark/analysis/wavelet/cwt/cwt_estimator.py",
        "lrdbenchmark/analysis/wavelet/cwt/cwt_estimator_improved.py",
        "lrdbenchmark/analysis/wavelet/cwt/cwt_estimator_numba_optimized.py",
        
        # Old wavelet implementations
        "lrdbenchmark/analysis/wavelet/log_variance/wavelet_log_variance_estimator.py",
        "lrdbenchmark/analysis/wavelet/log_variance/wavelet_log_variance_estimator_numba_optimized.py",
        "lrdbenchmark/analysis/wavelet/variance/wavelet_variance_estimator.py",
        "lrdbenchmark/analysis/wavelet/variance/wavelet_variance_estimator_numba_optimized.py",
        "lrdbenchmark/analysis/wavelet/whittle/wavelet_whittle_estimator.py",
        "lrdbenchmark/analysis/wavelet/whittle/wavelet_whittle_estimator_numba_optimized.py",
        
        # Old MFDFA implementations
        "lrdbenchmark/analysis/multifractal/mfdfa/mfdfa_estimator.py",
        "lrdbenchmark/analysis/multifractal/mfdfa/mfdfa_estimator_numba_optimized.py",
        
        # Old wavelet leaders implementations
        "lrdbenchmark/analysis/multifractal/wavelet_leaders/multifractal_wavelet_leaders_estimator.py",
        
        # Old ML implementations (keep only unified versions)
        "lrdbenchmark/analysis/machine_learning/random_forest_estimator.py",
        "lrdbenchmark/analysis/machine_learning/svr_estimator.py",
        "lrdbenchmark/analysis/machine_learning/gradient_boosting_estimator.py",
        
        # Old production files
        "lrdbenchmark/analysis/machine_learning/production_random_forest_estimator.py",
        "lrdbenchmark/analysis/machine_learning/production_ml_system.py",
        "lrdbenchmark/analysis/machine_learning/advanced_training_system.py",
        "lrdbenchmark/analysis/machine_learning/enhanced_ml_estimators.py",
        "lrdbenchmark/analysis/machine_learning/train_once_apply_many.py",
        "lrdbenchmark/analysis/machine_learning/unified_feature_extractor.py",
        "lrdbenchmark/analysis/machine_learning/ml_model_factory.py",
    ]
    
    return files_to_remove

def remove_files(files_to_remove: List[str], dry_run: bool = True) -> None:
    """Remove specified files and directories."""
    base_path = Path(".")
    
    for file_path in files_to_remove:
        full_path = base_path / file_path
        
        if full_path.exists():
            if dry_run:
                print(f"[DRY RUN] Would remove: {file_path}")
            else:
                try:
                    if full_path.is_dir():
                        shutil.rmtree(full_path)
                        print(f"Removed directory: {file_path}")
                    else:
                        full_path.unlink()
                        print(f"Removed file: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

def main():
    """Main cleanup function."""
    print("LRDBenchmark Cleanup Script")
    print("=" * 50)
    
    files_to_remove = find_duplicate_files()
    
    print(f"Found {len(files_to_remove)} files/directories to remove:")
    print()
    
    # Show what would be removed
    remove_files(files_to_remove, dry_run=True)
    
    print()
    response = input("Do you want to proceed with removal? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        print("\nRemoving files...")
        remove_files(files_to_remove, dry_run=False)
        print("\nCleanup completed!")
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    main()
