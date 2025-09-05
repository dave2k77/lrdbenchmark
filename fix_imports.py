#!/usr/bin/env python3
"""
Fix import statements in __init__.py files after cleanup
"""

import os
import re

def fix_imports_in_file(filepath):
    """Fix import statements in a single file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Fix common import patterns
        replacements = [
            (r'from \.rs_estimator import', 'from .rs_estimator_unified import'),
            (r'from \.dma_estimator import', 'from .dma_estimator_unified import'),
            (r'from \.higuchi_estimator import', 'from .higuchi_estimator_unified import'),
            (r'from \.dfa_estimator import', 'from .dfa_estimator_unified import'),
            (r'from \.gph_estimator import', 'from .gph_estimator_unified import'),
            (r'from \.whittle_estimator import', 'from .whittle_estimator_unified import'),
            (r'from \.periodogram_estimator import', 'from .periodogram_estimator_unified import'),
            (r'from \.cwt_estimator import', 'from .cwt_estimator_unified import'),
            (r'from \.variance_estimator import', 'from .variance_estimator_unified import'),
            (r'from \.log_variance_estimator import', 'from .log_variance_estimator_unified import'),
            (r'from \.mfdfa_estimator import', 'from .mfdfa_estimator_unified import'),
            (r'from \.wavelet_leaders_estimator import', 'from .wavelet_leaders_estimator_unified import'),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Fixed imports in: {filepath}")
            return True
        else:
            print(f"No changes needed in: {filepath}")
            return False
            
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Fix all __init__.py files."""
    files_to_fix = [
        'lrdbenchmark/analysis/temporal/rs/__init__.py',
        'lrdbenchmark/analysis/temporal/dma/__init__.py',
        'lrdbenchmark/analysis/temporal/higuchi/__init__.py',
        'lrdbenchmark/analysis/temporal/dfa/__init__.py',
        'lrdbenchmark/analysis/spectral/gph/__init__.py',
        'lrdbenchmark/analysis/spectral/whittle/__init__.py',
        'lrdbenchmark/analysis/spectral/periodogram/__init__.py',
        'lrdbenchmark/analysis/wavelet/cwt/__init__.py',
        'lrdbenchmark/analysis/wavelet/variance/__init__.py',
        'lrdbenchmark/analysis/wavelet/log_variance/__init__.py',
        'lrdbenchmark/analysis/wavelet/whittle/__init__.py',
        'lrdbenchmark/analysis/multifractal/mfdfa/__init__.py',
        'lrdbenchmark/analysis/multifractal/wavelet_leaders/__init__.py',
    ]
    
    fixed_count = 0
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            if fix_imports_in_file(filepath):
                fixed_count += 1
        else:
            print(f"File not found: {filepath}")
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()
