#!/usr/bin/env python3
"""
Update all notebooks to use the new simplified LRDBenchmark API.

This script updates all notebooks to:
1. Use simplified imports from lrdbenchmark
2. Use new GPU utilities instead of manual memory management
3. Remove verbose import paths
4. Add proper error handling
"""

import json
import os
from pathlib import Path


def update_notebook_imports(notebook_path):
    """Update a single notebook to use new imports."""
    print(f"Updating {notebook_path}...")
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    updated = False
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            # Replace old verbose imports with new simplified ones
            old_imports = [
                "from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion",
                "from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise", 
                "from lrdbenchmark.models.data_models.arfima.arfima_model import ARFIMAModel",
                "from lrdbenchmark.models.data_models.mrw.mrw_model import MultifractalRandomWalk",
                "from lrdbenchmark.models.data_models.alpha_stable.alpha_stable_model import AlphaStableModel",
                "from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator",
                "from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator",
                "from lrdbenchmark.analysis.spectral.gph.gph_estimator_unified import GPHEstimator",
                "from lrdbenchmark.analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator",
                "from lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified import RandomForestEstimator",
                "from lrdbenchmark.analysis.machine_learning.svr_estimator_unified import SVREstimator",
                "from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator_unified import GradientBoostingEstimator",
                "from lrdbenchmark.analysis.machine_learning.cnn_estimator_unified import CNNEstimator",
                "from lrdbenchmark.analysis.machine_learning.lstm_estimator_unified import LSTMEstimator",
                "from lrdbenchmark.analysis.machine_learning.gru_estimator_unified import GRUEstimator",
                "from lrdbenchmark.analysis.machine_learning.transformer_estimator_unified import TransformerEstimator"
            ]
            
            # Check if any old imports exist
            has_old_imports = any(old_import in source for old_import in old_imports)
            
            if has_old_imports:
                # Remove old imports
                for old_import in old_imports:
                    source = source.replace(old_import + '\n', '')
                
                # Add new simplified imports
                if 'from lrdbenchmark import' not in source:
                    # Find the right place to insert imports
                    lines = source.split('\n')
                    insert_idx = 0
                    for i, line in enumerate(lines):
                        if 'import numpy' in line or 'import pandas' in line:
                            insert_idx = i + 1
                            break
                    
                    # Insert new imports
                    new_imports = """# LRDBenchmark imports - using simplified API
from lrdbenchmark import (
    # Data models
    FBMModel, FGNModel, ARFIMAModel, MRWModel, AlphaStableModel,
    # Classical estimators  
    RSEstimator, DFAEstimator, GPHEstimator, WhittleEstimator,
    # Machine Learning estimators
    RandomForestEstimator, SVREstimator, GradientBoostingEstimator,
    # Neural Network estimators
    CNNEstimator, LSTMEstimator, GRUEstimator, TransformerEstimator,
    # GPU utilities
    gpu_is_available, get_device_info, clear_gpu_cache, monitor_gpu_memory
)"""
                    
                    lines.insert(insert_idx, new_imports)
                    source = '\n'.join(lines)
                
                # Replace manual GPU memory management with new utilities
                if 'def check_gpu_memory():' in source:
                    # Remove old GPU functions
                    lines = source.split('\n')
                    new_lines = []
                    skip_until_def = False
                    
                    for line in lines:
                        if 'def check_gpu_memory():' in line:
                            skip_until_def = True
                            continue
                        elif skip_until_def and line.strip() == '' and 'def ' in lines[lines.index(line) + 1] if lines.index(line) + 1 < len(lines) else True:
                            skip_until_def = False
                            continue
                        elif skip_until_def:
                            continue
                        else:
                            new_lines.append(line)
                    
                    source = '\n'.join(new_lines)
                
                # Add new GPU status check
                if 'print("✅ All imports successful!")' in source:
                    source = source.replace(
                        'print("✅ All imports successful!")',
                        '''# Check GPU status using new utilities
print("🔍 Checking system status...")
print(f"GPU Available: {gpu_is_available()}")
if gpu_is_available():
    device_info = get_device_info()
    print(f"GPU Device: {device_info.get('device_name', 'Unknown')}")
    print(f"GPU Memory: {device_info.get('memory_free', 0):.1f}GB free")

# Clear any existing GPU cache
clear_gpu_cache()

print("✅ All imports successful!")'''
                    )
                
                # Update class names
                source = source.replace('FractionalBrownianMotion', 'FBMModel')
                source = source.replace('FractionalGaussianNoise', 'FGNModel')
                source = source.replace('MultifractalRandomWalk', 'MRWModel')
                
                cell['source'] = source.split('\n')
                updated = True
    
    if updated:
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        print(f"✅ Updated {notebook_path}")
    else:
        print(f"ℹ️  No updates needed for {notebook_path}")


def main():
    """Update all notebooks in the notebooks directory."""
    notebooks_dir = Path("notebooks")
    
    if not notebooks_dir.exists():
        print("❌ Notebooks directory not found")
        return
    
    notebook_files = list(notebooks_dir.glob("*.ipynb"))
    
    if not notebook_files:
        print("❌ No notebook files found")
        return
    
    print(f"📝 Found {len(notebook_files)} notebook files")
    
    for notebook_path in notebook_files:
        if notebook_path.name.startswith('.'):
            continue
        update_notebook_imports(notebook_path)
    
    print("\n🎉 All notebooks updated successfully!")
    print("\nKey changes made:")
    print("✅ Replaced verbose imports with simplified lrdbenchmark imports")
    print("✅ Added new GPU utilities (gpu_is_available, get_device_info, clear_gpu_cache)")
    print("✅ Removed manual GPU memory management functions")
    print("✅ Updated class names (FractionalBrownianMotion → FBMModel, etc.)")
    print("✅ Added proper GPU status checking")


if __name__ == "__main__":
    main()
