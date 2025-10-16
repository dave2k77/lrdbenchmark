#!/usr/bin/env python3
"""
Fix all notebooks with the latest improvements.
"""

import json
import re
import os
from pathlib import Path

def fix_notebook(notebook_path):
    """Fix a single notebook with updated imports and class names."""
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Track changes
    changes_made = []
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            original_source = source
            
            # Replace old class names with new ones
            replacements = [
                ('FractionalBrownianMotion', 'FBMModel'),
                ('FractionalGaussianNoise', 'FGNModel'),
                ('MultifractalRandomWalk', 'MRWModel'),
                ('AlphaStableModel', 'AlphaStableModel'),  # Already correct
            ]
            
            for old_name, new_name in replacements:
                if old_name in source:
                    source = source.replace(old_name, new_name)
                    changes_made.append(f"Replaced {old_name} with {new_name}")
            
            # Replace old verbose imports with simplified ones
            import_replacements = [
                ('from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion', 
                 'from lrdbenchmark import FBMModel'),
                ('from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise', 
                 'from lrdbenchmark import FGNModel'),
                ('from lrdbenchmark.models.data_models.mrw.mrw_model import MultifractalRandomWalk', 
                 'from lrdbenchmark import MRWModel'),
                ('from lrdbenchmark.models.data_models.arfima.arfima_model import ARFIMAModel', 
                 'from lrdbenchmark import ARFIMAModel'),
                ('from lrdbenchmark.models.data_models.alpha_stable.alpha_stable_model import AlphaStableModel', 
                 'from lrdbenchmark import AlphaStableModel'),
            ]
            
            for old_import, new_import in import_replacements:
                if old_import in source:
                    source = source.replace(old_import, new_import)
                    changes_made.append(f"Replaced verbose import with simplified import")
            
            # Replace manual GPU memory management with new utilities
            gpu_replacements = [
                ('check_gpu_memory()', 'gpu_is_available()'),
                ('clear_gpu_memory()', 'clear_gpu_cache()'),
            ]
            
            for old_func, new_func in gpu_replacements:
                if old_func in source:
                    source = source.replace(old_func, new_func)
                    changes_made.append(f"Replaced {old_func} with {new_func}")
            
            # Add new GPU utilities import if not present
            if 'gpu_is_available' in source or 'clear_gpu_cache' in source:
                if 'from lrdbenchmark import' in source and 'gpu_is_available' not in source:
                    # Add GPU utilities to existing import
                    source = re.sub(
                        r'(from lrdbenchmark import [^\\n]*)',
                        r'\1, gpu_is_available, get_device_info, clear_gpu_cache, monitor_gpu_memory',
                        source
                    )
                    changes_made.append("Added GPU utilities to import")
            
            # Update the cell if changes were made
            if source != original_source:
                cell['source'] = source.split('\n')
                # Add newline to each line except the last
                for i in range(len(cell['source']) - 1):
                    cell['source'][i] += '\n'
    
    # Write the updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    return changes_made

def fix_readme():
    """Fix the README.md file."""
    readme_path = "notebooks/README.md"
    
    if not os.path.exists(readme_path):
        return []
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    # Replace old class names in documentation
    replacements = [
        ('FractionalBrownianMotion', 'FBMModel'),
        ('FractionalGaussianNoise', 'FGNModel'),
        ('MultifractalRandomWalk', 'MRWModel'),
    ]
    
    for old_name, new_name in replacements:
        if old_name in content:
            content = content.replace(old_name, new_name)
            changes_made.append(f"Replaced {old_name} with {new_name}")
    
    if content != original_content:
        with open(readme_path, 'w') as f:
            f.write(content)
    
    return changes_made

def main():
    """Fix all notebooks and documentation."""
    
    notebooks_dir = Path("notebooks")
    notebook_files = list(notebooks_dir.glob("*.ipynb"))
    
    print(f"📝 Found {len(notebook_files)} notebook files")
    
    total_changes = 0
    
    for notebook_path in notebook_files:
        print(f"\nUpdating {notebook_path}...")
        changes = fix_notebook(notebook_path)
        
        if changes:
            print(f"✅ Updated {notebook_path}")
            print("Changes made:")
            for change in changes:
                print(f"  - {change}")
            total_changes += len(changes)
        else:
            print(f"ℹ️  No updates needed for {notebook_path}")
    
    # Fix README
    print(f"\nUpdating notebooks/README.md...")
    readme_changes = fix_readme()
    
    if readme_changes:
        print(f"✅ Updated notebooks/README.md")
        print("Changes made:")
        for change in readme_changes:
            print(f"  - {change}")
        total_changes += len(readme_changes)
    else:
        print(f"ℹ️  No updates needed for notebooks/README.md")
    
    print(f"\n🎉 All notebooks updated successfully!")
    print(f"Total changes made: {total_changes}")
    
    if total_changes > 0:
        print("\nKey changes made:")
        print("✅ Replaced old class names with new simplified names")
        print("✅ Replaced verbose imports with simplified lrdbenchmark imports")
        print("✅ Replaced manual GPU memory management with new utilities")
        print("✅ Added GPU utilities (gpu_is_available, get_device_info, clear_gpu_cache)")

if __name__ == "__main__":
    main()
