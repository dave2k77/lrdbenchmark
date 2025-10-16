#!/usr/bin/env python3
"""
Fix notebook 01 with the latest improvements.
"""

import json
import re

def fix_notebook_01():
    """Fix notebook 01 with updated imports and class names."""
    
    notebook_path = "notebooks/01_data_generation_and_visualisation.ipynb"
    
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
            
            # Update the cell if changes were made
            if source != original_source:
                cell['source'] = source.split('\n')
                # Add newline to each line except the last
                for i in range(len(cell['source']) - 1):
                    cell['source'][i] += '\n'
    
    # Write the updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Updated {notebook_path}")
    if changes_made:
        print("Changes made:")
        for change in changes_made:
            print(f"  - {change}")
    else:
        print("No changes needed")

if __name__ == "__main__":
    fix_notebook_01()
