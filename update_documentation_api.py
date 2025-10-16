#!/usr/bin/env python3
"""
Update all documentation to use the new simplified API consistently.
"""

import os
import re
import json
from pathlib import Path

def update_notebook_cell(cell):
    """Update a notebook cell to use simplified imports."""
    if cell.get('cell_type') != 'code':
        return cell
    
    source = cell.get('source', [])
    if isinstance(source, str):
        source = [source]
    
    updated_source = []
    for line in source:
        # Update verbose imports to simplified ones
        line = re.sub(
            r'from lrdbenchmark\.models\.data_models\.fbm\.fbm_model import FractionalBrownianMotion',
            'from lrdbenchmark import FBMModel',
            line
        )
        line = re.sub(
            r'from lrdbenchmark\.models\.data_models\.fgn\.fgn_model import FractionalGaussianNoise',
            'from lrdbenchmark import FGNModel',
            line
        )
        line = re.sub(
            r'from lrdbenchmark\.models\.data_models\.mrw\.mrw_model import MultifractalRandomWalk',
            'from lrdbenchmark import MRWModel',
            line
        )
        line = re.sub(
            r'from lrdbenchmark\.models\.data_models\.alpha_stable\.alpha_stable_model import AlphaStableModel',
            'from lrdbenchmark import AlphaStableModel',
            line
        )
        line = re.sub(
            r'from lrdbenchmark\.models\.data_models\.arfima\.arfima_model import ARFIMAModel',
            'from lrdbenchmark import ARFIMAModel',
            line
        )
        
        # Update class names in usage
        line = re.sub(r'FractionalBrownianMotion\(', 'FBMModel(', line)
        line = re.sub(r'FractionalGaussianNoise\(', 'FGNModel(', line)
        line = re.sub(r'MultifractalRandomWalk\(', 'MRWModel(', line)
        
        # Update method calls
        line = re.sub(r'\.generate\(n=', '.generate(length=', line)
        
        updated_source.append(line)
    
    cell['source'] = updated_source
    return cell

def update_notebook(file_path):
    """Update a Jupyter notebook file."""
    print(f"📝 Updating {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Update all cells
        if 'cells' in notebook:
            notebook['cells'] = [update_notebook_cell(cell) for cell in notebook['cells']]
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Updated {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def update_python_file(file_path):
    """Update a Python file."""
    print(f"📝 Updating {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply the same replacements
        content = re.sub(
            r'from lrdbenchmark\.models\.data_models\.fbm\.fbm_model import FractionalBrownianMotion',
            'from lrdbenchmark import FBMModel',
            content
        )
        content = re.sub(
            r'from lrdbenchmark\.models\.data_models\.fgn\.fgn_model import FractionalGaussianNoise',
            'from lrdbenchmark import FGNModel',
            content
        )
        content = re.sub(
            r'from lrdbenchmark\.models\.data_models\.mrw\.mrw_model import MultifractalRandomWalk',
            'from lrdbenchmark import MRWModel',
            content
        )
        content = re.sub(
            r'from lrdbenchmark\.models\.data_models\.alpha_stable\.alpha_stable_model import AlphaStableModel',
            'from lrdbenchmark import AlphaStableModel',
            content
        )
        content = re.sub(
            r'from lrdbenchmark\.models\.data_models\.arfima\.arfima_model import ARFIMAModel',
            'from lrdbenchmark import ARFIMAModel',
            content
        )
        
        # Update class names
        content = re.sub(r'FractionalBrownianMotion\(', 'FBMModel(', content)
        content = re.sub(r'FractionalGaussianNoise\(', 'FGNModel(', content)
        content = re.sub(r'MultifractalRandomWalk\(', 'MRWModel(', content)
        
        # Update method calls
        content = re.sub(r'\.generate\(n=', '.generate(length=', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Updated {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all documentation."""
    print("🔄 Updating Documentation API Consistency")
    print("=" * 50)
    
    # Update notebooks
    notebook_dir = Path("notebooks")
    if notebook_dir.exists():
        print(f"\n📓 Updating Jupyter Notebooks in {notebook_dir}")
        for nb_file in notebook_dir.glob("*.ipynb"):
            update_notebook(nb_file)
    
    # Update examples
    examples_dir = Path("examples")
    if examples_dir.exists():
        print(f"\n📝 Updating Examples in {examples_dir}")
        for py_file in examples_dir.glob("*.py"):
            update_python_file(py_file)
    
    # Update docs
    docs_dir = Path("docs")
    if docs_dir.exists():
        print(f"\n📚 Updating Documentation in {docs_dir}")
        for rst_file in docs_dir.glob("**/*.rst"):
            update_python_file(rst_file)
        for md_file in docs_dir.glob("**/*.md"):
            update_python_file(md_file)
    
    print("\n✅ Documentation API consistency update complete!")
    print("\n📋 Summary of changes:")
    print("- Updated verbose imports to simplified API")
    print("- Updated class names (FractionalBrownianMotion → FBMModel)")
    print("- Updated method calls (generate(n= → generate(length=)")
    print("- Applied to notebooks, examples, and documentation")

if __name__ == "__main__":
    main()
