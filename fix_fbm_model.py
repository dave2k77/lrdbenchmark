#!/usr/bin/env python3
"""
Fix FBMModel parameter names from 'n' to 'length'.
"""

import re

def fix_fbm_model():
    """Fix the FBMModel file."""
    file_path = "lrdbenchmark/models/data_models/fbm/fbm_model.py"
    
    print(f"📝 Fixing {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix method signatures
    content = re.sub(r'def _generate_.*?\(self, n: int', lambda m: m.group(0).replace('n: int', 'length: int'), content)
    content = re.sub(r'def _.*?\(self, n: int', lambda m: m.group(0).replace('n: int', 'length: int'), content)
    
    # Fix parameter names in method bodies
    content = re.sub(r'\bn\b', 'length', content)
    
    # Fix specific method calls that should remain as 'n' (like in comments)
    content = re.sub(r'# .*?n\b', lambda m: m.group(0), content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated {file_path}")

if __name__ == "__main__":
    fix_fbm_model()
