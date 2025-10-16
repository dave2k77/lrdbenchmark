#!/usr/bin/env python3
"""
Fix data model parameter names from 'n' to 'length' for consistency.
"""

import os
import re
from pathlib import Path

def fix_file(file_path):
    """Fix parameter names in a single file."""
    print(f"📝 Fixing {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix method signatures
        content = re.sub(r'def generate\(self, n: int', 'def generate(self, length: int', content)
        content = re.sub(r'def generate\(self, n=', 'def generate(self, length=', content)
        
        # Fix parameter documentation
        content = re.sub(r'n : int\n\s*Length of the time series', 'length : int\n            Length of the time series', content)
        content = re.sub(r'n : int\n\s*Number of points', 'length : int\n            Number of points', content)
        
        # Fix method calls within the same file
        content = re.sub(r'\(n,', '(length,', content)
        content = re.sub(r'\(n\)', '(length)', content)
        content = re.sub(r'= n', '= length', content)
        content = re.sub(r'< n', '< length', content)
        content = re.sub(r'> n', '> length', content)
        content = re.sub(r'<= n', '<= length', content)
        content = re.sub(r'>= n', '>= length', content)
        
        # Fix specific patterns
        content = re.sub(r'if n <= 0:', 'if length <= 0:', content)
        content = re.sub(r'if n < 10:', 'if length < 10:', content)
        content = re.sub(r'if n > 1000:', 'if length > 1000:', content)
        content = re.sub(r'if n > 500:', 'if length > 500:', content)
        content = re.sub(r'if n > 100:', 'if length > 100:', content)
        
        # Fix warning messages
        content = re.sub(r'f"Very short time series \(n=\{n\}\)', 'f"Very short time series (length={length})', content)
        content = re.sub(r'f"Large dataset \(n=\{n\}\)', 'f"Large dataset (length={length})', content)
        
        # Fix method parameter names in helper methods
        content = re.sub(r'def _generate_.*\(self, n: int', 'def _generate_\\1(self, length: int', content)
        content = re.sub(r'def _.*\(self, n: int', 'def _\\1(self, length: int', content)
        
        # Fix array creation
        content = re.sub(r'np\.zeros\(n\)', 'np.zeros(length)', content)
        content = re.sub(r'np\.ones\(n\)', 'np.ones(length)', content)
        content = re.sub(r'np\.arange\(n\)', 'np.arange(length)', content)
        content = re.sub(r'np\.linspace\(0, 1, n\)', 'np.linspace(0, 1, length)', content)
        content = re.sub(r'np\.random\.normal\(0, 1, n\)', 'np.random.normal(0, 1, length)', content)
        
        # Fix range and loop variables
        content = re.sub(r'for i in range\(n\):', 'for i in range(length):', content)
        content = re.sub(r'for i in range\(1, n\):', 'for i in range(1, length):', content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated {file_path}")
            return True
        else:
            print(f"ℹ️  No changes needed for {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all data model files."""
    print("🔧 Fixing Data Model Parameter Names")
    print("=" * 50)
    
    # Find all data model files
    data_models_dir = Path("lrdbenchmark/models/data_models")
    
    if not data_models_dir.exists():
        print("❌ Data models directory not found!")
        return
    
    # Get all Python files in data models
    python_files = list(data_models_dir.glob("**/*.py"))
    
    print(f"📚 Found {len(python_files)} Python files to check")
    
    updated_count = 0
    for file_path in python_files:
        if fix_file(file_path):
            updated_count += 1
    
    print(f"\n✅ Updated {updated_count} files")
    print("📋 Summary of changes:")
    print("- Changed 'n' parameter to 'length' in generate() methods")
    print("- Updated parameter documentation")
    print("- Fixed method calls and variable references")
    print("- Updated array creation and loop variables")

if __name__ == "__main__":
    main()
