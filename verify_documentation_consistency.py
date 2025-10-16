#!/usr/bin/env python3
"""
Verify that all documentation is consistent with the simplified API.
"""

import os
import re
from pathlib import Path

def check_file_consistency(file_path):
    """Check if a file uses the simplified API consistently."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for old verbose imports
        verbose_imports = [
            r'from lrdbenchmark\.models\.data_models\.fbm\.fbm_model import FractionalBrownianMotion',
            r'from lrdbenchmark\.models\.data_models\.fgn\.fgn_model import FractionalGaussianNoise',
            r'from lrdbenchmark\.models\.data_models\.mrw\.mrw_model import MultifractalRandomWalk',
            r'from lrdbenchmark\.analysis\.temporal\.rs\.rs_estimator import RSEstimator',
            r'from lrdbenchmark\.analysis\.temporal\.dfa\.dfa_estimator import DFAEstimator',
        ]
        
        for pattern in verbose_imports:
            if re.search(pattern, content):
                issues.append(f"Found verbose import: {pattern}")
        
        # Check for old class names in usage
        old_class_usage = [
            r'FractionalBrownianMotion\(',
            r'FractionalGaussianNoise\(',
            r'MultifractalRandomWalk\(',
        ]
        
        for pattern in old_class_usage:
            if re.search(pattern, content):
                issues.append(f"Found old class name usage: {pattern}")
        
        # Check for old method calls
        if re.search(r'\.generate\(n=', content):
            issues.append("Found old method call: .generate(n=")
        
        return issues
        
    except Exception as e:
        return [f"Error reading file: {e}"]

def main():
    """Check all documentation files for consistency."""
    print("🔍 Verifying Documentation API Consistency")
    print("=" * 50)
    
    # Check different file types
    file_patterns = [
        ("README.md", "README.md"),
        ("docs/**/*.rst", "RST Documentation"),
        ("docs/**/*.md", "Markdown Documentation"),
        ("notebooks/*.ipynb", "Jupyter Notebooks"),
        ("examples/*.py", "Example Files"),
    ]
    
    total_issues = 0
    files_checked = 0
    
    for pattern, description in file_patterns:
        print(f"\n📚 Checking {description}")
        print("-" * 30)
        
        files = list(Path(".").glob(pattern))
        if not files:
            print(f"ℹ️  No files found matching {pattern}")
            continue
        
        for file_path in files:
            if file_path.is_file():
                files_checked += 1
                issues = check_file_consistency(file_path)
                
                if issues:
                    print(f"❌ {file_path}:")
                    for issue in issues:
                        print(f"   - {issue}")
                    total_issues += len(issues)
                else:
                    print(f"✅ {file_path}")
    
    print(f"\n📊 Summary:")
    print(f"   Files checked: {files_checked}")
    print(f"   Total issues: {total_issues}")
    
    if total_issues == 0:
        print("🎉 All documentation is consistent with the simplified API!")
    else:
        print(f"⚠️  Found {total_issues} issues that need attention.")
        print("   Run the update scripts to fix these issues.")
    
    return total_issues == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
