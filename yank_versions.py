#!/usr/bin/env python3
"""
Script to yank (hide) specific versions from PyPI.
This hides versions from default installations but keeps them available for explicit requests.
"""

import subprocess
import sys

def yank_version(version, reason="Version contains issues"):
    """Yank a specific version from PyPI."""
    try:
        cmd = [
            "twine", "yank", 
            "--repository", "pypi",
            "--package", "lrdbenchmark",
            "--version", version,
            "--reason", reason
        ]
        
        print(f"Yanking version {version} from PyPI...")
        print(f"Reason: {reason}")
        print(f"Command: {' '.join(cmd)}")
        
        # Note: This requires authentication with PyPI
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully yanked version {version}")
        else:
            print(f"❌ Failed to yank version {version}")
            print(f"Error: {result.stderr}")
            
    except FileNotFoundError:
        print("❌ twine not found. Install with: pip install twine")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function to yank versions."""
    
    print("🔧 LRDBenchmark PyPI Version Yanking Tool")
    print("=" * 50)
    
    # List of versions to potentially yank
    versions_to_yank = [
        # Add versions you want to yank here
        # "2.0.0",  # Example
        # "2.1.0",  # Example
    ]
    
    if not versions_to_yank:
        print("ℹ️  No versions specified for yanking.")
        print("Edit this script to add versions you want to yank.")
        return
    
    print(f"Versions to yank: {versions_to_yank}")
    
    # Confirm before proceeding
    response = input("\n⚠️  This will hide these versions from new installations. Continue? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        for version in versions_to_yank:
            yank_version(version, f"Version {version} contains issues - use latest version")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()
