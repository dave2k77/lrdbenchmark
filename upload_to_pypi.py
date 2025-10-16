#!/usr/bin/env python3
"""
Manual PyPI upload script for LRDBenchmark v2.3.0
"""

import os
import subprocess
import sys

def upload_to_pypi():
    """Upload the built package to PyPI."""
    
    print("🚀 LRDBenchmark v2.3.0 PyPI Upload")
    print("=" * 40)
    
    # Check if packages exist
    if not os.path.exists("dist/lrdbenchmark-2.3.0.tar.gz"):
        print("❌ Source distribution not found!")
        return False
        
    if not os.path.exists("dist/lrdbenchmark-2.3.0-py3-none-any.whl"):
        print("❌ Wheel distribution not found!")
        return False
    
    print("✅ Found packages:")
    print("  📦 lrdbenchmark-2.3.0.tar.gz")
    print("  📦 lrdbenchmark-2.3.0-py3-none-any.whl")
    
    print("\n🔑 PyPI Upload Options:")
    print("1. Use PyPI API Token (recommended)")
    print("2. Use username/password")
    print("3. Use environment variables")
    
    choice = input("\nChoose option (1-3): ").strip()
    
    if choice == "1":
        # API Token method
        token = input("Enter your PyPI API token: ").strip()
        if not token:
            print("❌ No token provided!")
            return False
            
        cmd = [
            "twine", "upload", 
            "--username", "__token__",
            "--password", token,
            "dist/lrdbenchmark-2.3.0*"
        ]
        
    elif choice == "2":
        # Username/password method
        username = input("Enter PyPI username: ").strip()
        password = input("Enter PyPI password: ").strip()
        
        if not username or not password:
            print("❌ Username and password required!")
            return False
            
        cmd = [
            "twine", "upload",
            "--username", username,
            "--password", password,
            "dist/lrdbenchmark-2.3.0*"
        ]
        
    elif choice == "3":
        # Environment variables
        print("Set environment variables:")
        print("export TWINE_USERNAME=__token__")
        print("export TWINE_PASSWORD=your_api_token")
        print("Then run: twine upload dist/lrdbenchmark-2.3.0*")
        return True
        
    else:
        print("❌ Invalid choice!")
        return False
    
    print(f"\n📤 Uploading with command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully uploaded to PyPI!")
            print("🎉 lrdbenchmark v2.3.0 is now available!")
            print("\n📋 Next steps:")
            print("1. Verify: pip install lrdbenchmark==2.3.0")
            print("2. Check: https://pypi.org/project/lrdbenchmark/")
            return True
        else:
            print(f"❌ Upload failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function."""
    print("This script will help you upload LRDBenchmark v2.3.0 to PyPI.")
    print("You'll need your PyPI API token or username/password.")
    print("\nTo get a PyPI API token:")
    print("1. Go to https://pypi.org/account/login/")
    print("2. Go to Account Settings > API tokens")
    print("3. Create a new token with 'Upload packages' scope")
    print("4. Copy the token (starts with 'pypi-')")
    
    proceed = input("\nProceed with upload? (y/N): ").strip().lower()
    
    if proceed in ['y', 'yes']:
        upload_to_pypi()
    else:
        print("Upload cancelled.")

if __name__ == "__main__":
    main()
