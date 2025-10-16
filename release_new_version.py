#!/usr/bin/env python3
"""
Release a new version of LRDBenchmark with comprehensive improvements.
"""

import re
import subprocess
import sys

def update_version(new_version):
    """Update version in all relevant files."""
    
    # Update pyproject.toml
    with open('pyproject.toml', 'r') as f:
        content = f.read()
    
    new_content = re.sub(
        r'version = "[^"]+"',
        f'version = "{new_version}"',
        content
    )
    
    with open('pyproject.toml', 'w') as f:
        f.write(new_content)
    
    print(f"✅ Updated pyproject.toml to version {new_version}")
    
    # Update __init__.py
    with open('lrdbenchmark/__init__.py', 'r') as f:
        init_content = f.read()
    
    new_init_content = re.sub(
        r'__version__ = "[^"]+"',
        f'__version__ = "{new_version}"',
        init_content
    )
    
    with open('lrdbenchmark/__init__.py', 'w') as f:
        f.write(new_init_content)
    
    print(f"✅ Updated __init__.py to version {new_version}")

def build_and_upload():
    """Build and upload the package."""
    try:
        print("\n🔨 Building package...")
        subprocess.run(["python", "-m", "build"], check=True)
        
        print("\n📤 Uploading to PyPI...")
        subprocess.run(["twine", "upload", "dist/*"], check=True)
        
        print("✅ Package uploaded successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during build/upload: {e}")
    except FileNotFoundError:
        print("❌ Required tools not found. Install with: pip install build twine")

def main():
    """Main function."""
    print("🚀 LRDBenchmark New Version Release")
    print("=" * 40)
    
    # Get current version
    with open('pyproject.toml', 'r') as f:
        content = f.read()
    
    version_match = re.search(r'version = "([^"]+)"', content)
    current_version = version_match.group(1) if version_match else "unknown"
    
    print(f"Current version: {current_version}")
    
    # Suggest new version
    if current_version == "2.2.1":
        suggested_version = "2.3.0"  # Minor version bump for new features
    else:
        # Parse version and increment
        parts = current_version.split('.')
        if len(parts) >= 3:
            major, minor, patch = parts[0], parts[1], parts[2]
            suggested_version = f"{major}.{int(minor)+1}.0"
        else:
            suggested_version = "2.3.0"
    
    print(f"Suggested new version: {suggested_version}")
    
    new_version = input(f"Enter new version (or press Enter for {suggested_version}): ").strip()
    if not new_version:
        new_version = suggested_version
    
    print(f"\n📋 This will release version {new_version} with:")
    print("✅ Enhanced stability with custom exception hierarchy")
    print("✅ Lazy GPU initialization with CPU-first approach")
    print("✅ Persistent performance profiling")
    print("✅ Fixed JAX issues in data generation")
    print("✅ Comprehensive test coverage (>80%)")
    print("✅ Restructured dependencies (optional acceleration)")
    print("✅ Enhanced documentation with progressive examples")
    print("✅ Code cleanup (45 duplicate files removed)")
    print("✅ CI/CD pipeline with Python 3.8-3.12 support")
    
    response = input(f"\nProceed with version {new_version}? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        update_version(new_version)
        
        print(f"\n📋 Next steps:")
        print(f"1. Test the package: pip install lrdbenchmark=={new_version}")
        print(f"2. Commit changes: git add . && git commit -m 'Release v{new_version}'")
        print(f"3. Push to GitHub: git push origin master")
        print(f"4. Create GitHub release: git tag -a v{new_version} -m 'Release v{new_version}'")
        print(f"5. Upload to PyPI: python -m build && twine upload dist/*")
        
        upload_now = input(f"\nUpload to PyPI now? (y/N): ")
        if upload_now.lower() in ['y', 'yes']:
            build_and_upload()
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()
