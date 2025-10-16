#!/usr/bin/env python3
"""
Create a post-release for minor fixes without changing functionality.
Use this for documentation updates, metadata fixes, etc.
"""

import re
import subprocess
import sys

def create_post_release():
    """Create a post-release version."""
    
    # Read current version
    with open('pyproject.toml', 'r') as f:
        content = f.read()
    
    version_match = re.search(r'version = "([^"]+)"', content)
    if not version_match:
        print("❌ Could not find version in pyproject.toml")
        return
    
    current_version = version_match.group(1)
    print(f"Current version: {current_version}")
    
    # Check if it's already a post-release
    if '.post' in current_version:
        # Extract base version and increment post number
        base_version = current_version.split('.post')[0]
        post_num = int(current_version.split('.post')[1]) + 1
        new_version = f"{base_version}.post{post_num}"
    else:
        # Create first post-release
        new_version = f"{current_version}.post1"
    
    print(f"New post-release version: {new_version}")
    
    # Update pyproject.toml
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
    
    return new_version

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
    print("📦 LRDBenchmark Post-Release Creator")
    print("=" * 40)
    
    print("This will create a post-release for minor fixes.")
    print("Use this for documentation updates, metadata fixes, etc.")
    print("For major changes, use a new version number instead.")
    
    response = input("\nCreate post-release? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        new_version = create_post_release()
        
        if new_version:
            print(f"\n📋 Next steps:")
            print(f"1. Test the package: pip install lrdbenchmark=={new_version}")
            print(f"2. Commit changes: git add . && git commit -m 'Bump to {new_version}'")
            print(f"3. Push to GitHub: git push origin master")
            print(f"4. Upload to PyPI: python -m build && twine upload dist/*")
            
            upload_now = input("\nUpload to PyPI now? (y/N): ")
            if upload_now.lower() in ['y', 'yes']:
                build_and_upload()
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()
