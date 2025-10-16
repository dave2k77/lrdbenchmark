#!/usr/bin/env python3
"""
Create GitHub release for LRDBenchmark v2.3.0
"""

import json
import subprocess
import sys

def check_gh_cli():
    """Check if GitHub CLI is available."""
    try:
        result = subprocess.run(['gh', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def create_release_with_gh():
    """Create release using GitHub CLI."""
    print("🔧 Using GitHub CLI to create release...")
    
    # Read release notes
    try:
        with open('RELEASE_NOTES_v2.3.0.md', 'r') as f:
            release_notes = f.read()
    except FileNotFoundError:
        print("❌ Release notes file not found!")
        return False
    
    # Create release
    cmd = [
        'gh', 'release', 'create', 'v2.3.0',
        '--title', 'LRDBenchmark v2.3.0 - Comprehensive Improvements',
        '--notes', release_notes
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GitHub release created successfully!")
            print(f"Release URL: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed to create release: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def manual_instructions():
    """Provide manual instructions."""
    print("\n🔗 Manual Release Creation:")
    print("1. Go to: https://github.com/dave2k77/LRDBenchmark/releases")
    print("2. Click: 'Create a new release'")
    print("3. Choose tag: v2.3.0")
    print("4. Title: LRDBenchmark v2.3.0 - Comprehensive Improvements")
    print("5. Copy release notes from RELEASE_NOTES_v2.3.0.md")
    print("6. Click: 'Publish release'")
    print("\n📦 After creating the release:")
    print("✅ GitHub Actions will automatically trigger")
    print("✅ Package will be built and uploaded to PyPI")
    print("✅ lrdbenchmark v2.3.0 will be available on PyPI")

def main():
    """Main function."""
    print("🚀 LRDBenchmark v2.3.0 Release Creator")
    print("=" * 50)
    
    if check_gh_cli():
        print("✅ GitHub CLI found")
        if create_release_with_gh():
            print("\n🎉 Release created! Check GitHub Actions for upload progress.")
            print("📦 PyPI upload should happen automatically within a few minutes.")
        else:
            manual_instructions()
    else:
        print("❌ GitHub CLI not found")
        print("Installing GitHub CLI...")
        try:
            subprocess.run(['sudo', 'apt', 'install', 'gh', '-y'], check=True)
            if create_release_with_gh():
                print("\n🎉 Release created! Check GitHub Actions for upload progress.")
            else:
                manual_instructions()
        except:
            print("❌ Could not install GitHub CLI")
            manual_instructions()

if __name__ == "__main__":
    main()
