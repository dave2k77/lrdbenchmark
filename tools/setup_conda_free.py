#!/usr/bin/env python3
"""
Conda-free setup for LRDBenchmark with JAX GPU support
Uses only system Python and pip to avoid conda issues
"""

import os
import sys
import subprocess
import venv
from pathlib import Path

def run_command(cmd, check=True, capture_output=True):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, check=check)
        if capture_output and result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if capture_output and e.stderr:
            print(f"Stderr: {e.stderr}")
        return None

def check_system_python():
    """Check if system Python is available"""
    print("Checking system Python...")
    
    # Check Python version
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    
    print("✅ Python version is compatible")
    return True

def create_virtual_environment():
    """Create a virtual environment"""
    print("Creating virtual environment...")
    
    venv_path = Path.home() / "lrdbenchmark_venv"
    
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
        return venv_path
    
    try:
        venv.create(venv_path, with_pip=True)
        print(f"✅ Virtual environment created at {venv_path}")
        return venv_path
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return None

def get_venv_python(venv_path):
    """Get the Python executable path for the virtual environment"""
    if os.name == 'nt':  # Windows
        return venv_path / "Scripts" / "python.exe"
    else:  # Unix-like
        return venv_path / "bin" / "python"

def install_packages(venv_path):
    """Install required packages in the virtual environment"""
    print("Installing packages...")
    
    python_exe = get_venv_python(venv_path)
    
    # Upgrade pip first
    print("Upgrading pip...")
    run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install JAX with CUDA support
    print("Installing JAX with CUDA support...")
    result = run_command([
        str(python_exe), "-m", "pip", "install", 
        "jax[cuda12_pip]", 
        "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    ])
    
    if result is None or result.returncode != 0:
        print("❌ Failed to install JAX with CUDA, trying CPU version...")
        run_command([str(python_exe), "-m", "pip", "install", "jax", "jaxlib"])
    
    # Install other required packages
    packages = [
        "numpy", "scipy", "matplotlib", "pandas", "seaborn",
        "torch", "torchvision", "scikit-learn", "plotly",
        "joblib", "numba", "sympy", "pytest", "sphinx",
        "black", "flake8", "mypy", "pre-commit"
    ]
    
    print("Installing additional packages...")
    for package in packages:
        print(f"Installing {package}...")
        run_command([str(python_exe), "-m", "pip", "install", package])
    
    return python_exe

def test_jax(python_exe):
    """Test JAX functionality"""
    print("Testing JAX...")
    
    test_script = '''
import jax
import jax.numpy as jnp
import sys

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# Test computation
x = jnp.array([1, 2, 3, 4, 5])
y = jnp.sin(x)
print(f"Test computation: sin([1,2,3,4,5]) = {y}")

# Check GPU
gpu_devices = [d for d in jax.devices() if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
if gpu_devices:
    print(f"✅ JAX GPU detected: {gpu_devices}")
    print("🎉 JAX GPU setup successful!")
else:
    print("⚠️  JAX running on CPU only")
    print("ℹ️  GPU not available, but JAX is working")

print("✅ JAX test completed successfully!")
'''
    
    result = run_command([str(python_exe), "-c", test_script])
    return result is not None and result.returncode == 0

def create_activation_script(venv_path):
    """Create an activation script for the virtual environment"""
    print("Creating activation script...")
    
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate.bat"
        script_content = f'''@echo off
echo Activating LRDBenchmark virtual environment...
call "{venv_path}\\Scripts\\activate.bat"
echo Environment ready: lrdbenchmark_venv
'''
    else:  # Unix-like
        activate_script = Path("activate_lrdbenchmark.sh")
        script_content = f'''#!/bin/bash
echo "Activating LRDBenchmark virtual environment..."
source "{venv_path}/bin/activate"
export PYTHONPATH="{Path.cwd()}:$PYTHONPATH"
echo "Environment ready: lrdbenchmark_venv"
'''
    
    try:
        with open(activate_script, 'w') as f:
            f.write(script_content)
        
        if os.name != 'nt':  # Unix-like
            os.chmod(activate_script, 0o755)
        
        print(f"✅ Activation script created: {activate_script}")
        return activate_script
    except Exception as e:
        print(f"❌ Failed to create activation script: {e}")
        return None

def find_project_root():
    """Find the LRDBenchmark project root directory"""
    # Look for project root by finding pyproject.toml or lrdbenchmark directory
    current_dir = Path.cwd()
    
    # Check current directory first
    if (current_dir / "pyproject.toml").exists() or (current_dir / "lrdbenchmark").exists():
        return current_dir
    
    # Check parent directories
    for parent in current_dir.parents:
        if (parent / "pyproject.toml").exists() or (parent / "lrdbenchmark").exists():
            return parent
    
    # If not found, use current directory
    print(f"⚠️  Project root not found, using current directory: {current_dir}")
    return current_dir

def main():
    """Main setup function"""
    print("🚀 Setting up LRDBenchmark with conda-free approach...")
    print("=" * 60)
    
    # Find project root
    project_root = find_project_root()
    print(f"Project root: {project_root}")
    
    # Check system Python
    if not check_system_python():
        sys.exit(1)
    
    # Create virtual environment in project root
    venv_path = project_root / "lrdbenchmark_venv"
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
    else:
        try:
            venv.create(venv_path, with_pip=True)
            print(f"✅ Virtual environment created at {venv_path}")
        except Exception as e:
            print(f"❌ Failed to create virtual environment: {e}")
            sys.exit(1)
    
    # Install packages
    python_exe = get_venv_python(venv_path)
    
    # Upgrade pip first
    print("Upgrading pip...")
    run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install JAX with CUDA support
    print("Installing JAX with CUDA support...")
    result = run_command([
        str(python_exe), "-m", "pip", "install", 
        "jax[cuda12_pip]", 
        "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    ])
    
    if result is None or result.returncode != 0:
        print("❌ Failed to install JAX with CUDA, trying CPU version...")
        run_command([str(python_exe), "-m", "pip", "install", "jax", "jaxlib"])
    
    # Install other required packages
    packages = [
        "numpy", "scipy", "matplotlib", "pandas", "seaborn",
        "torch", "torchvision", "scikit-learn", "plotly",
        "joblib", "numba", "sympy", "pytest", "sphinx",
        "black", "flake8", "mypy", "pre-commit"
    ]
    
    print("Installing additional packages...")
    for package in packages:
        print(f"Installing {package}...")
        run_command([str(python_exe), "-m", "pip", "install", package])
    
    # Install the project itself if pyproject.toml exists
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        print("Installing LRDBenchmark project...")
        run_command([str(python_exe), "-m", "pip", "install", "-e", str(project_root)])
    
    # Test JAX
    if not test_jax(python_exe):
        print("❌ JAX test failed")
        sys.exit(1)
    
    # Create activation script
    activate_script = project_root / "activate_lrdbenchmark.sh"
    script_content = f'''#!/bin/bash
echo "Activating LRDBenchmark virtual environment..."
source "{venv_path}/bin/activate"
export PYTHONPATH="{project_root}:$PYTHONPATH"
echo "Environment ready: lrdbenchmark_venv"
'''
    
    try:
        with open(activate_script, 'w') as f:
            f.write(script_content)
        os.chmod(activate_script, 0o755)
        print(f"✅ Activation script created: {activate_script}")
    except Exception as e:
        print(f"❌ Failed to create activation script: {e}")
    
    print("=" * 60)
    print("🎉 Setup completed successfully!")
    print(f"Project root: {project_root}")
    print(f"Virtual environment: {venv_path}")
    print(f"Python executable: {python_exe}")
    
    print(f"To activate the environment, run: source {activate_script}")
    
    print("\nTo test JAX, run:")
    print(f"{python_exe} -c \"import jax; print(f'JAX devices: {{jax.devices()}}')\"")

if __name__ == "__main__":
    main()
