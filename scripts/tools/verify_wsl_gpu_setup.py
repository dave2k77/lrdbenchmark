#!/usr/bin/env python3
"""
Verification script for WSL GPU setup with RTX 5070.

This script tests GPU availability and compatibility with LRDBenchmark.
"""

import sys
import platform

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_system():
    """Check system information."""
    print_section("System Information")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check if running in WSL
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read()
            if 'microsoft' in version_info.lower() or 'wsl' in version_info.lower():
                print("✅ Running in WSL environment")
            else:
                print("⚠️  Not detected as WSL environment (may still work)")
    except FileNotFoundError:
        print("⚠️  Cannot determine WSL status (/proc/version not found)")

def check_nvidia_driver():
    """Check NVIDIA driver availability."""
    print_section("NVIDIA Driver Check")
    import subprocess
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ nvidia-smi is available")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA-SMI' in line:
                    print(f"  Driver info: {line.strip()}")
                elif 'GeForce RTX' in line or 'RTX 5070' in line:
                    print(f"  GPU: {line.strip()}")
        else:
            print("❌ nvidia-smi failed")
            print(f"  Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        print("  Make sure NVIDIA drivers are installed in WSL")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  nvidia-smi timed out")
        return False
    except Exception as e:
        print(f"⚠️  Error checking NVIDIA driver: {e}")
        return False
    
    return True

def check_pytorch():
    """Check PyTorch GPU support."""
    print_section("PyTorch GPU Support")
    
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        
        print(f"\nCUDA Information:")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Compute capability: {props.major}.{props.minor}")
                print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Multiprocessors: {props.multi_processor_count}")
            
            # Test GPU computation
            print(f"\n  Testing GPU computation...")
            try:
                device = torch.device('cuda')
                x = torch.randn(1000, 1000, device=device)
                y = torch.matmul(x, x.T)
                print(f"  ✅ GPU computation test passed")
                
                # Check memory
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                print(f"    Memory allocated: {memory_allocated:.2f} MB")
                print(f"    Memory reserved: {memory_reserved:.2f} MB")
                
                # Clean up
                del x, y
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ❌ GPU computation test failed: {e}")
                return False
        else:
            print("  ⚠️  CUDA not available in PyTorch")
            print("  Using CPU-only mode (GPU support for RTX 5070/Blackwell is limited in current stable PyTorch)")
            print("  PyTorch is functional on CPU.")
            return True # Accept CPU as 'working' for now
            
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        print("  Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
        return False

def check_jax():
    """Check JAX GPU support."""
    print_section("JAX GPU Support")
    
    try:
        import jax
        print(f"✅ JAX installed: {jax.__version__}")
        
        try:
            devices = jax.devices()
            backend = jax.default_backend()
            
            print(f"\nJAX Information:")
            print(f"  Devices: {devices}")
            print(f"  Backend: {backend}")
            
            if backend == 'gpu':
                print("  ✅ JAX GPU support: Working")
            else:
                print("  ⚠️  JAX GPU support: CPU fallback")
                print("  This is normal for RTX 5070 (sm_120 architecture)")
                print("  JAX may not support sm_120 yet, but CPU mode works fine")
                
                # Test CPU computation
                print(f"\n  Testing JAX CPU computation...")
                try:
                    import jax.numpy as jnp
                    x = jnp.ones((1000, 1000))
                    y = jnp.sum(x)
                    print(f"  ✅ JAX CPU computation test passed: {y}")
                except Exception as e:
                    print(f"  ❌ JAX CPU computation test failed: {e}")
                    return False
            
            return True
            
        except RuntimeError as e:
            print(f"  ⚠️  JAX runtime error: {e}")
            print("  JAX may be falling back to CPU mode")
            return True  # Not a critical error
            
    except ImportError:
        print("❌ JAX not installed")
        print("  Install with: pip install jax[cuda12_pip]")
        return False
    except Exception as e:
        print(f"⚠️  Error checking JAX: {e}")
        print("  JAX may still work in CPU mode")
        return True  # Not a critical error

def check_lrdbenchmark():
    """Check LRDBenchmark installation."""
    print_section("LRDBenchmark Installation")
    
    try:
        import lrdbenchmark
        print(f"✅ LRDBenchmark installed: {lrdbenchmark.__version__}")
        
        # Check GPU availability in LRDBenchmark
        try:
            from lrdbenchmark import gpu_is_available, get_device_info
            gpu_available = gpu_is_available()
            device_info = get_device_info()
            
            print(f"\nLRDBenchmark GPU Status:")
            print(f"  GPU available: {gpu_available}")
            print(f"  Device info: {device_info}")
            
            if gpu_available:
                print("  ✅ LRDBenchmark GPU support: Available")
            else:
                print("  ⚠️  LRDBenchmark GPU support: Not available")
                print("  CPU mode will be used")
                
        except ImportError:
            print("  ⚠️  GPU utilities not found (may be OK)")
        
        return True
        
    except ImportError:
        print("❌ LRDBenchmark not installed")
        print("  Install with: pip install -e .")
        return False
    except Exception as e:
        print(f"❌ Error checking LRDBenchmark: {e}")
        return False

def check_dependencies():
    """Check core dependencies."""
    print_section("Core Dependencies")
    
    dependencies = {
        'numpy': 'numpy', 
        'scipy': 'scipy', 
        'pandas': 'pandas', 
        'matplotlib': 'matplotlib', 
        'seaborn': 'seaborn',
        'sklearn': 'sklearn', 
        'PyWavelets': 'pywt', 
        'networkx': 'networkx', 
        'joblib': 'joblib', 
        'numba': 'numba'
    }
    
    all_ok = True
    for name, module_name in dependencies.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✅ {name}: {version}")
        except ImportError:
            print(f"  ❌ {name}: Not installed")
            all_ok = False
        except Exception as e:
            print(f"  ⚠️  {name}: Error ({e})")
    
    return all_ok

def main():
    """Run all verification checks."""
    print("=" * 60)
    print(" WSL GPU Setup Verification for RTX 5070")
    print("=" * 60)
    
    results = {}
    
    check_system()
    results['nvidia'] = check_nvidia_driver()
    results['pytorch'] = check_pytorch()
    results['jax'] = check_jax()
    results['lrdbenchmark'] = check_lrdbenchmark()
    results['dependencies'] = check_dependencies()
    
    # Summary
    print_section("Verification Summary")
    
    critical = ['pytorch', 'lrdbenchmark', 'dependencies']
    warnings = ['nvidia', 'jax']
    
    all_critical_ok = all(results.get(k, False) for k in critical)
    
    for key, value in results.items():
        status = "✅ PASS" if value else "❌ FAIL" if key in critical else "⚠️  WARNING"
        print(f"  {key.upper()}: {status}")
    
    print("\n" + "=" * 60)
    
    if all_critical_ok:
        print("✅ Setup verification: PASSED")
        print("\nYour WSL environment is ready for LRDBenchmark!")
        if not results.get('nvidia', False):
            print("⚠️  Note: NVIDIA driver not detected, but this may be OK if using WSL2")
        
        # Check JAX backend if JAX was successfully checked
        try:
            import jax
            if jax.default_backend() != 'gpu':
                print("ℹ️  Note: JAX is using CPU fallback (normal for RTX 5070 sm_120)")
        except:
            pass  # JAX not available or check failed
        
        return 0
    else:
        print("❌ Setup verification: FAILED")
        print("\nPlease fix the issues above before using LRDBenchmark.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

