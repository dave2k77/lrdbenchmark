#!/usr/bin/env python3
"""
Utilities for backend selection in unified estimators.
"""
import warnings

# --- Backend Availability ---
try:
    import os
    import logging
    
    # Set JAX to CPU-only mode before importing to prevent CUDA plugin initialization
    # This prevents CUDA_ERROR_NO_DEVICE errors when CUDA_VISIBLE_DEVICES is empty
    if 'JAX_PLATFORM_NAME' not in os.environ:
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    
    # Suppress JAX plugin initialization warnings/errors
    # These can occur when multiple CUDA plugins are installed
    logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL)
    logging.getLogger('jax_plugins').setLevel(logging.CRITICAL)
    
    # Import JAX after setting environment variables
    import jax
    
    # Test that JAX is actually functional
    try:
        _ = jax.devices()
        JAX_AVAILABLE = True
    except Exception:
        # JAX is installed but not functional (e.g., plugin conflicts)
        JAX_AVAILABLE = False
except ImportError:
    JAX_AVAILABLE = False

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# --- Selection Logic ---
def select_backend(requested_backend: str = 'auto') -> str:
    """
    Selects the best available compute backend.

    Priority: JAX (GPU/TPU) > Numba (CPU JIT) > NumPy (fallback).

    Args:
        requested_backend: The user-requested backend ('auto', 'jax', 
                           'numba', 'numpy').

    Returns:
        The name of the selected backend.
    """
    if requested_backend == 'auto':
        if JAX_AVAILABLE:
            return 'jax'
        if NUMBA_AVAILABLE:
            warnings.warn("JAX not available. Falling back to Numba.")
            return 'numba'
        warnings.warn("JAX and Numba not available. Falling back to NumPy.")
        return 'numpy'
    
    if requested_backend == 'jax':
        if JAX_AVAILABLE:
            return 'jax'
        warnings.warn("Requested backend 'jax' not available. Falling back to auto-selection.")
        return select_backend('auto')

    if requested_backend == 'numba':
        if NUMBA_AVAILABLE:
            return 'numba'
        warnings.warn("Requested backend 'numba' not available. Falling back to auto-selection.")
        return select_backend('auto')

    if requested_backend == 'numpy':
        return 'numpy'

    warnings.warn(f"Unknown backend '{requested_backend}' requested. Falling back to auto-selection.")
    return select_backend('auto')
