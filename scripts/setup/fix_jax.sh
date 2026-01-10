#!/bin/bash
source lrdbenchmark_venv/bin/activate
echo "Uninstalling existing JAX/Jaxlib..."
pip uninstall -y jax jaxlib
echo "Installing JAX with CUDA 12 support..."
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
echo "Verifying installation..."
python check_jax.py
