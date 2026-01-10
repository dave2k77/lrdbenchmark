#!/bin/bash
source lrdbenchmark_venv/bin/activate

# Find library paths
VENV_LIB="/mnt/c/Users/davia/OneDrive/Desktop/PhD Bioengineering Research/frameworks/lrdbenchmark/lrdbenchmark_venv/lib/python3.12/site-packages"
CUDNN_LIB="${VENV_LIB}/nvidia/cudnn/lib"
CUBLAS_LIB="${VENV_LIB}/nvidia/cublas/lib"
CUSOLVER_LIB="${VENV_LIB}/nvidia/cusolver/lib"
CUPTI_LIB="${VENV_LIB}/nvidia/cupti/lib"
NCCL_LIB="${VENV_LIB}/nvidia/nccl/lib"
CUSPARSE_LIB="${VENV_LIB}/nvidia/cusparse/lib"

export LD_LIBRARY_PATH="${CUDNN_LIB}:${CUBLAS_LIB}:${CUSOLVER_LIB}:${CUPTI_LIB}:${NCCL_LIB}:${CUSPARSE_LIB}:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Running JAX check with LD_LIBRARY_PATH..."
python simple_gpu_check.py
