Installation Guide
=================

This guide will help you install lrdbenchmark and its dependencies.

Requirements
------------

* Python 3.8 or higher (3.8-3.12 supported)
* pip (Python package installer)
* Optional: CUDA-compatible GPU for accelerated computations

**Machine Learning Dependencies**:
* scikit-learn (for SVR, Gradient Boosting, Random Forest)
* PyTorch (for CNN and deep learning models)
* JAX (optional, for GPU acceleration)
* NumPy, SciPy (for numerical computations)

Basic Installation
------------------

Install lrdbenchmark from PyPI:

.. code-block:: bash

   pip install lrdbenchmark

This will install lrdbenchmark with all required dependencies including:

* **15+ Estimators**: Classical, Machine Learning, and Neural Network methods
* **Unified ML Features**: 76-feature extraction pipeline with pre-trained models
* **Production-Ready ML**: SVR (29 features), Gradient Boosting (54 features), Random Forest (76 features)
* **Neural Networks**: CNN, LSTM, GRU, Transformer with automatic device selection
* **Comprehensive Benchmarking**: End-to-end benchmarking system with statistical analysis
* **Demonstration Notebooks**: 5 comprehensive Jupyter notebooks showcasing all features

Installation with Optional Dependencies
--------------------------------------

For GPU acceleration and additional features:

.. code-block:: bash

   # All acceleration libraries (JAX, PyTorch, Numba)
   pip install lrdbenchmark[accel-all]
   
   # Specific acceleration libraries
   pip install lrdbenchmark[accel-jax]      # JAX acceleration
   pip install lrdbenchmark[accel-pytorch]  # PyTorch acceleration
   pip install lrdbenchmark[accel-numba]    # Numba acceleration
   
   # Install with development dependencies
   pip install lrdbenchmark[dev]
   
   # Install with documentation dependencies
   pip install lrdbenchmark[docs]

Development Installation
------------------------

To install LRDBenchmark in development mode:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/dave2k77/lrdbenchmark.git
   cd lrdbenchmark
   
   # Install in development mode
   pip install -e .
   
   # Install with dashboard support
   pip install -e .[dashboard]
   
   # Install development dependencies
   pip install -r requirements-dev.txt

Conda Installation
------------------

Using conda:

.. code-block:: bash

   # Create a new conda environment with Python 3.11 (GPU-ready stack)
   conda create -n lrdbenchmark python=3.11
   conda activate lrdbenchmark
   
   # Install PyTorch with CUDA 12.1 support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Install JAX with CUDA 13 support
   pip install "jax[cuda13_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   
   # Install additional dependencies and the package
   conda install numpy scipy pandas scikit-learn numba -c conda-forge
   pip install lrdbenchmark

Docker Installation
-------------------

Pull the official LRDBenchmark Docker image:

.. code-block:: bash

   docker pull lrdbenchmark/lrdbenchmark:latest
   
   # Run with GPU support
   docker run --gpus all -it lrdbenchmark/lrdbenchmark:latest

Or build from Dockerfile:

.. code-block:: bash

   git clone https://github.com/dave2k77/lrdbenchmark.git
   cd lrdbenchmark
   docker build -t lrdbenchmark .
   docker run -it lrdbenchmark

Verification
------------

After installation, verify that LRDBenchmark is working correctly:

.. code-block:: python

   import lrdbenchmark
   print(f"lrdbenchmark version: {lrdbenchmark.__version__}")
   
   # Test simplified API imports
   from lrdbenchmark import FBMModel, RSEstimator, CNNEstimator, LSTMEstimator
   print("Simplified API imports working!")
   
   # Test basic functionality
   model = FBMModel(H=0.7, sigma=1.0)
   data = model.generate(length=100, seed=42)
   print(f"Generated {len(data)} samples")
   
   # Test estimator
   estimator = RSEstimator()
   result = estimator.estimate(data)
   print(f"Hurst estimate: {result['hurst_parameter']:.3f}")

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**ImportError: No module named 'torch'**
   Install PyTorch separately: ``pip install torch``

**CUDA not found**
   Install CUDA toolkit or use CPU-only version: ``pip install lrdbenchmark[cpu]``

**JAX installation issues**
   On Windows, JAX may require special installation. See `JAX installation guide <https://github.com/google/jax#installation>`_.

**Memory issues with large datasets**
   Consider using smaller batch sizes or reducing data length in benchmarks.

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

For optimal performance:

1. **Use GPU acceleration** when available
2. **Install optimized BLAS libraries** (Intel MKL, OpenBLAS)
3. **Enable JIT compilation** for JAX backends
4. **Use appropriate batch sizes** for your hardware

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Set these environment variables for optimal performance:

.. code-block:: bash

   # Enable JAX optimizations
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_ALLOCATOR=platform
   
   # PyTorch optimizations
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1

Next Steps
-----------

After successful installation, proceed to:

* :doc:`quickstart` - Get started with LRDBenchmark
* :doc:`notebooks/notebooks_overview` - Comprehensive demonstration notebooks
* :doc:`api/estimators` - Detailed estimator documentation
* :doc:`examples/comprehensive_demo` - Example notebooks and scripts

**Recommended Learning Path**:

1. **Start with Notebooks**: Run the 5 demonstration notebooks in order
2. **Quick Start Guide**: Learn basic usage patterns
3. **API Documentation**: Explore detailed estimator documentation
4. **Examples**: Try advanced usage scenarios
