# Runtime Support Policy

This document captures the officially supported Python, NumPy, and accelerator stacks for `lrdbenchmark`. It is updated whenever the dependency policy or CI coverage changes.

## Python

- **Fully supported**: Python 3.10, 3.11, and 3.12 (all platforms). Every PR must pass the automated test suite on these versions.
- **Best effort**: Python 3.9 may work for CPU-only workflows but is no longer part of the CI matrix.
- **End-of-life**: Versions older than 3.9 are not supported.

## NumPy

- **Preferred runtime**: NumPy 2.x. All estimators, analytics modules, and benchmarks must pass the unit and regression suites on NumPy 2.x.
- **Legacy compatibility**: NumPy 1.26.x is available for downstream projects that cannot yet adopt NumPy 2. Failures specific to the 1.26.x stack are triaged with lower priority unless they break previously functional APIs.

## Acceleration Extras

To ensure Python 3.12 compatibility, accelerator extras pin to modern releases:

- `accel-jax`: `jax>=0.4.28`, `jaxlib>=0.4.28`
- `accel-pytorch`: `torch>=2.2.0`
- `accel-numba`: `numba>=0.60.0`
- `accel-all`: combination of the above

When enabling GPU acceleration, match the backend versions documented above and refer to vendor instructions for specific CUDA builds (e.g., PyTorch wheels or JAX CUDA plugins).

## CI Enforcement

- Every pull request runs the test suite on Python 3.10–3.12 with NumPy 2.x and 1.26.x.
- Linting and type checking share the same matrix to ensure code style and typing issues are caught across runtimes.
- Accelerator smoke tests (JAX, PyTorch, Numba) run nightly on Python 3.11 with NumPy 2.x until full GPU CI coverage is available.

## Communication

- The `README.md` installation section and `docs/installation.rst` describe the current policy for end users.
- Release notes highlight any additions or removals from the support table.
- Breaking support changes require explicit approval from the maintainers and must land in a minor or major version bump.

