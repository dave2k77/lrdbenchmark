Documentation Guide
==================

Welcome to the LRDBenchmark documentation! This directory contains the source files for building the comprehensive documentation for LRDBenchmark, a Long-Range Dependence Benchmarking Toolkit with comprehensive adaptive classical estimators and intelligent optimization backend.

Building the Documentation
-------------------------

To build the documentation locally:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build HTML documentation:**
   ```bash
   make html
   ```

3. **View the documentation:**
   Open `_build/html/index.html` in your web browser

4. **Clean build files:**
   ```bash
   make clean
   ```

Documentation Structure
----------------------

- **index.rst**: Main documentation entry point with latest results
- **installation.rst**: Installation and setup instructions
- **quickstart.rst**: Quick start guide and basic usage
- **api/**: Complete API reference documentation
  - **data_models.rst**: Data model APIs (FBM, FGN, ARFIMA, MRW)
  - **estimators.rst**: Classical estimator APIs
  - **adaptive_estimators.rst**: Comprehensive adaptive estimator APIs
  - **contamination_factory.rst**: Contamination factory and EEG scenarios
  - **benchmark.rst**: Benchmarking framework APIs
  - **analytics.rst**: Analytics and monitoring APIs
- **examples/**: Code examples and demonstrations
  - **comprehensive_demo.rst**: Original comprehensive demo
  - **comprehensive_adaptive_demo.rst**: New adaptive estimator demo
- **research/**: Theoretical background and validation studies

ReadTheDocs.io Integration
-------------------------

This documentation is automatically built and hosted on ReadTheDocs.io at:
https://lrdbenchmark.readthedocs.io/

The documentation is automatically updated when changes are pushed to the main branch.

Contributing to Documentation
----------------------------

To contribute to the documentation:

1. Make changes to the `.rst` files in this directory
2. Test locally with `make html`
3. Commit and push your changes
4. ReadTheDocs.io will automatically rebuild

For more information, see the main project README.md file.
