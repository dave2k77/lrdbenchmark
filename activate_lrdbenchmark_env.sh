#!/bin/bash
# LRDBenchmark Environment Activation Script
# This script activates the dedicated conda environment for LRDBenchmark

echo "🚀 Activating LRDBenchmark Environment..."
echo "=========================================="

# Source conda
source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate lrdbenchmark

# Verify environment
echo "✅ Environment activated: lrdbenchmark"
echo "📍 Python version: $(python --version)"
echo "📍 Working directory: $(pwd)"

# Test LRDBenchmark import
echo "🧪 Testing LRDBenchmark import..."
python -c "import lrdbenchmark; print('✅ LRDBenchmark version:', lrdbenchmark.__version__)"

echo ""
echo "🎯 Environment ready! You can now:"
echo "   • Run Jupyter notebooks: jupyter notebook"
echo "   • Run Python scripts: python your_script.py"
echo "   • Install packages: pip install package_name"
echo "   • Deactivate: conda deactivate"
echo ""
echo "📚 Available notebooks:"
echo "   • notebooks/01_data_generation_and_visualisation.ipynb"
echo "   • notebooks/02_estimation_and_validation.ipynb"
echo "   • notebooks/03_custom_models_and_estimators.ipynb"
echo "   • notebooks/04_comprehensive_benchmarking.ipynb"
echo "   • notebooks/05_leaderboard_generation.ipynb"
echo ""
