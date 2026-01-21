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
echo "   • Run Python scripts: python your_script.py"
echo "   • Install packages: pip install package_name"
echo "   • Regenerate notebooks from markdown (see notebooks/markdown/)"
echo "   • Deactivate: conda deactivate"
echo ""
