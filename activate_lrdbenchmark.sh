#!/bin/bash
echo "Activating LRDBenchmark virtual environment..."

# Find the virtual environment
VENV_PATH="/home/davianc/LRDBenchmark/.venv"
PROJECT_ROOT="/home/davianc/LRDBenchmark"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    echo "Environment ready: .venv"
else
    echo "Virtual environment not found at $VENV_PATH"
    echo "Please run: python3 setup_conda_free.py"
    exit 1
fi
