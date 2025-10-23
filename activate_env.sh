#!/bin/bash
# This script activates the project's Python virtual environment.
echo "Activating the LRDBenchmark clean virtual environment..."
source "$(dirname "$0")/lrdbenchmark_venv_clean/bin/activate"
echo "Environment activated. Python executable: $(which python)"
