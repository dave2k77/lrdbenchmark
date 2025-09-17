#!/bin/bash
# Simple script to run commands with proper environment setup

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python environment switcher
python3 "$SCRIPT_DIR/switch_env.py"

# If a command was provided, run it with the environment
if [ $# -gt 0 ]; then
    echo "Running: $*"
    exec "$@"
fi
