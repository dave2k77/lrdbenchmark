#!/bin/bash
# This script activates the project's Python virtual environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="${SCRIPT_DIR}/lrdbenchmark_venv"

echo "Activating the LRDBenchmark clean virtual environment..."

if [ ! -d "${ENV_PATH}" ]; then
    echo "Error: expected virtual environment directory at ${ENV_PATH}" >&2
    echo "Please create it (e.g. python -m venv lrdbenchmark_venv) or update activate_env.sh." >&2
    exit 1
fi

if [ ! -f "${ENV_PATH}/bin/activate" ]; then
    echo "Error: missing activate script at ${ENV_PATH}/bin/activate." >&2
    exit 1
fi

source "${ENV_PATH}/bin/activate"

# Normalise VIRTUAL_ENV to the actual path (older virtualenv metadata used a mixed-case directory)
export VIRTUAL_ENV="${ENV_PATH}"
PATH="${ENV_PATH}/bin:${PATH}"
export PATH
hash -r 2>/dev/null || true

echo "Environment activated. Python executable: $(which python)"
