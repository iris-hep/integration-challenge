#!/bin/bash
# Source this script: source setup.sh

VENV_DIR="$(dirname "${BASH_SOURCE[0]}")/.venv"
REQUIREMENTS="$(dirname "${BASH_SOURCE[0]}")/requirements.txt"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating .venv..."
    python -m venv --system-site-packages "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --quiet -r "$REQUIREMENTS"
    echo ".venv created and packages installed."
else
    source "$VENV_DIR/bin/activate"
fi

setupATLAS
asetup StatAnalysis,0.7.3