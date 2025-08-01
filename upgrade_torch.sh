#!/bin/bash

# PyTorch Upgrade Script
echo "Upgrading PyTorch to fix security vulnerability..."

# Get current script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check and activate virtual environment
if [ -d "$SCRIPT_DIR/venv310" ]; then
    echo "Virtual environment found"
    source "$SCRIPT_DIR/venv310/bin/activate"
elif [ -d "$HOME/bizmob_project/venv310" ]; then
    echo "Virtual environment found"
    source "$HOME/bizmob_project/venv310/bin/activate"
else
    echo "Virtual environment not found"
    exit 1
fi

# Check current Python version
echo "Python version:"
python --version

# Check current PyTorch version
echo "Current PyTorch version:"
python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "PyTorch not installed"

# Try to upgrade to latest available version
echo "Upgrading PyTorch to latest available version..."
pip install --upgrade torch

# If that fails, try installing specific version
if [ $? -ne 0 ]; then
    echo "Trying to install PyTorch 2.2.2 (latest stable)..."
    pip install torch==2.2.2
fi

# Verify upgrade
echo "New PyTorch version:"
python -c "import torch; print(torch.__version__)"

echo "PyTorch upgrade completed!" 