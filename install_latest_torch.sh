#!/bin/bash

# Install Latest PyTorch Script
echo "Installing latest PyTorch version..."

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

# Check current PyTorch version
echo "Current PyTorch version:"
python -c "import torch; print(torch.__version__)"

# Try installing from PyTorch nightly (if available)
echo "Trying to install PyTorch nightly build..."
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu

# If that fails, try conda installation
if [ $? -ne 0 ]; then
    echo "PyTorch nightly not available, trying conda..."
    
    # Check if conda is available
    if command -v conda &> /dev/null; then
        echo "Installing PyTorch via conda..."
        conda install pytorch torchvision torchaudio -c pytorch-nightly
    else
        echo "Conda not available, trying alternative installation..."
        pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Verify installation
echo "New PyTorch version:"
python -c "import torch; print(torch.__version__)"

echo "PyTorch installation completed!" 