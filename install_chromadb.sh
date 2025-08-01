#!/bin/bash

# ChromaDB Installation Script
echo "Installing ChromaDB as alternative vector store..."

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

# Install ChromaDB
echo "Installing ChromaDB..."
pip install chromadb

# Verify installation
echo "Verifying ChromaDB installation..."
python -c "import chromadb; print('ChromaDB installed successfully')"

echo "ChromaDB installation completed!" 