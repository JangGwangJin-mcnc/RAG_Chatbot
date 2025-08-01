#!/bin/bash

# bizMOB Chatbot Run Script (macOS)
echo "Starting bizMOB Chatbot..."

# Get current script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script location: $SCRIPT_DIR"

# Check and activate virtual environment
if [ -d "$SCRIPT_DIR/venv310" ]; then
    echo "Virtual environment found"
    source "$SCRIPT_DIR/venv310/bin/activate"
elif [ -d "$HOME/bizmob_project/venv310" ]; then
    echo "Virtual environment found"
    source "$HOME/bizmob_project/venv310/bin/activate"
else
    echo "Virtual environment not found"
    echo "Please create virtual environment following the installation guide"
    exit 1
fi

# Change to bizmob_chatbot directory
cd "$SCRIPT_DIR/bizmob_chatbot"

# Apply PyTorch vulnerability fixes
echo "Applying PyTorch security fixes..."
python ../fix_torch_vulnerability.py

# Install ChromaDB if not available
echo "Checking ChromaDB availability..."
python -c "import chromadb" 2>/dev/null || {
    echo "Installing ChromaDB..."
    pip install chromadb
}

# Set additional environment variables
export TORCH_WARN_ON_LOAD=0
export TORCH_LOAD_WARN_ONLY=0
export PYTORCH_DISABLE_WARNINGS=1

# Run Streamlit
echo "Open http://localhost:8080 in your web browser"
echo "Press Ctrl+C to stop"
echo ""

streamlit run bizmob_chatbot.py --server.port 8080 --server.address 0.0.0.0 