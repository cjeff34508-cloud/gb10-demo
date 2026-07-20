#!/bin/bash
# Dell GB10 Demo Suite - WebUI Quick Start
# Run this script to launch the Streamlit app

set -euo pipefail

PROJECT_DIR="$HOME/gb10-demo"
VENV_PATH="$PROJECT_DIR/env/llm"

echo "======================================"
echo "Dell GB10 Demo Suite - WebUI Launcher"
echo "======================================"
echo ""

# Check if venv exists
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "✗ Error: venv not found at $VENV_PATH"
    echo "Please run: ~/gb10-demo/setup.sh first"
    exit 1
fi

# Activate venv
echo "Activating Python venv..."
source "$VENV_PATH/bin/activate"

# Install Streamlit if not present
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing Streamlit..."
    pip install -q streamlit plotly pandas psutil
fi

# Launch app
echo ""
echo "=========================================="
echo "✓ Launching Streamlit WebUI..."
echo "=========================================="
echo ""
echo "Open your browser to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$PROJECT_DIR/webui"
streamlit run streamlit_app.py --logger.level=warning
