#!/bin/bash
"""
HybrIK Server Setup Script
Sets up Python environment and dependencies
"""

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "🔧 Setting up HybrIK Server environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "📍 Python version: $PYTHON_VERSION"

if [[ $(python3 -c "import sys; print(sys.version_info >= (3, 8))") != "True" ]]; then
    echo "❌ Python 3.8+ required. Current: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch for Mac M2 (MPS support)
echo "🔥 Installing PyTorch with MPS support for Mac M2..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import torch
import torchvision
import flask
import numpy as np
import cv2
from PIL import Image
print('✅ All core dependencies installed successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Clone HybrIK repository: git clone https://github.com/jeffffffli/HybrIK.git"
echo "2. Download pretrained models to pretrained_models/"
echo "3. Configure settings.yaml with correct paths"
echo "4. Run server: ./scripts/run.sh"
echo ""
echo "To activate the environment manually:"
echo "  source venv/bin/activate"