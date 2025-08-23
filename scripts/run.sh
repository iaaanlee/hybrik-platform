#!/bin/bash
"""
HybrIK Server Startup Script
Usage: ./scripts/run.sh [dev|prod]
"""

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODE=${1:-dev}

echo "🚀 Starting HybrIK Server in $MODE mode..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check dependencies
echo "🔍 Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flask; print('Flask: OK')"

if [ "$MODE" = "prod" ]; then
    echo "🏭 Starting production server with Gunicorn..."
    exec gunicorn \
        --bind 0.0.0.0:5002 \
        --workers 1 \
        --timeout 60 \
        --keepalive 2 \
        --max-requests 1000 \
        --max-requests-jitter 50 \
        --log-level info \
        --access-logfile - \
        --error-logfile - \
        app:app
else
    echo "🛠️  Starting development server..."
    exec python app.py
fi