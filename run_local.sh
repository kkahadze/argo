#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r fastapi_app/requirements.txt

# Run the server
echo "Starting backend server on http://localhost:8000"
uvicorn fastapi_app.api:app --reload --port 8000
