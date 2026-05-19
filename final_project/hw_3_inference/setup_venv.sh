#!/bin/bash
# Setup virtual environment for client scripts on Triton server
set -e

echo "Creating venv..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements-client.txt

echo ""
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then run:            python clients/client_http.py --image photo.jpg --output enhanced.jpg"
