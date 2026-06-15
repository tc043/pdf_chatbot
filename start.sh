#!/usr/bin/env bash

# Determine script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment (.venv) not found!"
    echo "Please create it first by running: python3 -m venv .venv --system-site-packages --without-pip"
    exit 1
fi

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Run the app
echo "Launching PDF Chatbot..."
python run.py

# Deactivate
deactivate
