#!/usr/bin/env bash
# Launch the PDF Chatbot using the project's virtual environment.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ ! -d ".venv" ]; then
    echo "Virtual environment (.venv) not found." >&2
    echo "Create it and install the dependencies first:" >&2
    echo "    python3 -m venv .venv" >&2
    echo "    source .venv/bin/activate" >&2
    echo "    pip install -r requirements.txt" >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Launching PDF Chatbot..."
exec python run.py
