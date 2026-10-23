#!/usr/bin/env bash
set -e

# Change directory to project root
cd "$(dirname "$0")"

echo "=========================================================="
echo " Starting PixMe (AI Event Photo Platform)"
echo "=========================================================="

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi

echo "Starting FastAPI server on http://localhost:8000 ..."
echo "Buka browser Anda di: http://localhost:8000"
exec .venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
