#!/usr/bin/env bash
# Setup script: create venv, install dependencies, init .env
set -e

echo "==> Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo "==> Creating .env from template..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env — add your API_FOOTBALL_KEY if you want injury data."
fi

echo "==> Creating output directories..."
mkdir -p data/raw data/processed data/external models output

echo ""
echo "Setup complete. Activate your environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run the full pipeline with:"
echo "  python main.py run"
echo ""
echo "Or just a specific league to test quickly:"
echo "  python main.py run --leagues E0 --seasons 2022-23 2023-24"
