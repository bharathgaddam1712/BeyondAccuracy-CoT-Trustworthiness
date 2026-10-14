#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-blackwell.txt
python -m pip freeze > environment.lock.txt
python -m pytest -q
printf '\nSetup and CPU tests finished. Authenticate with: hf auth login\n'
