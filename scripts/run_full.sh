#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -u run_pipeline.py all --config configs/blackwell_full.json --gpus 0 1
