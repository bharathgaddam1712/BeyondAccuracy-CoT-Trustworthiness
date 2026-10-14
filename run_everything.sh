#!/usr/bin/env bash
# One command: environment -> tests -> GPU smoke -> full run -> tables/figures.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != Linux ]]; then
    printf 'Run this launcher on the Linux GPU server.\n' >&2
    exit 1
fi
command -v flock >/dev/null || { printf 'Install util-linux (flock) before running.\n' >&2; exit 1; }
exec 9>.wllfm-run.lock
flock -n 9 || { printf 'Another run_everything.sh process is already active in this folder.\n' >&2; exit 1; }
mkdir -p logs
log_path="logs/everything-$(date -u +%Y%m%dT%H%M%SZ).log"

run_all() {
    trap 'status=$?; printf "Run stopped (exit %s). Fix the reported error, then rerun the same command to resume.\n" "$status" >&2; exit "$status"' ERR
    set -E
    printf 'Preparing the pinned Python environment and running CPU tests...\n'
    bash scripts/setup_blackwell.sh
    source .venv/bin/activate
    python -m pip check
    export MPLBACKEND=Agg
    export PYTHONUNBUFFERED=1

    if ! python -c 'from huggingface_hub import get_token; raise SystemExit(0 if get_token() else 1)'; then
        printf 'Hugging Face authentication is required for the gated models.\n'
        if [[ ! -t 0 ]]; then
            printf 'Run hf auth login interactively first, or supply HF_TOKEN securely, then rerun.\n' >&2
            return 1
        fi
        hf auth login
    fi
    hf auth whoami

    printf '\nRunning all GPU smoke-test stages...\n'
    python -u run_pipeline.py all --config configs/blackwell_smoke.json --gpus 0 1
    python run_pipeline.py validate --config configs/blackwell_smoke.json

    printf '\nSmoke test passed. Running all full-experiment stages...\n'
    python -u run_pipeline.py all --config configs/blackwell_full.json --gpus 0 1
    python run_pipeline.py validate --config configs/blackwell_full.json
    python run_pipeline.py timing --config configs/blackwell_full.json

    printf '\nAll automated stages completed successfully.\n'
    printf 'Results: results/wllfm2026_full_v3/summary/\n'
    printf 'Figures: results/wllfm2026_full_v3/figures/\n'
    printf 'Validation: results/wllfm2026_full_v3/validation.json\n'
    printf 'Human paraphrase fidelity review remains required before paper-ready validation.\n'
}

# pipefail preserves experiment failures even when tee itself succeeds.
run_all 2>&1 | tee "$log_path"
