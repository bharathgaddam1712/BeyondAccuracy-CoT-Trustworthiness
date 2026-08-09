"""
Session init for Paper A — run at the start of every Kaggle session.
Copies src/ from slm-cot-code dataset.
Restores results/ from slm-cot-results dataset.
Sets HF_HOME to slm-cot-models dataset.
"""

import os
import shutil
from pathlib import Path


CODE_INPUT    = Path("/kaggle/input/slm-cot-code")
RESULTS_INPUT = Path("/kaggle/input/slm-cot-results")
MODELS_INPUT  = Path("/kaggle/input/slm-cot-models/hf_cache")
WORK_DIR      = Path("/kaggle/working/light_paper")


def init(base_dir=None) -> dict:
    base = Path(base_dir) if base_dir else WORK_DIR
    paths = {
        "work":    base,
        "src":     base / "src",
        "data":    base / "data",
        "results": base / "results",
        "figures": base / "figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # 1. Sync src/ from dataset
    if CODE_INPUT.exists():
        src_input = CODE_INPUT / "src"
        if src_input.exists():
            if paths["src"].exists():
                shutil.rmtree(paths["src"])
            shutil.copytree(src_input, paths["src"])
            print(f"[init] src/ synced from {src_input}")
        else:
            print(f"[init] Warning: {src_input} not found — using local src/")
    else:
        print("[init] slm-cot-code not attached — using local src/")

    # 2. Restore results/ CSVs
    if RESULTS_INPUT.exists():
        n = 0
        for csv_file in RESULTS_INPUT.glob("*.csv"):
            dest = paths["results"] / csv_file.name
            if not dest.exists():
                shutil.copy(csv_file, dest)
                n += 1
        print(f"[init] Restored {n} CSVs from slm-cot-results")
    else:
        print("[init] slm-cot-results not attached — starting fresh")

    # 3. Set HF cache to model dataset
    if MODELS_INPUT.exists():
        os.environ["HF_HOME"]             = str(MODELS_INPUT)
        os.environ["TRANSFORMERS_CACHE"]  = str(MODELS_INPUT)
        os.environ["HF_DATASETS_CACHE"]   = str(MODELS_INPUT / "datasets")
        print(f"[init] HF_HOME set to {MODELS_INPUT}")
    else:
        print("[init] slm-cot-models not attached — models will be downloaded fresh")

    print(f"[init] Working dir: {base}")
    return {k: str(v) if not isinstance(v, Path) else v for k, v in paths.items()}