"""
Session save for Paper A — run at the end of every Kaggle session.
Zips results/ with timestamp and prints upload instructions.
"""

import zipfile
from datetime import datetime
from pathlib import Path


def save(results_dir: Path):
    results_dir = Path(results_dir)
    csvs = list(results_dir.glob("*.csv"))
    if not csvs:
        print("[save] No CSVs found in results/ — nothing to save")
        return

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = Path("/kaggle/working") / f"results_snapshot_{ts}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv in csvs:
            zf.write(csv, csv.name)

    print(f"\n{'='*60}")
    print(f"[save] Saved {len(csvs)} CSVs → {zip_path}")
    print(f"\n⚠️  BEFORE CLOSING THIS SESSION:")
    print(f"  1. Go to Kaggle Output panel → download {zip_path.name}")
    print(f"  2. Unzip it")
    print(f"  3. Upload CSVs as a new version of 'slm-cot-results' dataset")
    print(f"  4. Re-attach that dataset to the next session")
    print(f"\n  CSVs saved: {[c.name for c in csvs]}")
    print(f"{'='*60}\n")