"""Original path retained; corrected shared analysis implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.entrypoints import legacy_main
if __name__ == "__main__":
    legacy_main("analyze")
