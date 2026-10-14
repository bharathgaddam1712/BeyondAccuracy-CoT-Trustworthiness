"""Check orchestration and failure propagation without installing or using GPUs."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def launcher_fixture(tmp_path):
    bash = (r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt"
            else shutil.which("bash"))
    if not bash or not Path(bash).exists():
        pytest.skip("Bash is required for launcher integration tests")
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / "run_everything.sh", tmp_path / "run_everything.sh")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/setup_blackwell.sh").write_text("exit 0\n")
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/activate").write_text(":\n")
    (tmp_path / "bin").mkdir()
    stubs = {
        "uname": "printf 'Linux\\n'\n",
        "flock": "exit 0\n",
        "hf": "exit 0\n",
        "python": '''printf '%s\\n' "$*" >> calls.txt
if [[ "$*" == *" all --config configs/blackwell_smoke.json"* && "${FAIL_SMOKE:-0}" == 1 ]]; then
    exit 23
fi
exit 0
''',
    }
    for name, body in stubs.items():
        path = tmp_path / "bin" / name
        path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
        path.chmod(0o755)
    return bash


@pytest.mark.parametrize("fail_smoke", [False, True])
def test_launcher_runs_all_stages_and_preserves_failure(tmp_path, fail_smoke):
    bash = launcher_fixture(tmp_path)
    result = subprocess.run(
        [bash, "-c", 'export PATH="$PWD/bin:$PATH"; bash run_everything.sh'],
        cwd=tmp_path,
        env=dict(os.environ, FAIL_SMOKE="1" if fail_smoke else "0"),
        capture_output=True, text=True,
    )
    calls = (tmp_path / "calls.txt").read_text()
    assert "all --config configs/blackwell_smoke.json" in calls
    if fail_smoke:
        assert result.returncode == 23, result.stdout + result.stderr
        assert "blackwell_full.json" not in calls
        assert "All automated stages completed successfully" not in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "all --config configs/blackwell_full.json" in calls
        assert "validate --config configs/blackwell_full.json" in calls
        assert "timing --config configs/blackwell_full.json" in calls
        assert "All automated stages completed successfully" in result.stdout
