import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def test_regex_gate_passes():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/test_regex.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((ROOT / "reports/regex_gate.json").read_text())
    assert rep["n_benign"] >= 200
    assert rep["hard_FP"] == 0
    assert rep["misses"] == []
