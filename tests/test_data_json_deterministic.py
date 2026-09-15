"""Regenerar data.json dos veces da byte a byte lo mismo."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

def _run_export():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "tools/export_web.py"],
                          capture_output=True, text=True, timeout=300, env=env)

def test_export_deterministic():
    assert _run_export().returncode == 0
    h1 = hashlib.sha256(Path("web/data.json").read_bytes()).hexdigest()
    r = _run_export()
    assert r.returncode == 0, r.stderr[-500:]
    h2 = hashlib.sha256(Path("web/data.json").read_bytes()).hexdigest()
    assert h1 == h2
