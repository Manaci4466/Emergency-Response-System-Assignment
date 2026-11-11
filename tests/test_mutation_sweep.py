"""Overarching check: the mutation sweep (tools/mutation_sweep.py) must catch every mutant.

Slow (runs the whole suite once per mutant, about 25 s), so it only runs on request:
    RUN_MUTATION_SWEEP=1 python -m pytest tests/test_mutation_sweep.py -s
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(os.environ.get("RUN_MUTATION_SWEEP") != "1",
                    reason="slow; set RUN_MUTATION_SWEEP=1 to run the mutation sweep")
def test_every_mutant_is_caught() -> None:
    proc = subprocess.run([sys.executable, "tools/mutation_sweep.py"], cwd=ROOT,
                          capture_output=True, text=True, timeout=1800)
    print(proc.stdout)
    assert proc.returncode == 0, proc.stdout[-2000:]
