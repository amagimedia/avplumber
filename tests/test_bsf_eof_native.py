"""Bounded input through encoder -> bsf -> mux ends cleanly (needs the native build)."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = os.environ.get("AVPLUMBER_SHORT_VIDEO")


def test_bsf_bounded_input_ends_cleanly(tmp_path):
    if importlib.util.find_spec("_avplumber") is None:
        pytest.skip("requires the native avplumber extension")
    if not FIXTURE or not Path(FIXTURE).is_file():
        pytest.skip("set AVPLUMBER_SHORT_VIDEO to a short video file")
    smoke = Path(__file__).resolve().parent / "smoke_bsf_eof.py"
    try:
        result = subprocess.run(
            [sys.executable, str(smoke), FIXTURE, str(tmp_path / "out.nut")],
            capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"bsf EOF smoke exceeded its deadline: {error.stdout!r}{error.stderr!r}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS bsf_eof" in result.stdout
