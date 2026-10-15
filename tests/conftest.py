"""Shared scaffolding for the standalone C++ unit tests under tests/cpp/ and the native-module
scripts under tests/."""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def cpp_binary(tmp_path):
    """Compile tests/cpp/<name>.cpp into a binary; pkg-config `libs` are linked and their
    absence skips the test, `sources` are extra .cpp files relative to the repo root."""
    def build(name, *, libs=(), sources=(), pthread=False):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            pytest.skip("no C++ compiler available")
        flags = []
        if libs:
            probe = subprocess.run(["pkg-config", "--cflags", "--libs", *libs], capture_output=True, text=True)
            if probe.returncode != 0:
                pytest.skip(f"development files not found: {' '.join(libs)}")
            flags = probe.stdout.split()
        binary = tmp_path / name
        subprocess.run([compiler, "-std=c++17", "-D__STDC_CONSTANT_MACROS", "-O0", "-g", "-Wall", "-Wextra", *(["-pthread"] if pthread else []),
                        "-I", str(ROOT / "src"), "-I", str(ROOT / "deps/avcpp/src"), "-I", str(ROOT / "deps/include"),
                        str(ROOT / "tests/cpp" / f"{name}.cpp"), *(str(ROOT / s) for s in sources),
                        "-o", str(binary), *flags], check=True)
        return binary
    return build


@pytest.fixture
def run_script():
    """Run a test file as a script in a fresh interpreter that imports this checkout's pyplumber and
    native module, as the pytest process does: a bare `python script.py` puts only the script's
    directory on sys.path."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH"))))}
    def run(script, *args, timeout):
        return subprocess.run([sys.executable, str(script), *map(str, args)],
                              capture_output=True, text=True, timeout=timeout, env=env)
    return run
