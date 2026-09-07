"""Exact-picture retention and stale-frame rejection, without GPU dependencies."""
import pathlib
import shutil
import subprocess


def test_mixer_snapshot(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    binary = tmp_path / 'mixer_snapshot'
    compiler = shutil.which('g++') or shutil.which('clang++')
    assert compiler, 'a C++ compiler is required'
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-I', str(root / 'src'),
                    str(root / 'tests/cpp/test_mixer_snapshot.cpp'), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)
