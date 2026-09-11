"""Cut identity and elapsed-clock accounting, without media/GPU dependencies."""
import pathlib
import shutil
import subprocess


def test_cut_latency(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    binary = tmp_path / "cut_latency"
    compiler = shutil.which("g++") or shutil.which("clang++")
    assert compiler, "a C++ compiler is required"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-pthread", "-I", str(root / "src"),
                    str(root / "tests/cpp/test_cut_latency.cpp"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)
