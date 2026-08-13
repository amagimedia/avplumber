"""CPU unit test for the YOLO detection decoder.

Compiles and runs tests/cpp/test_decode_detection.cpp, which exercises the real
yolo_base::DetectionDecoder logic (including the boxes_normalized rescale for
DeepStream/Triton-style YOLO exports) against a minimal NvInfer.h stub, so no
CUDA/TensorRT toolchain is required. Skips if no C++ compiler is available.
"""

import os
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
STUBS = os.path.join(REPO_ROOT, "tests", "cpp", "stubs")
TEST_CPP = os.path.join(REPO_ROOT, "tests", "cpp", "test_decode_detection.cpp")


def _find_compiler():
    for cc in ("g++", "clang++", "c++"):
        path = shutil.which(cc)
        if path:
            return path
    return None


def test_decode_detection(tmp_path):
    compiler = _find_compiler()
    if compiler is None:
        pytest.skip("no C++ compiler available")

    binary = os.path.join(str(tmp_path), "test_decode_detection")
    compile_cmd = [
        compiler,
        "-std=c++17",
        "-O0",
        "-Wall",
        "-Wextra",
        "-I", STUBS,   # minimal NvInfer.h stub must resolve before any real one
        "-I", SRC,
        TEST_CPP,
        "-o", binary,
    ]
    compile = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert compile.returncode == 0, (
        "compilation failed:\n" + compile.stdout + compile.stderr
    )

    run = subprocess.run([binary], capture_output=True, text=True)
    assert run.returncode == 0, "test binary failed:\n" + run.stdout + run.stderr
    assert "OK: all decode_detection tests passed" in run.stdout, run.stdout
