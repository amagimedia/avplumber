"""Layer transform geometry (sub-pixel crop, rotation, pivot) in the native layer resolver."""
from pathlib import Path
import subprocess

import pytest


def test_compositor_affine(cpp_binary):
    library = Path(__file__).resolve().parents[1] / "deps/avcpp/build/src/libavcpp.a"
    if not library.exists():
        pytest.skip("build avcpp first")
    binary = cpp_binary("test_compositor_affine", sources=(str(library),),
                        libs=("libavutil", "libavcodec", "libavformat"))
    subprocess.run([str(binary)], check=True, timeout=10)
