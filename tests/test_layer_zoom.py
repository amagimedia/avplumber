"""Picture-mapped layer zoom geometry (the smooth zoom between one camera's framings)."""
from pathlib import Path
import subprocess

import pytest


def test_layer_zoom(cpp_binary):
    library = Path(__file__).resolve().parents[1] / "deps/avcpp/build/src/libavcpp.a"
    if not library.exists():
        pytest.skip("build avcpp first")
    binary = cpp_binary("test_layer_zoom", sources=(str(library),), libs=("libavutil", "libavcodec", "libavformat"))
    subprocess.run([str(binary)], check=True, timeout=10)
