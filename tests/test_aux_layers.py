"""Repeated input pads and scene-to-tile clipping in the native layer resolver."""
from pathlib import Path
import subprocess

import pytest


def test_aux_layers(cpp_binary):
    library = Path(__file__).resolve().parents[1] / "deps/avcpp/build/src/libavcpp.a"
    if not library.exists():
        pytest.skip("build avcpp first")
    binary = cpp_binary("test_aux_layers", sources=(str(library),),
                        libs=("libavutil", "libavcodec", "libavformat"))
    subprocess.run([str(binary)], check=True, timeout=10)
