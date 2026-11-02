"""Repeated input pads and scene-to-tile clipping in the native layer resolver."""
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("name", ["test_aux_layers", "test_compositor_crop"])
def test_aux_layers(cpp_binary, name):
    library = Path(__file__).resolve().parents[1] / "deps/avcpp/build/src/libavcpp.a"
    if not library.exists():
        pytest.skip("build avcpp first")
    binary = cpp_binary(name, sources=(str(library),),
                        libs=("libavutil", "libavcodec", "libavformat"))
    subprocess.run([str(binary)], check=True, timeout=10)
