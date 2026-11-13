"""Move transition layer interpolation and the slot's move against the take clock."""
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("name", ["test_layer_move", "test_layer_move_offscreen"])
def test_layer_move(cpp_binary, name):
    library = Path(__file__).resolve().parents[1] / "deps/avcpp/build/src/libavcpp.a"
    if not library.exists():
        pytest.skip("build avcpp first")
    binary = cpp_binary(name, sources=(str(library),),
                        libs=("libavutil", "libavcodec", "libavformat", "libavdevice", "libavfilter", "libswscale"))
    subprocess.run([str(binary)], check=True, timeout=10)
