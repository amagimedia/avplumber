"""A take swaps or clears the preview, cools the PVW slot and wakes the followers."""
import subprocess


def test_mixer_preview_swap(cpp_binary):
    binary = cpp_binary("test_mixer_preview_swap", libs=("libavutil",),
                        sources=("src/util.cpp", "deps/avcpp/src/rational.cpp", "deps/avcpp/src/timestamp.cpp"), pthread=True)
    subprocess.run([str(binary)], check=True, timeout=10)
