"""The multiview tick a take lands on and the apply window for it, without a graph or GPU."""
import subprocess


def test_mixer_preview_follow(cpp_binary):
    # TickGrid takes an av::Rational, whose constructor lives in avcpp's rational.cpp.
    binary = cpp_binary("test_mixer_preview_follow", libs=("libavutil",), sources=("deps/avcpp/src/rational.cpp",))
    subprocess.run([str(binary)], check=True, timeout=10)
