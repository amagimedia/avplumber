"""Exercise the shared frame-selection contract without CUDA or FFmpeg."""
import subprocess


def test_mixer_playout(cpp_binary):
    # TickGrid takes an av::Rational, whose constructor lives in avcpp's rational.cpp.
    binary = cpp_binary("test_mixer_playout", libs=("libavutil",), sources=("deps/avcpp/src/rational.cpp",))
    subprocess.run([str(binary)], check=True, timeout=120)
