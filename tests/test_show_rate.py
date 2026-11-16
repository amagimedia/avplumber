"""Exact show rates for host sync (src/ClockOrigin.hpp), without CUDA or FFmpeg."""
import subprocess


def test_show_rate(cpp_binary):
    binary = cpp_binary("test_show_rate", libs=("libavutil",))
    subprocess.run([str(binary)], check=True, timeout=120)
