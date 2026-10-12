"""The per-tick timing histogram, without CUDA or FFmpeg."""
import subprocess


def test_tick_times(cpp_binary):
    binary = cpp_binary("test_tick_times")
    subprocess.run([str(binary)], check=True, timeout=60)
