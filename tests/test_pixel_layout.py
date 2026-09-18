"""Pixel-format geometry used by the CUDA compositor; needs libavutil headers only (no GPU)."""
import subprocess


def test_pixel_layout(cpp_binary):
    subprocess.run([str(cpp_binary("test_pixel_layout", libs=("libavutil",)))], check=True, timeout=10)
