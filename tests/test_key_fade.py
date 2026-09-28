"""Fade curves and the downstream key fade envelope (host side, no GPU)."""
import subprocess


def test_key_fade(cpp_binary):
    subprocess.run([str(cpp_binary("test_key_fade"))], check=True, timeout=10)


def test_fade_curve_expression(cpp_binary):
    binary = cpp_binary("test_fade_curve_expression", libs=("libavutil",))
    subprocess.run([str(binary)], check=True, timeout=10)
