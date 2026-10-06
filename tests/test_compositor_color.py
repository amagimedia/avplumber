"""Color validation for packed RGB(A) compositor inputs; no GPU required."""
import subprocess


def test_compositor_color(cpp_binary):
    subprocess.run([str(cpp_binary("test_compositor_color", libs=("libavutil",)))], check=True, timeout=10)
