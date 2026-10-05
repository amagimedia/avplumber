"""The draw nodes' private-picture gate and property copy; no GPU required."""
import subprocess


def test_draw_picture_frame(cpp_binary):
    subprocess.run([str(cpp_binary("test_draw_picture_frame", libs=("libavutil",)))], check=True, timeout=10)
