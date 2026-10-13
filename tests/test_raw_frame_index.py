"""Where the pictures of a headerless raw video file are and when they are shown; no GPU required."""
import subprocess


def test_raw_frame_index(cpp_binary):
    subprocess.run([str(cpp_binary("test_raw_frame_index", libs=("libavutil",)))], check=True, timeout=10)
