"""The colour fields a node stamps on frames: names parsed as FFmpeg spells them; no GPU required."""
import subprocess


def test_frame_color_tags(cpp_binary):
    subprocess.run([str(cpp_binary("test_frame_color_tags", libs=("libavutil",)))], check=True, timeout=10)
