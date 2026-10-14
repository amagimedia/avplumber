"""The held copies of repeat_last_frame are numbered after the last stamp, not rounded from the clock."""
import subprocess


def test_repeat_last_frame(cpp_binary):
    # HoldNumbering is header-only over libavutil's AVRational and av_rescale_q; no avcpp needed.
    binary = cpp_binary("test_repeat_last_frame", libs=("libavutil",))
    subprocess.run([str(binary)], check=True, timeout=60)
