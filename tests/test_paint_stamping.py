"""smooth_timestamps' numbering of browser frames against the playout deadline: counting, one slot
later when a paint would miss its guard, one slot back after a sustained lead; hold copies share it."""
import subprocess


def test_paint_stamping(cpp_binary):
    # PaintStamping and HoldNumbering are header-only over libavutil's AVRational and av_rescale_q.
    binary = cpp_binary("test_paint_stamping", libs=("libavutil",))
    subprocess.run([str(binary)], check=True, timeout=120)
