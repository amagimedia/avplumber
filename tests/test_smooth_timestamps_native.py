"""A full output queue must not advance the smoother's sequence on retry."""
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest


def test_backpressure_does_not_skip_presentation_slots(tmp_path):
    if importlib.util.find_spec("_avplumber") is None:
        pytest.skip("requires the native extension")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def run(directory):
    from pyplumber import AVPlumber
    from pyplumber.node import Input, Demux, DecVideo, SmoothTimestamps
    path = str(Path(directory) / "frames.nut")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=32x32:rate=50",
                    "-frames:v", "4", "-c:v", "rawvideo", "-f", "nut", path], check=True)
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    try:
        for cls, params in ((Input, {"url": path, "dst": "packets"}),
                            (Demux, {"src": "packets", "routing": {"v:0": "video"}}),
                            (DecVideo, {"src": "video", "dst": "decoded"})):
            avp.addNode(cls({"group": "fixture", **params}))
        decoded = avp.getEdge("decoded", "VideoFrame")
        avp.group("fixture").startNodes()
        frames = [decoded.get(5000) for _ in range(4)]
        source = avp.getEdge("input", "VideoFrame")
        avp.edges.planCapacity("output", 1)
        avp.addNode(SmoothTimestamps({"name": "smooth", "src": "input", "dst": "output", "fps": "50/1",
                                     "drift_window": 0}), early_create=True)
        source = avp.getEdge("input", "VideoFrame")
        output = avp.getEdge("output", "VideoFrame")
        for frame in frames:
            source.enqueue(frame)
        avp.node("smooth").start()
        received = []
        for _ in frames:
            deadline = time.monotonic() + 5
            while output.occupied != 1:
                assert time.monotonic() < deadline, errors
                time.sleep(.001)
            # The next frame has time to encounter backpressure before this one is drained.
            time.sleep(.03)
            received.append(output.get(1000).pts.timestamp)
        assert not errors, errors
        assert received == list(range(received[0], received[0] + 4)), received
    finally:
        avp.shutdown()


if __name__ == "__main__":
    run(sys.argv[1])
