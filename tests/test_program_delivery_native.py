"""Output rate conversion must expose inserted holds to the delivery observer."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def test_rate_conversion_holds_reach_delivery_counter(tmp_path):
    if importlib.util.find_spec("_avplumber") is None:
        pytest.skip("requires the native extension")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def run(directory):
    from pyplumber import AVPlumber
    from pyplumber.node import Input, Demux, DecVideo, ForceFPS, MixerDelivery
    path = str(Path(directory) / "frames.nut")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=32x32:rate=25",
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
        for index, frame in enumerate(frames):
            # A nominally 50 fps source delivered only alternate captures.
            frame.metadata["avp.source.browser"] = f"1:{index * 2}:20000000"
        avp.addNode(ForceFPS({"name": "clock", "src": "input", "dst": "clocked", "fps": "50/1"}),
                    early_create=True)
        avp.addNode(MixerDelivery({"name": "delivery", "src": "clocked", "dst": "output", "fps": "50/1"}),
                    early_create=True)
        avp.executeCommandsFromString("node.object.set delivery enabled true")
        source = avp.getEdge("input", "VideoFrame")
        output = avp.getEdge("output", "VideoFrame")
        for frame in frames:
            source.enqueue(frame)
        avp.node("delivery").start()
        avp.node("clock").start()
        identities = [output.get(5000).metadata["avp.source.browser"] for _ in range(7)]
        assert identities == [f"1:{i}:20000000" for i in (0, 0, 2, 2, 4, 4, 6)]
        # getObject is serialized with observation, but publication precedes it.
        import time
        deadline = time.monotonic() + 5
        while True:
            counts = avp.node("delivery").getObject("per_source").get("browser", {})
            if counts.get("frames") == 7:
                break
            assert time.monotonic() < deadline, counts
            time.sleep(.001)
        assert counts["repeats"] == counts["dropped"] == 3, counts
        assert not errors, errors
    finally:
        avp.shutdown()


if __name__ == "__main__":
    run(sys.argv[1])
