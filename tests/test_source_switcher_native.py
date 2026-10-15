"""A frame arriving during selector publication must survive to the next decision."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("selector_type", ["source_switcher", "mixer_selector"])
def test_selector_keeps_unexamined_arrivals(tmp_path, selector_type):
    if importlib.util.find_spec("_avplumber") is None:
        pytest.skip("requires the native extension")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path), selector_type],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def run(directory, selector_type):
    from pyplumber import AVPlumber
    from pyplumber.node import Input, Demux, DecVideo, SourceSwitcher
    class Selector(SourceSwitcher):
        TYPE = selector_type

    path = str(Path(directory) / "frames.nut")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=32x32:rate=30",
                    "-frames:v", "3", "-c:v", "rawvideo", "-f", "nut", path], check=True)
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
        frames = [decoded.get(5000) for _ in range(3)]
        for index, frame in enumerate(frames):
            frame.metadata["identity"] = str(index)
        active = avp.getEdge("active", "VideoFrame")
        fallback = avp.getEdge("fallback", "VideoFrame")
        avp.addNode(Selector({"name": "selector", "mixer": "fixture", "src": ["active", "fallback"],
                                    "dst": "output", "active": 0, "fallback_active": 1}),
                    early_create=True)
        output = avp.getEdge("output", "VideoFrame")

        def during_publication(frame):
            # This is the interleaving a concurrent producer can cause: the
            # selected fallback is published while the previously empty active
            # queue receives its next frame, before selector cleanup.
            if frame.metadata["identity"] == "1":
                active.enqueue(frames[2])

        output.addWiretapCallback(during_publication)
        fallback.enqueue(frames[1])
        avp.node("selector").start()
        assert [output.get(2000).metadata["identity"] for _ in range(2)] == ["1", "2"]
        assert not errors, errors
    finally:
        avp.shutdown()


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
