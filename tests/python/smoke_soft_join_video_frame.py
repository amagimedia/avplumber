"""Native regression: soft_join_video_frame stamps MetadataFrame aux onto VideoFrame.

Needs ffmpeg CLI and a built `_avplumber` (same host pattern as test_force_keyframe_native):
    python3 tests/python/smoke_soft_join_video_frame.py
"""

import json
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path

from _avplumber import MetadataFrame, Rational, VideoFrame
from pyplumber import AVPlumber
from pyplumber import node as n

NOPTS = -(1 << 63)
FPS = 50
COUNT = 160  # 3.2s at 50 fps — covers 0.4s..3.0s like the MetadataFrame smoke


def frame_sec(frame):
    pts = frame.pts
    return Fraction(pts.timestamp) * pts.timebase.num / pts.timebase.den


def decode_frames(path: Path, count: int):
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    for cls, params in (
        (n.Input, {"url": str(path), "dst": "packets"}),
        (n.Demux, {"src": "packets", "routing": {"v:0": "video"}}),
        (n.DecVideo, {"src": "video", "dst": "frames"}),
    ):
        avp.addNode(cls({"group": "fixture", **params}))
    edge = avp.getEdge("frames", "VideoFrame")
    result = []
    try:
        avp.group("fixture").startNodes()
        for _ in range(count):
            frame = edge.get(5000)
            assert not errors, errors
            result.append(frame)
        return result
    finally:
        avp.shutdown()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "frames.nut"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
            "-i", f"testsrc2=size=64x64:rate={FPS}",
            "-frames:v", str(COUNT),
            "-c:v", "rawvideo", "-f", "nut", str(path),
        ], check=True, capture_output=True, timeout=30)
        footage = decode_frames(path, COUNT)

    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    for name in ("input", "asr_events", "tagging_events", "output"):
        avp.edges.planCapacity(name, 256)

    avp.addNode(n.SoftJoinVideoFrame({
        "name": "soft_join",
        "group": "test",
        "src": ["input", "asr_events", "tagging_events"],
        "dst": "output",
        "into": ["asr", "tags"],
        "max_age": 1.25,
    }))

    source = avp.getEdge("input", "VideoFrame")
    asr = avp.getEdge("asr_events", "MetadataFrame")
    tags = avp.getEdge("tagging_events", "MetadataFrame")
    output = avp.getEdge("output", "VideoFrame")

    audio_tb = Rational(1, 16000)
    assert asr.enqueue(MetadataFrame(8000, audio_tb, {
        "event": "asr_final", "engine": "whisper", "text": "hello",
    }))
    assert tags.enqueue(MetadataFrame(16000, audio_tb, {
        "event": "audio_tags", "engine": "panns",
        "tags": [{"label": "Speech", "prob": 0.87}],
    }))
    assert asr.enqueue(MetadataFrame(24000, audio_tb, {
        "event": "asr_final", "engine": "whisper", "text": "world",
    }))

    # Sample frames near the MetadataFrame smoke timeline.
    want_secs = [0.4, 1.0, 1.2, 2.0, 3.0]
    picked = []
    for target in want_secs:
        best = min(footage, key=lambda f: abs(float(frame_sec(f)) - target))
        assert abs(float(frame_sec(best)) - target) < 0.03, (frame_sec(best), target)
        picked.append(best)

    try:
        avp.group("test").startNodes()
        for frame in picked:
            assert source.enqueue(frame)
        assert source.enqueue(VideoFrame())  # EOF marker
        assert asr.enqueue(MetadataFrame())
        assert tags.enqueue(MetadataFrame())

        deadline = time.monotonic() + 5
        outs = []
        while len(outs) < len(picked) + 1 and time.monotonic() < deadline:
            assert not errors, errors
            frame = output.tryGet(50)
            if frame is None:
                continue
            if frame.pts.timestamp == NOPTS or (frame.width <= 0 and frame.height <= 0):
                outs.append(None)
                break
            outs.append(frame)

        assert not errors, errors
        assert len(outs) >= len(picked), len(outs)

        def meta_json(frame, key):
            raw = frame.metadata.as_dict
            if key not in raw:
                return None
            val = raw[key]
            return json.loads(val) if isinstance(val, str) else val

        f0, f1, f2, f3, f4 = outs[:5]
        assert meta_json(f0, "asr") is None and meta_json(f0, "tags") is None

        a1 = meta_json(f1, "asr")
        t1 = meta_json(f1, "tags")
        assert a1["text"] == "hello" and a1["engine"] == "whisper"
        assert t1["engine"] == "panns" and t1["tags"][0]["label"] == "Speech"

        assert meta_json(f2, "asr")["text"] == "hello"
        assert meta_json(f3, "asr")["text"] == "world"
        assert meta_json(f3, "tags")["engine"] == "panns"

        assert meta_json(f4, "asr") is None, meta_json(f4, "asr")
        assert meta_json(f4, "tags") is None, meta_json(f4, "tags")
    finally:
        avp.shutdown()

    print("soft_join_video_frame hold-last / into / max_age passed", flush=True)


if __name__ == "__main__":
    main()
