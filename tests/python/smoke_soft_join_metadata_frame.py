"""Native regression: soft_join_metadata_frame hold-last joins aux onto MetadataFrame primary.

Run in a built avplumber environment:
    python3 tests/python/smoke_soft_join_metadata_frame.py
"""

import time

from _avplumber import MetadataFrame, Rational
from pyplumber import AVPlumber
from pyplumber import node as n

NOPTS = -(1 << 63)


def md_dict(frame):
    return dict(frame.metadata.as_dict)


def collect(edge, count, deadline):
    out = []
    while len(out) < count and time.monotonic() < deadline:
        frame = edge.tryGet(50)
        if frame is None:
            continue
        if frame.pts.timestamp == NOPTS:
            out.append(None)
            break
        out.append(frame)
    return out


def main():
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)

    for name in ("video_md", "asr_events", "tagging_events", "joined_md"):
        avp.edges.planCapacity(name, 64)

    avp.addNode(n.SoftJoinMetadataFrame({
        "name": "soft_join",
        "group": "test",
        "src": ["video_md", "asr_events", "tagging_events"],
        "dst": "joined_md",
        "into": ["asr", "tags"],
        "max_age": 1.25,
    }))

    video = avp.getEdge("video_md", "MetadataFrame")
    asr = avp.getEdge("asr_events", "MetadataFrame")
    tags = avp.getEdge("tagging_events", "MetadataFrame")
    joined = avp.getEdge("joined_md", "MetadataFrame")

    video_tb = Rational(1, 50)
    audio_tb = Rational(1, 16000)

    assert asr.enqueue(MetadataFrame(8000, audio_tb, {
        "event": "asr_final", "engine": "whisper", "text": "hello", "sec": 0.5,
    }))
    assert tags.enqueue(MetadataFrame(16000, audio_tb, {
        "event": "audio_tags", "engine": "panns",
        "tags": [{"label": "Speech", "prob": 0.87}], "sec": 1.0,
    }))
    assert asr.enqueue(MetadataFrame(24000, audio_tb, {
        "event": "asr_final", "engine": "whisper", "text": "world", "sec": 1.5,
    }))

    assert video.enqueue(MetadataFrame(20, video_tb, {"hls_chunk_index": 0}))
    assert video.enqueue(MetadataFrame(50, video_tb, {"hls_chunk_index": 1}))
    assert video.enqueue(MetadataFrame(60, video_tb, {"hls_chunk_index": 2}))
    assert video.enqueue(MetadataFrame(100, video_tb, {"hls_chunk_index": 3}))
    assert video.enqueue(MetadataFrame(150, video_tb, {"hls_chunk_index": 4}))
    assert video.enqueue(MetadataFrame())
    assert asr.enqueue(MetadataFrame())
    assert tags.enqueue(MetadataFrame())

    try:
        avp.group("test").startNodes()
        deadline = time.monotonic() + 5
        frames = collect(joined, 6, deadline)
        assert not errors, errors
        assert len(frames) >= 5, f"expected >=5 outputs, got {len(frames)}"

        f0, f1, f2, f3, f4 = frames[:5]
        assert f0.pts.timestamp == 20 and f0.pts.timebase.den == 50
        d0 = md_dict(f0)
        assert d0["hls_chunk_index"] == 0
        assert "asr" not in d0 and "tags" not in d0

        d1 = md_dict(f1)
        assert d1["hls_chunk_index"] == 1
        assert d1["asr"]["text"] == "hello"
        assert d1["asr"]["engine"] == "whisper"
        assert d1["tags"]["engine"] == "panns"
        assert d1["tags"]["tags"][0]["label"] == "Speech"
        assert "event" not in d1

        d2 = md_dict(f2)
        assert d2["asr"]["text"] == "hello"
        assert d2["tags"]["engine"] == "panns"

        d3 = md_dict(f3)
        assert d3["asr"]["text"] == "world"
        assert d3["tags"]["engine"] == "panns"

        d4 = md_dict(f4)
        assert d4["hls_chunk_index"] == 4
        assert "asr" not in d4, d4
        assert "tags" not in d4, d4

        if len(frames) < 6 or frames[5] is not None:
            more = collect(joined, 1, deadline)
            assert more and more[0] is None, "missing EOF on joined output"
    finally:
        avp.shutdown()

    print("soft_join_metadata_frame hold-last / into / max_age passed", flush=True)


if __name__ == "__main__":
    main()
