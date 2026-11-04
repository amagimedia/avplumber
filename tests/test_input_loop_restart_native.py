"""Native test: a looping file decodes the same on every pass, whatever picture it begins with.

input_rec marks the first video packet of each new pass, and the decoder drains and resets there,
so nothing of a pass is predicted from the pictures of the one before (only a pass that begins with
a picture that is not a keyframe is marked; one that begins with a keyframe needs no reset). The
fixture is an MPEG-2
transport stream that begins with predicted pictures (its first keyframe is cut off): a transport
stream has no index, so the loop's seek goes back to its first byte. Reset, the decoder shows
nothing until the next keyframe on every pass, as on the first; run on across the loop, it showed
the leading pictures predicted from the last pass's end (H.264 and HEVC behave alike)."""

import hashlib
import subprocess
import time

import pytest

pytest.importorskip("_avplumber")

from pyplumber import AVPlumber
from pyplumber import node as n

FPS = 30
FRAMES = 36
PASSES = 3
NOPTS = -(1 << 63)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    """A transport stream with B-frames, which begins with a keyframe whose first picture is presented
    after its decode time, and the same without its keyframe: it begins with predicted pictures."""
    work = tmp_path_factory.mktemp("loop-restart")
    whole, cut = work / "whole.ts", work / "cut.ts"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=160x120:rate={FPS}",
                    "-frames:v", str(FRAMES), "-c:v", "mpeg2video", "-bf", "2", "-g", "12", str(whole)],
                   check=True, capture_output=True, timeout=30)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(whole), "-c", "copy",
                    "-bsf:v", "noise=drop=eq(n\\,0)", str(cut)], check=True, capture_output=True, timeout=30)
    return {"whole": whole, "cut": cut}


def decode_passes(url):
    """(pts, md5 of the luma plane) of every decoded frame over PASSES passes, split at each wrap."""
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    avp.edges.planCapacity("*", 4096)
    for cls, params in (
        (n.InputRec, {"name": "input", "url": str(url), "dst": "packets", "loop": True}),
        (n.Demux, {"name": "demux", "src": "packets", "routing": {"v:0": "video"}}),
        (n.DecVideo, {"name": "decode", "src": "video", "dst": "decoded"}),
    ):
        avp.addNode(cls({"group": "input", **params}))
    frames = avp.getEdge("decoded", "VideoFrame")
    passes, last = [[]], None
    try:
        avp.group("input").startNodes()
        deadline = time.monotonic() + 30
        while len(passes) <= PASSES:
            assert not errors, errors
            assert time.monotonic() < deadline, f"only {sum(map(len, passes))} frames decoded"
            frame = frames.tryGet(10)
            if frame is None or frame.pts.timestamp == NOPTS:
                continue
            seconds = frame.pts.timestamp * frame.pts.timebase.num / frame.pts.timebase.den
            if last is not None and seconds < last - FRAMES / FPS / 2:   # the file began again
                passes.append([])
            last = seconds
            passes[-1].append((round(seconds, 4), hashlib.md5(bytes(frame.data[0])).hexdigest()))
    finally:
        avp.shutdown()
    return passes[:PASSES]


@pytest.mark.parametrize("name", ["whole", "cut"])
def test_every_pass_decodes_like_the_first(clips, name):
    """`whole`: the loop goes back to the opening keyframe (seeking to the first picture's
    presentation time landed after it in a transport stream, and later passes lost a GOP).
    `cut`: the decoder resets, so a pass shows nothing it cannot decode, as the first."""
    passes = decode_passes(clips[name])
    first = passes[0]
    if name == "whole":
        assert len(first) == FRAMES, f"the first pass gave {len(first)} of {FRAMES} frames"
    else:
        assert FRAMES // 2 < len(first) < FRAMES, f"the first pass gave {len(first)} frames: is its start cut?"
    for number, later in enumerate(passes[1:], start=2):
        assert later == first, (f"pass {number} differs from the first: "
                                f"{sum(a != b for a, b in zip(first, later))} of {len(first)} frames, "
                                f"{len(later)} frames against {len(first)}")
