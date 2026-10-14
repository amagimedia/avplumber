"""Native regression: one_to_many_nb hands out the frames one_to_many hands out, from an event loop.

Run in the built avplumber environment (needs the `ffmpeg` CLI with the native mpeg4 encoder):
    python3 tests/smoke_one_to_many_nb.py

One clip is decoded and fanned out to three edges with the third masked off, once by the
threaded node and once by the node that runs on an event loop. Both must give the two open
edges every picture, in order, and the third edge nothing.
"""

import hashlib
import pathlib
import subprocess
import sys
import tempfile
import time

from pyplumber import AVPlumber
from pyplumber import node as n

FPS, FRAMES = 30, 45
NOPTS = -(1 << 63)


def make_fixture(workdir):
    mp4 = workdir / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=160x120:rate={FPS}",
                    "-frames:v", str(FRAMES), "-c:v", "mpeg4", "-g", "12", str(mp4)], check=True)
    return mp4


def fan_out(url, node_class, extra):
    """What each of the three edges received: [(pts, digest of the luma plane)]."""
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    avp.edges.planCapacity("*", 127)   # nothing is dropped for lack of room
    for cls, params in (
        (n.InputRec, {"name": "input", "url": str(url), "dst": "packets"}),
        (n.Demux, {"name": "demux", "src": "packets", "routing": {"v:0": "video_packets"}}),
        (n.DecVideo, {"name": "decode", "src": "video_packets", "dst": "video"}),
        (node_class, {"name": "fan", "src": "video", "dst": ["first", "second", "masked"], "outputs": 0b011,
                      "drop": True, **extra}),
    ):
        avp.addNode(cls({"group": "probe", **params}))
    edges = [avp.getEdge(name, "VideoFrame") for name in ("first", "second", "masked")]
    received = [[], [], []]
    try:
        avp.group("probe").startNodes()
        deadline = time.monotonic() + 20
        settled = None   # when both open edges had every picture; a little longer shows nothing follows
        while settled is None or time.monotonic() < settled + .3:
            assert not errors, errors
            assert time.monotonic() < deadline, f"received {[len(r) for r in received]} pictures"
            for i, edge in enumerate(edges):
                frame = edge.tryGet(5 if i == 0 else 0)
                if frame is not None and frame.pts.timestamp != NOPTS:
                    received[i].append((frame.pts.timestamp, hashlib.md5(bytes(frame.data[0])).hexdigest()))
            if settled is None and min(len(received[0]), len(received[1])) >= FRAMES:
                settled = time.monotonic()
        assert not errors, errors
    finally:
        avp.shutdown()
    return received


def main():
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = make_fixture(pathlib.Path(tmp))
        threaded = fan_out(mp4, n.OneToMany, {})
        assert len(threaded[0]) == FRAMES and threaded[0] == threaded[1], [len(r) for r in threaded]
        assert not threaded[2], f"the masked edge received {len(threaded[2])} pictures"
        print(f"PASS one_to_many: {FRAMES} pictures on both open edges, none on the masked one", flush=True)
        looped = fan_out(mp4, n.OneToManyNb, {"event_loop": "fan"})
        assert looped[0] == threaded[0] and looped[1] == threaded[0], \
            f"one_to_many_nb gave {[len(r) for r in looped]} pictures, one_to_many {len(threaded[0])}"
        assert not looped[2], f"the masked edge received {len(looped[2])} pictures"
        print(f"PASS one_to_many_nb: the same {FRAMES} pictures in the same order", flush=True)
        try:
            fan_out(mp4, n.OneToManyNb, {"event_loop": "fan", "drop": False})
        except Exception as error:   # the engine reports a node that cannot be created
            assert "drop" in str(error), error
            print("PASS one_to_many_nb refuses to wait for room in an output", flush=True)
        else:
            raise AssertionError("one_to_many_nb accepted drop: false; an event loop must not wait for room")
    return 0


if __name__ == "__main__":
    sys.exit(main())
