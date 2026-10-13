"""Remote GPU smoke: input_rec routes its video stream to the decoder itself, as demux would.

One H.264 clip with an audio track nobody routes is decoded on NVDEC through
``input_rec -> demux -> dec_video`` and through ``input_rec(routing) -> dec_video``. The two
chains must agree on every picture and its timestamp over a single pass ending in EOF, over
loop passes that restart their timestamps, and over loop passes with continuous timestamps.

Run in the built avplumber environment (needs the `ffmpeg` CLI with libx264 and the native aac encoder):
    python3 tests/cuda/smoke_input_routing.py
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import subprocess
import tempfile

from _harness import drain, finish, make_avp, nv12_planes, start
from pyplumber import node as api

FPS, FRAMES = 30, 24
HWACCEL = "routing_gpu"


def make_fixture(path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=192x108:rate={FPS}",
                    "-f", "lavfi", "-i", "sine=r=48000", "-frames:v", str(FRAMES), "-t", str(FRAMES / FPS),
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "8", "-bf", "2", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(path)], check=True)


def play(path, *, routed, loop, continuous=False, limit=None):
    """(pts in seconds, luma digest) of the decoded pictures, `limit` of them or all of them
    until EOF, and whether EOF followed."""
    avp, errors = make_avp(HWACCEL, capacity=4)
    reader = {"name": "input", "url": str(path), "loop": loop, "loop_continuous_ts": loop and continuous}
    if routed:
        head = [api.InputRec({**reader, "routing": {"?v:0": "video_packets"}})]
    else:
        head = [api.InputRec({**reader, "dst": "packets"}),
                api.Demux({"name": "demux", "src": "packets", "routing": {"?v:0": "video_packets"},
                           "wait_for_keyframe": False})]
    nodes = head + [
        api.DecVideo({"name": "decode", "src": "video_packets", "dst": "decoded",
                      "pixel_format": "?cuda", "hwaccel": HWACCEL, "options": {"threads": 1}}),
        api.FilterVideo({"name": "download", "src": "decoded", "dst": "pictures", "hwaccel": HWACCEL,
                         "graph": "hwdownload,format=nv12"}),
    ]
    pictures, state = [], {}
    try:
        edge = start(avp, nodes, "probe", "pictures")
        for frame in drain(edge, errors, timeout=20, limit=limit, state=state):
            pts = frame.pts
            pictures.append((Fraction(pts.timestamp) * pts.timebase.num / pts.timebase.den,
                             hashlib.sha1(nv12_planes(frame)[0].tobytes()).hexdigest()))
        assert not errors, errors
    finally:
        finish(avp, nodes)
    return pictures, state.get("eof", False)


def both(path, **how):
    routed, eof = play(path, routed=True, **how)
    demuxed, demux_eof = play(path, routed=False, **how)
    first = next((i for i, (a, b) in enumerate(zip(routed, demuxed)) if a != b), None)
    assert routed == demuxed, f"{how}: routed {len(routed)} pictures, demux {len(demuxed)}, first difference at {first}"
    assert eof == demux_eof, f"{how}: EOF routed={eof} demux={demux_eof}"
    return routed, eof


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "clip.mp4"
        make_fixture(path)

        pictures, eof = both(path, loop=False)
        assert len(pictures) == FRAMES and eof, (len(pictures), eof)
        print(f"PASS single pass: {FRAMES} identical pictures, then EOF on both chains", flush=True)

        pictures, _ = both(path, loop=True, limit=FRAMES * 2 + FRAMES // 2)
        times = [t for t, _ in pictures]
        assert any(b < a for a, b in zip(times, times[1:])), "the looped clip never wrapped"
        print(f"PASS loop: {len(pictures)} identical pictures over wrapping passes", flush=True)

        pictures, _ = both(path, loop=True, continuous=True, limit=FRAMES * 2 + FRAMES // 2)
        times = [t for t, _ in pictures]
        steps = {b - a for a, b in zip(times, times[1:])}
        assert steps == {Fraction(1, FPS)}, sorted(float(s) for s in steps)
        print(f"PASS continuous loop: {len(pictures)} identical pictures, one frame period apart", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
