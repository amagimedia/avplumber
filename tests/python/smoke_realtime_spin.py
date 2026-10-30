"""Native regression: `realtime` with a time base finer than the millisecond clock does not spin.

Run in the built avplumber environment (needs the `ffmpeg` CLI):
    python3 tests/python/smoke_realtime_spin.py

At 30 fps a frame lasts 33.333 ms, so the release time of one frame in three falls a third of a
millisecond after a whole millisecond. The event loop's timer counts whole milliseconds and fires
at that millisecond; a node that then waits for the remaining third asks for a time that has
already passed, and is called again and again until the clock turns over. At 25 fps every release
time is a whole millisecond and nothing spins. So the loop thread's CPU per paced frame must be
about the same at both rates; with the spin it is several times higher at 30 fps.
"""

import os
import pathlib
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

from pyplumber import AVPlumber
from pyplumber import node as n
from _avplumber import MetadataFrame, Rational

SOURCES = 16
SECONDS = 5
NOPTS = -(1 << 63)


def loop_thread_ticks():
    """utime + stime of this process's event loop threads, in clock ticks."""
    total = 0
    for task in pathlib.Path("/proc/self/task").iterdir():
        try:
            if (task / "comm").read_text().strip() != "EventLoop":
                continue
            fields = (task / "stat").read_text().rpartition(")")[2].split()
            total += int(fields[11]) + int(fields[12])
        except OSError:
            pass
    return total


def paced(raw, fps):
    """(loop CPU in ms per paced frame, the PTS steps of one source) with SOURCES sources at *fps*."""
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    avp.edges.planCapacity("*", 64)
    for i in range(SOURCES):
        for cls, params in (
            (n.InputRec, {"name": f"input_{i}", "url": str(raw), "dst": f"packets_{i}", "loop": True,
                          "loop_continuous_ts": True, "format": "rawvideo",
                          "options": {"pixel_format": "gray", "video_size": "64x48", "framerate": f"{fps}/1"}}),
            (n.Demux, {"name": f"demux_{i}", "src": f"packets_{i}", "routing": {"v:0": f"video_{i}"}}),
            (n.DecVideo, {"name": f"decode_{i}", "src": f"video_{i}", "dst": f"decoded_{i}", "codec": "rawvideo"}),
            # As the mixer paces its file sources: the host clock, counted in 1/120000 s.
            (n.Realtime, {"name": f"realtime_{i}", "src": f"decoded_{i}", "dst": f"paced_{i}", "set_pts": True,
                          "time_base": "1/120000", "negative_time_tolerance": .25, "discontinuity_threshold": 1.0}),
            *([(n.NullSink, {"name": f"sink_{i}", "src": f"paced_{i}"})] if i else []),
        ):
            avp.addNode(cls({"group": "input", **params}))
    watched = avp.getEdge("paced_0", "VideoFrame")
    stamps = []
    def drain(until):
        while time.monotonic() < until:
            frame = watched.tryGet(10)
            if frame is not None and frame.pts.timestamp != NOPTS:
                stamps.append(Fraction(frame.pts.timestamp) * frame.pts.timebase.num / frame.pts.timebase.den)
    try:
        avp.group("input").startNodes()
        drain(time.monotonic() + 1.5)   # let every source find its place on the clock
        del stamps[:]
        ticks, started = loop_thread_ticks(), time.monotonic()
        drain(started + SECONDS)
        used_ms = (loop_thread_ticks() - ticks) * 1000 / os.sysconf("SC_CLK_TCK")
        elapsed = time.monotonic() - started
        assert not errors, errors
    finally:
        avp.shutdown()
    frames = SOURCES * fps * elapsed
    assert abs(len(stamps) - fps * elapsed) <= 3, f"{fps} fps: {len(stamps)} frames in {elapsed:.2f} s"
    return used_ms / frames, [b - a for a, b in zip(stamps, stamps[1:])]


def backpressure():
    """A stalled sink preserves ordered frames, exact PTS and EOF through catch-up."""
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    avp.edges.planCapacity("pending", 64)
    avp.edges.planCapacity("paced", 3)
    source = avp.getEdge("pending", "MetadataFrame")
    output = avp.getEdge("paced", "MetadataFrame")
    avp.addNode(n.Realtime({
        "name": "pacer", "group": "test", "src": "pending", "dst": "paced",
        "set_pts": True, "time_base": "1/120000", "forward_eof": True,
        "negative_time_tolerance": 10, "discontinuity_threshold": 20,
    }))
    for i in range(32):
        assert source.enqueue(MetadataFrame(i, Rational(1, 100), {"sequence": i}))
    assert source.enqueue(MetadataFrame())
    try:
        avp.group("test").startNodes()
        deadline = time.monotonic() + 5
        while output.occupied < 3:
            assert not errors, errors
            assert time.monotonic() < deadline, "pacer did not fill its output"
            time.sleep(.005)
        time.sleep(.5)  # more than eight frames become due while the output is full
        stamps = []
        sequence = []
        eof = False
        while time.monotonic() < deadline:
            assert not errors, errors
            frame = output.tryGet(100)
            if frame is None:
                continue
            if frame.pts.timestamp == NOPTS:
                eof = True
                break
            sequence.append(frame.metadata["sequence"])
            stamps.append(Fraction(frame.pts.timestamp) * frame.pts.timebase.num / frame.pts.timebase.den)
        assert eof, "pacer failed to forward EOF after draining its backlog"
        assert sequence == list(range(32)), sequence
        assert set(b - a for a, b in zip(stamps, stamps[1:])) == {Fraction(1, 100)}
        assert output.tryGet(50) is None, "duplicate output after EOF"
    finally:
        avp.shutdown()
    print("PASS backpressure: 32 ordered frames, exact PTS, then EOF")


def main():
    backpressure()
    with tempfile.TemporaryDirectory() as tmp:
        raw = pathlib.Path(tmp) / "gray.raw"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30",
                        "-frames:v", "30", "-pix_fmt", "gray", "-f", "rawvideo", str(raw)], check=True)
        per_frame = {}
        for fps in (25, 30):
            per_frame[fps], steps = paced(raw, fps)
            # The stamped schedule stays exact whenever the frame actually left.
            assert set(steps) == {Fraction(1, fps)}, f"{fps} fps: steps {sorted(set(float(s) for s in steps))[:6]}"
            print(f"{fps} fps: {per_frame[fps] * 1000:.0f} us of loop CPU per paced frame, {len(steps) + 1} exact steps")
        ratio = per_frame[30] / per_frame[25]
        assert ratio < 1.8, f"the loop spins at 30 fps: {ratio:.1f} times the CPU per frame of 25 fps"
        print(f"PASS no spin: 30 fps costs {ratio:.2f} times the loop CPU per frame of 25 fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
