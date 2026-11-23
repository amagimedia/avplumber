"""Native regression: mux_nb and output_nb write the file mux and output write, from an event loop.

Run in the built avplumber environment (needs the `ffmpeg` CLI with the native mpeg4 encoder):
    python3 tests/smoke_mux_output_nb.py

One clip is decoded, encoded and written as MPEG-TS, once by the threaded nodes and once by the
nodes that run on an event loop, with one-packet queues so the muxer meets a full output queue.
Both files must be the same, byte for byte.
"""

import hashlib
import pathlib
import subprocess
import sys
import tempfile
import time

from pyplumber import AVPlumber
from pyplumber import node as n

FPS, FRAMES = 30, 90


def make_fixture(workdir):
    mp4 = workdir / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=160x120:rate={FPS}",
                    "-frames:v", str(FRAMES), "-c:v", "mpeg4", "-g", "12", str(mp4)], check=True)
    return mp4


def write(url, target, mux, output, extra, mux_extra=None):
    """The bytes of *target* after the clip went through *mux* and *output*."""
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    for edge in ("encoded", "muxed"):
        avp.edges.planCapacity(edge, 1)
    for cls, params in (
        (n.InputRec, {"name": "input", "url": str(url), "dst": "packets"}),
        (n.Demux, {"name": "demux", "src": "packets", "routing": {"v:0": "video_packets"}}),
        (n.DecVideo, {"name": "decode", "src": "video_packets", "dst": "video"}),
        (n.ForceFPS, {"name": "rate", "src": "video", "dst": "frames", "fps": f"{FPS}/1"}),
        (n.EncVideo, {"name": "encoder", "src": "frames", "dst": "encoded", "codec": "mpeg2video",
                      "options": {"threads": 1, "g": 12, "bf": 0}}),
        (mux, {"name": "mux", "src": ["encoded"], "dst": "muxed", "ts_sort_wait": 0, **extra, **(mux_extra or {})}),
        (output, {"name": "output", "src": "muxed", "url": str(target), "format": "mpegts", **extra}),
    ):
        avp.addNode(cls({"group": "probe", **params}))
    try:
        avp.group("probe").startNodes()
        deadline, size, steady = time.monotonic() + 30, -1, time.monotonic()
        while time.monotonic() < steady + 1:   # the file has stopped growing
            assert not errors, errors
            assert time.monotonic() < deadline, f"{target} still grows at {size} bytes"
            time.sleep(.05)
            now = target.stat().st_size if target.exists() else 0
            if now != size or not now:
                size, steady = now, time.monotonic()
        assert not errors, errors
    finally:
        avp.shutdown()
    return target.read_bytes()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        mp4 = make_fixture(tmp)
        threaded = write(mp4, tmp / "threaded.ts", n.Mux, n.Output, {})
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-show_entries", "stream=nb_read_packets",
                                "-of", "csv=p=0", str(tmp / "threaded.ts")], check=True, capture_output=True, text=True)
        assert probe.stdout.strip().rstrip(",") == str(FRAMES), f"mux and output wrote {probe.stdout!r} packets, not {FRAMES}"
        print(f"PASS mux, output: {FRAMES} packets, {len(threaded)} bytes", flush=True)
        looped = write(mp4, tmp / "looped.ts", n.MuxNb, n.OutputNb, {"event_loop": "packets"})
        assert looped == threaded, (f"mux_nb and output_nb wrote {len(looped)} bytes "
                                    f"({hashlib.md5(looped).hexdigest()}), mux and output {len(threaded)} "
                                    f"({hashlib.md5(threaded).hexdigest()})")
        print("PASS mux_nb, output_nb: the same file, byte for byte", flush=True)
        try:
            write(mp4, tmp / "refused.ts", n.MuxNb, n.OutputNb, {"event_loop": "packets"}, {"ts_sort_wait": 2.5})
        except Exception as error:   # the engine reports a node that cannot be created
            assert "ts_sort_wait" in str(error), error
            print("PASS mux_nb refuses to wait for another stream's packet", flush=True)
        else:
            raise AssertionError("mux_nb accepted ts_sort_wait 2.5; an event loop must not wait")
    return 0


if __name__ == "__main__":
    sys.exit(main())
