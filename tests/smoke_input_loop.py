"""Native regression: input_rec `loop` keeps its timestamps, `loop_continuous_ts` makes them rise.

Run in the built avplumber environment (needs the `ffmpeg` CLI with the native mpeg4 and aac encoders):
    python3 tests/smoke_input_loop.py

Fixtures are 30-frame 60 fps clips: an mp4 with B-frames and AAC audio, and headerless rawvideo
read the way the mixer reads v210 files. Each case reads four loop passes. The paced case adds
`realtime(set_pts)` as in the mixer's source chain: without continuous timestamps it resyncs at each
wrap and emits the first frame of the next pass immediately, one frame period early.
"""

import pathlib
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

from pyplumber import AVPlumber
from pyplumber import node as n

FPS = 60
FRAMES = 30
PASSES = 4
NOPTS = -(1 << 63)


def seconds(ts):
    return Fraction(ts.timestamp) * ts.timebase.num / ts.timebase.den


def make_fixtures(workdir):
    mp4 = workdir / "bframes.mp4"
    raw = workdir / "gray.raw"
    lavfi = f"testsrc2=size=160x120:rate={FPS}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", lavfi, "-f", "lavfi", "-i", "sine=r=48000",
                    "-frames:v", str(FRAMES), "-t", str(FRAMES / FPS), "-c:v", "mpeg4", "-bf", "2", "-g", "12",
                    "-c:a", "aac", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", lavfi, "-frames:v", str(FRAMES),
                    "-pix_fmt", "gray", "-f", "rawvideo", str(raw)], check=True)
    return mp4, raw


def read_loop(url, *, continuous, input_params=None, audio=False, video_codec=None, paced=False):
    """Video PTS (seconds, output order) and audio packet PTS over PASSES loop passes; with
    `paced`, the video PTS are those `realtime(set_pts)` stamps on the host clock."""
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    avp.edges.planCapacity("*", 4096)
    routing = {"v:0": "video_packets", **({"a:0": "audio_packets"} if audio else {})}
    decoder = {"name": "decode", "src": "video_packets", "dst": "decoded",
               **({"codec": video_codec} if video_codec else {})}
    for cls, params in (
        (n.InputRec, {"name": "input", "url": str(url), "dst": "packets", "loop": True,
                      "loop_continuous_ts": continuous, **(input_params or {})}),
        (n.Demux, {"name": "demux", "src": "packets", "routing": routing}),
        (n.DecVideo, decoder),
        *([(n.Realtime, {"name": "realtime", "src": "decoded", "dst": "paced", "set_pts": True})]
          if paced else []),
    ):
        avp.addNode(cls({"group": "input", **params}))
    frames = avp.getEdge("paced" if paced else "decoded", "VideoFrame")
    audio_edge = avp.getEdge("audio_packets", "Packet") if audio else None
    video, sound = [], []
    try:
        avp.group("input").startNodes()
        deadline = time.monotonic() + 20 + (PASSES * FRAMES / FPS if paced else 0)
        while len(video) < PASSES * FRAMES:
            assert not errors, errors
            assert time.monotonic() < deadline, f"only {len(video)} frames decoded"
            frame = frames.tryGet(10)
            if frame is not None and frame.pts.timestamp != NOPTS:
                video.append(seconds(frame.pts))
            while audio_edge is not None and (packet := audio_edge.tryGet(0)) is not None:
                if packet.pts.timestamp != NOPTS:
                    sound.append(seconds(packet.pts))
    finally:
        avp.shutdown()
    return video, sound


def check_restart(name, video):
    """Default `loop`: every pass repeats the file's own timestamps."""
    passes = [video[i * FRAMES:(i + 1) * FRAMES] for i in range(PASSES)]
    assert all(p == passes[0] for p in passes), f"{name}: passes differ"
    print(f"PASS {name} default loop: {PASSES} passes restart at {float(passes[0][0]):.3f} s", flush=True)


def check_continuous(name, video, sound=(), file_sound=()):
    """Video steps one frame period across wraps; audio pass k is the file's audio shifted by
    k video spans, so A/V stays in sync even though the AAC track is a frame longer."""
    step = Fraction(1, FPS)
    steps = {b - a for a, b in zip(video, video[1:])}
    assert steps == {step}, f"{name}: video steps {sorted(float(s) for s in steps)}"
    if sound:
        per_pass = next(i for i in range(1, len(file_sound)) if file_sound[i] < file_sound[i - 1])
        span = FRAMES * step
        for k in range(len(sound) // per_pass):
            got = sound[k * per_pass:(k + 1) * per_pass]
            assert got == [t + k * span for t in file_sound[:per_pass]], f"{name}: audio pass {k} out of sync"
    print(f"PASS {name} continuous loop: {len(video)} frames, constant {float(step) * 1000:.3f} ms step"
          + (f", {len(sound)} audio packets shifted in sync" if sound else ""), flush=True)


def check_paced(name, video, *, continuous):
    """`realtime(set_pts)` output: every step one frame period (continuous), or at least one
    wrap emitted early after a resync (default loop, the behaviour being fixed)."""
    period = Fraction(1, FPS)
    tolerance = Fraction(4, 1000)  # host scheduling jitter on a busy machine
    steps = [b - a for a, b in zip(video, video[1:])]
    early = [float(s) * 1000 for s in steps if s < period - 2 * tolerance]
    if continuous:
        off = [float(s) * 1000 for s in steps if abs(s - period) > tolerance]
        assert not off, f"{name}: paced steps off by more than 4 ms: {off} ms"
        print(f"PASS {name} paced continuous loop: {len(steps)} steps within 4 ms of the period", flush=True)
    else:
        assert early, f"{name}: default loop showed no resync; the paced check cannot detect the bug"
        print(f"PASS {name} paced default loop reproduces the resync: {len(early)} early steps "
              f"({', '.join(f'{s:.1f}' for s in early)} ms)", flush=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        mp4, raw = make_fixtures(pathlib.Path(tmp))
        raw_input = {"format": "rawvideo",
                     "options": {"pixel_format": "gray", "video_size": "160x120", "framerate": f"{FPS}/1"}}
        video, file_sound = read_loop(mp4, continuous=False, audio=True)
        check_restart("mp4", video)
        check_continuous("mp4", *read_loop(mp4, continuous=True, audio=True), file_sound=file_sound)
        check_paced("mp4", read_loop(mp4, continuous=False, paced=True)[0], continuous=False)
        check_paced("mp4", read_loop(mp4, continuous=True, paced=True)[0], continuous=True)
        check_restart("rawvideo", read_loop(raw, continuous=False, input_params=raw_input,
                                            video_codec="rawvideo")[0])
        check_continuous("rawvideo", read_loop(raw, continuous=True, input_params=raw_input,
                                               video_codec="rawvideo")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
