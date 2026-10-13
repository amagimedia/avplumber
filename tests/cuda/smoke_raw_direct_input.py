"""Remote GPU smoke for raw_video_file_cuda: end of file, loop timestamps, a fresh read on
every pass and byte-exact upload from page-locked memory (cacheable and write-combined).

The download is a verification boundary; the node itself never reads its pictures back.
"""
import os
from pathlib import Path
import tempfile

import numpy as np

from _harness import drain, finish, make_avp, wait_for
from pyplumber import node as api

WIDTH, HEIGHT, FPS, FRAMES = 96, 64, 30, 6


def picture(luma, dtype=np.uint8, shift=0):
    y = np.full((HEIGHT, WIDTH), luma << shift, dtype)
    uv = np.tile(np.array([96 << shift, 160 << shift], dtype), (HEIGHT // 2, WIDTH // 2))
    return y.tobytes() + uv.tobytes()


def reader(avp, path, *, fmt="nv12", **parameters):
    node = api.RawVideoFileCuda({
        "name": "reader", "url": str(path), "dst": "pictures", "hwaccel": "raw_gpu", "group": "probe",
        "auto_restart": "off", "pixel_format": fmt, "video_size": f"{WIDTH}x{HEIGHT}",
        "framerate": f"{FPS}/1", **parameters})
    avp.addNode(node)
    return node


def downloaded(avp, fmt):
    node = api.FilterVideo({"name": "verify", "src": "pictures", "dst": "result", "group": "probe",
                            "hwaccel": "raw_gpu", "graph": f"hwdownload,format={fmt}"})
    avp.addNode(node)
    return node


def samples(frame, sample_bytes=1):
    return b"".join(np.frombuffer(frame.data[i], np.uint8)
                    .reshape(rows, frame.linesize[i])[:, :WIDTH * sample_bytes].tobytes()
                    for i, rows in enumerate((HEIGHT, HEIGHT // 2)))


def seconds(frame):
    pts = frame.pts
    return pts.timestamp * pts.timebase.num / pts.timebase.den


def check_single_pass(directory):
    """A file read once gives its pictures, then the end-of-file marker, then the node finishes."""
    path = Path(directory) / "once.raw"
    path.write_bytes(b"".join(picture(32 + 16 * i) for i in range(FRAMES)))
    avp, errors = make_avp("raw_gpu", capacity=2)
    nodes = []
    try:
        nodes.append(reader(avp, path, loop=False))
        output = avp.getEdge("pictures", "VideoFrame")
        avp.group("probe").startNodes()
        state = {}
        frames = list(drain(output, errors, timeout=10, state=state))
        assert not errors, errors
        assert state["eof"], "no end-of-file marker after the last picture"
        assert [f.pts.timestamp for f in frames] == list(range(FRAMES)), [f.pts.timestamp for f in frames]
        assert all((f.pts.timebase.num, f.pts.timebase.den) == (1, FPS) for f in frames)
        assert all((f.width, f.height) == (WIDTH, HEIGHT) for f in frames)
        wait_for(lambda: not avp.node("reader").isWorking, "node finish after the end of file", errors=errors)
        print(f"PASS: {FRAMES} pictures, the end-of-file marker, then the node finished", flush=True)
    finally:
        finish(avp, nodes)


def check_loop_timestamps(directory, continuous):
    path = Path(directory) / "loop.raw"
    path.write_bytes(b"".join(picture(32 + 16 * i) for i in range(FRAMES)))
    avp, errors = make_avp("raw_gpu", capacity=2)
    nodes = []
    try:
        nodes.append(reader(avp, path, loop=True, loop_continuous_ts=continuous))
        output = avp.getEdge("pictures", "VideoFrame")
        avp.group("probe").startNodes()
        stamps = [f.pts.timestamp for f in drain(output, errors, timeout=10, limit=5 * FRAMES)]
        assert not errors, errors
        expected = [n if continuous else n % FRAMES for n in range(5 * FRAMES)]
        assert stamps == expected, stamps
        print(f"PASS: {len(stamps)} looped pictures, timestamps "
              f"{'counting on' if continuous else 'starting again'} at each pass", flush=True)
    finally:
        finish(avp, nodes)


def check_pixels(directory, fmt, write_combined):
    """Every picture arrives byte-exact, and a picture rewritten in the file while the node
    loops shows up within a pass: each pass is read from the file, nothing is kept."""
    dtype, shift = (np.uint8, 0) if fmt == "nv12" else (np.uint16, 6)
    sample_bytes = np.dtype(dtype).itemsize
    pictures = [picture(32 + 16 * i, dtype, shift) for i in range(FRAMES)]
    replacement = picture(200, dtype, shift)
    path = Path(directory) / f"pixels_{fmt}_{int(write_combined)}.raw"
    path.write_bytes(b"".join(pictures))
    avp, errors = make_avp("raw_gpu", capacity=2)
    nodes = []
    try:
        nodes.append(reader(avp, path, fmt=fmt, loop=True, loop_continuous_ts=True, write_combined=write_combined,
                            frame_color={"color_trc": "bt709", "color_primaries": "bt709",
                                         "colorspace": "bt709", "color_range": "tv"}))
        nodes.append(downloaded(avp, fmt))
        output = avp.getEdge("result", "VideoFrame")
        avp.group("probe").startNodes()
        seen = []
        for frame in drain(output, errors, timeout=10, limit=3 * FRAMES):
            data = samples(frame, sample_bytes)
            assert data in pictures, f"upload changed the raw {fmt} samples"
            seen.append(pictures.index(data))
        assert not errors, errors
        assert seen == [n % FRAMES for n in range(3 * FRAMES)], seen
        # Rewrite picture 2 where it lies. The node and the two edges hold fewer than two passes
        # of pictures read before the rewrite, so the third pass from here shows the file as it is.
        with open(path, "r+b") as file:
            os.pwrite(file.fileno(), replacement, 2 * len(pictures[0]))
            os.fsync(file.fileno())
        later = [samples(frame, sample_bytes) for frame in drain(output, errors, timeout=10, limit=3 * FRAMES)]
        assert not errors, errors
        assert len(later) == 3 * FRAMES, len(later)
        assert replacement in later[2 * FRAMES:], "a picture rewritten in the file never appeared: passes are not read anew"
        assert pictures[2] not in later[2 * FRAMES:], "the old picture is still shown two passes after the rewrite"
        print(f"PASS: byte-exact {fmt} upload from {'write-combined' if write_combined else 'cacheable'} "
              f"page-locked memory, each pass read from the file", flush=True)
    finally:
        finish(avp, nodes)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        check_single_pass(directory)
        for continuous in (True, False):
            check_loop_timestamps(directory, continuous)
        for fmt in ("nv12", "p010le"):
            for write_combined in (False, True):
                check_pixels(directory, fmt, write_combined)
