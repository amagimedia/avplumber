"""Raw byte simulator -> GPU v210 unpack -> CUDA scale -> CPU pixel verification.

Run on an NVIDIA host with the FFmpeg 8.1 avplumber Python module and NumPy.
The download is solely the test's verification boundary. No MXL service or
NVDEC/NVENC is used. Each run also compares against FFmpeg's CPU v210 decoder,
and the HLG fixture family's transfer checkpoints are asserted up front.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from _harness import drain, finish, frame_planes, make_avp, start, v210_chain
from v210_fixture import frame_stride, hlg_planes, sample_planes, write_fixture

BT709 = {"color_range": "tv", "colorspace": "bt709", "color_primaries": "bt709",
         "color_trc": "bt709", "chroma_location": "left"}


def check_hlg_fixture():
    """BT.2100 checkpoints E = 0, 1/12, 1 -> Y 64, 502, 940 with neutral chroma."""
    y, u, v = hlg_planes(96, 8, index=3)
    for i, code in enumerate((64, 502, 940)):
        np.testing.assert_array_equal(y[:2, 12 * i:12 * (i + 1)], code)
        np.testing.assert_array_equal(u[:2, 6 * i:6 * (i + 1)], 512)
        np.testing.assert_array_equal(v[:2, 6 * i:6 * (i + 1)], 512)
    assert y.min() >= 64 and y.max() <= 940


def cpu_reference(ffmpeg, path, width, height, frames):
    decoded = subprocess.run([
        ffmpeg, "-v", "error", "-threads", "1", "-f", "v210",
        "-video_size", f"{width}x{height}", "-framerate", "60", "-i", str(path),
        "-frames:v", str(frames), "-pix_fmt", "yuv422p10le", "-f", "rawvideo", "pipe:1",
    ], check=True, capture_output=True, timeout=60).stdout
    assert len(decoded) == width * height * 4 * frames
    values = np.frombuffer(decoded, dtype="<u2").reshape(frames, -1)
    for index, frame in enumerate(values):
        size = width * height
        planes = (frame[:size].reshape(height, width),
                  frame[size:size + size // 2].reshape(height, width // 2),
                  frame[size + size // 2:].reshape(height, width // 2))
        for actual, expected in zip(planes, sample_planes(width, height, index)):
            np.testing.assert_array_equal(actual, expected)


def run_graph(path, width, height, frames, stride, fmt, scale, timeout):
    from pyplumber.node import FilterVideo

    avp, errors = make_avp("v210_gpu")
    nodes = []
    src = v210_chain(nodes, "t", path, width=width, height=height, stride=stride, fmt=fmt,
                     hwaccel="v210_gpu", color=BT709, sample_aspect_ratio="4/3")
    nodes.append(FilterVideo({
        "name": "verify", "src": src, "dst": "result", "hwaccel": "v210_gpu",
        "graph": (f"scale_cuda=w={width * 2}:h={height * 2}:interp_algo=nearest,"
                  if scale else "") + f"hwdownload,format={fmt}"}))
    try:
        result = start(avp, nodes, "test", "result")
        deadline = time.monotonic() + timeout
        state = {}
        for index, frame in enumerate(drain(result, errors, timeout, state=state)):
            assert index < frames, "extra frame"
            expected = sample_planes(width, height, index)
            if scale:
                expected = tuple(plane.repeat(2, axis=0).repeat(2, axis=1) for plane in expected)
            for actual, reference in zip(frame_planes(frame, fmt), expected):
                np.testing.assert_array_equal(actual, reference)
            assert frame.pts.timestamp == index * 1500, "PTS rescale differs"
            assert (frame.pts.timebase.num, frame.pts.timebase.den) == (1, 90000)
            assert (frame.sampleAspectRatio.num, frame.sampleAspectRatio.den) == (4, 3)
        assert not errors, errors
        assert state["eof"] and state["count"] == frames, \
            f"EOF/count mismatch: eof={state['eof']}, frames={state['count']}/{frames}"
        while any(avp.node(n).isWorking for n in ("in_t", "demux_t", "unpack_t", "verify")):
            assert time.monotonic() < deadline, "EOF workers did not finish"
            time.sleep(0.01)
    finally:
        finish(avp, nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default="/usr/local/bin/ffmpeg")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--scale", action="store_true", help="also exercise nearest 2x CUDA scaling")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    check_hlg_fixture()
    with tempfile.TemporaryDirectory(prefix="avp-v210-") as root:
        path = Path(root) / "frames.v210"
        stride = write_fixture(path, args.width, args.height, args.frames, args.stride)
        # FFmpeg's v210 demuxer uses the standard 48-pixel row alignment.
        reference = path
        if stride != frame_stride(args.width):
            reference = Path(root) / "reference.v210"
            write_fixture(reference, args.width, args.height, args.frames)
        cpu_reference(args.ffmpeg, reference, args.width, args.height, args.frames)
        for fmt in ("p210le", "yuv422p10le"):
            run_graph(path, args.width, args.height, args.frames, stride, fmt, args.scale, args.timeout)
            print(f"PASS {args.width}x{args.height} @ 60 fps, {args.frames} frames, "
                  f"stride={stride}, format={fmt}, scale={args.scale}", flush=True)


if __name__ == "__main__":
    main()
