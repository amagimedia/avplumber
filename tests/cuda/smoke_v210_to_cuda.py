"""Raw byte simulator -> GPU v210 unpack -> CUDA scale -> CPU pixel verification.

Run on an NVIDIA host with the FFmpeg 8.1 avplumber Python module and NumPy.
The download is solely the test's verification boundary. No MXL service or
NVDEC/NVENC is used. Each run also compares against FFmpeg's CPU v210 decoder.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from v210_fixture import frame_stride, sample_planes, write_fixture


def frame_planes(frame, fmt):
    if fmt == "p210le":
        y, uv = [np.frombuffer(data, dtype="<u2").reshape(frame.height, pitch // 2)
                 for data, pitch in zip(frame.data[:2], frame.linesize[:2])]
        y, uv = y[:, :frame.width], uv[:, :frame.width]
        assert not np.any(y & 63) and not np.any(uv & 63), "P210 low bits must be zero"
        return y >> 6, uv[:, 0::2] >> 6, uv[:, 1::2] >> 6
    return tuple(np.frombuffer(data, dtype="<u2").reshape(frame.height, pitch // 2)[:, :width]
                 for data, pitch, width in zip(frame.data[:3], frame.linesize[:3],
                                                [frame.width, frame.width // 2, frame.width // 2]))


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
    from pyplumber import AVPlumber
    from pyplumber.node import Demux, FilterVideo, Input, V210ToCuda

    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(tuple(map(str, error)))
    avp.edges.planCapacity("*", 3)
    avp.executeCommandsFromString('hwaccel.init {"name":"v210_gpu","type":"cuda"}')
    # The gray rawvideo demuxer only frames the bytes into stride*height packets;
    # declaring its byte width this way permits nonstandard v210 row strides.
    nodes = [
        Input({"name": "input", "url": str(path), "format": "rawvideo", "dst": "packets",
               "options": {"pixel_format": "gray", "video_size": f"{stride}x{height}", "framerate": "60"}}),
        Demux({"name": "demux", "src": "packets", "routing": {"v:0": "packed"}}),
        V210ToCuda({"name": "unpack", "src": "packed", "dst": "cuda", "hwaccel": "v210_gpu",
                    "width": width, "height": height, "stride": stride, "fps": "60/1",
                    "timebase": "1/90000", "format": fmt, "sample_aspect_ratio": "4/3",
                    "color_range": "tv", "colorspace": "bt709", "color_primaries": "bt709",
                    "color_trc": "bt709", "chroma_location": "left"}),
        FilterVideo({"name": "verify", "src": "cuda", "dst": "result", "hwaccel": "v210_gpu",
                     "graph": (f"scale_cuda=w={width * 2}:h={height * 2}:interp_algo=nearest,"
                               if scale else "") + f"hwdownload,format={fmt}"}),
    ]
    try:
        for node in nodes:
            node.parameters.update({"group": "test", "auto_restart": "off"})
            avp.addNode(node)
        del node
        result = avp.getEdge("result", "VideoFrame")
        avp.group("test").startNodes()
        deadline = time.monotonic() + timeout
        index = 0
        eof = False
        while time.monotonic() < deadline and not errors:
            frame = result.tryGet(100)
            if frame is None:
                continue
            if frame.pts.timestamp == -(1 << 63):
                eof = True
                break
            assert index < frames, "extra frame"
            expected = sample_planes(width, height, index)
            if scale:
                expected = tuple(plane.repeat(2, axis=0).repeat(2, axis=1) for plane in expected)
            for actual, reference in zip(frame_planes(frame, fmt), expected):
                np.testing.assert_array_equal(actual, reference)
            assert frame.pts.timestamp == index * 1500, "PTS rescale differs"
            assert (frame.pts.timebase.num, frame.pts.timebase.den) == (1, 90000)
            assert (frame.sampleAspectRatio.num, frame.sampleAspectRatio.den) == (4, 3)
            index += 1
        assert not errors, errors
        assert eof and index == frames, f"EOF/count mismatch: eof={eof}, frames={index}/{frames}"
        while any(avp.node(name).isWorking for name in ("input", "demux", "unpack", "verify")):
            assert time.monotonic() < deadline, "EOF workers did not finish"
            time.sleep(0.01)
    finally:
        nodes.clear()
        avp.shutdown()


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
