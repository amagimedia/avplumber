"""Remote GPU smoke for raw NV12/P010: loop, pacing and byte-exact upload/compositing.

The download is a verification boundary for this CPU/GPU interop source.
"""
from pathlib import Path
import tempfile

import numpy as np

from _harness import drain, finish, frame_planes, make_avp
from pyplumber import node as api
from pyplumber.mixer.color import Color
from pyplumber.mixer.inputs import build_raw420_input, build_v210_input
from v210_fixture import COLOR, sample_planes, write_fixture


def check(fmt, pinned, native_rate=False, *, composite=False):
    width, height, fps = 96, 64, 30
    color = Color("sdr" if fmt == "nv12" else "hlg")
    dtype, shift = (np.uint8, 0) if fmt == "nv12" else (np.uint16, 6)
    sample_bytes = np.dtype(dtype).itemsize
    frames = []
    for index in range(6):
        y = np.full((height, width), (32 + index * 16) << shift, dtype)
        uv = np.tile(np.array([96 << shift, 160 << shift], dtype), (height // 2, width // 2))
        frames.append(y.tobytes() + uv.tobytes())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pattern.raw"
        path.write_bytes(b"".join(frames))
        avp, errors = make_avp("raw_gpu", capacity=2)
        nodes = []
        try:
            edge = build_raw420_input(avp, api, "raw", str(path), width=width, height=height, pixel_format=fmt,
                                    fps=fps, group="probe", hwaccel="raw_gpu", loop=True, pinned=pinned,
                                    color=color, native_rate=native_rate)
            if composite:
                # Upload tags must satisfy the compositor without a separate color node.
                compositor = api.MixerCompositor({
                    "name": "compose", "src": [edge], "dst": "composed", "group": "probe",
                    "hwaccel": "raw_gpu", "width": width, "height": height, "sw_format": fmt,
                    "color": color.transfer, "fps": f"{fps}/1", "active_inputs": 1,
                    "layers": [{"input": 0}],
                })
                nodes.append(compositor)
                avp.addNode(compositor)
                edge = "composed"
            verify = api.FilterVideo({"name": "verify", "src": edge, "dst": "result",
                                      "group": "probe", "hwaccel": "raw_gpu",
                                      "graph": f"hwdownload,format={fmt}"})
            nodes.append(verify)
            avp.addNode(verify)
            output = avp.getEdge("result", "VideoFrame")
            avp.group("probe").startNodes()
            seen, timestamps = [], []
            for frame in drain(output, errors, timeout=10, limit=24):
                data = b"".join(np.frombuffer(frame.data[i], np.uint8)
                                .reshape(rows, frame.linesize[i])[:, :width * sample_bytes].tobytes()
                                for i, rows in enumerate((height, height // 2)))
                assert data in frames, f"upload changed the raw {fmt} samples"
                seen.append(frames.index(data))
                pts = frame.pts
                timestamps.append(pts.timestamp * pts.timebase.num / pts.timebase.den)
            assert not errors, errors
            assert len(seen) == 24 and len(set(seen)) >= 3, seen
            assert any(b < a for a, b in zip(seen, seen[1:])), "raw clip did not loop"
            assert np.allclose(np.diff(timestamps), 1 / fps, atol=0.001), timestamps
            print(f"PASS: 24 paced {fmt} frames, looping, byte-exact upload ({'pinned' if pinned else 'hwupload'}, native_rate={native_rate}, composite={composite})", flush=True)
        finally:
            finish(avp, nodes)


def check_v210(width, native_rate=False):
    height, fps = 64, 30
    expected = [sample_planes(width, height, i) for i in range(6)]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pattern.v210"
        write_fixture(path, width, height, 6)
        avp, errors = make_avp("v210_gpu", capacity=2)
        nodes = []
        try:
            edge = build_v210_input(avp, api, "raw", str(path), width=width, height=height,
                                    fps=fps, group="probe", hwaccel="v210_gpu", loop=True,
                                    color=COLOR["ramp"], native_rate=native_rate)
            # Force a GPU read before release, exercising recycled pool ordering.
            verify = api.FilterVideo({"name": "verify", "src": edge, "dst": "result",
                                      "group": "probe", "hwaccel": "v210_gpu", "threads": 1,
                                      "graph": "scale_cuda=passthrough=0,hwdownload,format=p210le"})
            nodes.append(verify)
            avp.addNode(verify)
            output = avp.getEdge("result", "VideoFrame")
            avp.group("probe").startNodes()
            seen, timestamps = [], []
            # The first P210 scale kernel may JIT during asynchronous group
            # creation. Include cold startup in the bounded fixture budget.
            for frame in drain(output, errors, timeout=30, limit=24):
                planes = frame_planes(frame, "p210le")
                matches = [i for i, reference in enumerate(expected)
                           if all(np.array_equal(a, b) for a, b in zip(planes, reference))]
                assert matches, f"v210 unpack changed samples at width {width}"
                assert (frame.width, frame.height) == (width, height)
                seen.append(matches[0])
                pts = frame.pts
                timestamps.append(pts.timestamp * pts.timebase.num / pts.timebase.den)
            assert not errors, errors
            assert len(seen) == 24 and len(set(seen)) >= 3, seen
            assert any(b < a for a, b in zip(seen, seen[1:])), "v210 did not loop"
            assert np.allclose(np.diff(timestamps), 1 / fps, atol=.001), timestamps
            print(f"PASS: 24 paced v210 frames at width {width}, looping, byte-exact GPU unpack, native_rate={native_rate}", flush=True)
        finally:
            finish(avp, nodes)


if __name__ == "__main__":
    for native_rate in (False, True):
        for fmt in ("nv12", "p010le"):
            for pinned in (False, True):
                check(fmt, pinned, native_rate)
                if native_rate:
                    check(fmt, pinned, native_rate, composite=True)
        for width in (48, 50, 1920):
            check_v210(width, native_rate)
