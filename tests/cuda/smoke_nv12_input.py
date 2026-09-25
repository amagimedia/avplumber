"""Remote GPU smoke for raw NV12/P010: loop, pacing and byte-exact upload.

The download is a verification boundary for this CPU/GPU interop source.
"""
from pathlib import Path
import tempfile

import numpy as np

from _harness import drain, finish, make_avp
from pyplumber import node as api
from pyplumber.mixer.inputs import build_raw420_input


def check(fmt):
    width, height, fps = 96, 64, 30
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
                                    fps=fps, group="probe", hwaccel="raw_gpu", loop=True)
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
            print(f"PASS: 24 paced {fmt} frames, looping, byte-exact CPU → CUDA upload", flush=True)
        finally:
            finish(avp, nodes)


if __name__ == "__main__":
    for fmt in ("nv12", "p010le"):
        check(fmt)
