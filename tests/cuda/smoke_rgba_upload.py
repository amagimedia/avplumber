"""Verify pinned RGBA CPU/GPU upload, including alpha and negative row stride.

Requires the FFmpeg 9 RGBA pinned-upload patch. GPU readback is solely for
pixel verification; the production wipe stays on the GPU after upload.
"""
from pathlib import Path
import tempfile

import numpy as np

from _harness import drain, finish, make_avp
from pyplumber import node as api
from pyplumber.mixer.inputs import _pace, _raw_file_packets


def check(width, flipped):
    height, fps = 32, 30
    row, col = np.indices((height, width))
    alpha = np.array([0, 1, 127, 128, 254, 255], dtype=np.uint8)
    pictures = [np.stack(((col * 7 + i * 31) % 256, (row * 13 + i) % 256,
                          (col + row * 3 + i * 17) % 256, alpha[(col + row + i) % 6]),
                         axis=-1).astype(np.uint8) for i in range(6)]
    expected = [(frame[::-1] if flipped else frame).tobytes() for frame in pictures]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rgba.raw"
        path.write_bytes(b"".join(frame.tobytes() for frame in pictures))
        avp, errors = make_avp("rgba_gpu", capacity=2)
        nodes = []
        try:
            packets = _raw_file_packets(avp, api, "rgba", str(path), pixel_format="rgba",
                                       video_size=f"{width}x{height}", group="probe",
                                       fps=fps, fps_den=1, loop=True)
            for node in (
                api.DecVideo({"name": "decode", "src": packets, "dst": "decoded",
                              "codec": "rawvideo", "pixel_format": "rgba", "group": "probe"}),
                api.FilterVideo({"name": "upload", "src": "decoded", "dst": "uploaded",
                                 "graph": ("vflip," if flipped else "") + "hwupload_cuda=pinned=1",
                                 "hwaccel": "rgba_gpu", "threads": 1, "group": "probe"}),
            ):
                nodes.append(node)
                avp.addNode(node)
            paced = _pace(avp, api, "rgba", "uploaded", fps=fps, fps_den=1,
                          group="probe", event_loop=None)
            # Force a GPU consumer before pool reuse, then verify every RGBA byte.
            verify = api.FilterVideo({"name": "verify", "src": paced, "dst": "result",
                                      "graph": "scale_cuda=passthrough=0,hwdownload,format=rgba",
                                      "hwaccel": "rgba_gpu", "threads": 1, "group": "probe"})
            nodes.append(verify)
            avp.addNode(verify)
            output = avp.getEdge("result", "VideoFrame")
            avp.group("probe").startNodes()
            seen = []
            for frame in drain(output, errors, timeout=10, limit=24):
                data = np.frombuffer(frame.data[0], np.uint8).reshape(height, frame.linesize[0])
                data = data[:, :width * 4].tobytes()
                assert data in expected, "RGBA upload or GPU read changed color/alpha bytes"
                seen.append(expected.index(data))
            assert not errors, errors
            assert len(seen) == 24 and len(set(seen)) >= 3, seen
            assert any(b < a for a, b in zip(seen, seen[1:])), "clip did not loop"
            print(f"PASS: RGBA {width}x{height}, negative_stride={flipped}, 24 byte-exact frames incl. alpha", flush=True)
        finally:
            finish(avp, nodes)


if __name__ == "__main__":
    for width in (50, 960):
        for flipped in (False, True):
            check(width, flipped)
