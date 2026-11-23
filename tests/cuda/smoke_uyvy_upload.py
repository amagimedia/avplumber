"""Verify uyvy422_to_cuda: packed UYVY frames unpacked to NV12 on the device, byte for byte.

Luma is the packed picture's; a 4:2:0 chroma row is the rounded mean of its two picture rows, which
are of one field in an interlaced frame. Widths that end inside a kernel thread's 8 pixels and
inside a block are covered. GPU readback is solely for verification.
"""
from pathlib import Path
import tempfile

import numpy as np

from _harness import drain, finish, make_avp
from pyplumber import node as api
from pyplumber.mixer.inputs import _pace, _raw_file_packets


def reference(picture, interlaced):
    """NV12 planes of a packed picture (rows of Cb Y Cr Y bytes)."""
    luma, chroma = picture[:, 1::2], picture[:, 0::2].astype(np.uint16)
    rows = len(picture)
    first = [2 * c - (c & 1 if interlaced else 0) for c in range(rows // 2)]
    step = 2 if interlaced else 1
    second = [f + step if f + step < rows else f for f in first]
    return luma.tobytes(), ((chroma[first] + chroma[second] + 1) >> 1).astype(np.uint8).tobytes()


def check(width, interlaced):
    height, fps = 32, 30
    row, col = np.indices((height, width * 2))
    pictures = [((col * 7 + row * 13 + i * 31 + (col % 2) * row * 5) % 256).astype(np.uint8) for i in range(6)]
    expected = [reference(picture, interlaced) for picture in pictures]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "uyvy.raw"
        path.write_bytes(b"".join(picture.tobytes() for picture in pictures))
        avp, errors = make_avp("uyvy_gpu", capacity=2)
        nodes = []
        try:
            packets = _raw_file_packets(avp, api, "uyvy", str(path), pixel_format="uyvy422",
                                       video_size=f"{width}x{height}", group="probe",
                                       fps=fps, fps_den=1, loop=True)
            chain = [api.DecVideo({"name": "decode", "src": packets, "dst": "decoded",
                                   "codec": "rawvideo", "pixel_format": "uyvy422", "group": "probe"})]
            if interlaced:
                chain.append(api.FilterVideo({"name": "fields", "src": "decoded", "dst": "woven",
                                              "graph": "setfield=tff", "threads": 1, "group": "probe"}))
            chain.append(api.Uyvy422ToCuda({"name": "upload", "src": "woven" if interlaced else "decoded", "dst": "uploaded",
                                            "hwaccel": "uyvy_gpu", "frame_color": {"color_range": "tv"}, "group": "probe"}))
            for node in chain:
                nodes.append(node)
                avp.addNode(node)
            paced = _pace(avp, api, "uyvy", "uploaded", fps=fps, fps_den=1, group="probe", event_loop=None)
            verify = api.FilterVideo({"name": "verify", "src": paced, "dst": "result",
                                      "graph": "hwdownload,format=nv12", "hwaccel": "uyvy_gpu", "threads": 1, "group": "probe"})
            nodes.append(verify)
            avp.addNode(verify)
            output = avp.getEdge("result", "VideoFrame")
            avp.group("probe").startNodes()
            seen = []
            for frame in drain(output, errors, timeout=10, limit=24):
                planes = tuple(np.frombuffer(frame.data[p], np.uint8).reshape(-1, frame.linesize[p])[:height >> p, :width].tobytes()
                               for p in (0, 1))
                assert planes in expected, "UYVY unpack changed luma or chroma bytes"
                seen.append(expected.index(planes))
            assert not errors, errors
            assert len(seen) == 24 and len(set(seen)) >= 3, seen
            print(f"PASS: UYVY {width}x{height} to NV12, interlaced={interlaced}, 24 byte-exact frames", flush=True)
        finally:
            finish(avp, nodes)


if __name__ == "__main__":
    for width in (50, 258, 1920):
        for interlaced in (False, True):
            check(width, interlaced)
