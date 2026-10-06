"""Compositor interop matrix: source format x canvas format, 420/422, 8/10-bit.

Composes a flat-colour fullscreen source into each semiplanar canvas
(NV12 420/8, P010 420/10, P210 422/10) and checks the downloaded canvas
exactly. Flat colour stays constant through copy, 8->10 promotion (x4) and
420<->422 chroma resampling, so every combination has an exact expected value.
Covers same-format copy, same-subsampling depth promote, and cross-subsampling
(420->422) promote-in. Run on the NVIDIA host with the FFmpeg 8.1 avplumber
module + numpy.
"""

import argparse
from pathlib import Path
import tempfile

import numpy as np

from _harness import drain, finish, make_avp, start

W, H = 256, 128
# Semiplanar canvas formats under test: (sample_bytes, depth, chroma_h_shift)
FORMATS = {
    "nv12":   dict(sb=1, depth=8,  ch=1),   # 4:2:0 8-bit
    "p010le": dict(sb=2, depth=10, ch=1),   # 4:2:0 10-bit (data in high bits, shift 6)
    "p210le": dict(sb=2, depth=10, ch=0),   # 4:2:2 10-bit
}
# (source, canvas) pairs: copies, depth promotes, and 420->422 promote-in.
MATRIX = [("nv12", "nv12"), ("p010le", "p010le"), ("p210le", "p210le"),
          ("nv12", "p010le"), ("nv12", "p210le"), ("p010le", "p210le")]
# Flat 8-bit source codes; 10-bit sources use these <<2 so promotion is exact.
Y8, U8, V8 = 180, 110, 200


def write_flat(path, fmt):
    """One flat-colour raw frame in *fmt*. P010/P210 store the 10-bit code in
    the high bits, so the stored word is code<<6."""
    f = FORMATS[fmt]
    scale = 4 if f["depth"] == 10 else 1
    hishift = 6 if f["sb"] == 2 else 0
    dt = "<u2" if f["sb"] == 2 else np.uint8
    y = np.full((H, W), (Y8 * scale) << hishift, dt)
    uv = np.empty((H >> f["ch"], W), dt)          # interleaved Cb,Cr at half width
    uv[:, 0::2] = (U8 * scale) << hishift
    uv[:, 1::2] = (V8 * scale) << hishift
    with Path(path).open("wb") as s:
        s.write(y.tobytes()); s.write(uv.tobytes())


def run(root, src_fmt, canvas, timeout, *, transfer=None, reduction=1):
    from pyplumber.node import CudaRectOverlay, DecVideo, Demux, FilterVideo, Input
    from pyplumber.mixer.color import conversion_graph

    path = Path(root) / f"{src_fmt}.raw"
    if not path.exists():
        write_flat(path, src_fmt)
    avp, errors = make_avp("ix_gpu")
    upload = "hwupload"
    if transfer:
        upload += "," + conversion_graph(transfer, None, source=transfer)
    width, height = W // reduction, H // reduction
    nodes = [
        Input({"name": "in", "url": str(path), "format": "rawvideo", "dst": "pkt",
               "options": {"pixel_format": src_fmt, "video_size": f"{W}x{H}", "framerate": "60"}}),
        Demux({"name": "dx", "src": "pkt", "routing": {"v:0": "raw"}}),
        DecVideo({"name": "dec", "src": "raw", "dst": "cpu"}),
        FilterVideo({"name": "up", "src": "cpu", "dst": "gpu", "hwaccel": "ix_gpu", "graph": upload}),
        CudaRectOverlay({"name": "comp", "src": ["gpu"], "dst": "scene", "hwaccel": "ix_gpu",
                         "width": W, "height": H, "sw_format": canvas,
                         **({"color": transfer} if transfer else {}),
                         "active_inputs": 1,
                         "layers": [{"dst_x": 0, "dst_y": 0, "dst_w": width, "dst_h": height}]}),
        FilterVideo({"name": "down", "src": "scene", "dst": "out", "hwaccel": "ix_gpu",
                     "graph": f"hwdownload,format={canvas}"}),
    ]
    f = FORMATS[canvas]
    scale = 4 if f["depth"] == 10 else 1
    ey, eu, ev = Y8 * scale, U8 * scale, V8 * scale
    shift, dt = (6, "<u2") if f["sb"] == 2 else (0, np.uint8)
    try:
        out = start(avp, nodes, "ix", "out")
        state = {}
        for frame in drain(out, errors, timeout, 1, state):
            y = np.frombuffer(frame.data[0], dt).reshape(H, frame.linesize[0] // f["sb"])[:, :W] >> shift
            uv = np.frombuffer(frame.data[1], dt).reshape(H >> f["ch"], frame.linesize[1] // f["sb"])[:, :W] >> shift
            cy, cx = (height >> f["ch"]) // 2, (width // 2) & ~1
            assert abs(int(y[height // 2, width // 2]) - ey) <= 1, "luma mismatch"
            assert abs(int(uv[cy, cx]) - eu) <= 1 and abs(int(uv[cy, cx + 1]) - ev) <= 1, "chroma mismatch"
            if reduction > 1:
                assert np.all(y[height:, :] == 16 * scale), "background luma changed"
                assert np.all(uv[height >> f["ch"]:, :] == 128 * scale), "background chroma changed"
        assert not errors, errors
        assert state["count"] == 1, "no frame"
        print(f"PASS {src_fmt:>7} -> {canvas:<7} {transfer or 'untagged'} /{reduction} "
              f"(Y {ey} Cb {eu} Cr {ev})", flush=True)
    finally:
        finish(avp, nodes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeout", type=float, default=30)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="avp-ix-") as root:
        for src, canvas in MATRIX:
            run(root, src, canvas, args.timeout)
        for transfer in ("hlg", "pq"):
            for reduction in (1, 4):
                run(root, "p010le", "p210le", args.timeout, transfer=transfer, reduction=reduction)
    print("interop matrix OK")


if __name__ == "__main__":
    main()
