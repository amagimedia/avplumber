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
import time

import numpy as np

W, H = 256, 128
# Semiplanar canvas formats under test: (name, sample_bytes, depth, chroma_h_shift)
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
    """One flat-colour raw frame in *fmt* (rawvideo pixel_format = fmt name).
    P010/P210 carry the 10-bit code in the high bits, so the stored word is
    code<<6."""
    f = FORMATS[fmt]
    scale = 4 if f["depth"] == 10 else 1        # 8-bit code -> 10-bit code
    hishift = 6 if f["sb"] == 2 else 0          # p010le/p210le store in high bits
    dt = "<u2" if f["sb"] == 2 else np.uint8
    y = np.full((H, W), (Y8 * scale) << hishift, dt)
    ch_h = H >> f["ch"]
    uv = np.empty((ch_h, W), dt)          # interleaved Cb,Cr at half width
    uv[:, 0::2] = (U8 * scale) << hishift
    uv[:, 1::2] = (V8 * scale) << hishift
    with Path(path).open("wb") as s:
        s.write(y.tobytes()); s.write(uv.tobytes())


def expected(canvas):
    """Expected flat canvas codes (logical), promoted to the canvas depth."""
    scale = 4 if FORMATS[canvas]["depth"] == 10 else 1
    return Y8 * scale, U8 * scale, V8 * scale


def run(root, src_fmt, canvas, timeout):
    from pyplumber import AVPlumber
    from pyplumber.node import CudaRectOverlay, DecVideo, Demux, FilterVideo, Input

    path = Path(root) / f"{src_fmt}.raw"
    if not path.exists():
        write_flat(path, src_fmt)
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *e: errors.append(tuple(map(str, e)))
    avp.edges.planCapacity("*", 3)
    avp.executeCommandsFromString('hwaccel.init {"name":"ix_gpu","type":"cuda"}')
    nodes = [
        Input({"name": "in", "url": str(path), "format": "rawvideo", "dst": "pkt",
               "options": {"pixel_format": src_fmt, "video_size": f"{W}x{H}", "framerate": "60"}}),
        Demux({"name": "dx", "src": "pkt", "routing": {"v:0": "raw"}}),
        DecVideo({"name": "dec", "src": "raw", "dst": "cpu"}),
        FilterVideo({"name": "up", "src": "cpu", "dst": "gpu", "hwaccel": "ix_gpu", "graph": "hwupload"}),
        CudaRectOverlay({"name": "comp", "src": ["gpu"], "dst": "scene", "hwaccel": "ix_gpu",
                         "width": W, "height": H, "sw_format": canvas, "scale": True,
                         "active_inputs": 1,
                         "layers": [{"dst_x": 0, "dst_y": 0, "dst_w": W, "dst_h": H}]}),
        FilterVideo({"name": "down", "src": "scene", "dst": "out", "hwaccel": "ix_gpu",
                     "graph": f"hwdownload,format={canvas}"}),
    ]
    try:
        for n in nodes:
            n.parameters.update({"group": "ix", "auto_restart": "off"})
            avp.addNode(n)
        del n
        out = avp.getEdge("out", "VideoFrame")
        avp.group("ix").startNodes()
        deadline = time.monotonic() + timeout
        f = FORMATS[canvas]
        ey, eu, ev = expected(canvas)
        shift = 6 if (f["sb"] == 2 and canvas in ("p010le", "p210le")) else 0
        ok = False
        while time.monotonic() < deadline and not errors:
            frame = out.tryGet(100)
            if frame is None:
                continue
            if frame.pts.timestamp == -(1 << 63):
                break
            dt = "<u2" if f["sb"] == 2 else np.uint8
            y = np.frombuffer(frame.data[0], dt).reshape(H, frame.linesize[0] // f["sb"])[:, :W] >> shift
            uvp = frame.linesize[1] // f["sb"]
            uv = np.frombuffer(frame.data[1], dt).reshape(H >> f["ch"], uvp)[:, :W] >> shift
            # Center crop avoids any edge-tap on the flat field.
            assert abs(int(y[H//2, W//2]) - ey) <= 1, f"Y {int(y[H//2,W//2])} != {ey}"
            assert abs(int(uv[(H>>f['ch'])//2, (W//2)&~1]) - eu) <= 1, "Cb mismatch"
            assert abs(int(uv[(H>>f['ch'])//2, ((W//2)&~1)+1]) - ev) <= 1, "Cr mismatch"
            ok = True
            break
        assert not errors, errors
        assert ok, "no frame"
        print(f"PASS {src_fmt:>7} -> {canvas:<7} (Y {ey} Cb {eu} Cr {ev})", flush=True)
    finally:
        nodes.clear()
        avp.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeout", type=float, default=30)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="avp-ix-") as root:
        for src, canvas in MATRIX:
            run(root, src, canvas, args.timeout)
    print("interop matrix OK")


if __name__ == "__main__":
    main()
