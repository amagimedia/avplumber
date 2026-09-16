"""Two-source 10-bit mixer smoke: A/B cuda_rect_overlay -> transition_cuda ->
raw verification against an exact CPU reference.

P210 runs ingest packed v210 (HLG and SDR-promoted families) through
v210_to_cuda; the planar 444 run uploads a labeled CPU fixture. Scene A is a
two-tile grid (scaled path), scene B is source 0 fullscreen (copy path), and
the transition blends them at exact binary-fraction alphas so the float
arithmetic reproduces bit-exactly on the CPU. Run on the NVIDIA host with the
FFmpeg 8.1 avplumber module; the download is solely the verification boundary.
"""

import argparse
from pathlib import Path
import tempfile
import time

import numpy as np

from smoke_v210_to_cuda import frame_planes
from v210_fixture import COLOR, FAMILIES, write_fixture

W, H, FRAMES = 384, 216, 6
TRANSITIONS = (("fade", 0.0), ("fade", 0.25), ("fade", 1.0), ("wipe_left", 0.5))


def planes444(index, source):
    row = np.arange(H, dtype=np.int64)[:, None]
    x = np.arange(W, dtype=np.int64)[None, :]
    k = index + source * 1000
    y = 64 + (x + row * 17 + k * 101) % 877
    u = 64 + (x * 3 + row * 29 + k * 59) % 897
    v = 64 + (x * 7 + row * 13 + k * 83) % 897
    return y, u, v


def source_planes(family, index, source):
    if family == "444":
        return planes444(index, source)
    return tuple(p.astype(np.int64) for p in FAMILIES[family](W, H, index + source * 1000))


def write_444_fixture(path, source):
    with Path(path).open("wb") as stream:
        for index in range(FRAMES):
            for plane in planes444(index, source):
                stream.write(plane.astype("<u2").tobytes())


def half_scale(plane):
    # The production kernel at exact 2x reduction averages column then row
    # pairs (tx = ty = 0.5) and stores round-half-up; quarters are exact floats.
    p = plane.astype(np.float64)
    cols = (p[:, 0::2] + p[:, 1::2]) * 0.5
    rows = (cols[0::2, :] + cols[1::2, :]) * 0.5
    return np.floor(rows + 0.5).astype(np.int64)


def scene_a(family, index):
    """Two-tile top-row grid over depth-aware black."""
    cw = W if family == "444" else W // 2
    y = np.full((H, W), 64, np.int64)
    u = np.full((H, cw), 512, np.int64)
    v = np.full((H, cw), 512, np.int64)
    for source, x0 in ((0, 0), (1, W // 2)):
        sy, su, sv = source_planes(family, index, source)
        y[:H // 2, x0:x0 + W // 2] = half_scale(sy)
        cx0 = x0 if family == "444" else x0 // 2
        u[:H // 2, cx0:cx0 + su.shape[1] // 2] = half_scale(su)
        v[:H // 2, cx0:cx0 + sv.shape[1] // 2] = half_scale(sv)
    return y, u, v


def blend(a_planes, b_planes, mode, coef):
    """transition_cuda word arithmetic: per-plane wipe position over that
    plane's own pixel width, round-half-up store."""
    out = []
    for pa, pb in zip(a_planes, b_planes):
        width = pa.shape[1]
        if mode == "fade":
            alpha = np.full(width, coef)
        else:  # wipe_left
            alpha = ((np.arange(width) + 0.5) / width <= coef).astype(np.float64)
        out.append(np.floor(alpha * pb + (1 - alpha) * pa + 0.5).astype(np.int64))
    return out


def v210_chain(nodes, tag, path, stride, family, fmt, hwaccel):
    from pyplumber.node import Demux, Input, V210ToCuda
    nodes += [
        Input({"name": f"in_{tag}", "url": str(path), "format": "rawvideo", "dst": f"pkt_{tag}",
               "options": {"pixel_format": "gray", "video_size": f"{stride}x{H}", "framerate": "60"}}),
        Demux({"name": f"demux_{tag}", "src": f"pkt_{tag}", "routing": {"v:0": f"packed_{tag}"}}),
        V210ToCuda({"name": f"unpack_{tag}", "src": f"packed_{tag}", "dst": f"gpu_{tag}",
                    "hwaccel": hwaccel, "width": W, "height": H, "stride": stride, "fps": "60/1",
                    "timebase": "1/90000", "format": fmt, **COLOR[family]}),
    ]
    return f"gpu_{tag}"


def upload_chain(nodes, tag, path, hwaccel):
    from pyplumber.node import DecVideo, Demux, FilterVideo, Input
    nodes += [
        Input({"name": f"in_{tag}", "url": str(path), "format": "rawvideo", "dst": f"pkt_{tag}",
               "options": {"pixel_format": "yuv444p10le", "video_size": f"{W}x{H}", "framerate": "60"}}),
        Demux({"name": f"demux_{tag}", "src": f"pkt_{tag}", "routing": {"v:0": f"raw_{tag}"}}),
        DecVideo({"name": f"dec_{tag}", "src": f"raw_{tag}", "dst": f"cpu_{tag}"}),
        FilterVideo({"name": f"up_{tag}", "src": f"cpu_{tag}", "dst": f"gpu_{tag}",
                     "hwaccel": hwaccel, "graph": "hwupload"}),
    ]
    return f"gpu_{tag}"


def run(root, family, fmt, mode, coef, timeout):
    from pyplumber import AVPlumber
    from pyplumber.node import CudaRectOverlay, FilterVideo

    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(tuple(map(str, error)))
    avp.edges.planCapacity("*", 3)
    avp.executeCommandsFromString('hwaccel.init {"name":"mix_gpu","type":"cuda"}')
    nodes = []
    if family == "444":
        paths = []
        for source in (0, 1):
            path = Path(root) / f"src{source}.yuv444p10"
            if not path.exists():
                write_444_fixture(path, source)
            paths.append(path)
        edges = [upload_chain(nodes, tag, paths[src], "mix_gpu")
                 for tag, src in (("a0", 0), ("a1", 1), ("b0", 0))]
    else:
        from v210_fixture import frame_stride
        stride = frame_stride(W)
        paths = []
        for source in (0, 1):
            path = Path(root) / f"{family}_{source}.v210"
            if not path.exists():
                write_fixture(path, W, H, FRAMES, family=family, source=source)
            paths.append(path)
        edges = [v210_chain(nodes, tag, paths[src], stride, family, fmt, "mix_gpu")
                 for tag, src in (("a0", 0), ("a1", 1), ("b0", 0))]
    full = {"dst_x": 0, "dst_y": 0, "dst_w": W, "dst_h": H}
    nodes += [
        CudaRectOverlay({"name": "comp_a", "src": edges[:2], "dst": "scene_a", "hwaccel": "mix_gpu",
                         "width": W, "height": H, "sw_format": fmt, "scale": True, "active_inputs": 3,
                         "layers": [{"dst_x": 0, "dst_y": 0, "dst_w": W // 2, "dst_h": H // 2},
                                    {"dst_x": W // 2, "dst_y": 0, "dst_w": W // 2, "dst_h": H // 2}]}),
        CudaRectOverlay({"name": "comp_b", "src": [edges[2]], "dst": "scene_b", "hwaccel": "mix_gpu",
                         "width": W, "height": H, "sw_format": fmt, "scale": True, "active_inputs": 1,
                         "layers": [full]}),
        FilterVideo({"name": "trans", "src": ["scene_a", "scene_b"], "dst": "mixed",
                     "hwaccel": "mix_gpu",
                     "graph": f"transition_cuda=alpha='{coef}':mode={mode}:eval=init"}),
        FilterVideo({"name": "verify", "src": "mixed", "dst": "result", "hwaccel": "mix_gpu",
                     "graph": f"hwdownload,format={fmt}"}),
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
            assert index < FRAMES, "extra frame"
            reference = blend(scene_a(family, index),
                              source_planes(family, index, 0), mode, coef)
            for actual, expected in zip(planes_of(frame, fmt), reference):
                np.testing.assert_array_equal(actual, expected)
            index += 1
        assert not errors, errors
        assert eof and index == FRAMES, f"EOF/count mismatch: eof={eof}, frames={index}/{FRAMES}"
    finally:
        nodes.clear()
        avp.shutdown()


def planes_of(frame, fmt):
    if fmt == "p210le":
        return frame_planes(frame, fmt)
    y, u, v = [np.frombuffer(data, dtype="<u2").reshape(frame.height, pitch // 2)[:, :frame.width]
               for data, pitch in zip(frame.data[:3], frame.linesize[:3])]
    return y, u, v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--families", nargs="+", default=["sdr8", "hlg", "444"])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="avp-mix10-") as root:
        for family in args.families:
            fmt = "yuv444p10le" if family == "444" else "p210le"
            for mode, coef in TRANSITIONS:
                run(root, family, fmt, mode, coef, args.timeout)
                print(f"PASS {family}/{fmt} {mode} alpha={coef}", flush=True)


if __name__ == "__main__":
    main()
