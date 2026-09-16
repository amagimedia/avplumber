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

import numpy as np

from _harness import drain, finish, frame_planes, make_avp, start, v210_chain
from v210_fixture import COLOR, FAMILIES, frame_stride, write_fixture

W, H, FRAMES = 384, 216, 6
# Tail-guard frames: EOF through the dual-input transition drops a variable
# number of in-flight trailing frames (recorded as a follow-up finding; the
# single-input flush drains correctly). The margin exceeds the bounded queue
# depths, the scored FRAMES are compared exactly, and the run stops there
# instead of racing the flush.
GEN_FRAMES = FRAMES + 8
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
        for index in range(GEN_FRAMES):
            for plane in planes444(index, source):
                stream.write(plane.astype("<u2").tobytes())


def reduce_scale(plane, factor):
    # The production kernel at an exact even-integer reduction lands on
    # tx = ty = 0.5 between the two central taps; halves are exact floats and
    # the store rounds half-up.
    p = plane.astype(np.float64)
    cols = (p[:, factor // 2 - 1::factor] + p[:, factor // 2::factor]) * 0.5
    rows = (cols[factor // 2 - 1::factor, :] + cols[factor // 2::factor, :]) * 0.5
    return np.floor(rows + 0.5).astype(np.int64)


def grid(n):
    """Tile grid for n sources: 2x2 cells up to 4 sources, 4x4 up to 16."""
    cols = 2 if n <= 4 else 4
    return cols, W // cols, H // cols


def scene_a(family, index, n=2):
    """n-tile grid over depth-aware black; tiles reduce by the column count."""
    cols, tw, th = grid(n)
    cw = W if family == "444" else W // 2
    y = np.full((H, W), 64, np.int64)
    u = np.full((H, cw), 512, np.int64)
    v = np.full((H, cw), 512, np.int64)
    for source in range(n):
        sy, su, sv = source_planes(family, index, source)
        x0, y0 = (source % cols) * tw, (source // cols) * th
        y[y0:y0 + th, x0:x0 + tw] = reduce_scale(sy, cols)
        cx0, ctw = (x0, tw) if family == "444" else (x0 // 2, tw // 2)
        u[y0:y0 + th, cx0:cx0 + ctw] = reduce_scale(su, cols)
        v[y0:y0 + th, cx0:cx0 + ctw] = reduce_scale(sv, cols)
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


def run(root, family, fmt, mode, coef, timeout, n=2):
    from pyplumber.node import CudaRectOverlay, FilterVideo

    avp, errors = make_avp("mix_gpu")
    nodes = []
    chains = [(f"a{s}", s) for s in range(n)] + [("b0", 0)]
    if family == "444":
        paths = []
        for source in range(n):
            path = Path(root) / f"src{source}.yuv444p10"
            if not path.exists():
                write_444_fixture(path, source)
            paths.append(path)
        edges = [upload_chain(nodes, tag, paths[src], "mix_gpu") for tag, src in chains]
    else:
        stride = frame_stride(W)
        paths = []
        for source in range(n):
            path = Path(root) / f"{family}_{source}.v210"
            if not path.exists():
                write_fixture(path, W, H, GEN_FRAMES, family=family, source=source)
            paths.append(path)
        edges = [v210_chain(nodes, tag, paths[src], width=W, height=H, stride=stride, fmt=fmt,
                            hwaccel="mix_gpu", color=COLOR[family]) for tag, src in chains]
    full = {"dst_x": 0, "dst_y": 0, "dst_w": W, "dst_h": H}
    cols, tw, th = grid(n)
    layers_a = [{"dst_x": (s % cols) * tw, "dst_y": (s // cols) * th, "dst_w": tw, "dst_h": th}
                for s in range(n)]
    nodes += [
        CudaRectOverlay({"name": "comp_a", "src": edges[:n], "dst": "scene_a", "hwaccel": "mix_gpu",
                         "width": W, "height": H, "sw_format": fmt, "scale": True,
                         "active_inputs": (1 << n) - 1, "layers": layers_a}),
        CudaRectOverlay({"name": "comp_b", "src": [edges[n]], "dst": "scene_b", "hwaccel": "mix_gpu",
                         "width": W, "height": H, "sw_format": fmt, "scale": True, "active_inputs": 1,
                         "layers": [full]}),
        FilterVideo({"name": "trans", "src": ["scene_a", "scene_b"], "dst": "mixed",
                     "hwaccel": "mix_gpu", "dst_frame_rate": "60/1",
                     "defer_preliminary_init": True,
                     "graph": f"transition_cuda=alpha='{coef}':mode={mode}:eval=init"}),
        FilterVideo({"name": "verify", "src": "mixed", "dst": "result", "hwaccel": "mix_gpu",
                     "graph": f"hwdownload,format={fmt}"}),
    ]
    try:
        result = start(avp, nodes, "test", "result")
        state = {}
        for index, frame in enumerate(drain(result, errors, timeout, FRAMES, state)):
            reference = blend(scene_a(family, index, n),
                              source_planes(family, index, 0), mode, coef)
            for plane, (actual, expected) in enumerate(zip(frame_planes(frame, fmt), reference)):
                np.testing.assert_array_equal(actual, expected,
                                              err_msg=f"frame {index} plane {plane}")
        assert not errors, errors
        assert state["count"] == FRAMES, f"scored frames {state['count']}/{FRAMES}"
    finally:
        finish(avp, nodes)


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
        # 16 simultaneous full-resolution sources drawn as a 4x4 grid.
        for family in args.families:
            if family == "444":
                continue  # 17 CPU upload chains add nothing over the v210 grid
            run(root, family, "p210le", "fade", 0.25, args.timeout, n=16)
            print(f"PASS {family}/p210le 4x4 grid of 16, fade alpha=0.25", flush=True)


if __name__ == "__main__":
    main()
