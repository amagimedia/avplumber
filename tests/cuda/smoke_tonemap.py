"""Integration smoke for the tonemap_cuda FFmpeg filter (zero-copy, on GPU).

HLG v210 fixture -> v210_to_cuda (P210) -> scale_cuda=format=p010le ->
tonemap_cuda -> hwdownload NV12. The fixture's row-0 transfer checkpoints
(E = 0, 1/12, 1 -> codes 64, 502, 940 with neutral chroma) are compared
against the numpy oracle in tonemap_reference, so this validates the shipped
filter's colour math end to end rather than a copy of it. The filter uses
fast-math intrinsics, hence the +/-4 code tolerance.
"""

import argparse
from pathlib import Path
import tempfile

import numpy as np

from _harness import drain, finish, make_avp, start, v210_chain
from tonemap_reference import tonemap_codes
from v210_fixture import COLOR, write_fixture

W, H, FRAMES = 640, 360, 6
CHECK_Y = (64, 502, 940)      # row-0 gray patches, 12 columns each
CHECK_COLS = (6, 18, 30)      # patch centres


def run(op, timeout):
    from pyplumber.node import FilterVideo

    with tempfile.TemporaryDirectory(prefix="avp-tm-") as root:
        path = Path(root) / "hlg.v210"
        stride = write_fixture(path, W, H, FRAMES, family="hlg")
        avp, errors = make_avp("tm_gpu")
        nodes = []
        src = v210_chain(nodes, "hlg", path, width=W, height=H, stride=stride, fmt="p210le",
                         hwaccel="tm_gpu", color=COLOR["hlg"])
        nodes.append(FilterVideo({
            "name": "tm", "src": src, "dst": "sdr", "hwaccel": "tm_gpu",
            "graph": (f"scale_cuda=format=p010le,tonemap_cuda=transfer=hlg:tonemap={op}:peak=10,"
                      f"hwdownload,format=nv12")}))
        want = tonemap_codes(np.array(CHECK_Y), np.full(3, 512), np.full(3, 512), "hlg", 10.0, op)[0]
        try:
            out = start(avp, nodes, "t", "sdr")
            state = {}
            for f in drain(out, errors, timeout, FRAMES, state):
                y = np.frombuffer(f.data[0], np.uint8).reshape(H, f.linesize[0])[:, :W]
                uv = np.frombuffer(f.data[1], np.uint8).reshape(H // 2, f.linesize[1])[:, :W]
                got = y[0, list(CHECK_COLS)].astype(int)
                assert np.all(np.abs(got - want) <= 4), f"checkpoints {got.tolist()} vs oracle {want.tolist()}"
                assert np.all(np.abs(uv[0, :36].astype(int) - 128) <= 2), "neutral chroma drifted"
                assert 16 <= int(y.max()) <= 235, f"SDR luma out of range, max {y.max()}"
            assert not errors, errors
            assert state["count"] >= FRAMES - 2, f"only {state['count']}/{FRAMES} frames"
            print(f"PASS tonemap_cuda op={op}: checkpoints -> {want.tolist()} within 4 codes", flush=True)
        finally:
            finish(avp, nodes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--ops", nargs="+", default=["hable", "mobius", "reinhard"])
    args = p.parse_args()
    for op in args.ops:
        run(op, args.timeout)


if __name__ == "__main__":
    main()
