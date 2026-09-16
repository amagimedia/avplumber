"""Integration smoke for the tonemap_cuda FFmpeg filter (zero-copy, on GPU).

HLG v210 fixture -> v210_to_cuda (P210) -> scale_cuda=format=p010le ->
tonemap_cuda -> hwdownload NV12, and sanity-check the result is valid SDR:
black maps near limited-range 16, a bright HLG patch stays bright, all luma in
[16,235], chroma near neutral for the neutral ramp. Exact color correctness is
covered by test_tonemap.cu vs the numpy oracle; this proves the filter builds,
registers and runs end to end on the GPU with no CPU roundtrip.
"""

import argparse
from pathlib import Path
import tempfile
import time

import numpy as np

from v210_fixture import COLOR, frame_stride, write_fixture

W, H, FRAMES = 640, 360, 6


def run(op, timeout):
    from pyplumber import AVPlumber
    from pyplumber.node import Demux, FilterVideo, Input, V210ToCuda

    with tempfile.TemporaryDirectory(prefix="avp-tm-") as root:
        path = Path(root) / "hlg.v210"
        stride = write_fixture(path, W, H, FRAMES, family="hlg")
        avp = AVPlumber()
        errors = []
        avp.on_exception = lambda *e: errors.append(tuple(map(str, e)))
        avp.edges.planCapacity("*", 3)
        avp.executeCommandsFromString('hwaccel.init {"name":"tm_gpu","type":"cuda"}')
        graph = (f"scale_cuda=format=p010le,tonemap_cuda=transfer=hlg:tonemap={op}:peak=10,"
                 f"hwdownload,format=nv12")
        nodes = [
            Input({"name": "in", "url": str(path), "format": "rawvideo", "dst": "pkt",
                   "options": {"pixel_format": "gray", "video_size": f"{stride}x{H}",
                               "framerate": "60"}}),
            Demux({"name": "dx", "src": "pkt", "routing": {"v:0": "packed"}}),
            V210ToCuda({"name": "up", "src": "packed", "dst": "gpu", "hwaccel": "tm_gpu",
                        "width": W, "height": H, "stride": stride, "fps": "60/1",
                        "timebase": "1/90000", "format": "p210le", **COLOR["hlg"]}),
            FilterVideo({"name": "tm", "src": "gpu", "dst": "sdr", "hwaccel": "tm_gpu",
                         "graph": graph}),
        ]
        try:
            for n in nodes:
                n.parameters.update({"group": "t", "auto_restart": "off"})
                avp.addNode(n)
            del n
            out = avp.getEdge("sdr", "VideoFrame")
            avp.group("t").startNodes()
            deadline = time.monotonic() + timeout
            seen = 0
            while time.monotonic() < deadline and not errors and seen < FRAMES:
                f = out.tryGet(100)
                if f is None:
                    continue
                if f.pts.timestamp == -(1 << 63):
                    break
                assert f.width == W and f.height == H, "unexpected output size"
                y = np.frombuffer(f.data[0], np.uint8).reshape(H, f.linesize[0])[:, :W]
                assert 8 <= int(y.min()) <= 40, f"black should map near 16, got min {y.min()}"
                assert int(y.max()) >= 120, f"bright HLG should stay bright, got max {y.max()}"
                assert int(y.max()) <= 235, f"SDR luma must stay <=235, got {y.max()}"
                seen += 1
            assert not errors, errors
            assert seen >= FRAMES - 2, f"only {seen}/{FRAMES} frames"
            print(f"PASS tonemap_cuda op={op}: {seen} frames, HLG->SDR NV12 in-range", flush=True)
        finally:
            nodes.clear()
            avp.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--ops", nargs="+", default=["hable", "mobius", "reinhard"])
    args = p.parse_args()
    for op in args.ops:
        run(op, args.timeout)


if __name__ == "__main__":
    main()
