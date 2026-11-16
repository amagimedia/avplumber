"""GPU source boundary: exclude a neighbour without changing the moving picture.

Only fixture upload and verification download cross CPU. The production path
under test is the source-paced CUDA mixer, including per-frame metadata reset.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from _harness import drain, finish, make_avp, nv12_planes, start
from pyplumber import node as api
from pyplumber.mixer import source_compositor

W, H, FRAMES = 384, 216, 48


class Boundaries(api.PythonNode):
    def __init__(self, clipped):
        super().__init__({"data_type": "VideoFrame", "src": "input", "dst": "planned"})
        self.clipped = clipped
        self.tick = 0

    def process(self):
        frame = self._src.tryGet(200)
        if frame is None:
            return
        if frame.width > 0:
            layer = {
                "input": 0, "crop_subpixel": True,
                "crop": {"x": 160 + self.tick / 4, "y": 0, "w": 192, "h": H},
                "dst_w": W, "dst_h": H,
            }
            if self.clipped and (self.tick < 24 or self.tick >= 36):
                layer["source_clip"] = {
                    "x": 192 if self.tick < 24 else W + 10,
                    "y": 0, "w": 192, "h": H,
                }
            frame.metadata["layers"] = json.dumps({"0": layer})
            self.tick += 1
        self._dst.enqueue(frame)


def render(path, clipped):
    avp, errors = make_avp("@gpu")
    nodes = []
    try:
        nodes += [
            api.RawVideoFileCuda({
                "name": "fixture", "url": str(path), "dst": "input", "hwaccel": "@gpu",
                "pixel_format": "nv12", "video_size": f"{W}x{H}", "framerate": "25/1",
            }),
            Boundaries(clipped),
            source_compositor({
                "name": "mix", "src": ["planned"], "dst": "canvas", "hwaccel": "@gpu",
                "width": W, "height": H, "sw_format": "nv12", "metadata_key": "layers",
                "layers": [{"input": 0, "dst_w": W, "dst_h": H}],
            }),
            api.FilterVideo({
                "name": "measurement", "src": "canvas", "dst": "result", "hwaccel": "@gpu",
                "graph": "hwdownload,format=nv12",
            }),
        ]
        edge = start(avp, nodes, "probe", "result")
        state, pictures, pts = {}, [], []
        for frame in drain(edge, errors, timeout=60, state=state):
            pictures.append(tuple(p.copy() for p in nv12_planes(frame)))
            pts.append(frame.pts.timestamp * frame.pts.timebase.num / frame.pts.timebase.den)
        assert not errors and state["eof"] and len(pictures) == FRAMES, (errors, state)
        np.testing.assert_allclose(pts, np.arange(FRAMES) / 25, rtol=0, atol=1e-9)
        return pictures
    finally:
        finish(avp, nodes)


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "neighbours.nv12"
        y = np.full((H, W), 235, dtype=np.uint8)
        y[:, W // 2:] = np.linspace(80, 180, W // 2, dtype=np.uint8)
        uv = np.tile(np.array([32, 220], dtype=np.uint8), (H // 2, W // 2))
        uv[:, W // 2:] = np.tile(np.array([100, 150], dtype=np.uint8), (H // 2, W // 4))
        path.write_bytes((y.tobytes() + uv.tobytes()) * FRAMES)
        baseline, clipped = render(path, False), render(path, True)
        for tick, (before, after) in enumerate(zip(baseline, clipped)):
            if tick < 24:
                boundary = round((192 - (160 + tick / 4)) * 2)
                for plane, (a, b) in enumerate(zip(before, after)):
                    safe = (boundary + 7) // 2 * 2
                    np.testing.assert_array_equal(a[:, safe:], b[:, safe:])
                    assert np.all(b[:, :boundary - 4] == (16 if plane == 0 else 128))
                assert after[0].max() <= 180, "neighbour leaked through boundary/filter taps"
            elif tick < 36:
                for a, b in zip(before, after):
                    np.testing.assert_array_equal(a, b)  # omission clears the prior boundary
            else:
                assert np.all(after[0] == 16) and np.all(after[1] == 128)
        print("PASS source clip: 96 frames; neighbour excluded, affine motion unchanged, metadata reset, PTS/EOF")


if __name__ == "__main__":
    main()
