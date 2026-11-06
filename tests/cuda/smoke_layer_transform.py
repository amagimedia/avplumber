"""Remote GPU smoke for transformed layers through cuda_rect_overlay (the affine entries).

A smooth test picture is drawn by overlays that differ only in their layer: rotated 30 degrees
(the corners keep the clear value, the centre is the picture's centre), rotated 360 degrees and
switched at run time from 30 to 0 (both then byte-identical to the plain draw: the plain entries
draw them), and a 0.75x crop stepping 0.25 px per overlay over a luma ramp (the mean shifts by
exactly the ramp's slope times the step, monotone).
"""
from pathlib import Path
import json
import tempfile
import time

import numpy as np

from _harness import EOF, finish, make_avp
from pyplumber import node as api
from pyplumber.mixer.inputs import build_raw420_input

W, H, FPS = 384, 216, 30
RAMP_W, RAMP_H = 96, 48
FULL = {"dst_x": 0, "dst_y": 0, "dst_w": W, "dst_h": H}
OVERLAYS = {   # name: (source, canvas w, canvas h, layer)
    "plain": ("pic", W, H, FULL),
    "rot30": ("pic", W, H, {**FULL, "rotation_deg": 30}),
    "rot360": ("pic", W, H, {**FULL, "rotation_deg": 360}),
    "switch": ("pic", W, H, {**FULL, "rotation_deg": 30}),
    **{f"pan{k}": ("ramp", 64, 48, {"dst_x": 0, "dst_y": 0, "dst_w": 64, "dst_h": 48,
                                    "crop": {"x": 8 + k * 0.25, "y": 6, "w": 48, "h": 36}}) for k in range(4)},
}


def picture():
    y, x = np.mgrid[0:H, 0:W]
    luma = (110 + 80 * np.sin(x / 23.0) * np.cos(y / 31.0)).astype(np.uint8)
    return luma.tobytes() + bytes([128]) * (W * H // 2)


def ramp():
    luma = np.tile((20 + 2 * np.arange(RAMP_W)).astype(np.uint8), (RAMP_H, 1))
    return luma.tobytes() + bytes([128]) * (RAMP_W * RAMP_H // 2)


def luma(frame, w, h):
    return np.frombuffer(frame.data[0], np.uint8).reshape(h, frame.linesize[0])[:, :w].copy()


def main():
    with tempfile.TemporaryDirectory() as directory:
        paths = {"pic": Path(directory) / "pic.raw", "ramp": Path(directory) / "ramp.raw"}
        paths["pic"].write_bytes(picture() * 4)
        paths["ramp"].write_bytes(ramp() * 4)
        avp, errors = make_avp("gpu", capacity=4)
        nodes = []
        try:
            sources = {}
            for tag, (w, h) in {"pic": (W, H), "ramp": (RAMP_W, RAMP_H)}.items():
                sources[tag] = build_raw420_input(avp, api, tag, str(paths[tag]), width=w, height=h, pixel_format="nv12",
                                                  fps=FPS, group="probe", hwaccel="gpu", loop=True, pinned=True)
            for tag in sources:
                names = [n for n, o in OVERLAYS.items() if o[0] == tag]
                nodes.append(api.Split({"name": f"split_{tag}", "src": sources[tag], "dst": [f"to_{n}" for n in names],
                                        "group": "probe"}))
            for name, (_, w, h, layer) in OVERLAYS.items():
                nodes.append(api.CudaRectOverlay({"name": f"overlay_{name}", "src": [f"to_{name}"], "dst": f"o_{name}",
                                                  "hwaccel": "gpu", "group": "probe", "width": w, "height": h,
                                                  "layers": [layer]}))
                nodes.append(api.FilterVideo({"name": f"download_{name}", "src": f"o_{name}", "dst": f"r_{name}",
                                              "group": "probe", "hwaccel": "gpu", "graph": "hwdownload,format=nv12"}))
            for node in nodes:
                avp.addNode(node)
            edges = {name: avp.getEdge(f"r_{name}", "VideoFrame") for name in OVERLAYS}
            avp.group("probe").startNodes()

            seen = {name: [] for name in OVERLAYS}
            switched = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not errors and min(len(v) for v in seen.values()) < 40:
                for name, out in edges.items():
                    frame = out.tryGet(5)
                    if frame is None or frame.pts.timestamp == EOF:
                        continue
                    _, w, h, _ = OVERLAYS[name]
                    seen[name].append(luma(frame, w, h))
                if switched is None and len(seen["switch"]) >= 10:
                    avp.executeCommandsFromString("node.object.set overlay_switch layers " + json.dumps([FULL]))
                    switched = len(seen["switch"])
            assert not errors, errors

            plain = seen["plain"][-1]
            rot = seen["rot30"][-1]
            for y, x in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
                assert rot[y, x] == 16, f"rotated corner {x},{y} is {rot[y, x]}, not the clear value"
            centre = abs(int(rot[H // 2 - 1:H // 2 + 1, W // 2 - 1:W // 2 + 1].mean()) -
                         int(plain[H // 2 - 1:H // 2 + 1, W // 2 - 1:W // 2 + 1].mean()))
            assert centre <= 2, f"rotated centre is {centre} codes from the picture's centre"
            assert np.any(rot != plain), "rotation 30 drew the plain picture"
            print("PASS rotation_from_layers: corners clear, centre within", centre, "codes", flush=True)

            assert all(np.array_equal(f, plain) for f in seen["rot360"]), "rotation 360 is not the plain draw"
            late = seen["switch"][switched + 5:]
            assert late and all(np.array_equal(f, plain) for f in late), "switched to 0 deg but not the plain draw"
            assert not np.array_equal(seen["switch"][0], plain), "the switch overlay started unrotated"
            print(f"PASS rotation back to 0: rot360 and {len(late)} frames after the switch byte-identical to plain",
                  flush=True)

            means = [float(seen[f"pan{k}"][-1].mean()) for k in range(4)]
            steps = np.diff(means)
            # 0.25 px of crop at 0.75x on a 2-code/px ramp: 0.5 code a step; 0.05 px = 0.1 code.
            assert np.all(np.abs(steps - 0.5) <= 0.1), f"sub-pixel pan steps {steps} (means {means})"
            print("PASS subpixel_pan: mean luma steps", np.round(steps, 3).tolist(), flush=True)
        finally:
            finish(avp, nodes)


if __name__ == "__main__":
    main()
