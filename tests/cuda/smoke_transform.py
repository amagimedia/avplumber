"""Remote GPU smoke for cuda_transform against cuda_rect_overlay.

Each output of the transform node must be byte-identical to a cuda_rect_overlay drawing the same
layers from the same frame, carry that frame's timestamp, and an output with `fps` must take the
first frame of each slot. That holds for the two outputs the node passes on undrawn as well. The
transform's layers default to `filter: auto` (the 3:1 letterbox is multisampled), so the reference
overlay names that filter; a transform that defaulted to bilinear would differ there. The
downloads are the verification boundary; that those two were not copied shows only in the node's
log ("passes the input frame on").
"""
from pathlib import Path
import tempfile

import numpy as np

from _harness import finish, make_avp
from pyplumber import node as api
from pyplumber.mixer.inputs import build_raw420_input

EOF = -(1 << 63)
WIDTH, HEIGHT, FPS, FRAMES = 192, 108, 30, 30
OUTPUTS = {
    # name: (canvas width, canvas height, layers, fps)
    "letterbox": (64, 48, [{"dst_x": 0, "dst_y": 6, "dst_w": 64, "dst_h": 36}], None),
    "crop": (96, 64, [{"crop": {"x": 32, "y": 20, "w": 128, "h": 72},
                       "dst_x": 0, "dst_y": 4, "dst_w": 96, "dst_h": 54}], "15/1"),
    # Same size, format and storage as the input: the node passes the input frame on undrawn.
    "copy": (WIDTH, HEIGHT, [{"dst_x": 0, "dst_y": 0, "dst_w": WIDTH, "dst_h": HEIGHT}], None),
    "copy_half_rate": (WIDTH, HEIGHT, [{"dst_x": 0, "dst_y": 0, "dst_w": WIDTH, "dst_h": HEIGHT}], "15/1"),
}


def planes(frame, width, height):
    return b"".join(np.frombuffer(frame.data[i], np.uint8).reshape(rows, frame.linesize[i])[:, :width].tobytes()
                    for i, rows in enumerate((height, height // 2)))


def main():
    rng = np.random.default_rng(7)
    source = [rng.integers(16, 236, HEIGHT * WIDTH * 3 // 2, dtype=np.uint8).tobytes() for _ in range(6)]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "noise.raw"
        path.write_bytes(b"".join(source))
        avp, errors = make_avp("gpu", capacity=4)
        nodes = []
        try:
            edge = build_raw420_input(avp, api, "raw", str(path), width=WIDTH, height=HEIGHT, pixel_format="nv12",
                                      fps=FPS, group="probe", hwaccel="gpu", loop=True, pinned=True)
            branches = ["to_transform"] + [f"to_{name}" for name in OUTPUTS]
            nodes.append(api.Split({"name": "split", "src": edge, "dst": branches, "group": "probe"}))
            nodes.append(api.CudaTransform({
                "name": "transform", "src": "to_transform", "hwaccel": "gpu", "group": "probe",
                "outputs": [{"dst": f"t_{name}", "width": w, "height": h, "layers": layers,
                             **({"fps": fps} if fps else {})}
                            for name, (w, h, layers, fps) in OUTPUTS.items()]}))
            for name, (w, h, layers, _) in OUTPUTS.items():
                nodes.append(api.CudaRectOverlay({
                    "name": f"overlay_{name}", "src": [f"to_{name}"], "dst": f"o_{name}", "hwaccel": "gpu",
                    "group": "probe", "width": w, "height": h,
                    "layers": [{**layer, "filter": "auto"} for layer in layers]}))
                for side in "to":
                    nodes.append(api.FilterVideo({
                        "name": f"download_{side}_{name}", "src": f"{side}_{name}", "dst": f"r_{side}_{name}",
                        "group": "probe", "hwaccel": "gpu", "graph": "hwdownload,format=nv12"}))
            for node in nodes:
                avp.addNode(node)
            edges = {(side, name): avp.getEdge(f"r_{side}_{name}", "VideoFrame") for name in OUTPUTS for side in "to"}
            avp.group("probe").startNodes()

            seen = {key: {} for key in edges}
            import time
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not errors and min(len(seen[("o", n)]) for n in OUTPUTS) < FRAMES:
                for key, out in edges.items():
                    frame = out.tryGet(5)
                    if frame is None or frame.pts.timestamp == EOF:
                        continue
                    w, h = OUTPUTS[key[1]][:2]
                    seen[key][frame.pts.timestamp] = planes(frame, w, h)
                    timebase = (frame.pts.timebase.num, frame.pts.timebase.den)
            assert not errors, errors

            for name, (w, h, layers, fps) in OUTPUTS.items():
                ours, reference = seen[("t", name)], seen[("o", name)]
                common = sorted(set(ours) & set(reference))
                assert len(common) >= (FRAMES // 3 if fps else FRAMES - 4), (name, len(ours), len(reference))
                different = [pts for pts in common if ours[pts] != reference[pts]]
                assert not different, f"{name}: {len(different)} of {len(common)} frames differ from cuda_rect_overlay"
                assert len({reference[pts] for pts in common}) >= 3, f"{name}: source frames did not vary"
                if fps:
                    # The rule: the first input frame of each 1/fps slot, slots shifted by a quarter.
                    num, den = (int(v) for v in fps.split("/"))
                    slot = lambda pts: (4 * pts * timebase[0] * num // (timebase[1] * den) + 1) >> 2
                    full = sorted(seen[("t", "copy")])   # every input frame the node saw
                    lo, hi = max(min(ours), full[0]), min(max(ours), full[-1])
                    expected, last = [], None
                    for pts in full:
                        if slot(pts) != last:
                            last = slot(pts)
                            expected.append(pts)
                    expected = [pts for pts in expected if lo < pts < hi]
                    got = [pts for pts in sorted(ours) if lo < pts < hi]
                    assert got == expected, (f"{name}: fps output took {got[:12]}, the slot rule gives {expected[:12]}; "
                                             f"input {full[:16]} timebase {timebase}")
                    assert len(got) >= FRAMES // 3, (name, len(got))
                gaps = sorted(set(np.diff(sorted(reference)).tolist()))
                print(f"PASS {name}: {len(common)} frames byte-identical to cuda_rect_overlay"
                      + (f", {len(ours)} frames at {fps} taken on slot starts" if fps else "")
                      + f" (input timestamp steps {gaps})", flush=True)
            copy = seen[("t", "copy")]
            assert all(data in source for data in copy.values()), "1:1 output is not an exact copy of the source"
            print(f"PASS copy: {len(copy)} frames are exact copies of the source", flush=True)
        finally:
            finish(avp, nodes)


if __name__ == "__main__":
    main()
