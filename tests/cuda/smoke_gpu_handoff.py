"""Remote GPU smoke: a mixer_compositor that hands its frames on before the GPU has drawn them
(`gpu_handoff`) draws what one that waits for every draw draws.

Sixteen looped 1080p raw sources, each picture one luma value, from upload pools of a few frames
that are written again as soon as a frame is let go, fill a 4x4 grid. A source frame let go
before the kernel that reads it has run, or an output frame read before its draw has finished,
shows up as a tile that is not one whole source picture: every tile of every downloaded output
frame must be one of the source values, with the handoff on and off.

Run from the repository root with its CUDA Python module on PYTHONPATH.
"""
from pathlib import Path
import tempfile
import time

import numpy as np

from _harness import drain, finish, make_avp, nv12_planes, wait_for
from pyplumber import node as api
from pyplumber.mixer.color import Color
from pyplumber.mixer.inputs import build_raw420_input

WIDTH, HEIGHT, FPS, COLUMNS, ROWS = 1920, 1080, 30, 4, 4
TILE_W, TILE_H = WIDTH // COLUMNS, HEIGHT // ROWS
VALUES = [40, 70, 100, 130, 160, 190]   # one luma value per picture of the clip
FRAMES = 150


def playout(avp):
    try:
        return avp.node("compose").getObject("status")["playout"]
    except RuntimeError:   # the group is still creating its nodes
        return {"frames": 0, "missed_deadlines": 0}


def check(path, handoff):
    sources = COLUMNS * ROWS
    avp, errors = make_avp("handoff_gpu", capacity=3)
    nodes = []
    try:
        edges = [build_raw420_input(avp, api, str(i), str(path), width=WIDTH, height=HEIGHT, pixel_format="nv12",
                                    fps=FPS, group="probe", hwaccel="handoff_gpu", loop=True, direct=True,
                                    resident=True, pool_frames=3, color=Color("sdr"), native_rate=True)
                 for i in range(sources)]
        compositor = api.MixerCompositor({
            "name": "compose", "src": edges, "dst": "composed", "group": "probe", "hwaccel": "handoff_gpu",
            "width": WIDTH, "height": HEIGHT, "sw_format": "nv12", "color": "sdr", "fps": f"{FPS}/1",
            "active_inputs": (1 << sources) - 1, "gpu_handoff": handoff,
            "layers": [{"input": i, "dst_x": TILE_W * (i % COLUMNS), "dst_y": TILE_H * (i // COLUMNS),
                        "dst_w": TILE_W, "dst_h": TILE_H} for i in range(sources)],
        })
        verify = api.FilterVideo({"name": "verify", "src": "composed", "dst": "result", "group": "probe",
                                  "hwaccel": "handoff_gpu", "graph": "hwdownload,format=nv12"})
        for node in (compositor, verify):
            nodes.append(node)
            avp.addNode(node)
        output = avp.getEdge("result", "VideoFrame")
        avp.group("probe").startNodes()
        checked, torn = 0, []
        for frame in drain(output, errors, timeout=40, limit=FRAMES):
            luma, _ = nv12_planes(frame)
            for tile in range(sources):
                x, y = TILE_W * (tile % COLUMNS), TILE_H * (tile // COLUMNS)
                block = luma[y + 2:y + TILE_H - 2, x + 2:x + TILE_W - 2]
                low, high = int(block.min()), int(block.max())
                if high - low > 1 or min(abs(low - v) for v in VALUES) > 1:
                    torn.append((checked, tile, low, high))
            checked += 1
        state = playout(avp)
        assert not errors, errors
        assert checked == FRAMES, f"{checked} of {FRAMES} frames"
        assert not torn, f"{len(torn)} tiles are not one source picture, first {torn[:5]}"
        assert state["missed_deadlines"] == 0, state
        print(f"PASS gpu_handoff={handoff}: {checked} frames of {sources} 1080p tiles, every tile one whole "
              f"source picture, {state['frames']} ticks, no missed deadline", flush=True)
    finally:
        finish(avp, nodes)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "values.raw"
        uv = np.full((HEIGHT // 2, WIDTH), 128, np.uint8).tobytes()
        path.write_bytes(b"".join(np.full((HEIGHT, WIDTH), v, np.uint8).tobytes() + uv for v in VALUES))
        for handoff in (True, False):
            check(path, handoff)
