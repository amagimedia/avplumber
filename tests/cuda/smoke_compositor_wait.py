"""Remote GPU smoke: a mixer_compositor that looks for input at intervals composes what one
woken by every frame composes.

Twelve looped raw sources at the output rate feed one compositor, first woken by each frame
(the default), then with input_poll_ms. Both must draw every tick with no missed deadline and
no repeated or discarded source frame; the second must look for input a bounded number of
times per tick, however many sources there are.
"""
from pathlib import Path
import tempfile
import time

import numpy as np

from _harness import finish, make_avp, wait_for
from pyplumber import node as api
from pyplumber.mixer.color import Color
from pyplumber.mixer.inputs import build_raw420_input

WIDTH, HEIGHT, FPS, SOURCES, COLUMNS = 96, 64, 30, 12, 4
POLL_MS = 8


def playout(avp):
    try:
        return avp.node("compose").getObject("status")["playout"]
    except RuntimeError:   # the group is still creating its nodes
        return {"frames": 0}


def check(directory, poll_ms):
    path = Path(directory) / "pattern.raw"
    y = np.full((HEIGHT, WIDTH), 96, np.uint8)
    uv = np.full((HEIGHT // 2, WIDTH), 128, np.uint8)
    path.write_bytes((y.tobytes() + uv.tobytes()) * 6)
    avp, errors = make_avp("wait_gpu", capacity=3)
    nodes = []
    try:
        edges = [build_raw420_input(avp, api, str(i), str(path), width=WIDTH, height=HEIGHT, pixel_format="nv12",
                                    fps=FPS, group="probe", hwaccel="wait_gpu", loop=True, direct=True,
                                    color=Color("sdr"), native_rate=True) for i in range(SOURCES)]
        rows = -(-SOURCES // COLUMNS)
        compositor = api.MixerCompositor({
            "name": "compose", "src": edges, "dst": "composed", "group": "probe", "hwaccel": "wait_gpu",
            "width": WIDTH * COLUMNS, "height": HEIGHT * rows, "sw_format": "nv12", "color": "sdr",
            "fps": f"{FPS}/1", "active_inputs": (1 << SOURCES) - 1,
            "layers": [{"input": i, "dst_x": WIDTH * (i % COLUMNS), "dst_y": HEIGHT * (i // COLUMNS),
                        "dst_w": WIDTH, "dst_h": HEIGHT} for i in range(SOURCES)],
            **({} if poll_ms is None else {"input_poll_ms": poll_ms}),
        })
        sink = api.NullSink({"name": "sink", "src": "composed", "group": "probe"})
        for node in (compositor, sink):
            nodes.append(node)
            avp.addNode(node)
        avp.group("probe").startNodes()
        # Counters are published every 60 ticks; start counting once the sources run.
        wait_for(lambda: playout(avp)["frames"] >= 60, "the first composed ticks", errors=errors, timeout=30)
        avp.executeCommandsFromString("node.object.set compose reset_counters true")
        first = playout(avp)
        started = time.monotonic()
        wait_for(lambda: playout(avp)["frames"] >= first["frames"] + 180, "six seconds of ticks", errors=errors, timeout=30)
        last, elapsed = playout(avp), time.monotonic() - started
        assert not errors, errors
        assert last["input_poll_ms"] == (poll_ms or 0), last["input_poll_ms"]
        ticks = last["frames"] - first["frames"]
        assert abs(ticks / elapsed - FPS) < 3, f"{ticks} ticks in {elapsed:.1f} s"
        assert last["missed_deadlines"] == 0, last["missed_deadlines"]
        assert (last["repeats"], last["discarded"], last["overflow"]) == (0, 0, 0), last
        passes = (last["input_passes"] - first["input_passes"]) / ticks
        if poll_ms:
            # A look at each deadline, one after each draw, and the polls in between.
            assert passes <= 1000 / FPS / poll_ms + 4, f"{passes:.1f} looks for input per tick"
        print(f"PASS: {ticks} ticks of {SOURCES} sources, no missed deadline, repeat or discard; "
              f"{passes:.1f} looks for input per tick ({'every ' + str(poll_ms) + ' ms' if poll_ms else 'woken by every frame'})",
              flush=True)
    finally:
        finish(avp, nodes)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        for poll_ms in (None, POLL_MS):
            check(directory, poll_ms)
