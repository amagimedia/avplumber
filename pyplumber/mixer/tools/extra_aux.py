"""The setup's extra aux outputs: monitor buses of random source layouts, appended after the show's
own buses (setup_runtime.py), each on a Janus RTP port pair of its own."""

from itertools import count
import random

from pyplumber.mixer.config import ConfigError, parse_aux_buses

GRID = 8                 # layouts cut the canvas on a 1/8 grid
CELLS = (4, 16)          # source cells per layout, at most the sources there are
LAYOUTS = 5              # per bus, each with a different cell count
PORT_STEP = 4            # as the demo's own buses (5008, 5012): then 5016, 5020, ...


def aux_encode(encode):
    """A codec change also replaces the old profile; every AUX remains SDR."""
    codec = encode.get("codec", "h264_nvenc")
    return {**encode, "codec": codec, "profile": "main" if codec == "hevc_nvenc" else "high", "color": "sdr"}


def _even(value):
    return value // 2 * 2


def _layout(rng, cfg, cells):
    """*cells* cells of distinct sources tiling the canvas: the largest cell is cut across its longer
    side on a grid line until there are enough. While there are fewer than 16, the largest covers at
    least five of the 64 grid squares, so it always has a line to cut on."""
    rects = [(0, 0, GRID, GRID)]   # grid lines x0, y0, x1, y1
    while len(rects) < cells:
        rects.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
        x0, y0, x1, y1 = rects.pop()
        if x1 - x0 >= y1 - y0:
            cut = rng.randint(x0 + 1, x1 - 1)
            rects += [(x0, y0, cut, y1), (cut, y0, x1, y1)]
        else:
            cut = rng.randint(y0 + 1, y1 - 1)
            rects += [(x0, y0, x1, cut), (x0, cut, x1, y1)]
    xs = [_even(cfg.canvas_w * i // GRID) for i in range(GRID + 1)]
    ys = [_even(cfg.canvas_h * i // GRID) for i in range(GRID + 1)]
    rects.sort(key=lambda r: (r[1], r[0]))
    return {"cells": [{"role": "source", "source": source, "x": xs[x0], "y": ys[y0],
                       "w": xs[x1] - xs[x0], "h": ys[y1] - ys[y0]}
                      for source, (x0, y0, x1, y1) in zip(rng.sample(range(len(cfg.sources)), cells), rects)]}


def layouts(bus_id, cfg):
    """Up to LAYOUTS layouts of *cfg*'s sources with different cell counts, fewest first: the same
    for the same bus id and source list."""
    rng = random.Random(f"{bus_id}:{','.join(s.id for s in cfg.sources)}")
    sizes = range(min(CELLS[0], len(cfg.sources)), min(CELLS[1], len(cfg.sources)) + 1)
    return [_layout(rng, cfg, cells) for cells in sorted(rng.sample(sizes, min(LAYOUTS, len(sizes))))]


def extra_buses(cfg, kept, wanted, floor_port, encode):
    """*wanted* extra buses for the show *cfg* describes with its own buses; its sources are those
    the cells show. *kept*, the previous show's extra buses, come first, each keeping its layouts
    while they still fit: another orientation, or a cell beyond the sources, draws new ones. New
    buses take the next free ids aux0, aux1, ..., the labels Aux 0, Aux 1, ... by position, and RTP
    ports PORT_STEP apart above every port in use and *floor_port*. Every one encodes at *encode*,
    the setup's shared codec, NVENC preset and bitrate_kbps of extra aux outputs."""
    encode = aux_encode(encode)
    result = []
    for bus in kept[:wanted]:
        try:
            parse_aux_buses([bus], cfg)
        except ConfigError:
            specs = layouts(bus["id"], cfg)
            bus = {**bus, "layout": specs[0], "layouts": specs}
        result.append({**bus, "renditions": [{**bus["renditions"][0], **encode}]})
    port = max([floor_port, *(r.port for r in cfg.renditions), *(b.renditions[0].port for b in cfg.aux_buses),
                *(b["renditions"][0]["port"] for b in result)])
    taken = {b.id for b in cfg.aux_buses} | {b["id"] for b in result}
    ids = (f"aux{k}" for k in count() if f"aux{k}" not in taken)
    for position in range(len(result), wanted):
        bus_id, port = next(ids), port + PORT_STEP
        specs = layouts(bus_id, cfg)
        result.append({"id": bus_id, "label": f"Aux {position}", "layout": specs[0], "layouts": specs,
                       "renditions": [{"id": "monitor", "port": port, **encode}]})
    return result
