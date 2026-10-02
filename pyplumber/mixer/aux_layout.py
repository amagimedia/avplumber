"""AUX bus layouts as data: cells of the canvas, each drawing the preview ("pvw"), the program
("pgm"), an assigned scene ("slot") or one source ("source"), and the compositions a bus's
mixer_pvw_follow node sets for them. Code finds cells by role, never by position or count."""
from __future__ import annotations

from .config import ConfigError, scene_layers
from .control import source_mask_param

PRESETS = ("pgm_pvw_grid", "source_pages")
ROLES = {"pvw": (), "pgm": (), "slot": ("slot",), "source": ("source",)}   # role: its keys besides the rect
RECT = ("x", "y", "w", "h")


def _even(value):
    return int(value) // 2 * 2


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def grid_cells(cfg):
    """The pgm_pvw_grid preset: PVW and PGM over eight scene slots, in canvas pixels."""
    w, h = cfg.canvas_w, cfg.canvas_h
    xs = [_even(w * i // 4) for i in range(5)]
    ys = [_even(h * i // 4) for i in range(5)]
    cells = [{"role": "pvw", "x": 0, "y": 0, "w": xs[2], "h": ys[2]},
             {"role": "pgm", "x": xs[2], "y": 0, "w": w - xs[2], "h": ys[2]}]
    for i in range(8):
        col, row = i % 4, 2 + i // 4
        cells.append({"role": "slot", "slot": i, "x": xs[col], "y": ys[row],
                      "w": xs[col + 1] - xs[col], "h": ys[row + 1] - ys[row]})
    return cells


def page_grid(cfg):
    """Tiles of one source page: 2 by 6 on a portrait canvas, 4 by 3 on a landscape one."""
    w, h = cfg.canvas_w, cfg.canvas_h
    cols, rows = (2, 6) if h > w else (4, 3)
    cw, ch = w // cols, h // rows
    tw = _even(cw - 4)
    th = _even(tw * 9 / 16)
    if th > ch - 4:
        th = _even(ch - 4)
        tw = _even(th * 16 / 9)
    return [{"x": _even(c * cw + (cw - tw) / 2), "y": _even(r * ch + (ch - th) / 2), "w": tw, "h": th}
            for r in range(rows) for c in range(cols)]


def page_count(cfg):
    return -(-len(cfg.sources) // len(page_grid(cfg)))


def _cell(cfg, cell, where):
    if not isinstance(cell, dict) or cell.get("role") not in ROLES:
        raise ConfigError(f"{where}: role must be one of {', '.join(ROLES)}")
    keys = (*ROLES[cell["role"]], *RECT)
    if set(cell) != {"role", *keys}:
        raise ConfigError(f"{where}: a {cell['role']} cell has role, {', '.join(keys)}")
    if (not all(_integer(cell[k]) and cell[k] >= 0 and cell[k] % 2 == 0 for k in RECT) or not cell["w"] or
            not cell["h"] or cell["x"] + cell["w"] > cfg.canvas_w or cell["y"] + cell["h"] > cfg.canvas_h):
        raise ConfigError(f"{where}: x, y, w and h must be even pixels, w and h positive, "
                          f"inside the {cfg.canvas_w}x{cfg.canvas_h} canvas")
    if "slot" in cell and not (_integer(cell["slot"]) and cell["slot"] >= 0):
        raise ConfigError(f"{where}: slot must be a slot index")
    if "source" in cell and not (_integer(cell["source"]) and 0 <= cell["source"] < len(cfg.sources)):
        raise ConfigError(f"{where}: source must be a source index from 0 to {len(cfg.sources) - 1}")
    return {"role": cell["role"], **{k: cell[k] for k in keys}}


def parse_layout(cfg, spec):
    """*spec* validated and normalized: {"preset": "pgm_pvw_grid"}, {"preset": "source_pages",
    "page"} (0 when it names none) or {"cells": [...]}, whose slot cells number 0 to n-1."""
    if isinstance(spec, dict) and set(spec) == {"cells"} and isinstance(spec["cells"], list) and spec["cells"]:
        cells = [_cell(cfg, c, f"aux cell {i}") for i, c in enumerate(spec["cells"])]
        slots = sorted(c["slot"] for c in cells if c["role"] == "slot")
        if slots != list(range(len(slots))):
            raise ConfigError("aux slot cells must number 0 to n-1, each once")
        return {"cells": cells}
    preset = spec.get("preset") if isinstance(spec, dict) else None
    if preset == "pgm_pvw_grid" and set(spec) == {"preset"}:
        return {"preset": preset}
    if preset == "source_pages" and set(spec) <= {"preset", "page"}:
        page = spec.get("page", 0)
        if not (_integer(page) and 0 <= page < page_count(cfg)):
            raise ConfigError(f"source_pages page must be a page from 0 to {page_count(cfg) - 1}")
        return {"preset": preset, "page": page}
    raise ConfigError('aux layout is {"preset": "pgm_pvw_grid"}, {"preset": "source_pages", "page"} '
                      'or {"cells": [{"role", "x", "y", "w", "h"}, ...]}')


def same_kind(a, b):
    """Whether two layouts are one choice for the operator: the same preset (a page is runtime
    state), or the same cells."""
    return a.get("preset", a.get("cells")) == b.get("preset", b.get("cells"))


def layout_cells(cfg, spec, page=0):
    """The cells of a normalized *spec*; a source_pages layout shows *page*."""
    if "cells" in spec:
        return spec["cells"]
    if spec["preset"] == "pgm_pvw_grid":
        return grid_cells(cfg)
    grid = page_grid(cfg)
    first = page * len(grid)
    return [{"role": "source", "source": first + i, **rect} for i, rect in enumerate(grid[:len(cfg.sources) - first])]


def count(cells, role):
    return sum(c["role"] == role for c in cells)


def draws_program(cfg, specs):
    return any(count(layout_cells(cfg, spec), "pgm") for spec in specs)


def _layout(layers):
    return {"layers": layers, "active_inputs": source_mask_param(sum(1 << i for i in {l["input"] for l in layers}))}


def _tile(index, rect, z):
    """Input *index* contained in *rect*."""
    return {"input": index, "dst_x": rect["x"], "dst_y": rect["y"], "dst_w": rect["w"], "dst_h": rect["h"],
            "fit": "contain", "z": z}


def _cell_layers(cfg, scene, cell, first_z):
    """One layer per scene item, drawn into *cell*, z from *first_z* in item order."""
    indices = {s.id: i for i, s in enumerate(cfg.sources)}
    return [{**layer, "input": indices[name.split("#", 1)[0]], "z": first_z + z,
             "tile": {k: cell[k] for k in RECT},
             "scene_canvas": {"w": cfg.canvas_w, "h": cfg.canvas_h}}
            for z, (name, layer) in enumerate(scene_layers(cfg, scene).items())]


def _reserved(cfg):
    """z values each pvw cell reserves: the largest scene's item count."""
    return max(len(s.items) for s in cfg.scenes)


def pvw_layouts(cfg, cells):
    """Every scene's layers in each pvw cell, keyed by scene id: what the follower draws for that
    preview, below base_composition(). Empty layouts when no cell draws the preview."""
    pvw = [c for c in cells if c["role"] == "pvw"]
    return {scene.id: _layout([layer for k, cell in enumerate(pvw)
                               for layer in _cell_layers(cfg, scene, cell, k * _reserved(cfg))])
            for scene in cfg.scenes}


def base_composition(cfg, cells, scenes):
    """What the preview does not change: slot and source cells in list order, then the pgm cells,
    so the program is drawn over any cell it overlaps; z from above the pvw cells' reserve. *scenes*
    is indexed by slot."""
    definitions = {s.id: s for s in cfg.scenes}
    first = _reserved(cfg) * count(cells, "pvw")
    layers = []
    for cell in [c for c in cells if c["role"] in ("slot", "source")] + [c for c in cells if c["role"] == "pgm"]:
        if cell["role"] != "slot":
            layers.append(_tile(len(cfg.sources) if cell["role"] == "pgm" else cell["source"], cell, first + len(layers)))
        elif scenes[cell["slot"]]:
            layers.extend(_cell_layers(cfg, definitions[scenes[cell["slot"]]], cell, first + len(layers)))
    return _layout(layers)


def layer_count(cfg, cells, scenes):
    """Layers of the largest composition the follower sets: the largest scene in every pvw cell, then the base."""
    return _reserved(cfg) * count(cells, "pvw") + len(base_composition(cfg, cells, scenes)["layers"])


def max_layer_count(cfg, cells):
    """layer_count() with the largest scene in every slot."""
    largest = max(cfg.scenes, key=lambda s: len(s.items)).id
    return layer_count(cfg, cells, [largest] * count(cells, "slot"))


def check_assignments(cfg, cells, scenes, max_layers):
    """*scenes*, by slot index, must cover every slot cell, name known scenes, and fit *max_layers*."""
    if not isinstance(scenes, (list, tuple)) or len(scenes) < count(cells, "slot"):
        raise ConfigError(f"the layout needs at least {count(cells, 'slot')} scene assignments (null clears a slot)")
    definitions = {s.id for s in cfg.scenes}
    if any(s is not None and (not isinstance(s, str) or s not in definitions) for s in scenes):
        raise ConfigError("aux assignment references an unknown scene")
    layers = layer_count(cfg, cells, scenes)
    if layers > max_layers:
        raise ConfigError(f"the layout needs {layers} layers including {_reserved(cfg) * count(cells, 'pvw')} "
                          f"reserved for PVW; the bus's max_layers is {max_layers}")
