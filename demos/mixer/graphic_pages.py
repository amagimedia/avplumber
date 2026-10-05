"""The mixer's side of the graphics under graphics/: the page of each, and what its manifest says.

A graphic under graphics/<name>/ is a declaration for the motion engine (graphics/README.md).
Its page is graphics/host.html with the text of graphics/motion.js and of the graphic inlined,
delivered as a data: URL: the browser service fetches nothing, and needs no file mount.
Its EBU OGraf manifest, graphics/<name>/<name>.ograf.json, also says under "v_avplumber" what only
the mixer needs: the browser window of the graphic and, for a downstream key, its place on the canvas.
"""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
import re

GRAPHICS_DIR = Path(__file__).resolve().parent / "graphics"

# The manifest schema, https://ograf.ebu.io/v1/specification/json-schemas/graphics/schema.json:
# the fields it requires, the others it defines, and its rule that any further field starts with v_.
OGRAF_SCHEMA = "https://ograf.ebu.io/v1/specification/json-schemas/graphics/schema.json"
OGRAF_REQUIRED = ("$schema", "id", "name", "main", "supportsRealTime", "supportsNonRealTime")
OGRAF_OPTIONAL = ("version", "description", "author", "customActions", "actionDurations", "stepCount",
                  "schema", "renderRequirements", "thumbnails")
VENDOR = "v_avplumber"
# What a manifest may say under v_avplumber, and inside each of the two.
VENDOR_FIELDS = {"window": ("width", "height"), "key": ("order", "anchor", "above")}
# Where a key sits on the canvas: a corner, or a strip along the top or the bottom edge.
ANCHORS = ("top", "bottom", "top-left", "top-right", "bottom-left", "bottom-right")

# What host.html holds exactly once: the root element, which receives the data-* attributes, and
# one marker per inlined script.
ROOT_TAG = "<html"
MARKERS = {"motion": "/*@motion.js*/", "graphic": "/*@graphic.js*/"}


def _script(path: Path, root: Path) -> str:
    text = path.read_text(encoding="utf-8")
    # The HTML parser, not JavaScript, decides where an inline script ends.
    ends = re.search(r"</script|<!--", text, flags=re.I)
    if ends:
        raise ValueError(f"{path.relative_to(root).as_posix()}: {ends[0]!r} would end the inline script; "
                         "write it as two concatenated strings")
    return text


def _known(fields, known, where: str) -> None:
    for field in fields:
        if field not in known:
            raise ValueError(f'{where}: unknown field "{field}" (known: {", ".join(known)})')


def graphic_url(name: str, fps: float, *, root: Path = GRAPHICS_DIR, **data: object) -> str:
    """The data: URL of graphic *name* for a window that paints at *fps*.

    *fps* becomes data-fps of the page's <html>: the engine counts its whole frames in it. *data*
    become further data-* attributes, the graphic's data (the source id as ``source``, a label, ...).
    The URL depends only on the files and arguments, so an unchanged graphic keeps its window open.
    """
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be a positive number, not {fps!r}")
    for key in data:
        # data-my_key reads back as dataset.my_key; a hyphen or capital would change the name.
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError(f"data name {key!r} must be a lowercase word of letters, digits and _")
    path = root / name / "graphic.js"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not path.is_file():
        raise ValueError(f"no graphic {name!r}: expected {path}")
    shell = (root / "host.html").read_text(encoding="utf-8")
    if any(shell.count(part) != 1 for part in (ROOT_TAG, *MARKERS.values())):
        raise ValueError(f"host.html must hold {ROOT_TAG} and each of {', '.join(MARKERS.values())} exactly once")
    scripts = {MARKERS["motion"]: _script(root / "motion.js", root), MARKERS["graphic"]: _script(path, root)}
    attributes = "".join(f' data-{key}="{html.escape(str(value), quote=True)}"'
                         for key, value in {"fps": fps, **data}.items())
    # One pass over the shell alone: inlined text that happens to hold a marker is left as it is.
    page = re.sub("|".join(map(re.escape, (ROOT_TAG, *scripts))),
                  lambda found: scripts.get(found[0], ROOT_TAG + attributes), shell)
    return "data:text/html;base64," + base64.b64encode(page.encode("utf-8")).decode("ascii")


def manifests(root: Path = GRAPHICS_DIR) -> dict[str, dict]:
    """The OGraf manifest of every graphic under *root*, by name: <name>/<name>.ograf.json beside
    <name>/graphic.js. A manifest that OGraf or the linker would not accept raises ValueError."""
    found: dict[str, dict] = {}
    for script in sorted(root.glob("*/graphic.js")):
        name = script.parent.name
        path, where = script.with_name(f"{name}.ograf.json"), f"{name}/{name}.ograf.json"
        if not path.is_file():
            raise ValueError(f"{name}: no manifest {path.name} beside graphic.js")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from None
        if not isinstance(manifest, dict):
            raise ValueError(f"{where}: must be a JSON object")
        for field in OGRAF_REQUIRED:
            if field not in manifest:
                raise ValueError(f'{where}: needs "{field}"')
        for field in manifest:
            if field not in OGRAF_REQUIRED + OGRAF_OPTIONAL and not field.startswith("v_"):
                raise ValueError(f'{where}: unknown field "{field}"; OGraf allows only its own and v_ fields')
        if manifest["$schema"] != OGRAF_SCHEMA:
            raise ValueError(f'{where}: "$schema" must be {OGRAF_SCHEMA}')
        if manifest["main"] != "graphic.js":
            raise ValueError(f'{where}: "main" must be graphic.js, the file the page is linked from')
        twin = next((other for other, earlier in found.items() if earlier["id"] == manifest["id"]), None)
        if twin:
            raise ValueError(f'{where}: id "{manifest["id"]}" is also the id of {twin}')
        found[name] = manifest
    return found


def key_graphics(root: Path = GRAPHICS_DIR) -> dict[str, dict]:
    """The graphics a show can key over its program: those whose manifest has a v_avplumber "key",
    in the "order" the keys give. Per name: "label" for a menu, "window" as (width, height) of the
    browser window, which is the graphic's own rectangle, and the placement key_rects() reads:
    "anchor" and, where given, "above"."""
    keys: dict[str, dict] = {}
    order: dict[str, float] = {}
    for name, manifest in manifests(root).items():
        where = f"{name}/{name}.ograf.json: {VENDOR}"
        vendor = manifest.get(VENDOR, {})
        _known(vendor, VENDOR_FIELDS, where)
        for field, value in vendor.items():
            if not isinstance(value, dict):
                raise ValueError(f"{where}.{field}: must be an object")
            _known(value, VENDOR_FIELDS[field], f"{where}.{field}")
        if "key" not in vendor:
            continue
        window = tuple(vendor.get("window", {}).get(side) for side in ("width", "height"))
        if not all(isinstance(side, int) and not isinstance(side, bool) and side > 0 for side in window):
            raise ValueError(f'{where}.window: needs whole, positive "width" and "height"; a key\'s window is its own rectangle')
        key = vendor["key"]
        if key.get("anchor") not in ANCHORS:
            raise ValueError(f"{where}.key.anchor: must be one of {', '.join(ANCHORS)}")
        if isinstance(key.get("order"), bool) or not isinstance(key.get("order"), (int, float)):
            raise ValueError(f"{where}.key.order: must be a number; keys are listed from the lowest")
        order[name] = key["order"]
        description = manifest.get("description")
        keys[name] = {"label": f"{manifest['name']} · {description}" if description else manifest["name"],
                      "window": window, "anchor": key["anchor"], **({"above": key["above"]} if "above" in key else {})}
    for name, key in keys.items():
        where, below = f"{name}/{name}.ograf.json: {VENDOR}.key.above", key.get("above")
        if below is not None and not key["anchor"].startswith("bottom"):
            raise ValueError(f"{where}: only a bottom anchor stacks above another graphic")
        while below is not None:
            if below not in keys:
                raise ValueError(f'{where}: no key graphic "{below}"')
            if below == name:
                raise ValueError(f'{where}: "{name}" stacks above itself')
            below = keys[below].get("above")
    return {name: keys[name] for name in sorted(keys, key=lambda name: (order[name], name))}


def key_rects(width: int, height: int, root: Path = GRAPHICS_DIR) -> dict[str, tuple[int, int, int, int]]:
    """Canvas rectangle (x, y, w, h) of every key graphic on a *width* x *height* canvas.

    A window is scaled by the canvas's short side over 1080, so a 1080p canvas shows it 1:1, and
    sits one margin, 3 % of the short side, inside the corner its anchor names. An anchor without
    a side, "top" or "bottom", is a strip: scaled to span the canvas width, with the margin only
    above or below it. "above" puts a graphic one margin above the place of another, whether or
    not the show keys that one. Chromium paints only the graphic, never a transparent canvas.
    """
    keys = key_graphics(root)
    even = lambda value: max(2, round(value / 2) * 2)
    unit = min(width, height)
    margin = even(unit * 0.03)
    rects: dict[str, tuple[int, int, int, int]] = {}

    def place(name: str) -> tuple[int, int, int, int]:
        if name not in rects:
            key = keys[name]
            window_w, window_h = key["window"]
            edge, _, side = key["anchor"].partition("-")
            w, h = ((even(window_w * unit / 1080), even(window_h * unit / 1080)) if side
                    else (width, even(window_h * width / window_w)))
            x = {"": 0, "left": margin, "right": width - margin - w}[side]
            floor = place(key["above"])[1] if "above" in key else height
            rects[name] = (x, margin if edge == "top" else floor - margin - h, w, h)
        return rects[name]

    return {name: place(name) for name in keys}
