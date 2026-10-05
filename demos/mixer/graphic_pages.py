"""Link a graphic into one self-contained page for a browser window.

A graphic under graphics/<name>/ is a declaration for the motion engine (graphics/README.md).
Its page is graphics/host.html with the text of graphics/motion.js and of the graphic inlined,
delivered as a data: URL: the browser service fetches nothing, and needs no file mount.
"""

from __future__ import annotations

import base64
import html
import math
from pathlib import Path
import re

GRAPHICS_DIR = Path(__file__).resolve().parent / "graphics"

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
