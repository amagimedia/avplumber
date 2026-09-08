#!/usr/bin/env python3
"""Emit a mixer JSON document with the demo's layouts for the given sources.

    make_config.py --canvas 1080x1920 --fps 60 --wipe /media/wipe.mov \
        cam=/media/cam.mp4 page=https://example.org/page@1920x1080 > mixer.json

Each positional argument is ``id=path`` for a clip or ``id=url@WxH`` for a
browser page. The scenes are the same fullscreen and 2/4/8/16-box pages the
demo builds without a config, written out as plain data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layouts  # noqa: E402


def source_spec(arg: str) -> dict:
    sid, _, rest = arg.partition("=")
    if not sid or not rest:
        raise SystemExit(f"expected id=path or id=url@WxH, got {arg!r}")
    if "://" in rest and "@" in rest.rsplit("/", 1)[-1]:
        url, size = rest.rsplit("@", 1)
        w, h = (int(v) for v in size.lower().split("x"))
        return {"id": sid, "kind": "browser", "url": url, "width": w, "height": h}
    return {"id": sid, "kind": "video", "path": rest}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--canvas", default=f"{layouts.CANVAS_WIDTH}x{layouts.CANVAS_HEIGHT}")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--wipe", action="append", default=[], metavar="PATH")
    parser.add_argument("--fade-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)
    width, height = (int(v) for v in args.canvas.lower().split("x"))
    # One source per unique clip or page: repeated locations become references
    # to the first declaration, so a page shown eight times is one window.
    sources, position_ids, seen = [], [], {}
    for spec in (source_spec(a) for a in args.sources):
        key = (spec["kind"], spec.get("url") or spec.get("path"))
        if key not in seen:
            seen[key] = spec["id"]
            sources.append(spec)
        position_ids.append(seen[key])
    sx, sy = width / layouts.CANVAS_WIDTH, height / layouts.CANVAS_HEIGHT
    scenes = []
    for scene in layouts.all_scenes(len(position_ids)):
        items = [{"source": position_ids[p.source_index],
                  "dst": {"x": round(p.x * sx), "y": round(p.y * sy),
                          "w": round(p.width * sx), "h": round(p.height * sy)}, "fit": "contain"}
                 for p in scene.placements]
        scenes.append({"id": scene.name, "items": items})
    doc = {
        "canvas": {"width": width, "height": height, "fps": args.fps},
        "sources": sources,
        "wipes": [{"id": Path(p).stem, "path": p} for p in args.wipe],
        "control": {"direct": True, "fade_seconds": args.fade_seconds,
                    **({"default_wipe": Path(args.wipe[0]).stem} if args.wipe else {})},
        "scenes": scenes,
        "initial_scene": scenes[0]["id"],
    }
    json.dump(doc, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
