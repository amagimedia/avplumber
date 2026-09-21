#!/usr/bin/env python3
"""Emit a mixer JSON document with the demo's layouts for the given sources.

    make_config.py --canvas 1080x1920 --fps 60 --wipe /media/wipe.mov \
        cam=/media/cam.mp4 page=https://example.org/page@1920x1080 > mixer.json

Each positional argument is ``id=path`` for a clip or ``id=url@WxH`` for a
browser page. A ``:sdr``, ``:hlg`` or ``:pq`` suffix declares the clip's color
(``id=/media/assets/clip.mp4:hlg``); raw v210 takes its size too (``id=/media/assets/bars.v210@1920x1080:hlg``).
``--color hlg`` (or ``pq``) with ``--working-format p210le`` makes an HDR show:
the program rendition becomes HEVC Main10 and ``--sdr-port`` adds a tone-mapped
H.264 rendition, so one show feeds an HDR and an SDR mountpoint at once.
The scenes are the same fullscreen and 2/4/8/16-box pages the demo builds
without a config, written out as plain data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import layouts  # noqa: E402
from pyplumber.mixer import config as mixer_config  # noqa: E402
from pyplumber.mixer.color import TEN_BIT_FORMATS  # noqa: E402


COLORS = ("sdr", "hlg", "pq")


def scene_documents(source_ids, width, height):
    """Standard layouts scaled to the chosen canvas, with explicit source IDs."""
    sx, sy = width / layouts.CANVAS_WIDTH, height / layouts.CANVAS_HEIGHT
    return [{"id": scene.name, "items": [
        {"source": source_ids[p.source_index],
         "dst": {"x": round(p.x * sx), "y": round(p.y * sy),
                 "w": round(p.width * sx), "h": round(p.height * sy)}, "fit": "contain"}
        for p in scene.placements]} for scene in layouts.all_scenes(len(source_ids))]


def source_spec(arg: str) -> dict:
    sid, _, rest = arg.partition("=")
    if not sid or not rest:
        raise SystemExit(f"expected id=path[:color] or id=url@WxH, got {arg!r}")
    if "://" in rest and "@" in rest.rsplit("/", 1)[-1]:
        url, size = rest.rsplit("@", 1)
        w, h = (int(v) for v in size.lower().split("x"))
        return {"id": sid, "kind": "browser", "url": url, "width": w, "height": h, "color": "sdr"}
    color = None
    if rest.rsplit(":", 1)[-1] in COLORS:
        rest, color = rest.rsplit(":", 1)
    spec = {"id": sid, "kind": "video", "path": rest}
    if "@" in rest.rsplit("/", 1)[-1]:
        path, size = rest.rsplit("@", 1)
        if not path.lower().endswith(".v210"):
            raise SystemExit(f"only .v210 clips take a size, got {arg!r}")
        if color is None:
            raise SystemExit(f"v210 clips need a :sdr/:hlg/:pq color, got {arg!r}")
        w, h = (int(v) for v in size.lower().split("x"))
        spec = {"id": sid, "kind": "v210", "path": path, "width": w, "height": h}
    if color:
        spec["color"] = color
    return spec


def generate(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--canvas", default=f"{layouts.CANVAS_WIDTH}x{layouts.CANVAS_HEIGHT}")
    parser.add_argument("--fps", type=int, default=mixer_config.DEFAULT_FPS,
                        help="rate the compositor renders at (default: %(default)s)")
    parser.add_argument("--wipe", action="append", default=[], metavar="PATH")
    parser.add_argument("--fade-seconds", type=float, default=mixer_config.DEFAULT_FADE_SECONDS)
    parser.add_argument("--transition", default=mixer_config.DEFAULT_TRANSITION,
                        choices=mixer_config.TRANSITIONS,
                        help="what a direct-mode scene pick takes with (default: %(default)s)")
    parser.add_argument("--bitrate-kbps", type=int, default=2700,
                        help="program rendition bitrate (default: %(default)s)")
    parser.add_argument("--rendition-fps", type=int, default=0, metavar="FPS",
                        help="program rendition rate; 0 keeps the canvas rate")
    parser.add_argument("--profile", default="", help="NVENC profile of the program rendition (default: from depth)")
    parser.add_argument("--program-port", type=int, default=0,
                        help="program RTP port; 0 uses the mixer's --janus-video-port")
    parser.add_argument("--preset", default="p7", help="NVENC preset of the program rendition")
    parser.add_argument("--color", choices=COLORS, help="canvas color; hlg/pq need a 10-bit --working-format")
    parser.add_argument("--working-format", choices=mixer_config.WORKING_FORMATS, help="canvas storage")
    parser.add_argument("--sdr-port", type=int, metavar="PORT",
                        help="add a tone-mapped H.264 rendition on this RTP port (HDR canvas only)")
    parser.add_argument("--sdr-tonemap", default="mobius", choices=mixer_config.OPERATORS)
    parser.add_argument("--sdr-knee", type=float, default=0.9, help="tonemap_param of the SDR rendition")
    args = parser.parse_args(argv)
    hdr = args.color in ("hlg", "pq")
    if args.sdr_port and not hdr:
        raise SystemExit("--sdr-port needs an HDR canvas (--color hlg or pq)")
    if args.sdr_port and abs(args.sdr_port - (args.program_port or 5004)) < 2:
        raise SystemExit("HDR and SDR need distinct RTP ports and adjacent RTCP ports; pass --program-port, e.g. 5006")
    for port in (args.program_port, args.sdr_port):
        if port is not None and port != 0 and not 1 <= port < 65535:
            raise SystemExit("RTP ports must be between 1 and 65534 (RTCP uses the next port)")
    ten_bit = args.working_format in TEN_BIT_FORMATS
    width, height = (int(v) for v in args.canvas.lower().split("x"))
    # One source per unique clip or page: repeated locations become references
    # to the first declaration, so a page shown eight times is one window.
    sources, argument_ids, seen = [], [], {}
    for spec in (source_spec(a) for a in args.sources):
        key = (spec["kind"], spec.get("url") or spec.get("path"))
        if key not in seen:
            seen[key] = spec["id"]
            sources.append(spec)
        argument_ids.append(seen[key])
    # Grid slots take every distinct source before any repeat, so a 16-box of
    # 16 unique sources shows all sixteen instead of one of them three times.
    unique_ids = [s["id"] for s in sources]
    repeats = list(argument_ids)
    for sid in unique_ids:
        repeats.remove(sid)
    position_ids = unique_ids + repeats
    scenes = scene_documents(position_ids, width, height)
    rendition_fps = args.rendition_fps or args.fps
    program = {"id": "program", "target": "janus", "width": width, "height": height,
               "aspect": mixer_config.Rendition("program", width=width, height=height).aspect,
               "fps": rendition_fps, "bitrate_kbps": args.bitrate_kbps,
               "profile": args.profile or ("main10" if ten_bit else "baseline"), "preset": args.preset,
               **({"codec": "hevc_nvenc"} if ten_bit else {}),
               **({"port": args.program_port} if args.program_port else {})}
    renditions = [program]
    if args.sdr_port:
        renditions.append({**program, "id": "sdr", "port": args.sdr_port, "codec": "h264_nvenc",
                           "profile": "baseline", "tonemap": args.sdr_tonemap, "tonemap_param": args.sdr_knee})
    canvas = {"width": width, "height": height, "fps": args.fps,
              **({"working_format": args.working_format} if args.working_format else {}),
              **({"color": args.color} if args.color else {})}
    doc = {
        "canvas": canvas,
        "renditions": renditions,
        "sources": sources,
        "wipes": [{"id": Path(p).stem, "path": p} for p in args.wipe],
        "control": {"direct": True, "fade_seconds": args.fade_seconds,
                    "transition": args.transition,
                    **({"default_wipe": Path(args.wipe[0]).stem} if args.wipe else {})},
        "scenes": scenes,
        "initial_scene": scenes[0]["id"],
    }
    return doc


def main(argv=None) -> None:
    json.dump(generate(argv), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
