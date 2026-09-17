"""Mixer configuration file: sources, wipes and scenes as data.

See doc/research/2026-09-08-mixer-config-schema.md. The loader validates the
document and turns scene items into compositor layers. Every source is one
input chain however many scenes reference it; a source used more than once in
the same scene gets alias names (``id#2``, ``id#3``...) that share its frames.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .color import Color, declared_color, OPERATORS, TRANSFER_TAGS, YUV_FORMATS

FITS = ("stretch", "contain", "cover")
TRANSITIONS = ("cut", "fade", "wipe")
# Compositor/transition working formats: the semiplanar family only. NV12 is the
# 8-bit default; P010/P210 are 10-bit 4:2:0/4:2:2. Planar layouts are excluded on
# purpose: the compositor cannot promote 8-bit sources or draw the RGBA wipe
# onto them, so they only fail later.
WORKING_FORMATS = ("nv12", "p010le", "p210le")
DEFAULT_FPS = 30          # canvas.fps when the document does not say
DEFAULT_FADE_SECONDS = 0.5
DEFAULT_TRANSITION = "cut"


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Source:
    id: str
    kind: str                      # "browser" | "video" | "v210"
    location: str                  # url (browser) or path (video/v210 raw file)
    width: int = 0
    height: int = 0
    fps: int = 0                   # browser paint rate; 0 = canvas fps
    loop: bool = True
    # Empty fields require complete decoded frame metadata. Raw inputs must
    # declare a contract because their bytes carry no color metadata.
    color_trc: str = ""
    color_primaries: str = ""
    colorspace: str = ""
    color_range: str = ""
    filter_graph: str = ""         # optional CUDA source filter, before scene/alias fan-out
    filter_output_format: str = ""

    @property
    def color(self):
        tags = {k: getattr(self, k) for k in ("color_trc", "color_primaries", "colorspace", "color_range")}
        return Color.parse(tags) if any(tags.values()) else None


@dataclass(frozen=True)
class Rendition:
    """One encoded output. The compositor renders once, at the canvas rate;
    each rendition re-times and rescales that program for its own target."""
    id: str
    target: str = "janus"            # "janus" or a file path
    width: int = 0                   # 0 keeps the canvas size
    height: int = 0
    fps: int = 0                     # 0 keeps the canvas rate
    bitrate_kbps: int = 3000
    codec: str = ""                  # "" auto-selects: HEVC for a 10-bit program, else H.264
    profile: str = ""                # "" lets the encoder pick main/main10/baseline
    preset: str = "p7"               # NVENC quality preset
    port: int = 0                    # janus target: 0 keeps the configured port
    # An explicit operator requests SDR. Otherwise codec/color select the target
    # and automatic HDR-to-SDR conversion uses clip to preserve SDR reference white.
    tonemap: str = ""
    tonemap_peak: float = 10.0       # source peak in REFERENCE_WHITE units (HLG 1000 nits)
    tonemap_desat: float = 0.0       # 0 keeps saturation; FFmpeg's 0.5 default washes colours out
    tonemap_param: float = 0.0       # operator knee in reference-white units; 1.0 = SDR range untouched
    # HDR10 static metadata for PQ outputs (HLG needs none). 0 derives MaxCLL from
    # tonemap_peak and MaxFALL as 40% of it, the usual 1000/400 pair.
    max_cll: int = 0
    max_fall: int = 0

    color: str = ""                # empty: inherit canvas, except H.264/tonemap imply SDR

    @property
    def aspect(self) -> str:
        from math import gcd
        if not (self.width and self.height):
            return ""
        divisor = gcd(self.width, self.height)
        return f"{self.width // divisor}:{self.height // divisor}"


@dataclass(frozen=True)
class Wipe:
    id: str
    path: str
    duration_seconds: float = 0.0  # 0 = probed by the mixer
    name: str = ""                 # label for the control surfaces

    @property
    def label(self) -> str:
        return self.name or self.id


@dataclass(frozen=True)
class Item:
    source: str
    dst: Rect
    fit: str = "contain"
    crop: Optional[Rect] = None


@dataclass(frozen=True)
class Scene:
    id: str
    items: Tuple[Item, ...]


@dataclass(frozen=True)
class MixerConfig:
    canvas_w: int
    canvas_h: int
    fps: int
    sources: Tuple[Source, ...]
    scenes: Tuple[Scene, ...]
    wipes: Tuple[Wipe, ...] = ()
    renditions: Tuple[Rendition, ...] = ()
    initial_scene: str = ""
    direct: bool = True            # control surfaces: scene picks go straight to program
    fade_seconds: float = DEFAULT_FADE_SECONDS
    transition: str = DEFAULT_TRANSITION   # what a pick takes with in direct mode
    default_wipe: str = ""
    working_format: str = "nv12"   # canvas.working_format: compositor/transition sw_format
    out_color: Color = Color()     # canvas color contract; renditions convert from it and signal it (VUI)
    wipe_color: str = ""          # optional explicit override for all alpha wipe clips

    def source(self, id: str) -> Source:
        return next(s for s in self.sources if s.id == id)

    def settings(self) -> Dict[str, Any]:
        """What a control surface needs: direct mode, fade length, wipe library."""
        wipes = [{"id": w.id, "name": w.label, "path": w.path,
                  "duration_seconds": w.duration_seconds} for w in self.wipes]
        default = next((w for w in self.wipes if w.id == self.default_wipe), None)
        return {"direct": self.direct, "fade_seconds": self.fade_seconds,
                "transition": self.transition,
                "wipe_file": default.path if default else "", "default_wipe": self.default_wipe,
                "wipes": wipes}

    @property
    def alias_counts(self) -> Dict[str, int]:
        """Highest number of times each source appears within one scene."""
        counts = {s.id: 1 for s in self.sources}
        for scene in self.scenes:
            seen: Dict[str, int] = {}
            for item in scene.items:
                seen[item.source] = seen.get(item.source, 0) + 1
            for source, n in seen.items():
                counts[source] = max(counts[source], n)
        return counts


def alias_name(source_id: str, occurrence: int) -> str:
    return source_id if occurrence == 1 else f"{source_id}#{occurrence}"


class ConfigError(ValueError):
    pass


def _rect(obj: Any, where: str) -> Rect:
    if not isinstance(obj, dict) or any(k not in obj for k in ("x", "y", "w", "h")):
        raise ConfigError(f"{where}: rect needs x, y, w, h")
    r = Rect(int(obj["x"]), int(obj["y"]), int(obj["w"]), int(obj["h"]))
    if r.w <= 0 or r.h <= 0:
        raise ConfigError(f"{where}: rect needs positive w and h")
    return r


_SOURCE_KEYS = {"browser": ("url", "width", "height"), "video": ("path",), "v210": ("path", "width", "height")}


def _parse_source(s: Dict[str, Any], where: str, fps: int) -> Source:
    sid, kind = str(s.get("id", "")), s.get("kind")
    if not sid or "#" in sid:
        raise ConfigError(f"{where}: id required (no '#')")
    if kind not in _SOURCE_KEYS:
        raise ConfigError(f"{where}: kind must be browser, video or v210")
    if not all(k in s for k in _SOURCE_KEYS[kind]):
        raise ConfigError(f"{where}: {kind} source needs {', '.join(_SOURCE_KEYS[kind])}")
    source_filter = s.get("filter", "")
    if not isinstance(source_filter, str):
        raise ConfigError(f"{where}: filter must be a CUDA filter graph string")
    filter_format = str(s.get("filter_output_format", ""))
    if filter_format and filter_format not in YUV_FORMATS or source_filter and not filter_format:
        raise ConfigError(f"{where}: custom filter requires filter_output_format (CUDA YUV storage)")
    if source_filter and kind == "browser":
        raise ConfigError(f"{where}: browser source filters are unsupported; preserve packed RGB alpha")
    try:
        color = declared_color(s)
        if kind != "video" and color is None:
            raise ValueError("raw input has no color metadata; declare an explicit color setting")
        if kind == "browser" and color != Color():
            raise ValueError("browser input supports SDR only")
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from e
    return Source(sid, kind, str(s.get("url", s.get("path"))), width=int(s.get("width", 0)),
                  height=int(s.get("height", 0)), fps=int(s.get("fps", fps)) if kind == "browser" else 0,
                  loop=bool(s.get("loop", True)), filter_graph=source_filter,
                  filter_output_format=filter_format, **(color.tags if color else {}))


def _parse_rendition(r: Dict[str, Any], where: str, canvas_w: int, canvas_h: int, fps: int) -> Rendition:
    rid = str(r.get("id", ""))
    if not rid:
        raise ConfigError(f"{where}: id required")
    base = Rendition(rid, width=canvas_w, height=canvas_h, fps=fps)
    # Every other key coerces to its field's type; unknown keys are ignored.
    rendition = replace(base, **{f.name: type(getattr(base, f.name))(r[f.name])
                                 for f in fields(Rendition) if f.name in r and f.name != "id"})
    if rendition.tonemap and rendition.tonemap not in OPERATORS:
        raise ConfigError(f"{where}: unsupported tone-map operator")
    if rendition.color:
        try:
            Color.parse(rendition.color)
        except ValueError as e:
            raise ConfigError(f"{where}: {e}") from e
    if rendition.width <= 0 or rendition.height <= 0:
        raise ConfigError(f"{where}: width and height must be positive")
    if rendition.fps <= 0 or rendition.bitrate_kbps <= 0:
        raise ConfigError(f"{where}: fps and bitrate_kbps must be positive")
    if rendition.tonemap_peak < 2.03 or rendition.tonemap_desat < 0 or rendition.tonemap_param < 0:
        raise ConfigError(f"{where}: tonemap_peak >= 2.03 (203 nits), tonemap_desat >= 0 and tonemap_param >= 0")
    if rendition.max_cll < 0 or rendition.max_fall < 0 or rendition.max_fall > max(rendition.max_cll, rendition.tonemap_peak * 100):
        raise ConfigError(f"{where}: max_cll and max_fall must be non-negative nits, max_fall no higher than MaxCLL")
    if rendition.tonemap == "mobius" and rendition.tonemap_param >= 1:
        # The Möbius shoulder maps [knee, peak] onto [knee, 1]; at knee 1.0 it degenerates to clip.
        raise ConfigError(f"{where}: mobius tonemap_param must be below 1.0 (0.9 keeps 90% of SDR white linear)")
    if rendition.fps > fps:
        raise ConfigError(f"{where}: fps {rendition.fps} exceeds the canvas rate {fps}; "
                          "a rendition can only re-time the program downwards")
    wanted = str(r.get("aspect", ""))
    if wanted and wanted != rendition.aspect:
        raise ConfigError(f"{where}: {rendition.width}x{rendition.height} is "
                          f"{rendition.aspect}, not {wanted}")
    return rendition


def _parse_control(control: Any, wipes: List[Wipe]) -> Dict[str, Any]:
    if not isinstance(control, dict):
        raise ConfigError("control must be an object")
    default_wipe = str(control.get("default_wipe", wipes[0].id if wipes else ""))
    if default_wipe and not any(w.id == default_wipe for w in wipes):
        raise ConfigError(f"control.default_wipe '{default_wipe}' is not a wipe")
    transition = str(control.get("transition", DEFAULT_TRANSITION))
    if transition not in TRANSITIONS:
        raise ConfigError(f"control.transition must be one of {TRANSITIONS}")
    if transition == "wipe" and not wipes:
        raise ConfigError("control.transition 'wipe' needs a wipe library")
    fade_seconds = float(control.get("fade_seconds", DEFAULT_FADE_SECONDS))
    if fade_seconds <= 0:
        raise ConfigError("control.fade_seconds must be positive")
    return {"direct": bool(control.get("direct", True)), "fade_seconds": fade_seconds,
            "transition": transition, "default_wipe": default_wipe}


def _unique(items: List[Any], where: str, label: str = "id") -> None:
    if any(x.id == items[-1].id for x in items[:-1]):
        raise ConfigError(f"{where}: duplicate {label} '{items[-1].id}'")


def parse(doc: Dict[str, Any]) -> MixerConfig:
    for key in ("canvas", "sources", "scenes"):
        if key not in doc:
            raise ConfigError(f"missing '{key}'")
    canvas = doc["canvas"]
    try:
        # fps is how often the compositor renders; renditions re-time from it.
        canvas_w, canvas_h = int(canvas["width"]), int(canvas["height"])
        fps = int(canvas.get("fps", DEFAULT_FPS))
    except (KeyError, TypeError, ValueError):
        raise ConfigError("canvas needs integer width, height and fps") from None
    if canvas_w <= 0 or canvas_h <= 0 or fps <= 0:
        raise ConfigError("canvas width, height and fps must be positive")
    working_format = str(canvas.get("working_format", "nv12"))
    if working_format not in WORKING_FORMATS:
        raise ConfigError(f"canvas.working_format must be one of {WORKING_FORMATS}")
    try:
        out_color = declared_color(canvas) or Color()
        out_color.validate_format(working_format)
    except ValueError as e:
        raise ConfigError(f"canvas: {e}") from e

    sources: List[Source] = []
    locations: Dict[Tuple[str, str], str] = {}
    for i, s in enumerate(doc["sources"]):
        where = f"sources[{i}]"
        sources.append(_parse_source(s, where, fps))
        _unique(sources, where)
        key = (sources[-1].kind, sources[-1].location)
        if key in locations:
            raise ConfigError(f"{where}: '{key[1]}' already declared as '{locations[key]}'; "
                              "reference that id instead (one decode per unique source)")
        locations[key] = sources[-1].id
    if not sources:
        raise ConfigError("sources must not be empty")

    renditions: List[Rendition] = []
    for i, r in enumerate(doc.get("renditions", [])):
        renditions.append(_parse_rendition(r, f"renditions[{i}]", canvas_w, canvas_h, fps))
        _unique(renditions, f"renditions[{i}]")

    wipes: List[Wipe] = []
    for i, w in enumerate(doc.get("wipes", [])):
        if "id" not in w or "path" not in w:
            raise ConfigError(f"wipes[{i}]: id and path required")
        wipes.append(Wipe(str(w["id"]), str(w["path"]), float(w.get("duration_seconds", 0)),
                          str(w.get("name", ""))))
        _unique(wipes, f"wipes[{i}]")
    if "wipe_dir" in doc:
        # Every clip in the directory joins the library under its file name.
        # Entries declared above keep their id, name and duration.
        wipes.extend(scan_wipe_dir(str(doc["wipe_dir"]), taken={w.path for w in wipes},
                                   ids={w.id for w in wipes}))

    ids = {s.id: s for s in sources}
    scenes: List[Scene] = []
    for i, sc in enumerate(doc["scenes"]):
        where = f"scenes[{i}]"
        if "id" not in sc or not isinstance(sc.get("items"), list) or not sc["items"]:
            raise ConfigError(f"{where}: id and a non-empty items list required")
        items: List[Item] = []
        for j, it in enumerate(sc["items"]):
            iw = f"{where}.items[{j}]"
            if it.get("source") not in ids:
                raise ConfigError(f"{iw}: unknown source '{it.get('source')}'")
            fit = it.get("fit", "contain")
            if fit not in FITS:
                raise ConfigError(f"{iw}: fit must be one of {FITS}")
            crop = _rect(it["crop"], iw + ".crop") if "crop" in it else None
            items.append(Item(str(it["source"]), _rect(it["dst"], iw + ".dst"), fit, crop))
        scenes.append(Scene(str(sc["id"]), tuple(items)))
        _unique(scenes, where, "scene id")
    if not scenes:
        raise ConfigError("scenes must not be empty")

    initial = str(doc.get("initial_scene", scenes[0].id))
    if not any(s.id == initial for s in scenes):
        raise ConfigError(f"initial_scene '{initial}' is not a scene")
    wipe_color = str(doc.get("wipe_color", ""))
    if wipe_color and wipe_color not in TRANSFER_TAGS:
        raise ConfigError("wipe_color must be sdr, hlg or pq")
    return MixerConfig(canvas_w, canvas_h, fps, tuple(sources), tuple(scenes), tuple(wipes), tuple(renditions),
                       initial_scene=initial, working_format=working_format, out_color=out_color,
                       wipe_color=wipe_color, **_parse_control(doc.get("control", {}), wipes))


WIPE_SUFFIXES = (".mov", ".webm", ".mkv", ".mp4", ".avi", ".png", ".gif")


def scan_wipe_dir(directory: str, taken=frozenset(), ids=frozenset()) -> List[Wipe]:
    """Every clip in *directory*, sorted, as a wipe named after its file."""
    root = Path(directory)
    if not root.is_dir():
        raise ConfigError(f"wipe_dir '{directory}' is not a directory")
    found: List[Wipe] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in WIPE_SUFFIXES:
            continue
        if str(entry) in taken or entry.stem in ids:
            continue   # already declared explicitly, with its own id and duration
        found.append(Wipe(entry.stem, str(entry)))
    return found


def load(path: str) -> MixerConfig:
    with open(path, "r", encoding="utf-8") as f:
        return parse(json.load(f))


def probe_video_size(path: str) -> Tuple[int, int]:
    """Width and height of the first video stream, via ffprobe or, failing that, ffmpeg -i."""
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    text = ""
    if ffprobe:
        out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=30)
        text = out.stdout.strip().replace(",", "x")
    elif ffmpeg:
        out = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True, timeout=30)
        match = re.search(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b", out.stderr)
        text = f"{match.group(1)}x{match.group(2)}" if match else ""
    else:
        raise ConfigError(f"neither ffprobe nor ffmpeg found; declare width and height for '{path}'")
    match = re.match(r"^(\d+)x(\d+)", text)
    if not match:
        raise ConfigError(f"cannot probe the size of '{path}'")
    return int(match.group(1)), int(match.group(2))


def with_probed_sizes(cfg: MixerConfig, probe=probe_video_size) -> MixerConfig:
    """Fill in video sizes the document did not declare; cover needs them."""
    sources = tuple(replace(s, **dict(zip(("width", "height"), probe(s.location))))
                    if s.kind == "video" and not (s.width and s.height) else s for s in cfg.sources)
    return replace(cfg, sources=sources)


def cover_crop(src_w: int, src_h: int, dst: Rect, crop: Optional[Rect]) -> Rect:
    """The centred region of the (cropped) source with the box's aspect."""
    base = crop or Rect(0, 0, src_w, src_h)
    if base.w * dst.h > base.h * dst.w:           # source wider than the box: trim the sides
        w = max(2, (base.h * dst.w // dst.h) & ~1)
        return Rect(base.x + (base.w - w) // 2, base.y, w, base.h)
    h = max(2, (base.w * dst.h // dst.w) & ~1)    # taller: trim top and bottom
    return Rect(base.x, base.y + (base.h - h) // 2, base.w, h)


def scene_layers(cfg: MixerConfig, scene: Scene) -> Dict[str, Dict[str, Any]]:
    """Compositor layers keyed by mixer source name; item order is z-order."""
    layers: Dict[str, Dict[str, Any]] = {}
    seen: Dict[str, int] = {}
    for z, item in enumerate(scene.items):
        seen[item.source] = seen.get(item.source, 0) + 1
        name = alias_name(item.source, seen[item.source])
        layer: Dict[str, Any] = {"dst_x": item.dst.x, "dst_y": item.dst.y,
                                 "dst_w": item.dst.w, "dst_h": item.dst.h, "z": z}
        crop = item.crop
        if item.fit == "cover":
            src = cfg.source(item.source)
            if not (src.width and src.height):
                raise ConfigError(f"scene '{scene.id}': cover of '{src.id}' needs its size (see with_probed_sizes)")
            crop = cover_crop(src.width, src.height, item.dst, crop)
            layer["fit"] = "stretch"
        else:
            layer["fit"] = item.fit
        if crop is not None:
            layer["crop"] = {"x": crop.x, "y": crop.y, "w": crop.w, "h": crop.h}
        layers[name] = layer
    return layers
