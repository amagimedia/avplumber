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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FITS = ("stretch", "contain", "cover")
TRANSITIONS = ("cut", "fade", "wipe")
DEFAULT_FPS = 30          # canvas.fps when the document does not say
DEFAULT_FADE_SECONDS = 0.5
DEFAULT_TRANSITION = "fade"


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Source:
    id: str
    kind: str                      # "browser" | "video"
    location: str                  # url (browser) or path (video)
    width: int = 0
    height: int = 0
    fps: int = 0                   # browser paint rate; 0 = canvas fps
    loop: bool = True


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
    codec: str = "h264_nvenc"
    profile: str = "baseline"        # WebRTC negotiates constrained baseline
    preset: str = "p7"               # NVENC quality preset
    port: int = 0                    # janus target: 0 keeps the configured port

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

    sources: List[Source] = []
    locations: Dict[Tuple[str, str], str] = {}
    for i, s in enumerate(doc["sources"]):
        where = f"sources[{i}]"
        sid, kind = str(s.get("id", "")), s.get("kind")
        if not sid or "#" in sid:
            raise ConfigError(f"{where}: id required (no '#')")
        if any(x.id == sid for x in sources):
            raise ConfigError(f"{where}: duplicate id '{sid}'")
        if kind == "browser":
            if not all(k in s for k in ("url", "width", "height")):
                raise ConfigError(f"{where}: browser source needs url, width, height")
            src = Source(sid, kind, str(s["url"]), int(s["width"]), int(s["height"]),
                         int(s.get("fps", fps)), bool(s.get("loop", True)))
        elif kind == "video":
            if "path" not in s:
                raise ConfigError(f"{where}: video source needs path")
            src = Source(sid, kind, str(s["path"]), int(s.get("width", 0)), int(s.get("height", 0)),
                         0, bool(s.get("loop", True)))
        else:
            raise ConfigError(f"{where}: kind must be browser or video")
        key = (kind, src.location)
        if key in locations:
            raise ConfigError(f"{where}: '{src.location}' already declared as '{locations[key]}'; "
                              "reference that id instead (one decode per unique source)")
        locations[key] = sid
        sources.append(src)
    if not sources:
        raise ConfigError("sources must not be empty")

    renditions: List[Rendition] = []
    for i, r in enumerate(doc.get("renditions", [])):
        where = f"renditions[{i}]"
        rid = str(r.get("id", ""))
        if not rid:
            raise ConfigError(f"{where}: id required")
        if any(x.id == rid for x in renditions):
            raise ConfigError(f"{where}: duplicate id '{rid}'")
        rendition = Rendition(
            rid, str(r.get("target", "janus")),
            int(r.get("width", canvas_w)), int(r.get("height", canvas_h)),
            int(r.get("fps", fps)), int(r.get("bitrate_kbps", 3000)),
            str(r.get("codec", "h264_nvenc")), str(r.get("profile", "baseline")),
            str(r.get("preset", "p7")), int(r.get("port", 0)))
        if rendition.width <= 0 or rendition.height <= 0:
            raise ConfigError(f"{where}: width and height must be positive")
        if rendition.fps <= 0 or rendition.bitrate_kbps <= 0:
            raise ConfigError(f"{where}: fps and bitrate_kbps must be positive")
        if rendition.fps > fps:
            raise ConfigError(f"{where}: fps {rendition.fps} exceeds the canvas rate {fps}; "
                              "a rendition can only re-time the program downwards")
        wanted = str(r.get("aspect", ""))
        if wanted and wanted != rendition.aspect:
            raise ConfigError(f"{where}: {rendition.width}x{rendition.height} is "
                              f"{rendition.aspect}, not {wanted}")
        renditions.append(rendition)

    wipes: List[Wipe] = []
    for i, w in enumerate(doc.get("wipes", [])):
        if "id" not in w or "path" not in w:
            raise ConfigError(f"wipes[{i}]: id and path required")
        if any(x.id == w["id"] for x in wipes):
            raise ConfigError(f"wipes[{i}]: duplicate id '{w['id']}'")
        wipes.append(Wipe(str(w["id"]), str(w["path"]), float(w.get("duration_seconds", 0)),
                          str(w.get("name", ""))))
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
        if any(x.id == sc["id"] for x in scenes):
            raise ConfigError(f"{where}: duplicate scene id '{sc['id']}'")
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
    if not scenes:
        raise ConfigError("scenes must not be empty")

    initial = str(doc.get("initial_scene", scenes[0].id))
    if not any(s.id == initial for s in scenes):
        raise ConfigError(f"initial_scene '{initial}' is not a scene")
    control = doc.get("control", {})
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
    return MixerConfig(canvas_w, canvas_h, fps, tuple(sources), tuple(scenes), tuple(wipes),
                       tuple(renditions), initial,
                       bool(control.get("direct", True)), fade_seconds, transition, default_wipe)


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
