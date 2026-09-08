"""Mixer configuration file: sources, wipes and scenes as data.

See doc/research/2026-09-08-mixer-config-schema.md. The loader validates the
document and turns scene items into compositor layers. Every source is one
input chain however many scenes reference it; a source used more than once in
the same scene gets alias names (``id#2``, ``id#3``...) that share its frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

FITS = ("stretch", "contain", "cover")


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
class Wipe:
    id: str
    path: str
    duration_seconds: float = 0.0  # 0 = probed by the mixer


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
    initial_scene: str = ""
    direct: bool = True            # TUI: scene picks go straight to program
    fade_seconds: float = 0.5
    default_wipe: str = ""

    def source(self, id: str) -> Source:
        return next(s for s in self.sources if s.id == id)

    def settings(self) -> Dict[str, Any]:
        """What a control surface needs: direct mode, fade length, wipe paths."""
        wipes = {w.id: w.path for w in self.wipes}
        return {"direct": self.direct, "fade_seconds": self.fade_seconds,
                "wipe_file": wipes.get(self.default_wipe, ""), "wipes": wipes}

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
        canvas_w, canvas_h, fps = int(canvas["width"]), int(canvas["height"]), int(canvas["fps"])
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

    wipes: List[Wipe] = []
    for i, w in enumerate(doc.get("wipes", [])):
        if "id" not in w or "path" not in w:
            raise ConfigError(f"wipes[{i}]: id and path required")
        if any(x.id == w["id"] for x in wipes):
            raise ConfigError(f"wipes[{i}]: duplicate id '{w['id']}'")
        wipes.append(Wipe(str(w["id"]), str(w["path"]), float(w.get("duration_seconds", 0))))

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
            src = ids[it["source"]]
            if fit == "cover" and not (src.width and src.height):
                raise ConfigError(f"{iw}: cover needs the source's width and height "
                                  f"(declare them on source '{src.id}')")
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
    return MixerConfig(canvas_w, canvas_h, fps, tuple(sources), tuple(scenes), tuple(wipes), initial,
                       bool(control.get("direct", True)), float(control.get("fade_seconds", 0.5)), default_wipe)


def load(path: str) -> MixerConfig:
    with open(path, "r", encoding="utf-8") as f:
        return parse(json.load(f))


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
            crop = cover_crop(src.width, src.height, item.dst, crop)
            layer["fit"] = "stretch"
        else:
            layer["fit"] = item.fit
        if crop is not None:
            layer["crop"] = {"x": crop.x, "y": crop.y, "w": crop.w, "h": crop.h}
        layers[name] = layer
    return layers
