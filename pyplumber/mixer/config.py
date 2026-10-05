"""Mixer configuration file: sources, wipes and scenes as data.

See doc/research/2026-09-08-mixer-config-schema.md. The loader validates the
document and turns scene items into compositor layers. Every source is one
input chain however many scenes reference it; a source used more than once in
the same scene gets alias names (``id#2``, ``id#3``...) that share its frames.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .color import Color, declared_color, default_codec, rendition_color, OPERATORS, TRANSFER_TAGS, YUV_FORMATS

FITS = ("stretch", "contain", "cover")
TRANSITIONS = ("cut", "fade", "wipe")
# Compositor/transition working formats: the semiplanar family only. NV12 is the
# 8-bit default; P010/P210 are 10-bit 4:2:0/4:2:2. Planar layouts are excluded on
# purpose: the compositor cannot promote 8-bit sources or draw the RGBA wipe
# onto them, so they only fail later.
WORKING_FORMATS = ("nv12", "p010le", "p210le")
# How nv12/p010 sources reach the GPU: FFmpeg hwupload after pacing (the library default), or
# raw_to_cuda's pinned staging on a private stream (opt-in; the setup recipe selects it. 60 fps
# A/B in demos/mixer/docs/cookbook/raw-uploads.html; the 110-input A/B is pending).
RAW_UPLOADS = ("hwupload", "pinned")
DEFAULT_FPS = 30          # canvas.fps when the document does not say
MAX_SOURCES = 193         # mixer_compositor active_inputs is a 193-bit pad mask (SourceMask, kSourceMaskBits)
DEFAULT_MAX_COMPOSITOR_LAYERS = 256
DEFAULT_FADE_SECONDS = 0.5
DEFAULT_TRANSITION = "cut"
FEEDS = ("dirty", "clean")
MAX_DSK_KEYS = 4          # the downstream keyer stays one small pass: program + up to four keys
# Fade easing presets, the names src/mixer/primitives/fade_curve.hpp parses; linear is what fades
# did before curves existed.
FADE_CURVES = ("linear", "ease-in", "ease-out", "ease-in-out")
DEFAULT_FADE_CURVE = "linear"
MAX_KEY_FADE_SECONDS = 10.0   # mixer_keyer rejects fade_inputs duration_ms above 10000
MAX_DPB_SIZE = 16             # H.264/HEVC allow at most 16 reference frames


def fade_curve(value: Any, where: str) -> str:
    """A fade curve preset name; *where* names the field or command in the error."""
    if not isinstance(value, str) or value not in FADE_CURVES:
        raise ConfigError(f"{where} must be one of {', '.join(FADE_CURVES)}")
    return value


def fade_color(value: Any, where: str) -> Optional[str]:
    """A dip colour "#RRGGBB" (lower-cased), or None for a plain mix; *where* names the field or command."""
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ConfigError(f"{where} must be a #RRGGBB colour or null")
    return value.lower()


DEFAULT_DSK_FADE_SECONDS = 0.4   # keys fade in and out unless a command or the show says 0 (cut)


def key_fade_seconds(value: Any, where: str) -> float:
    """A downstream key fade length; 0 is a cut."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= MAX_KEY_FADE_SECONDS:
        raise ConfigError(f"{where} must be a number of seconds from 0 to {MAX_KEY_FADE_SECONDS:g}")
    return float(value)


def default_latency_ms(fps):
    """Playout deadline: 2 output frames up to 30 fps (80 ms at 25, 67 at 29.97/30), 3 at 50/60
    (60/50 ms). Two 60 fps frames left 17 ms of slack for a late source or pass and missed
    deadlines under load; two frames at 25/30 fps held the measured shows cleanly."""
    return (2 if fps < 40 else 3) * 1000 / fps


def default_browser_ring_size(fps):
    return 6 if fps in (25, 30) else 9


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Source:
    id: str
    kind: str                      # "browser" | "video" | "v210" | "nv12" | "p010"
    location: str                  # browser URL or media path (v210/nv12/p010 are raw files)
    width: int = 0
    height: int = 0
    fps: int = 0                   # browser paint rate; 0 = canvas fps
    loop: bool = True
    hold_last_frame: bool = True   # browser: repeat the last frame while a failed page reloads
    # Empty fields require complete decoded frame metadata. Raw inputs must
    # declare a contract because their bytes carry no color metadata.
    color_trc: str = ""
    color_primaries: str = ""
    colorspace: str = ""
    color_range: str = ""
    filter_graph: str = ""         # optional CUDA source filter, before scene/alias fan-out
    filter_output_format: str = ""
    decode_storage: str = "cuda"
    extra_hw_frames: int = 3       # CUarray application headroom, beyond codec/working surfaces

    @property
    def decoder_params(self):
        if self.decode_storage != "cuarray":
            return {}
        return {"pixel_format": "cuarray", "auto_restart": "off", "options": {
            "threads": 1, "hwaccel_flags": "unsafe_output", "extra_hw_frames": self.extra_hw_frames}}

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
    tonemap_desat: float = 0.0       # 0 keeps saturation; FFmpeg's 0.5 default washes colors out
    tonemap_param: float = 0.0       # operator knee in reference-white units; 1.0 = SDR range untouched
    # HDR10 static metadata for PQ outputs (HLG needs none). 0 derives MaxCLL from
    # tonemap_peak and MaxFALL as 40% of it, the usual 1000/400 pair.
    max_cll: int = 0
    max_fall: int = 0

    color: str = ""                # empty: inherit canvas, except H.264/tonemap imply SDR
    feed: str = "dirty"            # "clean" omits the downstream keys; without keys both are the program
    dpb_size: int = 0              # NVENC reference frames kept (dpb_size); 0 lets NVENC choose

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
    blend: bool = False           # honour source alpha, e.g. transparent browser graphics


@dataclass(frozen=True)
class Scene:
    id: str
    items: Tuple[Item, ...]


@dataclass(frozen=True)
class DskKey:
    """One downstream key: an alpha browser source drawn over the finished program."""
    id: str
    source: str
    dst: Rect
    on: bool = False


@dataclass(frozen=True)
class AuxBusConfig:
    """One AUX bus; layouts are normalized aux_layout specs."""
    id: str
    scenes: Tuple[Optional[str], ...]    # slot assignments by slot index, at least one per slot cell
    renditions: Tuple[Rendition, ...]
    layout: Dict[str, Any]               # the initial layout
    layouts: Tuple[Dict[str, Any], ...]  # the operator's menu, the initial one first; a pgm cell adds the PGM pad
    max_layers: int                      # the compositor's layer budget, fixed at build
    # when a pvw cell changes on a take: "program", with the program output, or "pgm_tile", with
    # the pgm cell of the same frame (one pgm_delay_frames later).
    pvw_align: str = "program"
    latency_ms: Optional[float] = None   # the bus compositor's playout buffer; None: the main mixer's
    # aux ticks the PGM pad is matched back; parse_aux_buses defaults it to default_pgm_delay_frames
    # (1, or 2 at a 50/60 fps bus).
    pgm_delay_frames: int = 1
    full_rate: bool = False              # run at the canvas rate at 50/60 fps instead of half
    label: str = ""                      # the control page's name for it; "" names it by its preset or id


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
    fade_curve: str = DEFAULT_FADE_CURVE   # what an M/E fade eases with unless a take picks its own
    fade_color: Optional[str] = None       # "#rrggbb": what an M/E fade dips through unless a take picks; None mixes
    transition: str = DEFAULT_TRANSITION   # what a pick takes with in direct mode
    default_wipe: str = ""
    working_format: str = "nv12"   # canvas.working_format: compositor/transition sw_format
    raw_upload: str = "hwupload"   # canvas.raw_upload: one of RAW_UPLOADS
    latency_ms: Optional[float] = None   # canvas.latency_ms: playout buffer, default default_latency_ms(fps)
    out_color: Color = Color()     # canvas color contract; renditions convert from it and signal it (VUI)
    wipe_color: str = ""          # optional explicit override for all alpha wipe clips
    aux_buses: Tuple[AuxBusConfig, ...] = ()
    dsk_keys: Tuple[DskKey, ...] = ()
    dsk_fade_seconds: float = DEFAULT_DSK_FADE_SECONDS   # what a key change fades over unless the command says; 0 cuts
    dsk_fade_curve: str = DEFAULT_FADE_CURVE
    browser_ring_size: Optional[int] = None
    max_compositor_layers: int = DEFAULT_MAX_COMPOSITOR_LAYERS

    def __post_init__(self):
        if type(self.max_compositor_layers) is not int or not 1 <= self.max_compositor_layers <= 2_147_483_647:
            raise ConfigError("max_compositor_layers must be a positive 32-bit integer")
        if self.browser_ring_size is None:
            object.__setattr__(self, "browser_ring_size", default_browser_ring_size(self.fps))
        if type(self.browser_ring_size) is not int or not 1 <= self.browser_ring_size <= 64:
            raise ConfigError("browser_ring_size must be an integer from 1 to 64")

    def source(self, id: str) -> Source:
        return next(s for s in self.sources if s.id == id)

    def settings(self) -> Dict[str, Any]:
        """Canvas format, source counts, operator settings and preview outputs for control surfaces."""
        wipes = [{"id": w.id, "name": w.label, "path": w.path,
                  "duration_seconds": w.duration_seconds} for w in self.wipes]
        default = next((w for w in self.wipes if w.id == self.default_wipe), None)

        def codec(r):
            return r.codec or default_codec(self.working_format)
        preview_codecs = list(dict.fromkeys(
            "h265" if "hevc" in codec(r) else "h264"
            for r in self.renditions if r.target == "janus" and r.feed == "dirty"))
        keys = {"dsk_keys": [{"id": k.id, "source": k.source} for k in self.dsk_keys],
                "dsk_fade_seconds": self.dsk_fade_seconds, "dsk_fade_curve": self.dsk_fade_curve} if self.dsk_keys else {}
        # The keyer splits off the clean feed, so it exists only with keys.
        from .janus import janus_mountpoint_id

        def preview(r):
            color = "sdr" if rendition_color(self.out_color, codec(r), r.color or None, r.tonemap).transfer == "sdr" else "hdr"
            port = r.port or 5004
            return {"rendition": r.id, "codec": "h265" if "hevc" in codec(r) else "h264", "color": color,
                    "port": port, "mountpoint": janus_mountpoint_id(port), "fps": r.fps}

        programs = [preview(r) for r in self.renditions if r.target == "janus" and r.feed == "dirty"]
        previews = []
        for r in self.renditions if self.dsk_keys else ():
            if r.feed == "clean" and r.target == "janus":
                output = preview(r)
                previews.append({"bus": f"clean_{r.id}", "label": f"Program clean · {output['color'].upper()}", **output})
        previews += [{"bus": b.id, "label": aux_label(b.id, b.label, b.layout),
                      "layout": b.layout.get("preset", "cells"), "rendition": r.id,
                      "codec": "h265" if "hevc" in r.codec else "h264", "color": "sdr", "port": r.port, "mountpoint": janus_mountpoint_id(r.port), "fps": r.fps}
                     for b in self.aux_buses for r in b.renditions]
        return {"source_count": len(self.sources), "browser_ring_size": self.browser_ring_size,
                "preview_codecs": preview_codecs, **({"program_outputs": programs} if programs else {}),
                "canvas": {"width": self.canvas_w, "height": self.canvas_h, "fps": self.fps,
                           "working_format": self.working_format},
                "source_counts": {kind: sum(s.kind == kind for s in self.sources) for kind in _SOURCE_KEYS},
                # Of the "video" inputs, those declared HDR; an undeclared one counts as SDR.
                "hdr_video": sum(s.kind == "video" and s.color_trc not in ("", TRANSFER_TAGS["sdr"]) for s in self.sources),
                "direct": self.direct,
                "fade_seconds": self.fade_seconds, "fade_curve": self.fade_curve,
                "fade_color": self.fade_color,
                "transition": self.transition,
                "wipe_file": default.path if default else "", "default_wipe": self.default_wipe,
                "wipes": wipes, **keys,
                **({"aux_buses": [b.id for b in self.aux_buses]} if self.aux_buses else {}),
                **({"preview_outputs": previews} if previews else {})}

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


_SOURCE_KEYS = {"browser": ("url", "width", "height"), "video": ("path",),
                **{kind: ("path", "width", "height") for kind in ("v210", "nv12", "p010")}}


def _parse_source(s: Dict[str, Any], where: str, fps: int) -> Source:
    sid, kind = str(s.get("id", "")), s.get("kind")
    if not sid or "#" in sid:
        raise ConfigError(f"{where}: id required (no '#')")
    if kind not in _SOURCE_KEYS:
        raise ConfigError(f"{where}: kind must be one of {', '.join(_SOURCE_KEYS)}")
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
    storage, extra = s.get("decode_storage", "cuda"), s.get("extra_hw_frames", 3)
    if storage not in ("cuda", "cuarray") or kind != "video" and ("decode_storage" in s or "extra_hw_frames" in s):
        raise ConfigError(f"{where}: decode_storage applies to video sources and must be cuda or cuarray")
    if type(extra) is not int or not 0 <= extra <= 32:
        raise ConfigError(f"{where}: extra_hw_frames must be an integer from 0 to 32")
    hold_last_frame = s.get("hold_last_frame", True)
    if not isinstance(hold_last_frame, bool) or "hold_last_frame" in s and kind != "browser":
        raise ConfigError(f"{where}: hold_last_frame must be a boolean on a browser source")
    try:
        color = declared_color(s)
        if kind != "video" and color is None:
            raise ValueError("raw input has no color metadata; declare an explicit color setting")
        if kind == "browser" and color != Color():
            raise ValueError("browser input supports SDR only")
        if kind in ("nv12", "p010"):
            if kind == "nv12" and color != Color():
                raise ValueError("nv12 input supports SDR only")
            if any(type(s[k]) is not int or s[k] <= 0 or s[k] % 2 for k in ("width", "height")):
                raise ValueError(f"{kind} input needs positive even width and height")
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from e
    return Source(sid, kind, str(s.get("url", s.get("path"))), width=int(s.get("width", 0)),
                  height=int(s.get("height", 0)), fps=int(s.get("fps", fps)) if kind == "browser" else 0,
                  loop=bool(s.get("loop", True)), hold_last_frame=hold_last_frame, filter_graph=source_filter,
                  filter_output_format=filter_format, decode_storage=storage, extra_hw_frames=extra,
                  **(color.tags if color else {}))


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
    if rendition.feed not in FEEDS:
        raise ConfigError(f"{where}: feed must be one of {FEEDS}")
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
    if not 0 <= rendition.dpb_size <= MAX_DPB_SIZE:
        raise ConfigError(f"{where}: dpb_size must be from 0 (NVENC decides) to {MAX_DPB_SIZE}")
    if rendition.fps > fps:
        raise ConfigError(f"{where}: fps {rendition.fps} exceeds the canvas rate {fps}; "
                          "a rendition can only re-time the program downwards")
    wanted = str(r.get("aspect", ""))
    if wanted and wanted != rendition.aspect:
        raise ConfigError(f"{where}: {rendition.width}x{rendition.height} is "
                          f"{rendition.aspect}, not {wanted}")
    return rendition


def check_janus_ports(renditions, default_port: int = 5004) -> None:
    """Janus renditions need distinct RTP ports whose RTCP port (RTP + 1) is free too; port 0 is
    the mixer's --janus-video-port, 5004 by default. parse_aux_buses checks the main renditions
    together with the aux ones."""
    used = set()
    for r in renditions:
        if r.target != "janus":
            continue
        port = r.port or default_port
        if not 1 <= port < 65535 or used.intersection((port, port + 1)):
            raise ConfigError("Janus renditions need distinct RTP/RTCP port pairs in 1..65535")
        used.update((port, port + 1))


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
    return {"direct": bool(control.get("direct", True)),
            "fade_seconds": fade_seconds,
            "fade_curve": fade_curve(control.get("fade_curve", DEFAULT_FADE_CURVE), "control.fade_curve"),
            "fade_color": fade_color(control.get("fade_color"), "control.fade_color"),
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
    raw_upload = canvas.get("raw_upload", "hwupload")
    if raw_upload not in RAW_UPLOADS:
        raise ConfigError(f"canvas.raw_upload must be one of {', '.join(RAW_UPLOADS)}")
    latency_ms = canvas.get("latency_ms")
    if latency_ms is not None:
        try:
            latency_ms = float(latency_ms)
        except (TypeError, ValueError):
            raise ConfigError("canvas.latency_ms must be a number of milliseconds") from None
        if not 0 <= latency_ms < 6000 / fps:
            raise ConfigError(f"canvas.latency_ms must be at least 0 and below six frames ({6000 / fps:.1f} ms at {fps} fps)")
    try:
        out_color = declared_color(canvas) or Color()
        out_color.validate_format(working_format)
    except ValueError as e:
        raise ConfigError(f"canvas: {e}") from e

    sources: List[Source] = []
    locations: Dict[Tuple[str, str], Tuple[str, bool]] = {}
    for i, s in enumerate(doc["sources"]):
        where = f"sources[{i}]"
        sources.append(_parse_source(s, where, fps))
        _unique(sources, where)
        key = (sources[-1].kind, sources[-1].location)
        independent = s.get("independent", False)
        if not isinstance(independent, bool):
            raise ConfigError(f"{where}: independent must be a boolean")
        if key in locations and not (independent and locations[key][1]):
            raise ConfigError(f"{where}: '{key[1]}' already declared as '{locations[key][0]}'; "
                              "reference that id instead, or mark both independent for separate input chains")
        locations[key] = (sources[-1].id, independent)
    if not sources:
        raise ConfigError("sources must not be empty")
    if len(sources) > MAX_SOURCES:
        raise ConfigError(f"at most {MAX_SOURCES} sources per show: each is a compositor pad "
                          f"and the pad mask is {MAX_SOURCES} bits")

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
            blend = it.get("blend", False)
            if not isinstance(blend, bool):
                raise ConfigError(f"{iw}: blend must be a boolean")
            items.append(Item(str(it["source"]), _rect(it["dst"], iw + ".dst"), fit, crop, blend))
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
    cfg = MixerConfig(canvas_w, canvas_h, fps, tuple(sources), tuple(scenes), tuple(wipes), tuple(renditions),
                       browser_ring_size=doc.get("browser_ring_size"),
                       max_compositor_layers=doc.get("max_compositor_layers", DEFAULT_MAX_COMPOSITOR_LAYERS),
                       initial_scene=initial, working_format=working_format, raw_upload=raw_upload, latency_ms=latency_ms, out_color=out_color,
                       wipe_color=wipe_color, **_parse_control(doc.get("control", {}), wipes))
    return replace(cfg, aux_buses=parse_aux_buses(doc.get("aux_buses", []), cfg),
                   **_parse_dsk(doc.get("dsk", {}), ids, canvas_w, canvas_h))


def _parse_dsk(dsk: Any, sources: Dict[str, Source], canvas_w: int, canvas_h: int) -> Dict[str, Any]:
    """Keys are ordinary sources, so scenes may use them too; each one here is
    alpha-blended over the finished program, above transitions and wipes.
    Returns the MixerConfig dsk_* fields."""
    if not isinstance(dsk, dict) or not isinstance(dsk.get("keys", []), list):
        raise ConfigError("dsk must be an object with a keys list")
    keys: List[DskKey] = []
    for i, k in enumerate(dsk.get("keys", [])):
        where = f"dsk.keys[{i}]"
        if not isinstance(k, dict) or "id" not in k:
            raise ConfigError(f"{where}: id required")
        source = sources.get(k.get("source"))
        if source is None or source.kind != "browser":
            raise ConfigError(f"{where}: source must be a browser source (keys need its alpha)")
        on = k.get("on", False)
        if not isinstance(on, bool):
            raise ConfigError(f"{where}: on must be a boolean")
        dst = _rect(k["dst"], where + ".dst") if "dst" in k else Rect(0, 0, canvas_w, canvas_h)
        keys.append(DskKey(str(k["id"]), source.id, dst, on))
        _unique(keys, where)
    if len(keys) > MAX_DSK_KEYS:
        raise ConfigError(f"dsk: at most {MAX_DSK_KEYS} keys")
    return {"dsk_keys": tuple(keys),
            "dsk_fade_seconds": key_fade_seconds(dsk.get("fade_seconds", DEFAULT_DSK_FADE_SECONDS), "dsk.fade_seconds"),
            "dsk_fade_curve": fade_curve(dsk.get("fade_curve", DEFAULT_FADE_CURVE), "dsk.fade_curve")}


DEFAULT_AUX_LAYOUT = {"preset": "pgm_pvw_grid"}


def aux_label(bus_id, label="", layout=None):
    """The control page's name for a bus: its *label*, else that of its initial *layout*
    (DEFAULT_AUX_LAYOUT without one), else Aux <id>."""
    return label or {"pgm_pvw_grid": "Program preview", "source_pages": "Multiviewer"}.get(
        (layout or DEFAULT_AUX_LAYOUT).get("preset"), f"Aux {bus_id}")


def aux_fps(fps, full_rate=False):
    """A bus runs at half the canvas rate at 50/60 fps unless it opts into the full rate."""
    return fps // 2 if fps in (50, 60) and not full_rate else fps


PVW_ALIGNMENTS = ("program", "pgm_tile")


def default_pgm_delay_frames(aux_fps):
    """Aux frames the PGM pad is matched back unless the bus says: one, or two at a 50/60 fps
    bus (full_rate). Either way about 33 ms (40 at 25/50) for the finished program to travel
    from the main compositor, which releases it at its deadline, to the bus's deadline for it;
    one 60 fps frame would leave 16.7 ms."""
    return 2 if aux_fps > 30 else 1


def _parse_bus_timing(obj, canvas_fps):
    """pvw_align, latency_ms, pgm_delay_frames and full_rate of one bus; the first and third
    matter while its layout draws the preview or the program. Their consistency with the main
    latency is checked at build (AuxBus.build), where the main mixer's latency is known."""
    if not isinstance(obj.get("full_rate", False), bool):
        raise ConfigError("aux full_rate must be a boolean")
    latency = obj.get("latency_ms")
    if latency is not None and (isinstance(latency, bool) or not isinstance(latency, (int, float)) or
                                not math.isfinite(latency) or latency <= 0):
        raise ConfigError("aux latency_ms must be a positive number of milliseconds or null")
    align = obj.get("pvw_align", "program")
    if align not in PVW_ALIGNMENTS:
        raise ConfigError(f"aux pvw_align must be one of {', '.join(PVW_ALIGNMENTS)}")
    delay = obj.get("pgm_delay_frames", default_pgm_delay_frames(aux_fps(canvas_fps, obj.get("full_rate", False))))
    if isinstance(delay, bool) or not isinstance(delay, int) or not 0 <= delay <= 6:
        raise ConfigError("aux pgm_delay_frames must be an integer from 0 to 6")
    return {"pvw_align": align, "latency_ms": latency, "pgm_delay_frames": delay,
            "full_rate": obj.get("full_rate", False)}


def parse_aux_buses(values, cfg):
    """Each bus: its initial `layout` (pgm_pvw_grid by default) and the `layouts` the operator can
    switch to (by default the presets, pgm_pvw_grid only when `layout` has a pgm cell itself), its
    slot assignments `scenes`, and the layer budget `max_layers`, max_compositor_layers by default.
    The bus gets the PGM pad only when `layout` or `layouts` has a pgm cell."""
    from .aux_layout import PRESETS, check_assignments, count, draws_program, layout_cells, parse_layout, same_kind
    if not isinstance(values, list) or len(values) > 30:
        raise ConfigError("aux_buses must be a list of at most 30 buses")
    if values and cfg.fps not in (25, 30, 50, 60):
        raise ConfigError("aux buses support program rates 25, 30, 50 and 60")
    result, ids = [], set()
    for obj in values:
        bid = obj.get("id", "")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", bid) or bid in ids:
            raise ConfigError("aux bus IDs must be unique identifiers")
        ids.add(bid)
        layouts = [parse_layout(cfg, obj.get("layout", DEFAULT_AUX_LAYOUT))]
        offered = obj.get("layouts", [{"preset": p} for p in PRESETS
                                      if draws_program(cfg, layouts) or not draws_program(cfg, [{"preset": p}])])
        if not isinstance(offered, list) or len(offered) > 16:
            raise ConfigError(f"aux {bid}: layouts must be a list of at most 16 layouts")
        for spec in (parse_layout(cfg, spec) for spec in offered):
            if not any(same_kind(spec, known) for known in layouts):
                layouts.append(spec)
        if len(cfg.sources) + draws_program(cfg, layouts) > MAX_SOURCES:
            raise ConfigError("an aux bus needs one pad per unique source, plus PGM when a layout draws it; "
                              f"limit is {MAX_SOURCES}")
        max_layers = obj.get("max_layers", cfg.max_compositor_layers)
        if type(max_layers) is not int or not 1 <= max_layers <= 2_147_483_647:
            raise ConfigError(f"aux {bid}: max_layers must be a positive 32-bit integer")
        cells = layout_cells(cfg, layouts[0], layouts[0].get("page", 0))
        scenes = obj.get("scenes", [])
        if not isinstance(scenes, list):
            raise ConfigError(f"aux {bid}: scenes must be a list of scene IDs by slot (null clears a slot)")
        scenes = scenes + [None] * (count(cells, "slot") - len(scenes))
        check_assignments(cfg, cells, scenes, max_layers)
        timing = _parse_bus_timing(obj, cfg.fps)
        fps = aux_fps(cfg.fps, timing["full_rate"])
        renditions = obj.get("renditions", [])
        if len(renditions) != 1:
            raise ConfigError("aux requires one SDR H.264 or HEVC Janus rendition")
        # A monitor needs no more than one reference frame (no B-frames): dpb_size 1 unless set.
        r = _parse_rendition({"codec": "h264_nvenc", "color": "sdr", "dpb_size": 1, **renditions[0]},
                             f"aux {bid}", cfg.canvas_w, cfg.canvas_h, fps)
        if (r.target != "janus" or r.codec not in ("h264_nvenc", "hevc_nvenc") or r.color != "sdr" or
                (r.width, r.height, r.fps) != (cfg.canvas_w, cfg.canvas_h, fps)):
            raise ConfigError("aux rendition must be SDR H.264 or HEVC at canvas size and the aux frame rate")
        if not r.port:
            raise ConfigError("aux needs a distinct explicit Janus RTP/RTCP port pair")
        if not isinstance(obj.get("label", ""), str):
            raise ConfigError(f"aux {bid}: label must be a string")
        result.append(AuxBusConfig(bid, tuple(scenes), (r,), layouts[0], tuple(layouts), max_layers,
                                   label=obj.get("label", ""), **timing))
    check_janus_ports([*cfg.renditions, *(b.renditions[0] for b in result)])
    return tuple(result)


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


def load(path: str, *, max_compositor_layers: Optional[int] = None) -> MixerConfig:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if max_compositor_layers is not None:
        doc["max_compositor_layers"] = max_compositor_layers
    return parse(doc)


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
        if item.blend:
            layer["blend"] = True
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
