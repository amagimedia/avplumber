"""Manage one demo mixer process; the HTTP control server survives setup changes."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
import json
import logging
import math
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import threading
import time

from demo_recipe import allocate, validate_dsk, write_atomic
from extra_aux import aux_encode, extra_buses
from graphic_pages import key_graphics
from instance_profiles import INSTANCE_PROFILES, InstanceType
import janus_mountpoints
from pyplumber.mixer.color import TRANSFER_TAGS, default_codec
from pyplumber.mixer.control import AvpProtocolError
from pyplumber.mixer.aux_layout import parse_layout
from pyplumber.mixer.config import ConfigError, aux_fps, aux_label, parse, parse_aux_buses

DEMO_DIR = Path(__file__).resolve().parent
# Phase markers with durations: restart timings are read from the container log.
log = logging.getLogger("setup")
# encodes: the codec, NVENC preset and bitrate_kbps of each encoded output, by output id: the recipe's
# renditions (sdr, and hdr on a 10-bit canvas), the clean copy of the SDR program (sdr_clean), each
# of the show's own aux buses by bus id, and "extra", shared by every extra aux output. One the
# settings omit takes the profile's default (nvenc.defaults), an aux bus, own or extra, its "aux".
# A clean mixer stop releases browser DMA-BUF frames; a killed one leaves them quarantined
# until /workers/recover restarts the browser workers. Groups stop concurrently (mixer.py
# MixerApplication.stop), so this only bounds a hung stop. compose.yaml's stop_grace_period
# covers it plus the reap and the setup worker join in close().
STOP_TIMEOUT_SEC = 60
FAILED_START_STOP_TIMEOUT_SEC = 10
# /workers/recover answers once the affected browser workers have restarted, concurrently:
# each stops (<= 5 s), boots Electron and reopens its pages four at a time, every open
# bounded by the worker's 10 s request timeout.
RECOVER_TIMEOUT_SEC = 180
# Waits before restarting a mixer that exited on its own (a crash or a node panic). The
# sequence starts over once a mixer has run for HEALTHY_RUN_SEC.
RETRY_DELAYS_SEC = (2, 5, 15, 60)
HEALTHY_RUN_SEC = 300
# The demo is 1080p only: every source is a unique 1920x1080 input, in either orientation.
PROGRAM_SIZE = (1920, 1080)
DEFAULT_SETTINGS = dict(orientation="portrait", fps=60, bit_depth=10, chroma="422",
                        source_count=16, scene_count=32, layout="balanced", weights=[8, 4, 2, 0, 2, 0, 0],
                        encodes={}, dsk=[], clean_feed=False, extra_aux=0)
# The keys setup.html offers, in its order: the graphics whose manifest places them as a key.
DSK_CHOICES = [{"id": page, "label": key["label"]} for page, key in key_graphics().items()]


def for_mode(profile, bit_depth, chroma):
    """*profile* with the limits its instance measured apart on this canvas ("mode_limits")."""
    return {**profile, **profile.get("mode_limits", {}).get(f"{bit_depth}:{chroma}", {})}


def source_limit(profile, fps, bit_depth=8, chroma="420"):
    """The instance's per-rate total scaled by the canvas's share, never above what the NVDEC,
    browser and upload caps carry together (v210 inputs only on a 4:2:2 canvas). An unsupported
    pair (8-bit 4:2:2) is refused by the mode checks in recipe_for."""
    profile = for_mode(profile, bit_depth, chroma)
    total = int(profile["sources"][fps] * profile["mode_share"].get(f"{bit_depth}:{chroma}", 1.0))
    return min(total, profile["nvdec_decodes"][fps] + profile["browser_windows"] + profile["raw_upload_units"][fps]
               + (profile["hlg_v210"] if chroma == "422" else 0))


def extra_aux_limit(profile, cfg, encodes):
    """Extra aux outputs the instance's NVENC budget leaves beside *cfg*'s encodes, its renditions
    and aux buses; encodes above the budget on their own are refused. Each takes its frames/s at
    1920x1080, scaled by its pixels, times the profile's share of its codec at its preset; the clean
    feed counts even while off, and an extra bus is a canvas-size encode at aux_fps, both at
    their codec and preset in *encodes*. setup.html sums in the same order, so both round alike."""
    nvenc = profile["nvenc"]
    mode = for_mode(profile, 8 if cfg.working_format == "nv12" else 10,
                    "422" if cfg.working_format == "p210le" else "420")
    def share(width, height, fps, codec, preset):
        return fps * width * height / (1920 * 1080) * nvenc["pct_per_fps"]["hevc" if "hevc" in codec else "h264"][preset]
    encoded = [(r.width, r.height, r.fps, r.codec or default_codec(cfg.working_format), r.preset) for r in cfg.renditions]
    if not any(r.feed == "clean" for r in cfg.renditions):
        encoded.append((cfg.canvas_w, cfg.canvas_h, cfg.fps,
                        encodes["sdr_clean"].get("codec", "h264_nvenc"), encodes["sdr_clean"]["preset"]))
    encoded += [(r.width, r.height, r.fps, r.codec, r.preset) for b in cfg.aux_buses for r in b.renditions]
    used = sum(share(*e) for e in encoded)
    budget = mode.get("nvenc_budget_pct", {}).get(cfg.fps, nvenc["budget_pct"])
    max_outputs = mode.get("nvenc_max_outputs", {}).get(cfg.fps, nvenc.get("max_outputs", math.inf))
    output_slots = max_outputs - len(encoded)
    if output_slots < 0:
        raise ValueError(f"The instance supports at most {max_outputs} encoded outputs, including programs")
    if used > budget:
        raise ValueError(f"The encodes need {used:.1f}% of NVENC, above its {budget}% budget: choose faster presets")
    return min(output_slots, 30 - len(cfg.aux_buses), math.floor((budget - used) / share(
        cfg.canvas_w, cfg.canvas_h, aux_fps(cfg.fps), encodes["extra"].get("codec", "h264_nvenc"), encodes["extra"]["preset"])))


def _in_range(profile, kbps):
    low, high = profile["nvenc"]["bitrate_kbps"]
    return min(max(kbps, low), high)


def current_settings(profile, settings):
    """Saved *settings* in the current shape, every program encode filled in. Saved before extra
    aux outputs, they had none; the browser ring size the page no longer sets goes, and the show
    takes the frame rate's default (config.default_browser_ring_size); one program bitrate_kbps
    sets the SDR program and its clean copy, the HLG program in the defaults' ratio to it."""
    if not isinstance(settings, dict):
        return settings
    encodes, legacy = settings.get("encodes", {}), settings.get("bitrate_kbps")
    if not isinstance(encodes, dict) or len(encodes) > 64:
        raise ValueError("encodes must map at most 64 output ids to their preset and bitrate_kbps")
    defaults = profile["nvenc"]["defaults"]
    # Bound a legacy integer before converting to float; JSON can carry arbitrarily large ints.
    scale = (min(max(legacy, 0), profile["nvenc"]["bitrate_kbps"][1]) / defaults["sdr"]["bitrate_kbps"]
             if type(legacy) is int else 1)
    programs = {o: {**e, "bitrate_kbps": _in_range(profile, round(e["bitrate_kbps"] * scale))}
                for o, e in defaults.items() if o != "aux"}
    encodes = {**programs, "extra": dict(defaults["aux"]), **encodes}
    encodes = {o: {"codec": "hevc_nvenc" if o == "hdr" else "h264_nvenc", **e} if isinstance(e, dict) else e
               for o, e in encodes.items()}
    return {"extra_aux": 0, **{k: v for k, v in settings.items() if k not in ("browser_ring_size", "bitrate_kbps")},
            "encodes": encodes}


def _split_aux(buses, recipe):
    """*buses* as the instance's own and the setup's extra ones, which follow them: as many as
    *recipe*'s setup asked for, none without one (an adopted show)."""
    own = len(buses) - min(len(buses), (recipe.get("setup") or {}).get("extra_aux", 0))
    return buses[:own], buses[own:]


def _ports(buses):
    return {b["id"]: b["renditions"][0]["port"] for b in buses}


def _browser_ids(*shows):
    return {s["id"] for show in shows for s in show.get("sources", []) if s["kind"] == "browser"}


def source_counts(profile, total, weights, fps=25, reserved_browsers=0):
    """*reserved_browsers* are downstream-key pages: browser inputs outside the weighted mix."""
    counts = allocate(total, weights)
    # P010 uses twice the upload bytes of NV12; SDR/HDR decode share NVDEC.
    for indices, costs, limit, name in (
            ((2, 3), (1, 1), profile["hlg_v210"], "4:2:2 upload"),
            ((4,), (1,), profile["browser_windows"] - reserved_browsers, "Browser"),
            ((0, 1), (1, 1), profile["nvdec_decodes"][fps], "Combined NVDEC"),
            ((5, 6), (1, 2), profile["raw_upload_units"][fps], "Raw 4:2:0 upload units")):
        group = [(i, cost) for i, cost in zip(indices, costs) if i < len(weights)]
        if sum(counts[i] * cost for i, cost in group) > limit:
            size = min(limit, sum(counts[i] for i, _ in group))
            while True:
                capped = allocate(size, [weights[i] for i, _ in group]) if size else [0] * len(group)
                if sum(n * cost for n, (_, cost) in zip(capped, group)) <= limit:
                    break
                size -= 1
            remaining = [0 if i in indices else w for i, w in enumerate(weights)]
            if not any(remaining):
                raise ValueError(f"{name} is limited to {limit}; enable another source type")
            counts = source_counts(profile, total - size, remaining, fps, reserved_browsers)
            for (i, _), count in zip(group, capped):
                counts[i] = count
            break
    return counts


def recipe_for(profile, settings):
    """Accept only the bounded generic setup controls, never paths or commands; *profile* is the
    instance's entry in INSTANCE_PROFILES."""
    settings = current_settings(profile, settings)
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError("Expected orientation, fps, source_count, scene_count, bit_depth, chroma, layout and weights")
    for key, choices in (("orientation", ("portrait", "landscape")),
                         ("fps", (25, 30, 50, 60)),
                         ("bit_depth", (8, 10)),
                         ("chroma", ("420", "422")),
                         ("layout", ("balanced", "grids", "fullscreen"))):
        if settings[key] not in choices or key in ("fps", "bit_depth") and type(settings[key]) is not int:
            raise ValueError(f"Unsupported {key}")
    dsk = settings["dsk"]
    validate_dsk(dsk, settings["clean_feed"])
    # Key pages are sources too: they take their share of the same budget. A show above the
    # limit of its rate and canvas is scaled down to it, not refused: switching 110 SDR inputs
    # at 30 fps to a 10-bit canvas keeps the mix at the capacity of the new mode.
    limit = source_limit(profile, settings["fps"], settings["bit_depth"], settings["chroma"]) - len(dsk)
    if type(settings["source_count"]) is int and settings["source_count"] > limit >= 1:
        settings = {**settings, "source_count": limit}
    for key, maximum in (("source_count", limit), ("scene_count", profile.get("max_scenes", 192))):
        value = settings[key]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{key} must be an integer from 1 to {maximum}")
    # 30: parse_aux_buses's cap on all buses. The NVENC budget bounds it further once the show's own
    # buses are known (SetupRuntime.apply).
    if type(settings["extra_aux"]) is not int or not 0 <= settings["extra_aux"] <= 30:
        raise ValueError("extra_aux must be an integer from 0 to 30")
    low, high = profile["nvenc"]["bitrate_kbps"]
    for output, encode in settings["encodes"].items():
        if not isinstance(encode, dict) or encode.get("codec") not in ("h264_nvenc", "hevc_nvenc"):
            raise ValueError(f"encodes.{output}: codec must be h264_nvenc or hevc_nvenc")
        if output == "hdr" and encode["codec"] != "hevc_nvenc":
            raise ValueError("Program HDR requires hevc_nvenc (Main10)")
        presets = profile["nvenc"]["pct_per_fps"]["hevc" if encode["codec"] == "hevc_nvenc" else "h264"]
        if set(encode) != {"codec", "preset", "bitrate_kbps"} or encode["preset"] not in presets:
            raise ValueError(f"encodes.{output} must have a preset of {', '.join(presets)} and a bitrate_kbps")
        if type(encode["bitrate_kbps"]) is not int or not low <= encode["bitrate_kbps"] <= high:
            raise ValueError(f"encodes.{output}: bitrate_kbps must be an integer from {low} to {high}")
    weights = settings["weights"]
    most = max(profile["sources"].values())   # the page sends source counts as weights
    if not isinstance(weights, list) or len(weights) != 7 or any(type(w) is not int or not 0 <= w <= most for w in weights):
        raise ValueError(f"Provide seven integer source weights from 0 to {most}")
    if settings["bit_depth"] == 8 and (any(weights[1:4]) or weights[6]):
        raise ValueError("8-bit mode supports SDR 4:2:0 and browser sources only")
    if settings["bit_depth"] == 8 and settings["chroma"] != "420":
        raise ValueError("8-bit mode supports a 4:2:0 canvas only")
    if settings["chroma"] == "420" and any(weights[2:4]):
        raise ValueError("4:2:0 mode supports 4:2:0 and browser sources only")
    counts = source_counts(for_mode(profile, settings["bit_depth"], settings["chroma"]), settings["source_count"],
                           weights, settings["fps"], len(dsk))
    _check_hdr_decodes(profile, counts[1], settings["fps"])
    width, height = PROGRAM_SIZE
    recipe = json.loads((DEMO_DIR / "demo.example.json").read_text())
    recipe.update(source_count=settings["source_count"], scene_count=settings["scene_count"])
    def program_encode(output):
        encode = settings["encodes"][output]
        return {**encode, "profile": "main10" if output == "hdr" else "main" if encode["codec"] == "hevc_nvenc" else "baseline",
                "color": "hlg" if output == "hdr" else "sdr"}
    for rendition in recipe["renditions"]:
        rendition.update(program_encode(rendition["id"]))
    recipe["clean_rendition"] = program_encode("sdr_clean")
    canvas_width, canvas_height = (height, width) if settings["orientation"] == "portrait" else (width, height)
    recipe["canvas"].update(width=canvas_width, height=canvas_height, fps=settings["fps"])
    if settings["bit_depth"] == 8:
        recipe["canvas"].update(working_format="nv12", color="sdr")
        recipe["renditions"] = [recipe["renditions"][0]]
    else:
        recipe["canvas"]["working_format"] = "p010le" if settings["chroma"] == "420" else "p210le"
    recipe["generation"].update(width=width, height=height)
    for source in recipe["inputs"][:2]:
        source.update(profile.get("generated_decode", {}))
    recipe["inputs"].append({"id": "sdr420_raw", "kind": "generated", "color": "sdr",
                             "chroma": "420", "storage": "nv12"})
    recipe["inputs"].append({"id": "hlg420_raw", "kind": "generated", "color": "hlg",
                             "chroma": "420", "storage": "p010"})
    for source, count in zip(recipe["inputs"], counts):
        source["weight"] = count
        if source["kind"] == "browser":
            source.update(width=width, height=height)
    layouts = {"fullscreen": 2}
    mode = settings["layout"]
    if mode != "fullscreen":
        layouts.update({f"grid_{n}": 2 if mode == "grids" else 1
                        for n in (2, 4, 8, 16, 32, 64) if n <= settings["source_count"]})
        if mode == "balanced":
            layouts.update(pip=3, random=3)
            if counts[4] and sum(counts[:4]) + sum(counts[5:]):
                layouts["alpha_overlay"] = 2
                for index in (0, 5, 1, 6, 2, 3):
                    if counts[index]:
                        source = recipe["inputs"][index]
                        recipe["alpha_background"] = source["id"] + ("_001" if counts[index] > 1 else "_000")
                        if counts[index] == 1 and index in (0, 5):
                            source["pattern"] = "bars"
                        break
    recipe["dsk"], recipe["clean_feed"] = dsk, settings["clean_feed"]
    recipe["setup"] = settings
    recipe["layouts"] = layouts
    return recipe


def _check_hdr_decodes(profile, count, fps):
    maximum = profile.get("nvdec_hdr_decodes", {}).get(fps, math.inf)
    if count > maximum:
        raise ValueError(f"HDR NVDEC inputs are limited to {maximum} at {fps} fps to reserve VRAM")


def _fits(cfg, spec):
    try:
        parse_layout(cfg, spec)
        return True
    except ConfigError:
        return False


class SetupRuntime:
    def __init__(self, media_dir, recipe_path, bridge, instance_type, mixer_args=(), browser_url="http://127.0.0.1:9009",
                 janus_api=None):
        self.media_dir = Path(media_dir).resolve()
        self.media_dir.mkdir(parents=True, exist_ok=True)
        # prepare_demo stages every file in a .prepare-* directory; a killed preparation leaves it.
        for stale in self.media_dir.glob("**/.prepare-*"):
            shutil.rmtree(stale, ignore_errors=True)
        self.recipe_path = Path(recipe_path)
        self.bridge = bridge
        self.instance_type = InstanceType(instance_type)
        self.profile = INSTANCE_PROFILES[self.instance_type]
        self.mixer_args = list(mixer_args)
        self.browser_url = browser_url
        # Extra aux outputs need a Janus mountpoint each, made through its HTTP API; none without it.
        self.janus_api = janus_api
        self.janus_error = ""
        # The show's own buses, for setup.html's NVENC budget, and the RTP ports of its extra ones,
        # whose mountpoints a smaller setup prunes: from _previous_show, then each synced setup.
        self.aux_buses, self.extra_ports = [], {}
        self.settings = None
        with suppress(OSError, ValueError):   # resume reports a broken show
            previous, _, _ = self._previous_show()
            if not self.recipe_path.exists():
                self.settings = current_settings(self.profile, previous.get("setup"))
        self.lock = threading.Lock()
        # Held for a whole _stop(): a second caller must not signal a mixer that is already
        # stopping (a second SIGINT would abort its clean shutdown).
        self.stop_lock = threading.Lock()
        self.closing = threading.Event()
        self.process = None
        self.worker = None
        self.retries = 0
        self.retry_timer = None
        self.phase = "idle"
        self.message = "Choose settings and apply to start the mixer."
        # A restarted setup server must also invalidate existing control/player tabs.
        self.revision = time.time_ns() // 1_000_000

    def status(self):
        with self.lock:
            if self.phase == "running" and self.process and self.process.poll() is not None:
                self.phase, self.message = "error", f"Mixer exited ({self.process.returncode}). Apply to retry."
            return dict(phase=self.phase, message=" ".join(filter(None, (self.message, self.janus_error))),
                        settings=self.settings, revision=self.revision, instance_type=self.instance_type.value,
                        profile=self.profile, aux_buses=self.aux_buses, janus_api=bool(self.janus_api),
                        dsk_pages=DSK_CHOICES)

    def _status(self, phase, message):
        with self.lock:
            self.phase, self.message = phase, message

    def resume(self):
        """Start the stored setup, else an existing show; a failure is reported in the status."""
        if self.recipe_path.exists():
            try:
                self.apply()
            except Exception as exc:
                self._status("error", str(exc))
            return
        if not (self.media_dir / "mixer.demo.json").exists():
            return
        # Adopt an existing explicit show when adding setup controls to a demo.
        def start():
            try:
                self._start_recovering(self.media_dir / "mixer.demo.json", None)
                self._status("running", "Existing mixer ready. Apply replaces it with the selected generic sources.")
                log.info("Mixer ready (existing show)")
            except Exception as exc:
                self._status("error", str(exc))
        self._status("starting", "Starting existing mixer…")
        self.worker = threading.Thread(target=start, daemon=True)
        self.worker.start()

    def _check_idle(self):
        """Call with self.lock held."""
        if self.closing.is_set() or (self.worker and self.worker.is_alive()):
            raise RuntimeError("A setup change is already in progress")

    def apply(self, settings=None):
        # Before _preserve_aux asks the live mixer, which a running change may be stopping.
        with self.lock:
            self._check_idle()
        recipe = recipe_for(self.profile, settings) if settings is not None else json.loads(self.recipe_path.read_text())
        # Plan validation happens before stopping the live mixer or writing files.
        from prepare_demo import plan
        show, _, _ = plan(recipe, self.media_dir)
        if settings is not None or isinstance(recipe.get("setup"), dict):
            # Resume the saved input recipe, including its decoder/storage contracts, but
            # reapply current output limits before restarting an older, larger setup.
            recipe["setup"] = current_settings(self.profile, recipe["setup"])
            self._preserve_aux(recipe, show)
            show, _, _ = plan(recipe, self.media_dir)
        self._validate_capacity(show, recipe.get("setup"))
        with self.lock:
            self._check_idle()
            self._cancel_retry()
            self.phase, self.message = "preparing", "Preparing assets…"
            self.worker = threading.Thread(target=self._apply, args=(recipe, settings), daemon=True)
            self.worker.start()

    def _previous_show(self):
        """The current show, its own aux buses and the setup's extra ones."""
        config = self.media_dir / "mixer.demo.json"
        previous = json.loads(config.read_text()) if config.exists() else {}
        # An adopted explicit show can declare the same setup metadata as a generated recipe.
        stored = json.loads(self.recipe_path.read_text()) if self.recipe_path.exists() else previous
        own, extra = _split_aux((previous or stored).get("aux_buses", []), stored)
        self.aux_buses = [{"id": b["id"], "label": aux_label(b["id"], b.get("label", ""), b.get("layout")),
                           "full_rate": b.get("full_rate", False)} for b in own]
        self.extra_ports = _ports(extra)
        return previous, own, extra

    def _preserve_aux(self, recipe, show):
        """Keep instance outputs while replacing the generic sources and scenes: each bus keeps its
        live layout, layouts and slot assignments, less the scenes that no longer exist or fit its budget,
        at its encode settings. The setup's extra buses follow the own ones: as many as asked for, within
        the NVENC budget, which the show's encodes must fit without them."""
        previous, own, extra = self._previous_show()
        if "max_compositor_layers" in previous:
            recipe["max_compositor_layers"] = show["max_compositor_layers"] = previous["max_compositor_layers"]
        wanted, encodes = recipe["setup"]["extra_aux"], recipe["setup"]["encodes"]
        extra = extra[:wanted]   # fewer drops the last ones
        for bus in own:
            bus["renditions"][0].update(aux_encode(encodes.get(bus["id"]) or self.profile["nvenc"]["defaults"]["aux"]))
        live = {}
        if (own or extra) and self.process and self.process.poll() is None:
            # A show started without aux buses has no mixer.aux_status, and so no live state to keep.
            with suppress(AvpProtocolError):
                live = {b["id"]: b for b in json.loads(self.bridge.command("mixer.aux_status"))}
        cfg = parse(show)
        scene_ids = {s.id for s in cfg.scenes}
        unpaged = lambda spec: {k: v for k, v in spec.items() if k != "page"}   # pages follow the new sources
        for bus in own + extra:
            for rendition in bus["renditions"]:
                rendition.update(width=cfg.canvas_w, height=cfg.canvas_h, fps=aux_fps(cfg.fps, bus.get("full_rate", False)))
            state = live.get(bus["id"], bus)
            if "layout" in state:
                bus["layout"] = unpaged(state["layout"])
            if "layouts" in state:   # the live menu, and with it the PGM pad when a layout has a pgm cell
                bus["layouts"] = [unpaged(spec) for spec in state["layouts"]]
            # A layout whose cells the new canvas or source list no longer has is dropped, the live
            # one falling back to the first that still fits: a saved layout never refuses an Apply.
            if "layouts" in bus:
                bus["layouts"] = [spec for spec in bus["layouts"] if _fits(cfg, spec)]
            if "layout" in bus and not _fits(cfg, bus["layout"]):
                bus["layout"] = next(iter(bus.get("layouts", [])), {"preset": "source_pages"})
            scenes = [s if s in scene_ids else None for s in state.get("scenes", [])]
            for i, scene in enumerate(scenes):
                if not scene:
                    continue
                try:
                    parse_aux_buses([{**bus, "scenes": scenes[:i + 1]}], cfg)
                except ConfigError:
                    # A retained scene may have grown beyond the bus's draw budget.
                    scenes[i] = None
            bus["scenes"] = scenes
        mine = parse({**show, "aux_buses": own})
        limit = extra_aux_limit(self.profile, mine, encodes)
        if wanted and not self.janus_api:
            raise ValueError("Extra aux outputs need a Janus API (webui.py --janus-api)")
        # Mode/rate changes shrink the managed tail, just as source_count follows its new limit.
        wanted = recipe["setup"]["extra_aux"] = min(wanted, limit)
        if wanted:
            from prepare_demo import CLEAN_PORT   # clean feed or not, extra buses never take its port
            # Cells show the generic sources, not the key pages after them.
            own = own + extra_buses(replace(mine, sources=mine.sources[:recipe["source_count"]]), extra, wanted, CLEAN_PORT,
                                     encodes["extra"])
        if own:
            recipe["aux_buses"] = own
        else:
            recipe.pop("aux_buses", None)

    def remember_aux(self, bus_id, fields):
        """Persist an operator's change of a bus's ``layout`` or ``scenes`` (a mixer.aux,
        mixer.aux_layout or mixer.aux_page answer), so a resume or restart shows the same bus;
        its ``layouts`` too, which keep the PGM pad of a bus switched away from its pgm cells."""
        fields = {k: fields[k] for k in ("layout", "layouts", "scenes") if k in fields}
        with self.lock:   # an Apply carries the live bus over itself (_preserve_aux)
            if self.worker and self.worker.is_alive():
                return
            for path in (self.media_dir / "mixer.demo.json", self.recipe_path):
                doc = json.loads(path.read_text()) if path.exists() else {}
                bus = next((b for b in doc.get("aux_buses", []) if b.get("id") == bus_id), None)
                if bus is not None and any(bus.get(k) != v for k, v in fields.items()):
                    bus.update(fields)
                    write_atomic(path, json.dumps(doc, indent=2) + "\n")

    def _stop(self, timeout=STOP_TIMEOUT_SEC):
        with self.stop_lock:
            process = self.process
            if process and process.poll() is None:
                started = time.monotonic()
                log.info("Stopping mixer process %s", process.pid)
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    log.warning("Mixer shutdown timed out; killing process before browser recovery")
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                log.info("Mixer process exited (%s) in %.1f s", process.returncode, time.monotonic() - started)
            self.process = None

    def _recover_browsers(self, shows):
        from pyplumber.mixer.dmabuf_inputs import rest_request
        if self.process and self.process.poll() is None:
            raise RuntimeError("Cannot recover browser buffers while the mixer is running")
        ids = sorted(_browser_ids(*shows))
        if not ids:
            return False
        status = rest_request(self.browser_url, "GET", "/status") or {}
        if not any(w["id"] in ids and w.get("stats", {}).get("quarantinedFrameCount", 0)
                   for w in status.get("windows", [])):
            return False
        self._status("starting", "Recovering browser workers…")
        started = time.monotonic()
        log.info("Recovering browser workers with quarantined frames")
        rest_request(self.browser_url, "POST", "/workers/recover", {"ids": ids}, timeout=RECOVER_TIMEOUT_SEC)
        log.info("Browser workers recovered in %.1f s", time.monotonic() - started)
        return True

    def _start_recovering(self, config, previous_show):
        shows = [json.loads(config.read_bytes()), previous_show or {}]
        self._recover_browsers(shows)
        self._close_removed_browsers(shows[1], shows[0])
        for attempt in range(2):
            try:
                self._start(config)
                break
            except Exception:
                self._stop(timeout=FAILED_START_STOP_TIMEOUT_SEC)
                # Disconnect/quarantine notification can arrive after the first
                # status check. Retry only when a worker was actually recovered.
                if attempt or self.closing.is_set() or not self._recover_browsers(shows):
                    raise
        # Every mixer that reached air is watched, a restored previous show included.
        threading.Thread(target=self._watch, args=(self.process,), daemon=True).start()

    def _validate_capacity(self, show, settings=None):
        """Saved and restored shows obey current limits without rewriting their input contracts."""
        cfg = parse(show)
        _check_hdr_decodes(self.profile, sum(s.kind == "video" and s.color_trc not in ("", TRANSFER_TAGS["sdr"])
                                            for s in cfg.sources), cfg.fps)
        encodes = current_settings(self.profile, settings or self.settings or {})["encodes"]
        extra_aux_limit(self.profile, cfg, encodes)

    def _watch(self, process):
        started = time.monotonic()
        code = process.wait()
        with self.stop_lock:   # a deliberate _stop() replaces self.process before releasing it
            if self.process is not process or self.closing.is_set():
                return
        with self.lock:
            if time.monotonic() - started >= HEALTHY_RUN_SEC:
                self.retries = 0
        self._schedule_retry(f"Mixer exited ({code})")

    def _schedule_retry(self, reason):
        with self.lock:
            others = self.worker and self.worker.is_alive() and self.worker is not threading.current_thread()
            if self.closing.is_set() or others:
                return   # shutting down, or a setup change owns the mixer now
            delay = RETRY_DELAYS_SEC[min(self.retries, len(RETRY_DELAYS_SEC) - 1)]
            self.retries += 1
            self.phase, self.message = "error", f"{reason}; restarting in {delay} s. Apply restarts it now."
            log.warning("%s; restarting in %d s", reason, delay)
            timer = threading.Timer(delay, lambda: self._retry(timer))
            timer.daemon = True
            self.retry_timer = timer
            timer.start()

    def _cancel_retry(self):
        """Call with self.lock held."""
        if self.retry_timer:
            self.retry_timer.cancel()
        self.retry_timer = None

    def _retry(self, timer):
        with self.lock:
            if self.retry_timer is not timer or self.closing.is_set() or (self.worker and self.worker.is_alive()):
                return   # cancelled, superseded or shutting down
            self.retry_timer = None
            self.phase, self.message = "starting", "Restarting mixer…"
            self.worker = threading.Thread(target=self._restart, daemon=True)
            self.worker.start()

    def _restart(self):
        started = time.monotonic()
        try:
            self._start_recovering(self.media_dir / "mixer.demo.json", None)
            with self.lock:
                self.revision += 1
                self.phase, self.message = "running", "Mixer ready (restarted automatically)."
            log.info("Mixer ready: restarted in %.1f s", time.monotonic() - started)
        except Exception as exc:
            self._stop()
            self._schedule_retry(f"Mixer restart failed: {exc}")

    def _start(self, config):
        started = time.monotonic()
        # Each explicit source has one normalization filter before shared fan-out.
        # It must be running even when no scene currently consumes that source.
        source_nodes = {f"mixer_color_{source['id']}" for source in json.loads(config.read_text())["sources"]}
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, "rb", buffering=0) as startup:
            try:
                self.process = subprocess.Popen(
                    [sys.executable, "-u", str(DEMO_DIR / "mixer.py"), "--config", str(config),
                     "--remote-control-port", str(self.bridge.port), "--dmabuf-rest", self.browser_url,
                     *self.mixer_args], start_new_session=True, pass_fds=(write_fd,),
                    env={**os.environ, "AVP_MIXER_STARTUP_FD": str(write_fd)})
            finally:
                os.close(write_fd)
            log.info("Mixer process %s started", self.process.pid)
            self._wait_started(source_nodes, startup, started)

    def _wait_started(self, source_nodes, startup, started):
        deadline = time.monotonic() + 180
        startup_complete = False
        last_problem = "waiting for mixer startup"
        while not self.closing.wait(.5):
            ready = False
            if time.monotonic() > deadline:
                raise RuntimeError(f"Mixer startup timed out: {last_problem}")
            if select.select([startup], [], [], 0)[0]:
                error = startup.read(4000)
                if error:
                    raise RuntimeError("Mixer startup failed: " + error.decode(errors="replace"))
                startup_complete = True
            if self.process.poll() is not None:
                raise RuntimeError(f"Mixer exited during startup ({self.process.returncode}); see container logs")
            # The native server also delays its greeting until setReady(). Wait on the
            # child pipe so a control connection cannot hide an early startup failure.
            if not startup_complete:
                continue
            try:
                state = self.bridge.state(timeout=30.0)
                if not state.get("status", {}).get("pgm_scene"):
                    last_problem = "waiting for a program scene"
                else:
                    # queues.json carries every edge (~160 KB at 48 sources) on one line, and
                    # the mixer answers it slowly while it is still filling its decoders, so
                    # give it far more than the default command budget.
                    queues = json.loads(self.bridge.command("queues.json", timeout=60.0) or "[]")
                    expected = {"janus_encoded"}
                    outputs = state.get("settings", {})
                    expected.update(f"janus_{o['rendition']}_encoded" for o in outputs.get("program_outputs", [])[1:])
                    expected.update(f"janus_{o['rendition']}_encoded" for o in outputs.get("preview_outputs", [])
                                    if o["bus"].startswith("clean_"))
                    expected.update(f"aux_{bid}_encoded" for bid in outputs.get("aux_buses", []))
                    encoded = {q["name"] for q in queues if q["enqueued_total"] > 0}
                    ready = expected <= encoded
                    last_problem = "waiting for encoded output"
            except Exception as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
            if ready:
                nodes = json.loads(self.bridge.command("nodes.json", timeout=60.0) or "[]")
                working = {n["name"] for n in nodes if n["type"] == "filter_video" and n["working"]}
                failed = source_nodes - working
                if failed:
                    raise RuntimeError("Source normalization not running: " + ", ".join(sorted(failed)) +
                                       "; see container logs")
                log.info("Mixer encoding in %.1f s", time.monotonic() - started)
                return
        raise RuntimeError("Setup server is stopping")

    def _close_removed_browsers(self, old_show, new_show):
        from pyplumber.mixer.dmabuf_inputs import rest_request
        removed = _browser_ids(old_show) - _browser_ids(new_show)
        if removed:
            status = rest_request(self.browser_url, "GET", "/status") or {}
            existing = {w["id"] for w in status.get("windows", [])}
            for name in sorted(removed & existing):
                rest_request(self.browser_url, "POST", "/window/close", {"id": name})

    def _mountpoints(self, show, extra_aux=0, prune=False, replace=False):
        """Replace codec contracts with encoders stopped; prune obsolete extras once on air."""
        if not self.janus_api:
            return
        try:
            janus_mountpoints.sync(self.janus_api, janus_mountpoints.outputs(show, extra_aux),
                                  prune=prune, replace=replace)
            self.janus_error = ""
            if prune:
                self.extra_ports = _ports(show.get("aux_buses", [])[-extra_aux:]) if extra_aux else {}
        except Exception as exc:
            if replace:
                raise
            self.janus_error = f"Janus mountpoints failed: {exc}."
            log.warning("Janus mountpoints: %s", exc)

    def _apply(self, recipe, settings):
        from prepare_demo import prepare
        config = self.media_dir / "mixer.demo.json"
        had_previous = config.exists()
        previous = None
        stopped = False
        recipe_show = {}
        previous_extras = len(self.extra_ports)
        wanted = (recipe.get("setup") or {}).get("extra_aux", 0)
        started = time.monotonic()
        try:
            previous = config.read_bytes() if had_previous else None
            # Complete missing assets while the old mixer continues to run.
            log.info("Preparing assets")
            prepare(recipe, self.media_dir, progress=lambda done, total:
                    self._status("preparing", f"Preparing assets… {done}/{total} ready."))
            log.info("Assets ready in %.1f s", time.monotonic() - started)
            if self.closing.is_set():
                raise RuntimeError("Setup server is stopping")
            self._status("starting", "Restarting mixer…")
            self._stop()
            stopped = True
            recipe_show = json.loads(config.read_bytes())
            self._mountpoints(recipe_show, wanted, replace=True)
            self._start_recovering(config, json.loads(previous or b"{}"))
            if settings is not None or recipe != json.loads(self.recipe_path.read_text()):
                write_atomic(self.recipe_path, json.dumps(recipe, indent=2) + "\n")
            self._mountpoints(recipe_show, wanted, prune=True)
            with self.lock:
                self.settings = current_settings(self.profile, recipe.get("setup"))
                self.revision += 1
                self.retries = 0
                self.phase, self.message = "running", "Mixer ready."
            log.info("Mixer ready: setup applied in %.1f s", time.monotonic() - started)
        except Exception as exc:
            message = str(exc)
            try:
                if stopped:
                    self._stop()
                if previous is not None:
                    write_atomic(config, previous.decode())
                    if stopped and isinstance(exc, TimeoutError):
                        # The browser service may still be restarting workers; restoring now would
                        # recover them a second time. The next Apply starts from a settled service.
                        message += "; the browser service is still busy, apply again when it settles."
                    elif stopped and not self.closing.is_set():
                        self._validate_capacity(json.loads(previous))
                        self._mountpoints(json.loads(previous), previous_extras, replace=True)
                        self._start_recovering(config, recipe_show)
                        with self.lock:
                            self.revision += 1
                        message += "; previous setup restored."
                elif not had_previous:
                    config.unlink(missing_ok=True)
            except Exception as rollback:
                message += f"; recovery failed: {rollback}"
            self._status("error", message)
            log.error("Setup failed after %.1f s: %s", time.monotonic() - started, message)

    def close(self):
        """Stop the mixer cleanly first: the container's stop grace period is spent on it,
        not on setup work that `closing` already cancels."""
        self.closing.set()
        with self.lock:
            self._cancel_retry()
        self._stop()
        if self.worker:
            self.worker.join(timeout=20)
        self._stop()   # a mixer the cancelled setup work started meanwhile
