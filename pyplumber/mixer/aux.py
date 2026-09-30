"""AUX outputs sharing the main mixer's ingest: scene multiviews and paged source views."""
from __future__ import annotations

import json
import re
import threading
import time
import uuid

from ..node import InternalNode
from .config import AuxBus, ConfigError, _parse_rendition, scene_layers
from .control import source_mask_param


class _PvwFollowNode(InternalNode):
    TYPE = "mixer_pvw_follow"


def aux_fps(fps):
    return fps // 2 if fps in (50, 60) else fps


def _even(value):
    return int(value) // 2 * 2


def validate_assignments(cfg, scenes):
    definitions = {s.id: s for s in cfg.scenes}
    if not isinstance(scenes, (list, tuple)) or len(scenes) != 8:
        raise ConfigError("multiview needs exactly eight scene assignments (null clears a slot)")
    if any(s is not None and (not isinstance(s, str) or s not in definitions) for s in scenes):
        raise ConfigError("multiview references an unknown scene")
    reserved = max(len(s.items) for s in cfg.scenes)
    count = reserved + 1 + sum(len(definitions[s].items) for s in scenes if s is not None)
    if count > cfg.max_compositor_layers:
        raise ConfigError(f"multiview needs {count} layers including {reserved} reserved for PVW; limit is {cfg.max_compositor_layers}")


def parse_aux_buses(values, cfg):
    if not isinstance(values, list) or len(values) > 30:
        raise ConfigError("aux_buses must be a list of at most 30 buses")
    if values and cfg.fps not in (25, 30, 50, 60):
        raise ConfigError("multiview supports program rates 25, 30, 50 and 60")
    result, ids = [], set()
    ports = {r.port for r in cfg.renditions if r.port}
    for obj in values:
        bid = obj.get("id", "")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", bid) or bid in ids:
            raise ConfigError("aux bus IDs must be unique identifiers")
        ids.add(bid)
        layout = obj.get("layout", {})
        preset = layout.get("preset", "pgm_pvw_grid")
        rotate_s = obj.get("rotate_s", 5)
        if preset == "source_pages":
            if set(layout) != {"preset"} or "scenes" in obj:
                raise ConfigError("source_pages takes no grid size or scenes: it pages through every source")
            if len(cfg.sources) > 128:
                raise ConfigError("source pages need one pad per unique source; limit is 128")
            if isinstance(rotate_s, bool) or not isinstance(rotate_s, (int, float)) or not 1 <= rotate_s <= 60:
                raise ConfigError("rotate_s must be 1 to 60 seconds")
            scenes = ()
        else:
            if preset != "pgm_pvw_grid" or layout.get("rows", 2) != 2 or layout.get("cols", 4) != 4 or "rotate_s" in obj:
                raise ConfigError("aux layout is pgm_pvw_grid (2 rows by 4 columns, no rotate_s) or source_pages")
            if len(cfg.sources) + 1 > 128:
                raise ConfigError("multiview needs one pad per unique source plus PGM; limit is 128")
            scenes = obj.get("scenes", [None] * 8)
            validate_assignments(cfg, scenes)
        renditions = obj.get("renditions", [])
        if len(renditions) != 1:
            raise ConfigError("v1 aux requires one SDR/H.264 Janus rendition")
        # A monitor needs no more than one reference frame (no B-frames): dpb_size 1 unless set.
        r = _parse_rendition({"codec": "h264_nvenc", "color": "sdr", "dpb_size": 1, **renditions[0]},
                             f"aux {bid}", cfg.canvas_w, cfg.canvas_h, aux_fps(cfg.fps))
        if (r.target != "janus" or r.codec != "h264_nvenc" or r.color != "sdr" or
                (r.width, r.height, r.fps) != (cfg.canvas_w, cfg.canvas_h, aux_fps(cfg.fps))):
            raise ConfigError("aux rendition must be SDR/H.264 at canvas size and the aux frame rate")
        if not r.port or r.port in ports or r.port + 1 in ports or r.port - 1 in ports:
            raise ConfigError("aux needs a distinct explicit Janus RTP/RTCP port pair")
        ports.add(r.port)
        result.append(AuxBus(bid, tuple(scenes), (r,), preset, float(rotate_s)))
    return tuple(result)


def multiview_cells(cfg):
    """PVW, PGM and the eight scene slots of the pgm_pvw_grid layout, in canvas pixels."""
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


def _layout(layers):
    return {"layers": layers, "active_inputs": source_mask_param(sum(1 << i for i in {l["input"] for l in layers}))}


def _cell_layers(cfg, scene, cell, first_z):
    """One layer per scene item, drawn into *cell*, z from *first_z* in item order."""
    indices = {s.id: i for i, s in enumerate(cfg.sources)}
    return [{**layer, "input": indices[name.split("#", 1)[0]], "z": first_z + z,
             "tile": {k: cell[k] for k in ("x", "y", "w", "h")},
             "scene_canvas": {"w": cfg.canvas_w, "h": cfg.canvas_h}}
            for z, (name, layer) in enumerate(scene_layers(cfg, scene).items())]


def pvw_layouts(cfg):
    """The PVW cell's layers of every scene, keyed by scene id: what mixer_pvw_follow draws for a
    preview. Their z is below the reserved count (the largest scene); base_composition starts there."""
    pvw = multiview_cells(cfg)[0]
    return {scene.id: _layout(_cell_layers(cfg, scene, pvw, 0)) for scene in cfg.scenes}


def base_composition(cfg, scenes):
    """The eight assigned scene tiles and the PGM tile: the part of the composition the
    preview does not change, z from the reserved count up."""
    validate_assignments(cfg, scenes)
    definitions = {s.id: s for s in cfg.scenes}
    _, pgm, *slots = multiview_cells(cfg)
    layers = []
    reserved = max(len(s.items) for s in cfg.scenes)
    for scene, cell in zip(scenes, slots):
        if scene:
            layers.extend(_cell_layers(cfg, definitions[scene], cell, reserved + len(layers)))
    layers.append({"input": len(cfg.sources), "dst_x": pgm["x"], "dst_y": pgm["y"],
                   "dst_w": pgm["w"], "dst_h": pgm["h"], "fit": "contain", "z": reserved + len(layers)})
    return _layout(layers)


def composition(cfg, scenes, preview):
    """The whole multiview: pvw_layouts()[preview] followed by base_composition(), exactly what
    the follower node sets for that preview."""
    base = base_composition(cfg, scenes)
    if not preview:
        return base
    layouts = pvw_layouts(cfg)
    if preview not in layouts:
        raise ConfigError(f"native preview references an unknown scene: {preview}")
    return _layout(layouts[preview]["layers"] + base["layers"])


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


def page_composition(cfg, page):
    grid = page_grid(cfg)
    first = page * len(grid)
    layers = [{"input": first + i, "dst_x": r["x"], "dst_y": r["y"], "dst_w": r["w"], "dst_h": r["h"],
               "fit": "contain", "z": i} for i, r in enumerate(grid[:len(cfg.sources) - first])]
    return {"layers": layers, "active_inputs": source_mask_param(sum(1 << l["input"] for l in layers))}


class _AuxOutput:
    """One AUX output: a compositor over subscribed sources, then SDR H.264 to Janus.

    Subclasses define the composition and the background thread that keeps it current."""
    pgm_edge = None   # the edge the program tap feeds, for layouts that show PGM
    pgm_delay_frames = 0   # ticks the PGM input (the last one) is matched back; see AuxMultiview

    def __init__(self, avp, api, mixer, cfg, bus):
        self.avp, self.api, self.mixer, self.cfg, self.bus = avp, api, mixer, cfg, bus
        self.prefix = f"aux_{bus.id}"
        self.group = self.prefix
        self.node_name = f"{self.prefix}_comp"
        self.hwaccel = f"{self.prefix}_gpu"
        self.edges = [f"{self.prefix}_source_{i}" for i in range(len(cfg.sources))]
        self.output_edge = f"{self.prefix}_out"
        self.lock, self.stopped = threading.Lock(), threading.Event()
        self.thread, self.listener, self.error = None, None, ""
        for source, edge in zip(cfg.sources, self.edges):
            mixer.add_aux_destination(source.id, edge)

    def inputs(self):
        return self.edges

    def layer_budget(self):
        return self.cfg.max_compositor_layers

    def latency_ms(self):
        # The main mixer's latency, but at least two aux ticks (Playout's default): at
        # 50/60 fps the aux runs at half rate, where the main default is only one tick.
        main_latency = self.mixer.latency_ms
        main = main_latency if main_latency is not None else 2000 / self.cfg.fps
        return max(main, 2000 / aux_fps(self.cfg.fps))

    def build(self, options):
        from .janus import JanusVideoConfig, build_janus_output
        fps = aux_fps(self.cfg.fps)
        latency = self.latency_ms()
        # Playout::setInputOffset's budget: the PGM delay is history the compositor keeps as well.
        if latency + self.pgm_delay_frames * 1000 / fps > 6000 / fps:
            raise ConfigError("main plus aux latency exceeds the six-frame aux history budget")
        inputs = self.inputs()
        for edge in [*inputs, self.output_edge, *(f"{self.prefix}_{suffix}" for suffix in
                      ("sdr", "fps", "keyframed", "video", "encoded", "repeat_headers", "video_rtp_mux"))]:
            self.avp.edges.planCapacity(edge, 1)
        self.avp.addNode(self.mixer.backend.compositor({
            "name": self.node_name, "src": inputs, "dst": self.output_edge,
            "width": self.cfg.canvas_w, "height": self.cfg.canvas_h,
            "sw_format": self.cfg.working_format, "color": self.cfg.out_color.transfer,
            "fps": str(fps), "latency_ms": latency, "warmup_timeout_ms": 250,
            "hwaccel": self.mixer.hwaccel, "output_hwaccel": self.hwaccel,
            "aux_mode": True, "subscriptions": inputs, "mixer": self.mixer.name,
            "max_layers": self.layer_budget(),
            "group": self.group, "auto_restart": "off", "on_error": "off",
            **({"pgm_delay_frames": self.pgm_delay_frames} if self.pgm_delay_frames else {}),
            **self.current_composition(),
        }, api=self.api), early_create=True)
        r = self.bus.renditions[0]
        converted = f"{self.prefix}_sdr"
        self.avp.addNode(self.api.FilterVideo({
            "name": converted, "src": self.output_edge, "dst": converted,
            "hwaccel": self.hwaccel, "group": self.group, "defer_preliminary_init": True,
            "threads": self.mixer.backend.graph_threads,
            "graph": self.mixer.backend.conversion("sdr", "nv12", source=self.cfg.out_color,
                                      source_format=self.cfg.working_format, tonemap=r.tonemap or "clip"),
        }))
        self.listener = build_janus_output(
            self.avp, self.api, converted,
            JanusVideoConfig(host=options.janus_host, video_port=r.port,
                             bitrate_kbps=r.bitrate_kbps, ssrc=options.janus_video_ssrc + r.port,
                             rtcp_bind=options.janus_rtcp_bind, rtcp_port=0),
            fps=fps, width=r.width, height=r.height, hwaccel=self.hwaccel, group=self.group,
            codec="h264_nvenc", preset=r.preset, profile=r.profile or "high", enc_format="nv12",
            prefix=self.prefix, failure_mode="off", dpb_size=r.dpb_size)

    def state(self):
        with self.lock:
            try:
                status = self.avp.node(self.node_name).getObject("status")
            except Exception:
                status = {"suspended": True}
            return {"id": self.bus.id, "layout": self.bus.layout, "error": self.error,
                    "canvas": {"w": self.cfg.canvas_w, "h": self.cfg.canvas_h}, **self.details(), **status}

    def _set(self, node, key, value):
        self.avp.executeCommandsFromString(f"node.object.set {node} {key} {json.dumps(value)}")

    def _publish(self, snapshot, status=None):
        """Apply *snapshot*. An automatic update passes the compositor's *status* and keeps
        an encoder-backpressure suspension; an operator's change resumes the bus."""
        if status is not None:
            snapshot = {**snapshot, "enabled": not status.get("suspended", False)}
        self._set(self.node_name, "composition", snapshot)

    def start(self):
        self.avp.group(self.group).startNodes()
        self.listener.start()
        self.thread = threading.Thread(target=self.run, name=self.prefix, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the page thread and RTCP listener. The application stops the group together
        with all others (demos/mixer MixerApplication.stop), and not after a panic."""
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=1)
        self.listener.stop()


class AuxMultiview(_AuxOutput):
    """PVW, PGM and eight assignable scene tiles; the PVW tile follows the main mixer's preview.

    A ``mixer_pvw_follow`` node (``<prefix>_pvw``) draws the preview: the mixer hands it every
    change with the program frame it takes effect on, and it sets the compositor's composition
    on the multiview frame whose PGM tile shows the take. It holds the PVW-cell layers of every
    scene and the base layout (tiles and PGM) published here at build and on every reassignment;
    this thread only checks that it holds the current base, and while the node is unreachable
    it polls the mixer's preview and sets the composition itself, as before the node existed."""

    def __init__(self, avp, api, mixer, cfg, bus):
        super().__init__(avp, api, mixer, cfg, bus)
        self.pgm_edge = f"{self.prefix}_pgm"
        self.follower = f"{self.prefix}_pvw"
        self.scenes, self.revision, self.preview = list(bus.scenes), uuid.uuid4().hex, ""
        self.follower_status = {}

    def inputs(self):
        return [*self.edges, self.pgm_edge]

    # The PGM tile is the finished program (the last input), which reaches this bus a
    # frame after the sources it is made of. Matching it one frame back keeps every
    # other input at the normal latency instead of holding all of them a frame longer.
    pgm_delay_frames = 1

    def current_composition(self):
        return composition(self.cfg, self.scenes, self.preview)

    def base(self, scenes=None, revision=None):
        return {"revision": revision or self.revision,
                **base_composition(self.cfg, self.scenes if scenes is None else scenes)}

    def build(self, options):
        super().build(options)
        self.avp.addNode(_PvwFollowNode({
            "name": self.follower, "mixer": self.mixer.name, "compositor": self.node_name,
            "fps": str(aux_fps(self.cfg.fps)), "latency_ms": self.latency_ms(),
            "pgm_delay_frames": self.pgm_delay_frames, "base": self.base(), "pvw": pvw_layouts(self.cfg),
            "group": self.group, "auto_restart": "off", "on_error": "off",
        }))

    def details(self):
        return {"scenes": list(self.scenes), "revision": self.revision, "cells": multiview_cells(self.cfg),
                "follower": self.follower_status}

    def _follower_status(self):
        """The follower's status, None while the node is unreachable (not created, stopped). A
        command sent to it then is logged and dropped, never an error here, so this is the test."""
        try:
            return self.avp.node(self.follower).getObject("status")
        except Exception as exc:
            self.follower_status = {"error": str(exc)}
            return None

    def assign(self, request):
        with self.lock:
            if request.get("expected_revision") != self.revision:
                return {"error": "Assignments changed; refresh before editing", "conflict": True,
                        "scenes": list(self.scenes), "revision": self.revision}
            scenes = request.get("scenes")
            base = self.base(scenes, uuid.uuid4().hex)
            if self._follower_status() is not None:
                self._set(self.follower, "base", base)
            else:
                self._publish(composition(self.cfg, scenes, self.preview))
            self.scenes, self.revision, self.error = list(scenes), base["revision"], ""
            return {"scenes": list(self.scenes), "revision": self.revision}

    def _follow(self):
        """One pass; returns the seconds until the next. A follower that restarted holds the base
        it was built with, so resend the current one when its revision differs. Only one of the
        two, the node or this thread, sets the composition at any time; both use the same shown
        preview."""
        with self.lock:
            status = self._follower_status()
            if status is not None:
                if status.get("base_revision") != self.revision:
                    self._set(self.follower, "base", self.base())
                self.follower_status, self.preview = status, status.get("pvw_scene", "")
                return 1.0
            try:
                status = self.avp.node(self.node_name).getObject("status")
                preview = status.get("pvw_scene", "")
                if preview != self.preview:
                    self._publish(composition(self.cfg, self.scenes, preview), status)
                    self.preview = preview
            except Exception as exc:
                self.error = str(exc)
            return 0.05

    def run(self):
        while not self.stopped.wait(self._follow()):
            pass


class AuxSourcePages(_AuxOutput):
    """Every source in equal tiles, one page at a time. Pages rotate every rotate_s seconds
    until the operator picks one, which holds it; only the shown page's sources are delivered."""

    def __init__(self, avp, api, mixer, cfg, bus):
        super().__init__(avp, api, mixer, cfg, bus)
        self.per_page = len(page_grid(cfg))
        self.pages = -(-len(cfg.sources) // self.per_page)
        self.page, self.auto, self.flipped_at = 0, True, time.monotonic()

    def current_composition(self):
        return page_composition(self.cfg, self.page)

    def layer_budget(self):
        return self.per_page   # a page never draws more tiles

    def details(self):
        first = self.page * self.per_page
        shown = self.cfg.sources[first:first + self.per_page]
        return {"page": self.page, "pages": self.pages, "auto": self.auto, "rotate_s": self.bus.rotate_s,
                "first": first + 1, "total": len(self.cfg.sources),
                "tiles": [{"id": s.id, "kind": s.kind, **rect} for s, rect in zip(shown, page_grid(self.cfg))]}

    def _show(self, page, status=None):
        self.flipped_at = time.monotonic()
        self._publish(page_composition(self.cfg, page), status)
        self.page, self.error = page, ""

    def turn(self, request):
        """Hold a page (``page`` or relative ``step``) or switch rotation (``auto``)."""
        def integer(value):
            return isinstance(value, int) and not isinstance(value, bool)
        with self.lock:
            if "auto" in request:
                if not isinstance(request["auto"], bool):
                    raise ConfigError("auto must be a boolean")
                self.auto, self.flipped_at = request["auto"], time.monotonic()
            elif integer(request.get("page")) and 0 <= request["page"] < self.pages:
                self.auto = False
                self._show(request["page"])
            elif integer(request.get("step")):
                self.auto = False
                self._show((self.page + request["step"]) % self.pages)
            else:
                raise ConfigError(f"aux_page needs auto, a page from 0 to {self.pages - 1}, or a step")
            return self.details()

    def _tick(self):
        """Show the next page when one is due; return the seconds until the next check."""
        with self.lock:
            due = self.flipped_at + self.bus.rotate_s - time.monotonic()
            if self.auto and self.pages > 1 and due <= 0:
                try:
                    self._show((self.page + 1) % self.pages, self.avp.node(self.node_name).getObject("status"))
                except Exception as exc:
                    self.error = str(exc)
                due = self.bus.rotate_s
            # A held page or a single page has nothing due: check back at the slow rate.
            return min(max(due, 0.05), 0.5) if self.auto and self.pages > 1 else 0.5

    def run(self):
        while not self.stopped.wait(self._tick()):
            pass


def make_aux(avp, api, mixer, cfg, bus):
    kind = AuxSourcePages if bus.layout == "source_pages" else AuxMultiview
    return kind(avp, api, mixer, cfg, bus)


def register_aux_commands(avp, buses):
    by_id = {b.bus.id: b for b in buses}

    def target(request, kind):
        bus = by_id.get(request.get("bus"))
        if not isinstance(bus, kind):
            raise ConfigError(f"Unknown {'scene multiview' if kind is AuxMultiview else 'source pages'} bus")
        return bus

    def assign(arg):
        request = json.loads(arg)
        return json.dumps(target(request, AuxMultiview).assign(request)) + "\n"

    def turn(arg):
        request = json.loads(arg)
        return json.dumps(target(request, AuxSourcePages).turn(request)) + "\n"

    avp.registerControlCommand("mixer.aux", assign, True)
    avp.registerControlCommand("mixer.aux_page", turn, True)
    avp.registerControlCommand("mixer.aux_status", lambda _arg: json.dumps([b.state() for b in buses]) + "\n", True)
