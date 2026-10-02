"""AUX outputs sharing the main mixer's ingest: scene multiviews and paged source views."""
from __future__ import annotations

import json
import threading
import time
import uuid

from ..node import InternalNode
from .config import ConfigError, aux_fps, scene_layers, validate_assignments
from .control import source_mask_param


class _PvwFollowNode(InternalNode):
    TYPE = "mixer_pvw_follow"


def _even(value):
    return int(value) // 2 * 2


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


def _tile(index, rect, z):
    """Input *index* contained in *rect*, a cell or page tile."""
    return {"input": index, "dst_x": rect["x"], "dst_y": rect["y"], "dst_w": rect["w"], "dst_h": rect["h"],
            "fit": "contain", "z": z}


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
    layers.append(_tile(len(cfg.sources), pgm, reserved + len(layers)))
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
    return _layout([_tile(first + i, r, i) for i, r in enumerate(grid[:len(cfg.sources) - first])])


class _AuxOutput:
    """One AUX output: a compositor over subscribed sources, then SDR H.264 to Janus.

    Subclasses define the composition and _step(), one pass of the background thread
    (run) that keeps it current."""
    pgm_edge = None   # the edge the program tap feeds, for layouts that show PGM
    pgm_delay_frames = 0   # ticks the PGM input (the last one) is matched back; see AuxMultiview

    def __init__(self, avp, api, mixer, cfg, bus):
        self.avp, self.api, self.mixer, self.cfg, self.bus = avp, api, mixer, cfg, bus
        self.fps = aux_fps(cfg.fps, bus.full_rate)
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

    def main_latency_ms(self):
        """The main mixer's playout buffer: when a program frame leaves its compositor."""
        return self.mixer.latency_ms

    def latency_ms(self):
        # The bus's own playout buffer, or the main mixer's (50 ms at 60 fps, 1.5 aux ticks at
        # half rate): the sources reach a bus when they reach the program compositors, so the
        # same buffer leaves it the same slack, and its PVW tile can leave with the program.
        return self.bus.latency_ms if self.bus.latency_ms is not None else self.main_latency_ms()

    def build(self, options):
        from .janus import JanusVideoConfig, build_janus_output
        fps = self.fps
        latency = self.latency_ms()
        # Playout::setInputOffset's budget: the PGM delay is history the compositor keeps as well.
        if latency + self.pgm_delay_frames * 1000 / fps > 6000 / fps:
            raise ConfigError("main plus aux latency exceeds the six-frame aux history budget")
        # The program frame leaves the main compositor at its deadline, main latency after its
        # pts, and is drawn here pgm_delay_frames aux ticks plus the bus latency after it. That
        # margin carries it through the output chain (snapshot, selectors, keyer, tap) to the bus:
        # below one program frame the PGM tile repeats or runs a tick late on every take.
        floor = 1000 / self.cfg.fps
        if self.pgm_edge and self.pgm_delay_frames * 1000 / fps + latency - self.main_latency_ms() < floor - 1e-6:
            raise ConfigError(f"aux {self.bus.id}: pgm_delay_frames * aux frame + latency_ms must exceed the main "
                              f"latency_ms ({self.main_latency_ms():g}) by a program frame ({floor:.1f} ms) at least, "
                              f"for the program frame to reach the bus in time")
        inputs = self.inputs()
        converted = f"{self.prefix}_sdr"
        for edge in [*inputs, self.output_edge, converted]:
            self.avp.edges.planCapacity(edge, 1)
        self.avp.addNode(self.mixer.canvas_compositor({
            "name": self.node_name, "src": inputs, "dst": self.output_edge,
            "fps": str(fps), "latency_ms": latency, "warmup_timeout_ms": 250,
            "output_hwaccel": self.hwaccel,
            "aux_mode": True, "subscriptions": inputs, "mixer": self.mixer.name,
            "max_layers": self.layer_budget(),
            "group": self.group, "auto_restart": "off", "on_error": "off",
            **({"pgm_delay_frames": self.pgm_delay_frames} if self.pgm_delay_frames else {}),
            **self.current_composition(),
        }, api=self.api), early_create=True)
        r = self.bus.renditions[0]
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
            prefix=self.prefix, failure_mode="off", dpb_size=r.dpb_size, edge_capacity=1)

    def state(self):
        with self.lock:
            try:
                status = self.avp.node(self.node_name).getObject("status")
            except Exception:
                status = {"suspended": True}
            return {"id": self.bus.id, "layout": self.bus.layout, "error": self.error,
                    "canvas": {"w": self.cfg.canvas_w, "h": self.cfg.canvas_h},
                    "fps": self.fps, "latency_ms": self.latency_ms(), **self.details(), **status}

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

    def run(self):
        """Until stop(): _step() makes one pass and returns the seconds until the next."""
        while not self.stopped.wait(self._step()):
            pass

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
        # The PGM tile is the finished program (the last input), which reaches this bus a
        # frame after the sources it is made of. Matching it back (default_pgm_delay_frames)
        # keeps every other input at the normal latency instead of holding all of them longer.
        self.pgm_delay_frames = bus.pgm_delay_frames
        self.scenes, self.revision, self.preview = list(bus.scenes), uuid.uuid4().hex, ""
        self.follower_status = {}

    def inputs(self):
        return [*self.edges, self.pgm_edge]

    def current_composition(self):
        return composition(self.cfg, self.scenes, self.preview)

    def base(self, scenes=None, revision=None):
        return {"revision": revision or self.revision,
                **base_composition(self.cfg, self.scenes if scenes is None else scenes)}

    def build(self, options):
        super().build(options)
        self.avp.addNode(_PvwFollowNode({
            "name": self.follower, "mixer": self.mixer.name, "compositor": self.node_name,
            "fps": str(self.fps), "latency_ms": self.latency_ms(), "main_latency_ms": self.main_latency_ms(),
            "pgm_delay_frames": self.pgm_delay_frames, "align": self.bus.pvw_align,
            "base": self.base(), "pvw": pvw_layouts(self.cfg),
            "group": self.group, "auto_restart": "off", "on_error": "off",
        }))

    def state(self):
        state = super().state()
        # An assignment reaches the compositor through the follower: pending until the follower
        # has set a composition with this revision's base. Unreachable, this thread sets it.
        with self.lock:
            follower = self._follower_status()
            if follower is not None and follower.get("applied_base_revision") != self.revision:
                state["composition_pending"] = True
        return state

    def details(self):
        return {"scenes": list(self.scenes), "revision": self.revision, "cells": multiview_cells(self.cfg),
                "pvw_align": self.bus.pvw_align, "pgm_delay_frames": self.pgm_delay_frames,
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

    def _step(self):
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

    def _step(self):
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
