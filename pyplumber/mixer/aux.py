"""AUX buses sharing the main mixer's ingest, each drawing a layout of cells (aux_layout) that the
operator can switch at runtime, and the one thread that keeps every bus current."""
from __future__ import annotations

import json
import threading
import time
import uuid

from ..node import InternalNode
from .aux_layout import (base_composition, check_assignments, count, draws_program, layout_cells, page_count,
                         page_grid, parse_layout, pvw_layouts)
from .config import ConfigError, aux_fps


class _PvwFollowNode(InternalNode):
    TYPE = "mixer_pvw_follow"


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


class AuxBus:
    """One AUX bus: a compositor over subscribed sources (and the program, last, when one of its
    layouts draws it), then SDR H.264 to Janus.

    A ``mixer_pvw_follow`` node (``<prefix>_pvw``) is the only writer of the composition: this class
    computes, for the current layout, every scene's preview layers and the base (everything else)
    with a new revision, and hands them to the node, which draws the mixer's preview changes with
    the program. A slot assignment, a layout switch and a page turn are each a new base."""

    def __init__(self, avp, api, mixer, cfg, bus):
        self.avp, self.api, self.mixer, self.cfg, self.bus = avp, api, mixer, cfg, bus
        self.fps = aux_fps(cfg.fps, bus.full_rate)
        self.prefix = f"aux_{bus.id}"
        self.group = self.prefix
        self.node_name = f"{self.prefix}_comp"
        self.follower = f"{self.prefix}_pvw"
        self.hwaccel = f"{self.prefix}_gpu"
        self.edges = [f"{self.prefix}_source_{i}" for i in range(len(cfg.sources))]
        # The finished program reaches the bus a frame after the sources it is made of. Matching it
        # back (default_pgm_delay_frames) keeps every other input at the normal latency.
        self.pgm_edge = f"{self.prefix}_pgm" if draws_program(cfg, bus.layouts) else None
        self.pgm_delay_frames = bus.pgm_delay_frames if self.pgm_edge else 0
        self.output_edge = f"{self.prefix}_out"
        self.lock = threading.Lock()
        self.listener, self.error, self.follower_status = None, "", {}
        self.layout, self.scenes = bus.layout, list(bus.scenes)
        self.page = bus.layout.get("page") or 0
        self.flipped_at = self.checked_at = time.monotonic()
        self.revision = uuid.uuid4().hex
        for source, edge in zip(cfg.sources, self.edges):
            mixer.add_aux_destination(source.id, edge)

    def inputs(self):
        return [*self.edges, self.pgm_edge] if self.pgm_edge else self.edges

    def cells(self):
        return layout_cells(self.cfg, self.layout, self.page)

    def main_latency_ms(self):
        """The main mixer's playout buffer: when a program frame leaves its compositor."""
        return self.mixer.latency_ms

    def latency_ms(self):
        # The bus's own playout buffer, or the main mixer's (50 ms at 60 fps, 1.5 aux ticks at
        # half rate): the sources reach a bus when they reach the program compositors, so the
        # same buffer leaves it the same slack, and its pvw cells can change with the program.
        return self.bus.latency_ms if self.bus.latency_ms is not None else self.main_latency_ms()

    def layout_object(self, resume=True):
        """The follower's ``layout``: every scene's preview layers and the base, with the revision.
        *resume* false keeps an encoder-backpressure suspension (an automatic page turn)."""
        cells = self.cells()
        return {"revision": self.revision, "pvw": pvw_layouts(self.cfg, cells),
                "base": base_composition(self.cfg, cells, self.scenes), "resume": resume}

    def build(self, options):
        """The bus's nodes, all in its own group: compositor, follower, conversion, encoder, Janus output."""
        from .janus import JanusVideoConfig, build_janus_output
        fps = self.fps
        latency = self.latency_ms()
        # Playout::setInputOffset's budget: the PGM delay is history the compositor keeps as well.
        if latency + self.pgm_delay_frames * 1000 / fps > 6000 / fps:
            raise ConfigError("main plus aux latency exceeds the six-frame aux history budget")
        # The program frame leaves the main compositor at its deadline, main latency after its
        # pts, and is drawn here pgm_delay_frames aux ticks plus the bus latency after it. That
        # margin carries it through the output chain (snapshot, selectors, keyer, tap) to the bus:
        # below one program frame a pgm cell repeats or runs a tick late on every take.
        floor = 1000 / self.cfg.fps
        if self.pgm_edge and self.pgm_delay_frames * 1000 / fps + latency - self.main_latency_ms() < floor - 1e-6:
            raise ConfigError(f"aux {self.bus.id}: pgm_delay_frames * aux frame + latency_ms must exceed the main "
                              f"latency_ms ({self.main_latency_ms():g}) by a program frame ({floor:.1f} ms) at least, "
                              f"for the program frame to reach the bus in time")
        inputs = self.inputs()
        converted = f"{self.prefix}_sdr"
        for edge in [*inputs, self.output_edge, converted]:
            self.avp.edges.planCapacity(edge, 1)
        initial = self.layout_object()
        self.avp.addNode(self.mixer.canvas_compositor({
            "name": self.node_name, "src": inputs, "dst": self.output_edge,
            "fps": str(fps), "latency_ms": latency, "warmup_timeout_ms": 250,
            "output_hwaccel": self.hwaccel,
            "aux_mode": True, "subscriptions": inputs, "mixer": self.mixer.name,
            "max_layers": self.bus.max_layers,
            "group": self.group, "auto_restart": "off", "on_error": "off",
            **({"pgm_delay_frames": self.pgm_delay_frames} if self.pgm_delay_frames else {}),
            **initial["base"],
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
        self.avp.addNode(_PvwFollowNode({
            "name": self.follower, "mixer": self.mixer.name, "compositor": self.node_name,
            "fps": str(fps), "latency_ms": latency, "main_latency_ms": self.main_latency_ms(),
            "pgm_delay_frames": self.pgm_delay_frames, "align": self.bus.pvw_align,
            "layout": initial, "group": self.group, "auto_restart": "off", "on_error": "off",
        }))

    def start(self):
        self.avp.group(self.group).startNodes()
        self.listener.start()

    def stop(self):
        """Stop the RTCP listener. The application stops the group together with all others
        (demos/mixer MixerApplication.stop), and not after a panic."""
        self.listener.stop()

    def _status(self, node):
        """*node*'s status, or None while it is unreachable (not created, stopped)."""
        try:
            return self.avp.node(node).getObject("status")
        except Exception as exc:
            if node == self.follower:
                self.follower_status = {"error": str(exc)}
            return None

    def _send(self, resume=True):
        """The current layout under a new revision to the follower. While the follower is
        unreachable the binding logs and drops the command; tick() resends it."""
        self.revision = uuid.uuid4().hex
        self.avp.executeCommandsFromString(
            f"node.object.set {self.follower} layout {json.dumps(self.layout_object(resume))}")

    def _pages(self):
        """A source_pages layout's page shown, page count, rotation and sources shown."""
        if self.layout.get("preset") != "source_pages":
            return {}
        per_page = len(page_grid(self.cfg))
        return {"page": self.page, "pages": page_count(self.cfg), "auto": self.layout["page"] is None,
                "rotate_s": self.layout["rotate_s"], "first": self.page * per_page + 1, "total": len(self.cfg.sources)}

    def _summary(self):
        return {"layout": self.layout, "scenes": list(self.scenes), "revision": self.revision, **self._pages()}

    def state(self):
        with self.lock:
            status = self._status(self.node_name) or {"suspended": True}
            follower = self._status(self.follower)
            if follower is not None:
                self.follower_status = follower
            sources = self.cfg.sources
            cells = [{**c, "id": sources[c["source"]].id, "kind": sources[c["source"]].kind}
                     if c["role"] == "source" else c for c in self.cells()]
            # Pending until the compositor draws this revision: the follower sets it a wake after
            # the change, and the compositor stages it until its new inputs have frames.
            pending = status.get("composition_pending", False) or status.get("composition_revision") != self.revision
            return {"id": self.bus.id, "error": self.error, "layouts": list(self.bus.layouts), "cells": cells,
                    "max_layers": self.bus.max_layers, "canvas": {"w": self.cfg.canvas_w, "h": self.cfg.canvas_h},
                    "fps": self.fps, "latency_ms": self.latency_ms(), "pvw_align": self.bus.pvw_align,
                    "pgm_delay_frames": self.pgm_delay_frames, "follower": self.follower_status,
                    "pvw_scene": self.follower_status.get("pvw_scene", ""), **self._summary(), **status,
                    "composition_pending": pending}

    def assign(self, request):
        """Slot assignments: the complete ``scenes`` list as the status reports it, by slot index,
        against ``expected_revision``."""
        with self.lock:
            if request.get("expected_revision") != self.revision:
                return {"error": "Assignments changed; refresh before editing", "conflict": True, **self._summary()}
            scenes, cells = request.get("scenes"), self.cells()
            if not count(cells, "slot"):
                raise ConfigError(f"aux {self.bus.id}: the layout has no scene slots")
            if not isinstance(scenes, list) or len(scenes) != len(self.scenes):
                raise ConfigError(f"aux {self.bus.id}: scenes must list all {len(self.scenes)} assignments (null clears one)")
            check_assignments(self.cfg, cells, scenes, self.bus.max_layers)
            self.scenes, self.error = list(scenes), ""
            self._send()
            return self._summary()

    def set_layout(self, request):
        """Switch to ``layout``, validated and within the layer budget; assignments keep their slots."""
        with self.lock:
            layout = parse_layout(self.cfg, request.get("layout"), self.bus.rotate_s)
            if not self.pgm_edge and draws_program(self.cfg, [layout]):
                raise ConfigError(f"aux {self.bus.id} has no program input: its layouts draw no pgm cell")
            page = layout.get("page") or 0
            cells = layout_cells(self.cfg, layout, page)
            scenes = self.scenes + [None] * (count(cells, "slot") - len(self.scenes))
            check_assignments(self.cfg, cells, scenes, self.bus.max_layers)
            self.layout, self.page, self.scenes, self.error = layout, page, scenes, ""
            self.flipped_at = time.monotonic()
            self._send()
            return self._summary()

    def turn(self, request):
        """Hold a page (``page`` or relative ``step``) or switch rotation (``auto``) of a source_pages
        layout: the layout's ``page``, null while rotating."""
        with self.lock:
            if self.layout.get("preset") != "source_pages":
                raise ConfigError(f"aux {self.bus.id} shows no source pages")
            pages = page_count(self.cfg)
            if "auto" in request:
                if not isinstance(request["auto"], bool):
                    raise ConfigError("auto must be a boolean")
                page = None if request["auto"] else self.page
            elif _integer(request.get("page")) and 0 <= request["page"] < pages:
                page = request["page"]
            elif _integer(request.get("step")):
                page = (self.page + request["step"]) % pages
            else:
                raise ConfigError(f"aux_page needs auto, a page from 0 to {pages - 1}, or a step")
            self.layout, self.flipped_at = {**self.layout, "page": page}, time.monotonic()
            if "auto" not in request:
                self.page, self.error = page, ""
                self._send()
            return self._summary()

    def tick(self, now):
        """One pass of the shared scheduler: turn a due page, and once a second resend the layout to
        a follower that restarted (it holds the one it was built with). Returns the seconds until
        this bus is due again."""
        with self.lock:
            rotating = self.layout.get("preset") == "source_pages" and self.layout["page"] is None and page_count(self.cfg) > 1
            try:
                if rotating and now >= self.flipped_at + self.layout["rotate_s"]:
                    self.page, self.flipped_at = (self.page + 1) % page_count(self.cfg), now
                    self._send(resume=False)
                if now >= self.checked_at + 1:
                    self.checked_at = now
                    follower = self._status(self.follower)
                    if follower is not None:
                        self.follower_status = follower
                        if follower.get("layout_revision") != self.revision:
                            self._send()
            except Exception as exc:
                self.error = str(exc)
            due = self.checked_at + 1
            return (min(due, self.flipped_at + self.layout["rotate_s"]) if rotating else due) - now


class AuxBuses:
    """Every AUX bus by id, their control commands, and one thread for all of them that turns
    pages and resends layouts (AuxBus.tick), however many buses there are."""

    def __init__(self, avp, buses):
        self.avp = avp
        self.buses = {bus.bus.id: bus for bus in buses}
        self.stopped, self.thread = threading.Event(), None

    def __iter__(self):
        return iter(list(self.buses.values()))

    def __len__(self):
        return len(self.buses)

    def start(self):
        for bus in self:
            bus.start()
        self.thread = threading.Thread(target=self.run, name="aux", daemon=True)
        self.thread.start()

    def tick(self):
        """Every bus's pass; the seconds until the next, from 50 ms to one second."""
        now = time.monotonic()
        return min(max(min((bus.tick(now) for bus in self), default=1.0), 0.05), 1.0)

    def run(self):
        while not self.stopped.wait(self.tick()):
            pass

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=1)
        for bus in self:
            bus.stop()

    def register_commands(self):
        def command(method):
            def handle(arg):
                request = json.loads(arg)
                bus = self.buses.get(request.get("bus"))
                if bus is None:
                    raise ConfigError(f"Unknown aux bus {request.get('bus')!r}")
                return json.dumps(method(bus, request)) + "\n"
            return handle
        self.avp.registerControlCommand("mixer.aux", command(AuxBus.assign), True)
        self.avp.registerControlCommand("mixer.aux_layout", command(AuxBus.set_layout), True)
        self.avp.registerControlCommand("mixer.aux_page", command(AuxBus.turn), True)
        self.avp.registerControlCommand("mixer.aux_status", lambda _arg: json.dumps([b.state() for b in self]) + "\n", True)
