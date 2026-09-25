"""Scene multiviews sharing the main mixer's ingest and native preview state."""
from __future__ import annotations

import json
import re
import threading
import uuid

from .config import AuxBus, ConfigError, _parse_rendition, scene_layers
from .control import source_mask_param


def aux_fps(fps):
    return fps // 2 if fps in (50, 60) else fps


def validate_assignments(cfg, scenes):
    definitions = {s.id: s for s in cfg.scenes}
    if not isinstance(scenes, (list, tuple)) or len(scenes) != 8:
        raise ConfigError("multiview needs exactly eight scene assignments (null clears a slot)")
    if any(s is not None and (not isinstance(s, str) or s not in definitions) for s in scenes):
        raise ConfigError("multiview references an unknown scene")
    reserved = max(len(s.items) for s in cfg.scenes)
    count = reserved + 1 + sum(len(definitions[s].items) for s in scenes if s is not None)
    if count > 256:
        raise ConfigError(f"multiview needs {count} layers including {reserved} reserved for PVW; limit is 256")


def parse_aux_buses(values, cfg):
    if not isinstance(values, list) or len(values) > 30:
        raise ConfigError("aux_buses must be a list of at most 30 buses")
    if values and len(cfg.sources) + 1 > 128:
        raise ConfigError("multiview needs one pad per unique source plus PGM; limit is 128")
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
        if (layout.get("preset", "pgm_pvw_grid") != "pgm_pvw_grid" or
                layout.get("rows", 2) != 2 or layout.get("cols", 4) != 4):
            raise ConfigError("v1 multiview layout is pgm_pvw_grid, 2 rows by 4 columns")
        scenes = obj.get("scenes", [None] * 8)
        validate_assignments(cfg, scenes)
        renditions = obj.get("renditions", [])
        if len(renditions) != 1:
            raise ConfigError("v1 aux requires one SDR/H.264 Janus rendition")
        r = _parse_rendition({"codec": "h264_nvenc", "color": "sdr", **renditions[0]},
                             f"aux {bid}", cfg.canvas_w, cfg.canvas_h, aux_fps(cfg.fps))
        if (r.target != "janus" or r.codec != "h264_nvenc" or r.color != "sdr" or
                (r.width, r.height, r.fps) != (cfg.canvas_w, cfg.canvas_h, aux_fps(cfg.fps))):
            raise ConfigError("aux rendition must be SDR/H.264 at canvas size and the aux frame rate")
        if not r.port or r.port in ports or r.port + 1 in ports or r.port - 1 in ports:
            raise ConfigError("aux needs a distinct explicit Janus RTP/RTCP port pair")
        ports.add(r.port)
        result.append(AuxBus(bid, tuple(scenes), (r,)))
    return tuple(result)


def composition(cfg, scenes, preview):
    validate_assignments(cfg, scenes)
    definitions = {s.id: s for s in cfg.scenes}
    indices = {s.id: i for i, s in enumerate(cfg.sources)}
    w, h = cfg.canvas_w, cfg.canvas_h
    xs = [(w * i // 4) // 2 * 2 for i in range(5)]
    ys = [(h * i // 4) // 2 * 2 for i in range(5)]
    places = [(preview, (0, 0, xs[2], ys[2]))]
    for i, scene in enumerate(scenes):
        col, row = i % 4, 2 + i // 4
        places.append((scene, (xs[col], ys[row], xs[col + 1] - xs[col], ys[row + 1] - ys[row])))
    layers = []
    for scene, (x, y, tw, th) in places:
        if not scene:
            continue
        if scene not in definitions:
            raise ConfigError(f"native preview references an unknown scene: {scene}")
        for name, layer in scene_layers(cfg, definitions[scene]).items():
            layers.append({**layer, "input": indices[name.split("#", 1)[0]], "z": len(layers),
                           "tile": {"x": x, "y": y, "w": tw, "h": th},
                           "scene_canvas": {"w": w, "h": h}})
    layers.append({"input": len(indices), "dst_x": xs[2], "dst_y": 0,
                   "dst_w": w - xs[2], "dst_h": ys[2], "fit": "contain", "z": len(layers)})
    mask = sum(1 << i for i in {layer["input"] for layer in layers})
    return {"layers": layers, "active_inputs": source_mask_param(mask)}


class AuxMultiview:
    def __init__(self, avp, api, mixer, cfg, bus):
        self.avp, self.api, self.mixer, self.cfg, self.bus = avp, api, mixer, cfg, bus
        self.prefix = f"aux_{bus.id}"
        self.group = self.prefix
        self.node_name = f"{self.prefix}_comp"
        self.hwaccel = f"{self.prefix}_gpu"
        self.edges = [f"{self.prefix}_source_{i}" for i in range(len(cfg.sources))]
        self.pgm_edge, self.output_edge = f"{self.prefix}_pgm", f"{self.prefix}_out"
        self.scenes, self.revision, self.preview = list(bus.scenes), uuid.uuid4().hex, ""
        self.lock, self.stopped = threading.Lock(), threading.Event()
        self.thread, self.listener, self.error = None, None, ""
        for source, edge in zip(cfg.sources, self.edges):
            mixer.add_aux_destination(source.id, edge)

    def build(self, options):
        from .janus import JanusVideoConfig, build_janus_output
        fps = aux_fps(self.cfg.fps)
        main_latency = self.mixer.latency_ms
        # Allow one aux frame for the rendered PGM tile to arrive.
        latency = (main_latency if main_latency is not None else 2000 / self.cfg.fps) + 1000 / fps
        if latency > 6000 / fps:
            raise ConfigError("main plus aux latency exceeds the six-frame aux history budget")
        inputs = [*self.edges, self.pgm_edge]
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
            "group": self.group, "auto_restart": "off", "on_error": "off",
            **composition(self.cfg, self.scenes, self.preview),
        }, api=self.api), early_create=True)
        r = self.bus.renditions[0]
        converted = f"{self.prefix}_sdr"
        self.avp.addNode(self.api.FilterVideo({
            "name": converted, "src": self.output_edge, "dst": converted,
            "hwaccel": self.hwaccel, "group": self.group, "defer_preliminary_init": True,
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
            prefix=self.prefix, failure_mode="off")

    def state(self):
        with self.lock:
            try:
                status = self.avp.node(self.node_name).getObject("status")
            except Exception:
                status = {"suspended": True}
            return {"id": self.bus.id, "scenes": list(self.scenes), "revision": self.revision,
                    "pvw_scene": self.preview, "error": self.error, **status}

    def assign(self, request):
        with self.lock:
            if request.get("expected_revision") != self.revision:
                return {"error": "Assignments changed; refresh before editing", "conflict": True,
                        "scenes": list(self.scenes), "revision": self.revision}
            scenes = request.get("scenes")
            snapshot = composition(self.cfg, scenes, self.preview)
            self._publish(snapshot)
            self.scenes, self.revision, self.error = list(scenes), uuid.uuid4().hex, ""
            return {"scenes": list(self.scenes), "revision": self.revision}

    def _publish(self, snapshot):
        self.avp.executeCommandsFromString(f"node.object.set {self.node_name} composition {json.dumps(snapshot)}")

    def start(self):
        self.avp.group(self.group).startNodes()
        self.listener.start()
        def follow_preview():
            while not self.stopped.wait(0.05):
                try:
                    with self.lock:
                        status = self.avp.node(self.node_name).getObject("status")
                        preview = status.get("pvw_scene", "")
                        if preview != self.preview:
                            snapshot = composition(self.cfg, self.scenes, preview)
                            snapshot["enabled"] = not status.get("suspended", False)
                            self._publish(snapshot)
                            self.preview = preview
                except Exception as exc:
                    self.error = str(exc)
        self.thread = threading.Thread(target=follow_preview, name=self.prefix, daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=1)
        self.listener.stop()
        self.avp.group(self.group).stopNodes()


def register_aux_commands(avp, buses):
    by_id = {b.bus.id: b for b in buses}
    def assign(arg):
        req = json.loads(arg)
        if req.get("bus") not in by_id:
            raise ConfigError("Unknown aux bus")
        return json.dumps(by_id[req["bus"]].assign(req)) + "\n"
    avp.registerControlCommand("mixer.aux", assign, True)
    avp.registerControlCommand("mixer.aux_status", lambda _arg: json.dumps([b.state() for b in buses]) + "\n", True)
