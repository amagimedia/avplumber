"""An engine that answers the control protocol from memory: no graph, no GPU.

`python -m pyplumber.rest --fake` serves the REST API over it, for writing a client away from
a mixer host, and the gateway's tests run on it. It keeps what a client can observe through the
API (Program and Preview, keys, an AUX bus of each mode, gains) and records every line it was sent.
It does not model time: a take is on Program at once, a key is on or off without a fade, and a
source scheduled to resume is playing at once.
"""

from __future__ import annotations

import json
import threading
import time
import uuid

from pyplumber.mixer.control import AvpProtocolError


class _Refused(Exception):
    """The engine's `500 ERROR: <text>`."""


class FakeEngine:
    """A `pyplumber.rest.ControlClient`. `sent` lists the lines received, oldest first."""

    def __init__(self, mixer: str = "mixer", scenes=("cam1", "cam2", "cam3", "wide"), keys=("logo", "lower_third"),
                 audio_sources=("cam1", "cam2")):
        self.mixer, self.scenes = mixer, list(scenes)
        self.sent: list[str] = []
        self.pgm, self.pvw, self.lever = self.scenes[0], "", 0.0
        self.keys = {key: {"id": key, "source": f"{key}_page", "on": False, "rate": None} for key in keys}
        self.all_rate, self.prepared, self.ftb_on = 0.4, [], False
        layouts = [{"preset": "pgm_pvw_grid"}, {"preset": "source_pages", "page": 0}]
        self.aux = {"id": "mv", "revision": uuid.uuid4().hex, "layout": layouts[0], "scenes": self.scenes[:2],
                    "layouts": layouts, "mode": "follow"}
        # An external bus: its picture is the composition a client sets, not the layout it was built with.
        self.external = {"id": "monitor", "revision": uuid.uuid4().hex, "layout": layouts[0], "scenes": self.scenes[:2],
                         "layouts": layouts[:1], "mode": "external", "composition": None}
        self.gains = {source: 0.0 for source in audio_sources}
        # Show source pads a client may repoint: scenes double as source ids in this fake.
        self.sources = {scene: {"id": scene, "url": f"fake://{scene}", "loop": False, "kind": "video"}
                        for scene in self.scenes}
        self.processor = {"output": "program", "id": "duck", "type": "ducker", "enabled": True, "threshold_db": -40.0,
                          "depth_db": -20.0, "attack_ms": 10.0, "hold_ms": 300.0, "release_ms": 800.0, "gain_db": 0.0}
        self._lock = threading.Lock()
        self._commands = {
            "version": lambda _arg: "avplumber fake engine (pyplumber.rest.fake)\n",
            "nodes.json": lambda _arg: self._nodes(),
            "node.param.get": lambda arg: self._node(arg.split()[0])["params"],
            "node.object.get": lambda arg: {"node": self._node(arg.split()[0])["name"], "object": arg.split()[1]},
            "queues.json": lambda _arg: [{"name": f"{scene}_video", "capacity": 15, "occupied": 0} for scene in self.scenes],
            "mixer.status": lambda arg: self._for_mixer(arg, self._status),
            "mixer.scenes": lambda arg: self._for_mixer(arg, lambda: self.scenes),
            "mixer.settings": lambda _arg: self._settings(),
            "mixer.preview": lambda arg: self._preview(self._scene(json.loads(arg))),
            **{f"mixer.{take}": lambda arg: self._take(self._scene(json.loads(arg))) for take in ("cut", "fade", "wipe", "move")},
            "mixer.interrupt": lambda _arg: None,
            "mixer.lever": lambda arg: setattr(self, "lever", json.loads(arg)["position"]),
            "mixer.dsk": lambda arg: self._dsk(json.loads(arg)),
            "mixer.dsk_status": lambda _arg: self._keys(),
            "mixer.ftb": lambda arg: self._ftb(json.loads(arg)),
            "mixer.ftb_status": lambda _arg: {"on": self.ftb_on},
            "mixer.aux": lambda arg: self._aux_scenes(self._bus(json.loads(arg))),
            "mixer.aux_layout": lambda arg: self._aux_layout(self._bus(json.loads(arg))),
            "mixer.aux_page": lambda arg: self._aux_page(self._bus(json.loads(arg))),
            "mixer.aux_composition": lambda arg: self._aux_composition(self._bus(json.loads(arg), "external")),
            "mixer.aux_status": lambda _arg: [{key: value for key, value in bus.items() if key != "layouts"}
                                              for bus in (self.aux, self.external)],
            "mixer.aux_layouts": lambda _arg: {bus["id"]: bus["layouts"] for bus in (self.aux, self.external)},
            "mixer.audio_gain": lambda arg: self._audio_gain(json.loads(arg)),
            "mixer.audio_gain_status": lambda _arg: self._gains(),
            "mixer.audio_processor": lambda arg: self._audio_processor(json.loads(arg)),
            "mixer.audio_processor_status": lambda _arg: [self.processor],
            "mixer.source.set": lambda arg: self._source_set(json.loads(arg)),
            "mixer.source.prepare": lambda arg: self._source_prepare(json.loads(arg)),
            "mixer.source.media": lambda arg: self._source_media(json.loads(arg)),
            "mixer.source.resume_at": lambda arg: self._source_resume_at(json.loads(arg)),
        }

    def command(self, line: str, timeout: float | None = None) -> str | None:
        name, _, arg = line.partition(" ")
        with self._lock:
            self.sent.append(line)
            handler = self._commands.get(name)
            if handler is None:
                raise AvpProtocolError(f"400 Unknown command: {name}")
            try:
                reply = handler(arg.strip())
            except _Refused as exc:
                raise AvpProtocolError(f"500 ERROR: {exc}") from None
        return reply if reply is None or isinstance(reply, str) else json.dumps(reply) + "\n"

    # --- engine ---

    def _nodes(self):
        return [*({"name": f"input_{scene}", "type": "input", "working": True,
                   "params": {"name": f"input_{scene}", "type": "input", "url": f"fake://{scene}", "dst": f"{scene}_video"}}
                  for scene in self.scenes),
                {"name": "comp_a", "type": "mixer_compositor", "working": True,
                 "params": {"name": "comp_a", "type": "mixer_compositor", "src": [f"{scene}_video" for scene in self.scenes]}}]

    def _node(self, name):
        node = next((node for node in self._nodes() if node["name"] == name), None)
        if node is None:
            raise _Refused(f"node not found: {name}")
        return node

    # --- Program and Preview ---

    def _for_mixer(self, name, answer):
        if name != self.mixer:
            raise _Refused(f"unknown mixer: {name}")
        return answer()

    def _status(self):
        return {"pgm_scene": self.pgm, "pvw_scene": self.pvw, "pvw_slot_scene": self.pvw, "transition": "idle",
                "transition_on_air": False, "transition_paused": False, "transition_manual": False,
                "transition_progress": 0.0, "transition_duration_ms": 0, "now_pts_ms": int(time.monotonic() * 1000),
                "lever": {"position": self.lever, "origin": 0.0, "engaged": 0 < self.lever < 1}}

    def _settings(self):
        return {"source_count": len(self.scenes), "canvas": {"width": 1920, "height": 1080, "fps": 30, "working_format": "nv12"},
                "dsk_keys": [{"id": key["id"], "source": key["source"]} for key in self.keys.values()],
                "aux_buses": [{"id": self.aux["id"]}, {"id": self.external["id"]}], "audio_sources": list(self.gains),
                "audio_processors": [{key: value for key, value in self.processor.items() if key != "gain_db"}], "wipes": []}

    def _scene(self, request):
        self._for_mixer(request.get("mixer"), lambda: None)
        if request.get("scene") not in self.scenes:
            raise _Refused(f"unknown scene: {request.get('scene')}")
        return request["scene"]

    def _preview(self, scene):
        self.pvw = scene

    def _take(self, scene):
        if scene != self.pgm:
            self.pgm, self.pvw = scene, self.pgm   # the scene that left Program is the one previewed

    # --- keys ---

    def _keys(self):
        return [{**key, "all_rate": self.all_rate} for key in self.keys.values()]

    def _dsk(self, request):
        if "all_rate" in request:
            self.all_rate = request["all_rate"]
            return self._keys()
        names = request.get("keys", [request.get("key")])
        if any(name not in self.keys for name in names):
            raise _Refused("dsk needs a known key and a boolean on")
        if "rate" in request:
            self.keys[request["key"]]["rate"] = request["rate"]
        elif request.get("prepare"):
            self.prepared = names
        else:
            for name in names:
                self.keys[name]["on"] = not self.keys[name]["on"] if request.get("toggle") else request["on"]
        return self._keys()

    def _ftb(self, request):
        self.ftb_on = not self.ftb_on if request.get("toggle") else request["on"]
        return {"on": self.ftb_on}

    # --- the AUX bus ---

    def _bus(self, request, mode="follow"):
        """*request*, for a bus of *mode*: each mode refuses the other's commands, as the mixer does."""
        bus = next((bus for bus in (self.aux, self.external) if bus["id"] == request.get("bus")), None)
        if bus is None:
            raise _Refused(f"Unknown aux bus {request.get('bus')!r}")
        if bus["mode"] != mode:
            raise _Refused(f"aux {bus['id']}: this command needs mode {mode} (the bus is {bus['mode']})")
        return request

    def _aux_changed(self, **changes):
        self.aux.update(changes, revision=uuid.uuid4().hex)
        return {key: value for key, value in self.aux.items() if key != "id"}

    def _aux_scenes(self, request):
        if request.get("expected_revision") != self.aux["revision"]:
            return {"error": "Assignments changed; refresh before editing", "conflict": True}
        if len(request["scenes"]) != len(self.aux["scenes"]) or any(s is not None and s not in self.scenes for s in request["scenes"]):
            raise _Refused(f"aux mv: scenes must list all {len(self.aux['scenes'])} assignments (null clears one)")
        return self._aux_changed(scenes=request["scenes"])

    def _aux_layout(self, request):
        if request["layout"].get("preset") not in ("pgm_pvw_grid", "source_pages"):
            raise _Refused("aux layout must be one of the bus's layouts")
        return self._aux_changed(layout={"page": 0, **request["layout"]} if request["layout"]["preset"] == "source_pages" else request["layout"])

    def _aux_page(self, request):
        if self.aux["layout"].get("preset") != "source_pages":
            raise _Refused("aux mv shows no source pages")
        pages = 2
        page = request["page"] if "page" in request else (self.aux["layout"]["page"] + request["step"]) % pages
        if not 0 <= page < pages:
            raise _Refused(f"aux_page needs a page from 0 to {pages - 1} or a step")
        return self._aux_changed(layout={**self.aux["layout"], "page": page})

    def _aux_composition(self, request):
        composition = request.get("composition")
        if not isinstance(composition, dict) or not isinstance(composition.get("layers"), list):
            raise _Refused("aux composition needs a layers list")
        if "active_inputs" not in composition:
            raise _Refused("aux composition needs active_inputs")
        self.external["composition"] = composition
        return {"bus": self.external["id"], "node": f"aux_{self.external['id']}_comp"}

    # --- audio ---

    def _gains(self):
        return [{"id": source, "gain_db": gain} for source, gain in self.gains.items()]

    def _audio_gain(self, request):
        if any(source not in self.gains for source in request["gains"]):
            raise _Refused("audio_gain: unknown source id")
        self.gains.update(request["gains"])
        return self._gains()

    def _audio_processor(self, request):
        if (request.get("output"), request.get("id")) != (self.processor["output"], self.processor["id"]):
            raise _Refused("audio_processor: unknown output or processor id")
        self.processor.update({key: value for key, value in request.items() if key not in ("output", "id")})
        return [self.processor]

    # --- sources ---

    def _source_set(self, request):
        self._for_mixer(request.get("mixer"), lambda: None)
        source = self.sources.get(request.get("id"))
        if source is None:
            raise _Refused(f"source.set: unknown source id {request.get('id')!r}")
        if source["kind"] not in ("video", "ndi"):
            raise _Refused(f"source.set: source {source['id']!r} kind {source['kind']!r} is unsupported")
        url, loop = request.get("url"), request.get("loop")
        if not isinstance(url, str) or not url or not isinstance(loop, bool):
            raise _Refused("source.set needs a non-empty url and a boolean loop")
        source.update(url=url, loop=loop)
        return {"id": source["id"], "url": url, "loop": loop}

    def _source_pad(self, request, who):
        self._for_mixer(request.get("mixer"), lambda: None)
        source = self.sources.get(request.get("id"))
        if source is None:
            raise _Refused(f"{who}: unknown source id {request.get('id')!r}")
        if source["kind"] != "video":
            raise _Refused(f"{who}: source {source['id']!r} kind {source['kind']!r} has no pause or seek")
        return source

    def _source_prepare(self, request):
        source = self._source_pad(request, "source.prepare")
        url, loop = request.get("url"), request.get("loop")
        cue = request.get("cue_in_ms", 0)
        if not isinstance(url, str) or not url or not isinstance(loop, bool):
            raise _Refused("source.prepare needs a non-empty url and a boolean loop")
        if not isinstance(cue, int) or isinstance(cue, bool) or cue < 0:
            raise _Refused("source.prepare: cue_in_ms must be a non-negative int")
        source.update(url=url, loop=loop, paused=True, position_ms=cue)
        return {"id": source["id"], "url": url, "loop": loop, "cue_in_ms": cue, "duration_ms": 60_000}

    def _source_media(self, request):
        source = self._source_pad(request, "source.media")
        action = request.get("action")
        if action not in ("play", "pause", "seek", "stop", "restart"):
            raise _Refused("source.media: unknown action")
        position = request.get("position_ms")
        if action == "seek":
            if not isinstance(position, int) or isinstance(position, bool) or position < 0:
                raise _Refused("source.media: seek needs a non-negative int position_ms")
            source["position_ms"] = position
        elif action == "play":
            source["paused"] = False
        elif action in ("pause", "stop"):
            source["paused"] = True
            if action == "stop":
                source["position_ms"] = 0
        elif action == "restart":
            source.update(paused=False, position_ms=0)
        reply = {"id": source["id"], "action": action}
        if action == "seek":
            reply["position_ms"] = position
        return reply

    def _source_resume_at(self, request):
        source = self._source_pad(request, "source.resume_at")
        at_ms = request.get("at_ms")
        if not isinstance(at_ms, int) or isinstance(at_ms, bool) or at_ms < 0:
            raise _Refused("source.resume_at needs a non-negative int at_ms")
        source["paused"] = False
        return {"id": source["id"], "at_ms": at_ms}
