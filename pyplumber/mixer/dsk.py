"""Downstream keyer: up to four alpha browser keys over the finished program.

The keyer runs after transitions and wipes, so a clean feed still carries every
scene and M/E transition and only the keys are missing. It is one compositor
pass clocked by the program (``clock_input``): each program frame renders at
once over every key's frame stamped for that tick, so keying adds no playout
latency and a late browser paint never stalls the program. A key cuts or fades
on and off, each on its own; with none visible, the program frame passes through
without any GPU work.
"""
from __future__ import annotations

import json
import logging
import threading

from .config import ConfigError, fade_curve, key_fade_seconds
from .control import source_mask_param
from .janus import keyframe_command


class DownstreamKeyer:
    def __init__(self, avp, api, mixer, cfg, *, group):
        self.avp, self.api, self.mixer, self.cfg, self.group = avp, api, mixer, cfg, group
        self.node_name = "dsk_comp"
        self.keys = cfg.dsk_keys
        self.edges = [f"dsk_key_{k.id}" for k in self.keys]
        self.on = {k.id: k.on for k in self.keys}
        self.lock = threading.Lock()
        for key, edge in zip(self.keys, self.edges):
            # Keys subscribe to the shared source like an aux bus: the source
            # never waits for the keyer. Off keys stay subscribed, so a key comes
            # on with the next program frame.
            mixer.add_aux_destination(key.source, edge)

    def _mask(self, on: dict):
        return source_mask_param(1 | sum(1 << (i + 1) for i, k in enumerate(self.keys) if on[k.id]))

    def build(self, program: str, *, clean: bool) -> dict:
        """Key *program*; return the edge of each feed. A clean edge exists only when asked for."""
        feeds = {"dirty": "program_dirty"}
        keyed = program
        if clean:
            keyed, feeds["clean"] = "dsk_program", "program_clean"
            self.avp.addNode(self.api.Split({"name": "split_clean", "src": program, "dst": [feeds["clean"], keyed],
                                             "group": self.group, "on_error": "panic"}))
        for edge in self.edges:
            self.avp.edges.planCapacity(edge, 1)
        w, h = self.cfg.canvas_w, self.cfg.canvas_h
        layers = [{"dst_x": 0, "dst_y": 0, "dst_w": w, "dst_h": h}]
        layers += [{"input": i + 1, "dst_x": k.dst.x, "dst_y": k.dst.y, "dst_w": k.dst.w, "dst_h": k.dst.h,
                    "z": i + 1, "blend": True} for i, k in enumerate(self.keys)]
        self.avp.addNode(self.mixer.canvas_compositor({
            "name": self.node_name, "src": [keyed, *self.edges], "dst": feeds["dirty"],
            "subscriptions": ["", *self.edges], "clock_input": 0,
            "fps": f"{self.mixer.fps_num}/{self.mixer.fps_den}",
            "max_layers": len(layers), "layers": layers, "active_inputs": self._mask(self.on),
            # Program frames carry the scene compositor's per-frame layer metadata;
            # a distinct key keeps it from rearranging the keyer's layers.
            "metadata_key": "dsk_layers_v1",
            "group": self.group,
        }, api=self.api))
        return feeds

    def set(self, request) -> dict:
        """Put *key* on or off air. fade_seconds and curve default to the config's dsk ones;
        0 seconds cuts. Only a key that changes takes this fade: the keyer never restarts
        another key's, and a command that leaves *key* as it is (a cut included) lets its
        running fade finish."""
        key, on = request.get("key"), request.get("on")
        if key not in self.on or not isinstance(on, bool):
            raise ConfigError("dsk needs a known key and a boolean on")
        seconds = key_fade_seconds(request.get("fade_seconds", self.cfg.dsk_fade_seconds), "dsk fade_seconds")
        curve = fade_curve(request.get("curve", self.cfg.dsk_fade_curve), "dsk curve")
        duration_ms = round(seconds * 1000)
        with self.lock:
            changed = self.on[key] != on
            mask = self._mask({**self.on, key: on})
            # self.on changes only once the keyer took the command, so a rejected one reports nothing switched.
            if duration_ms:
                self.avp.executeCommandsFromString(f"node.object.set {self.node_name} fade_inputs " + json.dumps(
                    {"active_inputs": mask, "duration_ms": duration_ms, "curve": curve}))
            else:
                self.avp.executeCommandsFromString(f"node.object.set {self.node_name} active_inputs {json.dumps(mask)}")
                # Like an M/E cut: a key cut changes the picture at once, so the preview receiver
                # should not wait for the next periodic keyframe. A fade changes it gradually.
                if changed and self.mixer.keyframe_node:
                    try:
                        self.avp.executeCommandsFromString(keyframe_command(self.mixer.keyframe_node))
                    except Exception as exc:   # the key already switched; a missed keyframe only delays the preview
                        logging.getLogger(__name__).warning("dsk: keyframe trigger failed: %s", exc)
            self.on[key] = on
            return self.state()

    def state(self) -> list:
        return [{"id": k.id, "source": k.source, "on": self.on[k.id]} for k in self.keys]


def register_dsk_commands(avp, keyer):
    avp.registerControlCommand("mixer.dsk", lambda arg: json.dumps(keyer.set(json.loads(arg))) + "\n", True)
    avp.registerControlCommand("mixer.dsk_status", lambda _arg: json.dumps(keyer.state()) + "\n", True)
