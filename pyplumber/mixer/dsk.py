"""Downstream keyer: up to four alpha browser keys over the finished program.

The keyer runs after transitions and wipes, so a clean feed still carries every
scene and M/E transition and only the keys are missing. It is one compositor
pass clocked by the program (``clock_input``): each program frame renders at
once over every key's frame stamped for that tick, so keying adds no playout
latency and a late browser paint never stalls the program. A key cuts or fades
on and off, each on its own, or with the background transition (Next Transition);
with none visible, the program frame passes through without any GPU work. The
keyer also fades the keyed program to black (FTB): an internal key holding one
black frame over the whole canvas. The clean feed is never faded.
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
        self.ftb_input, self.ftb_edge, self.ftb_on = len(self.keys) + 1, "dsk_ftb", False   # nothing feeds the edge
        self.lock = threading.Lock()
        for key, edge in zip(self.keys, self.edges):
            # Keys subscribe to the shared source like an aux bus: the source
            # never waits for the keyer. Off keys stay subscribed, so a key comes
            # on with the next program frame.
            mixer.add_aux_destination(key.source, edge)

    def _mask(self, on: dict):
        return source_mask_param(1 | sum(1 << (i + 1) for i, k in enumerate(self.keys) if on[k.id])
                                 | (1 << self.ftb_input if self.ftb_on else 0))

    def build(self, program: str, *, clean: bool) -> dict:
        """Key *program*; return the edge of each feed. A clean edge exists only when asked for."""
        feeds = {"dirty": "program_dirty"}
        keyed = program
        if clean:
            keyed, feeds["clean"] = "dsk_program", "program_clean"
            self.avp.addNode(self.api.Split({"name": "split_clean", "src": program, "dst": [feeds["clean"], keyed],
                                             "group": self.group, "on_error": "panic"}))
        for edge in self.edges:
            # A real paint can supersede a held copy at the same timestamp.
            # Retain both until the keyer drains them and selects the newest.
            self.avp.edges.planCapacity(edge, 2)
        w, h = self.cfg.canvas_w, self.cfg.canvas_h
        layers = [{"dst_x": 0, "dst_y": 0, "dst_w": w, "dst_h": h}]
        layers += [{"input": i + 1, "dst_x": k.dst.x, "dst_y": k.dst.y, "dst_w": k.dst.w, "dst_h": k.dst.h,
                    "z": i + 1, "blend": True} for i, k in enumerate(self.keys)]
        # The fade to black covers everything, the keys included.
        layers.append({"input": self.ftb_input, "dst_x": 0, "dst_y": 0, "dst_w": w, "dst_h": h,
                       "z": self.ftb_input, "blend": True})
        self.avp.addNode(self.mixer.canvas_compositor({
            "name": self.node_name, "src": [keyed, *self.edges, self.ftb_edge], "dst": feeds["dirty"],
            "mixer": self.mixer.name, "ftb_input": self.ftb_input,
            "subscriptions": ["", *self.edges, ""], "clock_input": 0,
            "fps": f"{self.mixer.fps_num}/{self.mixer.fps_den}",
            "max_layers": len(layers), "layers": layers, "active_inputs": self._mask(self.on),
            # The keys whose rectangles the keyer may compose alone, by input; the others take the whole canvas.
            "bounded_keys": [i + 1 for i, k in enumerate(self.keys) if k.bounded],
            # Program frames carry the scene compositor's per-frame layer metadata;
            # a distinct key keeps it from rearranging the keyer's layers.
            "metadata_key": "dsk_layers_v1",
            "group": self.group,
        }, keyer=True, api=self.api))
        return feeds

    def set(self, request) -> dict:
        """Put *key* on or off air. fade_seconds and curve default to the config's dsk ones;
        0 seconds cuts. Only a key that changes takes this fade: the keyer never restarts
        another key's, and a command that leaves *key* as it is (a cut included) lets its
        running fade finish. The group form, keys=[...], changes all named keys on one
        program frame; its explicit cuts also finish running envelopes. toggle=True
        resolves against native targets, not a controller's potentially stale copy.
        prepare=True selects the keys that switch with each following background take, on its
        clock (an empty list clears the selection); fade_seconds and curve do not apply to it."""
        key, on = request.get("key"), request.get("on")
        keys = request.get("keys", [key])
        prepare = request.get("prepare", False)
        toggle = request.get("toggle", False)
        if not isinstance(prepare, bool) or not isinstance(toggle, bool):
            raise ConfigError("dsk prepare and toggle must be booleans")
        if (not isinstance(keys, list) or any(not isinstance(k, str) or k not in self.on for k in keys)
                or len(set(keys)) != len(keys) or (not prepare and not keys)
                or (not prepare and not toggle and not isinstance(on, bool))):
            raise ConfigError("dsk needs a known key and a boolean on")
        seconds = key_fade_seconds(request.get("fade_seconds", self.cfg.dsk_fade_seconds), "dsk fade_seconds")
        curve = fade_curve(request.get("curve", self.cfg.dsk_fade_curve), "dsk curve")
        duration_ms = round(seconds * 1000)
        with self.lock:
            self.state()  # another controller or a prepared take may have changed the targets
            if "keys" in request or prepare or toggle:
                affected = source_mask_param(sum(1 << (i + 1) for i, k in enumerate(self.keys) if k.id in keys))
                targets = {**self.on, **{k: not self.on[k] if toggle else on for k in keys}}
                value = {"inputs": affected} if prepare else {"inputs": affected, "duration_ms": duration_ms, "curve": curve,
                                                              "active_inputs": self._mask(targets), "toggle": toggle}
                self.avp.executeCommandsFromString(f"node.object.set {self.node_name} " +
                    ("prepare_inputs " if prepare else "key_changes ") + json.dumps(value))
                if not prepare:
                    self.on = targets
                    if not duration_ms:
                        self._keyframe()
                return self.state()
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
                    self._keyframe()
            self.on[key] = on
            return self.state()

    def ftb(self, request) -> dict:
        """Fade the keyed program to black or back: on, or toggle=True; fade_seconds (0 cuts)
        and curve default to the config's fade ones."""
        on, toggle = request.get("on"), request.get("toggle", False)
        if not isinstance(toggle, bool) or (not toggle and not isinstance(on, bool)):
            raise ConfigError("ftb needs a boolean on, or toggle")
        seconds = key_fade_seconds(request.get("fade_seconds", self.cfg.fade_seconds), "ftb fade_seconds")
        curve = fade_curve(request.get("curve", "linear"), "ftb curve")
        with self.lock:
            self._rendered()
            self.ftb_on = not self.ftb_on if toggle else on
            self.avp.executeCommandsFromString(f"node.object.set {self.node_name} key_changes " + json.dumps(
                {"inputs": source_mask_param(1 << self.ftb_input), "active_inputs": self._mask(self.on),
                 "duration_ms": round(seconds * 1000), "curve": curve}))
            if not seconds:
                self._keyframe()
            return self.ftb_status()

    def _rendered(self) -> dict:
        try:
            rows = {row["input"]: row for row in self.avp.node(self.node_name).getObject("keys")}
        except Exception:
            return {}  # startup/older module: visibility is unknown, never infer it from the target
        if "on" in rows.get(self.ftb_input, {}):
            self.ftb_on = rows[self.ftb_input]["on"]
        return rows

    def ftb_status(self) -> dict:
        """The fade to black as the last program frame rendered it."""
        ftb = dict(self._rendered().get(self.ftb_input, {}))
        ftb.pop("input", None)
        ftb.pop("next_take", None)
        return {"on": self.ftb_on, **ftb}

    def state(self) -> list:
        rendered = self._rendered()
        rows = []
        for i, k in enumerate(self.keys):
            native = dict(rendered.get(i + 1, {}))
            native.pop("input", None)
            if "on" in native:
                self.on[k.id] = native["on"]
            rows.append({"id": k.id, "source": k.source, "on": self.on[k.id], **native})
        return rows

    def _keyframe(self):
        if not self.mixer.keyframe_node:
            return
        try:
            self.avp.executeCommandsFromString(keyframe_command(self.mixer.keyframe_node))
        except Exception as exc:
            logging.getLogger(__name__).warning("dsk: keyframe trigger failed: %s", exc)


def register_dsk_commands(avp, keyer):
    avp.registerControlCommand("mixer.dsk", lambda arg: json.dumps(keyer.set(json.loads(arg))) + "\n", True)
    avp.registerControlCommand("mixer.ftb", lambda arg: json.dumps(keyer.ftb(json.loads(arg))) + "\n", True)
    avp.registerControlCommand("mixer.dsk_status", lambda _arg: json.dumps(keyer.state()) + "\n", True, True)   # polled: not logged
    avp.registerControlCommand("mixer.ftb_status", lambda _arg: json.dumps(keyer.ftb_status()) + "\n", True, True)
