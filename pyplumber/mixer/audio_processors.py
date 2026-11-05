"""The show document's optional ``audio`` section: extra instructions for the channels audio_mixer
creates on its own.

audio_mixer makes an input channel per ``has_audio`` source (named by the source id) and an output
channel per bus (``program``, and each AUX bus with an ``audio_port``, named by the bus id). This
section creates none of them: an entry names one of those channels and adds to it. For now that is
``processors`` on an output, and the one processor type is a sidechain ``ducker``::

    "audio": {"channels": [
      {"id": "program", "type": "output", "processors": [
        {"id": "program-duck", "type": "ducker",
         "target": ["rene", "genaro"], "sidechain": ["denise", "leo"], "sidechain_tap": "post",
         "threshold_db": -40, "depth_db": -20, "attack_ms": 10, "hold_ms": 300, "release_ms": 500}]}]}

A ducker lowers its ``target`` inputs on that output by ``depth_db`` while the sum of its
``sidechain`` inputs peaks at or above ``threshold_db``, and for ``hold_ms`` after. ``post`` hears
the sidechain as the output mixes it (audio-follow-video, operator gain), ``pre`` by the operator
gain alone, whether it is on air or not. Its parameters change live through ``mixer.audio_processor``
(set_processor, src/nodes/audio_mixer.cpp); which processors exist is fixed by the document.
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .config import ConfigError

CHANNEL_TYPES = ("input", "output")
SIDECHAIN_TAPS = ("post", "pre")
# (low, high) of each live parameter; a value outside is an error, not clamped, so a typo shows.
DUCKER_RANGES = {"threshold_db": (-90.0, 0.0), "depth_db": (-90.0, 0.0), "attack_ms": (0.0, 5000.0),
                 "hold_ms": (0.0, 10000.0), "release_ms": (0.0, 20000.0)}


@dataclass(frozen=True)
class Ducker:
    id: str
    target: Tuple[str, ...]
    sidechain: Tuple[str, ...]
    sidechain_tap: str = "post"
    enabled: bool = True
    threshold_db: float = -40.0
    depth_db: float = -20.0
    attack_ms: float = 10.0
    hold_ms: float = 300.0
    release_ms: float = 500.0
    type: str = "ducker"

    def document(self) -> Dict[str, Any]:
        """As the show document writes it."""
        d = asdict(self)
        return {"id": d.pop("id"), "type": d.pop("type"), "target": list(self.target),
                "sidechain": list(self.sidechain), **{k: v for k, v in d.items() if k not in ("target", "sidechain")}}


@dataclass(frozen=True)
class AudioChannel:
    id: str
    type: str
    processors: Tuple[Ducker, ...] = ()


def ducker_params(obj: Dict[str, Any], where: str) -> Dict[str, Any]:
    """The live parameters `obj` sets, checked: `enabled` and the DUCKER_RANGES ones."""
    out: Dict[str, Any] = {}
    if "enabled" in obj:
        if not isinstance(obj["enabled"], bool):
            raise ConfigError(f"{where}: enabled must be a boolean")
        out["enabled"] = obj["enabled"]
    for key, (low, high) in DUCKER_RANGES.items():
        if key not in obj:
            continue
        value = obj[key]
        # A bool passes isinstance(x, int); a non-finite number is not JSON.
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or not low <= value <= high:
            raise ConfigError(f"{where}: {key} must be a number from {low:g} to {high:g}")
        out[key] = float(value)
    return out


def _ids(value: Any, where: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise ConfigError(f"{where} must be a non-empty list of input channel ids")
    if len(set(value)) != len(value):
        raise ConfigError(f"{where} names a channel twice")
    return tuple(value)


def _parse_ducker(p: Any, where: str) -> Ducker:
    if not isinstance(p, dict) or not isinstance(p.get("id"), str) or not p["id"]:
        raise ConfigError(f"{where}: an object with an id required")
    if p.get("type") != "ducker":
        raise ConfigError(f"{where}: type must be ducker")
    known = {"id", "type", "target", "sidechain", "sidechain_tap", "enabled", *DUCKER_RANGES}
    unknown = sorted(set(p) - known)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {', '.join(unknown)}")
    target, sidechain = _ids(p.get("target"), f"{where}.target"), _ids(p.get("sidechain"), f"{where}.sidechain")
    both = sorted(set(target) & set(sidechain))
    if both:
        raise ConfigError(f"{where}: {', '.join(both)} both target and sidechain")
    tap = p.get("sidechain_tap", "post")
    if tap not in SIDECHAIN_TAPS:
        raise ConfigError(f"{where}: sidechain_tap must be one of {', '.join(SIDECHAIN_TAPS)}")
    return Ducker(p["id"], target, sidechain, tap, **ducker_params(p, where))


def parse_audio(raw: Any) -> Tuple[AudioChannel, ...]:
    """The section's shape. Whether its channels exist is resolve()'s, once audio_mixer's are known."""
    if raw is None:
        return ()
    if not isinstance(raw, dict) or set(raw) - {"channels"}:
        raise ConfigError("audio must be an object with channels")
    entries = raw.get("channels", [])
    if not isinstance(entries, list):
        raise ConfigError("audio.channels must be a list")
    channels: List[AudioChannel] = []
    for i, c in enumerate(entries):
        where = f"audio.channels[{i}]"
        if not isinstance(c, dict) or not isinstance(c.get("id"), str) or c.get("type") not in CHANNEL_TYPES:
            raise ConfigError(f"{where}: id and a type of {' or '.join(CHANNEL_TYPES)} required")
        if set(c) - {"id", "type", "processors"}:
            raise ConfigError(f"{where}: unknown keys {', '.join(sorted(set(c) - {'id', 'type', 'processors'}))}")
        if any(ch.id == c["id"] and ch.type == c["type"] for ch in channels):
            raise ConfigError(f"{where}: {c['type']} channel {c['id']} listed twice")
        procs = c.get("processors", [])
        if not isinstance(procs, list):
            raise ConfigError(f"{where}.processors must be a list")
        if procs and c["type"] != "output":
            raise ConfigError(f"{where}: processors are only on output channels for now")
        parsed = tuple(_parse_ducker(p, f"{where}.processors[{j}]") for j, p in enumerate(procs))
        if len({p.id for p in parsed}) != len(parsed):
            raise ConfigError(f"{where}: processor ids must be unique")
        channels.append(AudioChannel(c["id"], c["type"], parsed))
    return tuple(channels)


def resolve(channels: Sequence[AudioChannel], inputs: Sequence[str], outputs: Sequence[str]) -> Dict[str, list]:
    """audio_mixer's `processors` parameter, {output id: [processor, ...]}, for the channels it
    created: `inputs` (source ids) and `outputs` (bus ids). A channel the section names that does
    not exist is an error: a typo must not quietly do nothing."""
    created = {"input": set(inputs), "output": set(outputs)}
    out: Dict[str, list] = {}
    for ch in channels:
        if ch.id not in created[ch.type]:
            raise ConfigError(f"audio.channels: no {ch.type} channel {ch.id!r} (the mixer has "
                              f"{', '.join(sorted(created[ch.type])) or 'none'})")
        for p in ch.processors:
            missing = sorted(set(p.target + p.sidechain) - created["input"])
            if missing:
                raise ConfigError(f"audio.channels {ch.id} processor {p.id}: no input channel {', '.join(missing)}")
        if ch.processors:
            out[ch.id] = [p.document() for p in ch.processors]
    return out


def processors_settings(channels: Sequence[AudioChannel]) -> List[Dict[str, Any]]:
    """The processors as pages list them: one row per processor, with its output."""
    return [{"output": ch.id, **p.document()} for ch in channels for p in ch.processors]


class AudioProcessorController:
    """Live parameters of the document's processors, sent to audio_mixer's set_processor."""

    def __init__(self, avp, node_name: str, channels: Sequence[AudioChannel]):
        self.avp, self.node_name = avp, node_name
        self.rows = {(row["output"], row["id"]): row for row in processors_settings(channels)}
        self.gain_db = {key: 0.0 for key in self.rows}
        self.lock = threading.Lock()

    def _rendered(self) -> None:
        # Best-effort: a mid-startup node degrades to the cache, like audio_gain's.
        try:
            status = self.avp.node(self.node_name).getObject("status")
        except Exception:
            return
        for bus in status.get("buses", []):
            for p in bus.get("processors", []):
                key = (bus.get("id"), p.get("id"))
                if key in self.rows:
                    self.rows[key].update({k: p[k] for k in ("enabled", *DUCKER_RANGES) if k in p})
                    self.gain_db[key] = p.get("gain_db", 0.0)

    def set(self, request: Any) -> list:
        if not isinstance(request, dict):
            raise ConfigError("audio_processor needs an object")
        key = (request.get("output"), request.get("id"))
        if key not in self.rows:
            raise ConfigError("audio_processor: unknown output or processor id")
        params = ducker_params({k: v for k, v in request.items() if k not in ("output", "id")}, "audio_processor")
        if not params:
            raise ConfigError("audio_processor: nothing to set")
        with self.lock:
            self.avp.executeCommandsFromString(
                f"node.object.set {self.node_name} set_processor " +
                json.dumps({"output": key[0], "id": key[1], **params}))
            self.rows[key].update(params)
            return self._state()

    def state(self) -> list:
        with self.lock:
            self._rendered()
            return self._state()

    def _state(self) -> list:
        return [{**row, "gain_db": self.gain_db[key]} for key, row in self.rows.items()]


def register_audio_processor_commands(avp, controller: AudioProcessorController):
    avp.registerControlCommand("mixer.audio_processor", lambda arg: json.dumps(controller.set(json.loads(arg))) + "\n", True)
    avp.registerControlCommand("mixer.audio_processor_status", lambda _arg: json.dumps(controller.state()) + "\n", True, True)
