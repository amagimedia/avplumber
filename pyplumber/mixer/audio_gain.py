"""Per-source audio gain: a persistent operator volume, independent of AFV's crossfade gain.

Multiplies into audio_mixer's own gain (set_source_gain, src/nodes/audio_mixer.cpp), so
it needs no edges or nodes of its own and never changes the mixed sources' format.
"""
from __future__ import annotations

import json
import math
import threading

from .config import ConfigError

DB_MIN, DB_MAX = -70.0, 10.0


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(gain: float) -> float:
    return 20.0 * math.log10(gain) if gain > 1e-6 else DB_MIN


class AudioGainController:
    def __init__(self, avp, node_name: str, source_ids: list):
        self.avp, self.node_name = avp, node_name
        self.gains_db = {sid: 0.0 for sid in source_ids}
        self.lock = threading.Lock()

    def _rendered(self) -> None:
        # Best-effort: a mid-startup node degrades to the cache, like dsk.py's _rendered().
        try:
            status = self.avp.node(self.node_name).getObject("status")
        except Exception:
            return
        for s in status.get("sources", []):
            if s["id"] in self.gains_db and "gain" in s:
                self.gains_db[s["id"]] = linear_to_db(s["gain"])

    def set(self, request) -> list:
        gains = request.get("gains")
        if not isinstance(gains, dict) or not gains:
            raise ConfigError("audio_gain needs a non-empty gains object")
        if any(sid not in self.gains_db for sid in gains):
            raise ConfigError("audio_gain: unknown source id")
        with self.lock:
            self._rendered()   # another controller may have changed a gain since the last poll
            for source_id, db in gains.items():
                if not isinstance(db, (int, float)):
                    raise ConfigError("audio_gain: gain_db must be a number")
                db = max(DB_MIN, min(DB_MAX, float(db)))
                if self.gains_db[source_id] == db:
                    continue
                self.avp.executeCommandsFromString(
                    f"node.object.set {self.node_name} set_source_gain " +
                    json.dumps({"source": source_id, "gain": db_to_linear(db)}))
                self.gains_db[source_id] = db
            return self.state()

    def state(self) -> list:
        return [{"id": sid, "gain_db": db} for sid, db in self.gains_db.items()]


def register_audio_gain_commands(avp, controller: AudioGainController):
    avp.registerControlCommand("mixer.audio_gain", lambda arg: json.dumps(controller.set(json.loads(arg))) + "\n", True)
    avp.registerControlCommand("mixer.audio_gain_status", lambda _arg: json.dumps(controller.state()) + "\n", True, True)
