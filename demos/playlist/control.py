"""Playlist control protocol: JSON over AVPlumber's control server.

The backend registers ``playlist.status`` and ``playlist.<verb>`` commands.
The TUI talks to them through ``RemoteClient``; ``--dry-run`` uses
``LocalClient`` on an in-memory controller with the same interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from avpmixer.control import AvpConnection
from playlist import (Clip, ElementMode, PlaylistController, PlaylistMode, Transition,
                      TransportState, now_ms)

VERBS = ("play", "pause", "stop", "toggle", "next", "prev", "select", "take", "hold",
         "park", "mode", "transition", "end_mode", "enable", "add", "remove", "move", "edit")


def clip_to_json(clip: Clip) -> Dict[str, Any]:
    return {"id": clip.item_id, "name": clip.name, "url": clip.url, "cue_in": clip.play_from_ms,
            "cue_out": clip.play_to_ms, "duration": clip.duration_ms, "speed": clip.speed,
            "end": clip.element_mode.value, "disabled": clip.disabled}


def clip_from_json(data: Dict[str, Any], item_id: Optional[str] = None) -> Clip:
    kwargs = dict(url=data["url"], name=data.get("name", ""),
                  element_mode=ElementMode(data.get("end", "PlayToEnd")),
                  play_from_ms=int(data.get("cue_in") or 0),
                  play_to_ms=None if data.get("cue_out") in (None, "") else int(data["cue_out"]),
                  duration_ms=None if data.get("duration") in (None, "") else int(data["duration"]),
                  disabled=bool(data.get("disabled", False)), speed=float(data.get("speed") or 1.0))
    if item_id or data.get("id"):
        kwargs["item_id"] = item_id or data["id"]
    return Clip(**kwargs)


def status_json(ctl: PlaylistController, now: int) -> Dict[str, Any]:
    s = ctl.status(now)
    return {
        "now_ms": now, "transport": s.transport.value, "mode": s.mode.value,
        "transition": ctl.transition.value, "transition_ms": ctl.transition_ms,
        "selected": s.selected_index, "active": s.active_index, "pending": s.pending_index,
        "next": s.next_index, "position_ms": s.position_ms, "end_ms": s.end_ms,
        "next_at_ms": s.next_at_ms, "output_alive": s.output_alive,
        "error": s.error, "error_index": s.error_index,
        "clips": [dict(clip_to_json(c), state=ctl.element_state(i).value)
                  for i, c in enumerate(ctl.clips)],
    }


def apply_verb(ctl: PlaylistController, verb: str, arg: Dict[str, Any]) -> None:
    """Dispatch one control verb onto the controller (caller holds the lock)."""
    i = arg.get("index")
    if verb == "play":
        ctl.play()
    elif verb == "pause":
        ctl.pause()
    elif verb == "stop":
        ctl.stop()
    elif verb == "toggle":
        ctl.toggle()
    elif verb == "next":
        ctl.next()
    elif verb == "prev":
        ctl.prev()
    elif verb == "select":
        ctl.select(i)
    elif verb == "take":
        ctl.element_play(i)
    elif verb == "hold":
        ctl.element_pause(i)
    elif verb == "park":
        ctl.element_stop(i)
    elif verb == "mode":
        ctl.set_mode(PlaylistMode(arg["mode"]))
    elif verb == "transition":
        ctl.set_transition(Transition(arg["transition"]), arg.get("duration_ms"))
    elif verb == "end_mode":
        ctl.set_element_mode(i, ElementMode(arg["end"]), arg.get("duration"))
    elif verb == "enable":
        ctl.set_disabled(i, not bool(arg["enabled"]))
    elif verb == "add":
        ctl.insert_clip(len(ctl.clips) if i is None else i, clip_from_json(arg["clip"]))
    elif verb == "remove":
        ctl.remove_clip(i)
    elif verb == "move":
        ctl.reorder_clip(i, arg["to"])
    elif verb == "edit":
        ctl.replace_clip(i, clip_from_json(arg["clip"], ctl.clips[i].item_id))
    else:
        raise ValueError(f"unknown playlist verb {verb!r}")


@dataclass
class Snapshot:
    """Typed view of ``status_json`` for the TUI."""
    data: Dict[str, Any]

    def __getattr__(self, name):
        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def clips(self) -> List[Dict[str, Any]]:
        return self.data["clips"]

    @property
    def playing(self) -> bool:
        return self.data["transport"] == TransportState.PLAYING.value

    def seconds_to_next(self) -> Optional[float]:
        if self.data.get("next_at_ms") is None:
            return None
        return max(0.0, (self.data["next_at_ms"] - self.data["now_ms"]) / 1000)


class LocalClient:
    """Same surface as RemoteClient, driving a controller in this process."""

    def __init__(self, controller: PlaylistController, clock=now_ms):
        self.ctl = controller
        self.clock = clock
        self.connected = True

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def status(self) -> Snapshot:
        now = self.clock()
        self.ctl.poll(now)
        return Snapshot(status_json(self.ctl, now))

    async def send(self, verb: str, **arg) -> None:
        try:
            apply_verb(self.ctl, verb, arg)
        except (ValueError, IndexError) as exc:
            self.ctl.set_error(str(exc))


class RemoteClient:
    def __init__(self, host: str, port: int):
        self.conn = AvpConnection(host, port)

    @property
    def connected(self) -> bool:
        return self.conn.connected

    async def connect(self) -> None:
        await self.conn.connect()

    async def disconnect(self) -> None:
        await self.conn.disconnect()

    async def status(self) -> Snapshot:
        content = await self.conn.command("playlist.status")
        if not content:
            raise RuntimeError("playlist.status returned no content")
        return Snapshot(json.loads(content))

    async def send(self, verb: str, **arg) -> None:
        await self.conn.command(f"playlist.{verb} {json.dumps(arg, separators=(',', ':'))}")
