"""Safe per-source input-chain restart and playout primitives.

A stopping input ends its stream with an end-of-stream frame. Left in a group queue it ends the
node that reads it after the next start; passed on, it can end the mixer's own nodes behind the
compositors. Terminal nodes stop first so the frame stays inside, then every node stops, then the
group's own edges are cleared before params change and the group starts again.

``mixer.source.media`` / ``prepare`` / ``resume_at`` need ``MixerOptions.playback_control`` so each
pad has a pause team and a sync team. Pause and seek retry while the chain has no frame yet
("Can not determine current time"); that path uses a raising control runner (loopback TCP when a
``control_port`` is set, else an injected ``run_command``). Playhead position is not reported by
the engine — clients keep their own estimate.
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .config import ConfigError

# Chains built by ``build_input`` (file / live video and NDI). Browser, raw, MXL and VMM pads need
# other primitives (navigate, unpack params, IPC) and refuse ``mixer.source.set``.
SETTABLE_KINDS = frozenset({"video", "ndi"})
MEDIA_ACTIONS = frozenset({"play", "pause", "seek", "stop", "restart"})
# Pause / seek need a shown frame; just after group.start there may be none yet.
_CURRENT_TIME_ERR = "Can not determine current time"


@dataclass(frozen=True)
class SourcePad:
    """One show source's input group, addressed by show source id."""
    id: str
    index: int
    kind: str
    group: str
    input_node: str
    sync_team: Optional[str] = None
    pause_team: Optional[str] = None
    continuous_loop: bool = True


def group_shape(nodes: list[dict]) -> tuple[list[str], list[str]]:
    """From the parameters of a group's *nodes*: names of its last nodes (their output leaves the
    group) and the edges between its own nodes."""
    def names(params: dict, key: str) -> set[str]:
        value = params.get(key) or []
        return {value} if isinstance(value, str) else set(value)

    produced = set().union(*(names(n, "dst") for n in nodes))
    consumed = set().union(*(names(n, "src") for n in nodes))
    last = [n["name"] for n in nodes if names(n, "dst") and not names(n, "dst") & consumed]
    return last, sorted(produced & consumed)


def stop_group(avp, group: str, timeout: float = 5.0) -> None:
    """Stop a group's nodes so that it can start again: nothing of the old stream stays in it or
    leaves it."""
    handle = avp.group(group)
    nodes = list(handle.sortedNodes)
    last, edges = group_shape([{**dict(node.parameters), "name": node.name} for node in nodes])
    for node in nodes:
        if node.name in last:
            node.stopAndWait()
    handle.stopNodes()
    deadline = time.monotonic() + timeout
    while any(node.isWorking for node in nodes):
        if time.monotonic() > deadline:
            raise RuntimeError(f"{group} did not stop within {timeout:g} s")
        time.sleep(0.005)
    for edge in edges:
        avp.getEdge(edge).clear()


def source_pads(sources, *, external: Optional[set[str]] = None,
                playback_control: bool = False) -> dict[str, SourcePad]:
    """Pads for every show source, named the way ``build_application`` wires them."""
    external = external or set()
    pads = {}
    for index, source in enumerate(sources):
        group = f"input_{index}"
        # Sync team: audio / NDI always; every build_input chain when playback_control is on.
        sync = (f"input_{index}_av_sync"
                if playback_control or source.kind == "ndi" or getattr(source, "has_audio", False)
                else None)
        pads[source.id] = SourcePad(
            id=source.id, index=index, kind=source.kind if source.id not in external else "vmm",
            group=group, input_node=f"input_{index}",
            sync_team=sync,
            pause_team=(f"input_{index}_pause" if playback_control else None),
            continuous_loop=True)
    return pads


def probe_duration_ms(url: str, *, timeout: float = 20.0) -> Optional[int]:
    """Clip length in ms via ffprobe, or None when the URL has no duration (live) or probe fails."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", url],
            capture_output=True, text=True, timeout=timeout)
        text = out.stdout.strip()
        if not text or text.upper() == "N/A":
            return None
        return int(float(text) * 1000)
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def _tcp_command(port: int, text: str, *, host: str = "127.0.0.1", timeout: float = 10.0) -> Optional[str]:
    """One raising exchange on the control port (``executeCommandsFromString`` only logs failures)."""
    conn = socket.create_connection((host, port), timeout=timeout)
    try:
        stream = conn.makefile("rw", encoding="utf-8", newline="\n")
        greeting = stream.readline()
        if not greeting.startswith("100"):
            raise RuntimeError(f"avplumber control: unexpected greeting {greeting.strip()!r}")
        stream.write(text + "\n")
        stream.flush()
        head = stream.readline()
        if not head:
            raise ConnectionError("avplumber closed the control connection")
        code = int(head.split(" ", 1)[0])
        body = None
        if code == 201:
            lines = []
            while (line := stream.readline()) not in ("\n", "\r\n", ""):
                lines.append(line)
            body = "".join(lines)
        if not 200 <= code < 300:
            raise RuntimeError(f"{text.split(' ', 1)[0]}: {head.strip()}")
        return body
    finally:
        conn.close()


def _is_position(value) -> bool:
    """A count of milliseconds: an int, and not a bool (which Python counts among them)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _url_and_loop(request: dict, who: str) -> tuple[str, bool]:
    url = request.get("url")
    if not isinstance(url, str) or not url:
        raise ConfigError(f"{who} needs a non-empty url")
    if "loop" not in request:
        raise ConfigError(f"{who} needs a boolean loop")
    loop = request["loop"]
    if not isinstance(loop, bool):
        raise ConfigError(f"{who}: loop must be a boolean")
    return url, loop


def _no_other_fields(request: dict, who: str, *fields: str) -> None:
    unknown = set(request) - {"mixer", "id", *fields}
    if unknown:
        raise ConfigError(f"{who}: unsupported fields {sorted(unknown)}")


class SourceController:
    def __init__(self, avp, pads: dict[str, SourcePad], *, mixer: str = "mixer",
                 control_port: int = 0,
                 run_command: Optional[Callable[[str], Optional[str]]] = None,
                 probe: Callable[[str], Optional[int]] = probe_duration_ms):
        self.avp, self.pads, self.mixer = avp, pads, mixer
        self.control_port = control_port
        self._run_command = run_command
        self._probe = probe
        self.lock = threading.Lock()
        # Pause and seek retries serialize: a failed pause can leave the team needing resume first.
        self._playback_lock = threading.Lock()

    def set(self, request: dict) -> dict:
        pad = self._pad(request, "source.set")
        url, loop = _url_and_loop(request, "source.set")
        # Optional fields the chain does not yet retarget (audio track, deinterlace) stay for a
        # follow-up; refuse them so a client does not think they took effect.
        _no_other_fields(request, "source.set", "url", "loop")

        with self.lock:
            self._set_unlocked(pad, url, loop)
        return {"id": pad.id, "url": url, "loop": loop}

    def prepare(self, request: dict) -> dict:
        """Load a URL, pause on the cue-in frame, report duration when ffprobe knows it."""
        pad = self._pad(request, "source.prepare", playback=True)
        url, loop = _url_and_loop(request, "source.prepare")
        cue = request.get("cue_in_ms", 0)
        if not _is_position(cue):
            raise ConfigError("source.prepare: cue_in_ms must be a non-negative int")
        _no_other_fields(request, "source.prepare", "url", "loop", "cue_in_ms")

        with self.lock:
            self._set_unlocked(pad, url, loop)
            duration_ms = self._probe(url)
            # Pause on the first frame, then seek to cue-in; the pause node holds that frame
            # (pass_on_seek off) until resume … at.
            self._playback_command(f"pause {pad.pause_team} now", wait_s=10.0)
            self._playback_command(f"seek {pad.sync_team} now {cue}", wait_s=5.0)
        return {"id": pad.id, "url": url, "loop": loop, "cue_in_ms": cue,
                "duration_ms": duration_ms}

    def media(self, request: dict) -> dict:
        pad = self._pad(request, "source.media", playback=True)
        action = request.get("action")
        if action not in MEDIA_ACTIONS:
            raise ConfigError(f"source.media: action must be one of {sorted(MEDIA_ACTIONS)}")
        position_ms = request.get("position_ms")
        if action == "seek":
            if not _is_position(position_ms):
                raise ConfigError("source.media: seek needs a non-negative int position_ms")
        elif position_ms is not None:
            raise ConfigError("source.media: position_ms is only valid with action seek")
        _no_other_fields(request, "source.media", "action", "position_ms")

        pause, team = pad.pause_team, pad.sync_team
        with self.lock:
            if action == "play":
                self._control(f"resume {pause}")
            elif action == "pause":
                self._playback_command(f"pause {pause} now")
            elif action in ("seek", "restart", "stop"):
                target = 0 if action != "seek" else position_ms
                self._playback_command(f"seek {team} now {target}")
                if action == "restart":
                    self._control(f"resume {pause}")
                elif action == "stop":
                    self._playback_command(f"pause {pause} now")
        reply = {"id": pad.id, "action": action}
        if action == "seek":
            reply["position_ms"] = position_ms
        return reply

    def resume_at(self, request: dict) -> dict:
        """Schedule ``resume <pause_team> at <at_ms>`` (wallclock ms, mixer start_pts domain)."""
        pad = self._pad(request, "source.resume_at", playback=True)
        at_ms = request.get("at_ms")
        if not _is_position(at_ms):
            raise ConfigError("source.resume_at needs a non-negative int at_ms")
        position_ms = request.get("position_ms")
        if position_ms is not None and not _is_position(position_ms):
            raise ConfigError("source.resume_at: position_ms must be a non-negative int when set")
        _no_other_fields(request, "source.resume_at", "at_ms", "position_ms")

        with self.lock:
            self._control(f"resume {pad.pause_team} at {at_ms}")
        reply = {"id": pad.id, "at_ms": at_ms}
        if position_ms is not None:
            # Echo only: the engine does not track playhead; clients keep their own clock.
            reply["position_ms"] = position_ms
        return reply

    def _pad(self, request: dict, who: str, *, playback: bool = False) -> SourcePad:
        """The pad a request names, for this mixer; with *playback*, one that can pause and seek."""
        mixer = request.get("mixer", self.mixer)
        if mixer != self.mixer:
            raise ConfigError(f"unknown mixer: {mixer}")
        source_id = request.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise ConfigError(f"{who} needs a non-empty id")
        pad = self.pads.get(source_id)
        if pad is None:
            raise ConfigError(f"{who}: unknown source id {source_id!r}")
        if pad.kind not in SETTABLE_KINDS:
            raise ConfigError(f"{who}: source {source_id!r} kind {pad.kind!r} is unsupported "
                              f"(need one of {sorted(SETTABLE_KINDS)})")
        if playback and not (pad.pause_team and pad.sync_team):
            raise ConfigError(f"{who}: source {source_id!r} has no pause/sync teams "
                              f"(build with MixerOptions.playback_control=True)")
        return pad

    def _set_unlocked(self, pad: SourcePad, url: str, loop: bool) -> None:
        continuous = bool(loop and pad.continuous_loop)
        stop_group(self.avp, pad.group)
        self._param(pad.input_node, "url", url)
        self._param(pad.input_node, "loop", loop)
        self._param(pad.input_node, "loop_continuous_ts", continuous)
        # The chain paces on one realtime team; the new input starts its own timeline, so the
        # team starts again too (otherwise it waits on the old one).
        if pad.sync_team:
            self.avp.executeCommandsFromString(f"realtime.team.reset {pad.sync_team}")
        if pad.pause_team:
            self.avp.executeCommandsFromString(f"resume {pad.pause_team}")
        self.avp.executeCommandsFromString(f"group.start {pad.group}")

    def _param(self, node: str, key: str, value: Any) -> None:
        self.avp.executeCommandsFromString(f"node.param.set {node} {key} {json.dumps(value)}")

    def _control(self, text: str) -> Optional[str]:
        """Run a control line; raises on refusal when a raising runner is available."""
        if self._run_command is not None:
            return self._run_command(text)
        if self.control_port:
            return _tcp_command(self.control_port, text)
        self.avp.executeCommandsFromString(text)
        return None

    def _playback_command(self, text: str, wait_s: float = 3.0) -> None:
        """Pause and seek need a frame of the clip to have been shown; just after a source starts
        there may be none yet, so they are tried again for a moment.

        A pause that fails that way has already paused its team on avplumber builds before the
        pause fix: the input, which waits on the team, stops reading and the frame never comes. So
        the team is resumed before the pause is tried again."""
        words = text.split()
        team = words[1] if len(words) > 2 and words[0] == "pause" else None
        with self._playback_lock:
            deadline = time.monotonic() + wait_s
            while True:
                try:
                    self._control(text)
                    return
                except RuntimeError as exc:
                    if _CURRENT_TIME_ERR not in str(exc) or time.monotonic() > deadline:
                        raise
                    if team is not None:
                        self._control(f"resume {team}")
                time.sleep(0.1)


def register_source_commands(avp, controller: SourceController) -> None:
    def handle(method):
        return lambda arg: json.dumps(method(json.loads(arg))) + "\n"

    avp.registerControlCommand("mixer.source.set", handle(controller.set), True)
    avp.registerControlCommand("mixer.source.prepare", handle(controller.prepare), True)
    avp.registerControlCommand("mixer.source.media", handle(controller.media), True)
    avp.registerControlCommand("mixer.source.resume_at", handle(controller.resume_at), True)
