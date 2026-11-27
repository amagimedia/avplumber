"""Safe per-source input-chain restart and playout primitives.

A stopping input ends its stream with an end-of-stream frame. Left in a group queue it ends the
node that reads it after the next start; passed on, it can end the mixer's own nodes behind the
compositors. Terminal nodes stop first so the frame stays inside, then every node stops, then the
group's own edges are cleared before params change and the group starts again.

``mixer.source.media`` / ``prepare`` / ``resume_at`` need ``MixerOptions.playback_control`` so each
video source's pad has a pause team and a sync team. Pause and seek retry while the chain has no
frame yet ("Can not determine current time"); to see that refusal their lines go through a runner
that raises (``control_command`` on the instance's control port, or an injected ``run_command``).
Playhead position is not reported by the engine — clients keep their own estimate.
"""
from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from .config import ConfigError
from .control import AvpConnection, AvpProtocolError

# Pads whose head node is ``input_<index>`` with a retargetable ``url`` / ``loop``. Browser needs
# navigate (dma-browser); VMM (cuda_ipc) refuses ``mixer.source.set``.
SETTABLE_KINDS = frozenset({"video", "ndi", "v210", "nv12", "p010", "mxl"})
# Pause and seek are input_rec's: a file, or a stream it can seek. An NDI feed is read by the plain
# input node, which waits on no pause team and has no position to seek to. Raw and MXL have none.
PLAYBACK_KINDS = frozenset({"video"})
# Live MXL reader URL as ``config._mxl_flow`` builds it: absolute domain path and a flow id.
_MXL_URL = re.compile(r"^mxl://(/[^?]*)\?id=(.+)$")
MEDIA_ACTIONS = frozenset({"play", "pause", "seek", "stop", "restart"})
# Pause / seek need a shown frame; just after group.start there may be none yet. The control
# protocol has no error codes: this text, from RealTimeTeam (src/RealTimeTeam.hpp), is the only
# sign of that refusal, so a change of its wording there must be made here too.
_CURRENT_TIME_ERR = "Can not determine current time"
# How long each step of a source command may take. A command answers when its work is done, so
# these bound the answer: COMMAND_BUDGET_S is the longest, prepare's, whose probe runs beside the rest.
STOP_TIMEOUT_S = 5.0          # the source's group to stop
FIRST_FRAME_WAIT_S = 10.0     # the pause after a start: the input opens the URL and shows a frame
SEEK_WAIT_S = 5.0
PROBE_TIMEOUT_S = 20.0        # ffprobe opens the URL itself
COMMAND_BUDGET_S = max(PROBE_TIMEOUT_S, STOP_TIMEOUT_S + FIRST_FRAME_WAIT_S + SEEK_WAIT_S)


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


def stop_group(avp, group: str, timeout: float = STOP_TIMEOUT_S) -> None:
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


def chain_teams(index: int, source, *, playback_control: bool) -> tuple[Optional[str], Optional[str]]:
    """The (realtime sync team, pause team) of the ``build_input`` chain of show source *index*:
    what ``build_application`` wires and ``source_pads`` addresses. A sync team where audio or
    NDI already share one (lip-sync, live backlog), and on every chain *playback_control* gives
    playback (``PLAYBACK_KINDS``), which also has the pause team."""
    playback = playback_control and source.kind in PLAYBACK_KINDS
    sync = f"input_{index}_av_sync" if playback or source.kind == "ndi" or source.has_audio else None
    return sync, (f"input_{index}_pause" if playback else None)


def continuous_loop_for(kind: str, *, raw_upload: str = "hwupload") -> bool:
    """Whether ``loop_continuous_ts`` follows ``loop`` on this kind's head node.

    ``nv12``/``p010`` with the default ``hwupload`` path run ``setpts=N*…`` and were built with
    ``loop_continuous_ts`` false; every other settable kind stamps continuous timestamps when
    looping (direct/pinned raw, v210, video, NDI)."""
    return not (kind in ("nv12", "p010") and raw_upload == "hwupload")


def source_pads(sources, *, external: Optional[set[str]] = None,
                playback_control: bool = False, raw_upload: str = "hwupload") -> dict[str, SourcePad]:
    """Pads for every show source, named the way ``build_application`` wires them. Only a
    ``build_input`` chain has teams. *raw_upload* is the show's ``canvas.raw_upload``."""
    external = external or set()
    pads = {}
    for index, source in enumerate(sources):
        kind = source.kind if source.id not in external else "vmm"
        # Teams exist only on ``build_input`` chains (video / NDI); raw pads pace without them.
        sync, pause = (chain_teams(index, source, playback_control=playback_control)
                       if kind in ("video", "ndi") else (None, None))
        pads[source.id] = SourcePad(
            id=source.id, index=index, kind=kind, group=f"input_{index}", input_node=f"input_{index}",
            sync_team=sync, pause_team=pause, continuous_loop=continuous_loop_for(kind, raw_upload=raw_upload))
    return pads


def probe_duration_ms(url: str, *, timeout: float = PROBE_TIMEOUT_S) -> Optional[int]:
    """Clip length in ms via ffprobe, or None when the URL has no duration (live) or probe fails."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        # After -i: a client names the URL, and one starting with "-" would be read as an option.
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", "-i", url],
            capture_output=True, text=True, timeout=timeout)
        text = out.stdout.strip()
        if not text or text.upper() == "N/A":
            return None
        return int(float(text) * 1000)
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def control_command(port: int, text: str, *, timeout: float = 10.0) -> Optional[str]:
    """One exchange on this instance's own control port, for a line whose refusal the caller
    acts on: ``executeCommandsFromString`` only logs it. A refusal raises RuntimeError with the
    command's name and the engine's status line.

    A control command's handler may call it: the server runs every connection on a thread of
    its own, and the mixer enables it on the default address, loopback. Not for a thread that
    runs an event loop."""
    async def exchange():
        connection = AvpConnection("127.0.0.1", port)
        await asyncio.wait_for(connection.connect(), timeout)
        try:
            return await connection.command(text, timeout)
        finally:
            await connection.disconnect()

    try:
        return asyncio.run(exchange())
    except AvpProtocolError as exc:
        raise RuntimeError(f"{text.split(' ', 1)[0]}: {exc}") from None


_SCHEME_CHARS = re.compile(r"[A-Za-z0-9+.-]*")   # libavformat's URL_SCHEME_CHARS


def url_protocol(url: str) -> str:
    """The name of the protocol FFmpeg opens *url* with: "file" for a path. libavformat answers
    this with avio_find_protocol_name, which has no Python binding here; this is its rule
    (url_find_protocol in libavformat/avio.c) for a host without DOS paths. The name is the whole
    scheme: "crypto+https" is not "https"."""
    end = _SCHEME_CHARS.match(url).end()
    if url[end:end + 1] == ":" or (url.startswith("subfile,") and ":" in url[end + 1:]):
        return url[:end]
    return "file"


class SourceUrls:
    """Which URLs ``mixer.source.set`` and ``prepare`` may give an input
    (``MixerOptions.source_urls``).

    *allowed* holds protocol names as FFmpeg has them ("srt", "https"; "file" is every file) and
    absolute directories, which allow the files below them. None allows every URL. The rule is
    about the URL a request names, not about what an allowed file refers to (a playlist)."""

    def __init__(self, allowed: Optional[Iterable[str]] = None):
        self.protocols: Optional[frozenset[str]] = None
        self.directories: tuple[str, ...] = ()
        if allowed is None:
            return
        wrong = "source_urls is a list of protocol names (srt, https, file) and absolute directories, not {!r}"
        if isinstance(allowed, str):   # would be read as a list of its letters
            raise ValueError(wrong.format(allowed))
        entries = list(allowed)
        for entry in entries:
            if not isinstance(entry, str) or not (entry.startswith("/") or entry and _SCHEME_CHARS.fullmatch(entry)):
                raise ValueError(wrong.format(entry))
        self.protocols = frozenset(entry for entry in entries if not entry.startswith("/"))
        self.directories = tuple(entry for entry in entries if entry.startswith("/"))

    def refusal(self, url: str) -> Optional[str]:
        """Why *url* is not allowed, or None when it is."""
        if self.protocols is None:
            return None
        protocol = url_protocol(url)
        if protocol in self.protocols:
            return None
        if protocol != "file":
            return f"protocol {protocol!r}"
        # As the file protocol reads it; resolved, so that neither ".." nor a link leads outside.
        path = os.path.realpath(url.removeprefix("file:"))
        if any(path.startswith(os.path.realpath(directory).rstrip("/") + "/") for directory in self.directories):
            return None
        return "a file outside the allowed directories"


def _is_position(value) -> bool:
    """A count of milliseconds: an int, and not a bool (which Python counts among them)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _no_other_fields(request: dict, who: str, *fields: str) -> None:
    unknown = set(request) - {"mixer", "id", *fields}
    if unknown:
        raise ConfigError(f"{who}: unsupported fields {sorted(unknown)}")


class SourceController:
    STOP_TIMEOUT_S = STOP_TIMEOUT_S

    def __init__(self, avp, pads: dict[str, SourcePad], *, mixer: str = "mixer",
                 control_port: int = 0,
                 run_command: Optional[Callable[[str], Optional[str]]] = None,
                 probe: Callable[[str], Optional[int]] = probe_duration_ms,
                 allowed_urls: Optional[Iterable[str]] = None,
                 hints: Optional[dict[str, dict]] = None,
                 browser_urls: Optional[dict[str, str]] = None):
        """*run_command* sends one control line and raises RuntimeError with the engine's reason
        when it is refused: ``control_command`` on *control_port* unless given. With neither, the
        playback commands are refused (``mixer.source.set`` needs no refusal and works).
        *allowed_urls* is ``SourceUrls``' list; a ValueError when it is malformed.
        *hints* seeds ``url``/``loop`` per pad id when the input node has not been set yet.
        *browser_urls* is shared with ``BrowserController`` (id → page URL)."""
        self.avp, self.pads, self.mixer = avp, pads, mixer
        self._urls = SourceUrls(allowed_urls)
        self._hints = hints or {}
        self._browser_urls = browser_urls
        if run_command is None and control_port:
            run_command = functools.partial(control_command, control_port)
        self._run_command = run_command
        self._probe = probe
        # One command at a time on a pad; pads do not wait for each other (a prepare takes seconds).
        self._locks = {source_id: threading.Lock() for source_id in pads}

    def list(self) -> dict:
        """Every show pad in index order: ``id``, ``kind``, and ``url``/``loop`` when known."""
        sources = []
        for pad in sorted(self.pads.values(), key=lambda p: p.index):
            item: dict[str, Any] = {"id": pad.id, "kind": pad.kind}
            hint = self._hints.get(pad.id, {})
            if pad.kind in SETTABLE_KINDS:
                params = dict(self.avp.node(pad.input_node).parameters)
                url = params["url"] if "url" in params else hint.get("url")
                loop = params["loop"] if "loop" in params else hint.get("loop")
                if url is not None:
                    item["url"] = url
                if loop is not None:
                    item["loop"] = loop
            elif pad.kind == "browser" and self._browser_urls is not None and pad.id in self._browser_urls:
                item["url"] = self._browser_urls[pad.id]
            sources.append(item)
        return {"sources": sources}

    def set(self, request: dict) -> dict:
        pad = self._pad(request, "source.set")
        url, loop = self._url_and_loop(request, "source.set")
        # Optional fields the chain does not yet retarget (audio track, deinterlace) stay for a
        # follow-up; refuse them so a client does not think they took effect.
        _no_other_fields(request, "source.set", "url", "loop")
        if pad.kind == "mxl":
            if loop:
                raise ConfigError("source.set: an mxl source cannot loop")
            if not _MXL_URL.match(url):
                raise ConfigError("source.set: an mxl source needs url mxl://<domain>?id=<flow> "
                                 "(absolute domain path and a video flow id)")

        with self._locks[pad.id]:
            self._set_unlocked(pad, url, loop)
        return {"id": pad.id, "url": url, "loop": loop}

    def prepare(self, request: dict) -> dict:
        """Load a URL, pause on the cue-in frame, report duration when ffprobe knows it."""
        pad = self._pad(request, "source.prepare", playback=True)
        url, loop = self._url_and_loop(request, "source.prepare")
        cue = request.get("cue_in_ms", 0)
        if not _is_position(cue):
            raise ConfigError("source.prepare: cue_in_ms must be a non-negative int")
        _no_other_fields(request, "source.prepare", "url", "loop", "cue_in_ms")

        # The probe opens the URL itself and may take as long as the load: it runs beside it,
        # and not under the pad's lock.
        probed: list[Optional[int]] = []
        probe = threading.Thread(target=lambda: probed.append(self._probe(url)), name="source-probe", daemon=True)
        probe.start()
        with self._locks[pad.id]:
            self._set_unlocked(pad, url, loop)
            # Pause on the first frame, then seek to cue-in; the pause node holds that frame
            # (pass_on_seek off) until resume … at.
            self._playback_command(f"pause {pad.pause_team} now", wait_s=FIRST_FRAME_WAIT_S)
            self._playback_command(f"seek {pad.sync_team} now {cue}", wait_s=SEEK_WAIT_S)
        probe.join()
        return {"id": pad.id, "url": url, "loop": loop, "cue_in_ms": cue,
                "duration_ms": probed[0] if probed else None}

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
        with self._locks[pad.id]:
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
        """Schedule ``resume <pause_team> at <at_ms>``: the engine's wallclock in ms, the clock
        ``mixer.status`` reports as ``now_pts_ms``."""
        pad = self._pad(request, "source.resume_at", playback=True)
        at_ms = request.get("at_ms")
        if not _is_position(at_ms):
            raise ConfigError("source.resume_at needs a non-negative int at_ms")
        _no_other_fields(request, "source.resume_at", "at_ms")

        with self._locks[pad.id]:
            self._control(f"resume {pad.pause_team} at {at_ms}")
        return {"id": pad.id, "at_ms": at_ms}

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
        if playback and pad.kind not in PLAYBACK_KINDS:
            raise ConfigError(f"{who}: source {source_id!r} kind {pad.kind!r} has no pause or seek "
                              f"(need one of {sorted(PLAYBACK_KINDS)})")
        if playback and not (pad.pause_team and pad.sync_team):
            raise ConfigError(f"{who}: source {source_id!r} has no pause/sync teams "
                              f"(build with MixerOptions.playback_control=True)")
        if playback and self._run_command is None:
            # Pause and seek are tried again on the engine's refusal, which the binding only logs.
            raise ConfigError(f"{who} needs the engine's control port (MixerOptions.remote_control_port)")
        return pad

    def _url_and_loop(self, request: dict, who: str) -> tuple[str, bool]:
        url = request.get("url")
        if not isinstance(url, str) or not url:
            raise ConfigError(f"{who} needs a non-empty url")
        refusal = self._urls.refusal(url)
        if refusal:
            raise ConfigError(f"{who}: url is not allowed here: {refusal}")
        if "loop" not in request:
            raise ConfigError(f"{who} needs a boolean loop")
        loop = request["loop"]
        if not isinstance(loop, bool):
            raise ConfigError(f"{who}: loop must be a boolean")
        return url, loop

    def _set_unlocked(self, pad: SourcePad, url: str, loop: bool) -> None:
        continuous = bool(loop and pad.continuous_loop)
        try:
            stop_group(self.avp, pad.group, self.STOP_TIMEOUT_S)
        except RuntimeError as exc:
            # Not started again here: a node is still running and the group's queues were not
            # cleared, so they may hold the old stream's end, which would end the started chain.
            raise RuntimeError(f"{exc}; source {pad.id!r} stays stopped until a mixer.source.set succeeds") from None
        try:
            self._param(pad.input_node, "url", url)
            self._param(pad.input_node, "loop", loop)
            self._param(pad.input_node, "loop_continuous_ts", continuous)
            # The chain paces on one realtime team; the new input starts its own timeline, so the
            # team starts again too (otherwise it waits on the old one).
            if pad.sync_team:
                self.avp.executeCommandsFromString(f"realtime.team.reset {pad.sync_team}")
            if pad.pause_team:
                self.avp.executeCommandsFromString(f"resume {pad.pause_team}")
        finally:
            # After a failure too: the group is stopped and empty, and stays stopped until asked
            # to start. With the parameters it has, the source plays again instead of going dark.
            self.avp.executeCommandsFromString(f"group.start {pad.group}")

    def _param(self, node: str, key: str, value: Any) -> None:
        self.avp.executeCommandsFromString(f"node.param.set {node} {key} {json.dumps(value)}")
        # The binding only logs a refused line; the node's parameters say whether it was taken.
        if dict(self.avp.node(node).parameters).get(key) != value:
            raise RuntimeError(f"{node}: {key} was not set; the engine's log has the reason")

    def _control(self, text: str) -> Optional[str]:
        """Run a playback line; raises RuntimeError when the engine refuses it."""
        return self._run_command(text)

    def _playback_command(self, text: str, wait_s: float = 3.0) -> None:
        """Pause and seek need a frame of the clip to have been shown; just after a source starts
        there may be none yet, so they are tried again for a moment. A pause that fails this way
        leaves its team as it was (``PauseControlTeam``), so nothing is undone between tries."""
        deadline = time.monotonic() + wait_s
        while True:
            try:
                self._control(text)
                return
            except RuntimeError as exc:
                if _CURRENT_TIME_ERR not in str(exc) or time.monotonic() > deadline:
                    raise
            time.sleep(0.1)


def register_source_commands(avp, controller: SourceController) -> None:
    def handle(method):
        return lambda arg: json.dumps(method(json.loads(arg))) + "\n"

    avp.registerControlCommand("mixer.sources",
                               lambda _arg: json.dumps(controller.list()) + "\n", True, True)
    avp.registerControlCommand("mixer.source.set", handle(controller.set), True)
    avp.registerControlCommand("mixer.source.prepare", handle(controller.prepare), True)
    avp.registerControlCommand("mixer.source.media", handle(controller.media), True)
    avp.registerControlCommand("mixer.source.resume_at", handle(controller.resume_at), True)
