"""Safe per-source input-chain restart: stop from the terminals, clear EOS, repoint URL/loop.

A stopping input ends its stream with an end-of-stream frame. Left in a group queue it ends the
node that reads it after the next start; passed on, it can end the mixer's own nodes behind the
compositors. Terminal nodes stop first so the frame stays inside, then every node stops, then the
group's own edges are cleared before params change and the group starts again.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from .config import ConfigError

# Chains built by ``build_input`` (file / live video and NDI). Browser, raw, MXL and VMM pads need
# other primitives (navigate, unpack params, IPC) and refuse ``mixer.source.set``.
SETTABLE_KINDS = frozenset({"video", "ndi"})


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


def source_pads(sources, *, external: Optional[set[str]] = None) -> dict[str, SourcePad]:
    """Pads for every show source, named the way ``build_application`` wires them."""
    external = external or set()
    pads = {}
    for index, source in enumerate(sources):
        group = f"input_{index}"
        # Stock builds wire a sync team for audio and NDI; pause teams arrive with Stage 2 options.
        sync = (f"input_{index}_av_sync"
                if source.kind == "ndi" or getattr(source, "has_audio", False) else None)
        pads[source.id] = SourcePad(
            id=source.id, index=index, kind=source.kind if source.id not in external else "vmm",
            group=group, input_node=f"input_{index}",
            sync_team=sync, pause_team=None, continuous_loop=True)
    return pads


class SourceController:
    def __init__(self, avp, pads: dict[str, SourcePad], *, mixer: str = "mixer"):
        self.avp, self.pads, self.mixer = avp, pads, mixer
        self.lock = threading.Lock()

    def set(self, request: dict) -> dict:
        mixer = request.get("mixer", self.mixer)
        if mixer != self.mixer:
            raise ConfigError(f"unknown mixer: {mixer}")
        source_id = request.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise ConfigError("source.set needs a non-empty id")
        pad = self.pads.get(source_id)
        if pad is None:
            raise ConfigError(f"source.set: unknown source id {source_id!r}")
        if pad.kind not in SETTABLE_KINDS:
            raise ConfigError(f"source.set: source {source_id!r} kind {pad.kind!r} is unsupported "
                              f"(need one of {sorted(SETTABLE_KINDS)})")
        url = request.get("url")
        if not isinstance(url, str) or not url:
            raise ConfigError("source.set needs a non-empty url")
        if "loop" not in request:
            raise ConfigError("source.set needs a boolean loop")
        loop = request["loop"]
        if not isinstance(loop, bool):
            raise ConfigError("source.set: loop must be a boolean")
        # Optional fields the chain does not yet retarget (audio track, deinterlace) stay for a
        # follow-up; refuse them so a client does not think they took effect.
        unknown = set(request) - {"mixer", "id", "url", "loop"}
        if unknown:
            raise ConfigError(f"source.set: unsupported fields {sorted(unknown)}")

        continuous = bool(loop and pad.continuous_loop)
        with self.lock:
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
        return {"id": source_id, "url": url, "loop": loop}

    def _param(self, node: str, key: str, value: Any) -> None:
        self.avp.executeCommandsFromString(f"node.param.set {node} {key} {json.dumps(value)}")


def register_source_commands(avp, controller: SourceController) -> None:
    avp.registerControlCommand(
        "mixer.source.set",
        lambda arg: json.dumps(controller.set(json.loads(arg))) + "\n", True)
