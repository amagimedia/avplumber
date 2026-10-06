"""Media clips replayed from GPU memory.

An optional module beside the mixer graph: it inserts a ``clip_cache`` node at
the end of a normal decode chain and splits that chain into two groups. The
loader group is started at startup, once per clip, to fill the cache; the player group
runs from startup on, idle, and a live transition only arms it, so a take costs
no file open, no decoder and no node or thread startup. Nothing here reaches
into the compositor or the orchestrator.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, Iterable

log = logging.getLogger(__name__)

# The shared store the cache node fills. The mixer reads it by this name for
# mixer.status, so the status report and the node see the same clips.
STORE = "clips"


# The loader chain as MixerGraphBuilder names it, in stream order, and the edges
# between its nodes. Its last node feeds the cache.
LOADER_NODES = ("wipe_input", "wipe_demux", "wipe_dec", "wipe_fmt", "wipe_rt")
LOADER_EDGES = ("wipe_raw_pkt", "wipe_v_pkt", "wipe_dec_out", "wipe_fmt_out")
DECODED_EDGE = "wipe_dec_out"
CACHE_INPUT_EDGE = "wipe_rt_out"
CACHE_NODE = "wipe_cache"


def loader_group(mixer_name: str) -> str:
    """Group holding the decode chain; started only to fill the cache."""
    return f"{mixer_name}_wipe_load"


def cache_node(*, name: str, src: str, dst: str, group: str, fps: str,
               store: str = STORE, budget_mb: float | None = None,
               url: str = "") -> Dict[str, Any]:
    """Parameters for the clip_cache node.

    ``url`` names the clip the decode chain delivers when both groups start
    together (a preload without a running player). The running node is told
    what to load with its ``load`` object and what to replay with ``play``,
    which is how the mixer arms a media wipe.
    """
    node: Dict[str, Any] = {"name": name, "src": src, "dst": dst, "group": group,
                            "fps": fps, "cache": store, "url": url}
    if budget_mb is not None:
        node["budget_mb"] = budget_mb
    return node


def _complete(avp, mixer_name: str) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """The status entries of the clips held whole, by path, and the cache node's status."""
    status = avp.node(f"{mixer_name}_{CACHE_NODE}").getObject("status")
    return {c["path"]: c for c in status["clips"] if c["complete"]}, status


def require_cached(avp, mixer_name: str, clips: Iterable[str]) -> None:
    """Raise unless every one of *clips* is held whole.

    preload() checks its own clip only, and a later load evicts earlier clips when
    the budget does not hold them all; call this after the last load.
    """
    complete, status = _complete(avp, mixer_name)
    evicted = [clip for clip in clips if clip not in complete]
    if evicted:
        raise RuntimeError(
            f"the wipe cache does not hold every clip: {', '.join(evicted)} evicted to make room; "
            f"{status['bytes'] / 1048576:.0f} MiB cached, budget {status['budget_bytes'] / 1048576:.0f} MiB")


def preload(avp, mixer_name: str, clip: str, *, timeout_sec: float, poll_sec: float = 0.02,
            check: Callable[[], None] = lambda: None) -> Dict[str, Any]:
    """Decode *clip* once into the running cache and return its status entry.

    A load ends when the loader chain's end-of-stream marker reaches the cache,
    never because frames stopped arriving. The cached frame count must then equal
    the number of frames the decoder delivered; a clip that differs, or of which
    nothing was cached, is dropped and loaded once more, and a second such load
    raises. Each load has *timeout_sec* to end, and its loader as long again to
    stop; running out is a failure. *check* is called on every poll and raises
    to abort. A clip already held whole is returned as it is.
    """
    held = _complete(avp, mixer_name)[0].get(clip)
    if held:   # the node ignores a load of a clip it holds, so no end would follow
        return held
    cache = f"{mixer_name}_{CACHE_NODE}"
    decoded_edge = f"{mixer_name}_{DECODED_EDGE}"
    group = avp.group(loader_group(mixer_name))
    value = json.dumps(clip)   # the control commands parse the value as JSON

    def wait(done: Callable[[], bool], what: str, deadline: float) -> None:
        while not done():
            check()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"wipe preload timed out waiting for {what} {clip}")
            time.sleep(poll_sec)

    def stop_and_discard() -> None:
        # stopNodes() only requests the stop. A node stalled in a GPU call outlives it
        # and still delivers, so the next load waits until every node is gone, drops
        # what they left between them, and lets the cache discard what reached it.
        group.stopNodes()
        deadline = time.monotonic() + timeout_sec   # its own: a load that ended in time is not failed by its stop
        wait(lambda: not any(avp.node(f"{mixer_name}_{node}").isWorking for node in LOADER_NODES),
             "the loader to stop after", deadline)
        for edge in LOADER_EDGES:
            avp.getEdge(f"{mixer_name}_{edge}").clear()
        wait(lambda: avp.getEdge(f"{mixer_name}_{CACHE_INPUT_EDGE}").occupied == 0,
             "the cache to discard late frames of", deadline)

    def load() -> tuple[Dict[str, Any] | None, int]:
        deadline = time.monotonic() + timeout_sec
        decoded = avp.getEdge(decoded_edge).enqueued_total
        # The reader needs the clip before its group starts, to open the file; the
        # running cache is told which clip the chain is about to deliver.
        avp.executeCommandsFromString(
            f"node.param.set {mixer_name}_{LOADER_NODES[0]} url {value}\n"
            f"node.object.set {cache} load {value}")
        group.startNodes()
        held = None

        def ended() -> bool:
            nonlocal held
            complete, status = _complete(avp, mixer_name)
            held = complete.get(clip)
            return held is not None or status["loading"] != clip

        wait(ended, "the end of", deadline)
        # The decoder ends its output with one end-of-stream marker, which the edge counts.
        expected = avp.getEdge(decoded_edge).enqueued_total - decoded - 1
        stop_and_discard()
        return held, expected

    for attempt in ("first", "second"):
        held, expected = load()
        if held and held["frames"] == expected:
            return held
        # Nothing cached has no count to compare: the decoder's is not final when the
        # cache gives a clip up over its budget.
        found = (f"{held['frames']} frames cached, {expected} frames decoded" if held else
                 "ended without being cached (no picture in it, or over the cache budget)")
        avp.executeCommandsFromString(f"node.object.set {cache} forget {value}")
        log.warning("wipe clip is not whole after the %s load: %s: %s", attempt, clip, found)
    raise RuntimeError(f"wipe clip is not whole after a second load: {clip}: {found}")
