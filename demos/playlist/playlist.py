#!/usr/bin/env python3
"""Single-playlist player: N replay-style worker chains -> source_switcher ->
one realtime reclock -> Janus H.264 RTP.  Built entirely on existing AVPlumber
nodes (see demos/replay).  No C++ changes.

The playlist *logic* (modes, resolve_next, edits) and the *graph plan* (node
specs + emitted control commands) are kept free of the live backend so the whole
feature set is unit-testable through a fake command sink -- this doubles as a
regression harness for AVP's playlist-relevant nodes (source_switcher, pause,
realtime, force_fps, speed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

GPU_DEVICE = "cuda0"

# Shared reclock: every worker's speed node paces against the single downstream
# realtime node so all clips share one output clock (see demos/replay wiring).
OUTPUT_REALTIME_NODE = "pl_realtime"
SYNC_TEAM = "pl_sync"


# --------------------------------------------------------------------------- #
# Playback model
# --------------------------------------------------------------------------- #
class PlaylistMode(str, Enum):
    """Level 1 -- whole-playlist behaviour (the loop x current_clip_only matrix)."""
    PLAY_ALL = "PlayAll"          # advance through list, stop at end
    PLAY_CURRENT = "PlayCurrent"  # play current clip, then stop
    LOOP_ALL = "LoopAll"          # advance, wrap to first at end
    LOOP_CURRENT = "LoopCurrent"  # repeat current clip forever

    @property
    def loops(self) -> bool:
        return self in (PlaylistMode.LOOP_ALL, PlaylistMode.LOOP_CURRENT)

    @property
    def current_only(self) -> bool:
        return self in (PlaylistMode.PLAY_CURRENT, PlaylistMode.LOOP_CURRENT)


class ElementMode(str, Enum):
    """Level 2 -- per-clip hand-off trigger."""
    PLAY_TO_END = "PlayToEnd"  # advance on EOF / play_to
    TIMED = "Timed"            # advance after duration_ms
    LOOP_SELF = "LoopSelf"     # media loops; only manual next/goto ends it


@dataclass
class Clip:
    url: str
    name: str = ""
    element_mode: ElementMode = ElementMode.PLAY_TO_END
    play_from_ms: int = 0
    play_to_ms: Optional[int] = None
    duration_ms: Optional[int] = None  # required when TIMED
    disabled: bool = False
    speed: float = 1.0

    def __post_init__(self):
        if not self.url:
            raise ValueError("clip url is required")
        if not self.name:
            self.name = self.url.rsplit("/", 1)[-1]
        if self.element_mode == ElementMode.TIMED and not self.duration_ms:
            raise ValueError(f"clip {self.name!r}: Timed element requires duration_ms")
        if self.speed <= 0:
            raise ValueError("speed must be > 0")


@dataclass
class PlaylistStatus:
    clip_count: int
    current_index: Optional[int]
    next_index: Optional[int]
    mode: PlaylistMode
    playing: bool
    active_worker: int
    last_command: str = ""
    error: str = ""


# --------------------------------------------------------------------------- #
# Pure sequencer -- no backend, trivially testable
# --------------------------------------------------------------------------- #
def _step_enabled(clips: List[Clip], start: int, direction: int, wrap: bool) -> Optional[int]:
    """First enabled index walking from *start* (exclusive) in *direction*."""
    n = len(clips)
    if n == 0:
        return None
    idx = start
    for _ in range(n):
        idx += direction
        if idx < 0 or idx >= n:
            if not wrap:
                return None
            idx %= n
        if not clips[idx].disabled:
            return idx
    return None


def resolve_next(clips: List[Clip], current: Optional[int], mode: PlaylistMode,
                 direction: int = +1) -> Optional[int]:
    """Which clip index plays after *current*, or None to stop.

    Encodes the 4 playlist modes x element modes.  This is the single source of
    truth used for natural end-of-clip, manual next, and manual prev.
    """
    if not clips or current is None:
        return None
    cur = clips[current]
    # Level 2: a self-looping clip never advances on its own.
    if direction == +1 and cur.element_mode == ElementMode.LOOP_SELF \
            and not mode.current_only:
        return current
    # Level 1: current-only modes never leave the current clip on their own.
    if mode.current_only:
        return current if mode.loops else None
    return _step_enabled(clips, current, direction, wrap=mode.loops)


def first_enabled(clips: List[Clip]) -> Optional[int]:
    for i, c in enumerate(clips):
        if not c.disabled:
            return i
    return None


# --------------------------------------------------------------------------- #
# Graph plan -- node specs + edge names, no live API
# --------------------------------------------------------------------------- #
@dataclass
class NodeSpec:
    type: str
    name: str
    group: str
    params: dict


def worker_group(i: int) -> str:
    return f"worker{i}"


def worker_out_edge(i: int) -> str:
    return f"worker{i}_out"


def pause_node(i: int) -> str:
    return f"worker{i}_pause"


def pause_team(i: int) -> str:
    """PauseControlTeam that freezes/releases worker *i* (pause/resume commands)."""
    return f"worker{i}_pauseteam"


def speed_team(i: int) -> str:
    return f"worker{i}_speed"


def worker_node_names(i: int) -> List[str]:
    """Node names of worker *i*, in the order plan_worker_nodes emits them.
    Teardown deletes them in reverse (sinks before sources)."""
    e = lambda s: f"worker{i}_{s}"
    return [e("input"), e("demux"), e("decode"), e("speed"), e("scale"),
            e("fps"), pause_node(i)]


def plan_worker_nodes(i: int, clip: Clip, fps: int, width: int, height: int) -> List[NodeSpec]:
    """One replay-style decode chain for worker *i*, ending in a Pause (held)
    then the shared switcher edge.  Mirrors demos/replay/replay.py 727-748."""
    g = worker_group(i)
    e = lambda s: f"worker{i}_{s}"
    preseek = clip.play_from_ms / 1000.0
    specs = [
        NodeSpec("input_rec", e("input"), g, {
            "url": clip.url, "dst": e("packets"), "timeout": -1,
            "preseek": preseek, "loop": clip.element_mode == ElementMode.LOOP_SELF,
            "stop_delay": 3000, "timestamp_source": "wallclock"}),
        NodeSpec("demux", e("demux"), g, {
            "src": e("packets"), "routing": {"v:0": e("vpackets")}}),
        NodeSpec("dec_video", e("decode"), g, {
            "src": e("vpackets"), "dst": e("decoded"), "pixel_format": "cuda",
            "hwaccel": GPU_DEVICE, "codec_map": {"h264": "h264_cuvid"},
            "hwaccel_only_for_codecs": ["h264"], "flush_magic": True}),
        NodeSpec("speed_video", e("speed"), g, {
            "src": e("decoded"), "dst": e("speeded"), "team": speed_team(i),
            "sync_team": SYNC_TEAM, "sync_node": OUTPUT_REALTIME_NODE,
            "speed": clip.speed}),
        NodeSpec("rescale_video", e("scale"), g, {
            "src": e("speeded"), "dst": e("scaled"),
            "width": width, "height": height, "hwaccel": GPU_DEVICE}),
        NodeSpec("force_fps", e("fps"), g, {
            "src": e("scaled"), "dst": e("normalized"), "fps": f"{fps}/1"}),
        # Held (paused) while this worker is a preload; released at the cut via
        # `resume <team>`.  paused=True freezes it the moment the group starts.
        NodeSpec("pause", pause_node(i), g, {
            "src": e("normalized"), "dst": worker_out_edge(i),
            "team": pause_team(i), "sync_team": SYNC_TEAM, "paused": True}),
    ]
    return specs


def plan_output_nodes(worker_count: int, fps: int, active: int,
                      timeline: str = "seq_tl") -> List[NodeSpec]:
    """source_switcher over all worker_out edges -> single realtime reclock ->
    probe edge.  The Janus encode/mux/output group is added verbatim from replay
    downstream of `pl_observed` by the live builder."""
    srcs = [worker_out_edge(i) for i in range(worker_count)]
    return [
        NodeSpec("source_switcher", "pl_switcher", "switch", {
            "src": srcs, "dst": "pl_switched", "active": active,
            "timeline": timeline}),
        NodeSpec("realtime<av::VideoFrame>", OUTPUT_REALTIME_NODE, "switch", {
            "src": "pl_switched", "dst": "pl_realtime_out", "team": SYNC_TEAM,
            "set_pts": True, "tick_period": f"1/{fps}",
            "negative_time_tolerance": 1 / fps,
            "negative_time_discard": 1 / fps, "discontinuity_threshold": 3}),
    ]


def contains_black_source(specs: List[NodeSpec]) -> bool:
    """Regression guard: the playlist graph must never contain a black/card
    generator (no sentinel) nor a configured black fallback on the switcher."""
    for s in specs:
        if s.type in ("sentinel_video", "sentinel_audio", "sentinel"):
            return True
        if s.type == "source_switcher" and "fallback_active" in s.params:
            return True
    return False


# --------------------------------------------------------------------------- #
# Cut command sequence -- unpause the incoming worker BEFORE flipping active
# --------------------------------------------------------------------------- #
def pause_command(worker: int, paused: bool) -> str:
    """Freeze/release a worker through its PauseControlTeam.  `pause <team> now`
    holds the worker on its current decoded frame (never black); `resume <team>`
    lets it run again -- exactly how demos/replay drives its pause teams."""
    team = pause_team(worker)
    return f"pause {team} now" if paused else f"resume {team}"


def flip_command(active: int, timeline: str = "seq_tl",
                 at_pts_ms: Optional[int] = None) -> str:
    if at_pts_ms is None:
        return f"node.object.set pl_switcher active {active}"
    return "timeline.set " + json.dumps({
        "name": timeline, "channel": "sel", "key": "active",
        "at": at_pts_ms, "val": active})


def cut_commands(target_worker: int, at_pts_ms: Optional[int] = None) -> List[str]:
    """Ordered: release the frozen incoming worker, THEN select it.  Doing the
    flip first would select an unproduced input and open an avoidable gap."""
    return [pause_command(target_worker, False),
            flip_command(target_worker, at_pts_ms=at_pts_ms)]


# --------------------------------------------------------------------------- #
# Controller -- drives everything through a command sink (fake in tests)
# --------------------------------------------------------------------------- #
class PlaylistController:
    """Playlist transport + edits.  All backend interaction goes through the
    injected `command(str)` sink, so tests capture the exact AVP command stream.
    """

    def __init__(self, command: Callable[[str], None], clips: List[Clip],
                 mode: PlaylistMode = PlaylistMode.LOOP_ALL, worker_count: int = 2,
                 fps: int = 30, width: int = 1920, height: int = 1080):
        if worker_count < 2:
            raise ValueError("need at least 2 workers")
        if not clips:
            raise ValueError("playlist must have at least one clip")
        self._command = command
        self.clips = list(clips)
        self.mode = mode
        self.worker_count = worker_count
        self.fps, self.width, self.height = fps, width, height
        self.playing = True
        self._active_worker = 0
        self._worker_clip: List[Optional[int]] = [None] * worker_count
        # worker 0's chain is created by the live builder; the rest are built on
        # demand.  `_built` tracks which names currently exist in the backend so
        # a rebuild deletes them first (node.add rejects a busy name).
        self._built = [False] * worker_count
        self._built[0] = True
        self.current_index = first_enabled(self.clips)
        self._worker_clip[0] = self.current_index
        self.last_command = ""
        self.error = ""
        self.ready = False
        self.observed_pts_ms: Optional[int] = None

    # -- backend observation (called by the live PositionProbe) ------------
    def observe_metadata(self, metadata) -> None:
        """Called from the graph thread for every frame reaching the output."""
        self.ready = True
        try:
            if "frame_ts" in metadata:
                self.observed_pts_ms = int(metadata["frame_ts"])
        except (KeyError, TypeError, ValueError):
            pass

    def set_error(self, message: str) -> None:
        self.error = message

    # -- helpers -----------------------------------------------------------
    def _send(self, cmd: str) -> None:
        self.last_command = cmd
        self._command(cmd)

    def _idle_worker(self) -> int:
        return (self._active_worker + 1) % self.worker_count

    def next_index(self, direction: int = +1) -> Optional[int]:
        return resolve_next(self.clips, self.current_index, self.mode, direction)

    # -- preload / cut -----------------------------------------------------
    def _teardown(self, worker: int) -> None:
        """Stop and delete a worker's nodes so their names/edges free up.
        group.stop alone leaves the nodes registered -> node.add would clash."""
        if not self._built[worker]:
            return
        self._send(f"group.stop {worker_group(worker)}")
        for name in reversed(worker_node_names(worker)):
            self._send(f"node.delete {name}")
        self._built[worker] = False
        self._worker_clip[worker] = None

    def _preload(self, worker: int, clip_index: int) -> None:
        """Rebuild an idle worker onto *clip_index*, frozen (paused) on frame 0."""
        self._teardown(worker)
        self._worker_clip[worker] = clip_index
        clip = self.clips[clip_index]
        for spec in plan_worker_nodes(worker, clip, self.fps, self.width, self.height):
            self._send("node.add " + json.dumps(
                {"type": spec.type, "name": spec.name, "group": spec.group, **spec.params}))
        self._built[worker] = True
        self._send(f"group.start {worker_group(worker)}")

    def _cut(self, target_worker: int, at_pts_ms: Optional[int] = None) -> None:
        for cmd in cut_commands(target_worker, at_pts_ms):
            self._send(cmd)
        self._active_worker = target_worker
        self.current_index = self._worker_clip[target_worker]

    def _rebuild_idle_for_next(self) -> None:
        """Preload resolve_next onto the idle worker (best-effort)."""
        nxt = self.next_index(+1)
        idle = self._idle_worker()
        if nxt is not None and self._worker_clip[idle] != nxt:
            self._preload(idle, nxt)

    # -- transport ---------------------------------------------------------
    def play(self) -> None:
        self.playing = True
        self._send(pause_command(self._active_worker, False))

    def pause(self) -> None:
        self.playing = False
        self._send(pause_command(self._active_worker, True))

    def toggle(self) -> None:
        self.pause() if self.playing else self.play()

    def stop(self) -> None:
        self.playing = False
        self._send(pause_command(self._active_worker, True))
        self.current_index = None

    def next(self) -> bool:
        return self._advance(+1)

    def prev(self) -> bool:
        return self._advance(-1)

    def _advance(self, direction: int) -> bool:
        target = self.next_index(direction)
        if target is None:
            self.error = "no next clip"
            return False
        self.error = ""
        idle = self._idle_worker()
        if self._worker_clip[idle] != target:      # not preloaded -> build now
            self._preload(idle, target)
        self._cut(idle)
        self._rebuild_idle_for_next()
        return True

    def goto(self, index: int, at_pts_ms: Optional[int] = None) -> bool:
        if not 0 <= index < len(self.clips):
            raise IndexError(index)
        if self.clips[index].disabled:
            self.error = "clip is disabled"
            return False
        self.error = ""
        idle = self._idle_worker()
        if self._worker_clip[idle] != index:
            self._preload(idle, index)
        self._cut(idle, at_pts_ms)
        self._rebuild_idle_for_next()
        return True

    # -- edits -------------------------------------------------------------
    def append_clip(self, clip: Clip) -> None:
        self.clips.append(clip)
        self._rebuild_idle_for_next()

    def insert_clip(self, index: int, clip: Clip) -> None:
        self.clips.insert(index, clip)
        if self.current_index is not None and index <= self.current_index:
            self.current_index += 1
        self._rebuild_idle_for_next()

    def remove_clip(self, index: int) -> None:
        if len(self.clips) <= 1:
            raise ValueError("cannot remove the last clip")
        removing_current = index == self.current_index
        self.clips.pop(index)
        if self.current_index is not None and index < self.current_index:
            self.current_index -= 1
        if removing_current:                        # guard 1: cut away now
            self.current_index = min(self.current_index or 0, len(self.clips) - 1)
            self._advance(+1)
        else:                                       # guard 2: refresh preload
            self._rebuild_idle_for_next()

    def reorder_clip(self, src: int, dst: int) -> None:
        clip = self.clips.pop(src)
        self.clips.insert(dst, clip)
        # track current across the move
        if self.current_index is not None:
            self.current_index = self.clips.index(clip) if src == self.current_index \
                else max(0, min(self.current_index, len(self.clips) - 1))
        self._rebuild_idle_for_next()

    def set_disabled(self, index: int, disabled: bool) -> None:
        self.clips[index].disabled = disabled
        self._rebuild_idle_for_next()

    def set_element_mode(self, index: int, mode: ElementMode,
                         duration_ms: Optional[int] = None) -> None:
        clip = self.clips[index]
        if mode == ElementMode.TIMED and not (duration_ms or clip.duration_ms):
            raise ValueError("Timed element requires duration_ms")
        clip.element_mode = mode
        if duration_ms:
            clip.duration_ms = duration_ms
        self._rebuild_idle_for_next()

    def set_mode(self, mode: PlaylistMode) -> None:
        self.mode = mode
        self._rebuild_idle_for_next()

    # -- status ------------------------------------------------------------
    def status(self) -> PlaylistStatus:
        return PlaylistStatus(
            clip_count=len(self.clips), current_index=self.current_index,
            next_index=self.next_index(+1), mode=self.mode, playing=self.playing,
            active_worker=self._active_worker, last_command=self.last_command,
            error=self.error)
