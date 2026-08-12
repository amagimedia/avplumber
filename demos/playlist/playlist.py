#!/usr/bin/env python3
"""Policy and graph plans for the single-playlist regression harness.

The Python code models the OBS MSE playlist policy while the live application
uses AVPlumber's existing C++ ``source_switcher`` and replay nodes unchanged.
Nothing in this module imports the live bindings, so policy and graph shape are
fully testable on a non-NVIDIA host.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List, Optional
from uuid import uuid4

GPU_DEVICE = "cuda0"
OUTPUT_REALTIME_NODE = "pl_realtime"
SYNC_TEAM = "pl_sync"
SWITCHER_NAME = "pl_switcher"
SWITCHER_TYPE = "source_switcher<av::VideoFrame>"
DEFAULT_SLOT_CAPACITY = 16


class PlaylistMode(str, Enum):
    PLAY_ALL = "PlayAll"
    PLAY_CURRENT = "PlayCurrent"
    LOOP_ALL = "LoopAll"
    LOOP_CURRENT = "LoopCurrent"

    @property
    def loops(self) -> bool:
        return self in (PlaylistMode.LOOP_ALL, PlaylistMode.LOOP_CURRENT)

    @property
    def current_only(self) -> bool:
        return self in (PlaylistMode.PLAY_CURRENT, PlaylistMode.LOOP_CURRENT)


class ElementMode(str, Enum):
    PLAY_TO_END = "PlayToEnd"
    TIMED = "Timed"
    LOOP_SELF = "LoopSelf"


class TransportState(str, Enum):
    STOPPED = "Stopped"
    LOADING = "Loading"
    PLAYING = "Playing"
    PAUSED = "Paused"


@dataclass
class Clip:
    url: str
    name: str = ""
    element_mode: ElementMode = ElementMode.PLAY_TO_END
    play_from_ms: int = 0
    play_to_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    disabled: bool = False
    speed: float = 1.0
    item_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("clip url is required")
        if not self.name:
            self.name = self.url.rsplit("/", 1)[-1]
        if self.play_from_ms < 0:
            raise ValueError("cue-in must be >= 0")
        if self.play_to_ms is not None and self.play_to_ms <= self.play_from_ms:
            raise ValueError("cue-out must be greater than cue-in")
        if self.element_mode is ElementMode.TIMED and not self.duration_ms:
            raise ValueError(f"clip {self.name!r}: Timed element requires duration_ms")
        if self.duration_ms is not None and self.duration_ms <= 0:
            raise ValueError("duration must be > 0")
        if self.speed <= 0:
            raise ValueError("speed must be > 0")
        if not self.item_id:
            raise ValueError("item_id is required")


@dataclass(frozen=True)
class BackendEvent:
    kind: str
    item_id: Optional[str] = None
    request_id: Optional[int] = None
    message: str = ""
    value: Optional[bool] = None
    position_ms: Optional[int] = None


@dataclass(frozen=True)
class PlaylistStatus:
    clip_count: int
    selected_index: Optional[int]
    active_index: Optional[int]
    pending_index: Optional[int]
    next_index: Optional[int]
    mode: PlaylistMode
    transport: TransportState
    output_alive: bool
    error: str = ""
    error_index: Optional[int] = None

    @property
    def playing(self) -> bool:
        return self.transport is TransportState.PLAYING

    @property
    def current_index(self) -> Optional[int]:
        return self.active_index


def _step_enabled(clips: List[Clip], start: int, direction: int,
                  wrap: bool) -> Optional[int]:
    if not clips:
        return None
    index = start
    for _ in range(len(clips)):
        index += direction
        if index < 0 or index >= len(clips):
            if not wrap:
                return None
            index %= len(clips)
        if not clips[index].disabled:
            return index
    return None


def first_enabled(clips: List[Clip]) -> Optional[int]:
    return next((index for index, clip in enumerate(clips) if not clip.disabled), None)


def resolve_automatic(clips: List[Clip], current: Optional[int],
                      mode: PlaylistMode) -> Optional[int]:
    """Resolve natural completion; element and playlist modes stay independent."""
    if current is None or not clips:
        return None
    if clips[current].element_mode is ElementMode.LOOP_SELF:
        return current
    if mode.current_only:
        return current if mode.loops else None
    return _step_enabled(clips, current, +1, wrap=mode.loops)


def resolve_manual(clips: List[Clip], current: Optional[int], mode: PlaylistMode,
                   direction: int) -> Optional[int]:
    """Manual navigation always escapes current-only and element-loop modes."""
    if direction not in (-1, +1):
        raise ValueError("direction must be -1 or +1")
    if not clips:
        return None
    if current is None:
        enabled = [index for index, clip in enumerate(clips) if not clip.disabled]
        if not enabled:
            return None
        return enabled[0] if direction > 0 else enabled[-1]
    target = _step_enabled(clips, current, direction, wrap=mode.loops)
    return current if target is None else target


@dataclass(frozen=True)
class NodeSpec:
    type: str
    name: str
    group: str
    params: dict


def item_group(slot: int) -> str:
    return f"pl_item_{slot}"


def item_edge(slot: int, suffix: str) -> str:
    return f"pl_item_{slot}_{suffix}"


def item_pause_team(slot: int) -> str:
    return f"pl_item_{slot}_pause_team"


def item_speed_team(slot: int) -> str:
    return f"pl_item_{slot}_speed_team"


def item_node_names(slot: int) -> List[str]:
    return [item_edge(slot, suffix) for suffix in (
        "input", "demux", "decode", "speed", "fps")]


def _timestamp_ms(value: int) -> str:
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def plan_item_nodes(slot: int, clip: Clip, fps: int = 30) -> List[NodeSpec]:
    """Existing replay-shaped source chain for one stable switcher slot."""
    group = item_group(slot)
    input_params = {
        "url": clip.url,
        "dst": item_edge(slot, "packets"),
        "timeout": -1,
        "preseek": 0,
        "stop_delay": 0,
        "timestamp_source": "wallclock",
        "pause_team": item_pause_team(slot),
        "loop": clip.element_mode is ElementMode.LOOP_SELF,
        "send_eof": True,
    }
    if clip.play_from_ms:
        input_params["start_ts"] = _timestamp_ms(clip.play_from_ms)
    if clip.play_to_ms is not None:
        input_params["stop_ts"] = _timestamp_ms(clip.play_to_ms)

    return [
        NodeSpec("input_rec", item_edge(slot, "input"), group, input_params),
        NodeSpec("demux", item_edge(slot, "demux"), group, {
            "src": item_edge(slot, "packets"),
            "routing": {"v:0": item_edge(slot, "video_packets")},
        }),
        NodeSpec("dec_video", item_edge(slot, "decode"), group, {
            "src": item_edge(slot, "video_packets"),
            "dst": item_edge(slot, "decoded"),
            "pixel_format": "cuda",
            "hwaccel": GPU_DEVICE,
            "codec_map": {"h264": "h264_cuvid"},
            "hwaccel_only_for_codecs": ["h264"],
            "flush_magic": True,
        }),
        NodeSpec("speed_video", item_edge(slot, "speed"), group, {
            "src": item_edge(slot, "decoded"),
            "dst": item_edge(slot, "speeded"),
            "team": item_speed_team(slot),
            "sync_team": SYNC_TEAM,
            "sync_node": OUTPUT_REALTIME_NODE,
            "speed": clip.speed,
        }),
        NodeSpec("force_fps", item_edge(slot, "fps"), group, {
            "src": item_edge(slot, "speeded"),
            "dst": item_edge(slot, "normalized"),
            "fps": f"{fps}/1",
        }),
    ]


def plan_switch_nodes(slot_capacity: int = DEFAULT_SLOT_CAPACITY,
                      fps: int = 30) -> List[NodeSpec]:
    if slot_capacity < 1:
        raise ValueError("slot capacity must be positive")
    return [
        NodeSpec(SWITCHER_TYPE, SWITCHER_NAME, "switch", {
            "src": [item_edge(slot, "normalized")
                    for slot in range(slot_capacity)],
            "dst": "pl_switched",
            "active": 0,
        }),
        NodeSpec("realtime<av::VideoFrame>", OUTPUT_REALTIME_NODE, "switch", {
            "src": "pl_switched",
            "dst": "pl_realtime_out",
            "team": SYNC_TEAM,
            "set_pts": True,
            "tick_period": f"1/{fps}",
            "negative_time_tolerance": 1 / fps,
            "negative_time_discard": 1 / fps,
            "discontinuity_threshold": 3,
        }),
    ]


_UNSET = object()


class PlaylistController:
    """Playlist policy with asynchronous, item-addressed backend operations."""

    def __init__(self, backend, clips: List[Clip],
                 mode: PlaylistMode = PlaylistMode.LOOP_ALL):
        if not clips:
            raise ValueError("playlist must have at least one clip")
        if len({clip.item_id for clip in clips}) != len(clips):
            raise ValueError("playlist item IDs must be unique")
        self._backend = backend
        self.clips = list(clips)
        self.mode = mode
        initial = first_enabled(self.clips)
        if initial is None:
            raise ValueError("playlist must have at least one enabled clip")
        self._selected_id: Optional[str] = self.clips[initial].item_id
        self._active_id: Optional[str] = None
        self._pending_id: Optional[str] = None
        self._request_id = 0
        self._pending_request: Optional[int] = None
        self._transport_before_loading = TransportState.STOPPED
        self.transport = TransportState.STOPPED
        self.error = ""
        self.error_item_id: Optional[str] = None
        self.observed_pts_ms: Optional[int] = None
        self._eof_item_id: Optional[str] = None
        self._timed_started_ms: Optional[int] = None
        self._timed_remaining_ms: Optional[int] = None
        self._last_poll_ms: Optional[int] = None

    def _index_for_id(self, item_id: Optional[str]) -> Optional[int]:
        if item_id is None:
            return None
        return next((index for index, clip in enumerate(self.clips)
                     if clip.item_id == item_id), None)

    def _clip_for_id(self, item_id: Optional[str]) -> Optional[Clip]:
        index = self._index_for_id(item_id)
        return None if index is None else self.clips[index]

    @property
    def selected_index(self) -> Optional[int]:
        return self._index_for_id(self._selected_id)

    @property
    def active_index(self) -> Optional[int]:
        return self._index_for_id(self._active_id)

    @property
    def pending_index(self) -> Optional[int]:
        return self._index_for_id(self._pending_id)

    @property
    def current_index(self) -> Optional[int]:
        return self.active_index

    @property
    def playing(self) -> bool:
        return self.transport is TransportState.PLAYING

    def _check_index(self, index: int) -> None:
        if not 0 <= index < len(self.clips):
            raise IndexError(index)

    def _clear_error(self) -> None:
        self.error = ""
        self.error_item_id = None

    def _reset_timer(self, item_id: Optional[str] = None) -> None:
        clip = self._clip_for_id(item_id if item_id is not None else self._active_id)
        self._timed_started_ms = None
        self._timed_remaining_ms = (
            clip.duration_ms if clip and clip.element_mode is ElementMode.TIMED else None)

    def _request_item(self, index: int) -> bool:
        self._check_index(index)
        clip = self.clips[index]
        self._selected_id = clip.item_id
        if clip.disabled:
            self.error = "element is disabled"
            self.error_item_id = clip.item_id
            return False
        self._request_id += 1
        request_id = self._request_id
        if self.transport is not TransportState.LOADING:
            self._transport_before_loading = self.transport
        self._pending_request = request_id
        self._pending_id = clip.item_id
        self.transport = TransportState.LOADING
        self._eof_item_id = None
        self._reset_timer(clip.item_id)
        self._clear_error()
        try:
            self._backend.play_item(request_id, clip.item_id, clip)
        except Exception as exc:
            self.notify_source_failed(clip.item_id, request_id, str(exc))
            return False
        return True

    def notify_source_ready(self, item_id: str,
                            request_id: Optional[int] = None) -> bool:
        request_id = self._pending_request if request_id is None else request_id
        if item_id != self._pending_id or request_id != self._pending_request:
            return False
        self._active_id = item_id
        self._pending_id = None
        self._pending_request = None
        self.transport = TransportState.PLAYING
        self._clear_error()
        self._reset_timer(item_id)
        return True

    def notify_source_failed(self, item_id: str, request_id: Optional[int],
                             message: str) -> bool:
        if item_id != self._pending_id:
            return False
        if request_id is not None and request_id != self._pending_request:
            return False
        self._pending_id = None
        self._pending_request = None
        self.transport = (self._transport_before_loading
                          if self._active_id is not None else TransportState.STOPPED)
        self.error = message
        self.error_item_id = item_id
        self._reset_timer(self._active_id)
        return True

    def notify_eof(self, item_id: Optional[str] = None) -> None:
        self._eof_item_id = self._active_id if item_id is None else item_id

    def observe_metadata(self, metadata) -> None:
        try:
            if "frame_ts" in metadata:
                self.observed_pts_ms = int(metadata["frame_ts"])
        except (KeyError, TypeError, ValueError):
            pass

    def set_error(self, message: str, item_id: Optional[str] = None) -> None:
        self.error = message
        self.error_item_id = item_id or self._selected_id

    def clear_error(self) -> None:
        self._clear_error()

    def _drain_backend_events(self) -> None:
        poll_events = getattr(self._backend, "poll_events", None)
        if poll_events is None:
            return
        for event in poll_events():
            if event.kind == "ready" and event.item_id is not None:
                self.notify_source_ready(event.item_id, event.request_id)
            elif event.kind == "failed" and event.item_id is not None:
                self.notify_source_failed(
                    event.item_id, event.request_id, event.message or "source failed")
            elif event.kind == "eof":
                self.notify_eof(event.item_id)
            elif event.kind == "error":
                self.set_error(event.message, event.item_id)
            elif event.kind == "position" and event.position_ms is not None:
                self.observed_pts_ms = event.position_ms

    def select(self, index: int) -> None:
        self._check_index(index)
        self._selected_id = self.clips[index].item_id

    def play(self) -> bool:
        if (self.transport is TransportState.PAUSED
                and self._active_id == self._selected_id
                and self._active_id is not None):
            self._backend.resume_item(self._active_id)
            self.transport = TransportState.PLAYING
            self._timed_started_ms = None
            return True
        if self.transport in (TransportState.PLAYING, TransportState.LOADING):
            return False
        target = self.selected_index
        if target is None:
            target = first_enabled(self.clips)
        return False if target is None else self._request_item(target)

    def pause(self) -> bool:
        if self.transport is not TransportState.PLAYING or self._active_id is None:
            return False
        if (self._timed_started_ms is not None and self._last_poll_ms is not None
                and self._timed_remaining_ms is not None):
            elapsed = max(0, self._last_poll_ms - self._timed_started_ms)
            self._timed_remaining_ms = max(0, self._timed_remaining_ms - elapsed)
        self._timed_started_ms = None
        self._backend.pause_item(self._active_id)
        self.transport = TransportState.PAUSED
        return True

    def stop(self) -> bool:
        if self._active_id is None and self._pending_id is None:
            return False
        self._backend.cancel_activation()
        pending = self._pending_id
        if pending is not None and pending != self._active_id:
            self._backend.stop_item(pending)
        if self._active_id is not None:
            self._backend.stop_item(self._active_id)
        self._pending_id = None
        self._pending_request = None
        self.transport = TransportState.STOPPED
        self._eof_item_id = None
        self._reset_timer(self._active_id)
        return True

    def _cancel_pending_item(self, item_id: str) -> bool:
        if item_id != self._pending_id:
            return False
        self._backend.cancel_activation()
        self._pending_id = None
        self._pending_request = None
        self.transport = (self._transport_before_loading
                          if self._active_id is not None else TransportState.STOPPED)
        self._reset_timer(self._active_id)
        return True

    def toggle(self) -> bool:
        return self.pause() if self.transport is TransportState.PLAYING else self.play()

    def element_play(self, index: int) -> bool:
        self.select(index)
        item_id = self.clips[index].item_id
        if item_id == self._active_id and self.transport is TransportState.PLAYING:
            return False
        if item_id == self._active_id and self.transport is TransportState.PAUSED:
            return self.play()
        return self._request_item(index)

    def element_pause(self, index: int) -> bool:
        self.select(index)
        item_id = self.clips[index].item_id
        self._backend.pause_item(item_id)
        if item_id == self._active_id and self.transport is TransportState.PLAYING:
            self.transport = TransportState.PAUSED
            self._timed_started_ms = None
        return True

    def element_stop(self, index: int) -> bool:
        self.select(index)
        item_id = self.clips[index].item_id
        self._cancel_pending_item(item_id)
        self._backend.stop_item(item_id)
        if item_id == self._active_id:
            self.transport = TransportState.STOPPED
            self._reset_timer(item_id)
        return True

    def goto(self, index: int) -> bool:
        return self.element_play(index)

    def _navigation_origin(self) -> Optional[int]:
        return self.active_index if self.active_index is not None else self.selected_index

    def _navigate(self, direction: int) -> bool:
        origin = self._navigation_origin()
        target = resolve_manual(self.clips, origin, self.mode, direction)
        return False if target is None or target == origin else self._request_item(target)

    def next(self) -> bool:
        return self._navigate(+1)

    def prev(self) -> bool:
        return self._navigate(-1)

    def next_index(self, direction: int = +1) -> Optional[int]:
        return resolve_manual(self.clips, self._navigation_origin(), self.mode, direction)

    def _complete_active(self) -> bool:
        index = self.active_index
        if index is None:
            return False
        target = resolve_automatic(self.clips, index, self.mode)
        return self.stop() if target is None else self._request_item(target)

    def poll(self, now_ms: int) -> None:
        self._drain_backend_events()
        self._last_poll_ms = now_ms
        if self.transport is not TransportState.PLAYING or self._active_id is None:
            return
        clip = self._clip_for_id(self._active_id)
        if clip is None:
            return
        if self._eof_item_id == self._active_id:
            self._eof_item_id = None
            if clip.element_mode in (ElementMode.PLAY_TO_END, ElementMode.LOOP_SELF):
                self._complete_active()
                return
        if clip.element_mode is not ElementMode.TIMED or self._timed_remaining_ms is None:
            return
        if self._timed_started_ms is None:
            self._timed_started_ms = now_ms
            return
        if now_ms - self._timed_started_ms >= self._timed_remaining_ms:
            self._complete_active()

    def append_clip(self, clip: Clip) -> int:
        if self._index_for_id(clip.item_id) is not None:
            raise ValueError("playlist item IDs must be unique")
        self.clips.append(clip)
        return len(self.clips) - 1

    def insert_clip(self, index: int, clip: Clip) -> None:
        if not 0 <= index <= len(self.clips):
            raise IndexError(index)
        if self._index_for_id(clip.item_id) is not None:
            raise ValueError("playlist item IDs must be unique")
        self.clips.insert(index, clip)

    def remove_clip(self, index: int) -> Clip:
        self._check_index(index)
        if len(self.clips) == 1:
            raise ValueError("cannot remove the last element")
        removed = self.clips.pop(index)
        self._cancel_pending_item(removed.item_id)
        self._backend.remove_item(removed.item_id)
        if removed.item_id == self._active_id:
            self._active_id = None
            self.transport = TransportState.STOPPED
        if removed.item_id == self._selected_id:
            candidate = min(index, len(self.clips) - 1)
            if self.clips[candidate].disabled:
                candidate = first_enabled(self.clips)
            self._selected_id = None if candidate is None else self.clips[candidate].item_id
        return removed

    def reorder_clip(self, src: int, dst: int) -> None:
        self._check_index(src)
        self._check_index(dst)
        self.clips.insert(dst, self.clips.pop(src))

    def set_disabled(self, index: int, disabled: bool) -> None:
        self._check_index(index)
        clip = self.clips[index]
        clip.disabled = disabled
        if not disabled:
            return
        self._cancel_pending_item(clip.item_id)
        self._backend.stop_item(clip.item_id)
        if clip.item_id == self._active_id:
            self.transport = TransportState.STOPPED
        if clip.item_id == self._selected_id:
            target = _step_enabled(self.clips, index, +1, wrap=True)
            self._selected_id = None if target is None else self.clips[target].item_id

    def update_clip(self, index: int, *, play_from_ms=_UNSET, play_to_ms=_UNSET,
                    duration_ms=_UNSET, speed=_UNSET) -> None:
        self._check_index(index)
        old = self.clips[index]
        changes = {name: value for name, value in (
            ("play_from_ms", play_from_ms), ("play_to_ms", play_to_ms),
            ("duration_ms", duration_ms), ("speed", speed)) if value is not _UNSET}
        updated = replace(old, **changes)
        self.clips[index] = updated
        if not changes or old.item_id != self._active_id:
            return
        if set(changes) == {"speed"} and self.transport in (
                TransportState.PLAYING, TransportState.PAUSED):
            self._backend.set_speed(old.item_id, updated.speed)
            return
        if self.transport in (TransportState.PLAYING, TransportState.PAUSED):
            self._request_item(index)

    def replace_clip(self, index: int, clip: Clip) -> None:
        """Replace all editable settings while preserving graph-slot identity."""
        self._check_index(index)
        old = self.clips[index]
        if clip.item_id != old.item_id:
            raise ValueError("editing an element must preserve its item ID")
        self.clips[index] = clip
        if old.item_id == self._active_id and self.transport in (
                TransportState.PLAYING, TransportState.PAUSED):
            self._request_item(index)

    def set_element_mode(self, index: int, mode: ElementMode,
                         duration_ms: Optional[int] = None) -> None:
        self._check_index(index)
        old = self.clips[index]
        duration = old.duration_ms if duration_ms is None else duration_ms
        if mode is ElementMode.TIMED and not duration:
            raise ValueError("Timed element requires duration_ms")
        self.clips[index] = replace(
            old, element_mode=mode, duration_ms=duration)
        if old.item_id == self._active_id and self.transport in (
                TransportState.PLAYING, TransportState.PAUSED):
            self._request_item(index)

    def set_mode(self, mode: PlaylistMode) -> None:
        self.mode = mode

    def element_state(self, index: int) -> TransportState:
        self._check_index(index)
        item_id = self.clips[index].item_id
        if item_id == self._pending_id:
            return TransportState.LOADING
        if item_id == self._active_id:
            return self.transport
        return TransportState.STOPPED

    def status(self) -> PlaylistStatus:
        try:
            output_alive = bool(self._backend.output_alive())
        except Exception:
            output_alive = False
        active = self.active_index
        return PlaylistStatus(
            clip_count=len(self.clips),
            selected_index=self.selected_index,
            active_index=active,
            pending_index=self.pending_index,
            next_index=resolve_automatic(self.clips, active, self.mode),
            mode=self.mode,
            transport=self.transport,
            output_alive=output_alive,
            error=self.error,
            error_index=self._index_for_id(self.error_item_id),
        )


class InMemoryBackend:
    """Non-printing asynchronous fake used by dry-run TUI and tests."""

    def __init__(self, auto_ready: bool = True):
        self.calls = []
        self.events: List[BackendEvent] = []
        self.alive = True
        self.auto_ready = auto_ready

    def play_item(self, request_id: int, item_id: str, clip: Clip) -> None:
        self.calls.append(("play_item", request_id, item_id, clip))
        if self.auto_ready:
            self.events.append(BackendEvent("ready", item_id, request_id))

    def resume_item(self, item_id: str) -> None:
        self.calls.append(("resume_item", item_id))

    def pause_item(self, item_id: str) -> None:
        self.calls.append(("pause_item", item_id))

    def stop_item(self, item_id: str) -> None:
        self.calls.append(("stop_item", item_id))

    def cancel_activation(self) -> None:
        self.calls.append(("cancel_activation",))

    def set_speed(self, item_id: str, speed: float) -> None:
        self.calls.append(("set_speed", item_id, speed))

    def remove_item(self, item_id: str) -> None:
        self.calls.append(("remove_item", item_id))

    def poll_events(self) -> List[BackendEvent]:
        events, self.events = self.events, []
        return events

    def output_alive(self) -> bool:
        return self.alive

    def clear(self) -> None:
        self.calls.clear()
