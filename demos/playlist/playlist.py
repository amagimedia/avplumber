#!/usr/bin/env python3
"""Playlist policy for the mixer-backed playlist demo.

The controller decides *what* plays and *when* it ends; the backend (see
``engine.py``) turns that into native mixer transitions scheduled on the host
monotonic clock.  Nothing here imports the native bindings, so the policy is
fully testable on any machine.

Time values are milliseconds on the same monotonic clock AVPlumber uses for
``start_pts_ms`` (``time.monotonic_ns() // 1_000_000``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List, Optional
from uuid import uuid4

SLOT_CAPACITY = 16


def now_ms() -> int:
    """Milliseconds on the clock AVPlumber schedules with (CLOCK_MONOTONIC)."""
    return time.monotonic_ns() // 1_000_000


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


class Transition(str, Enum):
    CUT = "Cut"
    FADE = "Fade"
    WIPE = "Wipe"


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

    @property
    def repeats(self) -> bool:
        """The picture wraps to cue-in on its own while the element stays on air."""
        return self.element_mode in (ElementMode.LOOP_SELF, ElementMode.TIMED)

    def media_span_ms(self, media_length_ms: Optional[int]) -> Optional[int]:
        """Wallclock time of one pass from cue-in to cue-out, or None if unknown."""
        end = self.play_to_ms if self.play_to_ms is not None else media_length_ms
        if end is None or end <= self.play_from_ms:
            return None
        return int(round((end - self.play_from_ms) / self.speed))

    def span_ms(self, media_length_ms: Optional[int]) -> Optional[int]:
        """Wallclock time the element occupies on air; None for unknown or LoopSelf."""
        if self.element_mode is ElementMode.TIMED:
            return self.duration_ms
        if self.element_mode is ElementMode.LOOP_SELF:
            return None
        return self.media_span_ms(media_length_ms)


@dataclass(frozen=True)
class BackendEvent:
    kind: str                      # on_air | failed | error | health
    item_id: Optional[str] = None
    request_id: Optional[int] = None
    message: str = ""
    value: Optional[bool] = None
    at_ms: Optional[int] = None


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
    position_ms: Optional[int] = None
    end_ms: Optional[int] = None
    next_at_ms: Optional[int] = None
    error: str = ""
    error_index: Optional[int] = None

    @property
    def playing(self) -> bool:
        return self.transport is TransportState.PLAYING


def _step_enabled(clips: List[Clip], start: int, direction: int, wrap: bool) -> Optional[int]:
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


def resolve_automatic(clips: List[Clip], current: Optional[int], mode: PlaylistMode) -> Optional[int]:
    """Where playback goes when the current element completes."""
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


_UNSET = object()


class PlaylistController:
    """Playlist policy driving an asynchronous, item-addressed backend.

    Backend protocol (all non-blocking):
      cue(request_id, clip, at_ms, transition, transition_ms) -> event on_air/failed
      pause(item_id) / resume(item_id) / park(item_id) / remove(item_id)
      media_length_ms(clip) -> int | None
      poll_events() -> list[BackendEvent];  output_alive() -> bool
    Every chain loops between its cue points, so an element that repeats
    (LoopCurrent, Timed, LoopSelf) needs no backend action at its end.
    """

    def __init__(self, backend, clips: List[Clip], mode: PlaylistMode = PlaylistMode.LOOP_ALL,
                 transition: Transition = Transition.CUT, transition_ms: int = 500):
        if not clips:
            raise ValueError("playlist must have at least one clip")
        if len({clip.item_id for clip in clips}) != len(clips):
            raise ValueError("playlist item IDs must be unique")
        initial = first_enabled(clips)
        if initial is None:
            raise ValueError("playlist must have at least one enabled clip")
        self._backend = backend
        self.clips = list(clips)
        self.mode = mode
        self.transition = transition
        self.transition_ms = transition_ms
        self.transport = TransportState.STOPPED
        self.error = ""
        self.error_item_id: Optional[str] = None
        self._selected_id: Optional[str] = self.clips[initial].item_id
        self._active_id: Optional[str] = None
        self._pending_id: Optional[str] = None      # manual take in flight
        self._armed_id: Optional[str] = None        # automatic next, scheduled
        self._request_id = 0
        self._pending_request: Optional[int] = None
        self._armed_request: Optional[int] = None
        self._start_ms: Optional[int] = None
        self._end_ms: Optional[int] = None
        self._paused_at_ms: Optional[int] = None
        self._now_ms = 0

    # ---- lookups -------------------------------------------------------
    def _index_for_id(self, item_id: Optional[str]) -> Optional[int]:
        if item_id is None:
            return None
        return next((i for i, clip in enumerate(self.clips) if clip.item_id == item_id), None)

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
    def playing(self) -> bool:
        return self.transport is TransportState.PLAYING

    def _check_index(self, index: int) -> None:
        if not 0 <= index < len(self.clips):
            raise IndexError(index)

    def _next_request(self) -> int:
        self._request_id += 1
        return self._request_id

    # ---- errors --------------------------------------------------------
    def set_error(self, message: str, item_id: Optional[str] = None) -> None:
        self.error = message
        self.error_item_id = item_id or self._selected_id

    def clear_error(self) -> None:
        self.error = ""
        self.error_item_id = None

    # ---- scheduling ----------------------------------------------------
    def _disarm(self) -> None:
        if self._armed_id is not None:
            self._backend.park(self._armed_id)
        self._armed_id = None
        self._armed_request = None

    def _arm_next(self) -> None:
        """Schedule what happens at the active element's end, natively."""
        self._disarm()
        index = self.active_index
        if index is None or self._end_ms is None:
            return
        target = resolve_automatic(self.clips, index, self.mode)
        if target is None or target == index:
            return                              # end-stop and self-repeat are handled in poll()
        clip = self.clips[target]
        self._armed_id = clip.item_id
        self._armed_request = self._next_request()
        self._backend.cue(self._armed_request, clip, self._end_ms,
                          self.transition, self.transition_ms)

    def _went_on_air(self, item_id: str, at_ms: int) -> None:
        previous = self._active_id
        self._active_id = item_id
        self._pending_id = self._pending_request = None
        self._armed_id = self._armed_request = None
        self.transport = TransportState.PLAYING
        self._paused_at_ms = None
        self._start_ms = at_ms
        clip = self._clip_for_id(item_id)
        span = clip.span_ms(self._backend.media_length_ms(clip))
        self._end_ms = None if span is None else at_ms + span
        if clip.element_mode is not ElementMode.LOOP_SELF and span is None:
            self.set_error("length unknown; set a cue-out", item_id)
        else:
            self.clear_error()
        if previous is not None and previous != item_id:
            self._backend.park(previous)
        self._arm_next()

    def _take(self, index: int, *, select: bool = True) -> bool:
        self._check_index(index)
        clip = self.clips[index]
        if select:
            self._selected_id = clip.item_id
        if clip.disabled:
            self.set_error("element is disabled", clip.item_id)
            return False
        self._disarm()
        self._pending_id = clip.item_id
        self._pending_request = self._next_request()
        if self._active_id is None:
            self.transport = TransportState.LOADING
        self.clear_error()
        try:
            self._backend.cue(self._pending_request, clip, None,
                              self.transition, self.transition_ms)
        except Exception as exc:  # noqa: BLE001
            self.notify_failed(clip.item_id, self._pending_request, str(exc))
            return False
        return True

    # ---- backend events ------------------------------------------------
    def notify_on_air(self, item_id: str, request_id: Optional[int], at_ms: Optional[int]) -> bool:
        if request_id not in (self._pending_request, self._armed_request) or request_id is None:
            return False
        self._went_on_air(item_id, self._now_ms if at_ms is None else at_ms)
        return True

    def notify_failed(self, item_id: str, request_id: Optional[int], message: str) -> bool:
        if request_id == self._pending_request and request_id is not None:
            self._pending_id = self._pending_request = None
            if self._active_id is None:
                self.transport = TransportState.STOPPED
        elif request_id == self._armed_request and request_id is not None:
            self._armed_id = self._armed_request = None
        else:
            return False
        self.set_error(message, item_id)
        return True

    def _drain_backend_events(self) -> None:
        for event in self._backend.poll_events():
            if event.kind == "on_air" and event.item_id is not None:
                self.notify_on_air(event.item_id, event.request_id, event.at_ms)
            elif event.kind == "failed" and event.item_id is not None:
                self.notify_failed(event.item_id, event.request_id, event.message or "source failed")
            elif event.kind == "error":
                self.set_error(event.message, event.item_id)

    def poll(self, now_ms: int) -> None:
        self._now_ms = now_ms
        self._drain_backend_events()
        if self.transport is not TransportState.PLAYING or self._end_ms is None:
            return
        if now_ms < self._end_ms:
            return
        index = self.active_index
        target = resolve_automatic(self.clips, index, self.mode)
        if target is None:
            self.stop()                          # PlayAll/PlayCurrent reached the end
        elif target == index:
            self._start_ms, self._end_ms = self._end_ms, None
            clip = self.clips[index]
            span = clip.span_ms(self._backend.media_length_ms(clip))
            self._end_ms = None if span is None else self._start_ms + span
            self._arm_next()
        elif self._armed_id is None and self._pending_id is None:
            # The armed transition failed earlier: cut now rather than hang on
            # the finished element.  The error stays visible until a take works.
            message = self.error
            self._take(target, select=False)
            if message and not self.error:
                self.error = message
        # Otherwise the armed element reports on_air itself.

    # ---- transport -----------------------------------------------------
    def select(self, index: int) -> None:
        self._check_index(index)
        self._selected_id = self.clips[index].item_id

    def play(self) -> bool:
        if self.transport is TransportState.PAUSED and self._active_id is not None:
            self._backend.resume(self._active_id)
            self.transport = TransportState.PLAYING
            if self._end_ms is not None and self._paused_at_ms is not None:
                shift = self._now_ms - self._paused_at_ms
                self._start_ms += shift
                self._end_ms += shift
            self._paused_at_ms = None
            self._arm_next()
            return True
        if self.transport in (TransportState.PLAYING, TransportState.LOADING):
            return False
        target = self.active_index if self.active_index is not None else first_enabled(self.clips)
        return False if target is None else self._take(target, select=False)

    def pause(self) -> bool:
        if self.transport is not TransportState.PLAYING or self._active_id is None:
            return False
        self._disarm()
        self._backend.pause(self._active_id)
        self._paused_at_ms = self._now_ms
        self.transport = TransportState.PAUSED
        return True

    def stop(self) -> bool:
        if self._active_id is None and self._pending_id is None:
            return False
        self._disarm()
        if self._pending_id is not None and self._pending_id != self._active_id:
            self._backend.park(self._pending_id)
        if self._active_id is not None:
            self._backend.park(self._active_id)
        self._pending_id = self._pending_request = None
        self.transport = TransportState.STOPPED
        self._start_ms = self._end_ms = self._paused_at_ms = None
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
        return self._take(index)

    def element_pause(self, index: int) -> bool:
        self.select(index)
        if self.clips[index].item_id == self._active_id:
            return self.pause()
        self._backend.pause(self.clips[index].item_id)
        return True

    def element_stop(self, index: int) -> bool:
        self.select(index)
        item_id = self.clips[index].item_id
        if item_id in (self._active_id, self._pending_id):
            return self.stop()
        if item_id == self._armed_id:
            self._disarm()
        else:
            self._backend.park(item_id)
        return True

    def _navigation_origin(self) -> Optional[int]:
        return self.active_index if self.active_index is not None else self.selected_index

    def _navigate(self, direction: int) -> bool:
        origin = self._navigation_origin()
        target = resolve_manual(self.clips, origin, self.mode, direction)
        return False if target is None or target == origin else self._take(target)

    def next(self) -> bool:
        return self._navigate(+1)

    def prev(self) -> bool:
        return self._navigate(-1)

    # ---- editing -------------------------------------------------------
    def _reschedule_if_active(self, item_id: str) -> None:
        if item_id == self._active_id and self.transport in (TransportState.PLAYING, TransportState.PAUSED):
            self._take(self.active_index, select=False)
        elif item_id == self._armed_id:
            self._arm_next()

    def insert_clip(self, index: int, clip: Clip) -> None:
        if not 0 <= index <= len(self.clips):
            raise IndexError(index)
        if self._index_for_id(clip.item_id) is not None:
            raise ValueError("playlist item IDs must be unique")
        if len(self.clips) >= SLOT_CAPACITY:
            raise ValueError(f"playlist holds at most {SLOT_CAPACITY} elements")
        self.clips.insert(index, clip)
        self._arm_next()

    def remove_clip(self, index: int) -> Clip:
        self._check_index(index)
        if len(self.clips) == 1:
            raise ValueError("cannot remove the last element")
        removed = self.clips.pop(index)
        if removed.item_id in (self._active_id, self._pending_id):
            self.stop()
            self._active_id = None
        if removed.item_id == self._armed_id:
            self._armed_id = self._armed_request = None
        self._backend.remove(removed.item_id)
        if removed.item_id == self._selected_id:
            candidate = min(index, len(self.clips) - 1)
            if self.clips[candidate].disabled:
                candidate = first_enabled(self.clips)
            self._selected_id = None if candidate is None else self.clips[candidate].item_id
        self._arm_next()
        return removed

    def reorder_clip(self, src: int, dst: int) -> None:
        self._check_index(src)
        self._check_index(dst)
        self.clips.insert(dst, self.clips.pop(src))
        self._arm_next()

    def set_disabled(self, index: int, disabled: bool) -> None:
        self._check_index(index)
        clip = self.clips[index]
        clip.disabled = disabled
        if disabled:
            if clip.item_id in (self._active_id, self._pending_id):
                self.stop()
            elif clip.item_id == self._armed_id:
                self._disarm()
            if clip.item_id == self._selected_id:
                target = _step_enabled(self.clips, index, +1, wrap=True)
                self._selected_id = None if target is None else self.clips[target].item_id
        self._arm_next()

    def update_clip(self, index: int, *, play_from_ms=_UNSET, play_to_ms=_UNSET,
                    duration_ms=_UNSET, speed=_UNSET) -> None:
        self._check_index(index)
        changes = {name: value for name, value in (
            ("play_from_ms", play_from_ms), ("play_to_ms", play_to_ms),
            ("duration_ms", duration_ms), ("speed", speed)) if value is not _UNSET}
        if changes:
            self.replace_clip(index, replace(self.clips[index], **changes))

    def replace_clip(self, index: int, clip: Clip) -> None:
        """Replace all editable settings while preserving slot identity."""
        self._check_index(index)
        if clip.item_id != self.clips[index].item_id:
            raise ValueError("editing an element must preserve its item ID")
        self.clips[index] = clip
        self._reschedule_if_active(clip.item_id)

    def set_element_mode(self, index: int, mode: ElementMode, duration_ms: Optional[int] = None) -> None:
        self._check_index(index)
        old = self.clips[index]
        duration = old.duration_ms if duration_ms is None else duration_ms
        if mode is ElementMode.TIMED and not duration:
            raise ValueError("Timed element requires duration_ms")
        self.replace_clip(index, replace(old, element_mode=mode, duration_ms=duration))

    def set_mode(self, mode: PlaylistMode) -> None:
        self.mode = mode
        self._arm_next()

    def set_transition(self, transition: Transition, duration_ms: Optional[int] = None) -> None:
        self.transition = transition
        if duration_ms is not None:
            if duration_ms <= 0:
                raise ValueError("transition duration must be positive")
            self.transition_ms = duration_ms
        self._arm_next()

    # ---- status --------------------------------------------------------
    def element_state(self, index: int) -> TransportState:
        self._check_index(index)
        item_id = self.clips[index].item_id
        if item_id == self._pending_id:
            return TransportState.LOADING
        if item_id == self._active_id:
            return self.transport
        return TransportState.STOPPED

    def position_ms(self, now_ms: Optional[int] = None) -> Optional[int]:
        clip = self._clip_for_id(self._active_id)
        if clip is None or self._start_ms is None:
            return None
        now = self._now_ms if now_ms is None else now_ms
        if self.transport is TransportState.PAUSED and self._paused_at_ms is not None:
            now = self._paused_at_ms
        if self.transport is TransportState.STOPPED:
            return None
        elapsed = now - self._start_ms
        if clip.repeats:
            pass_ms = clip.media_span_ms(self._backend.media_length_ms(clip))
            if pass_ms:
                elapsed %= pass_ms
        return clip.play_from_ms + int(elapsed * clip.speed)

    def status(self, now_ms: Optional[int] = None) -> PlaylistStatus:
        try:
            output_alive = bool(self._backend.output_alive())
        except Exception:  # noqa: BLE001
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
            position_ms=self.position_ms(now_ms),
            end_ms=self._end_ms if self.transport is TransportState.PLAYING else None,
            next_at_ms=self._end_ms if self._armed_id is not None else None,
            error=self.error,
            error_index=self._index_for_id(self.error_item_id),
        )


class InMemoryBackend:
    """Non-printing fake used by the dry-run TUI and tests.

    ``cue`` reports on_air immediately (``at_ms`` None) or at the scheduled
    time once ``poll_events`` is called after that time.
    """

    DEFAULT_LENGTH_MS = 10_000

    def __init__(self, auto_on_air: bool = True, clock=None):
        self.calls = []
        self.events: List[BackendEvent] = []
        self.alive = True
        self.auto_on_air = auto_on_air
        self.lengths = {}
        self._scheduled: List[BackendEvent] = []
        self._clock = clock or (lambda: 0)

    def cue(self, request_id, clip, at_ms, transition, transition_ms) -> None:
        self.calls.append(("cue", request_id, clip.item_id, at_ms, transition, transition_ms))
        if not self.auto_on_air:
            return
        event = BackendEvent("on_air", clip.item_id, request_id, at_ms=at_ms)
        (self._scheduled if at_ms is not None else self.events).append(event)

    def pause(self, item_id) -> None:
        self.calls.append(("pause", item_id))

    def resume(self, item_id) -> None:
        self.calls.append(("resume", item_id))

    def park(self, item_id) -> None:
        self.calls.append(("park", item_id))
        self._scheduled = [e for e in self._scheduled if e.item_id != item_id]

    def remove(self, item_id) -> None:
        self.calls.append(("remove", item_id))

    def media_length_ms(self, clip) -> Optional[int]:
        return self.lengths.get(clip.url, self.DEFAULT_LENGTH_MS)

    def poll_events(self) -> List[BackendEvent]:
        now = self._clock()
        due = [e for e in self._scheduled if e.at_ms <= now]
        self._scheduled = [e for e in self._scheduled if e.at_ms > now]
        events, self.events = self.events + due, []
        return events

    def output_alive(self) -> bool:
        return self.alive

    def clear(self) -> None:
        self.calls.clear()
