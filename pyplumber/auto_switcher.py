"""Auto-switcher policy for a pyplumber video mixer.

``AutoSwitcher`` runs a background thread that periodically reads the
``Speaker`` registry and issues ``MixerGraphBuilder.fade()`` calls when the
active speaker changes.

Usage
-----
    from pyplumber.auto_switcher import AutoSwitcher

    switcher = AutoSwitcher(
        mixer=mx,
        registry=speaker_registry,
        n_inputs=2,
        scene_for_input=lambda i: f"videoconf_{i}",
        min_dwell_speaking_s=0.6,
        min_dwell_program_s=2.5,
        fade_duration_s=0.6,
    )
    switcher.start()
    # ... run the pipeline ...
    switcher.stop()
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .audio_vad import Speaker
from .mixer import MixerGraphBuilder


class AutoSwitcher:
    """Background policy thread that drives a ``MixerGraphBuilder``.

    Decision logic (runs every ``tick_s`` seconds)
    -----------------------------------------------
    1. If a priority VAD-only speaker is active, choose that input without
       requiring visual speech or loudest-camera status.
    2. Otherwise, find the input where Silero audio VAD and visual speech are
       both active, the measured RMS level is above ``min_active_level_db``,
       and the input has held that state for at least
       ``min_dwell_speaking_s`` seconds. Choose the highest audio loudness
       when several inputs qualify.
    3. If that input differs from the current program AND the last
       transition completed at least ``min_dwell_program_s`` seconds ago,
       trigger a scene change to the configured scene for that input.
    4. A ``cooldown_s`` lockout after each transition prevents oscillation
       even if min_dwell_program_s has elapsed.
    """

    def __init__(
        self,
        mixer: MixerGraphBuilder,
        registry: Speaker,
        n_inputs: int,
        scene_for_input: Callable[[int], str] = lambda i: f"full_face_{i}",
        min_dwell_speaking_s: float = 0.6,
        min_dwell_program_s: float = 2.5,
        fade_duration_s: float = 0.6,
        cooldown_s: float = 1.5,
        tick_s: float = 0.25,
        min_active_level_db: float = -60.0,
        switch_pts_lead_ms: Optional[int] = None,
        special_speaker_index: Optional[int] = None,
        special_speaker_margin_db: float = 0.0,
        vad_only_priority_speaker_index: Optional[int] = None,
        program_scene_getter: Optional[Callable[[], Optional[str]]] = None,
        program_scene_poll_s: float = 0.5,
    ) -> None:
        self._mixer = mixer
        self._registry = registry
        self._n_inputs = n_inputs
        self._scene_for_input = scene_for_input
        self._min_dwell_speaking = min_dwell_speaking_s
        self._min_dwell_program = min_dwell_program_s
        self._fade_duration = fade_duration_s
        self._cooldown = cooldown_s
        self._tick = tick_s
        self._min_active_level_db = min_active_level_db
        self._switch_pts_lead_ms = switch_pts_lead_ms
        self._special_speaker_index = special_speaker_index
        self._special_speaker_margin_db = special_speaker_margin_db
        self._vad_only_priority_speaker_index = vad_only_priority_speaker_index
        self._program_scene_getter = program_scene_getter
        self._program_scene_poll_s = program_scene_poll_s

        self._current_input: Optional[int] = None
        self._current_scene: Optional[str] = None
        self._last_switch_ts: float = 0.0
        self._next_program_scene_poll_ts: float = 0.0
        self._pending_auto_scene: Optional[str] = None
        self._pending_auto_scene_until: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AutoSwitcher")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Manual override helpers (can be called from any thread)
    # ------------------------------------------------------------------

    def force_scene(self, scene_name: str) -> None:
        """Immediately cut to an arbitrary scene and reset the dwell timer."""
        self._mixer.cut(scene_name)
        self._current_scene = scene_name
        self._last_switch_ts = time.monotonic()

    def force_input(self, index: int) -> None:
        """Cut to the auto-switch scene for *index* and reset the dwell timer."""
        scene = self._scene_for_input(index)
        if scene in self._mixer.scenes():
            self._mixer.cut(scene)
            self._current_input = index
            self._current_scene = scene
            self._last_switch_ts = time.monotonic()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._tick_once()
            self._stop_event.wait(timeout=self._tick)

    def _switch_start_pts_ms(self, entry, now: float) -> Optional[int]:
        if self._switch_pts_lead_ms is None:
            return None
        if entry.audio_event_pts_ms is None or entry.audio_event_observed_ts is None:
            return None
        estimated_media_now_ms = entry.audio_event_pts_ms + int(
            round((now - entry.audio_event_observed_ts) * 1000.0)
        )
        return estimated_media_now_ms + self._switch_pts_lead_ms

    def _try_switch_to(self, index: int, now: float, entry=None) -> bool:
        if (now - self._last_switch_ts) < self._min_dwell_program:
            return False

        scene = self._scene_for_input(index)
        if scene == self._current_scene:
            self._current_input = index
            return True
        if scene not in self._mixer.scenes():
            return False

        start_pts_ms = self._switch_start_pts_ms(entry, now) if entry is not None else None
        try:
            #self._mixer.fade(scene, duration_sec=self._fade_duration)
            if start_pts_ms is None:
                self._mixer.cut(scene)
            else:
                self._mixer.cut(scene, start_pts_ms=start_pts_ms)
        except Exception as exc:
            print(
                f"[auto_switcher] switch to {scene} failed"
                f"{f' at pts={start_pts_ms}ms' if start_pts_ms is not None else ''}: {exc}",
                flush=True,
            )
            return False
        print(
            f"[auto_switcher] switch to {scene}"
            f"{f' at pts={start_pts_ms}ms' if start_pts_ms is not None else ''}",
            flush=True,
        )
        self._current_input = index
        self._current_scene = scene
        self._pending_auto_scene = scene
        self._pending_auto_scene_until = now + 3.0
        self._last_switch_ts = now
        return True

    def _tick_once(self) -> None:
        now = time.monotonic()
        self._observe_program_scene(now)

        # Respect post-transition cooldown.
        if (now - self._last_switch_ts) < self._cooldown:
            return

        priority_index = self._vad_only_priority_speaker_index
        if priority_index is not None:
            entry = self._registry.get(priority_index)
            if entry is not None and entry.speaking:
                speaking_since = entry.audio_since or entry.last_change_ts
                speaking_duration_s = now - speaking_since
                if speaking_duration_s >= self._min_dwell_speaking:
                    self._try_switch_to(priority_index, now, entry)
                    return

        # Find candidates: audio VAD and visual speech both active, with usable RMS.
        candidates = []

        for i in range(self._n_inputs):
            entry = self._registry.get(i)
            if entry is None:
                continue
            if not entry.speaking or not entry.visual_speaking:
                continue
            if entry.combined_speaking_since is None:
                continue
            speaking_duration_s = now - entry.combined_speaking_since
            if speaking_duration_s < self._min_dwell_speaking:
                continue
            if entry.level_db <= self._min_active_level_db:
                continue
            candidates.append(entry)

        if not candidates:
            return

        candidates.sort(key=lambda e: e.level_db, reverse=True)
        best = candidates[0]
        if (
            self._special_speaker_index is not None
            and best.index == self._special_speaker_index
            and len(candidates) > 1
            and best.level_db < candidates[1].level_db + self._special_speaker_margin_db
        ):
            best = candidates[1]
        best_index = best.index

        self._try_switch_to(best_index, now, best)

    def _observe_program_scene(self, now: float) -> None:
        if self._program_scene_getter is None or now < self._next_program_scene_poll_ts:
            return
        self._next_program_scene_poll_ts = now + self._program_scene_poll_s
        try:
            scene = self._program_scene_getter()
        except Exception:
            return
        if not scene:
            return

        if self._pending_auto_scene is not None:
            if scene == self._pending_auto_scene:
                self._pending_auto_scene = None
                self._current_scene = scene
            elif now < self._pending_auto_scene_until:
                return
            else:
                self._pending_auto_scene = None

        if self._current_scene is None:
            self._current_scene = scene
            return
        if scene == self._current_scene:
            return

        self._current_scene = scene
        self._last_switch_ts = now
        observer = getattr(self._scene_for_input, "observe_program_scene", None)
        if observer is not None:
            observer(scene, now)
