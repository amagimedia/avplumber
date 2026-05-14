"""Auto-switcher policy for a pyplumber video mixer.

``AutoSwitcher`` runs a background thread that periodically reads the
``Speaker`` registry (written by ``RmsVadNode`` instances) and issues
``MixerGraphBuilder.fade()`` calls when the active speaker changes.

Usage
-----
    from pyplumber.auto_switcher import AutoSwitcher

    switcher = AutoSwitcher(
        mixer=mx,
        registry=speaker_registry,
        n_inputs=2,
        full_face_scene=lambda i: f"full_face_{i}",
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
    1. Find the input that has been speaking continuously for at least
       ``min_dwell_speaking_s`` seconds with the highest level.
    2. If that input differs from the current program AND the last
       transition completed at least ``min_dwell_program_s`` seconds ago,
       trigger a ``mixer.fade`` to ``full_face_<candidate>``.
    3. A ``cooldown_s`` lockout after each transition prevents oscillation
       even if min_dwell_program_s has elapsed.
    """

    def __init__(
        self,
        mixer: MixerGraphBuilder,
        registry: Speaker,
        n_inputs: int,
        full_face_scene: Callable[[int], str] = lambda i: f"full_face_{i}",
        min_dwell_speaking_s: float = 0.6,
        min_dwell_program_s: float = 2.5,
        fade_duration_s: float = 0.6,
        cooldown_s: float = 1.5,
        tick_s: float = 0.25,
    ) -> None:
        self._mixer = mixer
        self._registry = registry
        self._n_inputs = n_inputs
        self._full_face_scene = full_face_scene
        self._min_dwell_speaking = min_dwell_speaking_s
        self._min_dwell_program = min_dwell_program_s
        self._fade_duration = fade_duration_s
        self._cooldown = cooldown_s
        self._tick = tick_s

        self._current_input: Optional[int] = None
        self._last_switch_ts: float = 0.0
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
        self._last_switch_ts = time.monotonic()

    def force_input(self, index: int) -> None:
        """Cut to the fullscreen face scene for *index* and reset the dwell timer."""
        scene = self._full_face_scene(index)
        if scene in self._mixer.scenes():
            self._mixer.cut(scene)
            self._current_input = index
            self._last_switch_ts = time.monotonic()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._tick_once()
            self._stop_event.wait(timeout=self._tick)

    def _tick_once(self) -> None:
        now = time.monotonic()

        # Respect post-transition cooldown.
        if (now - self._last_switch_ts) < self._cooldown:
            return

        # Find the best candidate: speaking for long enough.
        best_index: Optional[int] = None
        best_level: float = -999.0

        for i in range(self._n_inputs):
            entry = self._registry.get(i)
            if entry is None:
                continue
            if not entry.speaking:
                continue
            if entry.speaking_duration_s < self._min_dwell_speaking:
                continue
            if entry.level_db > best_level:
                best_level = entry.level_db
                best_index = i

        if best_index is None:
            return

        # Same as current program — nothing to do.
        if best_index == self._current_input:
            return

        # Respect minimum time between program changes.
        if (now - self._last_switch_ts) < self._min_dwell_program:
            return

        scene = self._full_face_scene(best_index)
        if scene not in self._mixer.scenes():
            return

        self._mixer.fade(scene, duration_sec=self._fade_duration)
        self._current_input = best_index
        self._last_switch_ts = now
