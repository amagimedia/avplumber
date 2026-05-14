"""Voice Activity Detection based on per-frame RMS energy.

The ``RmsVadNode`` is a ``python_node_si`` (single audio input, no output)
that runs a simple energy-level state machine and writes results to a
shared ``Speaker`` registry.  Everything lives in a single module so that
swapping to a neural VAD (e.g. Silero-VAD) is a one-class change:

    class SileroVadNode(RmsVadNode):
        def _classify(self, samples) -> float:
            # call ONNX runtime here, return probability in [0, 1]
            ...

Usage
-----
    from pyplumber.audio_vad import Speaker, RmsVadNode

    registry = Speaker()

    for i, audio_edge in enumerate(audio_edges):
        node = RmsVadNode(
            {"src": audio_edge, "group": f"input_{i}", "name": f"vad_{i}"},
            index=i,
            registry=registry,
        )
        avp.addNode(node)
"""

from __future__ import annotations

import math
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .node import PythonNode

# AV_SAMPLE_FMT_FLTP = 3  (float planar; standard FFmpeg enum value)
_AV_SAMPLE_FMT_FLTP = 3


@dataclass
class SpeakerEntry:
    """Latest VAD state for one input."""
    index: int
    speaking: bool = False
    level_db: float = -96.0
    # Wall-clock time (time.monotonic()) of last state transition.
    last_change_ts: float = field(default_factory=time.monotonic)
    # Accumulated continuous speaking duration at the last update.
    speaking_duration_s: float = 0.0


class Speaker:
    """Thread-safe registry of per-input speaking state.

    Updated by ``RmsVadNode`` threads; read by ``AutoSwitcher``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[int, SpeakerEntry] = {}

    def update(
        self,
        index: int,
        speaking: bool,
        level_db: float,
        speaking_duration_s: float,
    ) -> None:
        with self._lock:
            entry = self._entries.get(index)
            if entry is None:
                entry = SpeakerEntry(index=index)
                self._entries[index] = entry
            if entry.speaking != speaking:
                entry.last_change_ts = time.monotonic()
            entry.speaking = speaking
            entry.level_db = level_db
            entry.speaking_duration_s = speaking_duration_s

    def get(self, index: int) -> Optional[SpeakerEntry]:
        with self._lock:
            return self._entries.get(index)

    def all(self) -> List[SpeakerEntry]:
        with self._lock:
            return list(self._entries.values())

    def loudest_speaker(self) -> Optional[SpeakerEntry]:
        """Return the speaking input with the highest sustained level."""
        with self._lock:
            candidates = [e for e in self._entries.values() if e.speaking]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.level_db)


class RmsVadNode(PythonNode):
    """Audio sink that computes per-frame RMS and drives a hysteresis state machine.

    Expects the audio edge to carry **fltp** (float planar, AV_SAMPLE_FMT_FLTP=3)
    samples, which is the output format of a ``resample_audio`` node configured
    with ``dst_sample_format: "fltp"``.

    State machine
    -------------
    - *inactive* → *active*: when the rolling-window mean exceeds
      ``enter_db`` continuously for at least ``hold_on_s`` seconds.
    - *active* → *inactive*: when the rolling-window mean drops below
      ``leave_db`` for at least ``hold_off_s`` seconds.

    Subclass and override ``_classify`` to replace RMS with a neural model
    (e.g. Silero-VAD) while keeping the rest of the logic intact.
    """

    def __init__(
        self,
        args: dict,
        index: int,
        registry: Speaker,
        enter_db: float = -35.0,
        leave_db: float = -45.0,
        hold_on_s: float = 0.3,
        hold_off_s: float = 0.5,
        window_s: float = 0.2,
    ) -> None:
        if "dst" in args:
            raise ValueError("RmsVadNode is a sink (python_node_si): do not pass 'dst'")
        super().__init__(args)
        self._index = index
        self._registry = registry
        self._enter_db = enter_db
        self._leave_db = leave_db
        self._hold_on_s = hold_on_s
        self._hold_off_s = hold_off_s
        self._window_s = window_s

        self._speaking = False
        # Wall-clock time when we entered the current candidate state.
        self._candidate_since: Optional[float] = None
        self._candidate_state: Optional[bool] = None

        # For measuring speaking duration.
        self._speaking_start: Optional[float] = None
        self._speaking_accum: float = 0.0

        # Rolling window: list of (timestamp, level_db) tuples.
        self._window: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Override this to plug in a neural model
    # ------------------------------------------------------------------

    def _classify(self, samples) -> float:
        """Return a probability in [0, 1] that audio is active speech.

        The default implementation maps RMS dB to a linear ramp between
        ``leave_db`` (→ 0.0) and ``enter_db`` (→ 1.0).

        Subclasses may override this to use a neural model.  ``samples``
        is an ``av::AudioSamples`` object with ``samplesCount``,
        ``sampleRate``, ``channelsCount``, and ``data(channel) -> bytes``.
        The override should NOT call super(); ``_rms_db`` is available as a
        helper if the model still needs energy as a feature.
        """
        level_db = self._rms_db(samples)
        if level_db >= self._enter_db:
            return 1.0
        if level_db <= self._leave_db:
            return 0.0
        span = self._enter_db - self._leave_db
        return (level_db - self._leave_db) / span

    # ------------------------------------------------------------------
    # process() is called by the C++ thread loop
    # ------------------------------------------------------------------

    def _rms_db(self, samples) -> float:
        """Compute peak-channel RMS level in dBFS from an AudioSamples object."""
        if not samples.isComplete or samples.samplesCount == 0:
            return -96.0
        if samples.sampleFormat != _AV_SAMPLE_FMT_FLTP:
            return -96.0
        n = samples.samplesCount
        peak_rms_sq = 0.0
        for ch in range(samples.channelsCount):
            raw = samples.data(ch)
            if not raw:
                continue
            floats = struct.unpack(f"{n}f", raw[:n * 4])
            ms = sum(x * x for x in floats) / n
            if ms > peak_rms_sq:
                peak_rms_sq = ms
        if peak_rms_sq <= 0.0:
            return -96.0
        return 20.0 * math.log10(math.sqrt(peak_rms_sq) + 1e-12)

    def process(self) -> None:
        samples = self._src.tryGet(timeout_ms=100)
        if samples is None:
            return
        if not samples.isComplete:
            return

        now = time.monotonic()
        level_db = self._rms_db(samples)
        prob = self._classify(samples)

        # Rolling window of (timestamp, probability).
        self._window.append((now, prob))
        cutoff = now - self._window_s
        self._window = [(t, p) for t, p in self._window if t >= cutoff]

        mean_prob = sum(p for _, p in self._window) / len(self._window) if self._window else 0.0

        # Hysteresis thresholds on probability: 0.5 to enter, 0.3 to stay active.
        enter_prob = 0.5
        leave_prob = 0.3
        candidate = (mean_prob >= enter_prob) if not self._speaking else (mean_prob >= leave_prob)

        if candidate != self._candidate_state:
            self._candidate_state = candidate
            self._candidate_since = now

        hold = self._hold_on_s if candidate else self._hold_off_s
        if (
            self._candidate_since is not None
            and (now - self._candidate_since) >= hold
            and candidate != self._speaking
        ):
            self._speaking = candidate
            if self._speaking:
                self._speaking_start = now
            else:
                if self._speaking_start is not None:
                    self._speaking_accum += now - self._speaking_start
                self._speaking_start = None

        speaking_dur = self._speaking_accum
        if self._speaking and self._speaking_start is not None:
            speaking_dur += now - self._speaking_start

        self._registry.update(
            index=self._index,
            speaking=self._speaking,
            level_db=level_db,
            speaking_duration_s=speaking_dur,
        )
