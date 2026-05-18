"""History-aware scene selection for auto-switching."""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Callable, Iterable, Optional

from pyplumber.mixer import MixerGraphBuilder

from .config import CANVAS_H, CANVAS_W
from .shot_rules import SHOT_RULES, profile_names


def fixed_scene_selector(layout: str) -> Callable[[int], str]:
    return lambda index: f"{layout}_{index}"


class AutoShotSceneBuilder:
    """Reusable auto-switch scene templates with dynamic slot routing."""

    PIP_SCENE = "auto_pip"
    VSTACK2_SCENE = "auto_vstack2"
    VSTACK3_SCENE = "auto_vstack3"

    def __init__(self, mx: MixerGraphBuilder, *, n_inputs: int, preheated=None) -> None:
        self._mx = mx
        self._n_inputs = n_inputs
        self._preheated = preheated

    def register_initial_scenes(self) -> None:
        if self._n_inputs >= 2:
            self.pip(0, 1)
            self.vstack2([0, 1])
        if self._n_inputs >= 3:
            self.vstack3([0, 1, 2])

    def pip(self, speaker: int, companion: int) -> str:
        companion = self._valid_companion(speaker, companion)
        pip_w = CANVAS_W // 3
        pip_h = (pip_w * 9 // 16) & ~1
        pip_x = CANVAS_W - pip_w - 16
        pip_y = 16

        if self._preheated:
            self._mx.define_scene(
                self.PIP_SCENE,
                {
                    self._preheated.source("face_full"): {
                        "graph": self._preheated.graph("face_full"),
                        "dst_x": 0,
                        "dst_y": 0,
                    },
                    self._preheated.source("orig_pip_thumb"): {
                        "graph": self._preheated.graph("orig_pip_thumb"),
                        "dst_x": pip_x,
                        "dst_y": pip_y,
                    },
                },
                controls=[
                    self._preheated.control("face_full", 0, speaker),
                    self._preheated.control("orig_pip_thumb", 0, companion),
                ],
            )
        else:
            self._mx.define_scene(
                self.PIP_SCENE,
                {
                    f"face_{speaker}": {
                        "graph": f"scale_cuda=w={CANVAS_W}:h={CANVAS_H}:interp_algo=lanczos",
                        "dst_x": 0,
                        "dst_y": 0,
                    },
                    f"orig_{companion}": {
                        "graph": f"scale_cuda=w={pip_w}:h={pip_h}:interp_algo=lanczos",
                        "dst_x": pip_x,
                        "dst_y": pip_y,
                    },
                },
            )
        return self.PIP_SCENE

    def vstack2(self, speakers: list[int]) -> str:
        cams = self._fill_unique(speakers, 2)
        tile_h = 608
        gap = (CANVAS_H - 2 * tile_h) // 2
        self._define_stack_scene(self.VSTACK2_SCENE, cams, top=gap, tile_h=tile_h)
        return self.VSTACK2_SCENE

    def vstack3(self, speakers: list[int]) -> str:
        cams = self._fill_unique(speakers, 3)
        tile_h = 608
        top = (CANVAS_H - 3 * tile_h) // 2
        self._define_stack_scene(self.VSTACK3_SCENE, cams, top=top, tile_h=tile_h)
        return self.VSTACK3_SCENE

    def _define_stack_scene(self, scene_name: str, cams: list[int], *, top: int, tile_h: int) -> None:
        graph = f"scale_cuda=w={CANVAS_W}:h={tile_h}:interp_algo=lanczos"
        sources = {}
        controls = []
        for slot, cam in enumerate(cams):
            if self._preheated:
                source = self._preheated.source("orig_stack", slot)
                sources[source] = {
                    "graph": self._preheated.graph("orig_stack"),
                    "dst_x": 0,
                    "dst_y": top + slot * tile_h,
                }
                controls.append(self._preheated.control("orig_stack", slot, cam))
            else:
                sources[f"orig_{cam}"] = {
                    "graph": graph,
                    "dst_x": 0,
                    "dst_y": top + slot * tile_h,
                }
        self._mx.define_scene(scene_name, sources, controls=controls)

    def _valid_companion(self, speaker: int, companion: int) -> int:
        if companion != speaker and 0 <= companion < self._n_inputs:
            return companion
        for i in range(self._n_inputs):
            if i != speaker:
                return i
        return speaker

    def _fill_unique(self, speakers: list[int], count: int) -> list[int]:
        result = []
        for speaker in speakers:
            if 0 <= speaker < self._n_inputs and speaker not in result:
                result.append(speaker)
        for i in range(self._n_inputs):
            if len(result) >= count:
                break
            if i not in result:
                result.append(i)
        return result[:count]


class HistoryAwareShotSelector:
    """Choose a concrete mixer scene from dominant-speaker decisions.

    The selector keeps only switch-event history.  When the conversation moves
    quickly between two or three distinct speakers, it prefers stack scenes
    containing those speakers.  Outside that window it picks a single-speaker
    scene family from the configured weighted profile.
    """

    def __init__(
        self,
        scenes: Iterable[str],
        *,
        n_inputs: int,
        profile_name: str = "talkshow-balanced",
        fallback_layout: str = "videoconf",
        scene_builder: Optional[AutoShotSceneBuilder] = None,
        seed: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if profile_name not in SHOT_RULES:
            raise ValueError(f"unknown shot profile: {profile_name}")
        self.rules = SHOT_RULES[profile_name]
        self._scenes = set(scenes)
        self._n_inputs = n_inputs
        self._fallback_layout = fallback_layout
        self._scene_builder = scene_builder
        self._rng = random.Random(seed)
        self._clock = clock
        self._speaker_history: deque[tuple[int, float]] = deque(maxlen=12)
        self._recent_scenes: deque[str] = deque(maxlen=self.rules.avoid_recent_scenes)
        self._history_version = 0
        self._active_stack_signature: Optional[tuple[int, ...]] = None
        self._active_stack_until = 0.0
        self._active_stack_history_version = 0
        self._manual_suggestion_family: Optional[str] = None
        self._manual_suggestion_speaker: Optional[int] = None
        self._manual_suggestion_until = 0.0

    def initial_scene(self, speaker_index: int = 0) -> str:
        for family in (self._fallback_layout, "videoconf", "full_face"):
            scene = f"{family}_{speaker_index}"
            if scene in self._scenes:
                return scene
        scene = self._weighted_single_speaker_scene(speaker_index)
        if scene is None:
            raise ValueError(f"no auto-switch scene available for speaker {speaker_index}")
        return scene

    def __call__(self, speaker_index: int) -> str:
        now = self._clock()
        self._remember_speaker(speaker_index, now)

        recent = self._recent_distinct_speakers(now)
        scene = self._manual_suggestion_scene(speaker_index, recent, now)
        if scene is None:
            scene = self._conversation_scene(recent, now)
        if scene is None:
            scene = self._weighted_single_speaker_scene(speaker_index, recent)
        if scene is None:
            scene = self.initial_scene(speaker_index)

        self._recent_scenes.append(scene)
        return scene

    def observe_program_scene(self, scene_name: str, now: Optional[float] = None) -> None:
        """Treat an externally selected PGM scene as a short-lived layout hint."""
        if now is None:
            now = self._clock()
        family, speaker_index = self._parse_manual_suggestion(scene_name)
        if family is None or self.rules.manual_suggestion_window_s <= 0:
            self._clear_manual_suggestion()
            return

        self._manual_suggestion_family = family
        self._manual_suggestion_speaker = speaker_index
        self._manual_suggestion_until = now + self.rules.manual_suggestion_window_s
        self._active_stack_signature = None
        self._active_stack_until = 0.0
        print(
            f"[auto_switcher] manual layout suggestion: {family} "
            f"for {self.rules.manual_suggestion_window_s:.1f}s from {scene_name}",
            flush=True,
        )

    def _remember_speaker(self, speaker_index: int, now: float) -> None:
        if not 0 <= speaker_index < self._n_inputs:
            return
        if self._speaker_history and self._speaker_history[-1][0] == speaker_index:
            self._speaker_history[-1] = (speaker_index, now)
            return
        self._speaker_history.append((speaker_index, now))
        self._history_version += 1

    def _recent_distinct_speakers(self, now: float) -> list[int]:
        speakers: list[int] = []
        cutoff = now - self.rules.conversation_window_s
        for speaker, ts in reversed(self._speaker_history):
            if ts < cutoff:
                break
            if speaker not in speakers:
                speakers.append(speaker)
            if len(speakers) >= 3:
                break
        return speakers

    def _manual_suggestion_scene(
        self,
        speaker_index: int,
        recent_speakers: list[int],
        now: float,
    ) -> Optional[str]:
        family = self._manual_suggestion_family
        if family is None:
            return None
        if now >= self._manual_suggestion_until:
            self._clear_manual_suggestion()
            return None

        scene = self._scene_for_family(family, speaker_index, recent_speakers)
        if scene is None:
            self._clear_manual_suggestion()
            return None

        if self._manual_suggestion_speaker is None or speaker_index != self._manual_suggestion_speaker:
            self._clear_manual_suggestion()
        return scene

    def _clear_manual_suggestion(self) -> None:
        self._manual_suggestion_family = None
        self._manual_suggestion_speaker = None
        self._manual_suggestion_until = 0.0

    def _parse_manual_suggestion(self, scene_name: str) -> tuple[Optional[str], Optional[int]]:
        for family in self.rules.manual_suggestion_families:
            if family in ("videoconf", "full_face"):
                prefix = f"{family}_"
                if not scene_name.startswith(prefix):
                    continue
                speaker = self._parse_int_suffix(scene_name[len(prefix):])
                if speaker is not None and 0 <= speaker < self._n_inputs:
                    return family, speaker
            elif family == "pip":
                parts = scene_name.split("_")
                if len(parts) >= 3 and parts[0] == "pip":
                    speaker = self._parse_int_suffix(parts[1])
                    if speaker is not None and 0 <= speaker < self._n_inputs:
                        return family, speaker
        return None, None

    @staticmethod
    def _parse_int_suffix(value: str) -> Optional[int]:
        try:
            return int(value)
        except ValueError:
            return None

    def _conversation_scene(self, recent_speakers: list[int], now: float) -> Optional[str]:
        if (
            self._active_stack_signature is not None
            and now >= self._active_stack_until
            and self._history_version <= self._active_stack_history_version
        ):
            return None
        for rule in self.rules.stack_rules:
            if len(recent_speakers) < rule.distinct_speakers:
                continue
            speakers = recent_speakers[:rule.distinct_speakers]
            signature = (rule.distinct_speakers, *speakers)
            if self._active_stack_signature == signature and now >= self._active_stack_until:
                return None
            scene = self._stack_scene(rule.shot_family, speakers)
            if scene is None:
                continue
            if self._active_stack_signature != signature:
                self._active_stack_signature = signature
                self._active_stack_until = now + self.rules.stack_hold_s
                self._active_stack_history_version = self._history_version
            return scene
        return None

    def _weighted_single_speaker_scene(
        self,
        speaker_index: int,
        recent_speakers: Optional[list[int]] = None,
    ) -> Optional[str]:
        weighted_families: list[tuple[str, int]] = []
        for family, weight in self.rules.single_speaker_weights:
            if self._family_available(family, speaker_index):
                weighted_families.append((family, weight))
        if not weighted_families:
            return None

        total = sum(weight for _, weight in weighted_families)
        pick = self._rng.uniform(0, total)
        cursor = 0.0
        for family, weight in weighted_families:
            cursor += weight
            if pick <= cursor:
                return self._scene_for_family(family, speaker_index, recent_speakers or [])
        return self._scene_for_family(weighted_families[-1][0], speaker_index, recent_speakers or [])

    def _family_available(self, family: str, speaker_index: int) -> bool:
        if family == "pip" and self._scene_builder is not None:
            return self._n_inputs >= 2
        return bool(self._family_candidates(family, speaker_index))

    def _scene_for_family(self, family: str, speaker_index: int, recent_speakers: list[int]) -> Optional[str]:
        if family == "pip" and self._scene_builder is not None:
            companion = self._pip_companion(speaker_index, recent_speakers)
            return self._scene_builder.pip(speaker_index, companion)
        candidates = self._family_candidates(family, speaker_index)
        return self._choose_scene(candidates) if candidates else None

    def _pip_companion(self, speaker_index: int, recent_speakers: list[int]) -> int:
        for speaker in recent_speakers:
            if speaker != speaker_index:
                return speaker
        candidates = [i for i in range(self._n_inputs) if i != speaker_index]
        return self._rng.choice(candidates) if candidates else speaker_index

    def _family_candidates(self, family: str, speaker_index: int) -> list[str]:
        if family in ("videoconf", "full_face"):
            scene = f"{family}_{speaker_index}"
            return [scene] if scene in self._scenes else []
        if family == "pip":
            prefix = f"pip_{speaker_index}_"
            return sorted(s for s in self._scenes if s.startswith(prefix))
        return []

    def _stack_scene(self, family: str, speakers: list[int]) -> Optional[str]:
        if self._scene_builder is not None:
            if family == "vstack2":
                return self._scene_builder.vstack2(speakers)
            if family == "vstack3":
                return self._scene_builder.vstack3(speakers)
        if family == "vstack2":
            exact = f"vstack_{speakers[0]}_{speakers[1]}"
            prefix = "vstack_"
            expected_len = 3
        elif family == "vstack3":
            exact = f"vstack3_{speakers[0]}_{speakers[1]}_{speakers[2]}"
            prefix = "vstack3_"
            expected_len = 4
        else:
            return None

        if exact in self._scenes:
            return exact

        wanted = set(speakers)
        containing_all: list[str] = []
        containing_current: list[str] = []
        for scene in self._scenes:
            if not scene.startswith(prefix):
                continue
            parts = scene.split("_")
            if len(parts) != expected_len:
                continue
            try:
                cams = {int(p) for p in parts[1:]}
            except ValueError:
                continue
            if wanted.issubset(cams):
                containing_all.append(scene)
            elif speakers[0] in cams:
                containing_current.append(scene)

        if containing_all:
            return self._choose_scene(containing_all)
        if containing_current:
            return self._choose_scene(containing_current)
        return None

    def _choose_scene(self, candidates: list[str]) -> str:
        fresh = [scene for scene in candidates if scene not in self._recent_scenes]
        pool = fresh or candidates
        return self._rng.choice(sorted(pool))
