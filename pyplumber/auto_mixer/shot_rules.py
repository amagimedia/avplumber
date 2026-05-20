"""Shot-selection rules for auto mixer profiles."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MANUAL_SUGGESTION_WINDOW_S = 15.0
DEFAULT_MANUAL_SUGGESTION_FAMILIES = (
    "videoconf",
    "full_face",
    "pip",
    "vstack2",
    "vstack3",
)


@dataclass(frozen=True)
class StackRule:
    distinct_speakers: int
    shot_family: str


@dataclass(frozen=True)
class ShotRules:
    name: str
    single_speaker_weights: tuple[tuple[str, int], ...]
    stack_rules: tuple[StackRule, ...]
    conversation_window_s: float
    stack_hold_s: float
    manual_suggestion_window_s: float
    manual_suggestion_families: tuple[str, ...]
    avoid_recent_scenes: int


SHOT_RULES: dict[str, ShotRules] = {
    "talkshow-conference": ShotRules(
        name="talkshow-conference",
        single_speaker_weights=(("videoconf", 100),),
        stack_rules=(
            StackRule(distinct_speakers=3, shot_family="vstack3"),
            StackRule(distinct_speakers=2, shot_family="vstack2"),
        ),
        conversation_window_s=6.0,
        stack_hold_s=4.0,
        manual_suggestion_window_s=DEFAULT_MANUAL_SUGGESTION_WINDOW_S,
        manual_suggestion_families=DEFAULT_MANUAL_SUGGESTION_FAMILIES,
        avoid_recent_scenes=3,
    ),
    "talkshow-balanced": ShotRules(
        name="talkshow-balanced",
        single_speaker_weights=(("videoconf", 50), ("full_face", 30), ("pip", 20)),
        stack_rules=(
            StackRule(distinct_speakers=3, shot_family="vstack3"),
            StackRule(distinct_speakers=2, shot_family="vstack2"),
        ),
        conversation_window_s=6.0,
        stack_hold_s=4.0,
        manual_suggestion_window_s=DEFAULT_MANUAL_SUGGESTION_WINDOW_S,
        manual_suggestion_families=DEFAULT_MANUAL_SUGGESTION_FAMILIES,
        avoid_recent_scenes=3,
    ),
    "talkshow-variety": ShotRules(
        name="talkshow-variety",
        single_speaker_weights=(("videoconf", 35), ("full_face", 30), ("pip", 35)),
        stack_rules=(
            StackRule(distinct_speakers=3, shot_family="vstack3"),
            StackRule(distinct_speakers=2, shot_family="vstack2"),
        ),
        conversation_window_s=8.0,
        stack_hold_s=5.0,
        manual_suggestion_window_s=DEFAULT_MANUAL_SUGGESTION_WINDOW_S,
        manual_suggestion_families=DEFAULT_MANUAL_SUGGESTION_FAMILIES,
        avoid_recent_scenes=4,
    ),
}


def profile_names() -> tuple[str, ...]:
    return tuple(SHOT_RULES)
