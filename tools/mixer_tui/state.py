"""Typed status parsing for the mixer TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MixerStatus:
    pgm_scene: str = ""
    pvw_scene: str = ""
    transition: str = "idle"


@dataclass(frozen=True)
class AutoSwitchStatus:
    transition_mode: Optional[str] = None
    fade_duration_s: Optional[float] = None
    wipe_file: Optional[str] = None


def parse_mixer_status(content: str) -> MixerStatus:
    data = json.loads(content.strip())
    return MixerStatus(
        pgm_scene=data.get("pgm_scene", ""),
        pvw_scene=data.get("pvw_scene", ""),
        transition=data.get("transition", "idle"),
    )


def parse_auto_switch_status(content: str) -> AutoSwitchStatus:
    data = json.loads(content.strip())
    duration = data.get("fade_duration_s")
    return AutoSwitchStatus(
        transition_mode=data.get("transition_mode") if isinstance(data.get("transition_mode"), str) else None,
        fade_duration_s=float(duration) if duration is not None else None,
        wipe_file=data.get("wipe_file") if isinstance(data.get("wipe_file"), str) else None,
    )


def parse_scene_list(content: str) -> list[str]:
    data = json.loads(content.strip())
    if not isinstance(data, list):
        raise ValueError("mixer.scenes response is not a list")
    return [str(item) for item in data]
