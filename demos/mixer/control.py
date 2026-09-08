"""Generic mixer control protocol helpers shared by the TUI and tests."""

from __future__ import annotations

import json
from dataclasses import dataclass



@dataclass(frozen=True)
class MixerStatus:
    pgm_scene: str = ""
    pvw_scene: str = ""
    transition: str = "idle"


def mixer_command(command: str, mixer: str, **payload) -> str:
    payload.setdefault("mixer", mixer)
    return f"mixer.{command} {json.dumps(payload, separators=(',', ':'))}"


def parse_mixer_status(content: str) -> MixerStatus:
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("mixer.status response must be an object")
    return MixerStatus(
        pgm_scene=str(data.get("pgm_scene", "")),
        pvw_scene=str(data.get("pvw_scene", "")),
        transition=str(data.get("transition", "idle")),
    )


def parse_scene_list(content: str) -> list[str]:
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError("mixer.scenes response must be a list")
    return [str(scene) for scene in data]
