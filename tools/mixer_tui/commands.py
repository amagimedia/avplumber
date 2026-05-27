"""Command builders for avplumber mixer and TUI utility actions."""

from __future__ import annotations

import json
from typing import Any, Optional

TAKE_COMMAND_PREFIXES = ("mixer.cut", "mixer.fade", "mixer.wipe")
AI_TRANSITION_MODES = ("cut", "fade", "wipe")


def mixer_command(command: str, mixer_name: str, **payload: Any) -> str:
    payload.setdefault("mixer", mixer_name)
    return f"mixer.{command} {json.dumps(payload, separators=(',', ':'))}"


def auto_switch_set_command(
    *,
    fade_duration_s: float,
    transition_mode: Optional[str] = None,
    wipe_file: Optional[str] = None,
) -> str:
    payload: dict[str, Any] = {"fade_duration_s": fade_duration_s}
    if transition_mode is not None:
        payload["transition_mode"] = transition_mode
    if wipe_file:
        payload["wipe_file"] = wipe_file
    return f"auto_switch.set {json.dumps(payload, separators=(',', ':'))}"


def overlay_toggle_commands(
    *,
    enabled: bool,
    mixer_name: str,
    overlay_source_otm_name: str,
    overlay_otm_name: str,
    overlay_selector_name: str,
) -> list[str]:
    payload = {
        "mixer": mixer_name,
        "enabled": enabled,
        "source_otm": overlay_source_otm_name,
        "overlay_otm": overlay_otm_name,
        "selector": overlay_selector_name,
    }
    return [f"mixer.overlay {json.dumps(payload, separators=(',', ':'))}"]
