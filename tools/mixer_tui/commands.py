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
    overlay_source_otm_name: str,
    overlay_otm_name: str,
    overlay_selector_name: str,
) -> list[str]:
    if enabled:
        return [
            f"node.object.set {overlay_source_otm_name} outputs 1",
            f"node.object.set {overlay_otm_name} outputs 3",
            f"node.object.set {overlay_selector_name} active 1",
        ]
    return [
        f"node.object.set {overlay_otm_name} outputs 3",
        f"node.object.set {overlay_selector_name} active 0",
        f"node.object.set {overlay_otm_name} outputs 1",
        f"node.object.set {overlay_source_otm_name} outputs 0",
    ]
