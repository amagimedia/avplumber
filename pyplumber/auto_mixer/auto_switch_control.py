"""Auto-switch policy settings exposed through the main AVPlumber control port."""

from __future__ import annotations

import json
from typing import Any, Dict

from pyplumber.auto_switcher import AutoSwitcher
from pyplumber.auto_mixer.native_exceptions import NativeExceptionRegistry


def _parse_set_payload(payload: str) -> Dict[str, Any]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON payload: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("payload must be a JSON object")
    allowed = {"transition_mode", "fade_duration_s", "wipe_file"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(unknown)}")
    return data


def auto_switch_status(
    switcher: AutoSwitcher,
    native_exceptions: NativeExceptionRegistry | None = None,
) -> Dict[str, Any]:
    status = switcher.settings()
    if native_exceptions is not None:
        status["native_exceptions"] = native_exceptions.summary()
    return status


def handle_auto_switch_command(
    switcher: AutoSwitcher,
    command: str,
    payload: str = "",
    *,
    native_exceptions: NativeExceptionRegistry | None = None,
) -> Dict[str, Any]:
    if command == "auto_switch.status":
        return auto_switch_status(switcher, native_exceptions)
    if command == "auto_switch.exceptions":
        if native_exceptions is None:
            return {"total": 0, "events": [], "by_node": [], "by_type": []}
        return native_exceptions.snapshot()
    if command == "auto_switch.set":
        if not payload.strip():
            raise ValueError("auto_switch.set requires a JSON object payload")
        settings = _parse_set_payload(payload)
        return switcher.configure(
            transition_mode=settings.get("transition_mode"),
            fade_duration_s=settings.get("fade_duration_s"),
            wipe_file=settings.get("wipe_file"),
        )
    if command == "auto_switch.mode":
        mode = payload.strip()
        if not mode:
            raise ValueError("auto_switch.mode requires cut, fade, or wipe")
        return switcher.set_transition_mode(mode)
    if command == "auto_switch.fade_duration":
        duration = payload.strip()
        if not duration:
            raise ValueError("auto_switch.fade_duration requires seconds")
        return switcher.set_fade_duration(float(duration))
    raise ValueError(f"Unknown command: {command}")


class AutoSwitchControlCommands:
    """Registers auto-switch commands on the main AVPlumber control port."""

    COMMANDS = (
        "auto_switch.status",
        "auto_switch.exceptions",
        "auto_switch.set",
        "auto_switch.mode",
        "auto_switch.fade_duration",
    )

    def __init__(
        self,
        avp,
        switcher: AutoSwitcher,
        *,
        native_exceptions: NativeExceptionRegistry | None = None,
    ) -> None:
        self.avp = avp
        self.switcher = switcher
        self.native_exceptions = native_exceptions

    def start(self) -> None:
        for command in self.COMMANDS:
            self.avp.registerControlCommand(
                command,
                lambda payload, command=command: self._handle(command, payload),
                True,
            )

    def stop(self) -> None:
        return None

    def _handle(self, command: str, payload: str) -> str:
        result = handle_auto_switch_command(
            self.switcher,
            command,
            payload,
            native_exceptions=self.native_exceptions,
        )
        return json.dumps(result, sort_keys=True) + "\n"
