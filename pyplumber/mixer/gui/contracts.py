"""Structural interfaces for applications embedding the shared mixer GUI.

Applications implement these methods without inheriting from the demo or from
SetupRuntime. Input discovery, credentials and preparation stay in the caller.
"""
from typing import Any, Protocol


class GpuTelemetry(Protocol):
    def snapshot(self) -> list[dict[str, Any]]:
        """Return the shared GPU/NVENC sample used by /api/state."""
        ...


class SetupController(Protocol):
    def page(self) -> bytes:
        """Return the application's setup HTML."""
        ...

    def status(self) -> dict[str, Any]:
        """Include phase, message and revision; idle/starting gate mixer polling."""
        ...

    def apply(self, settings: dict[str, Any]) -> None:
        """Validate application settings and schedule their application."""
        ...

    def remember_aux(self, bus_id: str | None, fields: dict[str, Any]) -> None:
        """Persist an accepted AUX change, or do nothing if persistence is unneeded."""
        ...
