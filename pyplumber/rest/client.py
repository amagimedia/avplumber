"""What the gateway needs from a connection to the engine's control port."""

from __future__ import annotations

from typing import Protocol


class ControlClient(Protocol):
    """Sends one control-protocol line and returns the reply's content (None for `200 OK`).

    `pyplumber.mixer.gui.MixerBridge` is one: it serializes callers and reconnects when the
    engine restarts. A command the engine refuses raises `pyplumber.mixer.control.AvpProtocolError`
    with the engine's status line; an unreachable engine raises ConnectionError or OSError, a
    reply that does not arrive in time TimeoutError. Implementations must be safe to call from
    several threads: the gateway serves each request on a worker thread.
    """

    def command(self, line: str, timeout: float | None = None) -> str | None:
        ...
