"""Small TCP control socket for Python auto-switch policy settings."""

from __future__ import annotations

import json
import socketserver
import threading
from typing import Any, Dict

from pyplumber.auto_switcher import AutoSwitcher
from pyplumber.auto_mixer.native_exceptions import NativeExceptionRegistry


class _AutoSwitchTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address,
        handler_cls,
        switcher: AutoSwitcher,
        native_exceptions: NativeExceptionRegistry | None,
    ):
        super().__init__(address, handler_cls)
        self.switcher = switcher
        self.native_exceptions = native_exceptions


class _AutoSwitchRequestHandler(socketserver.StreamRequestHandler):
    def setup(self) -> None:
        super().setup()
        self._send_line(100, "AUTO SWITCH READY")

    def handle(self) -> None:
        for raw in self.rfile:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line in ("bye", "quit"):
                self._send_line(200, "BYE")
                return
            try:
                self._handle_command(line)
            except ValueError as exc:
                self._send_line(400, str(exc))
            except Exception as exc:
                self._send_line(500, str(exc))

    @property
    def switcher(self) -> AutoSwitcher:
        return self.server.switcher

    @property
    def native_exceptions(self) -> NativeExceptionRegistry | None:
        return self.server.native_exceptions

    def _handle_command(self, line: str) -> None:
        command, _, payload = line.partition(" ")
        if command == "auto_switch.status":
            status = self.switcher.settings()
            if self.native_exceptions is not None:
                status["native_exceptions"] = self.native_exceptions.summary()
            self._send_json(status)
            return
        if command == "auto_switch.exceptions":
            if self.native_exceptions is None:
                self._send_json({"total": 0, "events": [], "by_node": [], "by_type": []})
                return
            self._send_json(self.native_exceptions.snapshot())
            return
        if command == "auto_switch.set":
            if not payload.strip():
                raise ValueError("auto_switch.set requires a JSON object payload")
            settings = self._parse_payload(payload)
            result = self.switcher.configure(
                transition_mode=settings.get("transition_mode"),
                fade_duration_s=settings.get("fade_duration_s"),
                wipe_file=settings.get("wipe_file"),
            )
            self._send_json(result)
            return
        if command == "auto_switch.mode":
            mode = payload.strip()
            if not mode:
                raise ValueError("auto_switch.mode requires cut, fade, or wipe")
            self._send_json(self.switcher.set_transition_mode(mode))
            return
        if command == "auto_switch.fade_duration":
            duration = payload.strip()
            if not duration:
                raise ValueError("auto_switch.fade_duration requires seconds")
            self._send_json(self.switcher.set_fade_duration(float(duration)))
            return
        self._send_line(404, "Unknown command")

    def _parse_payload(self, payload: str) -> Dict[str, Any]:
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

    def _send_line(self, code: int, status: str) -> None:
        self.wfile.write(f"{code} {status}\n".encode("utf-8"))

    def _send_json(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True)
        self.wfile.write(f"201 OK\n{body}\n\n".encode("utf-8"))


class AutoSwitchControlServer:
    """Background line-protocol server for auto-switch runtime settings."""

    def __init__(
        self,
        switcher: AutoSwitcher,
        *,
        host: str = "0.0.0.0",
        port: int,
        native_exceptions: NativeExceptionRegistry | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self._server = _AutoSwitchTCPServer(
            (host, self.port),
            _AutoSwitchRequestHandler,
            switcher,
            native_exceptions,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="AutoSwitchControlServer",
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=timeout)
