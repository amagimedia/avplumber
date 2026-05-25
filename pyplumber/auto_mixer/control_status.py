"""Small synchronous reader for avplumber mixer status."""

from __future__ import annotations

import json
import socket
import time
from typing import Optional


class MixerProgramSceneReader:
    """Poll ``mixer.status`` over the text control protocol."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        mixer_name: str = "mixer",
        timeout_s: float = 0.2,
        reconnect_s: float = 1.0,
    ) -> None:
        self._host = host
        self._port = port
        self._mixer_name = mixer_name
        self._timeout_s = timeout_s
        self._reconnect_s = reconnect_s
        self._sock: Optional[socket.socket] = None
        self._file = None
        self._next_connect_ts = 0.0

    def status(self) -> Optional[dict]:
        if not self._ensure_connected():
            return None
        try:
            assert self._sock is not None
            self._sock.sendall(f"mixer.status {self._mixer_name}\n".encode("utf-8"))
            code, content = self._read_response()
            if code != 201 or not content:
                return None
            data = json.loads(content)
            return data if isinstance(data, dict) else None
        except Exception:
            self.close()
            return None

    def program_scene(self) -> Optional[str]:
        data = self.status()
        if not data:
            return None
        scene = data.get("pgm_scene")
        return scene if isinstance(scene, str) and scene else None

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._next_connect_ts = time.monotonic() + self._reconnect_s

    def _ensure_connected(self) -> bool:
        if self._sock is not None:
            return True
        now = time.monotonic()
        if now < self._next_connect_ts:
            return False
        try:
            sock = socket.create_connection((self._host, self._port), timeout=self._timeout_s)
            sock.settimeout(self._timeout_s)
            self._sock = sock
            self._file = sock.makefile("rb", buffering=0)
            code, _ = self._read_response()
            if code != 100:
                self.close()
                return False
            return True
        except Exception:
            self.close()
            return False

    def _read_response(self) -> tuple[int, str]:
        assert self._file is not None
        line = self._file.readline().decode("utf-8", errors="replace")
        if not line:
            raise EOFError("control connection closed")
        code_text = line.split(maxsplit=1)[0]
        code = int(code_text)
        if code != 201:
            return code, ""

        content = []
        while True:
            line = self._file.readline().decode("utf-8", errors="replace")
            if not line:
                raise EOFError("control connection closed")
            if line.rstrip("\r\n") == "":
                break
            content.append(line)
        return code, "".join(content)
