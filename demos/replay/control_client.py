"""A client for the Rust avplumber's control protocol, and the process behind it.

The protocol is one command per line and one status line per reply:
``200 OK``, ``201 OK`` followed by a body and an empty line, ``400`` for an
unknown command, ``500`` for a failed one (``doc/control_protocol.md``).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path


class ControlError(RuntimeError):
    """The server refused a command."""


def find_avplumber(explicit: str | os.PathLike | None = None) -> Path:
    """The Rust ``avplumber`` executable: ``explicit``, else ``$AVPLUMBER_BIN``.

    Nothing is taken from ``$PATH`` on purpose: the C++ binary of the same name
    is usually installed there, and it does not run these graphs.
    """
    candidate = explicit or os.environ.get("AVPLUMBER_BIN")
    if not candidate:
        raise FileNotFoundError(
            "no avplumber binary: pass --avplumber or set AVPLUMBER_BIN to the Rust "
            "executable (cargo build -p avplumber_nodes --features ffmpeg7_1,async --bin avplumber)"
        )
    path = Path(candidate).expanduser()
    if not path.is_file():
        resolved = shutil.which(str(candidate))
        if resolved is None:
            raise FileNotFoundError(f"avplumber binary not found: {candidate}")
        path = Path(resolved)
    return path.resolve()


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def parse_endpoint(value: str, default_port: int = 20200) -> tuple[str, int]:
    host, sep, port = value.rpartition(":")
    if not sep:
        return value, default_port
    return host or "127.0.0.1", int(port)


class ControlClient:
    """One connection, safe to share between threads: a command and its reply
    are one critical section."""

    def __init__(self, host: str, port: int, *, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._reader = None

    def connect(self, *, retries: int = 1, delay: float = 0.1) -> None:
        last: Exception | None = None
        for _ in range(max(retries, 1)):
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
                break
            except OSError as exc:
                last = exc
                time.sleep(delay)
        else:
            raise ConnectionError(f"cannot connect to avplumber at {self.host}:{self.port}: {last}")
        sock.settimeout(self.timeout)
        self._sock = sock
        self._reader = sock.makefile("r", encoding="utf-8", newline="\n")
        greeting = self._reader.readline().strip()
        if not greeting.startswith("100"):
            raise ConnectionError(f"unexpected greeting from avplumber: {greeting!r}")

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def command(self, line: str) -> str:
        """Sends one command; returns the reply body (empty for ``200 OK``)."""
        if self._sock is None or self._reader is None:
            raise ConnectionError("not connected")
        with self._lock:
            self._sock.sendall((line.rstrip("\n") + "\n").encode("utf-8"))
            status = self._reader.readline()
            if not status:
                raise ConnectionError("avplumber closed the connection")
            status = status.rstrip("\n")
            if status.startswith("201"):
                lines = []
                while True:
                    part = self._reader.readline()
                    if part in ("", "\n"):
                        break
                    lines.append(part.rstrip("\n"))
                return "\n".join(lines)
            if status.startswith("200"):
                return ""
            if status == "BYE":
                return "BYE"
            raise ControlError(f"{line!r}: {status}")

    def status(self, group: str) -> dict:
        return json.loads(self.command(f"playback.status {group}"))

    def group_status(self, group: str) -> dict:
        return json.loads(self.command(f"group.status {group}"))

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is None:
            return
        try:
            with self._lock:
                sock.sendall(b"bye\n")
        except OSError:
            pass
        finally:
            sock.close()


class AvplumberProcess:
    """The Rust executable serving the control protocol on a local port."""

    def __init__(self, binary: Path, *, host: str = "127.0.0.1", port: int | None = None,
                 log_level: str = "info", log_path: Path | None = None):
        self.binary = Path(binary)
        self.host = host
        self.port = port or free_port(host)
        self.log_level = log_level
        self.log_path = log_path
        self._process: subprocess.Popen | None = None
        self._log_file = None

    def start(self) -> None:
        if self._process is not None:
            return
        stderr = subprocess.DEVNULL
        if self.log_path is not None:
            self._log_file = open(self.log_path, "ab")
            stderr = self._log_file
        self._process = subprocess.Popen(
            [str(self.binary), "--port", str(self.port), "--bind", self.host, "--log", self.log_level],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )

    def connect(self, timeout: float = 5.0) -> ControlClient:
        client = ControlClient(self.host, self.port, timeout=timeout)
        deadline = time.monotonic() + timeout
        while True:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f"avplumber exited with status {self._process.returncode}")
            try:
                client.connect()
                return client
            except ConnectionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def stop(self, timeout: float = 5.0) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def run_batch(self, script: Path, timeout: float | None = None) -> subprocess.CompletedProcess:
        """Runs ``script`` to completion without serving: a transcode."""
        return subprocess.run(
            [str(self.binary), "--log", self.log_level, str(script)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
