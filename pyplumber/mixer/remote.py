"""A mixer session against a separate ``avplumber`` process.

Graph changes are control-protocol lines (``node.add``, ``group.start``,
``queue.plan_capacity``, ``mixer.*``). The process is started on its own;
this client does not embed it.

The TUI keeps connecting to ``--remote-control-port``. That listener is here,
because ``mixer.aux``, ``mixer.dsk`` and ``mixer.settings`` are Python. Every
other command is forwarded to the avplumber process. The greeting waits until
:meth:`RemoteAvplumber.setReady`, which is when the show has finished starting.
"""

from __future__ import annotations

import json
import logging
import socket
import threading

log = logging.getLogger("mixer.remote")


class ControlError(RuntimeError):
    pass


def parse_endpoint(text: str) -> tuple[str, int]:
    """``HOST:PORT``. The host may be a name or an address; the port is the last field."""
    host, sep, port_text = text.rpartition(":")
    if not sep or not host:
        raise ValueError(f"avplumber endpoint must be HOST:PORT, got {text!r}")
    try:
        port = int(port_text)
    except ValueError:
        raise ValueError(f"avplumber endpoint must be HOST:PORT, got {text!r}") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"avplumber port {port} is out of range")
    return host, port


def parse_reply(raw: bytes) -> tuple[int, str, str | None]:
    """One protocol reply: status line, and a body when the code is 201."""
    text = raw.decode("utf-8")
    line, sep, rest = text.partition("\n")
    if not sep:
        raise ControlError(f"truncated control reply {raw!r}")
    code_text, _, status = line.partition(" ")
    try:
        code = int(code_text)
    except ValueError:
        raise ControlError(f"control reply has no status code: {line!r}") from None
    if code != 201:
        return code, status, None
    lines = []
    for part in rest.splitlines(keepends=True):
        if part in ("\n", "\r\n"):
            break
        lines.append(part)
    return code, status, "".join(lines)


class ControlClient:
    """One connection. Commands are serialized: the process handles each to completion."""

    def __init__(self, host: str, port: int):
        try:
            sock = socket.create_connection((host, port), timeout=10)
        except OSError as exc:
            raise ControlError(f"cannot connect to avplumber at {host}:{port}: {exc}") from exc
        sock.settimeout(None)
        self._file = sock.makefile("rwb", buffering=0)
        self._sock = sock
        self._lock = threading.Lock()
        greeting = self._file.readline()
        if not greeting.startswith(b"100 "):
            raise ControlError(f"unexpected greeting from {host}:{port}: {greeting!r}")

    def exchange(self, line: str) -> bytes:
        """Send one command and return the raw reply, including a non-2xx status."""
        with self._lock:
            self._file.write(line.encode("utf-8") + b"\n")
            return self._read_reply()

    def command(self, line: str) -> str | None:
        """Send one command. Returns the 201 body, or None for an empty success."""
        raw = self.exchange(line)
        code, status, body = parse_reply(raw)
        if not 200 <= code < 300:
            raise ControlError(f"{code} {status}")
        return body

    def close(self) -> None:
        with self._lock:
            try:
                self._sock.close()
            except OSError:
                pass

    def _read_reply(self) -> bytes:
        line = self._file.readline()
        if not line:
            raise ControlError("avplumber closed the control connection")
        code_text = line.split(b" ", 1)[0]
        if code_text != b"201":
            return line
        chunks = [line]
        while True:
            part = self._file.readline()
            if not part:
                raise ControlError("avplumber closed a multiline response")
            chunks.append(part)
            if part in (b"\n", b"\r\n"):
                break
        return b"".join(chunks)


class _Edges:
    def __init__(self, avp: RemoteAvplumber):
        self._avp = avp

    def planCapacity(self, name: str, capacity: int) -> None:
        self._avp.client.command(f"queue.plan_capacity {name} {capacity}")


class RemoteGroup:
    def __init__(self, avp: RemoteAvplumber, name: str):
        self._avp = avp
        self.name = name

    def startNodes(self) -> None:
        self._avp.client.command(f"group.start {self.name}")
        self._avp._started.add(self.name)

    def stopNodes(self) -> None:
        try:
            self._avp.client.command(f"group.stop {self.name}")
        except ControlError as exc:
            log.warning("stopping group %s: %s", self.name, exc)

    def status(self) -> dict:
        body = self._avp.client.command(f"group.status {self.name}")
        if not body:
            raise ControlError(f"group.status {self.name} returned an empty body")
        status = json.loads(body)
        if not isinstance(status, dict):
            raise ControlError(f"group.status {self.name} is not an object")
        return status

    def is_running(self) -> bool:
        return self.status().get("state") == "running"


class RemoteNode:
    def __init__(self, avp: RemoteAvplumber, name: str):
        self._avp = avp
        self.name = name

    def getObject(self, key: str):
        body = self._avp.client.command(f"node.object.get {self.name} {key}")
        if body is None:
            return None
        return json.loads(body)

    def isWorking(self) -> bool:
        """True while the node's group is running and this node has not failed.

        The control protocol has no per-node running flag. A node is working
        when ``node.add`` put it in a group and that group's status is
        ``running`` without a ``failed`` or ``panicked`` outcome for the node.
        """
        group = self._avp._node_groups.get(self.name)
        if not group or group not in self._avp._started:
            return False
        status = self._avp.group(group).status()
        if status.get("state") != "running":
            return False
        prefix_failed = f"failed:{self.name}:"
        prefix_panicked = f"panicked:{self.name}:"
        return not any(
            outcome.startswith((prefix_failed, prefix_panicked))
            for outcome in status.get("outcomes") or ()
        )


class RemoteEdge:
    """Named edge. Queue depth and clear are not control commands."""

    def __init__(self, name: str):
        self.name = name

    def _unavailable(self, what: str):
        raise ControlError(f"{what} of edge {self.name!r} is not a control command")

    @property
    def enqueued_total(self) -> int:
        self._unavailable("enqueued_total")

    @property
    def occupied(self) -> int:
        self._unavailable("occupied")

    def clear(self) -> None:
        self._unavailable("clear")


class _Manager:
    def __init__(self, avp: RemoteAvplumber):
        self._avp = avp

    @property
    def shouldWork(self) -> bool:
        return self._avp._working


def _json_default(value):
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"node parameter {value!r} is not JSON")


class RemoteAvplumber:
    def __init__(self, client: ControlClient):
        self.client = client
        self.edges = _Edges(self)
        self.manager = _Manager(self)
        self.on_exception = lambda _name, _kind, _message: None
        self._node_groups: dict[str, str] = {}
        self._started: set[str] = set()
        self._local: dict = {}
        self._working = True
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._server: socket.socket | None = None
        self._server_thread: threading.Thread | None = None

    @classmethod
    def connect(cls, host: str, port: int) -> RemoteAvplumber:
        log.info("connecting to avplumber at %s:%s", host, port)
        return cls(ControlClient(host, port))

    def addNode(self, node, early_create: bool = False, start: bool = False) -> None:
        del early_create, start  # node.add creates the node; group.start runs it
        params = node.parameters
        self.client.command("node.add " + json.dumps(params, default=_json_default, separators=(",", ":")))
        group = params.get("group")
        name = params.get("name")
        if isinstance(name, str) and isinstance(group, str):
            self._node_groups[name] = group
        node.add_to_avplumber(self)

    def executeCommandsFromString(self, text: str) -> None:
        for raw in text.splitlines():
            line = raw.strip()
            if line:
                self.client.command(line)

    def registerControlCommand(self, name: str, handler, *_ignored) -> None:
        self._local[name] = handler

    def group(self, name: str) -> RemoteGroup:
        return RemoteGroup(self, name)

    def node(self, name: str) -> RemoteNode:
        return RemoteNode(self, name)

    def getEdge(self, name: str, data_type=None) -> RemoteEdge:
        del data_type
        return RemoteEdge(name)

    def enableControlServer(self, port: int, host: str = "0.0.0.0") -> None:
        if not port:
            return
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        server.settimeout(0.5)
        self._server = server
        self._server_thread = threading.Thread(target=self._serve, name="mixer-control", daemon=True)
        self._server_thread.start()
        log.info("mixer control listening on %s:%s", host, port)

    def setReady(self) -> None:
        self._ready.set()

    def poll(self) -> None:
        """Drop ``shouldWork`` when a started group has failed, so the supervisor loop exits."""
        if not self._working:
            return
        for name in list(self._started):
            try:
                status = self.group(name).status()
            except ControlError as exc:
                self._working = False
                self.on_exception(name, "group", str(exc))
                return
            bad = next(
                (outcome for outcome in status.get("outcomes") or ()
                 if outcome.startswith(("failed:", "panicked:"))),
                None,
            )
            if status.get("state") == "failed" or bad:
                self._working = False
                self.on_exception(name, "group", status.get("last_error") or bad or "failed")
                return

    def shutdown(self) -> None:
        self._working = False
        self._ready.set()
        self._stop.set()
        for name in list(self._started):
            self.group(name).stopNodes()
        server = self._server
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        thread = self._server_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        self.client.close()

    def registerWithWebUI(self, url: str, name: str, description: str) -> None:
        del url, name, description
        log.warning("the separate avplumber process is not registered with a web UI")

    def heartbeat(self) -> None:
        return None

    def _serve(self) -> None:
        server = self._server
        assert server is not None
        while not self._stop.is_set():
            try:
                conn, _addr = server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn: socket.socket) -> None:
        try:
            while not self._ready.is_set():
                if self._stop.wait(0.2):
                    return
            conn.sendall(b"100 VTR READY\n")
            reader = conn.makefile("rb", buffering=0)
            while not self._stop.is_set():
                line = reader.readline()
                if not line:
                    return
                text = line.decode("utf-8").strip()
                if not text:
                    continue
                if text == "bye":
                    conn.sendall(b"BYE\n")
                    return
                conn.sendall(self._dispatch(text))
        except (OSError, ControlError) as exc:
            log.debug("control client ended: %s", exc)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _dispatch(self, line: str) -> bytes:
        cmd, _, rest = line.partition(" ")
        handler = self._local.get(cmd)
        if handler is not None:
            try:
                body = handler(rest)
            except Exception as exc:
                return f"500 ERROR: {exc}\n".encode()
            text = "" if body is None else str(body)
            if text.strip() in ("", "ok"):
                return b"200 OK\n"
            if not text.endswith("\n"):
                text += "\n"
            return f"201 OK\n{text}\n".encode()
        try:
            return self.client.exchange(line)
        except ControlError as exc:
            return f"500 ERROR: {exc}\n".encode()
