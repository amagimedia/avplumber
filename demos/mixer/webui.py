#!/usr/bin/env python3
"""Browser control surface for the generic mixer demo.

Serves one static page and bridges it to AVPlumber's line protocol, so the
mixer needs no HTTP server of its own and the page needs no build step:

    GET  /              the page
    GET  /api/state     status + scenes + settings in one round trip
    POST /api/command   {"command": "cut"|"fade"|"wipe"|"preview", "scene": ...}

The bridge owns a single serialized control connection and reconnects when the
mixer restarts, so the page can stay open across a demo restart.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from avpmixer.control import AvpConnection, mixer_command

PAGE = Path(__file__).with_name("webui") / "index.html"
TAKE_COMMANDS = ("cut", "fade", "wipe", "preview", "interrupt")


class MixerBridge:
    """Serialized access to the mixer's control connection from HTTP threads."""

    def __init__(self, host: str, port: int, mixer: str, timeout: float = 10.0,
                 transition: str | None = None):
        self.mixer = mixer
        self.timeout = timeout
        self.transition = transition
        self._connection = AvpConnection(host, port)
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="mixer-bridge").start()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.timeout)

    def command(self, line: str) -> str | None:
        """Send one command, reconnecting once if the mixer was restarted."""
        try:
            if not self._connection.connected:
                self._run(self._connection.connect())
            return self._run(self._connection.command(line))
        except Exception:
            self._run(self._connection.disconnect())
            self._run(self._connection.connect())
            return self._run(self._connection.command(line))

    def state(self) -> dict:
        """Everything the page redraws from, in one poll."""
        state: dict = {"mixer": self.mixer}
        state["status"] = json.loads(self.command(f"mixer.status {self.mixer}") or "{}")
        state["scenes"] = json.loads(self.command(f"mixer.scenes {self.mixer}") or "[]")
        try:
            state["settings"] = json.loads(self.command(f"mixer.settings {self.mixer}") or "{}")
        except Exception:
            state["settings"] = {}   # older mixers, or one started without a config
        if self.transition is not None:
            state["settings"]["transition"] = self.transition
        return state

    def take(self, request: dict) -> None:
        command = request.get("command")
        if command not in TAKE_COMMANDS:
            raise ValueError(f"command must be one of {', '.join(TAKE_COMMANDS)}")
        payload = {k: v for k, v in request.items() if k not in ("command", "mixer")}
        if command != "interrupt" and not payload.get("scene"):
            raise ValueError(f"{command} needs a scene")
        self.command(mixer_command(command, self.mixer, **payload))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "avplumber-mixer-webui"

    def __init__(self, bridge: MixerBridge, *args, **kwargs):
        self.bridge = bridge
        super().__init__(*args, **kwargs)

    def log_message(self, *_args) -> None:
        pass   # one line per poll would drown the demo logs

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        path = self.path.partition("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            try:
                self._send_json(200, self.bridge.state())
            except Exception as exc:
                self._send_json(503, {"error": str(exc)})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.partition("?")[0] != "/api/command":
            self._send_json(404, {"error": "not found"})
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            self.bridge.take(json.loads(body or b"{}"))
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(502, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True})


def serve(bridge: MixerBridge, bind: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((bind, port), partial(Handler, bridge))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="mixer control host")
    parser.add_argument("--port", type=int, default=7777, help="mixer control port")
    parser.add_argument("--mixer", default="mixer")
    parser.add_argument("--bind", default="0.0.0.0", help="address to serve the page on")
    parser.add_argument("--http-port", type=int, default=7681)
    parser.add_argument("--transition", choices=("cut", "fade", "wipe"),
                        help="Override the page's initial transition without changing the running mixer")
    args = parser.parse_args(argv)
    server = serve(MixerBridge(args.host, args.port, args.mixer, transition=args.transition),
                   args.bind, args.http_port)
    print(f"mixer web UI on http://{args.bind}:{args.http_port} "
          f"controlling {args.mixer} at {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
