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
import csv
import json
import subprocess
import threading
import signal
import time
from urllib.parse import urlsplit
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pyplumber.mixer.control import AvpConnection, mixer_command

PAGE = Path(__file__).with_name("webui") / "index.html"
TAKE_COMMANDS = ("cut", "fade", "wipe", "preview", "interrupt")


class GpuStats:
    """Share one bounded nvidia-smi sample across all viewers each second."""

    def __init__(self):
        self.lock = threading.Lock()
        self.next_sample = 0
        self.values = []

    def snapshot(self):
        if not self.lock.acquire(blocking=False):
            return self.values
        try:
            if time.monotonic() < self.next_sample:
                return self.values
            self.next_sample = time.monotonic() + 1
            output = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.decoder,utilization.encoder,memory.used,memory.total",
                "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=1)
            keys = ("index", "gpu", "decoder", "encoder", "memory_used_mib", "memory_total_mib")
            values = []
            for row in csv.reader(output.splitlines()):
                if len(row) != len(keys) or not row[0].strip().isdigit():
                    continue
                values.append(dict(zip(keys, (int(v) if v.strip().isdigit() else None for v in row))))
            self.values = values
        except (OSError, subprocess.SubprocessError):
            self.values = []
        finally:
            self.lock.release()
        return self.values


class MixerBridge:
    """Serialized access to the mixer's control connection from HTTP threads."""

    def __init__(self, host: str, port: int, mixer: str, timeout: float = 10.0,
                 transition: str | None = None):
        self.mixer = mixer
        self.port = port
        self._lock = threading.Lock()
        self.timeout = timeout
        self.transition = transition
        self._connection = AvpConnection(host, port)
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="mixer-bridge").start()

    def _run(self, coro, timeout: float | None = None):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(self.timeout if timeout is None else timeout + 1.0)
        except Exception:
            future.cancel()
            raise

    def command(self, line: str, timeout: float | None = None) -> str | None:
        """Send one command, reconnecting once if the mixer was restarted. `timeout` bounds
        both the exchange and the wait for it, for replies that outgrow the default."""
        budget = self.timeout if timeout is None else timeout
        with self._lock:
            try:
                if not self._connection.connected:
                    self._run(self._connection.connect())
                return self._run(self._connection.command(line, budget), budget)
            except Exception:
                self._run(self._connection.disconnect())
                self._run(self._connection.connect())
                return self._run(self._connection.command(line, budget), budget)

    def state(self, timeout: float | None = None) -> dict:
        """Everything the page redraws from, in one poll. A caller polling a mixer that is still
        building its graph passes a longer `timeout`: the mixer runs control commands on the
        thread that is busy starting nodes, so replies stall for seconds during startup even
        though they take a millisecond once it is running."""
        state: dict = {"mixer": self.mixer}
        state["status"] = json.loads(self.command(f"mixer.status {self.mixer}", timeout) or "{}")
        state["scenes"] = json.loads(self.command(f"mixer.scenes {self.mixer}", timeout) or "[]")
        try:
            state["settings"] = json.loads(self.command(f"mixer.settings {self.mixer}", timeout) or "{}")
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

    def __init__(self, bridge: MixerBridge, *args, setup=None, gpu=None, **kwargs):
        self.bridge = bridge
        self.setup_manager = setup
        self.gpu = gpu
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
        elif path in ("/setup", "/setup/"):
            self._send(200, Path(__file__).with_name("setup.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/setup" and self.setup_manager:
            self._send_json(200, self.setup_manager.status())
        elif path == "/api/state":
            if self.setup_manager:
                status = self.setup_manager.status()
                if status["phase"] in ("idle", "starting"):
                    self._send_json(503, {"error": status["message"]})
                    return
            try:
                state = self.bridge.state()
                state["gpus"] = self.gpu.snapshot()
                if self.setup_manager:
                    state["setup_revision"] = self.setup_manager.status()["revision"]
                self._send_json(200, state)
            except Exception as exc:
                self._send_json(503, {"error": str(exc)})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.partition("?")[0] == "/api/setup" and self.setup_manager:
            self.close_connection = True
            # The setup endpoint accepts same-origin JSON, never shell commands.
            origin = self.headers.get("Origin")
            if (origin and urlsplit(origin).netloc != self.headers.get("Host")) or self.headers.get("Sec-Fetch-Site") == "cross-site":
                self._send_json(403, {"error": "Use the setup page on this instance"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected a JSON setup request of at most 4096 bytes")
                self.setup_manager.apply(json.loads(self.rfile.read(size)))
                self._send_json(202, self.setup_manager.status())
            except (ValueError, KeyError, TypeError) as exc:
                self._send_json(400, {"error": str(exc)})
            except RuntimeError as exc:
                self._send_json(409, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return
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


def serve(bridge: MixerBridge, bind: str, port: int, setup=None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((bind, port), partial(Handler, bridge, setup=setup, gpu=GpuStats()))
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
    parser.add_argument("--manage-setup", action="store_true", help="Own and restart the demo mixer process")
    parser.add_argument("--media-dir", type=Path, default=Path("/media"))
    parser.add_argument("--recipe", type=Path, default=Path("/media/demo.json"))
    parser.add_argument("--dmabuf-rest", default="http://127.0.0.1:9009")
    parser.add_argument("--mixer-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    bridge = MixerBridge(args.host, args.port, args.mixer, transition=args.transition)
    setup = None
    if args.manage_setup:
        from setup_runtime import SetupRuntime
        setup = SetupRuntime(args.media_dir, args.recipe, bridge, args.mixer_args, args.dmabuf_rest)
        if args.recipe.exists() or (args.media_dir / "mixer.demo.json").exists():
            try:
                setup.resume()
            except Exception as exc:
                setup._status("error", str(exc))
    server = serve(bridge, args.bind, args.http_port, setup=setup)
    print(f"mixer web UI on http://{args.bind}:{args.http_port} "
          f"controlling {args.mixer} at {args.host}:{args.port}", flush=True)
    def stop(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if setup:
            setup.close()


if __name__ == "__main__":
    main()
