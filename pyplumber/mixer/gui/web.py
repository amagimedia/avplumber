#!/usr/bin/env python3
"""Shared mixer control surface and output video wall.

Serves static pages and bridges them to AVPlumber's line protocol, so the
mixer needs no HTTP server of its own and the pages need no build step:

    GET  /              the page
    GET  /wall          every output side by side, nothing to control
    GET  /outputs.js    the output list both pages play from
    GET  /api/state     status + scenes + settings in one round trip
    GET  /api/status    mixer.status alone, for scripts that poll often
    POST /api/command   {"command": "cut"|"fade"|"wipe"|"preview", "scene": ...}
                        (a fade may carry "curve": "linear"|"ease-in"|"ease-out"|"ease-in-out",
                        and "color": "#RRGGBB" to dip through that colour instead of mixing)
                        {"command": "dsk", "key": ..., "on": true|false, "fade_seconds"?: s, "curve"?: ...}
                        A cut, fade or wipe superseded by a newer one before it reached
                        the mixer answers {"ok": true, "superseded": true}.

The bridge owns a single serialized control connection and reconnects when the
mixer restarts, so the page can stay open across a mixer restart. Takes must not
queue behind polling: every tab shares one /api/state reply for STATE_TTL_S, and
a burst of program takes (keyboard auto-repeat) collapses to its newest one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
from urllib.parse import urlsplit
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .bridge import AUX_COMMANDS, MixerBridge
from .contracts import GpuTelemetry, SetupController
from .priority import CRITICAL_NICE_PERIOD_S, CriticalNice
from .telemetry import GpuStats, HostStats, RecentCounts, compute_price, delivery_totals

PAGE = Path(__file__).with_name("assets") / "index.html"
WALL = PAGE.with_name("wall.html")
OUTPUTS_JS = PAGE.with_name("outputs.js")
# The page reads the backend's options from this script before its first request; the file holds `{}`, the defaults.
CONFIG_SCRIPT = '<script id="config" type="application/json">%s</script>'


def page(config: dict, path: Path = PAGE) -> bytes:
    """The page at `path` with `config`, what it needs before its first request (where the player is),
    in its config script."""
    html = path.read_text("utf-8")
    placeholder = CONFIG_SCRIPT % "{}"
    if placeholder not in html:
        raise RuntimeError(f"{path} has no config script to fill")
    # `<` is escaped so no option value can end the script element.
    return html.replace(placeholder, CONFIG_SCRIPT % json.dumps(config).replace("<", "\\u003c"), 1).encode("utf-8")


def normalize_preview_base(value: str) -> str:
    """--preview-base: where the page finds the player, a path on the page's own origin (`/preview/`)
    or a URL. The player resolves its own files and its Janus requests against it, so it ends with a
    slash; empty keeps the default."""
    return value + "/" if value and not value.endswith("/") else value


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "avplumber-mixer-webui"

    def __init__(self, bridge: MixerBridge, *args, setup: SetupController | None = None, gpu: GpuTelemetry | None = None, host_stats=None, config=None,
                 recent=None, **kwargs):
        self.bridge = bridge
        self.setup_manager = setup
        self.gpu = gpu if gpu is not None else GpuStats()
        self.host_stats = host_stats
        self.recent = recent
        self.config = config or {}
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args) -> None:
        pass   # one line per poll would drown the logs

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
            self._send(200, page(self.config), "text/html; charset=utf-8")
        elif path in ("/wall", "/wall/"):
            self._send(200, page(self.config, WALL), "text/html; charset=utf-8")
        elif path in ("/outputs.js", "/setup.js"):
            self._send(200, PAGE.with_name(path[1:]).read_bytes(), "text/javascript; charset=utf-8")
        elif path in ("/setup", "/setup/") and self.setup_manager:
            self._send(200, self.setup_manager.page(), "text/html; charset=utf-8")
        elif path == "/api/setup" and self.setup_manager:
            self._send_json(200, self.setup_manager.status())
        elif path in ("/api/state", "/api/status"):
            if self.setup_manager:
                status = self.setup_manager.status()
                if status["phase"] in ("idle", "starting"):
                    self._send_json(503, {"error": status["message"]})
                    return
            try:
                if path == "/api/status":
                    self._send_json(200, self.bridge.status())
                    return
                state = self.bridge.state()
                state["gpus"] = self.gpu.snapshot()
                state["host"] = self.host_stats.values if self.host_stats else None
                # Deadline misses and AUX drops of the last ten minutes; without a sampler the page shows totals.
                state["recent"] = self.recent.values if self.recent else None
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
                settings = json.loads(self.rfile.read(size))
                if not isinstance(settings, dict):
                    raise ValueError("Expected a JSON setup object")
                self.setup_manager.apply(settings)
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
            request = json.loads(body or b"{}")
            result = self.bridge.take(request)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(502, {"error": str(exc)})
            return
        if self.setup_manager and request.get("command") in AUX_COMMANDS:
            try:
                self.setup_manager.remember_aux(request.get("bus"), result)
            except Exception as exc:   # the bus is live; only its persistence failed
                print(f"Could not persist aux {request.get('bus')}: {exc}", flush=True)
        self._send_json(200, {"ok": True, "superseded": True} if result is False else {"ok": True})


def serve(bridge: MixerBridge, bind: str, port: int, setup: SetupController | None = None, host_stats: HostStats | None = None,
          preview_base: str | None = None, price: dict | None = None, *, gpu: GpuTelemetry | None = None,
          recent: RecentCounts | None = None) -> ThreadingHTTPServer:
    """`preview_base`: where the page's players load from, when not port 8080 of the page's host.
    Normalized here, so every caller's page carries a base the player resolves its files against.
    `setup` owns application-specific input/settings policy. `gpu` may be shared
    with another server; when omitted, this server creates one cached sampler."""
    preview_base = normalize_preview_base(preview_base or "")
    config: dict = {"preview_base": preview_base} if preview_base else {}
    if setup:
        config["setup"] = True
    if price:
        config["compute_price"] = price
    server = ThreadingHTTPServer((bind, port), partial(Handler, bridge, setup=setup, gpu=gpu if gpu is not None else GpuStats(), host_stats=host_stats,
                                                        config=config, recent=recent))
    server.daemon_threads = True
    return server


def parse_args(argv: list[str] | None = None, *, configure=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="mixer control host")
    parser.add_argument("--port", type=int, default=7777, help="mixer control port")
    parser.add_argument("--mixer", default="mixer")
    parser.add_argument("--bind", default="0.0.0.0", help="address to serve the page on")
    parser.add_argument("--http-port", type=int, default=7681)
    # nargs="?": compose.yaml always passes the flag, with an empty value when MIXER_PREVIEW_BASE is unset.
    parser.add_argument("--preview-base", nargs="?", const="", default="",
                        help="Where the page's players load from: a path on the page's own origin, such as /preview/ "
                             "behind a reverse proxy, or a URL (default: port 8080 of the page's host)")
    parser.add_argument("--transition", choices=("cut", "fade", "wipe"),
                        help="Override the page's initial transition without changing the running mixer")
    parser.add_argument("--compute-price", type=Path, default=os.environ.get("MIXER_COMPUTE_PRICE") or None,
                        help="JSON file with public hourly_usd, label, as_of and source_url for compute + GPU; "
                             "enables USD per input source per hour. Env: MIXER_COMPUTE_PRICE")
    # A string default goes through type=int like a command-line value, so a bad env value is a usage error.
    parser.add_argument("--critical-nice", type=int, metavar="N", default=os.environ.get("MIXER_CRITICAL_NICE", "0"),
                        help="Keep the deadline-critical mixer threads at nice -N, 1 to 20, rechecked every "
                             f"{CRITICAL_NICE_PERIOD_S} s; needs CAP_SYS_NICE and a managed mixer. 0, the default, "
                             "leaves them alone. Env: MIXER_CRITICAL_NICE")
    if configure:
        configure(parser)
    args = parser.parse_args(argv)
    if not 0 <= args.critical_nice <= 20:
        parser.error("--critical-nice must be 0 (off) or 1 to 20")
    return args


def main(argv: list[str] | None = None, *, configure=None, setup_factory=None) -> None:
    args = parse_args(argv, configure=configure)
    price = compute_price(args.compute_price)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    bridge = MixerBridge(args.host, args.port, args.mixer, transition=args.transition)
    setup = setup_factory(args, bridge) if setup_factory else None
    if args.critical_nice and setup is None:
        raise ValueError("--critical-nice needs a managed mixer process")
    if setup:
        setup.resume()
    def mixer_pid():
        process = setup.process if setup else None   # read once: the setup worker thread clears it when the mixer stops
        return process.pid if process else None
    host_stats = HostStats(mixer_pid)
    host_stats.start()
    recent = RecentCounts(lambda: delivery_totals(bridge.state(timeout=3)))
    recent.start()
    if args.critical_nice:
        CriticalNice(mixer_pid, args.critical_nice).start()
    server = serve(bridge, args.bind, args.http_port, setup=setup, host_stats=host_stats,
                   preview_base=args.preview_base, price=price, recent=recent)
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
