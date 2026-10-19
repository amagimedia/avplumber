#!/usr/bin/env python3
"""Shared mixer control surface and output video wall.

Serves static pages and bridges them to AVPlumber's line protocol, so the
mixer needs no HTTP server of its own and the pages need no build step:

    GET  /              the page
    GET  /wall          every output side by side, nothing to control
    GET  /outputs.js    the output list both pages play from
    GET  /api/state     status + scenes + settings in one round trip
    GET  /api/lever     a WebSocket for the lever: text messages {"position": 0..1, "duration_sec"?, "curve"?,
                        "color"?, "wipe_file"?, "t_ms"?, "seq"?, "client"?}, one per position, applied as
                        POST /api/command "lever" is; one connection keeps a fader's positions in order and
                        off the page's other requests
    GET  /api/keys      small cached rendered-key and fade-to-black tally, sampled separately from show telemetry
    GET  /api/banks     the source banks: {"banks": [[scene|null, ...x9], x3] | null}, shared by every controller
    POST /api/banks     {"banks": [[...], [...], [...]]}: replaces them (kept with the show's settings)
    GET  /api/outputs   the show's settings and the GPU figures alone, for player pages
    GET  /api/status    mixer.status alone, for scripts that poll often
    POST /api/command   {"command": "cut"|"fade"|"wipe"|"preview", "scene": ...}
                        (a fade may carry "curve": "linear"|"ease-in"|"ease-out"|"ease-in-out",
                        and "color": "#RRGGBB" to dip through that colour instead of mixing)
                        {"command": "dsk", "key": ..., "on": true|false, "fade_seconds"?: s, "curve"?: ...}
                        (without fade_seconds a key takes its own rate, set by {"command": "dsk", "key": ...,
                        "rate": s}, 0 cuts; until it has one, or after "rate": null, it takes the All rate, set
                        by {"command": "dsk", "all_rate": s}, which starts at the show's dsk fade_seconds and
                        is also the group form's default; each key row reports "rate", its own or null, and
                        "all_rate")
                        {"command": "dsk", "keys": [...], "on": true|false | "toggle": true, ...}
                        {"command": "dsk", "keys": [...], "prepare": true}: the keys that switch with the
                        next background transitions (Next Transition), on the transition's own clock
                        {"command": "ftb", "on": true|false | "toggle": true, "fade_seconds"?: s, "curve"?: ...}
                        fades the keyed program (not the clean feed) to black and back
                        {"command": "lever", "position": 0..1, "duration_sec"?: s, "curve"?: ..., "color"?: ...,
                        "wipe_file"?: path, "t_ms"?: ms, "seq"?: n, "client"?: id}
                        the lever arm (T-bar) for a panel, a MIDI fader or this page: from the end it rests
                        at it mixes to Preview (with a cached clip's "wipe_file": wipes, the clip following
                        the lever), the other end completes, its start cancels, each once the picture got
                        there (a position off that end meanwhile starts the next take then); positions sent
                        faster than the mixer takes them collapse to the newest, an end position only into
                        the same end. "t_ms", when the controller sampled the position on its own clock,
                        paces positions against network jitter; "seq" numbered per "client", a late one is
                        dropped
                        A cut, fade or wipe superseded by a newer one before it reached
                        the mixer answers {"ok": true, "superseded": true}.

The bridge owns a serialized control connection, and one for the lever, and reconnects
when the mixer restarts, so the page can stay open across a mixer restart. Takes must not
queue behind polling: every tab shares one /api/state reply for STATE_TTL_S, and
a burst of program takes (keyboard auto-repeat) collapses to its newest one.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import signal
import threading
from urllib.parse import urlsplit
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .bridge import AUX_COMMANDS, MixerBridge
from .contracts import GpuTelemetry, SetupController
from .priority import CRITICAL_NICE_PERIOD_S, CriticalNice
from .telemetry import (BrowserCpuStats, GpuStats, HostStats, RecentCounts, compute_price, playout_counters, playout_summary,
                        startup_report)

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


class SourceBanks:
    """The scenes on the nine buttons of the Program and Preset rows, in three banks, shared by every
    controller of the show and persisted by the setup manager when there is one."""
    BANKS, BUTTONS = 3, 9

    def __init__(self, setup: SetupController | None = None):
        # A setup controller of an embedding application may predate source banks: they then
        # live only as long as this server.
        self.setup = setup if callable(getattr(setup, "source_banks", None)) else None
        self.lock, self.banks, self.loaded = threading.Lock(), None, self.setup is None

    @classmethod
    def _checked(cls, banks):
        """A copy of `banks`, or ValueError when they are not BANKS lists of BUTTONS scene ids or nulls."""
        if (not isinstance(banks, list) or len(banks) != cls.BANKS
                or any(not isinstance(b, list) or len(b) > cls.BUTTONS for b in banks)
                or any(s is not None and not isinstance(s, str) for b in banks for s in b)):
            raise ValueError(f"banks must be {cls.BANKS} lists of at most {cls.BUTTONS} scene ids or nulls")
        return [list(b) for b in banks]

    def get(self):
        with self.lock:
            if not self.loaded:
                # Read once: a stored file that cannot be used leaves the buttons empty rather than
                # failing every state poll.
                try:
                    stored = self.setup.source_banks()
                    self.banks = None if stored is None else self._checked(stored)
                except (OSError, ValueError) as exc:
                    logging.getLogger(__name__).warning("source banks: cannot use the stored banks: %s", exc)
                self.loaded = True
            return self.banks

    def set(self, banks):
        banks = self._checked(banks)
        with self.lock:
            self.banks, self.loaded = banks, True
            if self.setup:
                self.setup.remember_banks(self.banks)
        return self.banks


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "avplumber-mixer-webui"

    def __init__(self, bridge: MixerBridge, *args, setup: SetupController | None = None, gpu: GpuTelemetry | None = None, host_stats=None, config=None,
                 recent=None, browser_cpu=None, banks: SourceBanks | None = None, **kwargs):
        self.bridge = bridge
        self.banks = banks or SourceBanks(setup)
        self.setup_manager = setup
        self.gpu = gpu if gpu is not None else GpuStats()
        self.host_stats = host_stats
        self.recent = recent
        self.browser_cpu = browser_cpu
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

    def _lever_socket(self) -> None:
        """Serves one WebSocket (RFC 6455) of lever positions until the client closes it. Only what a
        browser sends is handled: masked text frames of one fragment, ping and close."""
        key = self.headers.get("Sec-WebSocket-Key")
        if not key or self.headers.get("Upgrade", "").lower() != "websocket":
            self._send_json(400, {"error": "expected a WebSocket upgrade"})
            return
        accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.wfile.flush()
        self.close_connection = True

        def frame(opcode: int, payload: bytes = b"") -> None:
            self.wfile.write(bytes([0x80 | opcode, len(payload)]) + payload)   # control frames: < 126 bytes
            self.wfile.flush()

        warn = partial(logging.getLogger(__name__).warning, "lever: %s")

        def send() -> None:
            try:
                self.bridge.send_lever()
            except Exception as exc:
                warn(exc)

        while True:
            head = self.rfile.read(2)
            if len(head) < 2:
                return
            opcode, length = head[0] & 0x0F, head[1] & 0x7F
            if length == 126:
                length = int.from_bytes(self.rfile.read(2), "big")
            elif length == 127:
                length = int.from_bytes(self.rfile.read(8), "big")
            if length > 4096:
                frame(0x8)
                return
            mask = self.rfile.read(4) if head[1] & 0x80 else bytes(4)
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(self.rfile.read(length)))
            if opcode == 0x8:
                frame(0x8)
                return
            if opcode == 0x9:
                frame(0xA, data[:125])
            elif opcode == 0x1:
                try:
                    # Sent on a thread of its own, the socket read on meanwhile: positions arriving
                    # while one is with the mixer collapse (MixerBridge.queue_lever).
                    if self.bridge.queue_lever({**json.loads(data), "command": "lever"}):
                        threading.Thread(target=send, daemon=True, name="lever").start()
                except Exception as exc:   # one bad position must not close a fader's channel
                    warn(exc)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        path = self.path.partition("?")[0]
        if path == "/api/lever":
            self._lever_socket()
            return
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
        elif path == "/api/outputs":
            # A player page needs to know which outputs exist and when the show changed. The whole
            # state is hundreds of kilobytes and makes the mixer answer for every bus; this is not.
            try:
                reply = {"settings": self.bridge.settings(), "gpus": self.gpu.snapshot()}
                if self.setup_manager:
                    status = self.setup_manager.status()
                    if status["phase"] in ("idle", "starting"):
                        self._send_json(503, {"error": status["message"]})
                        return
                    reply["setup_revision"] = status["revision"]
                self._send_json(200, reply)
            except Exception as exc:
                self._send_json(503, {"error": str(exc)})
        elif path == "/api/keys":
            try:
                self._send_json(200, self.bridge.keys())
            except Exception as exc:
                self._send_json(503, {"error": str(exc)})
        elif path == "/api/banks":
            self._send_json(200, {"banks": self.banks.get()})
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
                state["source_banks"] = self.banks.get()
                state["host"] = self.host_stats.values if self.host_stats else None
                # Chromium's CPU from the browser service; None hides the THREAD tip's Chromium part.
                state["chromium"] = self.browser_cpu.values if self.browser_cpu else None
                # The browser windows' drops and Chromium's skipped paints over the last ten minutes.
                state["browser_drops"] = self.browser_cpu.drops.values if self.browser_cpu else None
                # Deadline misses and AUX drops of the last ten minutes; without a sampler the page shows totals.
                state["recent"] = self.recent.values if self.recent else None
                # The samplers rebase independently after startup. Do not display the previous
                # show's browser drops beside this show's delivery counts while one catches up.
                on_air = (state["recent"] or {}).get("on_air")
                if on_air is not None and (state["browser_drops"] or {}).get("since") != on_air:
                    state["browser_drops"] = None
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
        if self.path.partition("?")[0] == "/api/banks":
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                self._send_json(200, {"banks": self.banks.set(json.loads(body or b"{}").get("banks"))})
            except (ValueError, AttributeError) as exc:
                self._send_json(400, {"error": str(exc)})
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
          recent: RecentCounts | None = None, browser_cpu: BrowserCpuStats | None = None) -> ThreadingHTTPServer:
    """`preview_base`: where the page's players load from, when not port 8080 of the page's host.
    Normalized here, so every caller's page carries a base the player resolves its files against.
    `setup` owns application-specific input/settings policy. `gpu` may be shared
    with another server; when omitted, this server creates one cached sampler. `browser_cpu` polls
    the browser service's CPU figures for the page."""
    preview_base = normalize_preview_base(preview_base or "")
    config: dict = {"preview_base": preview_base} if preview_base else {}
    if setup:
        config["setup"] = True
    if price:
        config["compute_price"] = price
    server = ThreadingHTTPServer((bind, port), partial(Handler, bridge, setup=setup, gpu=gpu if gpu is not None else GpuStats(), host_stats=host_stats,
                                                        config=config, recent=recent, browser_cpu=browser_cpu,
                                                        banks=SourceBanks(setup)))
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
    # Deadline misses, AUX drops and the inputs' repeats and discards, every 5 s, from the mixer's
    # on-air moment on (the mixer resets them then; the window starts again with each new one).
    recent = RecentCounts(lambda: playout_counters(bridge), summary=playout_summary)
    recent.start()
    # Only an application that names the browser service (--dmabuf-rest) shows Chromium's CPU and
    # its windows' drops; the drop window starts from the counters the mixer read as it went on air.
    browser_url = getattr(args, "dmabuf_rest", None)
    browser_cpu = BrowserCpuStats(browser_url, on_air=lambda: startup_report(bridge)) if browser_url else None
    if browser_cpu:
        browser_cpu.start()
    if args.critical_nice:
        CriticalNice(mixer_pid, args.critical_nice).start()
    server = serve(bridge, args.bind, args.http_port, setup=setup, host_stats=host_stats,
                   preview_base=args.preview_base, price=price, recent=recent, browser_cpu=browser_cpu)
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
