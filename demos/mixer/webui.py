#!/usr/bin/env python3
"""Browser control surface for the generic mixer demo.

Serves one static page and bridges it to AVPlumber's line protocol, so the
mixer needs no HTTP server of its own and the page needs no build step:

    GET  /              the page
    GET  /api/state     status + scenes + settings in one round trip
    GET  /api/status    mixer.status alone, for scripts that poll often
    POST /api/command   {"command": "cut"|"fade"|"wipe"|"preview", "scene": ...}
                        (a fade may carry "curve": "linear"|"ease-in"|"ease-out"|"ease-in-out",
                        and "color": "#RRGGBB" to dip through that colour instead of mixing)
                        {"command": "dsk", "key": ..., "on": true|false, "fade_seconds"?: s, "curve"?: ...}
                        A cut, fade or wipe superseded by a newer one before it reached
                        the mixer answers {"ok": true, "superseded": true}.

The bridge owns a single serialized control connection and reconnects when the
mixer restarts, so the page can stay open across a demo restart. Takes must not
queue behind polling: every tab shares one /api/state reply for STATE_TTL_S, and
a burst of program takes (keyboard auto-repeat) collapses to its newest one.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import json
import logging
import os
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
PROGRAM_TAKES = ("cut", "fade", "wipe")
STATE_TTL_S = 0.2


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


class HostStats:
    """Host load, CPU use and the mixer's busiest thread, sampled once a second on one thread and
    shared by every viewer. Inside the container /proc/loadavg and /proc/stat still describe the
    whole host, not the container's share. `values` is None where /proc is missing and until the
    second sample; its thread fields are None while no mixer process runs."""

    def __init__(self, mixer_pid=lambda: None, proc=Path("/proc")):
        self.mixer_pid = mixer_pid   # the mixer's pid, or None while none runs
        self.proc = proc
        self.values = None
        self._last = None   # (monotonic time, cpu counters, pid, {tid: (comm, ticks)}) of the previous sample

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="host-stats").start()

    def _run(self):
        while True:
            try:
                self.sample()
            except (OSError, ValueError):   # no /proc (macOS), or a torn read of it
                self.values = None
            time.sleep(1)

    def sample(self):
        load1 = float((self.proc / "loadavg").read_text().split()[0])
        lines = (self.proc / "stat").read_text().splitlines()
        # user nice system idle iowait irq softirq steal; guest time is already inside user and nice.
        cpu = [int(v) for v in lines[0].split()[1:9]]
        pid, at = self.mixer_pid(), time.monotonic()
        threads = self._thread_ticks(pid) if pid else {}
        last, self._last = self._last, (at, cpu, pid, threads)
        if last is None:
            return
        last_at, last_cpu, last_pid, last_threads = last
        total, idle = sum(cpu) - sum(last_cpu), cpu[3] + cpu[4] - last_cpu[3] - last_cpu[4]
        values = dict(load1=load1, vcpus=sum(line[3:4].isdigit() for line in lines if line.startswith("cpu")),
                      cpu_pct=round((total - idle) / total * 100) if total > 0 else None, thread_pct=None, thread_name=None)
        if pid and pid == last_pid:   # a restarted mixer's thread ids say nothing about the previous ones
            name, ticks = max(((name, ticks - last_threads.get(tid, ("", 0))[1]) for tid, (name, ticks) in threads.items()),
                              key=lambda item: item[1], default=(None, 0))
            if name is not None:
                values.update(thread_pct=round(ticks * 100 / os.sysconf("SC_CLK_TCK") / (at - last_at)), thread_name=name)
        self.values = values   # a fresh dict: HTTP threads read the previous one meanwhile

    def _thread_ticks(self, pid):
        """{tid: (comm, utime + stime)} for every thread of *pid*: one listing and one read per thread."""
        tasks = self.proc / str(pid) / "task"
        try:
            tids = os.listdir(tasks)
        except OSError:
            return {}   # the mixer exited since its pid was read
        ticks = {}
        for tid in tids:
            try:
                stat = (tasks / tid / "stat").read_text()
            except OSError:
                continue   # the thread exited between the listing and the read
            head, _, rest = stat.rpartition(")")   # comm may hold spaces and parentheses
            fields = rest.split()
            ticks[tid] = (head.partition("(")[2], int(fields[11]) + int(fields[12]))   # stat fields 14 and 15
        return ticks


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
        self._init_sharing()

    def _init_sharing(self) -> None:
        """State every HTTP thread shares: the cached /api/state reply and the take queue."""
        self._state_lock = threading.Lock()
        self._state: dict | None = None
        self._state_at = self._changed_at = float("-inf")
        self._takes = threading.Condition()
        self._take_arrivals = 0
        self._take_queue: collections.deque[int] = collections.deque()   # waiting, in arrival order
        self._waiting_program: int | None = None
        self._take_running = False

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

    def status(self, timeout: float | None = None) -> dict:
        return json.loads(self.command(f"mixer.status {self.mixer}", timeout) or "{}")

    def state(self, timeout: float | None = None) -> dict:
        """Everything the page redraws from, in one poll. Polls within STATE_TTL_S of each
        other share one reply, and a poll arriving while another is in flight waits for it,
        so any number of tabs costs one set of commands; a take or key change expires it.
        A caller polling a mixer that is still building its graph passes a longer `timeout`:
        the mixer runs control commands on the thread that is busy starting nodes, so replies
        stall for seconds during startup even though they take a millisecond once it is running."""
        with self._state_lock:
            now = time.monotonic()
            if self._state is None or now - self._state_at >= STATE_TTL_S or self._state_at <= self._changed_at:
                self._state, self._state_at = self._query_state(timeout), now
            return dict(self._state)   # callers add their own top-level keys

    def _query_state(self, timeout: float | None) -> dict:
        state: dict = {"mixer": self.mixer}
        state["status"] = self.status(timeout)
        state["scenes"] = json.loads(self.command(f"mixer.scenes {self.mixer}", timeout) or "[]")
        try:
            state["settings"] = json.loads(self.command(f"mixer.settings {self.mixer}", timeout) or "{}")
        except Exception:
            state["settings"] = {}   # older mixers, or one started without a config
        if self.transition is not None:
            state["settings"]["transition"] = self.transition
        if state["settings"].get("aux_buses"):
            state["aux_buses"] = json.loads(self.command("mixer.aux_status", timeout) or "[]")
        if state["settings"].get("dsk_keys"):
            state["dsk"] = json.loads(self.command("mixer.dsk_status", timeout) or "[]")
        return state

    def take(self, request: dict):
        """Returns the mixer's answer to an aux, aux_page or dsk command; for a take, whether it was sent
        (False when a newer program take superseded it unsent)."""
        command = request.get("command")
        if command in ("aux", "aux_page", "dsk"):
            payload = {k: v for k, v in request.items() if k != "command"}
            try:
                result = json.loads(self.command(f"mixer.{command} " + json.dumps(payload)) or "{}")
            finally:
                self._changed_at = time.monotonic()
            if isinstance(result, dict) and result.get("error"):
                raise ValueError(result["error"])
            return result
        if command not in TAKE_COMMANDS:
            raise ValueError(f"command must be one of {', '.join(TAKE_COMMANDS)}")
        payload = {k: v for k, v in request.items() if k not in ("command", "mixer")}
        if command != "interrupt" and not payload.get("scene"):
            raise ValueError(f"{command} needs a scene")
        return self._send_in_order(mixer_command(command, self.mixer, **payload), command in PROGRAM_TAKES)

    def _send_in_order(self, line: str, program: bool) -> bool:
        """Takes reach the mixer one at a time, in arrival order. A cut, fade or wipe sets the
        whole program state, so a newer one supersedes the one still waiting: a burst collapses
        to its last take, which is never dropped or overtaken. A preview or interrupt changes
        only part of it; it is never superseded and never supersedes a program take."""
        with self._takes:
            self._take_arrivals += 1
            mine = self._take_arrivals
            if program:
                if self._waiting_program is not None:
                    self._take_queue.remove(self._waiting_program)
                    self._takes.notify_all()   # the superseded take answers now
                self._waiting_program = mine
            self._take_queue.append(mine)
            while mine in self._take_queue and (self._take_running or self._take_queue[0] != mine):
                self._takes.wait()
            if mine not in self._take_queue:
                return False
            self._take_queue.popleft()
            if self._waiting_program == mine:
                self._waiting_program = None
            self._take_running = True
        try:
            self.command(line)
        finally:
            with self._takes:
                self._take_running = False
                self._changed_at = time.monotonic()
                self._takes.notify_all()
        return True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "avplumber-mixer-webui"

    def __init__(self, bridge: MixerBridge, *args, setup=None, gpu=None, host=None, **kwargs):
        self.bridge = bridge
        self.setup_manager = setup
        self.gpu = gpu
        self.host = host
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
                state["host"] = self.host.values if self.host else None
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
            request = json.loads(body or b"{}")
            result = self.bridge.take(request)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(502, {"error": str(exc)})
            return
        if self.setup_manager and request.get("command") == "aux":
            try:
                self.setup_manager.remember_aux(request.get("bus"), result["scenes"])
            except Exception as exc:   # the tiles are live; only their persistence failed
                print(f"Could not persist aux {request.get('bus')}: {exc}", flush=True)
        self._send_json(200, {"ok": True, "superseded": True} if result is False else {"ok": True})


def serve(bridge: MixerBridge, bind: str, port: int, setup=None, host: HostStats | None = None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((bind, port), partial(Handler, bridge, setup=setup, gpu=GpuStats(), host=host))
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
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
    host = HostStats(lambda: setup.process.pid if setup and setup.process else None)
    host.start()
    server = serve(bridge, args.bind, args.http_port, setup=setup, host=host)
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
