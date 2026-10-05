#!/usr/bin/env python3
"""Browser control surface for the generic mixer demo.

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
import math
import os
import re
import selectors
import subprocess
import threading
import signal
import time
from urllib.parse import urlsplit
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from instance_profiles import InstanceType
from pyplumber.mixer.control import AvpConnection, mixer_command

log = logging.getLogger("webui")
PAGE = Path(__file__).with_name("webui") / "index.html"
WALL = PAGE.with_name("wall.html")
OUTPUTS_JS = PAGE.with_name("outputs.js")
# The page reads the backend's options from this script before its first request; the file holds `{}`, the defaults.
CONFIG_SCRIPT = '<script id="config" type="application/json">%s</script>'
TAKE_COMMANDS = ("cut", "fade", "wipe", "preview", "interrupt")
PROGRAM_TAKES = ("cut", "fade", "wipe")
AUX_COMMANDS = ("aux", "aux_layout", "aux_page")   # answered with the bus's layout, layouts and scenes, which setup keeps
STATE_TTL_S = 0.2


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


def compute_price(path: Path | None) -> dict | None:
    """Public hourly compute price, supplied by deployment rather than queried on each UI poll."""
    if path is None:
        return None
    value = json.loads(path.read_text("utf-8"))
    rate = value.get("hourly_usd")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("compute price hourly_usd must be a positive finite number")
    for key in ("label", "as_of", "source_url"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"compute price needs {key}")
    if urlsplit(value["source_url"]).scheme != "https":
        raise ValueError("compute price source_url must use HTTPS")
    return {key: value[key] for key in ("hourly_usd", "label", "as_of", "source_url")}


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
                "nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.decoder,utilization.encoder,"
                              "memory.used,memory.total,power.draw,enforced.power.limit,encoder.stats.sessionCount",
                "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=1)
            keys = ("index", "gpu", "decoder", "encoder", "memory_used_mib", "memory_total_mib")
            values = []
            session_counts = {}
            for row in csv.reader(output.splitlines()):
                if len(row) != len(keys) + 3 or not row[0].strip().isdigit():
                    continue
                sample = dict(zip(keys, (int(v) if v.strip().isdigit() else None for v in row[:len(keys)])))
                for key, raw in zip(("power_draw_w", "power_limit_w"), row[len(keys):-1]):
                    try:
                        watts = float(raw)
                        sample[key] = watts if math.isfinite(watts) and watts >= 0 else None
                    except ValueError:
                        sample[key] = None
                sample.update(encoder_sessions=None, encoder_fps=None, encoder_mpix_s=None)
                session_counts[sample["index"]] = int(row[-1]) if row[-1].strip().isdigit() else None
                values.append(sample)
            try:
                totals = _encoder_snapshot(session_counts)
                for sample in values:
                    index = sample["index"]
                    total = totals.get(index, (0, 0.0, 0.0) if session_counts[index] == 0 else None)
                    if total is not None and session_counts[index] == total[0]:
                        sample.update(zip(("encoder_sessions", "encoder_fps", "encoder_mpix_s"), total))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass   # Missing session telemetry must not hide utilization or power.
            self.values = values
        except (OSError, subprocess.SubprocessError):
            self.values = []
        finally:
            self.lock.release()
        return self.values


def _encoder_snapshot(session_counts):
    """encodersessions loops by default. Stop after a complete count-verified first table."""
    expected = {index: count for index, count in session_counts.items() if count}
    if not expected:
        return {}
    with subprocess.Popen(["nvidia-smi", "encodersessions"], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL) as process:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + 1
                output, totals = bytearray(), {}
                while len(output) < 1024 * 1024:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        break
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    output.extend(chunk)
                    if not output.endswith(b"\n"):
                        continue   # Never publish a row cut off halfway through a pipe read.
                    totals = _encoder_totals(output.decode("ascii"))
                    if all(totals.get(index) is not None and totals[index][0] >= count
                           for index, count in expected.items()):
                        return totals
                return totals if output.endswith(b"\n") else {}
        finally:
            process.kill()
            process.wait()


def _encoder_totals(output):
    """Aggregate complete driver session rows; a malformed row invalidates its GPU's totals."""
    totals = {}
    for line in output.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if not fields[0].isdigit():
            raise ValueError("unrecognized encoder session output")
        index = int(fields[0])
        try:
            if len(fields) != 8 or not all(fields[i].isdigit() for i in (1, 2, 4, 5)):
                raise ValueError("incomplete encoder session")
            width, height, fps, latency = map(float, fields[4:])
            if (width <= 0 or height <= 0 or not all(math.isfinite(v) and v >= 0 for v in (width, height, fps, latency))
                    or fields[3] in ("-", "N/A", "[N/A]")):
                raise ValueError("invalid encoder session")
            total = totals.setdefault(index, (0, 0.0, 0.0))
            if total is not None:
                total = (total[0] + 1, total[1] + fps, total[2] + width * height * fps / 1e6)
                totals[index] = total if all(math.isfinite(v) for v in total) else None
        except ValueError:
            totals[index] = None
    return totals


class HostStats:
    """Host load, CPU use and the mixer's busiest thread, sampled once a second on one thread and
    shared by every viewer. Inside the container /proc/loadavg and /proc/stat still describe the
    whole host, not the container's share. `values` is None where /proc is missing and until the
    second sample; its mixer fields are None while no mixer process runs."""

    def __init__(self, mixer_pid=lambda: None, proc=Path("/proc")):
        self.mixer_pid = mixer_pid   # the mixer's pid, or None while none runs
        self.proc = proc
        self.values = None
        self._last = None   # Previous time, host counters, pid, thread counters and process counters.

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="host-stats").start()

    def _run(self):
        failing = False   # the first failure of an outage is logged; a working sample ends it
        while True:
            try:
                self.sample()
                failing = False
            except Exception as exc:   # no /proc (macOS), a torn read, the mixer stopping mid-sample: the page hides the meters
                if not failing:
                    log.warning("Host stats unavailable: %s: %s", type(exc).__name__, exc)
                failing, self.values = True, None
            time.sleep(1)

    def sample(self):
        load1 = float((self.proc / "loadavg").read_text().split()[0])
        lines = (self.proc / "stat").read_text().splitlines()
        # user nice system idle iowait irq softirq steal; guest time is already inside user and nice.
        cpu = [int(v) for v in lines[0].split()[1:9]]
        pid, at = self.mixer_pid(), time.monotonic()
        threads = self._thread_ticks(pid) if pid else {}
        process = self._process_ticks(pid) if pid else None
        last, self._last = self._last, (at, cpu, pid, threads, process)
        if last is None:
            return
        last_at, last_cpu, last_pid, last_threads, last_process = last
        total, idle = sum(cpu) - sum(last_cpu), cpu[3] + cpu[4] - last_cpu[3] - last_cpu[4]
        values = dict(load1=load1, vcpus=sum(line[3:4].isdigit() for line in lines if line.startswith("cpu")),
                      cpu_pct=round((total - idle) / total * 100) if total > 0 else None,
                      mixer_cpu_pct=None, thread_pct=None, thread_name=None)
        elapsed = at - last_at
        same_process = pid and pid == last_pid and process and last_process and process[1] == last_process[1]
        if same_process and elapsed > 0:
            ticks = process[0] - last_process[0]
            if ticks >= 0:
                values["mixer_cpu_pct"] = round(ticks * 100 / os.sysconf("SC_CLK_TCK") / elapsed, 1)
        if pid and pid == last_pid and elapsed > 0 and (not process or not last_process or same_process):
            name, ticks = max(((name, ticks - last_threads.get(tid, ("", 0))[1]) for tid, (name, ticks) in threads.items()),
                              key=lambda item: item[1], default=(None, 0))
            if name is not None:
                values.update(thread_pct=round(ticks * 100 / os.sysconf("SC_CLK_TCK") / elapsed), thread_name=name)
        self.values = values   # a fresh dict: HTTP threads read the previous one meanwhile

    def _thread_ticks(self, pid):
        """{tid: (comm, utime + stime)} for every thread of *pid*: one listing and one read per thread."""
        ticks = {}
        for tid, stat in _thread_files(self.proc, pid, "stat"):
            name, used, _ = _cpu_stat(stat)
            ticks[tid] = (name, used)
        return ticks

    def _process_ticks(self, pid):
        try:
            return _cpu_stat((self.proc / str(pid) / "stat").read_text())[1:]
        except (OSError, ValueError, IndexError):
            return None


def _cpu_stat(stat):
    head, _, rest = stat.rpartition(")")   # comm may hold spaces and parentheses
    fields = rest.split()
    # Process utime/stime include exited threads; summing current task stats would lose their CPU time.
    return head.partition("(")[2], int(fields[11]) + int(fields[12]), int(fields[19])


def _thread_files(proc, pid, name):
    """(tid, text of /proc/pid/task/tid/name) for every thread of *pid*: one listing and one read
    per thread. Nothing when the mixer exited since its pid was read; a thread that exits between
    the listing and its read is left out."""
    tasks = proc / str(pid) / "task"
    try:
        tids = os.listdir(tasks)
    except OSError:
        return
    for tid in tids:
        try:
            yield tid, (tasks / tid / name).read_text()
        except OSError:
            continue


# Threads whose lateness shows on air, matched by comm: the kernel's thread name, cut to 15
# characters. avplumber names a node's thread after the node (src/util.cpp set_thread_name), so
# these are the scene compositors mixer_comp_a/b; "dsk_comp", the downstream keyer's compositor
# (pyplumber/mixer/dsk.py, no mixer_ prefix), which every program frame passes through after
# mixer_snapshot_output; the snapshot nodes mixer_snapshot_a/b/output, all "mixer_snapshot_"
# once cut; "EventLoop", the tick thread (src/EventLoop.hpp); the CUDA driver's own event thread
# "cuda-EvtHandlr"; "udp-tx", FFmpeg's paced UDP sender, which exists because the Janus RTP URL
# sets bitrate and fifo_size; and every NVENC node: janus_encoder, janus_<rendition>_encoder
# (janus_hdr_encoder arrives as "janus_hdr_encod", hence "_encod", not "encoder") and
# aux_<bus>_encoder, which the cut hides when the bus id is longer than five characters.
# The pass-through nodes between these stages (OneToMany mixer_otm_*, SourceSwitcher
# mixer_out_sel/mixer_wipe_sel, Split split_clean, janus_force_keyframe, janus_format,
# janus_repeat_headers, janus_mux, janus_rtp_output) are left at nice 0: each runs for
# microseconds per frame and sleeps otherwise, which CFS already wakes promptly. Widen the set
# only after measuring that it moves the missed-deadline counter.
CRITICAL_THREADS = re.compile(r"^(mixer_comp_|dsk_comp|mixer_snapshot|EventLoop|cuda-EvtHandlr|udp-tx)|_encod")
CRITICAL_NICE_PERIOD_S = 10


class CriticalNice:
    """Keep the mixer's deadline-critical threads (CRITICAL_THREADS) at nice -`level`, checked every
    CRITICAL_NICE_PERIOD_S with one task listing and one comm read per thread. Nice only, never a
    real-time class: a spinning thread must stay preemptible. Linux applies PRIO_PROCESS with a thread
    id to that thread alone. Thread ids are reused, so every match is set each period (idempotent);
    the log names each thread once per mixer process. Opt-in: needs CAP_SYS_NICE in the container."""

    def __init__(self, mixer_pid, level, proc=Path("/proc")):
        self.mixer_pid = mixer_pid   # the mixer's pid, or None while none runs
        self.level = level
        self.proc = proc
        self.enabled = True   # cleared on EPERM: without CAP_SYS_NICE every period would fail the same way
        self._logged = (None, set())   # (pid, {(tid, comm)}) already named in the log

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="critical-nice").start()

    def _run(self):
        while self.enabled:
            self.apply()
            time.sleep(CRITICAL_NICE_PERIOD_S)

    def apply(self):
        pid = self.mixer_pid()
        if not pid or not self.enabled:
            return
        applied = set()
        for tid, comm in _thread_files(self.proc, pid, "comm"):
            comm = comm.strip()
            if not CRITICAL_THREADS.search(comm):
                continue
            try:
                os.setpriority(os.PRIO_PROCESS, int(tid), -self.level)
            except ProcessLookupError:
                continue   # the thread exited after the listing
            except PermissionError:
                log.warning("Cannot set nice -%d on mixer thread %s (%s): the container needs CAP_SYS_NICE; "
                            "critical-thread priority is off", self.level, comm, tid)
                self.enabled = False
                return
            applied.add((tid, comm))
        logged_pid, logged = self._logged
        if logged_pid != pid:   # a restarted mixer: its thread ids say nothing about the previous ones
            logged = set()
        new = applied - logged
        if new:
            log.info("Mixer %s: nice -%d on %d %s: %s", pid, self.level, len(new), "thread" if len(new) == 1 else "threads",
                     ", ".join(f"{comm} ({tid})" for tid, comm in sorted(new, key=lambda item: (item[1], int(item[0])))))
        self._logged = (pid, logged | new)


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
        """Returns the mixer's answer to an aux, aux_layout, aux_page or dsk command; for a take, whether it was sent
        (False when a newer program take superseded it unsent)."""
        command = request.get("command")
        if command in ("dsk", *AUX_COMMANDS):
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

    def __init__(self, bridge: MixerBridge, *args, setup=None, gpu=None, host_stats=None, config=None, **kwargs):
        self.bridge = bridge
        self.setup_manager = setup
        self.gpu = gpu
        self.host_stats = host_stats
        self.config = config or {}
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
            self._send(200, page(self.config), "text/html; charset=utf-8")
        elif path in ("/wall", "/wall/"):
            self._send(200, page(self.config, WALL), "text/html; charset=utf-8")
        elif path == "/outputs.js":
            self._send(200, OUTPUTS_JS.read_bytes(), "text/javascript; charset=utf-8")
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
                state["host"] = self.host_stats.values if self.host_stats else None
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


def serve(bridge: MixerBridge, bind: str, port: int, setup=None, host_stats: HostStats | None = None,
          preview_base: str | None = None, price: dict | None = None) -> ThreadingHTTPServer:
    """`preview_base`: where the page's players load from, when not port 8080 of the page's host.
    Normalized here, so every caller's page carries a base the player resolves its files against."""
    preview_base = normalize_preview_base(preview_base or "")
    config = {"preview_base": preview_base} if preview_base else {}
    if price:
        config["compute_price"] = price
    server = ThreadingHTTPServer((bind, port), partial(Handler, bridge, setup=setup, gpu=GpuStats(), host_stats=host_stats,
                                                        config=config))
    server.daemon_threads = True
    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--manage-setup", action="store_true", help="Own and restart the demo mixer process")
    instance_types = [t.value for t in InstanceType]
    parser.add_argument("--instance-type", choices=instance_types,
                        help="The host type whose measured source limits the setup applies (instance_profiles.py); "
                             "required with --manage-setup")
    parser.add_argument("--media-dir", type=Path, default=Path("/media"))
    parser.add_argument("--recipe", type=Path, default=Path("/media/demo.json"))
    parser.add_argument("--dmabuf-rest", default="http://127.0.0.1:9009")
    parser.add_argument("--janus-api", metavar="URL",
                        help="Janus HTTP API (e.g. http://127.0.0.1:8088/janus, no API secret) for the setup's "
                             "extra aux outputs, one Streaming mountpoint each; without it the setup offers none")
    # A string default goes through type=int like a command-line value, so a bad env value is a usage error.
    parser.add_argument("--critical-nice", type=int, metavar="N", default=os.environ.get("MIXER_CRITICAL_NICE", "0"),
                        help="Keep the deadline-critical mixer threads at nice -N, 1 to 20, rechecked every "
                             f"{CRITICAL_NICE_PERIOD_S} s; needs CAP_SYS_NICE and --manage-setup. 0, the default, "
                             "leaves them alone. Env: MIXER_CRITICAL_NICE")
    parser.add_argument("--mixer-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    if not 0 <= args.critical_nice <= 20:
        parser.error("--critical-nice must be 0 (off) or 1 to 20")
    if args.critical_nice and not args.manage_setup:
        parser.error("--critical-nice needs --manage-setup: the mixer pid comes from the managed process")
    if args.manage_setup and not args.instance_type:
        parser.error("--manage-setup needs --instance-type: source limits are measured per host type "
                     f"({', '.join(instance_types)}); another machine needs its own measured profile "
                     "in instance_profiles.py")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    price = compute_price(args.compute_price)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    bridge = MixerBridge(args.host, args.port, args.mixer, transition=args.transition)
    setup = None
    if args.manage_setup:
        from setup_runtime import SetupRuntime
        setup = SetupRuntime(args.media_dir, args.recipe, bridge, args.instance_type, args.mixer_args, args.dmabuf_rest,
                             args.janus_api)
        setup.resume()
    def mixer_pid():
        process = setup.process if setup else None   # read once: the setup worker thread clears it when the mixer stops
        return process.pid if process else None
    host_stats = HostStats(mixer_pid)
    host_stats.start()
    if args.critical_nice:
        CriticalNice(mixer_pid, args.critical_nice).start()
    server = serve(bridge, args.bind, args.http_port, setup=setup, host_stats=host_stats,
                   preview_base=args.preview_base, price=price)
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
