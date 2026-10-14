"""Reusable NVIDIA/NVENC, host and browser-service CPU telemetry; independent of the web UI's HTTP
server and of input policy."""
from __future__ import annotations

import collections
import csv
import json
import logging
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

log = logging.getLogger("webui")


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
                sample: dict[str, int | float | None] = dict(zip(keys, (int(v) if v.strip().isdigit() else None for v in row[:len(keys)])))
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
            assert process.stdout is not None
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


def delivery_totals(state: dict) -> dict | None:
    """The page's two delivery counters from one mixer state: deadlines missed by the program
    slots and the AUX buses, and frames the AUX outputs dropped. None while the state lacks a
    count or a bus its settings name, as the page shows a dash then."""
    playout = list(((state.get("status") or {}).get("playout") or {}).values())
    aux = state.get("aux_buses") or []
    if not playout or len(aux) != len((state.get("settings") or {}).get("aux_buses") or []):
        return None
    missed = [(row or {}).get("missed_deadlines") for row in playout + [bus.get("playout") for bus in aux]]
    drops = [bus.get("output_drops") for bus in aux]
    if any(isinstance(n, bool) or not isinstance(n, int) for n in missed + drops):
        return None
    return {"missed_deadlines": sum(missed), "output_drops": sum(drops)}


def startup_report(bridge) -> dict | None:
    """`mixer.startup`: how the running mixer went on air (its `released_at` is the on-air moment,
    `browser_windows` the browser service's counters then), or None from a mixer without the
    command or one not answering."""
    try:
        report = json.loads(bridge.command("mixer.startup", 3) or "null")
    except Exception:
        return None
    return report if isinstance(report, dict) else None


ON_AIR = "on_air"


def playout_counters(bridge) -> dict | None:
    """delivery_totals() and, where the show names its compositor inputs (settings "inputs"), each
    source's repeats and discards from both program-slot compositors' per_input counters: two small
    commands per call, which RecentCounts makes every 5 s. A repeat with a discard beside it is two
    paints in one tick slot (jitter); a repeat alone is a tick no frame arrived for. With the
    mixer's on-air moment (startup_report) as ON_AIR, so that the window starts again with a show
    that went on air anew; the counters themselves start from zero then."""
    state = bridge.state(timeout=3)
    totals = delivery_totals(state)
    inputs = (state.get("settings") or {}).get("inputs")
    if totals is None:
        return None
    report = startup_report(bridge)
    if report and report.get("released_at") is not None:
        totals[ON_AIR] = report["released_at"]
    if not inputs:
        return totals
    slots = [json.loads(bridge.command(f"node.object.get {bridge.mixer}_comp_{slot} per_input", 3) or "[]")
             for slot in ("a", "b")]
    return {**totals, **input_counters(inputs, slots)}


def input_counters(inputs, slots):
    """{("input", source, group, counter): total} from per_input lists in compositor input order;
    group is "browser" for a browser (dmabuf) source and "other" for the rest. A source scenes
    use twice has an input per use, all fed the same frames: its first input speaks for it."""
    counters, seen = {}, set()
    for index, (source, kind) in enumerate(inputs):
        if source in seen:
            continue
        seen.add(source)
        group = "browser" if kind == "browser" else "other"
        for counter in ("repeats", "discarded"):
            values = [rows[index][counter] for rows in slots if index < len(rows)]
            if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
                raise ValueError("malformed per_input counters")
            counters[("input", source, group, counter)] = sum(values)
    return counters


def playout_summary(rises):
    """RecentCounts' summary: the plain counters as they are, and the inputs' rises as
    {"inputs": {"browser": {...}, "other": {...}}}, each with its sources, repeats and discards,
    the browser group also with its worst source (most repeats, then most discards) or None."""
    plain = {key: value for key, value in rises.items() if not isinstance(key, tuple)}
    per_source = {}
    for (_, source, group, counter), value in ((k, v) for k, v in rises.items() if isinstance(k, tuple)):
        per_source.setdefault((group, source), {"repeats": 0, "discarded": 0})[counter] = value
    if not per_source:
        return plain
    groups = {}
    for name in ("browser", "other"):
        rows = [(source, c) for (group, source), c in per_source.items() if group == name]
        summary = {"sources": len(rows), "repeats": sum(c["repeats"] for _, c in rows),
                   "discarded": sum(c["discarded"] for _, c in rows)}
        if name == "browser":
            worst = max(rows, key=lambda row: (row[1]["repeats"], row[1]["discarded"]), default=None)
            summary["worst"] = ({"source": worst[0], **worst[1]}
                                if worst and (worst[1]["repeats"] or worst[1]["discarded"]) else None)
        groups[name] = summary
    return {**plain, "inputs": groups}


class RecentCounts:
    """How far counters that only grow rose in the last `window_s` seconds. The page shows this
    instead of the totals since the mixer started, which keep the misses of a start or of a
    disturbance long past on screen for good. `read` returns the totals as a dict, or None while
    there are none. One thread samples every `period_s`, so the window is full whether or not a
    page is open. `values` is None until the first sample; `covered_s` is how much of the window
    the samples span so far. Totals that fall belong to a mixer that started again and counts from
    zero: everything it counted happened inside the window. Counters that appear or go away (another
    show's inputs) start the window again from the current totals. An ON_AIR key among the totals is
    not a counter but the mixer's on-air moment: when it changes, the mixer went on air anew with
    its counters reset, so everything it counted is recent (whether or not every total fell), and
    `values` carries it. `summary` turns the rises into what the page reads (by default they are
    the values themselves)."""

    def __init__(self, read, window_s: float = 600, period_s: float = 5, clock=time.monotonic, summary=None):
        self.read, self.window_s, self.period_s, self.clock = read, window_s, period_s, clock
        self.summary = summary or (lambda rises: rises)
        self.values = None
        self.on_air = None
        self._samples: list[tuple[float, dict]] = []

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="recent-counts").start()

    def _run(self):
        while True:
            try:
                self.sample()
            except Exception:   # no mixer yet, or one busy starting: the last values stand until it answers
                pass
            time.sleep(self.period_s)

    def sample(self):
        totals = self.read()
        if totals is None:
            return
        totals = dict(totals)
        on_air = totals.pop(ON_AIR, None)
        now = self.clock()
        last = self._samples[-1][1] if self._samples else None
        if last is not None and (on_air != self.on_air or any(totals.get(key, count) < count for key, count in last.items())):
            self._samples = [(now, dict.fromkeys(totals, 0))]
        elif last is not None and set(totals) != set(last):
            self._samples = []
        self.on_air = on_air
        self._samples.append((now, dict(totals)))
        while self._samples[0][0] < now - self.window_s:
            self._samples.pop(0)
        since, base = self._samples[0]
        self.values = {"window_s": self.window_s, "covered_s": round(now - since),
                       **({ON_AIR: on_air} if on_air is not None else {}),
                       **self.summary({key: count - base[key] for key, count in totals.items()})}


class HostStats:
    """Host load, CPU use and the mixer's busiest threads, sampled once a second on one thread and
    shared by every viewer. Inside the container /proc/loadavg and /proc/stat still describe the
    whole host, not the container's share. `values` is None where /proc is missing and until the
    second sample; its mixer fields are None while no mixer process runs."""

    THREAD_KINDS = 8     # kinds of thread listed, the largest total first; the rest are "other"
    KINDS_WINDOW = 5     # samples a kind's figures span: one thread's second is a tick or two

    def __init__(self, mixer_pid=lambda: None, proc=Path("/proc")):
        self.mixer_pid = mixer_pid   # the mixer's pid, or None while none runs
        self.proc = proc
        self.values = None
        self._last = None   # Previous time, host counters, pid, thread counters and process counters.
        self._thread_history = []   # (time, pid, thread counters) of the last KINDS_WINDOW samples

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
                      mixer_cpu_pct=None, thread_pct=None, thread_name=None, thread_count=None, thread_kinds=None)
        elapsed = at - last_at
        same_process = pid and pid == last_pid and process and last_process and process[1] == last_process[1]
        if same_process and process is not None and last_process is not None and elapsed > 0:
            ticks = process[0] - last_process[0]
            if ticks >= 0:
                values["mixer_cpu_pct"] = round(ticks * 100 / os.sysconf("SC_CLK_TCK") / elapsed, 1)
        if pid and pid == last_pid and elapsed > 0 and (not process or not last_process or same_process):
            hz = os.sysconf("SC_CLK_TCK")
            name, ticks = max(((name, ticks - last_threads.get(tid, ("", 0))[1]) for tid, (name, ticks) in threads.items()),
                              key=lambda item: item[1], default=(None, 0))
            if name is not None:
                # The kinds over a few seconds, from the oldest sample of this process still kept.
                self._thread_history = [h for h in self._thread_history if h[1] == pid][-(self.KINDS_WINDOW - 1):]
                since_at, _, since = self._thread_history[0] if self._thread_history else (last_at, pid, last_threads)
                self._thread_history.append((last_at, pid, last_threads))
                used = [(ticks - since.get(tid, ("", 0))[1], name) for tid, (name, ticks) in threads.items()]
                values.update(thread_pct=round(ticks * 100 / hz / elapsed), thread_name=name, thread_count=len(threads),
                              thread_kinds=thread_kinds(used, hz * (at - since_at), self.THREAD_KINDS))
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


def thread_kinds(used, tick_seconds, limit):
    """Threads summed by kind: [{"name", "count", "pct", "avg_pct", "max_pct"}], the kind that used the most
    CPU in total first (then the one with the busiest thread), at most *limit* kinds and then the
    rest as "other". *used* holds (ticks, thread name).

    A show has hundreds of threads that each use a fraction of a percent and together a core:
    one per source for input, decoding, upload, fan-out. Digits in a name become N, so
    input_12 and input_140 are both input_N."""
    kinds = {}
    for ticks, name in used:
        kind = kinds.setdefault(re.sub(r"\d+", "N", name), {"count": 0, "ticks": 0, "max": 0, "cut": True})
        kind["count"] += 1
        kind["ticks"] += ticks
        kind["max"] = max(kind["max"], ticks)
        kind["cut"] &= len(name) >= 15
    # The kernel keeps 15 characters of a name, so aux_aux1_encoder and aux_aux10_encoder arrive
    # as aux_aux1_encode and aux_aux10_encod: one kind under two spellings.
    for name in sorted((n for n, k in kinds.items() if k["cut"]), key=len, reverse=True):
        shorter = next((n for n, k in kinds.items() if n != name and k["cut"] and name.startswith(n)), None)
        if shorter:
            longer = kinds.pop(name)
            kinds[shorter].update(count=kinds[shorter]["count"] + longer["count"], ticks=kinds[shorter]["ticks"] + longer["ticks"],
                                  max=max(kinds[shorter]["max"], longer["max"]))
    pct = lambda ticks: round(ticks * 100 / tick_seconds, 1) if tick_seconds > 0 else 0.0
    ranked = sorted(kinds.items(), key=lambda item: (-item[1]["ticks"], -item[1]["max"], item[0]))
    row = lambda name, count, ticks, most: {"name": name, "count": count, "pct": pct(ticks),
                                            "avg_pct": pct(ticks / count), "max_pct": pct(most)}
    rows = [row(name, k["count"], k["ticks"], k["max"]) for name, k in ranked[:limit]]
    rest = [k for _, k in ranked[limit:]]
    if rest:
        rows.append(row("other", sum(k["count"] for k in rest), sum(k["ticks"] for k in rest), max(k["max"] for k in rest)))
    return rows


class BrowserCpuStats:
    """Chromium's CPU use, from the browser service's GET /metrics/cpu (deps/dma-browser): the mixer
    container cannot see the browser container's processes. Polled every 5 s on one thread and
    shared by every viewer. `values` is chromium_summary() of the last answer, or None until the
    second answer (the first only takes the service's baseline) and while the service is down. A
    browser image without the endpoint answers 404; it is asked again every `absent_s` seconds."""

    PATH = "/metrics/cpu"

    # Every 5 s: the hover needs no more, and each answer costs the browser service a /proc read.
    def __init__(self, base_url: str, period_s: float = 5, absent_s: float = 30, timeout_s: float = 1, fetch=None,
                 fetch_status=None, clock=time.monotonic, on_air=None, wall=time.time):
        self.base_url = base_url.rstrip("/")
        self.period_s, self.absent_s, self.timeout_s = period_s, absent_s, timeout_s
        self.fetch = fetch or (lambda: self._get(self.PATH))
        self.fetch_status = fetch_status or (lambda: self._get("/status"))
        self.values = None
        self.error = None   # why the last poll has no values, or None
        # The windows' drops, paints and sends over the last ten minutes, from GET /status at the
        # same cadence: browser_drop_summary() of the rises, or None before the first answer.
        self.drops = RecentIncrements(clock=clock, summary=browser_drop_summary, group=lambda key: key[0])
        # `on_air` gives the mixer's startup report (telemetry.startup_report) or None. The windows
        # count from before the mixer exists (their pages load and drop frames while the show is
        # built), so when the report's on-air moment is new the drop window starts again from the
        # counters the mixer read at that moment: nothing from the warm-up is recent.
        self.on_air, self.clock, self.wall = on_air, clock, wall
        self._on_air_seen = None

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="browser-cpu").start()

    def _get(self, path):
        with urllib.request.urlopen(self.base_url + path, timeout=self.timeout_s) as response:
            return json.loads(response.read())

    def _run(self):
        failing = False   # the first failed poll of an outage is logged; a good one ends it
        metrics_due = 0.0
        while True:
            now = time.monotonic()
            if now >= metrics_due:   # an image without /metrics/cpu is asked again after absent_s
                metrics_due = now + self.poll()
                if self.error and not failing:
                    log.warning("Chromium CPU unavailable from %s%s: %s", self.base_url, self.PATH, self.error)
                failing = self.error is not None
            self.poll_status()
            time.sleep(self.period_s)

    def poll_status(self):
        """One GET /status into the drop window, after a new on-air moment has started it again
        (rebase_on_air). A failed one leaves the last figures standing; the next good one counts
        what rose meanwhile (or, for a restarted window, since its restart)."""
        try:
            self.rebase_on_air()
        except Exception as exc:   # a report the mixer should not have given: the window goes on as it was
            log.warning("Browser drop window not rebased on air: %s: %s", type(exc).__name__, exc)
        try:
            self.drops.sample(window_counters(self.fetch_status()))
        except Exception:   # down or restarting: never end the polling thread
            pass

    def rebase_on_air(self):
        """Start the drop window again from the mixer's on-air counters when its on-air moment is
        new. A report without them (the service did not answer the mixer then) starts the window
        with the next poll instead."""
        report = self.on_air() if self.on_air else None
        released_at = report.get("released_at") if report else None
        if released_at is None or released_at == self._on_air_seen:
            return
        self._on_air_seen = released_at
        windows = report.get("browser_windows")
        at = self.clock() - max(0.0, self.wall() - released_at)
        if windows is None:
            self.drops.restart(at, epoch=released_at)
            log.warning("Browser drop window restarted at on air without the mixer's browser counters")
            return
        counters = window_counters({"windows": [{"id": wid, "stats": stats} for wid, stats in windows.items()]})
        self.drops.restart(at, epoch=released_at, totals=counters)

    def poll(self) -> float:
        """One request; returns how long to wait before the next."""
        self.error = None
        try:
            self.values = chromium_summary(self.fetch())
            return self.period_s
        except urllib.error.HTTPError as exc:
            self.values, self.error = None, f"HTTP {exc.code}"
            return self.absent_s if exc.code == 404 else self.period_s
        except Exception as exc:   # down, restarting, or an unexpected answer: never end the polling thread
            self.values, self.error = None, f"{type(exc).__name__}: {exc}"
            return self.period_s


class RecentIncrements:
    """How much each counter rose over the last `window_s` seconds, counter by counter: a counter
    that falls started again (a window's worker restarted) and counts from zero, one that appears
    (a window opened) counts from zero, one that goes away keeps what it added inside the window.
    The first sample is the baseline. `values` is {"window_s", "covered_s", **summary(rises)}, or
    None before the first sample. Unlike RecentCounts, one counter's restart leaves the others; with
    `group` (key -> group), one fallen counter restarts every counter of its group (a window's)."""

    def __init__(self, window_s: float = 600, clock=time.monotonic, summary=None, group=None):
        self.window_s, self.clock = window_s, clock
        self.summary = summary or (lambda rises: rises)
        self.group = group or (lambda key: key)
        self.values = None
        self.epoch = None   # what the window counts from (a mixer's on-air moment), in `values` as "since"
        self._last = None   # (time, totals) of the last sample
        self._steps = collections.deque()   # (start, end, {key: rise}) between successive samples

    def restart(self, at, *, epoch=None, totals=None):
        """Forget the window: it counts from *at* (the clock's time) on, as `epoch` names. With
        *totals*, the counters' values at *at*, the next sample's rise is counted from them; without,
        the next sample is the baseline and `values` stay as they were until it comes."""
        self._steps.clear()
        self._last = (at, dict(totals)) if totals is not None else None
        self.epoch = epoch
        if totals is not None:
            self.values = {"window_s": self.window_s, "covered_s": round(self.clock() - at),
                           **({"since": epoch} if epoch is not None else {}), **self.summary(dict.fromkeys(totals, 0))}

    def sample(self, totals):
        now = self.clock()
        if self._last is not None:
            since, before = self._last
            restarted = {self.group(key) for key, value in totals.items() if key in before and value < before[key]}
            rises = {key: value - before[key] if key in before and self.group(key) not in restarted else value
                     for key, value in totals.items()}
            self._steps.append((since, now, rises))
        while self._steps and self._steps[0][1] < now - self.window_s:
            self._steps.popleft()
        self._last = (now, dict(totals))
        sums = dict.fromkeys(totals, 0)
        for _, _, rises in self._steps:
            for key, value in rises.items():
                sums[key] = sums.get(key, 0) + value
        start = self._steps[0][0] if self._steps else now
        self.values = {"window_s": self.window_s, "covered_s": round(now - start),
                       **({"since": self.epoch} if self.epoch is not None else {}), **self.summary(sums)}


WINDOW_COUNTERS = {"paints": "paintCount", "sent": "txFrameCount", "dropped": "droppedFrames"}


def window_counters(status):
    """{(window id, counter): total} from the browser service's GET /status: paints, sent and
    dropped frames, drops by reason as (id, "reason", reason), and (id, "skipped") from an app
    that counts the paints Chromium skipped (stats.paintsSkipped)."""
    counters = {}
    count = lambda value: isinstance(value, int) and not isinstance(value, bool) and value >= 0
    for window in status.get("windows") or []:
        wid, stats = window["id"], window["stats"]
        for key, field in WINDOW_COUNTERS.items():
            if not count(stats[field]):
                raise ValueError(f"malformed browser window counter {field}")
            counters[(wid, key)] = stats[field]
        for reason, value in (stats.get("droppedReasons") or {}).items():
            if not count(value):
                raise ValueError("malformed browser drop reason")
            counters[(wid, "reason", reason)] = value
        if "paintsSkipped" in stats:
            if not count(stats["paintsSkipped"]):
                raise ValueError("malformed paintsSkipped")
            counters[(wid, "skipped")] = stats["paintsSkipped"]
    return counters


def browser_drop_summary(rises):
    """The LAT tip's browser rows from window_counters() rises: windows seen, paints, sent and
    dropped in all, drops by reason (most first), the window that dropped most (or None), and the
    paints Chromium skipped, None while no window reports them (an older app)."""
    ids = {key[0] for key in rises}
    total = lambda name: sum(value for key, value in rises.items() if key[1:] == (name,))
    reasons = {}
    for key, value in rises.items():
        if len(key) == 3 and value:
            reasons[key[2]] = reasons.get(key[2], 0) + value
    dropped = {wid: rises.get((wid, "dropped"), 0) for wid in ids}
    worst = max(sorted(dropped), key=lambda wid: dropped[wid], default=None)
    skipped = [value for key, value in rises.items() if key[1:] == ("skipped",)]
    return {"windows": len(ids), "paints": total("paints"), "sent": total("sent"), "dropped": total("dropped"),
            "reasons": sorted(reasons.items(), key=lambda item: (-item[1], item[0])),
            "worst": {"id": worst, "dropped": dropped[worst]} if worst is not None and dropped[worst] else None,
            "skipped": sum(skipped) if skipped else None}


CHROMIUM_PROCESS_NAMES = {"browser": "browser", "gpu": "GPU process", "renderer": "renderer",
                          "utility": "utility", "zygote": "zygote", "other": "other"}
CHROMIUM_GPU_THREAD_NAMES = {"viz": "viz compositor", "main": "GPU main", "io": "IO", "other": "other"}


def chromium_summary(raw):
    """The page's Chromium section from one GET /metrics/cpu answer, or None while the answer has
    no interval yet. Rows have the THREAD tip's shape, {"name", "count", "pct", "avg_pct", "max_pct"}: the
    process types the largest total first and the GPU process's threads in a fixed order (viz
    compositor, GPU main, IO, other), under Chromium's cores' worth in all."""
    if raw.get("intervalS") is None:
        return None
    number = lambda value: value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None

    def rows(items, names):
        out = []
        for item in items or []:
            name, count, pct, max_pct = item.get("type", item.get("name")), item["n"], number(item["pct"]), number(item["maxPct"])
            if not isinstance(count, int) or isinstance(count, bool) or pct is None or max_pct is None:
                raise ValueError("malformed Chromium CPU row")
            out.append({"name": names.get(name, str(name)), "count": count, "pct": pct,
                        "avg_pct": round(pct / count, 1) if count > 0 else 0.0, "max_pct": max_pct})
        return out

    processes = sorted(rows(raw.get("processes"), CHROMIUM_PROCESS_NAMES), key=lambda r: (-r["pct"], -r["max_pct"]))
    order = list(CHROMIUM_GPU_THREAD_NAMES.values())
    gpu = sorted(rows(raw.get("gpuThreads"), CHROMIUM_GPU_THREAD_NAMES), key=lambda r: order.index(r["name"]) if r["name"] in order else len(order))
    cores = raw.get("cores") or {}
    total = number(raw.get("totalPct"))
    return {"total_pct": total, "cores_used": round(total / 100, 2) if total is not None else None,
            "vcpus": cores.get("hostVcpus"), "processes": processes, "gpu_threads": gpu}


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
