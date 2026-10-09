"""Reusable NVIDIA/NVENC and host telemetry; independent of HTTP and input policy."""
from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path
import selectors
import subprocess
import threading
import time
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


class RecentCounts:
    """How far counters that only grow rose in the last `window_s` seconds. The page shows this
    instead of the totals since the mixer started, which keep the misses of a start or of a
    disturbance long past on screen for good. `read` returns the totals as a dict, or None while
    there are none. One thread samples every `period_s`, so the window is full whether or not a
    page is open. `values` is None until the first sample; `covered_s` is how much of the window
    the samples span so far. Totals that fall belong to a mixer that started again and counts from
    zero: everything it counted happened inside the window."""

    def __init__(self, read, window_s: float = 600, period_s: float = 5, clock=time.monotonic):
        self.read, self.window_s, self.period_s, self.clock = read, window_s, period_s, clock
        self.values = None
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
        now = self.clock()
        if self._samples and any(totals[key] < count for key, count in self._samples[-1][1].items()):
            self._samples = [(now, dict.fromkeys(totals, 0))]
        self._samples.append((now, dict(totals)))
        while self._samples[0][0] < now - self.window_s:
            self._samples.pop(0)
        since, base = self._samples[0]
        self.values = {"window_s": self.window_s, "covered_s": round(now - since),
                       **{key: count - base[key] for key, count in totals.items()}}


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
        if same_process and process is not None and last_process is not None and elapsed > 0:
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
