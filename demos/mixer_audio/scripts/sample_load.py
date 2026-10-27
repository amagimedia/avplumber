#!/usr/bin/env python3
"""Sample host/GPU/mixer load for a fixed window and print one JSON report.

Written for the mixer_audio vs mixer (video-only) A/B on the AWS T4 demo host,
which has no sysstat (no mpstat/pidstat) -- everything here reads /proc,
cgroup v2 and nvidia-smi directly. Run it on the host:

    python3 -I sample_load.py --seconds 120 --label audio > audio.json

Per-thread CPU is included because avplumber names its worker threads after
graph nodes, so the audio chain (decode_audio_N / resample_audio_N /
audio_mixer) can be attributed without a profiler.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup/system.slice")
CLK_TCK = os.sysconf("SC_CLK_TCK")
GPU_FIELDS = ["utilization.gpu", "utilization.memory", "utilization.encoder",
              "utilization.decoder", "memory.used", "power.draw", "clocks.sm", "temperature.gpu"]


def read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def cpu_totals():
    """Per-field jiffies of /proc/stat's aggregate line, plus per-core busy totals."""
    fields = ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"]
    out = {"cores": {}}
    for line in read("/proc/stat").splitlines():
        parts = line.split()
        if parts and parts[0] == "cpu":
            out["total"] = dict(zip(fields, (int(v) for v in parts[1:9])))
        elif parts and re.fullmatch(r"cpu\d+", parts[0]):
            vals = dict(zip(fields, (int(v) for v in parts[1:9])))
            out["cores"][parts[0]] = vals
    return out


def meminfo():
    info = {}
    for line in read("/proc/meminfo").splitlines():
        key, _, rest = line.partition(":")
        info[key] = int(rest.split()[0])  # kB
    return info


def containers():
    """{service name: cgroup dir} for the running compose containers."""
    names = subprocess.run(["docker", "ps", "--format", "{{.ID}} {{.Names}}"],
                           capture_output=True, text=True, check=True).stdout.split("\n")
    found = {}
    for line in names:
        if not line.strip():
            continue
        cid, name = line.split()[0], line.split()[1]
        for d in CGROUP.glob(f"docker-{cid}*.scope"):
            found[name] = d
    return found


def cgroup_cpu(d):
    stat = {}
    for line in read(d / "cpu.stat").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            stat[parts[0]] = int(parts[1])
    return {"usage_usec": stat.get("usage_usec", 0), "user_usec": stat.get("user_usec", 0),
            "system_usec": stat.get("system_usec", 0),
            "memory_current": int(read(d / "memory.current").strip() or 0)}


def mixer_pid():
    """PID of the pyplumber.mixer process (not the webui supervisor)."""
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        cmd = read(p / "cmdline").replace("\0", " ")
        if "pyplumber.mixer" in cmd and "--config" in cmd:
            return int(p.name), cmd.strip()
    return None, ""


def proc_cpu(pid):
    parts = read(f"/proc/{pid}/stat").rsplit(") ", 1)
    if len(parts) != 2:
        return None
    f = parts[1].split()
    return {"utime": int(f[11]), "stime": int(f[12]), "threads": int(f[17]),
            "rss_kb": int(read(f"/proc/{pid}/statm").split()[1]) * 4 if read(f"/proc/{pid}/statm") else 0}


def thread_cpu(pid):
    """{tid: (comm, utime+stime jiffies)} for every thread of pid."""
    out = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        raw = read(f"/proc/{pid}/task/{tid}/stat")
        head, _, rest = raw.rpartition(") ")
        if not rest:
            continue
        comm = head.partition(" (")[2]
        f = rest.split()
        out[tid] = (comm, int(f[11]) + int(f[12]))
    return out


def http_json(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {"error": repr(exc)}


def gpu_sampler(interval):
    cmd = ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}",
           "--format=csv,noheader,nounits", f"--loop={interval}"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)


def summarize(values):
    if not values:
        return None
    s = sorted(values)
    return {"mean": round(sum(s) / len(s), 2), "min": s[0], "max": s[-1],
            "p50": s[len(s) // 2], "p95": s[min(len(s) - 1, int(len(s) * 0.95))], "n": len(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--state-url", default="http://127.0.0.1:7681/api/state")
    ap.add_argument("--outputs-url", default="http://127.0.0.1:7681/api/outputs")
    ap.add_argument("--top-threads", type=int, default=25)
    args = ap.parse_args()

    pid, cmdline = mixer_pid()
    if pid is None:
        print("no pyplumber.mixer process found", file=sys.stderr)
    conts = containers()

    gpu = gpu_sampler(int(max(1, args.interval)))
    t0 = time.monotonic()
    cpu0, mem0 = cpu_totals(), meminfo()
    cg0 = {name: cgroup_cpu(d) for name, d in conts.items()}
    p0 = proc_cpu(pid) if pid else None
    th0 = thread_cpu(pid) if pid else {}
    state0 = http_json(args.state_url)
    loadavg0 = read("/proc/loadavg").split()[:3]

    # 1 Hz samples of the cheap /proc counters, for variability within the window.
    load_series, run_series = [], []
    while time.monotonic() - t0 < args.seconds:
        time.sleep(args.interval)
        la = read("/proc/loadavg").split()
        if la:
            load_series.append(float(la[0]))
            run_series.append(int(la[3].split("/")[0]) if "/" in la[3] else 0)

    t1 = time.monotonic()
    elapsed = t1 - t0
    cpu1, mem1 = cpu_totals(), meminfo()
    cg1 = {name: cgroup_cpu(d) for name, d in conts.items()}
    p1 = proc_cpu(pid) if pid else None
    th1 = thread_cpu(pid) if pid else {}
    state1 = http_json(args.state_url)
    loadavg1 = read("/proc/loadavg").split()[:3]
    gpu.terminate()
    gpu_out, _ = gpu.communicate(timeout=15)

    # --- CPU ---
    d = {k: cpu1["total"][k] - cpu0["total"][k] for k in cpu0["total"]}
    total = sum(d.values()) or 1
    ncpu = len(cpu0["cores"])
    cpu_pct = {k: round(100.0 * v / total, 2) for k, v in d.items()}
    cpu_pct["busy"] = round(100.0 - cpu_pct["idle"], 2)
    core_busy = []
    for core, v0 in cpu0["cores"].items():
        v1 = cpu1["cores"][core]
        dd = {k: v1[k] - v0[k] for k in v0}
        tot = sum(dd.values()) or 1
        core_busy.append(round(100.0 * (tot - dd["idle"]) / tot, 1))

    # --- GPU ---
    gpu_rows = []
    for line in gpu_out.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(GPU_FIELDS):
            continue
        try:
            gpu_rows.append([float(c) for c in cells])
        except ValueError:
            continue
    gpu_stats = {f: summarize([r[i] for r in gpu_rows]) for i, f in enumerate(GPU_FIELDS)}

    # --- threads ---
    threads = []
    for tid, (comm, j1) in th1.items():
        j0 = th0.get(tid, (comm, 0))[1] if tid in th0 else 0
        cpu = 100.0 * (j1 - j0) / CLK_TCK / elapsed
        if cpu > 0.05:
            threads.append({"tid": tid, "comm": comm, "cpu_pct": round(cpu, 2)})
    threads.sort(key=lambda t: -t["cpu_pct"])
    groups = {}
    for t in threads:
        # Collapse node-indexed thread names (resample_audio_17 -> resample_audio_N).
        key = re.sub(r"[_-]?\d+$", "_N", t["comm"])
        g = groups.setdefault(key, {"cpu_pct": 0.0, "threads": 0})
        g["cpu_pct"] += t["cpu_pct"]
        g["threads"] += 1
    for g in groups.values():
        g["cpu_pct"] = round(g["cpu_pct"], 2)

    report = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_sec": round(elapsed, 1),
        "ncpu": ncpu,
        "mixer_pid": pid,
        "mixer_cmdline": cmdline,
        "cpu_pct_of_host": cpu_pct,
        "cpu_cores_busy_pct": {"per_core": core_busy, "mean": round(sum(core_busy) / len(core_busy), 1),
                               "min": min(core_busy), "max": max(core_busy)},
        "loadavg": {"start": loadavg0, "end": loadavg1,
                    "sampled_1min": summarize(load_series), "runnable": summarize(run_series)},
        "memory_kb": {"start": {k: mem0[k] for k in ("MemTotal", "MemAvailable", "MemFree", "Cached")},
                      "end": {k: mem1[k] for k in ("MemTotal", "MemAvailable", "MemFree", "Cached")}},
        "gpu": gpu_stats,
        "containers": {},
        "mixer_process": None,
        "threads_top": threads[: args.top_threads],
        "thread_groups": dict(sorted(groups.items(), key=lambda kv: -kv[1]["cpu_pct"])),
        "state_before": state0,
        "state_after": state1,
    }
    for name in cg1:
        if name not in cg0:
            continue
        usec = cg1[name]["usage_usec"] - cg0[name]["usage_usec"]
        report["containers"][name] = {
            "cpu_pct_of_host": round(100.0 * usec / 1e6 / elapsed / ncpu, 2),
            "cpu_cores": round(usec / 1e6 / elapsed, 3),
            "user_cores": round((cg1[name]["user_usec"] - cg0[name]["user_usec"]) / 1e6 / elapsed, 3),
            "system_cores": round((cg1[name]["system_usec"] - cg0[name]["system_usec"]) / 1e6 / elapsed, 3),
            "memory_mb": round(cg1[name]["memory_current"] / 1e6, 1),
        }
    if p0 and p1:
        jiffies = (p1["utime"] + p1["stime"]) - (p0["utime"] + p0["stime"])
        report["mixer_process"] = {
            "cpu_cores": round(jiffies / CLK_TCK / elapsed, 3),
            "cpu_pct_of_host": round(100.0 * jiffies / CLK_TCK / elapsed / ncpu, 2),
            "user_cores": round((p1["utime"] - p0["utime"]) / CLK_TCK / elapsed, 3),
            "system_cores": round((p1["stime"] - p0["stime"]) / CLK_TCK / elapsed, 3),
            "threads": p1["threads"], "rss_mb": round(p1["rss_kb"] / 1024, 1),
        }
    report["outputs"] = http_json(args.outputs_url)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
