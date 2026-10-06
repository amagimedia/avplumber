"""Record comparable mixer, GPU and CPU measurements on the NVIDIA host.

CPU percentages use 100% for one logical CPU; host_cpu_percent uses 100% for
the whole machine. Run without compilation or unrelated GPU jobs during an A/B.
The JSON includes counter deltas, not cumulative misses since mixer startup.
"""

import argparse
import asyncio
import csv
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
import urllib.request


GPU_FIELDS = (
    "memory.used", "memory.total", "utilization.gpu", "utilization.memory",
    "utilization.decoder", "utilization.encoder", "clocks.current.sm", "power.draw",
)
COUNTERS = ("frames", "missed_deadlines", "repeats", "discarded", "overflow")


def read_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def source_progress(port):
    if port is None:
        return None
    from pyplumber.mixer.control import AvpConnection

    async def read():
        connection = AvpConnection("127.0.0.1", port)
        try:
            await connection.connect()
            queues = json.loads(await connection.command("queues.json", timeout=30))
            return {"at": time.monotonic(), "frames": {
                q["name"]: q["enqueued_total"] for q in queues
                if re.fullmatch(r"input_\d+_(decoded|fps)", q["name"])
                or q["name"].endswith("_encoded")}}
        finally:
            await connection.disconnect()

    return asyncio.run(read())


def cpu_snapshot(pids):
    threads = {}
    for pid in pids:
        for path in Path(f"/proc/{pid}/task").glob("*/stat"):
            try:
                data = path.read_text()
            except FileNotFoundError:
                continue
            end = data.rfind(")")
            fields = data[end + 2:].split()
            # Linux stat fields 14/15 are utime/stime, field 22 guards TID reuse.
            key = f"{pid}:{path.parent.name}:{fields[19]}"
            threads[key] = {"name": data[data.find("(") + 1:end],
                            "ticks": int(fields[11]) + int(fields[12])}
    host = Path("/proc/stat").read_text().splitlines()
    ticks = list(map(int, host[0].split()[1:9]))
    return {"at": time.monotonic(), "threads": threads,
            "host_total": sum(ticks), "host_idle": ticks[3] + ticks[4]}


def cpu_delta(before, after):
    elapsed = after["at"] - before["at"]
    scale = 100 / os.sysconf("SC_CLK_TCK") / elapsed
    per_thread = []
    for key, row in after["threads"].items():
        if key in before["threads"]:
            per_thread.append({"key": key, "name": row["name"],
                               "cpu_percent": (row["ticks"] - before["threads"][key]["ticks"]) * scale})
    per_thread.sort(key=lambda row: row["cpu_percent"], reverse=True)
    total = after["host_total"] - before["host_total"]
    idle = after["host_idle"] - before["host_idle"]
    return {"cpu_percent": sum(row["cpu_percent"] for row in per_thread),
            "host_cpu_percent": 100 * (total - idle) / total if total else 0,
            "threads": len(after["threads"]),
            "active_threads": sum(row["cpu_percent"] > 0 for row in per_thread),
            "created_threads": len(after["threads"].keys() - before["threads"].keys()),
            "exited_threads": len(before["threads"].keys() - after["threads"].keys()),
            "per_thread": per_thread}


def gpu_snapshot():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS),
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    return [dict(zip(GPU_FIELDS, [float(value.strip()) if value.strip() != "[N/A]"
                                 else None for value in row]))
            for row in csv.reader(io.StringIO(output))]


def mixer_counters(state):
    status = state["status"]
    counters = {"program_" + slot: values for slot, values in status["playout"].items()}
    for bus in state["aux_buses"]:
        counters["aux_" + bus["id"]] = {
            **bus["playout"], "output_drops": bus.get("output_drops", 0),
        }
    return counters


def counter_delta(before, after):
    result = {}
    for name, values in after.items():
        if name not in before:
            raise RuntimeError(f"counter {name} appeared during measurement")
        delta = {key: values.get(key, 0) - before[name].get(key, 0)
                 for key in (*COUNTERS, "output_drops")}
        if any(value < 0 for value in delta.values()):
            raise RuntimeError(f"counter reset during measurement: {name}")
        result[name] = delta
    if before.keys() != after.keys():
        raise RuntimeError("mixer output set changed during measurement")
    return result


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {}
    return {"min": values[0], "mean": statistics.mean(values),
            "p95": values[min(len(values) - 1, int(len(values) * .95))], "max": values[-1]}


def capture(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pids = args.pid
    if args.container:
        rows = subprocess.check_output(
            ["docker", "top", args.container, "-eo", "pid"], text=True).splitlines()[1:]
        pids += [int(row.strip()) for row in rows]
    if not pids:
        raise ValueError("provide --pid or --container for process/thread measurements")
    setup = read_json(args.url + "/api/setup")
    state_before = read_json(args.url + "/api/state")
    sources_before = source_progress(args.control_port)
    first = previous = cpu_snapshot(pids)
    samples = []
    deadline = first["at"] + args.duration
    while True:
        target = min(deadline, first["at"] + (len(samples) + 1) * args.interval)
        time.sleep(max(0, target - time.monotonic()))
        current = cpu_snapshot(pids)
        state = read_json(args.url + "/api/state")
        if state.get("setup_revision") != state_before.get("setup_revision"):
            raise RuntimeError("setup changed during measurement")
        samples.append({"elapsed": current["at"] - first["at"],
                        "cpu": cpu_delta(previous, current), "gpu": gpu_snapshot(),
                        "counters": mixer_counters(state), "host": state.get("host"),
                        "pgm_scene": state["status"].get("pgm_scene"),
                        "cut_latency": state["status"].get("cut_latency")})
        previous = current
        if current["at"] >= deadline:
            break
    summary = {"elapsed": current["at"] - first["at"],
               "cpu": cpu_delta(first, current),
               "cpu_percent": distribution(sample["cpu"]["cpu_percent"] for sample in samples),
               "host_cpu_percent": distribution(sample["cpu"]["host_cpu_percent"] for sample in samples),
               "gpu": {field: distribution(sample["gpu"][0][field] for sample in samples)
                       for field in GPU_FIELDS},
               "counter_deltas": counter_delta(mixer_counters(state_before), mixer_counters(state))}
    sources_after = source_progress(args.control_port)
    if sources_before is not None:
        before, after = sources_before["frames"], sources_after["frames"]
        if not before or before.keys() != after.keys():
            raise RuntimeError("source edge set is empty or changed during measurement")
        deltas = {name: count - before[name] for name, count in after.items()}
        if any(count < 0 for count in deltas.values()):
            raise RuntimeError("source frame counter reset during measurement")
        summary["source_progress"] = {
            "elapsed": sources_after["at"] - sources_before["at"],
            "frames": deltas, "stalled": [name for name, count in deltas.items() if not count]}
    report = {"label": args.label, "logical_cpus": os.cpu_count(), "pids": pids,
              "setup": setup, "state_before": state_before, "state_after": state,
              "cpu_before": first, "cpu_after": current,
              "sources_before": sources_before, "sources_after": sources_after,
              "samples": samples, "summary": summary}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({**summary, "cpu": {key: value for key, value in summary["cpu"].items()
                                       if key != "per_thread"}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:17681")
    parser.add_argument("--container")
    parser.add_argument("--control-port", type=int, help="also measure source and encoded-output progress; requires pyplumber on PYTHONPATH")
    parser.add_argument("--pid", action="append", type=int, default=[])
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0:
        parser.error("duration and interval must be positive")
    capture(args)


if __name__ == "__main__":
    main()
