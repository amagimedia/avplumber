#!/usr/bin/env python3
import argparse
import json
import socket
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


QueueSampleMap = Dict[str, Dict[str, float]]
PairDef = Tuple[str, str, str]


STAGE_PAIRS: List[PairDef] = [
    ("Preprocess", "v_dec_for_yolo", "v_pre_yolo"),
    ("Inference total path", "v_pre_yolo", "v_inferred"),
    ("Shot classifier", "v_inferred", "v_classified"),
    ("Player tracker", "v_classified", "v_tracked_players"),
    ("Ball tracker", "v_tracked_players", "v_ball_tracked"),
    ("Ball handler", "v_ball_tracked", "v_tracked"),
    ("Drawing debug overlays", "v_1080p_with_md", "v_bbox_handler"),
    ("Smooth crop / reframer", "v_bbox_handler", "v_viewport_md"),
    ("Viewport draw", "v_viewport_md", "v_annotated_cuda"),
]

MODEL_PAIRS: List[PairDef] = [
    ("Players model", "v_pre_yolo", "v_post_players"),
    ("Ball model", "v_pre_yolo", "v_post_ball"),
    ("Pose model", "v_pre_yolo", "v_post_pose"),
    ("Court polygon", "v_post_pose", "v_post_court"),
    ("Pose + court path", "v_pre_yolo", "v_post_court"),
]

TOTAL_PAIRS: List[PairDef] = [
    ("Before inference total", "v_dec_for_yolo", "v_inferred"),
    ("After inference total", "v_inferred", "v_annotated_cuda"),
]


class QueueClient:
    def fetch_queues(self) -> List[dict]:
        raise NotImplementedError


class TcpQueueClient(QueueClient):
    def __init__(self, host: str, port: int, timeout: float):
        self.host = host
        self.port = port
        self.timeout = timeout

    def fetch_queues(self) -> List[dict]:
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.sendall(b"queues.json\nbye\n\n")
            data = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data += chunk
        return _parse_control_response(data.decode("utf-8", "replace"))


class SshQueueClient(QueueClient):
    def __init__(self, ssh_target: str, port: int, timeout: float, identity: Optional[str], ssh_extra: Sequence[str]):
        self.ssh_target = ssh_target
        self.port = port
        self.timeout = timeout
        self.identity = identity
        self.ssh_extra = list(ssh_extra)

    def fetch_queues(self) -> List[dict]:
        cmd = ["ssh"]
        if self.identity:
            cmd.extend(["-i", self.identity])
        cmd.extend(self.ssh_extra)
        remote_cmd = (
            "python3 - <<'PY'\n"
            "import socket\n"
            f"s = socket.create_connection(('localhost', {self.port}), timeout={self.timeout})\n"
            "s.sendall(b'queues.json\\nbye\\n\\n')\n"
            "data = b''\n"
            "while True:\n"
            "    chunk = s.recv(65536)\n"
            "    if not chunk:\n"
            "        break\n"
            "    data += chunk\n"
            "s.close()\n"
            "print(data.decode('utf-8', 'replace'), end='')\n"
            "PY"
        )
        cmd.extend([self.ssh_target, remote_cmd])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ssh command failed")
        return _parse_control_response(proc.stdout)


def _parse_control_response(text: str) -> List[dict]:
    lines = [
        line for line in text.splitlines()
        if line and not line.startswith(("100 ", "201 ", "BYE"))
    ]
    payload = "".join(lines).strip()
    if not payload:
        raise RuntimeError("empty queues.json response")
    parsed = json.loads(payload)
    if not isinstance(parsed, list):
        raise RuntimeError("queues.json did not return an array")
    return parsed


def pair_queue_names(*groups: Iterable[PairDef]) -> List[str]:
    names = set()
    for group in groups:
        for _, src, dst in group:
            names.add(src)
            names.add(dst)
    return sorted(names)


def collect_samples(client: QueueClient, names: Sequence[str], duration_s: float, poll_s: float) -> QueueSampleMap:
    samples: QueueSampleMap = {name: {} for name in names}
    deadline = time.time() + duration_s
    while time.time() < deadline:
        started = time.time()
        snapshot = client.fetch_queues()
        by_name = {item["name"]: item for item in snapshot}
        for name in names:
            item = by_name.get(name)
            if not item:
                continue
            last_ts = item.get("last_ts_seconds")
            if last_ts is None:
                continue
            pts_key = f"{float(last_ts):.3f}"
            samples[name].setdefault(pts_key, started)
        elapsed = time.time() - started
        remaining = poll_s - elapsed
        if remaining > 0:
            time.sleep(remaining)
    return samples


def summarize_pairs(samples: QueueSampleMap, pairs: Sequence[PairDef]) -> List[dict]:
    results = []
    for label, src, dst in pairs:
        src_map = samples[src]
        dst_map = samples[dst]
        common_pts = sorted(set(src_map).intersection(dst_map))
        deltas_ms = []
        for pts in common_pts:
            delta_ms = (dst_map[pts] - src_map[pts]) * 1000.0
            if delta_ms >= -1.0:
                deltas_ms.append(max(0.0, delta_ms))
        result = {
            "label": label,
            "count": len(deltas_ms),
            "avg_ms": (sum(deltas_ms) / len(deltas_ms)) if deltas_ms else None,
            "min_ms": min(deltas_ms) if deltas_ms else None,
            "max_ms": max(deltas_ms) if deltas_ms else None,
        }
        results.append(result)
    return results


def format_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if value < 1.0:
        return "<1.000 ms"
    return f"{value:.3f} ms"


def print_markdown_table(title: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    print(title)
    print("| Stage | Avg Latency |")
    print("|---|---:|")
    for row in rows:
        print(f"| {row['label']} | {format_ms(row['avg_ms'])} |")
    print()


def print_json_report(stage_rows: Sequence[dict], total_rows: Sequence[dict], model_rows: Sequence[dict]) -> None:
    print(json.dumps({
        "stages": stage_rows,
        "totals": total_rows,
        "models": model_rows,
    }, indent=2))


def build_client(args: argparse.Namespace) -> QueueClient:
    if args.ssh:
        return SshQueueClient(
            ssh_target=args.ssh,
            port=args.port,
            timeout=args.timeout,
            identity=args.identity,
            ssh_extra=args.ssh_extra or [],
        )
    return TcpQueueClient(host=args.host, port=args.port, timeout=args.timeout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure queue-derived latency across avplumber graph stages by matching identical frame PTS values."
    )
    parser.add_argument("--host", default="localhost", help="Control API host for direct TCP probing.")
    parser.add_argument("--port", type=int, default=20200, help="Control API port.")
    parser.add_argument("--duration", type=float, default=3.0, help="Sampling duration in seconds.")
    parser.add_argument("--poll", type=float, default=0.005, help="Polling interval in seconds.")
    parser.add_argument("--timeout", type=float, default=1.0, help="Socket timeout in seconds.")
    parser.add_argument("--ssh", help="Optional SSH target such as user@host.")
    parser.add_argument("--identity", help="Optional SSH identity file.")
    parser.add_argument("--ssh-extra", action="append", default=[], help="Additional raw ssh argument. Repeat as needed.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = build_client(args)
    queue_names = pair_queue_names(STAGE_PAIRS, MODEL_PAIRS, TOTAL_PAIRS)
    samples = collect_samples(client, queue_names, args.duration, args.poll)
    stage_rows = summarize_pairs(samples, STAGE_PAIRS)
    total_rows = summarize_pairs(samples, TOTAL_PAIRS)
    model_rows = summarize_pairs(samples, MODEL_PAIRS)

    if args.format == "json":
        print_json_report(stage_rows, total_rows, model_rows)
        return 0

    print_markdown_table("Stage latency:", stage_rows)
    print_markdown_table("Totals:", total_rows)
    print_markdown_table("Model latency:", model_rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
