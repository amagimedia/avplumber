#!/usr/bin/env python3
"""
ANSI-art console waveform display for envelope files produced by the
write_audio_envelope node.

Usage:
  python3 waveform_console_demo.py <path_to_envelope_dir> [options]

Example:
  python3 waveform_console_demo.py /tmp/envelope
  python3 waveform_console_demo.py /tmp/envelope --level "1/25" --metric rms --width 60 --height 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Envelope format: 1 byte per value, 0.5 dB resolution, -127..0 dB
# byte = 0   -> -127 dB (or below), linear ~ 0
# byte = 254 -> 0 dB,    linear = 1.0
# Formula: dB = -127 + byte * 0.5  =>  linear = 10^(dB/20)
def envelope_byte_to_linear(b: int) -> float:
    if b <= 0:
        return 0.0
    if b >= 254:
        return 1.0
    db = -127.0 + b * 0.5
    return 10.0 ** (db / 20.0)


def load_index(dirpath: Path) -> dict:
    index_path = dirpath / "index.json"
    if not index_path.is_file():
        sys.exit(f"Missing {index_path}")
    with open(index_path) as f:
        return json.load(f)


def find_metric(index: dict, metric_id: str) -> dict:
    metrics = index.get("metrics", {})
    if metric_id not in metrics:
        sys.exit(f"Metric '{metric_id}' not found in index.json. Available: {list(metrics.keys())}")
    return metrics[metric_id]


def read_envelope_metric(
    filepath: Path,
    channels: int,
    offset_bytes: int,
    stride_bytes: int,
    channel: int = 0,
    max_samples: Optional[int] = None,
) -> List[float]:
    """Read one metric for one channel (or max across channels if channel < 0)."""
    bytes_per_sample = 3 * channels  # pos_peak, neg_peak, rms per channel
    with open(filepath, "rb") as f:
        data = f.read()
    n = len(data) // bytes_per_sample
    if max_samples is not None and n > max_samples:
        n = max_samples
    out = []
    for i in range(n):
        base = i * bytes_per_sample
        if channel >= 0:
            b = data[base + offset_bytes + channel * stride_bytes]
            out.append(envelope_byte_to_linear(b))
        else:
            # max across channels
            vals = [
                data[base + offset_bytes + ch * stride_bytes]
                for ch in range(channels)
            ]
            out.append(envelope_byte_to_linear(max(vals)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ANSI-art console waveform from write_audio_envelope output",
    )
    ap.add_argument("dir", type=Path, help="Directory containing index.json and level_*.bin")
    ap.add_argument("--level", default="", help="Level key (duration_sec, e.g. 1/25). Default: first level in index.")
    ap.add_argument("--metric", default="rms", choices=["positive_peak", "negative_peak", "rms"], help="Metric to display (default rms)")
    ap.add_argument("--channel", type=int, default=-1, help="Channel index (default -1 = max across channels)")
    ap.add_argument("--width", type=int, default=72, help="Display width in chars (default 72)")
    ap.add_argument("--height", type=int, default=12, help="Display height in rows (default 12)")
    ap.add_argument("--no-ansi", action="store_true", help="Disable ANSI color")
    args = ap.parse_args()

    dirpath = args.dir.resolve()
    if not dirpath.is_dir():
        sys.exit(f"Not a directory: {dirpath}")

    index = load_index(dirpath)
    channels = index.get("channels", 1)
    levels = index.get("levels", {})
    if not levels:
        sys.exit("No levels in index.json")
    level_key = args.level if args.level else next(iter(levels))
    if level_key not in levels:
        sys.exit(f"Level '{level_key}' not found. Available: {list(levels.keys())}")
    level_info = levels[level_key]
    bin_name = level_info.get("file", "level.bin") if isinstance(level_info, dict) else level_info
    bin_path = dirpath / bin_name
    if not bin_path.is_file():
        sys.exit(f"Missing {bin_path}")

    metric = find_metric(index, args.metric)
    offset = metric.get("offset_bytes", 0)
    stride = metric.get("stride_bytes", 3)

    # Read envelope; limit samples to width so we don't downsample too much
    samples = read_envelope_metric(
        bin_path, channels, offset, stride,
        channel=args.channel if 0 <= args.channel < channels else -1,
        max_samples=args.width * 4,
    )
    if not samples:
        print("No envelope data.")
        return

    # Downsample to width (take max in each bucket to preserve peaks)
    w, h = args.width, args.height
    step = len(samples) / w
    display = []
    for c in range(w):
        i0 = int(c * step)
        i1 = int((c + 1) * step)
        if i1 <= i0:
            i1 = i0 + 1
        display.append(max(samples[i0:i1]))

    # Build rows: row 0 = top (loud), row h-1 = bottom (quiet)
    grid = [[" "] * w for _ in range(h)]
    for c in range(w):
        v = display[c]
        # map [0,1] to row index: 0 -> bottom row, 1 -> top row
        r = int((1.0 - v) * (h - 1) + 0.5)
        r = max(0, min(h - 1, r))
        grid[r][c] = "#"

    # Fill below the waveform so it looks like a solid bar
    for c in range(w):
        v = display[c]
        r = int((1.0 - v) * (h - 1) + 0.5)
        r = max(0, min(h - 1, r))
        for rr in range(r + 1, h):
            grid[rr][c] = ":"  # faint fill below

    green = "" if args.no_ansi else "\033[32m"
    dim = "" if args.no_ansi else "\033[2m"
    reset = "" if args.no_ansi else "\033[0m"

    print(f"{dim}Envelope: {dirpath} level={level_key!r} metric={args.metric} ({len(samples)} samples -> {w} cols){reset}")
    print()
    for row in grid:
        print(f"{green}{''.join(row)}{reset}")
    print(f"{dim}+{'-' * (w - 2)}+{reset}")


if __name__ == "__main__":
    main()
