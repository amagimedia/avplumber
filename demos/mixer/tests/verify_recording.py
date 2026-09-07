"""Decode a numbered mixer recording once and check every visible source.

Run against recordings of frame_codes.py fixtures. Every frame must match the
expected scene; only explicitly marked transition intervals permit unreadable
blends. A returning source must have advanced while hidden.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layouts import CANVAS_WIDTH, CANVAS_HEIGHT, cell_size, fitted_even_size, GRID_SHAPES
from frame_codes import read_code


def read_sources(image):
    candidates = []
    for capacity in (1, 2, 4, 8, 16):
        width, height = cell_size(capacity)
        columns = GRID_SHAPES.get(capacity, (1, 1))[0]
        fitted_w, fitted_h = fitted_even_size(1920, 1080, width, height)
        for slot in range(capacity):
            x = slot % columns * width + (width - fitted_w) // 2
            y = slot // columns * height + (height - fitted_h) // 2
            code = read_code(image[y:y+fitted_h, x:x+fitted_w])
            if code is not None:
                candidates.append({"capacity": capacity, "slot": slot,
                                   "source": code[0], "id": code[1]})
    return candidates


def continuity(rows, loop_frames):
    sources = sorted({code["source"] for row in rows for code in row["codes"]})
    results = []
    for source in sources:
        samples = [(row["frame"], code["id"]) for row in rows
                   for code in row["codes"] if code["source"] == source]
        offsets = Counter((identity - index) % loop_frames for index, identity in samples)
        expected = offsets.most_common(1)[0][0]
        anomalies = [{"frame": index, "id": identity} for index, identity in samples
                     if (identity - index) % loop_frames != expected]
        results.append({"source": source, "readable_frames": len(samples),
                        "offset": expected, "anomalies": anomalies})
    return results


def visibility_errors(rows, schedule):
    """Validate a complete scene schedule; frame ranges are [start, end)."""
    cursor = 0
    errors = []
    for interval in schedule:
        start, end = interval["start"], interval["end"]
        if start != cursor or end <= start or end > len(rows):
            raise ValueError("scene schedule must cover every frame once, in order")
        cursor = end
        if interval.get("transition", False):
            continue
        expected = set(interval["sources"])
        if not expected:
            raise ValueError("scene interval must specify expected sources")
        for row in rows[start:end]:
            seen = {code["source"] for code in row["codes"]}
            if missing := sorted(expected - seen):
                errors.append({"frame": row["frame"], "missing_sources": missing})
    if cursor != len(rows):
        raise ValueError("scene schedule must cover every frame once, in order")
    return errors


def packet_timing(video, start, seconds, fps):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts:stream=time_base", "-of", "json", video,
    ]))
    timebase = Fraction(data["streams"][0]["time_base"])
    all_pts = [int(packet["pts"]) for packet in data["packets"] if "pts" in packet]
    begin = all_pts[0] * timebase + Fraction(str(start))
    end = begin + Fraction(str(seconds))
    selected = [pts for pts in all_pts if begin <= pts * timebase < end]
    errors = []
    for index, pts in enumerate(selected):
        expected = selected[0] + Fraction(index, fps) / timebase
        if abs(pts - expected) > 1:
            errors.append({"packet": index, "pts": pts})
    return {"packets": len(selected), "unexpected_pts": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--loop-frames", type=int, default=7200)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--schedule", type=Path,
                        help="JSON scene intervals; default requires all sources in every frame")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    process = subprocess.Popen([
        "ffmpeg", "-v", "error", "-ss", str(args.start), "-i", args.video,
        "-t", str(args.seconds), "-an", "-pix_fmt", "gray", "-fps_mode", "passthrough",
        "-f", "rawvideo", "pipe:1",
    ], stdout=subprocess.PIPE)
    rows = []
    while data := process.stdout.read(CANVAS_WIDTH * CANVAS_HEIGHT):
        if len(data) != CANVAS_WIDTH * CANVAS_HEIGHT:
            raise RuntimeError("partial decoded frame")
        image = np.frombuffer(data, np.uint8).reshape(CANVAS_HEIGHT, CANVAS_WIDTH)
        rows.append({"frame": len(rows), "codes": read_sources(image)})
    if process.wait():
        raise RuntimeError("recording decode failed")
    timing = packet_timing(args.video, args.start, args.seconds, args.fps)
    schedule = (json.loads(args.schedule.read_text()) if args.schedule else
                [{"start": 0, "end": len(rows), "sources": list(range(args.sources))}])
    visibility = visibility_errors(rows, schedule) if rows else []
    result = {"frames": len(rows), "sources": continuity(rows, args.loop_frames),
              "visibility_errors": visibility, "schedule": schedule,
              "timing": timing, "rows": rows}
    args.output.write_text(json.dumps(result, indent=2))
    seen = {source["source"] for source in result["sources"]}
    missing = sorted(set(range(args.sources)) - seen)
    print(json.dumps({"frames": len(rows), "missing_sources": missing,
                      "visibility_errors": len(visibility),
                      "timing": {"packets": timing["packets"],
                                 "unexpected_pts": len(timing["unexpected_pts"])}, "sources": [
        {**source, "anomalies": len(source["anomalies"])} for source in result["sources"]
    ]}, indent=2))
    if (not rows or missing or visibility or not result["sources"] or not timing["packets"] or timing["unexpected_pts"]
            or any(source["anomalies"] for source in result["sources"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
