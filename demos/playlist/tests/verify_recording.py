#!/usr/bin/env python3
"""Check a program recording for frame-perfect playlist transitions.

Decodes the recording with ffmpeg, reads the burned-in frame code of every
output frame and reports, per element boundary, the last frame of the outgoing
element and the first frame of the incoming one.  With ``--playlist`` the
expected cue-in/cue-out frames are checked too.

    python3 tests/verify_recording.py program.mp4 --playlist playlist.json --fps 30

Exit status 0 means: no unreadable frames outside declared transitions, no
repeated frames at speed 1, no gaps inside elements, and every boundary lands
on the expected frames.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_codes import read_code  # noqa: E402

WIDTH, HEIGHT = 1920, 1080


def decode_codes(path: str) -> List[Optional[Tuple[int, int]]]:
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{WIDTH}x{HEIGHT}", "-"], stdout=subprocess.PIPE)
    codes = []
    size = WIDTH * HEIGHT
    while True:
        data = proc.stdout.read(size)
        if len(data) < size:
            break
        codes.append(read_code(np.frombuffer(data, np.uint8).reshape(HEIGHT, WIDTH)))
    proc.wait()
    return codes


@dataclass
class Segment:
    clip: int
    first: int
    last: int
    frames: int
    repeats: int = 0
    gaps: int = 0
    unreadable_before: int = 0


@dataclass
class Report:
    segments: List[Segment] = field(default_factory=list)
    boundaries: List[dict] = field(default_factory=list)
    unreadable: int = 0
    problems: List[str] = field(default_factory=list)
    anomalies: List[dict] = field(default_factory=list)   # where repeats, gaps and unreadable runs sit


def analyze(codes: List[Optional[Tuple[int, int]]], speed_of=None) -> Report:
    report = Report()
    segment: Optional[Segment] = None
    pending_unreadable = 0
    previous: Optional[Tuple[int, int]] = None
    for index, code in enumerate(codes):
        if code is None:
            report.unreadable += 1
            if pending_unreadable == 0:
                report.anomalies.append({"at": index, "kind": "unreadable", "after": previous})
            pending_unreadable += 1
            continue
        clip, frame = code
        if segment is None or clip != segment.clip:
            if segment is not None:
                report.boundaries.append({"at_output_frame": index, "from": [segment.clip, segment.last],
                                          "to": [clip, frame], "unreadable_between": pending_unreadable})
            segment = Segment(clip, frame, frame, 1, unreadable_before=pending_unreadable)
            report.segments.append(segment)
        else:
            step = frame - previous[1]
            speed = 1.0 if speed_of is None else speed_of(clip)
            if step == 0 and speed >= 1.0:
                segment.repeats += 1
                report.anomalies.append({"at": index, "kind": "repeat", "frame": frame})
            elif step > max(1, round(speed)) + pending_unreadable * max(1, round(speed)) or step < 0:
                segment.gaps += 1
                report.anomalies.append({"at": index, "kind": "gap", "from": previous[1], "to": frame})
            segment.last, segment.frames = frame, segment.frames + 1
        pending_unreadable = 0
        previous = code
    return report


def check_expectations(report: Report, playlist: Optional[list], fps: int, transition_frames: int) -> None:
    for seg in report.segments:
        if seg.repeats:
            report.problems.append(f"clip {seg.clip}: {seg.repeats} repeated frame(s)")
        if seg.gaps:
            report.problems.append(f"clip {seg.clip}: {seg.gaps} gap(s)")
    for b in report.boundaries:
        if b["unreadable_between"] > transition_frames:
            report.problems.append(f"boundary {b['from']}->{b['to']}: {b['unreadable_between']} unreadable frames")
    if not playlist:
        return
    by_clip = {int(Path(c["url"]).name.split("-")[0]): c for c in playlist}
    for b in report.boundaries:
        incoming = by_clip.get(b["to"][0])
        outgoing = by_clip.get(b["from"][0])
        if incoming:
            expected_first = round(incoming.get("cue_in", 0) * fps / 1000)
            if b["to"][1] != expected_first:
                report.problems.append(
                    f"clip {b['to'][0]} started at frame {b['to'][1]}, expected cue-in frame {expected_first}")
        if outgoing and outgoing.get("cue_out") is not None and outgoing.get("end", "PlayToEnd") == "PlayToEnd":
            expected_last = round(outgoing["cue_out"] * fps / 1000) - 1
            if b["from"][1] != expected_last:
                report.problems.append(
                    f"clip {b['from'][0]} ended at frame {b['from'][1]}, expected cue-out frame {expected_last}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording")
    parser.add_argument("--playlist", type=Path, help="JSON element list used for the run")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--transition-frames", type=int, default=0,
                        help="unreadable frames allowed at a boundary (fade/wipe duration)")
    parser.add_argument("--json", type=Path, help="write the report here")
    args = parser.parse_args(argv)

    playlist = json.loads(args.playlist.read_text()) if args.playlist else None
    speeds = None
    if playlist:
        speed_map = {int(Path(c["url"]).name.split("-")[0]): float(c.get("speed", 1.0)) for c in playlist}
        speeds = lambda clip: speed_map.get(clip, 1.0)  # noqa: E731
    codes = decode_codes(args.recording)
    report = analyze(codes, speeds)
    check_expectations(report, playlist, args.fps, args.transition_frames)
    summary = {"frames": len(codes), "unreadable": report.unreadable,
               "segments": [vars(s) for s in report.segments], "boundaries": report.boundaries,
               "anomalies": report.anomalies, "problems": report.problems}
    text = json.dumps(summary, indent=2)
    if args.json:
        args.json.write_text(text)
    print(text)
    print("PASS" if not report.problems else f"FAIL ({len(report.problems)} problem(s))")
    return 0 if not report.problems else 1


if __name__ == "__main__":
    sys.exit(main())
