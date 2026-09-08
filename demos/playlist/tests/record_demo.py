#!/usr/bin/env python3
"""Scripted operator actions for the demo recording, with chapter marks.

Run while ``server.py --record program.mp4`` and ``capture_tui.cjs`` are
running.  Sends the same control verbs the TUI sends, at fixed times, and
writes ``chapters.json`` (seconds since the script started) for the docs page.

    python3 tests/record_demo.py --port 7778 --out chapters.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from control import RemoteClient  # noqa: E402

# 15-second movie: manual take, two scheduled cuts, a fade. Element 3 runs 2 s..8 s,
# element 4 is Timed 4 s at 2x, so the cuts land at ~7 s and ~11 s after the take.
SHORT = [
    (0.0, "Take: manual cut", "transition", {"transition": "Cut"}),
    (0.5, None, "take", {"index": 2}),
    (7.0, "Scheduled cut at cue-out", None, {}),
    (11.0, "Scheduled cut after a Timed element", None, {}),
    (12.5, "Fade", "transition", {"transition": "Fade", "duration_ms": 800}),
    (13.0, None, "take", {"index": 0}),
]

# (seconds from start, chapter title or None, verb, arguments)
SCRIPT = [
    (0.0, "Scheduled cuts through the playlist", "mode", {"mode": "LoopAll"}),
    (0.0, None, "take", {"index": 0}),
    (26.0, "Manual take and Next", "take", {"index": 2}),
    (30.0, None, "next", {}),
    (35.0, "Fade and interruption", "transition", {"transition": "Fade", "duration_ms": 1200}),
    (36.0, None, "take", {"index": 4}),
    (36.6, None, "take", {"index": 0}),          # interrupts the fade half-way
    (41.0, None, "transition", {"transition": "Cut"}),
    (42.0, "Pause and resume", "pause", {}),
    (45.0, None, "play", {}),
    (49.0, "Edit cue points on air", "edit", {"index": 1, "clip": {
        "url": "", "name": "02-smpte", "cue_in": 4000, "cue_out": 7000, "speed": 1.0}}),
    (50.0, None, "take", {"index": 1}),
    (55.0, "Cold element", "add", {"clip": {"url": "", "name": "05 again", "cue_in": 5000, "cue_out": 8000}}),
    (56.0, None, "take", {"index": 5}),
    (62.0, None, "stop", {}),
]


async def run(host: str, port: int, out: Path, short: bool) -> None:
    client = RemoteClient(host, port)
    await client.connect()
    status = await client.status()
    urls = {i: c["url"] for i, c in enumerate(status.clips)}
    chapters, start = [], time.monotonic()
    start_wallclock_ms = int(time.time() * 1000)
    for at, title, verb, arg in (SHORT if short else SCRIPT):
        await asyncio.sleep(max(0.0, start + at - time.monotonic()))
        arg = json.loads(json.dumps(arg))
        if "clip" in arg and not arg["clip"]["url"]:
            arg["clip"]["url"] = urls[arg.get("index", 4)]
        if title:
            chapters.append({"t": round(time.monotonic() - start, 1), "title": title})
        if verb is None:
            continue                                   # chapter mark only
        await client.send(verb, **arg)
        s = await client.status()
        shown = json.dumps(arg) if arg else ""
        print(f"{time.monotonic() - start:6.1f}s {verb:10} {shown:<40} active={s.active} "
              f"transport={s.transport} error={s.error!r}", flush=True)
    out.write_text(json.dumps({"start_wallclock_ms": start_wallclock_ms, "chapters": chapters}, indent=2))
    await client.disconnect()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7778)
    p.add_argument("--out", type=Path, default=Path("chapters.json"))
    p.add_argument("--short", action="store_true", help="15-second movie script")
    args = p.parse_args(argv)
    asyncio.run(run(args.host, args.port, args.out, args.short))
    return 0


if __name__ == "__main__":
    sys.exit(main())
