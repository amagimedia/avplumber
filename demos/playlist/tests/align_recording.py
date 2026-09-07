#!/usr/bin/env python3
"""Compute the program-to-TUI offset for compose_recording.sh.

Uses the frame codes: the manual take in ``record_demo.py --short`` puts element
3 (clip 3) on air at script time 0.5 s.  The program time of clip 3's first frame
and the TUI capture time of the same instant give the offset in seconds.

    python3 tests/align_recording.py movie.ts chapters.json movie/tui.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_recording import decode_codes  # noqa: E402

FPS = 30


def main() -> int:
    program, chapters, tui = sys.argv[1:4]
    codes = decode_codes(program)
    first_clip3 = next(i for i, c in enumerate(codes) if c and c[0] == 3)
    chap = json.loads(Path(chapters).read_text())
    cap = json.loads(Path(tui).read_text())
    take_wallclock = chap["start_wallclock_ms"] + 500
    tui_seconds = (take_wallclock - cap["startWallclock"]) / 1000
    program_seconds = first_clip3 / FPS
    offset = program_seconds - tui_seconds
    print(json.dumps({"program_take_frame": first_clip3, "program_take_s": round(program_seconds, 3),
                      "tui_take_s": round(tui_seconds, 3), "program_offset_s": round(offset, 3),
                      "chapters": [{"t": round(c["t"] + tui_seconds - 0.5, 1), "title": c["title"]}
                                   for c in chap["chapters"]]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
