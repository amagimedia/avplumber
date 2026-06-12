#!/usr/bin/env python3
"""Tactical-dot jitter metric.

Reads the draw_tactical_court dot_log ndjson (one record per tracked dot per
frame: filtered x/y and raw rx/ry court-ft) and reports per-track jitter —
the p50/p90 of frame-to-frame 2nd differences (ft/frame^2) — for the raw
measurements and the filtered positions in the same run, plus speed sanity.
Smooth human motion has near-zero 2nd difference at 25 fps; foot/H noise
shows up directly here. No ground truth needed.

Usage: dot_jitter_eval.py <dot_log.ndjson>
"""
import json
import sys
from collections import defaultdict

import numpy as np

tracks = defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line)
    tracks[r["id"]].append((r["frame"], r["x"], r["y"], r["rx"], r["ry"],
                            r.get("fb", 0)))

acc_f, acc_r, speeds, n_pts = [], [], [], 0
fb_frac = []
for tid, rows in tracks.items():
    rows.sort()
    f = np.array([r[0] for r in rows], dtype=float)
    xy = np.array([[r[1], r[2]] for r in rows])
    rxy = np.array([[r[3], r[4]] for r in rows])
    fb_frac.append(np.mean([r[5] for r in rows]))
    # only consecutive-frame triples (gaps break the 2nd difference)
    for arr, out in ((xy, acc_f), (rxy, acc_r)):
        d2 = arr[2:] - 2 * arr[1:-1] + arr[:-2]
        ok = (np.diff(f)[1:] == 1) & (np.diff(f)[:-1] == 1)
        out.extend(np.hypot(d2[ok, 0], d2[ok, 1]))
    v = np.hypot(*np.diff(xy, axis=0).T)
    speeds.extend(v[np.diff(f) == 1])
    n_pts += len(rows)


def pct(a, q):
    return float(np.percentile(np.asarray(a), q)) if len(a) else float("nan")


print(f"tracks={len(tracks)} points={n_pts} "
      f"fallback_frac={np.mean(fb_frac):.2f}")
print(f"jitter RAW      ft/f^2: p50={pct(acc_r, 50):.3f} "
      f"p90={pct(acc_r, 90):.3f} p99={pct(acc_r, 99):.3f}")
print(f"jitter FILTERED ft/f^2: p50={pct(acc_f, 50):.3f} "
      f"p90={pct(acc_f, 90):.3f} p99={pct(acc_f, 99):.3f}")
print(f"speed FILTERED  ft/f:   p50={pct(speeds, 50):.3f} "
      f"p90={pct(speeds, 90):.3f} p99={pct(speeds, 99):.3f} "
      f"(sprint ~ 1.2)")
