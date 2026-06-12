#!/usr/bin/env python3
"""Line-provenance breakdown: for every displayed frame, where did the court
overlay actually come from, and was it verified against evidence?

Joins the stamped calib log (what viewers saw: frame, solved, age, zone)
with the solver sidecar (per-solve: err_ft presence = curve-refined,
seg_conf.arc = px distance of the published arc from the frame's own mask
boundary). Classes:

  fit-frame     stamped on the very frame that was solved (gap == 0)
  extrapolated  motion-compensated from a FRESH fit (solver age == 0, gap > 0)
  held          source solve was itself a hold/coast (solver age > 0)
  invalid       nothing drawn

Quality of the source solve (arc vs own mask evidence):
  on-line <=12px | near <=30px | off >30px | unverified (no arc evidence)

Usage: line_provenance_eval.py <stamped.ndjson> <solver.ndjson>
"""
import json
import sys
from collections import Counter

stamped = [json.loads(l) for l in open(sys.argv[1])]
solver = {r["frame"]: r for r in (json.loads(l) for l in open(sys.argv[2]))}

prov = Counter()
qual = Counter()
gaps = []
n = len(stamped)
for r in stamped:
    if not r.get("valid"):
        prov["invalid"] += 1
        continue
    if r.get("src") == "track":
        prov["tracked (locked line)"] += 1
        continue
    if r.get("zone") == "mid":
        prov["mid (no arcs)"] += 1
        continue
    src = solver.get(r.get("solved", -1))
    gap = r["frame"] - r.get("solved", r["frame"])
    gaps.append(gap)
    src_age = src.get("age", 0) if src else 0
    if src_age > 0:
        prov["held"] += 1
    elif gap == 0:
        prov["fit-frame"] += 1
    else:
        prov["extrapolated"] += 1
    arc = (src or {}).get("seg_conf", {}).get("arc")
    if arc is None:
        qual["unverified (no arc evidence)"] += 1
    elif arc[0] <= 12:
        qual["on-line (<=12px)"] += 1
    elif arc[0] <= 30:
        qual["near (<=30px)"] += 1
    else:
        qual["OFF (>30px)"] += 1

print(f"total frames: {n}")
print("\n-- overlay provenance (all frames) --")
for k, v in prov.most_common():
    print(f"  {k:24s} {v:5d}  {100*v/n:5.1f}%")
drawn = sum(v for k, v in prov.items() if k not in ("invalid",))
print("\n-- source-solve arc quality (frames with arcs drawn) --")
tq = sum(qual.values())
for k in ("on-line (<=12px)", "near (<=30px)", "OFF (>30px)",
          "unverified (no arc evidence)"):
    v = qual.get(k, 0)
    print(f"  {k:30s} {v:5d}  {100*v/max(1,tq):5.1f}%")
if gaps:
    gaps.sort()
    print(f"\nstamp gap frames: p50 {gaps[len(gaps)//2]} "
          f"p90 {gaps[int(0.9*len(gaps))]} max {gaps[-1]}")
