"""Where a browser source's repeats and discards come from: every frame's stamps and arrival times
through the show's browser chain, and the playout tick that shows or drops it.

Opens --windows pages that slide for 2 s every 6 s (the show's test page; --page for another) on an
isolated browser worker, builds the show's browser chain for each (ipc_dmabuf_source, drm_prime_to_cuda,
smooth_timestamps, repeat_last_frame as `hold=True` adds it) with a pass-through Python tap after every
node, feeds them into one MixerCompositor (1080p30, a tile each) and runs --seconds. Per window it writes
a CSV with one row per frame the compositor received: the receive time, the paint's own stamp, the smoothed
stamp, the stamp and arrival time at the compositor and the tick a Presentation-mode Playout, replayed
from those arrivals, shows it at or the frame it was superseded by. The summary checks the replay against
the compositor's own per_input counters and says which mechanism the stamps show:

  * `hold` flips: repeat_last_frame stamps each re-emitted frame with its own emission time rounded to the
    nearest slot. Two consecutive held frames with the same slot (a discard) or a slot apart twice (a
    repeat) while their emission phase sits at the half slot is that rounding.
  * late paints: a live paint arriving after its slot's deadline (margin < 0) is the "stamps versus
    arrival lateness" candidate.
  * smooth_timestamps re-alignments inside a slide (a live slot step other than +1 without a 100 ms gap).

A diagnosis tool, not a pass/fail smoke. Never point it at a shared production worker. The isolated
browser must run with the show's capture mode (DMA_BROWSER_ZERO_COPY_EXPORT=1) and, because the taps hold
each frame a little longer, DMA_BROWSER_EXPORT_BUFFERS=8 rather than the show's 6; the PRECONDITION line
(about 60 paints per cycle, no browser drops) says whether a run can be read.
"""
import argparse
import base64
import csv
import json
import math
import os
import statistics
import sys
import time
import uuid
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import finish, make_avp, start  # noqa: E402
from pyplumber.mixer.config import default_browser_frame_queue  # noqa: E402
from pyplumber.mixer.dmabuf_inputs import (  # noqa: E402
    browser_capture_mode, dmabuf_cuda_input_nodes, open_windows, refresh_windows, rest_request, wait_for_sockets,
)

EOF = -(1 << 63)
NS = 1_000_000_000
FPS = 30
GROUP = "timing"
SEQ_KEY = "probe_seq"

# A 6 s cycle: 2 s of motion, 4 s without a change, so the page paints 30 fps then nothing.
PAGE = ('<body style="margin:0;background:#123">'
        '<div style="position:absolute;left:0;top:40%;width:200px;height:200px;background:#e83;'
        'animation:s 6s linear infinite"></div>'
        '<style>@keyframes s{0%{transform:translateX(0)}33.333%{transform:translateX(1700px)}'
        '100%{transform:translateX(1700px)}}</style>')


# --- the compositor's tick grid (src/mixer/primitives/TickGrid.hpp) ---------------------------------

def tick_time(k):
    return (k * NS) // FPS


def nearest_index(t_ns):
    period = NS
    return ((t_ns + 1) * FPS - 1 + period // 2) // period


def at_or_before(t_ns):
    return ((t_ns + 1) * FPS - 1) // NS


LATENCY_NS = tick_time(2)          # default_latency_ms at 30 fps: two frames
PERIOD_NS = NS / FPS


def pts_ns(ts, num, den):
    """A frame pts as the compositor reads it: nanoseconds, nearest."""
    return int(round(ts * num * NS / den))


def grid_frac(t_ns):
    """Where in its slot an instant falls, 0 at a tick, 0.5 at the half slot (the rounding boundary)."""
    k = at_or_before(t_ns)
    return (t_ns - tick_time(k)) / (tick_time(k + 1) - tick_time(k))


# --- taps -------------------------------------------------------------------------------------------

class Tap:
    """Pass-through PythonNode that reports every frame and the monotonic time it went by."""
    node_class = None

    @classmethod
    def make(cls, params, on_frame, errors):
        if cls.node_class is None:
            from pyplumber.node import PythonNode

            class TapNode(PythonNode):
                def __init__(self, params, on_frame, errors):
                    super().__init__(params)
                    self._on_frame, self._errors = on_frame, errors

                def process(self):
                    frame = self._src.tryGet(100)
                    if frame is None:
                        return
                    if frame.pts.timestamp != EOF:
                        try:
                            self._on_frame(frame, time.monotonic_ns())
                        except Exception as exc:   # a broken tap must not stop the chain
                            if len(self._errors) < 5:
                                self._errors.append(("tap", self.parameters.get("name"), repr(exc)))
                    self._dst.enqueue(frame)
            cls.node_class = TapNode
        return cls.node_class(params, on_frame, errors)


class WindowLog:
    """Everything the four taps of one window saw, keyed by a sequence number given after the import."""

    def __init__(self, index):
        self.index = index
        self.recv = {}        # in_pts_us -> receive time ns (after ipc_dmabuf_source)
        self.cuda = []        # (seq, in_pts_us, ns)            after drm_prime_to_cuda
        self.smooth = []      # (seq, ns, pts, num, den)        after smooth_timestamps
        self.comp = []        # (seq, ns, pts, num, den)        after repeat_last_frame = compositor input
        self._next_seq = 0

    def on_recv(self, frame, now):
        self.recv[frame.pts.timestamp] = now

    def on_cuda(self, frame, now):
        seq = self._next_seq
        self._next_seq += 1
        frame.metadata[SEQ_KEY] = str(seq)
        self.cuda.append((seq, frame.pts.timestamp, now))

    @staticmethod
    def _seq(frame):
        return int(frame.metadata[SEQ_KEY]) if SEQ_KEY in frame.metadata else -1

    def on_smooth(self, frame, now):
        pts = frame.pts
        self.smooth.append((self._seq(frame), now, pts.timestamp, pts.timebase.num, pts.timebase.den))

    def on_comp(self, frame, now):
        pts = frame.pts
        self.comp.append((self._seq(frame), now, pts.timestamp, pts.timebase.num, pts.timebase.den))


def splice_taps(nodes, prefix, log, errors):
    """A tap after every node of the chain, in the chain's own order (followed edge by edge from the
    receiver, whichever way the hold and the numbering are arranged), by renaming each output edge.
    Roles by position: the first node is the receiver (receive time), the second the importer (gives
    the seq), the last feeds the compositor (stamp and arrival); anything between is logged as the
    stamp entering the last node. Returns the taps and the final edge."""
    order = []
    node = next(n for n in nodes if n.parameters["name"] == f"{prefix}_receive")
    while node is not None:
        order.append(node)
        node = next((n for n in nodes if n.parameters.get("src") == node.parameters["dst"]), None)
    roles = [log.on_recv, log.on_cuda] + [log.on_smooth] * (len(order) - 3) + [log.on_comp]
    taps = []
    final = None
    for node, callback in zip(order, roles):
        edge = node.parameters["dst"]
        tapped = f"{edge}_t"
        for other in nodes:
            if other.parameters.get("src") == edge:
                other.parameters["src"] = tapped
        taps.append(Tap.make({"src": edge, "dst": tapped, "name": f"{node.parameters['name']}_tap", "group": GROUP},
                             callback, errors))
        final = tapped
    return taps, final


# --- the playout replay (src/mixer/Playout.hpp, TimestampMode::Presentation, one input) --------------

def replay(frames, slack_ns, count_from_ns, count_to_ns):
    """frames: list of dicts with 'arrival' and 'slot' in arrival order. Ticks are drawn at
    tick_time(k) + latency + slack. Returns (repeats, discards, discontinuities) counted for ticks drawn
    in [count_from_ns, count_to_ns] and fills each frame's 'sim' (tick or None) and 'outcome'."""
    pending = deque(frames)
    queue = []
    held = None
    index = None
    next_index = None
    repeats = discards = discontinuities = 0
    for f in frames:
        f["sim"], f["outcome"] = None, "pending"

    def counting(t):
        return count_from_ns <= t <= count_to_ns

    def discard(f, why, tick):
        nonlocal discards
        f["outcome"], f["sim"] = why, tick
        if tick is not None and counting(tick_time(tick) + LATENCY_NS):
            discards += 1

    def push(f):
        nonlocal next_index, index, discontinuities
        slot = f["slot"]
        if next_index is not None and slot < next_index - 1:
            for q in queue:
                discard(q, "discarded_discontinuity", at_or_before(f["arrival"] - LATENCY_NS))
            queue.clear()
            discontinuities += 1
        next_index = slot + 1
        queue.append(f)
        if index is None:
            index = slot
        # discardSuperseded as of this arrival
        due = at_or_before(f["arrival"] - LATENCY_NS)
        while len(queue) >= 2 and queue[1]["slot"] <= due:
            discard(queue.pop(0), "discarded_superseded", due)

    while pending or queue:
        if index is None:
            push(pending.popleft())
            continue
        draw_at = tick_time(index) + LATENCY_NS + slack_ns
        while pending and pending[0]["arrival"] <= draw_at:
            push(pending.popleft())
        if not pending and not queue:
            break
        if pending and not queue and pending[0]["arrival"] > draw_at:
            # Ticks drawn strictly before the next frame arrives: each repeats the held picture.
            # The last of them is the largest k with tick_time(k) + latency + slack < arrival; the
            # tick after it is drawn at or after the arrival, so the next pass pushes the frame.
            last_idle = at_or_before(pending[0]["arrival"] - LATENCY_NS - slack_ns - 1)
            if held is not None:
                for k in range(index, last_idle + 1):
                    if counting(tick_time(k) + LATENCY_NS):
                        repeats += 1
            index = last_idle + 1
            continue
        count = 0
        for q in queue:
            if q["slot"] > index:
                break
            count += 1
        if count:
            for q in queue[:count - 1]:
                discard(q, f"discarded_superseded_by_{queue[count - 1]['seq']}", index)
            shown = queue[count - 1]
            shown["outcome"], shown["sim"] = "shown", index
            held = shown
            del queue[:count]
        elif held is not None and counting(draw_at - slack_ns):
            repeats += 1
        index += 1
    return repeats, discards, discontinuities


# --- analysis ---------------------------------------------------------------------------------------

def assemble(log):
    """One record per frame the compositor received, in arrival order.

    A frame is live when its `probe_seq` is seen for the first time, held when the seq is reused. A frame
    without a seq (metadata that did not survive a node) is classified by timing instead: live if it left
    the smooth tap within 3 ms before it reached the compositor tap, held otherwise, and the live ones then
    take their seq from the cuda tap in order (the chain is FIFO). `unknown` counts the fallbacks."""
    cuda = {seq: (in_pts, ns) for seq, in_pts, ns in log.cuda}
    smooth = {}
    for seq, ns, pts, num, den in log.smooth:
        smooth.setdefault(seq, (ns, pts, num, den))
    smooth_times = sorted(ns for _, ns, _, _, _ in log.smooth) or sorted(ns for _, _, ns in log.cuda)
    cuda_in_order = [seq for seq, _, _ in log.cuda]
    seen = set()
    frames = []
    unknown = 0
    live_by_order = 0
    smooth_claimed = 0
    for seq, ns, pts, num, den in log.comp:
        if seq < 0:
            unknown += 1
            while smooth_claimed < len(smooth_times) and smooth_times[smooth_claimed] < ns - 3_000_000:
                smooth_claimed += 1
            if smooth_claimed < len(smooth_times) and smooth_times[smooth_claimed] <= ns:
                smooth_claimed += 1
                seq = cuda_in_order[live_by_order] if live_by_order < len(cuda_in_order) else -1
                live_by_order += 1
                kind = "live"
            else:
                kind = "held"
        else:
            kind = "held" if seq in seen else "live"
        in_pts, cuda_ns = cuda.get(seq, (None, None))
        slot = pts if (num, den) == (1, FPS) else nearest_index(pts_ns(pts, num, den))
        if seq >= 0:
            seen.add(seq)
        sm = smooth.get(seq)
        frames.append({
            "seq": seq, "kind": kind, "in_pts_us": in_pts,
            "recv_ns": log.recv.get(in_pts) if (in_pts is not None and kind == "live") else None,
            "cuda_ns": cuda_ns if kind == "live" else None,
            "smooth_ns": sm[0] if sm and kind == "live" else None,
            "smooth_pts": sm[1] if sm else None,
            "arrival": ns, "comp_pts": pts, "slot": slot,
            "grid_frac": grid_frac(ns),
            "margin_ms": (tick_time(slot) + LATENCY_NS - ns) / 1e6,
        })
    return frames, unknown


def cycles_of(frames):
    """Slides: runs of live frames separated by more than half a second without a paint."""
    cycles, current = [], None
    for i, f in enumerate(frames):
        if f["kind"] != "live":
            continue
        if current is None or f["arrival"] - frames[current[-1]]["arrival"] > NS // 2:
            current = []
            cycles.append(current)
        current.append(i)
    return cycles


def slope(values, wrap=None):
    """Least-squares slope per index; None below three points. With *wrap*, a step of more than half
    of it between neighbours is a wrap of a sawtooth (g runs from 0 to one slot) and is undone."""
    if len(values) < 3:
        return None
    if wrap:
        unwrapped, offset = [values[0]], 0.0
        for a, b in zip(values, values[1:]):
            if b - a > wrap / 2:
                offset -= wrap
            elif a - b > wrap / 2:
                offset += wrap
            unwrapped.append(b + offset)
        values = unwrapped
    xs = range(len(values))
    mx, my = statistics.fmean(xs), statistics.fmean(values)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / sxx if sxx else None


def analyse(frames, unknown, actual, count_from, count_to, slacks_ms):
    res = {"live": sum(f["kind"] == "live" for f in frames), "held": sum(f["kind"] == "held" for f in frames),
           "unknown": unknown}
    # consecutive slot steps, by the kinds on each side
    steps = {"held-held": {"dup": 0, "skip": 0, "n": 0}, "live-live": {"dup": 0, "skip": 0, "gap": 0, "n": 0},
             "handoff": {"dup": 0, "skip": 0, "clean": 0, "n": 0}}
    flip_fracs, quiet_fracs = [], []
    for a, b in zip(frames, frames[1:]):
        d = b["slot"] - a["slot"]
        dt = b["arrival"] - a["arrival"]
        if a["kind"] == "held" and b["kind"] == "held":
            key = "held-held"
            flipped = d != 1 and dt < 2.5 * PERIOD_NS
            (flip_fracs if flipped else quiet_fracs).append(b["grid_frac"])
        elif a["kind"] == "live" and b["kind"] == "live":
            key = "live-live"
            if dt > 0.1 * NS:
                steps[key]["gap"] += 1
                steps[key]["n"] += 1
                continue
        else:
            key = "handoff"
            if d == 1:
                steps[key]["clean"] += 1
        steps[key]["n"] += 1
        if d <= 0:
            steps[key]["dup"] += 1
        elif d >= 2 and (key != "held-held" or dt < 2.5 * PERIOD_NS):
            steps[key]["skip"] += 1
    res["steps"] = steps
    live = [f for f in frames if f["kind"] == "live"]
    res["live_late"] = sum(f["margin_ms"] < 0 for f in live)
    res["live_margin_ms"] = {"min": round(min((f["margin_ms"] for f in live), default=math.nan), 2),
                             "median": round(statistics.median([f["margin_ms"] for f in live]), 2) if live else None}
    held = [f for f in frames if f["kind"] == "held"]
    res["held_phase"] = {
        "median_frac": round(statistics.median([f["grid_frac"] for f in held]), 4) if held else None,
        "flip_frac_from_half": [round(abs(x - 0.5), 4) for x in sorted(flip_fracs, key=lambda x: abs(x - 0.5))][:5],
        "max_flip_from_half": round(max((abs(x - 0.5) for x in flip_fracs), default=math.nan), 4),
        "quiet_min_from_half": round(min((abs(x - 0.5) for x in quiet_fracs), default=math.nan), 4),
    }
    # per cycle: the paint phase (g: slot time minus the paint's own stamp) and the held frames' phase
    per_cycle = []
    for c in cycles_of(frames):
        gs = [(tick_time(frames[i]["slot"]) - frames[i]["in_pts_us"] * 1000) / 1e6
              for i in c if frames[i]["in_pts_us"] is not None]
        first, last = c[0], c[-1]
        after = [frames[j] for j in range(last + 1, len(frames)) if frames[j]["kind"] == "held"][:120]
        held_after = [f for f in after if f["arrival"] - frames[last]["arrival"] < 4.5 * NS]
        flips = sum(1 for a, b in zip(held_after, held_after[1:]) if b["slot"] - a["slot"] != 1)
        before = frames[first - 1] if first else None
        per_cycle.append({
            "t_s": round((frames[first]["arrival"] - frames[0]["arrival"]) / 1e9, 1),
            "paints": len(c), "g_ms": round(statistics.median(gs), 3) if gs else None,
            "held": len(held_after),
            "held_frac": round(statistics.median([f["grid_frac"] for f in held_after]), 4) if held_after else None,
            "held_flips": flips,
            "handoff_in": None if before is None else frames[first]["slot"] - before["slot"],
        })
    res["cycles"] = per_cycle
    res["paints_per_cycle"] = statistics.median([c["paints"] for c in per_cycle]) if per_cycle else None
    g = [c["g_ms"] for c in per_cycle if c["g_ms"] is not None]
    s = slope(g, wrap=PERIOD_NS / 1e6)
    res["g_slope_us_per_cycle"] = None if s is None else round(s * 1000, 1)
    # the replay against the compositor's counters
    sims = {}
    for slack in slacks_ms:
        r, d, disc = replay(frames, int(slack * 1e6), count_from, count_to)
        sims[str(slack)] = {"repeats": r, "discards": d, "discontinuities": disc}
    res["sim"] = sims
    res["actual"] = actual
    return res


def verdict(per_window, browser):
    lines = []
    drops = {}
    for stats in browser.values():
        if isinstance(stats, dict):
            for k, v in (stats.get("droppedReasons") or {}).items():
                drops[k] = drops.get(k, 0) + v
    paints = [w["paints_per_cycle"] for w in per_window if w["paints_per_cycle"] is not None]
    unknown = sum(w["unknown"] for w in per_window)
    lines.append(f"PRECONDITION paints per cycle {paints} (expect ~60 each; fewer means the browser skipped paints "
                 f"and the live stamps count behind); browser drops {drops or 'none'}; frames without probe_seq "
                 f"{unknown} (classified by timing if not 0)")
    flips = sum(w["steps"]["held-held"]["dup"] + w["steps"]["held-held"]["skip"] for w in per_window)
    late = sum(w["live_late"] for w in per_window)
    live_steps = sum(w["steps"]["live-live"]["dup"] + w["steps"]["live-live"]["skip"] for w in per_window)
    hand = sum(w["steps"]["handoff"]["dup"] + w["steps"]["handoff"]["skip"] for w in per_window)
    actual_r = sum(w["actual"]["repeats"] for w in per_window)
    actual_d = sum(w["actual"]["discarded"] for w in per_window)
    lines.append(f"VERDICT compositor counted repeats={actual_r} discards={actual_d} over the run")
    lines.append(f"VERDICT hold rounding flips (held-held dup/skip): {flips}; live paints late for their deadline: {late}; "
                 f"smooth re-alignments inside slides: {live_steps}; slide<->hold handoff steps: {hand}")
    far = [w["held_phase"]["max_flip_from_half"] for w in per_window if not math.isnan(w["held_phase"]["max_flip_from_half"])]
    near = [w["held_phase"]["quiet_min_from_half"] for w in per_window if not math.isnan(w["held_phase"]["quiet_min_from_half"])]
    if far:
        lines.append(f"VERDICT flips sit within {max(far):.3f} of the half slot ({max(far) * PERIOD_NS / 1e6:.2f} ms); "
                     f"the nearest quiet held frame was {min(near) if near else float('nan'):.3f} away")
    for w in per_window:
        best = min(w["sim"].items(), key=lambda kv: abs(kv[1]["repeats"] - w["actual"]["repeats"]) +
                   abs(kv[1]["discards"] - w["actual"]["discarded"]))
        lines.append(f"VERDICT window {w['index']}: replay (slack {best[0]} ms) repeats={best[1]['repeats']} "
                     f"discards={best[1]['discards']} vs compositor repeats={w['actual']['repeats']} "
                     f"discards={w['actual']['discarded']}; g drifts {w['g_slope_us_per_cycle']} us/cycle "
                     f"(33333 us browser interval vs 33333.33 predicts +60)")
    classes = {"hold rounding flips": flips, "late paints": late, "smooth re-alignments": live_steps,
               "slide<->hold handoff steps": hand}
    total = sum(classes.values())
    if actual_r + actual_d == 0 and total == 0:
        lines.append("VERDICT no event in this run: no window's phase was near a boundary; run longer or again")
    else:
        name, n = max(classes.items(), key=lambda kv: kv[1])
        share = f"{100 * n / total:.0f} % of {total} stamp events" if total else "no stamp event"
        lines.append(f"VERDICT dominant: {name} ({share}); a repeat+discard pair per flip means "
                     "repeat_last_frame's nearest-slot stamping of its own emission clock; late paints would "
                     "show as margin < 0 in the CSVs")
    return lines


# --- main -------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rest", default="http://127.0.0.1:19009")
    parser.add_argument("--socket-dir", default="/tmp/dma-test")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--warmup", type=float, default=12)
    parser.add_argument("--page", default=None, help="page URL (default: the 2 s slide every 6 s)")
    parser.add_argument("--no-hold", action="store_true", help="control run without repeat_last_frame")
    parser.add_argument("--input-poll-ms", type=int, default=0, help="the show's canvas.input_poll_ms, if set")
    parser.add_argument("--no-refresh", action="store_true",
                        help="do not reload the pages once the chains run (the show reloads; a reload can give "
                             "a page a new compositor)")
    parser.add_argument("--slack-ms", default="0,1,3", help="tick wake-up slacks the replay tries")
    parser.add_argument("--csv-slack-ms", type=float, default=1.0, help="the replay the CSV columns show")
    parser.add_argument("--out", default=None, help="directory for the CSVs and summary.json")
    args = parser.parse_args()
    from pyplumber import node as api

    url = args.page or "data:text/html;base64," + base64.b64encode(PAGE.encode()).decode("ascii")
    tag = uuid.uuid4().hex[:6]
    out_dir = args.out or f"{args.socket_dir}/probe_timing_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    names = [f"timing_{tag}_{i}" for i in range(args.windows)]
    slacks = [float(s) for s in args.slack_ms.split(",")]
    avp, errors = make_avp("timing_gpu", capacity=3)
    nodes = []
    logs = [WindowLog(i) for i in range(args.windows)]
    try:
        queue = default_browser_frame_queue(FPS)
        open_windows(args.rest, [{"id": n, "url": url, "width": 1920, "height": 1080, "fps": FPS,
                                  "ringSize": queue, "holdLastFrame": True} for n in names])
        wait_for_sockets([f"{args.socket_dir}/{n}.sock" for n in names], 60)
        mode = browser_capture_mode(args.rest)
        queue = mode.frame_queue(queue)
        edges = []
        for i, n in enumerate(names):
            chain, _ = dmabuf_cuda_input_nodes(
                api, prefix=f"w{i}", socket=f"{args.socket_dir}/{n}.sock", fps=FPS, drm_hwaccel=None,
                cuda_hwaccel="timing_gpu", source_group=GROUP, processing_group=GROUP, hold=not args.no_hold,
                preserve_alpha=False, browser_frame_queue=queue, zero_copy=mode.zero_copy)
            taps, edge = splice_taps(chain, f"w{i}", logs[i], errors)
            nodes += chain + taps
            edges.append(edge)
        cols = math.ceil(math.sqrt(args.windows))
        rows = math.ceil(args.windows / cols)
        w, h = 1920 // cols // 2 * 2, 1080 // rows // 2 * 2
        layers = [{"dst_x": (i % cols) * w, "dst_y": (i // cols) * h, "dst_w": w, "dst_h": h}
                  for i in range(args.windows)]
        nodes.append(api.MixerCompositor({
            "name": "compose", "src": edges, "dst": "out", "hwaccel": "timing_gpu", "width": 1920, "height": 1080,
            "sw_format": "nv12", "color": "sdr", "fps": f"{FPS}/1", "active_inputs": args.windows, "layers": layers,
            **({"input_poll_ms": args.input_poll_ms} if args.input_poll_ms else {})}))
        output = start(avp, nodes, GROUP, "out")
        if not args.no_refresh:
            refresh_windows(args.rest, names)
        print(f"capture {mode}; queue {queue}; hold {'off' if args.no_hold else 'on'}; out {out_dir}", flush=True)

        def pump(seconds, poll=None):
            deadline = time.monotonic() + seconds
            next_poll = 0
            while time.monotonic() < deadline and not errors:
                frame = output.tryGet(100)
                if frame is not None and frame.pts.timestamp == EOF:
                    break
                if poll and time.monotonic() >= next_poll:
                    poll()
                    next_poll = time.monotonic() + 2

        pump(args.warmup)
        avp.executeCommandsFromString("node.object.set compose reset_counters true")
        count_from = time.monotonic_ns()
        samples = []

        def poll():
            samples.append((time.monotonic_ns(), avp.node("compose").getObject("per_input")))

        pump(args.seconds, poll)
        poll()
        count_to = samples[-1][0]
        per_input = samples[-1][1]
        status = avp.node("compose").getObject("status")
        # The numbering's own counters (smooth_timestamps with latency_ms); older builds have none.
        stamping = []
        for i in range(args.windows):
            try:
                stamping.append(avp.node(f"w{i}_smooth").getObject("stamping"))
            except Exception as exc:
                stamping.append({"error": repr(exc)})
        browser = {}
        try:
            for wdw in (rest_request(args.rest, "GET", "/status") or {}).get("windows", []):
                if wdw["id"] in names:
                    browser[wdw["id"]] = wdw.get("stats", {})
        except Exception as exc:   # the summary is still worth writing
            browser["error"] = repr(exc)
    finally:
        for n in names:
            try:
                rest_request(args.rest, "POST", "/window/close", {"id": n})
            except Exception:
                pass
        finish(avp, nodes)

    per_window = []
    for i, log in enumerate(logs):
        frames, unknown = assemble(log)
        actual = per_input[i] if i < len(per_input) else {"repeats": 0, "discarded": 0, "missing": True}
        res = analyse(frames, unknown, actual, count_from, count_to, slacks)
        res["index"] = i
        res["stamping"] = stamping[i]
        per_window.append(res)
        # the CSV's sim/outcome columns are the replay at --csv-slack-ms
        replay(frames, int(args.csv_slack_ms * 1e6), count_from, count_to)
        with open(f"{out_dir}/window_{i}.csv", "w", newline="") as fh:
            cols_ = ["seq", "kind", "in_pts_us", "recv_ns", "cuda_ns", "smooth_ns", "smooth_pts", "arrival",
                     "comp_pts", "slot", "grid_frac", "margin_ms", "sim", "outcome"]
            writer = csv.DictWriter(fh, fieldnames=cols_, extrasaction="ignore")
            writer.writeheader()
            for f in frames:
                f["grid_frac"] = round(f["grid_frac"], 4)
                f["margin_ms"] = round(f["margin_ms"], 3)
                writer.writerow(f)
    # the per_input time series, to place episodes in time next to the CSVs
    with open(f"{out_dir}/per_input_samples.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_ns"] + [f"w{i}_{k}" for i in range(args.windows) for k in ("repeats", "discarded")])
        for t, sample in samples:
            writer.writerow([t] + [sample[i][k] if i < len(sample) else "" for i in range(args.windows)
                                   for k in ("repeats", "discarded")])
    summary = {"errors": errors[:5], "latency_ns": LATENCY_NS, "count_from": count_from, "count_to": count_to,
               "compositor": {"missed_deadlines": status["playout"].get("missed_deadlines"),
                              "frames": status["playout"].get("frames")},
               "browser": browser, "windows": per_window}
    with open(f"{out_dir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    # The smoke runner shows the terminal lines cut at 400 columns; summary.json holds everything.
    for w in per_window:
        hh, ll, ho = w["steps"]["held-held"], w["steps"]["live-live"], w["steps"]["handoff"]
        print(f"WINDOW {w['index']}: live={w['live']} held={w['held']} unknown={w['unknown']} "
              f"flips(held-held dup/skip)={hh['dup']}/{hh['skip']} live-live dup/skip/gap={ll['dup']}/{ll['skip']}/{ll['gap']} "
              f"handoff dup/skip/clean={ho['dup']}/{ho['skip']}/{ho['clean']} late={w['live_late']} "
              f"margin_min={w['live_margin_ms']['min']} held_frac={w['held_phase']['median_frac']} "
              f"flip_max_from_half={w['held_phase']['max_flip_from_half']} actual={w['actual']} "
              f"sim1ms={w['sim'].get('1.0') or w['sim'].get('1')}", flush=True)
        st = w["stamping"]
        if "error" not in st:
            print(f"STAMPING {w['index']}: mode={st.get('mode')} shifts_later={st.get('shifts_later')} "
                  f"shifts_back={st.get('shifts_back')} realignments={st.get('realignments')} "
                  f"shift_events={st.get('shift_events')}", flush=True)
    for line in verdict(per_window, browser):
        print(line, flush=True)
    print(f"RESULT missed_deadlines={summary['compositor']['missed_deadlines']} errors={errors[:3]} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
