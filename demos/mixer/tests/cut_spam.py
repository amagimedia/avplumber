#!/usr/bin/env python3
"""Cut/transition spam gate for a live mixer demo, driven through its web UI API.

    python3 cut_spam.py --url http://127.0.0.1:7681 --duration 60 --rate 4

Spaced baseline cuts, bursts of auto-repeated cuts, `--duration` s of cut/fade/wipe takes at
`--rate` with jitter, a 2 s pause and spaced recovery cuts. Needs the mixer's cut probe
(--cut-latency-encoder); see docs/latency.md "Cut spam gate" for what it measures.
Exit status 0 = PASS.
"""
import argparse, http.client, json, math, random, statistics, sys, threading, time
import urllib.error, urllib.request

EXPECTED_REJECTIONS = ("transition already in progress",)
THRESHOLDS = ("spam_p95_ms", "spam_max_ms", "recovery_p50_ms", "recovery_max_ms",
              "measured_ratio_min", "playout_repeats_max")
UNPROBED_WIPE_S = 3.0   # wipe without duration_seconds: the mixer probes the clip, we cannot
BURST_TAKES, BURST_GAP_S = 5, 0.05   # keyboard auto-repeat on a take key
TRANSITION_START = ("not measurable: the cut probe samples cuts only; mixer.status needs a "
                    "cut_latency-style 'transition' sample (id, kind, scene, state, ms) from command "
                    "receipt to the first encoded frame of the fade or wipe branch")


# ---- pure logic, unit-tested by test_cut_spam.py ----

def parse_mix(text):
    parts = text.split(":")
    weights = dict(zip(("cut", "fade", "wipe"), map(float, parts)))
    if len(parts) != 3 or min(weights.values()) < 0 or not any(weights.values()):
        raise ValueError(f"--mix wants three non-negative cut:fade:wipe weights, got {text!r}")
    return weights


def percentile(values, p):
    """Nearest-rank percentile, None without values."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)] if ordered else None


def summarize(values):
    return {"n": len(values), "p50": statistics.median(values) if values else None,
            "p95": percentile(values, 95), "max": max(values, default=None)}


def thresholds(fps, baseline, overrides=None):
    """Limits relative to this run's own spaced cuts (`baseline` is their summary), so a slower
    setup is judged on what hammering adds; a frame F at the show rate keeps them meaningful at
    25 fps as well as at 60. Latencies in ms, None without a baseline sample."""
    frame = 1000 / fps

    def above(base, frames, ms=0):
        return None if base is None else round(base + frames * frame + ms, 1)
    limits = {"spam_p95_ms": above(baseline["p50"], 2, 10), "spam_max_ms": above(baseline["p50"], 4, 10),
              "recovery_p50_ms": above(baseline["p50"], 1), "recovery_max_ms": above(baseline["max"], 1),
              "measured_ratio_min": 0.9, "playout_repeats_max": None}
    limits.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return limits


def probe_entries(cut_latency):
    """Every sample in status.cut_latency: current direct/previewed plus their `recent` lists."""
    for entry in (cut_latency or {}).values():
        if isinstance(entry, dict):   # skips the "endpoint" and "encoder" strings
            yield entry
            yield from entry.get("recent") or []


def latest_id(cut_latency):
    """Probe ids are shared by both categories and grow with every cut, measured or not."""
    return max((e.get("id", 0) for e in probe_entries(cut_latency)), default=0)


def measured(cut_latency, after_id):
    return {e["id"]: e for e in probe_entries(cut_latency)
            if e.get("state") == "measured" and e.get("ms") is not None and e.get("id", 0) > after_id}


def sample(cut_latency, probe_id, scene):
    """ms of cut `probe_id` once it is measured and went to `scene`, else None."""
    entry = measured(cut_latency, probe_id - 1).get(probe_id)
    return entry["ms"] if entry and entry.get("scene") == scene else None


def last_states(cut_latency, after_id):
    """{id: state} of every sample after `after_id` as this poll shows it; a poller that
    updates a dict with it keeps the last state it saw for each cut."""
    return {e["id"]: e.get("state") for e in probe_entries(cut_latency) if e.get("id", 0) > after_id}


def unmeasured_by_state(probe_ids, states):
    """Counts of the last state the poller saw for each of `probe_ids`, the eligible cuts never
    measured: {"interrupted": 105, "unseen": 2}. "interrupted": the next take cancelled a sample
    the encoder had not completed; "unseen": it came and went between two polls."""
    counts = {}
    for probe_id in probe_ids:
        state = states.get(probe_id) or "unseen"
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))


def eligible_cuts(takes, after_id, gap_s):
    """Probe ids of the spam cuts whose next take came at least `gap_s` later: a take cancels a
    pending measurement, so only these can be measured whatever the take rate. `takes` is
    [(sent_at_s, kind, outcome)] in sending order; each cut the mixer accepted took the next
    probe id after `after_id`. The last take has the pause before recovery after it."""
    eligible, probe_id = set(), after_id
    for i, (sent_at, kind, outcome) in enumerate(takes):
        if kind != "cut" or outcome != "sent":
            continue
        probe_id += 1
        if i + 1 == len(takes) or takes[i + 1][0] - sent_at >= gap_s:
            eligible.add(probe_id)
    return eligible


def playout_delta(before, after):
    """Change of both slot compositors' counters (mixer.status `playout`), summed: the slots take
    turns on program. None when they are not reported or did not advance."""
    if not before or not after:
        return None
    delta = {k: sum((after.get(slot) or {}).get(k, 0) - (before.get(slot) or {}).get(k, 0) for slot in "AB")
             for k in ("frames", "repeats", "missed_deadlines")}
    return delta if delta["frames"] > 0 else None


class PvwSamples:
    """The AUX multiviews' PVW change latencies, informational: `mixer.status` `pvw_latency` holds
    each follower's (`aux_<bus>_pvw`) last timed change, so every status poll feeds `observe` and
    a change counts once, by its revision, under its take's `kind`. `pvw_minus_pgm_ms` is how
    much later than the program frame the multiview frame with the new PVW tile leaves its
    compositor (0 when aligned, one aux tick when the change missed its tick: `late`, or when
    its tick had passed before the mixer published it: `unreachable`, every fade); the latencies
    run from the take's receipt to the two compositor deadlines, so the cut probe's
    encoder-output `ms` exceeds a cut's `pgm_latency_ms` by the program encoder's share, and a
    fade's include the fade."""

    def __init__(self):
        self.samples = {}

    def observe(self, status):
        for follower, entry in ((status or {}).get("pvw_latency") or {}).items():
            if isinstance(entry, dict) and entry.get("pvw_minus_pgm_ms") is not None:
                self.samples.setdefault((follower, entry.get("applied_revision")), entry)

    @staticmethod
    def block(entries):
        """Counts and summaries of the difference and the latencies over *entries*."""
        def of(key):
            return summarize([e[key] for e in entries if isinstance(e.get(key), (int, float))])
        unreachable = [bool(e.get("target_unreachable")) for e in entries]
        return {"n": len(entries), "pvw_minus_pgm": of("pvw_minus_pgm_ms"), "pvw_latency": of("pvw_latency_ms"),
                "pgm_latency": of("pgm_latency_ms"), "unreachable": sum(unreachable),
                "late": sum((e.get("last_target_error_ticks") or 0) > 0 and not u for e, u in zip(entries, unreachable))}

    def summary(self):
        """None without a sample; else the followers, their alignments and one block per take kind."""
        if not self.samples:
            return None
        entries = list(self.samples.values())
        kinds = sorted({e.get("kind") or "cut" for e in entries})
        return {"followers": sorted({follower for follower, _ in self.samples}),
                "align": sorted({str(e.get("align")) for e in entries}),
                "kinds": {kind: self.block([e for e in entries if (e.get("kind") or "cut") == kind]) for kind in kinds}}


def pick_scenes(names, explicit=None, prefix=""):
    chosen = explicit or [n for n in names if not n.startswith("aux") and n.startswith(prefix)]
    unknown = sorted(set(chosen) - set(names))
    if unknown:
        raise ValueError(f"unknown scenes: {', '.join(unknown)}")
    if len(set(chosen)) < 2:
        raise ValueError("need at least two scenes to cut between")
    return chosen


def transition_payloads(settings):
    """Command arguments per take kind, as the web UI sends them; wipe is None without a clip."""
    fade = {"duration_sec": settings.get("fade_seconds") or 0.5}
    if settings.get("fade_curve", "linear") != "linear":
        fade["curve"] = settings["fade_curve"]
    if settings.get("fade_color"):
        fade["color"] = settings["fade_color"]
    clips = settings.get("wipes") or []
    clip = next((w for w in clips if w.get("id") == settings.get("default_wipe")), clips[0] if clips else {})
    wipe = {"wipe_file": clip.get("path") or settings.get("wipe_file")}
    if clip.get("duration_seconds"):
        wipe["duration_sec"] = clip["duration_seconds"]
    return {"cut": {}, "fade": fade, "wipe": wipe if wipe["wipe_file"] else None}


def is_expected_rejection(message):
    return any(text in (message or "").lower() for text in EXPECTED_REJECTIONS)


def setup_check(before, after):
    """A managed mixer must still run the same setup revision; `before` None means unmanaged."""
    if before is None:
        return True, "not managed by this web UI"
    after = after or {}
    ok = after.get("phase") == "running" and after.get("revision") == before.get("revision")
    return ok, f"phase {after.get('phase')}, revision {before.get('revision')} -> {after.get('revision')}"


def verdict(limits, spam, burst, recovery, playout, errors, setup, scene):
    """[(criterion, ok, detail)]; `setup` and `scene` are (ok, detail), `playout` is a
    playout_delta. A criterion without samples fails."""
    def at_most(name, value, limit):
        return name, value is not None and limit is not None and value <= limit, f"{fmt(value)} ms, limit {fmt(limit)}"

    def ratio(name, stats):
        value, low, pool = stats.get("measured_ratio"), limits["measured_ratio_min"], stats.get("eligible")
        if pool == 0:
            return name, False, "no cut had the spam max limit before the next take; lower --rate"
        return name, value is not None and value >= low, f"{pct(value)} of {pool} cuts measured, minimum {pct(low)}"

    def counter(name, key, limit):
        if playout is None:
            return name, False, "mixer.status reports no advancing playout counters"
        if limit is None:   # repeats count every input showing its previous frame, e.g. a page that paints slower
            return name, True, f"{playout[key]} over {playout['frames']} frames, informational"
        return name, playout[key] <= limit, f"{playout[key]} over {playout['frames']} frames, limit {limit:g}"
    return [("no errors", not errors, f"{len(errors)} unexpected"), ("setup running", *setup),
            at_most("spam p95", spam["p95"], limits["spam_p95_ms"]),
            at_most("spam max", spam["max"], limits["spam_max_ms"]),
            ratio("spam measured", spam),
            at_most("burst p95", burst["p95"], limits["spam_p95_ms"]),
            at_most("burst max", burst["max"], limits["spam_max_ms"]),
            ratio("burst measured", burst),
            at_most("recovery p50", recovery["p50"], limits["recovery_p50_ms"]),
            at_most("recovery max", recovery["max"], limits["recovery_max_ms"]),
            counter("program missed deadlines", "missed_deadlines", 0),
            counter("program repeats", "repeats", limits["playout_repeats_max"]),
            ("program on last target", *scene)]


def fmt(value):
    return "-" if value is None else f"{value:.1f}"


def pct(value):
    return "-" if value is None else f"{value:.0%}"


# ---- live run against the web UI ----

def request(url, body=None, timeout=5.0):
    """(HTTP status, JSON body); status 0 when the server did not answer."""
    data = None if body is None else json.dumps(body).encode()
    try:
        req = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        text = exc.read()
        return exc.code, json.loads(text) if text.startswith(b"{") else {"error": exc.reason}
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return 0, {"error": str(exc)}


class Run:
    def __init__(self, url):
        self.url, self.errors, self.rejections = url.rstrip("/"), [], []
        self.pvw = PvwSamples()

    def state(self):
        code, body = request(self.url + "/api/state")
        if code != 200:
            self.errors.append(f"GET /api/state: HTTP {code} {body.get('error')}")
        return body if code == 200 else None

    def status(self):
        """mixer.status alone: the lightest poll the web UI offers."""
        code, body = request(self.url + "/api/status")
        if code != 200:
            self.errors.append(f"GET /api/status: HTTP {code} {body.get('error')}")
        if code == 200:
            self.pvw.observe(body)
        return body if code == 200 else {}

    def setup(self):
        code, body = request(self.url + "/api/setup")
        return body if code == 200 else None   # 404: this web UI does not manage the mixer

    def command(self, kind, scene, **extra):
        """"sent", "superseded" (the web UI replaced it with a newer take) or None on failure."""
        code, body = request(self.url + "/api/command", {"command": kind, "scene": scene, **extra})
        if code != 200:
            message = f"{kind} {scene}: HTTP {code} {body.get('error')}"
            (self.rejections if is_expected_rejection(body.get("error")) else self.errors).append(message)
            return None
        return "superseded" if body.get("superseded") else "sent"

    def settle(self, target, wait_s):
        """(True, ...) once `target` is on program with no transition running, within `wait_s`."""
        deadline = time.monotonic() + wait_s
        while True:
            status = self.status()
            if status.get("pgm_scene") == target and status.get("transition") == "idle":
                return True, f"{target} on program"
            if time.monotonic() > deadline:
                return False, f"program {status.get('pgm_scene')!r} ({status.get('transition')}), not {target!r}"
            time.sleep(0.1)


def next_target(scenes, previous, rng):
    return rng.choice([s for s in scenes if s != previous])


def spaced_cuts(run, scenes, count, rng, target):
    """One cut at a time, each waited for, as an operator would take them."""
    samples = []
    for _ in range(count):
        after = latest_id(run.status().get("cut_latency"))
        scene = next_target(scenes, target, rng)
        if run.command("cut", scene):
            target = scene
            for _ in range(30):
                time.sleep(0.1)
                found = measured(run.status().get("cut_latency"), after)
                if found:
                    samples.append(found[min(found)]["ms"])
                    break
        time.sleep(0.9 + rng.random() * 0.4)   # vary the phase against the frame tick
    return samples, target


def bursts(run, scenes, count, rng, target):
    """Auto-repeat on the cut key: BURST_TAKES cuts BURST_GAP_S apart, each sent without waiting
    for the previous reply, as the page does. The web UI may coalesce them. Only the last is
    measured: its probe latency plus its reply time, which bounds its wait in the web UI."""
    samples, superseded = [], 0
    for _ in range(count):
        chain = [next_target(scenes, target, rng)]
        while len(chain) < BURST_TAKES:
            chain.append(next_target(scenes, chain[-1], rng))
        replies = [(None, 0.0)] * len(chain)

        def send(i):
            start = time.monotonic()
            replies[i] = (run.command("cut", chain[i]), time.monotonic() - start)
        threads = [threading.Thread(target=send, args=(i,)) for i in range(len(chain))]
        for thread in threads:
            thread.start()
            time.sleep(BURST_GAP_S)
        for thread in threads:
            thread.join()
        superseded += sum(outcome == "superseded" for outcome, _ in replies)
        outcome, reply_s = replies[-1]
        if outcome == "sent":
            target = chain[-1]
            probe_id = latest_id(run.status().get("cut_latency"))
            for _ in range(30):
                ms = sample(run.status().get("cut_latency"), probe_id, target)
                if ms is not None:
                    samples.append(ms + reply_s * 1000)
                    break
                time.sleep(0.1)
        time.sleep(0.9 + rng.random() * 0.4)
    return samples, superseded, target


def spam(run, args, scenes, payloads, mix, rng, target, max_s):
    """Send takes on an absolute schedule, one at a time so they arrive in the order sent;
    a 10 Hz poller collects every cut measured meanwhile, including ones that land late.
    `max_s` is the spam max limit: the gap that makes a cut eligible and the settle slack."""
    after, samples, states, stop = latest_id(run.status().get("cut_latency")), {}, {}, threading.Event()

    def poll():
        while not stop.wait(0.1):
            cut_latency = run.status().get("cut_latency")
            samples.update(measured(cut_latency, after))
            states.update(last_states(cut_latency, after))
    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    sent, takes, last_kind = dict.fromkeys(mix, 0), [], "cut"
    next_at = time.monotonic()
    end = next_at + args.duration
    while next_at < end:
        time.sleep(max(0.0, next_at - time.monotonic()))
        kind = rng.choices(list(mix), list(mix.values()))[0]
        scene = next_target(scenes, target, rng)
        takes.append((time.monotonic(), kind, run.command(kind, scene, **payloads[kind])))
        if takes[-1][2] == "sent":
            sent[kind] += 1
            target, last_kind = scene, kind
        next_at += rng.uniform(0.5, 1.5) / args.rate
    # pgm_scene flips when a fade or wipe ends, and a take may start as late as a slow cut.
    length = payloads[last_kind].get("duration_sec", UNPROBED_WIPE_S if last_kind == "wipe" else 0)
    settled = run.settle(target, max(length, 1.0) + max_s)
    time.sleep(2)   # the pause before recovery; the poller still collects the last cut
    stop.set()
    poller.join()
    eligible = eligible_cuts(takes, after, max_s)
    stats = {**summarize([samples[i]["ms"] for i in sorted(samples)]), "sent": sent,
             "eligible": len(eligible), "measured_eligible": len(eligible & samples.keys()),
             "unmeasured": unmeasured_by_state(eligible - samples.keys(), states)}
    stats["measured_ratio"] = stats["measured_eligible"] / len(eligible) if eligible else None
    return stats, target, settled


def gate(args):
    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    rng, run = random.Random(seed), Run(args.url)
    state = run.state()
    if state is None:
        raise RuntimeError(f"no mixer state at {args.url}: {run.errors[-1]}")
    status, settings = state.get("status", {}), state.get("settings") or {}
    if not status.get("cut_latency"):
        raise RuntimeError("cut-latency probe is off; start the mixer with --cut-latency-encoder <encoder>")
    fps = args.fps or (settings.get("canvas") or {}).get("fps")
    if not fps:
        raise RuntimeError("no canvas fps in /api/state settings; pass --fps")
    scenes = pick_scenes(state.get("scenes") or [], args.scenes and args.scenes.split(","), args.scene_prefix)
    payloads = transition_payloads(settings)
    mix = {k: w for k, w in parse_mix(args.mix).items() if w and payloads[k] is not None}
    if not mix:
        raise ValueError("--mix leaves no take to send (wipes need a configured clip)")
    setup_before = run.setup()
    if setup_before and setup_before.get("phase") != "running":
        raise RuntimeError(f"setup phase is {setup_before.get('phase')!r}, not 'running'")

    baseline, target = spaced_cuts(run, scenes, args.baseline_cuts, rng, status.get("pgm_scene"))
    base = summarize(baseline)
    if base["p50"] is None:
        raise RuntimeError("no baseline cut was measured; every limit is relative to them")
    limits = thresholds(fps, base, {k: getattr(args, k) for k in THRESHOLDS})
    counters_before = run.status().get("playout")
    burst_ms, superseded, target = bursts(run, scenes, args.bursts, rng, target)
    spam_stats, target, after_spam = spam(run, args, scenes, payloads, mix, rng, target,
                                          limits["spam_max_ms"] / 1000)
    recovery, target = spaced_cuts(run, scenes, args.recovery_cuts, rng, target)
    at_end = run.settle(target, 1.0 + limits["recovery_max_ms"] / 1000)
    playout = playout_delta(counters_before, run.status().get("playout"))
    scene = (after_spam[0] and at_end[0], f"after spam: {after_spam[1]}; at end: {at_end[1]}")
    burst_stats = {**summarize(burst_ms), "bursts": args.bursts, "superseded": superseded,
                   "eligible": args.bursts, "measured_ratio": len(burst_ms) / args.bursts if args.bursts else None}
    criteria = verdict(limits, spam_stats, burst_stats, summarize(recovery), playout, run.errors,
                       setup_check(setup_before, run.setup()), scene)
    return {"pass": all(ok for _, ok, _ in criteria), "seed": seed, "fps": fps, "scenes": scenes,
            "mix": mix, "rate": args.rate, "duration_s": args.duration, "thresholds": limits,
            "baseline": base, "burst": burst_stats, "spam": spam_stats, "recovery": summarize(recovery),
            "playout": playout, "pvw": run.pvw.summary(), "transition_start": TRANSITION_START,
            "errors": run.errors, "rejections": run.rejections,
            "criteria": [{"name": n, "ok": ok, "detail": d} for n, ok, d in criteria]}


def pvw_line(pvw):
    """The cuts' block, the goal of the alignment; the other kinds by count. A fade's swap lands
    a tick after the program frame by construction, so its numbers say nothing about lateness."""
    if not pvw:
        return "pvw       no AUX multiview reported a timed PVW change (mixer.status pvw_latency)"
    kind = "cut" if "cut" in pvw["kinds"] else next(iter(pvw["kinds"]))
    block, d, lat = pvw["kinds"][kind], pvw["kinds"][kind]["pvw_minus_pgm"], pvw["kinds"][kind]["pvw_latency"]
    unreachable = f", {block['unreachable']} unreachable" if block["unreachable"] else ""
    others = "".join(f"; {k} n={b['n']}" for k, b in pvw["kinds"].items() if k != kind)
    return (f"pvw       {', '.join(pvw['followers'])} ({', '.join(pvw['align'])}): {kind} n={block['n']} PVW-PGM "
            f"p50={fmt(d['p50'])} p95={fmt(d['p95'])} max={fmt(d['max'])} ms, {block['late']} missed their tick{unreachable}; "
            f"PVW latency p50={fmt(lat['p50'])} max={fmt(lat['max'])} ms (receipt to compositor deadlines, informational){others}")


def print_summary(r):
    s, b, u, rec = r["spam"], r["baseline"], r["burst"], r["recovery"]
    mix = " ".join(f"{kind}:{weight:g}" for kind, weight in r["mix"].items())
    print(f"{len(r['scenes'])} scenes at {r['fps']:g} fps, mix {mix}, "
          f"{r['duration_s']:g} s at {r['rate']:g}/s, seed {r['seed']}\n"
          f"baseline  n={b['n']} p50={fmt(b['p50'])} max={fmt(b['max'])} ms\n"
          f"burst     {u['bursts']} x {BURST_TAKES} cuts, {u['superseded']} coalesced; last measured "
          f"{u['n']} ({pct(u['measured_ratio'])}); p50={fmt(u['p50'])} p95={fmt(u['p95'])} max={fmt(u['max'])} ms\n"
          f"spam      sent {s['sent']}; measured {s['n']} cuts, {s['measured_eligible']}/{s['eligible']} "
          f"eligible ({pct(s['measured_ratio'])}); p50={fmt(s['p50'])} p95={fmt(s['p95'])} max={fmt(s['max'])} ms\n"
          f"          unmeasured eligible by last probe state: "
          f"{' '.join(f'{state}={n}' for state, n in s['unmeasured'].items()) or 'none'}\n"
          f"recovery  n={rec['n']} p50={fmt(rec['p50'])} max={fmt(rec['max'])} ms\n"
          f"playout   {r['playout'] or 'not reported'}\n"
          f"{pvw_line(r.get('pvw'))}\n"
          f"fade/wipe start latency {r['transition_start']}\n"
          f"errors    {len(r['errors'])} unexpected, {len(r['rejections'])} expected rejections")
    for line in r["errors"][:10] + r["rejections"][:5]:
        print("  " + line)
    for c in r["criteria"]:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['name']:<24} {c['detail']}")
    print("PASS" if r["pass"] else "FAIL")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:7681", help="mixer web UI")
    p.add_argument("--duration", type=float, default=60, help="spam seconds")
    p.add_argument("--rate", type=float, default=4, help="spam commands per second")
    p.add_argument("--mix", default="6:2:2", help="cut:fade:wipe weights")
    p.add_argument("--scenes", help="comma-separated scenes (default: every scene not named aux*)")
    p.add_argument("--scene-prefix", default="", help="only default scenes starting with this")
    p.add_argument("--baseline-cuts", type=int, default=15)
    p.add_argument("--recovery-cuts", type=int, default=15)
    p.add_argument("--bursts", type=int, default=10, help=f"bursts of {BURST_TAKES} auto-repeated cuts")
    p.add_argument("--seed", type=int)
    p.add_argument("--fps", type=float, help="show fps, when /api/state reports no canvas")
    p.add_argument("--json", action="store_true", help="print one JSON object")
    for name in THRESHOLDS:
        p.add_argument("--" + name.replace("_", "-"), type=float, help="override the default limit")
    args = p.parse_args(argv)
    try:
        result = gate(args)
    except (RuntimeError, ValueError) as exc:
        result = {"pass": False, "error": str(exc)}
    if args.json or "error" in result:
        print(json.dumps(result) if args.json else f"FAIL  {result['error']}")
    else:
        print_summary(result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
