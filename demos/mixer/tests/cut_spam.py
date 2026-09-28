#!/usr/bin/env python3
"""Cut/transition spam gate for a live mixer demo, driven through its web UI API.

    python3 cut_spam.py --url http://127.0.0.1:7681 --duration 60 --rate 4

Spaced baseline cuts, then `--duration` s of cut/fade/wipe takes at `--rate` with jitter,
a 2 s pause and spaced recovery cuts. Needs the mixer's cut probe (--cut-latency-encoder);
see docs/latency.md "Cut spam gate" for what it measures. Exit status 0 = PASS.
"""
import argparse, http.client, json, math, random, statistics, sys, threading, time
import urllib.error, urllib.request

EXPECTED_REJECTIONS = ("transition already in progress",)
THRESHOLDS = ("spam_p95_ms", "spam_max_ms", "recovery_p50_ms", "recovery_max_ms")
UNPROBED_WIPE_S = 3.0   # wipe without duration_seconds: the mixer probes the clip, we cannot


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


def thresholds(fps, baseline_p50, overrides=None):
    """Latency limits in ms; the frame term keeps them meaningful at 25 fps as well as at 60."""
    frame = 1000 / fps
    limits = {"spam_p95_ms": max(150, 8 * frame), "spam_max_ms": max(300, 15 * frame),
              "recovery_p50_ms": None if baseline_p50 is None else baseline_p50 * 1.25 + frame,
              "recovery_max_ms": max(120, 6 * frame)}
    limits.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return {k: v if v is None else round(v, 1) for k, v in limits.items()}


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


def verdict(limits, spam, recovery, errors, setup, scene):
    """[(criterion, ok, detail)]; `setup` and `scene` are (ok, detail). No samples fails."""
    def at_most(name, value, limit):
        return name, value is not None and limit is not None and value <= limit, f"{fmt(value)} ms, limit {fmt(limit)}"
    return [("no errors", not errors, f"{len(errors)} unexpected"), ("setup running", *setup),
            at_most("spam p95", spam["p95"], limits["spam_p95_ms"]),
            at_most("spam max", spam["max"], limits["spam_max_ms"]),
            at_most("recovery p50", recovery["p50"], limits["recovery_p50_ms"]),
            at_most("recovery max", recovery["max"], limits["recovery_max_ms"]),
            ("program on last target", *scene)]


def fmt(value):
    return "-" if value is None else f"{value:.1f}"


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

    def state(self):
        code, body = request(self.url + "/api/state")
        if code != 200:
            self.errors.append(f"GET /api/state: HTTP {code} {body.get('error')}")
        return body if code == 200 else None

    def status(self):
        return (self.state() or {}).get("status", {})

    def setup(self):
        code, body = request(self.url + "/api/setup")
        return body if code == 200 else None   # 404: this web UI does not manage the mixer

    def command(self, kind, scene, **extra):
        code, body = request(self.url + "/api/command", {"command": kind, "scene": scene, **extra})
        if code != 200:
            message = f"{kind} {scene}: HTTP {code} {body.get('error')}"
            (self.rejections if is_expected_rejection(body.get("error")) else self.errors).append(message)
        return code == 200

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


def spam(run, args, scenes, payloads, mix, rng, target, settle_slack_s):
    """Send takes on an absolute schedule, one at a time so they arrive in the order sent;
    a 10 Hz poller collects every cut measured meanwhile, including ones that land late."""
    after, samples, stop = latest_id(run.status().get("cut_latency")), {}, threading.Event()

    def poll():
        while not stop.wait(0.1):
            samples.update(measured(run.status().get("cut_latency"), after))
    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    sent, last_kind = dict.fromkeys(mix, 0), "cut"
    next_at = time.monotonic()
    end = next_at + args.duration
    while next_at < end:
        time.sleep(max(0.0, next_at - time.monotonic()))
        kind = rng.choices(list(mix), list(mix.values()))[0]
        scene = next_target(scenes, target, rng)
        if run.command(kind, scene, **payloads[kind]):
            sent[kind] += 1
            target, last_kind = scene, kind
        next_at += rng.uniform(0.5, 1.5) / args.rate
    # pgm_scene flips when a fade or wipe ends, and a take may start as late as a slow cut.
    length = payloads[last_kind].get("duration_sec", UNPROBED_WIPE_S if last_kind == "wipe" else 0)
    settled = run.settle(target, max(length, 1.0) + settle_slack_s)
    stop.set()
    poller.join()
    return [samples[i]["ms"] for i in sorted(samples)], sent, target, settled


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
    limits = thresholds(fps, base["p50"], {k: getattr(args, k) for k in THRESHOLDS})
    spam_ms, sent, target, after_spam = spam(run, args, scenes, payloads, mix, rng, target,
                                             limits["spam_max_ms"] / 1000)
    time.sleep(2)
    recovery, target = spaced_cuts(run, scenes, args.recovery_cuts, rng, target)
    at_end = run.settle(target, 1.0 + limits["recovery_max_ms"] / 1000)
    scene = (after_spam[0] and at_end[0], f"after spam: {after_spam[1]}; at end: {at_end[1]}")
    spam_stats = {**summarize(spam_ms), "sent": sent,
                  "measured_ratio": len(spam_ms) / sent["cut"] if sent.get("cut") else None}
    criteria = verdict(limits, spam_stats, summarize(recovery), run.errors,
                       setup_check(setup_before, run.setup()), scene)
    return {"pass": all(ok for _, ok, _ in criteria), "seed": seed, "fps": fps, "scenes": scenes,
            "mix": mix, "rate": args.rate, "duration_s": args.duration, "thresholds": limits,
            "baseline": base, "spam": spam_stats, "recovery": summarize(recovery),
            "errors": run.errors, "rejections": run.rejections,
            "criteria": [{"name": n, "ok": ok, "detail": d} for n, ok, d in criteria]}


def print_summary(r):
    s, b, rec = r["spam"], r["baseline"], r["recovery"]
    ratio = "-" if s["measured_ratio"] is None else f"{s['measured_ratio']:.0%}"
    mix = " ".join(f"{kind}:{weight:g}" for kind, weight in r["mix"].items())
    print(f"{len(r['scenes'])} scenes at {r['fps']:g} fps, mix {mix}, "
          f"{r['duration_s']:g} s at {r['rate']:g}/s, seed {r['seed']}\n"
          f"baseline  n={b['n']} p50={fmt(b['p50'])} max={fmt(b['max'])} ms\n"
          f"spam      sent {s['sent']}; measured {s['n']}/{s['sent'].get('cut', 0)} cuts ({ratio}); "
          f"p50={fmt(s['p50'])} p95={fmt(s['p95'])} max={fmt(s['max'])} ms\n"
          f"recovery  n={rec['n']} p50={fmt(rec['p50'])} max={fmt(rec['max'])} ms\n"
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
    p.add_argument("--seed", type=int)
    p.add_argument("--fps", type=float, help="show fps, when /api/state reports no canvas")
    p.add_argument("--json", action="store_true", help="print one JSON object")
    for name in THRESHOLDS:
        p.add_argument("--" + name.replace("_", "-"), type=float, help="override the fps-derived limit")
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
