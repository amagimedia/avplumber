"""Manage one demo mixer process; the HTTP control server survives setup changes."""

from __future__ import annotations

from contextlib import suppress
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from demo_recipe import DSK_PAGES, allocate
from pyplumber.mixer.config import default_browser_ring_size

DEMO_DIR = Path(__file__).resolve().parent
# Phase markers with durations: restart timings are read from the container log.
log = logging.getLogger("setup")
# bitrate_kbps is the SDR (H.264) program bitrate. The recipe's other renditions keep their
# ratio to it, so an HDR output stays proportionally richer without a second control.
DEFAULT_BITRATE_KBPS = 6000
MIN_BITRATE_KBPS, MAX_BITRATE_KBPS = 500, 40000
# A clean mixer stop releases browser DMA-BUF frames; a killed one leaves them quarantined
# until /workers/recover restarts the browser workers. Groups stop concurrently (mixer.py
# MixerApplication.stop), so this only bounds a hung stop. compose.yaml's stop_grace_period
# covers it plus the reap and the setup worker join in close().
STOP_TIMEOUT_SEC = 60
# /workers/recover answers once the affected browser workers have restarted, concurrently:
# each stops (<= 5 s), boots Electron and reopens its pages four at a time, every open
# bounded by the worker's 10 s request timeout.
RECOVER_TIMEOUT_SEC = 180
# Waits before restarting a mixer that exited on its own (a crash or a node panic). The
# sequence starts over once a mixer has run for HEALTHY_RUN_SEC.
RETRY_DELAYS_SEC = (2, 5, 15, 60)
HEALTHY_RUN_SEC = 300
# The demo is 1080p only: every source is a unique 1920x1080 input, in either orientation.
PROGRAM_SIZE = (1920, 1080)
DEFAULT_SETTINGS = dict(orientation="portrait", fps=60, bit_depth=10, chroma="422",
                        source_count=16, scene_count=32, layout="balanced", weights=[8, 4, 2, 0, 2, 0, 0],
                        bitrate_kbps=DEFAULT_BITRATE_KBPS, browser_ring_size=default_browser_ring_size(60),
                        dsk=[], clean_feed=False)
# Measured on the T4: 110 inputs at 25/30 fps; 68 at 50/60 fps (the largest 60 fps show that
# passes the cut-spam gate, with its 3-frame deadline and the wipe cache). Keys count as inputs.
def browser_limit(fps):
    return 40   # five browser workers of eight windows (compose.yaml)


def nvdec_limit(fps):
    # 40 streams at 25 fps (1000 decoded frames/s) measured 81% NVDEC on the T4; about
    # 1100 frames/s stays near 90% at any rate, leaving room for content-dependent swings.
    return min(40, 1100 // fps)


def raw_upload_units(fps):
    return {25: 30, 30: 34}.get(fps, 17)


# Share of the per-rate total a 10-bit canvas carries, measured on the T4 at 30 fps (see
# docs/capacity.md): P010 and P210 canvases both pass the cut-spam gate at 90 sources; at 95 the
# GPU-side SDR-to-HLG work slows NVDEC to saturation and sources fall behind. The same share is
# assumed at 50/60 fps. The setup page carries the same table.
MODE_CAPACITY = {(8, "420"): 1.0, (10, "420"): 0.82, (10, "422"): 0.82}


def source_limit(fps, bit_depth=8, chroma="420"):
    # An unsupported pair (8-bit 4:2:2) is refused by the mode checks in recipe_for.
    return int((110 if fps <= 30 else 68) * MODE_CAPACITY.get((bit_depth, chroma), 1.0))


def _write_atomic(path, text):
    """A crash mid-write must not leave resume or a restart a torn show or recipe."""
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(text)
    os.replace(staged, path)


def _browser_ids(*shows):
    return {s["id"] for show in shows for s in show.get("sources", []) if s["kind"] == "browser"}


def source_counts(total, weights, fps=25, reserved_browsers=0):
    """*reserved_browsers* are downstream-key pages: browser inputs outside the weighted mix."""
    counts = allocate(total, weights)
    # P010 uses twice the upload bytes of NV12; SDR/HDR decode share NVDEC.
    for indices, costs, limit, name in (
            ((2,), (1,), 4, "HDR 4:2:2"), ((4,), (1,), browser_limit(fps) - reserved_browsers, "Browser"),
            ((0, 1), (1, 1), nvdec_limit(fps), "Combined NVDEC"),
            ((5, 6), (1, 2), raw_upload_units(fps), "Raw 4:2:0 upload units")):
        group = [(i, cost) for i, cost in zip(indices, costs) if i < len(weights)]
        if sum(counts[i] * cost for i, cost in group) > limit:
            size = min(limit, sum(counts[i] for i, _ in group))
            while True:
                capped = allocate(size, [weights[i] for i, _ in group])
                if sum(n * cost for n, (_, cost) in zip(capped, group)) <= limit:
                    break
                size -= 1
            remaining = [0 if i in indices else w for i, w in enumerate(weights)]
            if not any(remaining):
                raise ValueError(f"{name} is limited to {limit}; enable another source type")
            counts = source_counts(total - size, remaining, fps, reserved_browsers)
            for (i, _), count in zip(group, capped):
                counts[i] = count
            break
    return counts


def recipe_for(settings):
    """Accept only the bounded generic setup controls, never paths or commands."""
    if isinstance(settings, dict):
        settings = {"bit_depth": 10, "chroma": "420" if settings.get("bit_depth") == 8 else "422",
                    "bitrate_kbps": DEFAULT_BITRATE_KBPS, "browser_ring_size": default_browser_ring_size(settings.get("fps")),
                    "dsk": [], "clean_feed": False, **settings}
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError("Expected orientation, fps, source_count, scene_count, bit_depth, chroma, layout and weights")
    for key, choices in (("orientation", ("portrait", "landscape")),
                         ("fps", (25, 30, 50, 60)),
                         ("bit_depth", (8, 10)),
                         ("chroma", ("420", "422")),
                         ("layout", ("balanced", "grids", "fullscreen"))):
        if settings[key] not in choices:
            raise ValueError(f"Unsupported {key}")
    dsk = settings["dsk"]
    if (not isinstance(dsk, list) or len(set(dsk)) != len(dsk) or any(page not in DSK_PAGES for page in dsk)):
        raise ValueError(f"dsk must list distinct pages from {', '.join(DSK_PAGES)}")
    if not isinstance(settings["clean_feed"], bool) or settings["clean_feed"] and not dsk:
        raise ValueError("clean_feed must be a boolean and needs at least one dsk page")
    # Key pages are sources too: they take their share of the same budget. A show above the
    # limit of its rate and canvas is scaled down to it, not refused: switching 110 SDR inputs
    # at 30 fps to a 10-bit canvas keeps the mix at the capacity of the new mode.
    limit = source_limit(settings["fps"], settings["bit_depth"], settings["chroma"]) - len(dsk)
    if type(settings["source_count"]) is int and settings["source_count"] > limit >= 1:
        settings = {**settings, "source_count": limit}
    for key, maximum in (("source_count", limit), ("scene_count", 192), ("browser_ring_size", 64)):
        value = settings[key]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{key} must be an integer from 1 to {maximum}")
    bitrate = settings["bitrate_kbps"]
    if type(bitrate) is not int or not MIN_BITRATE_KBPS <= bitrate <= MAX_BITRATE_KBPS:
        raise ValueError(f"bitrate_kbps must be an integer from {MIN_BITRATE_KBPS} to {MAX_BITRATE_KBPS}")
    weights = settings["weights"]
    if not isinstance(weights, list) or len(weights) not in (5, 6, 7) or any(type(w) is not int or not 0 <= w <= 110 for w in weights):
        raise ValueError("Provide seven integer source weights from 0 to 110")
    weights = weights + [0] * (7 - len(weights))
    settings = {**settings, "weights": weights}
    if settings["bit_depth"] == 8 and (any(weights[1:4]) or weights[6]):
        raise ValueError("8-bit mode supports SDR 4:2:0 and browser sources only")
    if settings["bit_depth"] == 8 and settings["chroma"] != "420":
        raise ValueError("8-bit mode supports a 4:2:0 canvas only")
    if settings["chroma"] == "420" and any(weights[2:4]):
        raise ValueError("4:2:0 mode supports 4:2:0 and browser sources only")
    counts = source_counts(settings["source_count"], weights, settings["fps"], len(dsk))
    width, height = PROGRAM_SIZE
    recipe = json.loads((DEMO_DIR / "demo.example.json").read_text())
    recipe.update(source_count=settings["source_count"], scene_count=settings["scene_count"])
    recipe["browser_ring_size"] = settings["browser_ring_size"]
    # Scale every rendition by what the SDR one was asked to change by, so their relative
    # quality is preserved and the recipe's own numbers stay the reference.
    reference = recipe["renditions"][0]["bitrate_kbps"]
    for rendition in recipe["renditions"]:
        rendition["bitrate_kbps"] = max(1, round(rendition["bitrate_kbps"] * bitrate / reference))
    canvas_width, canvas_height = (height, width) if settings["orientation"] == "portrait" else (width, height)
    recipe["canvas"].update(width=canvas_width, height=canvas_height, fps=settings["fps"])
    if settings["bit_depth"] == 8:
        recipe["canvas"].update(working_format="nv12", color="sdr")
        recipe["renditions"] = [recipe["renditions"][0]]
    else:
        recipe["canvas"]["working_format"] = "p010le" if settings["chroma"] == "420" else "p210le"
    recipe["generation"].update(width=width, height=height)
    recipe["inputs"].append({"id": "sdr420_raw", "kind": "generated", "color": "sdr",
                             "chroma": "420", "storage": "nv12"})
    recipe["inputs"].append({"id": "hlg420_raw", "kind": "generated", "color": "hlg",
                             "chroma": "420", "storage": "p010"})
    for source, count in zip(recipe["inputs"], counts):
        source["weight"] = count
        if source["kind"] == "browser":
            source.update(width=width, height=height)
    layouts = {"fullscreen": 2}
    mode = settings["layout"]
    if mode != "fullscreen":
        layouts.update({f"grid_{n}": 2 if mode == "grids" else 1
                        for n in (2, 4, 8, 16, 32, 64) if n <= settings["source_count"]})
        if mode == "balanced":
            layouts.update(pip=3, random=3)
            if counts[4] and sum(counts[:4]) + sum(counts[5:]):
                layouts["alpha_overlay"] = 2
                for index in (0, 5, 1, 6, 2, 3):
                    if counts[index]:
                        source = recipe["inputs"][index]
                        recipe["alpha_background"] = source["id"] + ("_001" if counts[index] > 1 else "_000")
                        if counts[index] == 1 and index in (0, 5):
                            source["pattern"] = "bars"
                        break
    recipe["dsk"], recipe["clean_feed"] = dsk, settings["clean_feed"]
    recipe["setup"] = settings
    recipe["layouts"] = layouts
    return recipe


class SetupRuntime:
    def __init__(self, media_dir, recipe_path, bridge, mixer_args=(), browser_url="http://127.0.0.1:9009"):
        self.media_dir = Path(media_dir).resolve()
        self.media_dir.mkdir(parents=True, exist_ok=True)
        # prepare_demo stages every file in a .prepare-* directory; a killed preparation leaves it.
        for stale in self.media_dir.glob("**/.prepare-*"):
            shutil.rmtree(stale, ignore_errors=True)
        self.recipe_path = Path(recipe_path)
        self.bridge = bridge
        self.mixer_args = list(mixer_args)
        self.browser_url = browser_url
        self.lock = threading.Lock()
        # Held for a whole _stop(): a second caller must not signal a mixer that is already
        # stopping (a second SIGINT would abort its clean shutdown).
        self.stop_lock = threading.Lock()
        self.closing = threading.Event()
        self.process = None
        self.worker = None
        self.retries = 0
        self.retry_timer = None
        self.phase = "idle"
        self.message = "Choose settings and apply to start the mixer."
        self.settings = None
        # A restarted setup server must also invalidate existing control/player tabs.
        self.revision = time.time_ns() // 1_000_000

    def status(self):
        with self.lock:
            if self.phase == "running" and self.process and self.process.poll() is not None:
                self.phase, self.message = "error", f"Mixer exited ({self.process.returncode}). Apply to retry."
            return dict(phase=self.phase, message=self.message, settings=self.settings, revision=self.revision)

    def _status(self, phase, message):
        with self.lock:
            self.phase, self.message = phase, message

    def resume(self):
        if self.recipe_path.exists():
            self.apply()
            return
        # Adopt an existing explicit show when adding setup controls to a demo.
        def start():
            try:
                self._start_recovering(self.media_dir / "mixer.demo.json", None)
                self._status("running", "Existing mixer ready. Apply replaces it with the selected generic sources.")
                log.info("Mixer ready (existing show)")
            except Exception as exc:
                self._status("error", str(exc))
        self._status("starting", "Starting existing mixer…")
        self.worker = threading.Thread(target=start, daemon=True)
        self.worker.start()

    def _check_idle(self):
        """Call with self.lock held."""
        if self.closing.is_set() or (self.worker and self.worker.is_alive()):
            raise RuntimeError("A setup change is already in progress")

    def apply(self, settings=None):
        # Before _preserve_aux asks the live mixer, which a running change may be stopping.
        with self.lock:
            self._check_idle()
        recipe = recipe_for(settings) if settings is not None else json.loads(self.recipe_path.read_text())
        # Plan validation happens before stopping the live mixer or writing files.
        from prepare_demo import plan
        show, _, _ = plan(recipe, self.media_dir)
        if settings is not None:
            self._preserve_aux(recipe, show)
            plan(recipe, self.media_dir)
        with self.lock:
            self._check_idle()
            self._cancel_retry()
            self.phase, self.message = "preparing", "Preparing assets…"
            self.worker = threading.Thread(target=self._apply, args=(recipe, settings), daemon=True)
            self.worker.start()

    def _preserve_aux(self, recipe, show):
        """Keep instance outputs while replacing the generic sources and scenes."""
        from pyplumber.mixer.aux import aux_fps, validate_assignments
        from pyplumber.mixer.config import ConfigError, parse
        config = self.media_dir / "mixer.demo.json"
        previous = json.loads(config.read_text()) if config.exists() else {}
        if "max_compositor_layers" in previous:
            recipe["max_compositor_layers"] = show["max_compositor_layers"] = previous["max_compositor_layers"]
        buses = previous.get("aux_buses", [])
        if not buses:
            return
        live = {}
        if self.process and self.process.poll() is None:
            live = {b["id"]: b["scenes"] for b in json.loads(self.bridge.command("mixer.aux_status")) if "scenes" in b}
        cfg = parse(show)
        scene_ids = {s.id for s in cfg.scenes}
        for bus in buses:
            for rendition in bus["renditions"]:
                rendition.update(width=cfg.canvas_w, height=cfg.canvas_h, fps=aux_fps(cfg.fps, bus.get("full_rate", False)))
            if bus.get("layout", {}).get("preset") == "source_pages":
                continue   # pages follow the new source list by themselves
            assignments = [None] * 8
            for i, scene in enumerate(live.get(bus["id"], bus.get("scenes", [None] * 8))):
                if scene not in scene_ids:
                    continue
                assignments[i] = scene
                try:
                    validate_assignments(cfg, assignments)
                except ConfigError:
                    # A retained scene may have grown beyond the tile draw budget.
                    assignments[i] = None
            bus["scenes"] = assignments
        recipe["aux_buses"] = buses

    def remember_aux(self, bus_id, scenes):
        """Persist an operator's multiview assignment, so a resume or restart shows the same tiles."""
        with self.lock:   # an Apply carries live assignments over itself (_preserve_aux)
            if self.worker and self.worker.is_alive():
                return
            for path in (self.media_dir / "mixer.demo.json", self.recipe_path):
                doc = json.loads(path.read_text()) if path.exists() else {}
                bus = next((b for b in doc.get("aux_buses", []) if b.get("id") == bus_id), None)
                if bus is not None and bus.get("scenes") != scenes:
                    bus["scenes"] = scenes
                    _write_atomic(path, json.dumps(doc, indent=2) + "\n")

    def _stop(self):
        with self.stop_lock:
            process = self.process
            if process and process.poll() is None:
                started = time.monotonic()
                log.info("Stopping mixer process %s", process.pid)
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=STOP_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    log.warning("Mixer shutdown timed out; killing process before browser recovery")
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                log.info("Mixer process exited (%s) in %.1f s", process.returncode, time.monotonic() - started)
            self.process = None

    def _recover_browsers(self, shows):
        from pyplumber.mixer.dmabuf_inputs import rest_request
        if self.process and self.process.poll() is None:
            raise RuntimeError("Cannot recover browser buffers while the mixer is running")
        ids = sorted(_browser_ids(*shows))
        if not ids:
            return False
        status = rest_request(self.browser_url, "GET", "/status") or {}
        if not any(w["id"] in ids and w.get("stats", {}).get("quarantinedFrameCount", 0)
                   for w in status.get("windows", [])):
            return False
        self._status("starting", "Recovering browser workers…")
        started = time.monotonic()
        log.info("Recovering browser workers with quarantined frames")
        rest_request(self.browser_url, "POST", "/workers/recover", {"ids": ids}, timeout=RECOVER_TIMEOUT_SEC)
        log.info("Browser workers recovered in %.1f s", time.monotonic() - started)
        return True

    def _start_recovering(self, config, previous_show):
        shows = [json.loads(config.read_bytes()), previous_show or {}]
        self._recover_browsers(shows)
        self._close_removed_browsers(shows[1], shows[0])
        try:
            self._start(config)
        except Exception:
            self._stop()
            # Disconnect/quarantine notification can arrive after the first
            # status check. Retry only when a worker was actually recovered.
            if self.closing.is_set() or not self._recover_browsers(shows):
                raise
            self._start(config)
        # Every mixer that reached air is watched, a restored previous show included.
        threading.Thread(target=self._watch, args=(self.process,), daemon=True).start()

    def _watch(self, process):
        started = time.monotonic()
        code = process.wait()
        with self.stop_lock:   # a deliberate _stop() replaces self.process before releasing it
            if self.process is not process or self.closing.is_set():
                return
        with self.lock:
            if time.monotonic() - started >= HEALTHY_RUN_SEC:
                self.retries = 0
        self._schedule_retry(f"Mixer exited ({code})")

    def _schedule_retry(self, reason):
        with self.lock:
            others = self.worker and self.worker.is_alive() and self.worker is not threading.current_thread()
            if self.closing.is_set() or others:
                return   # shutting down, or a setup change owns the mixer now
            delay = RETRY_DELAYS_SEC[min(self.retries, len(RETRY_DELAYS_SEC) - 1)]
            self.retries += 1
            self.phase, self.message = "error", f"{reason}; restarting in {delay} s. Apply restarts it now."
            log.warning("%s; restarting in %d s", reason, delay)
            timer = threading.Timer(delay, lambda: self._retry(timer))
            timer.daemon = True
            self.retry_timer = timer
            timer.start()

    def _cancel_retry(self):
        """Call with self.lock held."""
        if self.retry_timer:
            self.retry_timer.cancel()
        self.retry_timer = None

    def _retry(self, timer):
        with self.lock:
            if self.retry_timer is not timer or self.closing.is_set() or (self.worker and self.worker.is_alive()):
                return   # cancelled, superseded or shutting down
            self.retry_timer = None
            self.phase, self.message = "starting", "Restarting mixer…"
            self.worker = threading.Thread(target=self._restart, daemon=True)
            self.worker.start()

    def _restart(self):
        started = time.monotonic()
        try:
            self._start_recovering(self.media_dir / "mixer.demo.json", None)
            with self.lock:
                self.revision += 1
                self.phase, self.message = "running", "Mixer ready (restarted automatically)."
            log.info("Mixer ready: restarted in %.1f s", time.monotonic() - started)
        except Exception as exc:
            self._stop()
            self._schedule_retry(f"Mixer restart failed: {exc}")

    def _start(self, config):
        started = time.monotonic()
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(DEMO_DIR / "mixer.py"), "--config", str(config),
             "--remote-control-port", str(self.bridge.port), "--dmabuf-rest", self.browser_url,
             *self.mixer_args], start_new_session=True)
        log.info("Mixer process %s started", self.process.pid)
        deadline = time.monotonic() + 180
        last_problem = "waiting for mixer control"
        while not self.closing.wait(.5):
            if self.process.poll() is not None:
                raise RuntimeError(f"Mixer exited during startup ({self.process.returncode}); see container logs")
            try:
                state = self.bridge.state(timeout=30.0)
                if not state.get("status", {}).get("pgm_scene"):
                    last_problem = "waiting for a program scene"
                else:
                    # queues.json carries every edge (~160 KB at 48 sources) on one line, and
                    # the mixer answers it slowly while it is still filling its decoders, so
                    # give it far more than the default command budget.
                    queues = json.loads(self.bridge.command("queues.json", timeout=60.0) or "[]")
                    expected = {"janus_encoded"}
                    if "h265" in state.get("settings", {}).get("preview_codecs", []):
                        expected.add("janus_hdr_encoded")
                    expected.update(f"aux_{bid}_encoded" for bid in state.get("settings", {}).get("aux_buses", []))
                    encoded = {q["name"] for q in queues if q["enqueued_total"] > 0}
                    if expected <= encoded:
                        log.info("Mixer encoding in %.1f s", time.monotonic() - started)
                        return
                    last_problem = "waiting for encoded output"
            except Exception as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
            if time.monotonic() > deadline:
                raise RuntimeError(f"Mixer startup timed out: {last_problem}")
        raise RuntimeError("Setup server is stopping")

    def _close_removed_browsers(self, old_show, new_show):
        from pyplumber.mixer.dmabuf_inputs import rest_request
        removed = _browser_ids(old_show) - _browser_ids(new_show)
        if removed:
            status = rest_request(self.browser_url, "GET", "/status") or {}
            existing = {w["id"] for w in status.get("windows", [])}
            for name in sorted(removed & existing):
                rest_request(self.browser_url, "POST", "/window/close", {"id": name})

    def _apply(self, recipe, settings):
        from prepare_demo import prepare
        config = self.media_dir / "mixer.demo.json"
        had_previous = config.exists()
        previous = None
        stopped = False
        recipe_show = {}
        started = time.monotonic()
        try:
            previous = config.read_bytes() if had_previous else None
            # Complete missing assets while the old mixer continues to run.
            log.info("Preparing assets")
            prepare(recipe, self.media_dir)
            log.info("Assets ready in %.1f s", time.monotonic() - started)
            if self.closing.is_set():
                raise RuntimeError("Setup server is stopping")
            self._status("starting", "Restarting mixer…")
            self._stop()
            stopped = True
            recipe_show = json.loads(config.read_bytes())
            self._start_recovering(config, json.loads(previous or b"{}"))
            if settings is not None:
                _write_atomic(self.recipe_path, json.dumps(recipe, indent=2) + "\n")
            with self.lock:
                self.settings = recipe.get("setup")
                self.revision += 1
                self.retries = 0
                self.phase, self.message = "running", "Mixer ready."
            log.info("Mixer ready: setup applied in %.1f s", time.monotonic() - started)
        except Exception as exc:
            message = str(exc)
            try:
                if stopped:
                    self._stop()
                if previous is not None:
                    _write_atomic(config, previous.decode())
                    if stopped and isinstance(exc, TimeoutError):
                        # The browser service may still be restarting workers; restoring now would
                        # recover them a second time. The next Apply starts from a settled service.
                        message += "; the browser service is still busy, apply again when it settles."
                    elif stopped and not self.closing.is_set():
                        self._start_recovering(config, recipe_show)
                        with self.lock:
                            self.revision += 1
                        message += "; previous setup restored."
                elif not had_previous:
                    config.unlink(missing_ok=True)
            except Exception as rollback:
                message += f"; recovery failed: {rollback}"
            self._status("error", message)
            log.error("Setup failed after %.1f s: %s", time.monotonic() - started, message)

    def close(self):
        """Stop the mixer cleanly first: the container's stop grace period is spent on it,
        not on setup work that `closing` already cancels."""
        self.closing.set()
        with self.lock:
            self._cancel_retry()
        self._stop()
        if self.worker:
            self.worker.join(timeout=20)
        self._stop()   # a mixer the cancelled setup work started meanwhile
