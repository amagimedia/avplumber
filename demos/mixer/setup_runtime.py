"""Manage one demo mixer process; the HTTP control server survives setup changes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from demo_recipe import allocate
from pyplumber.mixer.config import default_browser_ring_size

DEMO_DIR = Path(__file__).resolve().parent
# bitrate_kbps is the SDR (H.264) program bitrate. The recipe's other renditions keep their
# ratio to it, so an HDR output stays proportionally richer without a second control.
DEFAULT_BITRATE_KBPS = 6000
MIN_BITRATE_KBPS, MAX_BITRATE_KBPS = 500, 40000
DEFAULT_SETTINGS = dict(resolution="1920x1080", orientation="portrait", fps=60, bit_depth=10, chroma="422",
                        source_count=16, scene_count=32, layout="balanced", weights=[8, 4, 2, 0, 2, 0, 0],
                        bitrate_kbps=DEFAULT_BITRATE_KBPS, browser_ring_size=default_browser_ring_size(60))


def source_counts(total, weights, fps=25):
    counts = allocate(total, weights)
    # P010 uses twice the upload bytes of NV12; SDR/HDR decode share NVDEC.
    for indices, costs, limit, name in (
            ((2,), (1,), 4, "HDR 4:2:2"), ((4,), (1,), 32, "Browser"),
            ((0, 1), (1, 1), 40 if fps <= 30 else 20, "Combined NVDEC"),
            ((5, 6), (1, 2), min(28, 700 // fps), "Raw 4:2:0 upload units")):
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
            counts = source_counts(total - size, remaining, fps)
            for (i, _), count in zip(group, capped):
                counts[i] = count
            break
    return counts


def recipe_for(settings):
    """Accept only the bounded generic setup controls, never paths or commands."""
    if isinstance(settings, dict):
        settings = {"bit_depth": 10, "chroma": "420" if settings.get("bit_depth") == 8 else "422",
                    "bitrate_kbps": DEFAULT_BITRATE_KBPS, "browser_ring_size": default_browser_ring_size(settings.get("fps")), **settings}
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError("Expected resolution, orientation, fps, source_count, scene_count, bit_depth, chroma, layout and weights")
    for key, choices in (("resolution", ("1920x1080", "1280x720")),
                         ("orientation", ("portrait", "landscape")),
                         ("fps", (25, 30, 50, 60)),
                         ("bit_depth", (8, 10)),
                         ("chroma", ("420", "422")),
                         ("layout", ("balanced", "grids", "fullscreen"))):
        if settings[key] not in choices:
            raise ValueError(f"Unsupported {key}")
    # Higher rates use the 100-at-25-fps baseline; 25 fps retains the experimental 110-input ceiling.
    source_limit = 110 if settings["fps"] == 25 else 2500 // settings["fps"]
    for key, maximum in (("source_count", source_limit), ("scene_count", 192), ("browser_ring_size", 64)):
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
    counts = source_counts(settings["source_count"], weights, settings["fps"])
    width, height = map(int, settings["resolution"].split("x"))
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
    recipe["inputs"] = [source for source in recipe["inputs"] if source["kind"] != "download"]
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
    recipe["setup"] = settings
    recipe["layouts"] = layouts
    return recipe


class SetupRuntime:
    def __init__(self, media_dir, recipe_path, bridge, mixer_args=(), browser_url="http://127.0.0.1:9009"):
        self.media_dir = Path(media_dir).resolve()
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.recipe_path = Path(recipe_path)
        self.bridge = bridge
        self.mixer_args = list(mixer_args)
        self.browser_url = browser_url
        self.lock = threading.Lock()
        self.closing = threading.Event()
        self.process = None
        self.worker = None
        self.phase = "idle"
        self.message = "Choose settings and apply to start the mixer."
        self.settings = None
        # A restarted setup server must also invalidate existing control/player tabs.
        self.revision = time.time_ns() // 1_000_000

    def status(self):
        with self.lock:
            if self.phase == "running" and self.process.poll() is not None:
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
                self._start(self.media_dir / "mixer.demo.json")
                self._status("running", "Existing mixer ready. Apply replaces it with the selected generic sources.")
            except Exception as exc:
                self._status("error", str(exc))
        self._status("starting", "Starting existing mixer…")
        self.worker = threading.Thread(target=start, daemon=True)
        self.worker.start()

    def apply(self, settings=None):
        recipe = recipe_for(settings) if settings is not None else json.loads(self.recipe_path.read_text())
        # Plan validation happens before stopping the live mixer or writing files.
        from prepare_demo import plan
        show, _, _ = plan(recipe, self.media_dir)
        if settings is not None:
            self._preserve_aux(recipe, show)
            plan(recipe, self.media_dir)
        with self.lock:
            if self.closing.is_set() or (self.worker and self.worker.is_alive()):
                raise RuntimeError("A setup change is already in progress")
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
            live = {b["id"]: b["scenes"] for b in json.loads(self.bridge.command("mixer.aux_status"))}
        cfg = parse(show)
        scene_ids = {s.id for s in cfg.scenes}
        for bus in buses:
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
            for rendition in bus["renditions"]:
                rendition.update(width=cfg.canvas_w, height=cfg.canvas_h, fps=aux_fps(cfg.fps))
        recipe["aux_buses"] = buses

    def _stop(self):
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                # Large graphs stop input groups serially; killing them early
                # leaves browser DMA-BUF frames quarantined and blocks restart.
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.process = None

    def _start(self, config):
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(DEMO_DIR / "mixer.py"), "--config", str(config),
             "--remote-control-port", str(self.bridge.port), "--dmabuf-rest", self.browser_url,
             *self.mixer_args], start_new_session=True)
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
                        return
                    last_problem = "waiting for encoded output"
            except Exception as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
            if time.monotonic() > deadline:
                raise RuntimeError(f"Mixer startup timed out: {last_problem}")
        raise RuntimeError("Setup server is stopping")

    def _close_removed_browsers(self, old_show, new_show):
        from pyplumber.mixer.dmabuf_inputs import rest_request
        old_ids = {s["id"] for s in old_show.get("sources", []) if s["kind"] == "browser"}
        new_ids = {s["id"] for s in new_show.get("sources", []) if s["kind"] == "browser"}
        if old_ids - new_ids:
            status = rest_request(self.browser_url, "GET", "/status") or {}
            existing = {w["id"] for w in status.get("windows", [])}
            for name in sorted((old_ids - new_ids) & existing):
                rest_request(self.browser_url, "POST", "/window/close", {"id": name})

    def _apply(self, recipe, settings):
        from prepare_demo import prepare
        config = self.media_dir / "mixer.demo.json"
        had_previous = config.exists()
        previous = None
        stopped = False
        recipe_show = {}
        try:
            previous = config.read_bytes() if had_previous else None
            # Complete missing assets while the old mixer continues to run.
            prepare(recipe, self.media_dir)
            if self.closing.is_set():
                raise RuntimeError("Setup server is stopping")
            self._status("starting", "Restarting mixer…")
            self._stop()
            stopped = True
            recipe_show = json.loads(config.read_bytes())
            self._close_removed_browsers(json.loads(previous or b"{}"), recipe_show)
            self._start(config)
            if settings is not None:
                staged = self.recipe_path.with_suffix(".pending.json")
                staged.write_text(json.dumps(recipe, indent=2) + "\n")
                staged.replace(self.recipe_path)
            with self.lock:
                self.settings = recipe.get("setup")
                self.revision += 1
                self.phase, self.message = "running", "Mixer ready."
        except Exception as exc:
            message = str(exc)
            try:
                if stopped:
                    self._stop()
                    self._close_removed_browsers(recipe_show, json.loads(previous or b"{}"))
                if previous is not None:
                    config.write_bytes(previous)
                    if stopped and not self.closing.is_set():
                        self._start(config)
                        with self.lock:
                            self.revision += 1
                        message += "; previous setup restored."
                elif not had_previous:
                    config.unlink(missing_ok=True)
            except Exception as rollback:
                message += f"; recovery failed: {rollback}"
            self._status("error", message)

    def close(self):
        self.closing.set()
        if self.worker:
            self.worker.join(timeout=20)
        self._stop()
