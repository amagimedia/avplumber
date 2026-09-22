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

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS = dict(resolution="1920x1080", orientation="portrait", fps=60, bit_depth=10, chroma="422",
                        source_count=16, scene_count=32, layout="balanced", weights=[8, 4, 2, 0, 2])


def source_counts(total, weights):
    counts = allocate(total, weights)
    if counts[2] > 4:
        remaining = [w if i != 2 else 0 for i, w in enumerate(weights)]
        if not any(remaining):
            raise ValueError("HDR 4:2:2 is limited to four sources; enable another source type")
        counts = allocate(total - 4, remaining)
        counts[2] = 4
    return counts


def recipe_for(settings):
    """Accept only the bounded generic setup controls, never paths or commands."""
    if isinstance(settings, dict):
        settings = {"bit_depth": 10, "chroma": "420" if settings.get("bit_depth") == 8 else "422", **settings}
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
    source_limit = 32 if settings["fps"] in (50, 60) else 64
    for key, maximum in (("source_count", source_limit), ("scene_count", 128)):
        value = settings[key]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{key} must be an integer from 1 to {maximum}")
    weights = settings["weights"]
    if not isinstance(weights, list) or len(weights) != 5 or any(type(w) is not int or not 0 <= w <= 100 for w in weights):
        raise ValueError("Provide five integer source weights from 0 to 100")
    if settings["bit_depth"] == 8 and any(weights[1:4]):
        raise ValueError("8-bit mode supports SDR 4:2:0 and browser sources only")
    if settings["bit_depth"] == 8 and settings["chroma"] != "420":
        raise ValueError("8-bit mode supports a 4:2:0 canvas only")
    if settings["chroma"] == "420" and any(weights[2:4]):
        raise ValueError("4:2:0 mode supports 4:2:0 and browser sources only")
    counts = source_counts(settings["source_count"], weights)
    width, height = map(int, settings["resolution"].split("x"))
    recipe = json.loads((DEMO_DIR / "demo.example.json").read_text())
    recipe.update(source_count=settings["source_count"], scene_count=settings["scene_count"])
    canvas_width, canvas_height = (height, width) if settings["orientation"] == "portrait" else (width, height)
    recipe["canvas"].update(width=canvas_width, height=canvas_height, fps=settings["fps"])
    if settings["bit_depth"] == 8:
        recipe["canvas"].update(working_format="nv12", color="sdr")
        recipe["renditions"] = [recipe["renditions"][0]]
    else:
        recipe["canvas"]["working_format"] = "p010le" if settings["chroma"] == "420" else "p210le"
    recipe["generation"].update(width=width, height=height)
    recipe["inputs"] = [source for source in recipe["inputs"] if source["kind"] != "download"]
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
            if counts[4] and sum(counts[:4]):
                layouts["alpha_overlay"] = 2
                if counts[0]:
                    recipe["alpha_background"] = "sdr420_001" if counts[0] > 1 else "sdr420_000"
                    if counts[0] == 1:
                        recipe["inputs"][0]["pattern"] = "bars"
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
        plan(recipe, self.media_dir)
        with self.lock:
            if self.closing.is_set() or (self.worker and self.worker.is_alive()):
                raise RuntimeError("A setup change is already in progress")
            self.phase, self.message = "preparing", "Preparing assets…"
            self.worker = threading.Thread(target=self._apply, args=(recipe, settings), daemon=True)
            self.worker.start()

    def _stop(self):
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=15)
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
                state = self.bridge.state()
                queues = json.loads(self.bridge.command("queues.json") or "[]")
                encoded = [q for q in queues if q["name"] in ("janus_encoded", "janus_hdr_encoded")]
                if state.get("status", {}).get("pgm_scene") and encoded and all(q["enqueued_total"] > 0 for q in encoded):
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
