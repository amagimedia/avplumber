#!/usr/bin/env python3
"""
Avplumber Mixer TUI -- live production video switcher.

Scene composition (graphs, PiP positions) is defined on the avplumber side with
``mixer.scene`` JSON using a ``sources`` map: logical camera name -> ``graph``
plus ``dst_x`` / ``dst_y`` (see doc/mixer_orchestrator.md). This UI only refers
to scenes by name.

Scene names are fetched automatically from the server via ``mixer.scenes`` on
connect and refreshed every 10 seconds. No scene names need to be passed as
arguments. Selecting a preview scene preloads that scene into avplumber's
hidden PVW slot so a later CUT can take it without cold-loading the graph.

Usage:
    python tools/mixer_tui.py --host localhost --port 5555 --mixer mixer

Keyboard shortcuts:
    1-9        Select scene N on Preview bus, or direct cut when Direct is ON
    F1-F9      Direct CUT to scene N (skips preview step)
    t          Toggle direct scene-click cuts
    a          Cycle AI auto-switch transitions through CUT / FADE / WIPE
    o          Toggle HTML overlay on/off
    c          CUT  (take preview to program, hard cut)
    x          X-FADE (crossfade preview to program at set frame count)
    w          WIPE (uses --wipe-file path if set, otherwise prompts)
    e          Select/change wipe file path
    d          Focus the fade frame input field
    s          Force an immediate mixer.status refresh
    q / ctrl+c Quit

The Textual keys / shortcuts panel and footer legend are hidden by default.
"""

import argparse
import asyncio
import os
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Header,
    Input,
    Label,
    Static,
)

from avp_client import AvpConnection
from commands import (
    AI_TRANSITION_MODES,
    TAKE_COMMAND_PREFIXES,
    auto_switch_set_command,
    mixer_command,
    overlay_toggle_commands,
)
from state import parse_auto_switch_status, parse_mixer_status, parse_scene_list
from widgets import FadeFramesInput, SceneButton, WipeModal


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

OVERLAY_SETTLE_SECONDS = 0.2
DEFAULT_FADE_FPS = 30.0
DEFAULT_FADE_FRAMES = 15
MAX_FADE_FRAMES = 300

TRANSITION_SYMBOLS = {
    "idle": "─",
    "cut": "✂",
    "crossfade": "⟿",
    "wipe": "▶",
}

APP_CSS = """
Screen {
    background: $surface-darken-1;
}

/* ── Top bus row ── */
#bus_row {
    height: 9;
    margin: 0 0 1 0;
}

#pgm_panel {
    width: 1fr;
    height: 100%;
    border: heavy $error;
    background: $error-darken-3;
    align: center middle;
    content-align: center middle;
    padding: 0 2;
}

#pvw_panel {
    width: 1fr;
    height: 100%;
    border: heavy $success;
    background: $success-darken-3;
    align: center middle;
    content-align: center middle;
    padding: 0 2;
}

#pgm_panel.transitioning {
    border: heavy $warning;
    background: $warning-darken-3;
}

#pvw_panel.transitioning {
    border: heavy $warning;
    background: $warning-darken-3;
}

#pgm_label {
    color: $error-lighten-1;
    text-style: bold;
    width: 100%;
    content-align: center middle;
    text-align: center;
}

#pvw_label {
    color: $success-lighten-1;
    text-style: bold;
    width: 100%;
    content-align: center middle;
    text-align: center;
}

#pgm_scene_name {
    text-style: bold;
    width: 100%;
    content-align: center middle;
    text-align: center;
    color: $text;
}

#pvw_scene_name {
    text-style: bold;
    width: 100%;
    content-align: center middle;
    text-align: center;
    color: $text;
}

#pgm_sub {
    color: $error-lighten-2;
    width: 100%;
    content-align: center middle;
    text-align: center;
}

#pvw_sub {
    color: $success-lighten-2;
    width: 100%;
    content-align: center middle;
    text-align: center;
}

/* ── Scene row ── */
#scenes_container {
    height: 7;
    background: $panel;
    border: solid $primary-darken-2;
    align: left middle;
    padding: 0 1;
    margin-bottom: 1;
    overflow-x: auto;
}

/* ── Control bars ── */
#transition_bar {
    height: 5;
    background: $panel;
    border: solid $primary-darken-2;
    align: left middle;
    padding: 0 1;
    margin-bottom: 0;
}

#utility_bar {
    height: 5;
    background: $panel;
    border: solid $primary-darken-2;
    align: left middle;
    padding: 0 1;
    margin-top: 0;
    margin-bottom: 0;
}

#trans_mode_label {
    width: auto;
    min-width: 22;
    color: $text-muted;
    content-align: left middle;
}

#trans_mode_label.busy {
    color: $warning;
    text-style: bold;
}

#dur_label {
    width: auto;
    content-align: left middle;
    margin-left: 1;
    margin-right: 1;
    color: $text-muted;
}

#dur_input {
    width: 16;
    min-width: 16;
    max-width: 16;
    margin-right: 1;
}

#btn_ai_transition {
    margin-right: 1;
}

#btn_cut {
    margin-right: 1;
}

#btn_auto {
    margin-right: 1;
}

#btn_wipe {
    margin-right: 1;
}

#btn_change_wipe {
    margin-left: 0;
    margin-right: 1;
}

#btn_direct {
    margin-left: 0;
    margin-right: 1;
}

#btn_overlay {
    margin-left: 1;
}

/* ── Connection status ── */
#conn_status {
    height: 1;
    background: $error-darken-2;
    color: $error-lighten-1;
    content-align: center middle;
    text-align: center;
    display: none;
}

#conn_status.visible {
    display: block;
}
"""


class MixerTUI(App):
    """Live production video switcher TUI for avplumber."""

    TITLE = "Avplumber Mixer Switcher"
    CSS = APP_CSS

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("c", "cut", "CUT", show=False),
        Binding("x", "auto_transition", "X-FADE", show=False),
        Binding("w", "wipe", "WIPE", show=False),
        Binding("e", "change_wipe_path", "Wipe path", show=False),
        Binding("d", "focus_duration", "Fade frames", show=False),
        Binding("a", "toggle_ai_transition_mode", "AI transition", show=False, priority=True),
        Binding("t", "toggle_direct_cut_mode", "Direct cuts", show=False),
        Binding("o", "toggle_overlay", "Overlay", show=False),
        Binding("s", "refresh_status", "Refresh", show=False),
        # F1-F9: direct cut to scene (bound dynamically in on_key)
    ]

    # Reactive state from mixer.status
    pgm_scene: reactive[str] = reactive("", layout=True)
    pvw_scene_remote: reactive[str] = reactive("")
    transition_mode: reactive[str] = reactive("idle")

    # Scene list — fetched from the server; drives the scene button row.
    scenes: reactive[list] = reactive(list, layout=True)

    # Local PVW selection (TD's intent)
    pvw_selected: reactive[int] = reactive(-1, layout=True)
    overlay_enabled: reactive[bool] = reactive(False, layout=True)
    direct_cut_mode: reactive[bool] = reactive(False, layout=True)
    auto_control_connected: reactive[bool] = reactive(False, layout=True)
    auto_transition_mode: reactive[str] = reactive("n/a", layout=True)

    connected: reactive[bool] = reactive(False)

    def __init__(self, host: str, port: int, mixer: str,
                 overlay_otm: str,
                 overlay_source_otm: str,
                 overlay_selector: str,
                 wipe_file: Optional[str] = None,
                 wipe_dir: Optional[str] = None,
                 fade_fps: float = DEFAULT_FADE_FPS,
                 fade_frames: int = DEFAULT_FADE_FRAMES) -> None:
        super().__init__()
        self.avp_host = host
        self.avp_port = port
        self.fade_fps = fade_fps if fade_fps > 0 else DEFAULT_FADE_FPS
        self.default_fade_frames = min(MAX_FADE_FRAMES, max(1, int(fade_frames)))
        self.mixer_name = mixer
        self.overlay_otm_name = overlay_otm
        self.overlay_source_otm_name = overlay_source_otm
        self.overlay_selector_name = overlay_selector
        self.wipe_file: Optional[str] = wipe_file
        self.wipe_dir: Optional[str] = wipe_dir
        self._conn = AvpConnection(host, port)
        self._pending_action: Optional[str] = None  # "cut" | "auto" | "wipe:<path>"
        self._auto_poll_counter = 0
        self._auto_duration_initialized = False
        self._overlay_toggle_in_progress = False
        self._connection_timer = None
        self._poll_timer = None
        self._scene_timer = None

    # ── Layout ──────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()

        yield Static("", id="conn_status")

        with Horizontal(id="bus_row"):
            with Vertical(id="pgm_panel"):
                yield Label("◉  PROGRAM  ◉", id="pgm_label")
                yield Static("", id="pgm_scene_name")
                yield Static("ON AIR", id="pgm_sub")
            with Vertical(id="pvw_panel"):
                yield Label("●  PREVIEW  ●", id="pvw_label")
                yield Static("", id="pvw_scene_name")
                yield Static("READY", id="pvw_sub")

        with Horizontal(id="scenes_container"):
            pass  # populated dynamically by watch_scenes

        with Horizontal(id="transition_bar"):
            yield Static("Mode: idle", id="trans_mode_label")
            yield Button("✂ CUT", id="btn_cut", variant="error")
            yield Button("⟿ X-FADE", id="btn_auto", variant="success")
            yield Button("▶ MEDIA WIPE", id="btn_wipe", variant="primary")
            yield Static("Fade frames:", id="dur_label")
            yield FadeFramesInput(str(self.default_fade_frames), id="dur_input")

        with Horizontal(id="utility_bar"):
            yield Button("Direct: OFF", id="btn_direct", variant="default")
            yield Button("AI: n/a", id="btn_ai_transition", variant="default")
            yield Button("⚙ Wipe file…", id="btn_change_wipe", variant="default")
            yield Button("Overlay: OFF", id="btn_overlay", variant="warning")

    # ── On mount ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self._connection_timer = self.set_interval(0.5, self._connection_tick, name="mixer_avp_connect")
        self._poll_timer = self.set_interval(0.5, self._poll_status_tick, name="mixer_avp_poll")
        self._scene_timer = self.set_interval(10.0, self._fetch_scenes_tick, name="mixer_avp_scenes")
        self.run_worker(
            self._connection_tick(),
            group="mixer_avp_connect_now",
            exclusive=True,
            exit_on_error=False,
        )
        self.call_after_refresh(self._clear_initial_focus)
        self._update_wipe_button_tooltip()
        self._refresh_ai_transition_button()

    # ── Connection & polling workers ────────────────────────────────────────

    def _clear_initial_focus(self) -> None:
        self.set_focus(None)

    async def _connection_tick(self) -> None:
        """Connect or reconnect the shared control socket."""
        if self._conn.connected:
            self.connected = True
            return
        try:
            ok = await self._conn.ensure_connected()
        except Exception:
            ok = False
        self.connected = ok
        if ok:
            self._set_status_bar(False)
            self.auto_control_connected = True
            if self.auto_transition_mode == "n/a":
                self.auto_transition_mode = "cut"
            await self._fetch_scenes()
            await self._refresh_once()
            await self._refresh_auto_status_tick()
        else:
            self._set_status_bar(True, f"Connecting to {self.avp_host}:{self.avp_port}…")

    async def _fetch_scenes(self) -> None:
        """Fetch scene list from the server and update the reactive."""
        resp = await self._conn.command(f"mixer.scenes {self.mixer_name}", raise_for_status=False)
        if resp is None or resp.code != 201 or not resp.content:
            return
        try:
            scenes = parse_scene_list(resp.content)
            if isinstance(scenes, list) and scenes != self.scenes:
                self.scenes = scenes
        except Exception:
            pass

    async def _fetch_scenes_tick(self) -> None:
        if not self._conn.connected:
            return
        await self._fetch_scenes()

    async def _poll_status_tick(self) -> None:
        """Poll mixer.status."""
        if not self._conn.connected:
            return
        try:
            self._auto_poll_counter += 1
            if self._auto_poll_counter >= 2:  # every ~1 s
                self._auto_poll_counter = 0
                await self._refresh_auto_status_tick()

            resp = await self._conn.command(f"mixer.status {self.mixer_name}", raise_for_status=False)
            if resp is None:
                self.connected = False
                self._set_status_bar(True, f"Lost connection to {self.avp_host}:{self.avp_port}")
                return
            if resp.code == 201 and resp.content:
                try:
                    status = parse_mixer_status(resp.content)
                    self.pgm_scene = status.pgm_scene
                    self.pvw_scene_remote = status.pvw_scene
                    self.transition_mode = status.transition
                except Exception:
                    pass
        except Exception:
            self.connected = False
            self._set_status_bar(True, f"Lost connection to {self.avp_host}:{self.avp_port}")

    async def _refresh_auto_status_tick(self) -> None:
        try:
            await self._refresh_auto_status()
        except Exception:
            pass

    # ── Reactive watchers ────────────────────────────────────────────────────

    def watch_pgm_scene(self, value: str) -> None:
        self.query_one("#pgm_scene_name", Static).update(value or "(none)")
        self._refresh_scene_buttons()

    def watch_pvw_selected(self, value: int) -> None:
        name = self.scenes[value] if 0 <= value < len(self.scenes) else ""
        self.query_one("#pvw_scene_name", Static).update(name or "(none)")
        self._refresh_scene_buttons()
        if name:
            self._preview_scene(name)

    def watch_scenes(self, scenes: list) -> None:
        try:
            container = self.query_one("#scenes_container")
        except NoMatches:
            return
        for btn in list(container.query(SceneButton)):
            btn.remove()
        for i, name in enumerate(scenes):
            container.mount(SceneButton(i, name))
        # Keep pvw_selected in-range; auto-select first scene if nothing selected yet.
        if scenes and self.pvw_selected < 0:
            self.pvw_selected = 0
        elif self.pvw_selected >= len(scenes):
            self.pvw_selected = len(scenes) - 1 if scenes else -1
        self.call_after_refresh(self._refresh_scene_buttons)

    def watch_transition_mode(self, value: str) -> None:
        sym = TRANSITION_SYMBOLS.get(value, "?")
        label = self.query_one("#trans_mode_label", Static)
        busy = value != "idle"
        label.update(f"Mode: {sym} {value}")
        label.set_class(busy, "busy")
        pgm = self.query_one("#pgm_panel")
        pvw = self.query_one("#pvw_panel")
        pgm.set_class(busy, "transitioning")
        pvw.set_class(busy, "transitioning")

    def watch_overlay_enabled(self, value: bool) -> None:
        try:
            button = self.query_one("#btn_overlay", Button)
            button.label = "Overlay: ON" if value else "Overlay: OFF"
            button.variant = "primary" if value else "warning"
        except NoMatches:
            pass

    def watch_direct_cut_mode(self, value: bool) -> None:
        try:
            button = self.query_one("#btn_direct", Button)
            button.label = "Direct: ON" if value else "Direct: OFF"
            button.variant = "warning" if value else "default"
            button.tooltip = (
                "Scene clicks cut directly to program"
                if value else
                "Scene clicks load preview"
            )
        except NoMatches:
            pass

    def watch_auto_control_connected(self, value: bool) -> None:
        self._refresh_ai_transition_button()

    def watch_auto_transition_mode(self, value: str) -> None:
        self._refresh_ai_transition_button()

    def watch_connected(self, value: bool) -> None:
        if value:
            self._set_status_bar(False)
        else:
            self._set_status_bar(True, f"Disconnected from {self.avp_host}:{self.avp_port}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _set_status_bar(self, visible: bool, msg: str = "") -> None:
        try:
            bar = self.query_one("#conn_status", Static)
            bar.update(msg)
            bar.set_class(visible, "visible")
        except NoMatches:
            pass

    def _refresh_scene_buttons(self) -> None:
        pgm = self.pgm_scene
        pvw_idx = self.pvw_selected
        for btn in self.query(SceneButton):
            is_pgm = btn.scene_name == pgm
            is_pvw = (btn.scene_index == pvw_idx)
            btn.set_class(is_pgm, "pgm")
            btn.set_class(is_pvw and not is_pgm, "pvw")
            btn._refresh_content()

    def _pvw_scene_name(self) -> Optional[str]:
        if 0 <= self.pvw_selected < len(self.scenes):
            return self.scenes[self.pvw_selected]
        return None

    def _duration(self) -> float:
        return self._fade_frames() / self.fade_fps

    def _fade_frames(self) -> int:
        try:
            value = int(self.query_one("#dur_input", Input).value.strip())
            return min(MAX_FADE_FRAMES, max(1, value))
        except (ValueError, NoMatches):
            return self.default_fade_frames

    def _set_duration(self, duration_s: float) -> None:
        frames = max(1, int(round(duration_s * self.fade_fps)))
        self._set_fade_frames(frames)

    def _set_fade_frames(self, frames: int) -> None:
        try:
            clamped = min(MAX_FADE_FRAMES, max(1, int(frames)))
            self.query_one("#dur_input", Input).value = str(clamped)
        except NoMatches:
            pass

    def _wipe_files(self) -> list[str]:
        if not self.wipe_dir:
            return []
        try:
            names = sorted(os.listdir(self.wipe_dir))
        except OSError as exc:
            self.notify(f"Wipe directory unavailable: {exc}", severity="warning")
            return []

        paths: list[str] = []
        for name in names:
            if not name.lower().endswith(".mov"):
                continue
            path = os.path.join(self.wipe_dir, name)
            if os.path.isfile(path):
                paths.append(path)
        return paths

    def _wipe_modal(self) -> WipeModal:
        return WipeModal(
            self.wipe_file,
            wipe_dir=self.wipe_dir,
            wipe_files=self._wipe_files(),
        )

    def _refresh_ai_transition_button(self) -> None:
        try:
            button = self.query_one("#btn_ai_transition", Button)
        except NoMatches:
            return

        if not self.auto_control_connected:
            button.label = "AI: n/a"
            button.variant = "default"
            button.tooltip = f"Auto-switch control not available on {self.avp_host}:{self.avp_port}"
            return
        if self.auto_transition_mode == "fade":
            button.label = "AI: FADE"
            button.variant = "success"
            button.tooltip = "Automatic speaker switches crossfade using the Fade frames value"
        elif self.auto_transition_mode == "wipe":
            button.label = "AI: WIPE"
            button.variant = "primary"
            button.tooltip = "Automatic speaker switches use the selected media wipe"
        else:
            button.label = "AI: CUT"
            button.variant = "warning"
            button.tooltip = "Automatic speaker switches hard cut"

    async def _refresh_auto_status(self) -> None:
        if not self._conn.connected:
            self.auto_control_connected = False
            self.auto_transition_mode = "n/a"
            return
        resp = await self._conn.command("auto_switch.status", raise_for_status=False)
        if resp is None or resp.code != 201 or not resp.content:
            return
        self.auto_control_connected = True
        try:
            status = parse_auto_switch_status(resp.content)
        except Exception:
            return
        if status.transition_mode is not None:
            self.auto_transition_mode = status.transition_mode
        if status.fade_duration_s is not None and not self._auto_duration_initialized:
            try:
                self._set_duration(status.fade_duration_s)
                self._auto_duration_initialized = True
            except Exception:
                pass
        if status.wipe_file and not self.wipe_file:
            self.wipe_file = status.wipe_file
            self._update_wipe_button_tooltip()

    def _scene_index(self, scene_name: str) -> int:
        if not scene_name:
            return -1
        try:
            return self.scenes.index(scene_name)
        except ValueError:
            return -1

    def _apply_pgm_pvw_swap(self, old_pgm: str) -> None:
        """After a take, show the former program on preview (bus swap)."""
        idx = self._scene_index(old_pgm)
        if idx >= 0:
            self.pvw_selected = idx

    async def _wait_transition_idle(self, max_wait: float) -> None:
        """Poll mixer.status until transition is idle or max_wait elapses."""
        interval = 0.12
        waited = 0.0
        while waited < max_wait:
            await self._refresh_once()
            if self.transition_mode == "idle":
                return
            await asyncio.sleep(interval)
            waited += interval
        await self._refresh_once()

    def _idle_wait_budget(self, cmd: str) -> float:
        if cmd.startswith("mixer.cut"):
            return 4.0
        if cmd.startswith("mixer.fade"):
            return max(6.0, self._duration() + 5.0)
        if cmd.startswith("mixer.wipe"):
            return 120.0
        return 8.0

    def _select_pvw(self, index: int) -> None:
        if 0 <= index < len(self.scenes):
            self.pvw_selected = index

    def _select_or_cut_scene(self, index: int) -> None:
        if not (0 <= index < len(self.scenes)):
            return
        scene = self.scenes[index]
        if self.direct_cut_mode:
            if scene != self.pgm_scene:
                self._do_command(self._mixer_command("cut", scene=scene))
        else:
            self._select_pvw(index)

    def _mixer_take_in_progress(self) -> bool:
        return self.transition_mode != "idle"

    def _mixer_command(self, command: str, **payload) -> str:
        return mixer_command(command, self.mixer_name, **payload)

    # ── Actions ──────────────────────────────────────────────────────────────

    def action_cut(self) -> None:
        scene = self._pvw_scene_name()
        if scene:
            self._do_command(self._mixer_command("cut", scene=scene))

    def action_auto_transition(self) -> None:
        scene = self._pvw_scene_name()
        if scene:
            self._set_fade_frames(self._fade_frames())
            dur = self._duration()
            self._do_command(self._mixer_command("fade", scene=scene, duration_sec=dur))

    def action_wipe(self) -> None:
        if self._mixer_take_in_progress():
            return
        scene = self._pvw_scene_name()
        if not scene:
            return
        if self.wipe_file:
            self._do_command(self._mixer_command("wipe", scene=scene, wipe_file=self.wipe_file))
        else:
            self.push_screen(
                self._wipe_modal(),
                callback=lambda path: self._on_wipe_path_selected(scene, path, execute=True),
            )

    def action_change_wipe_path(self) -> None:
        """Open the wipe path modal without executing a wipe."""
        self.push_screen(
            self._wipe_modal(),
            callback=lambda path: self._on_wipe_path_selected(None, path, execute=False),
        )

    def _on_wipe_path_selected(self, scene: Optional[str], path: Optional[str], *, execute: bool) -> None:
        if path:
            self.wipe_file = path
            self._update_wipe_button_tooltip()
            if self.auto_transition_mode == "wipe":
                self._set_auto_transition_settings()
        if execute and path and scene:
            self._do_command(self._mixer_command("wipe", scene=scene, wipe_file=path))

    def _update_wipe_button_tooltip(self) -> None:
        try:
            btn = self.query_one("#btn_wipe", Button)
            btn.tooltip = self.wipe_file or "(no wipe file set)"
        except NoMatches:
            pass

    def _do_wipe(self, scene: str, path: Optional[str]) -> None:
        if path:
            self._do_command(self._mixer_command("wipe", scene=scene, wipe_file=path))

    @work(exclusive=True, thread=False, group="mixer_preview")
    async def _preview_scene(self, scene: str) -> None:
        if self._mixer_take_in_progress() or scene == self.pvw_scene_remote:
            return
        resp = await self._conn.command(self._mixer_command("preview", scene=scene), raise_for_status=False)
        if resp is not None and resp.code < 400:
            await self._refresh_once()

    def action_focus_duration(self) -> None:
        try:
            self.query_one("#dur_input", Input).focus()
        except NoMatches:
            pass

    def action_refresh_status(self) -> None:
        self.run_worker(
            self._poll_status_tick(),
            group="mixer_avp_poll_now",
            exclusive=True,
            exit_on_error=False,
        )

    def action_toggle_overlay(self) -> None:
        if self._overlay_toggle_in_progress:
            return
        self._overlay_toggle_in_progress = True
        self._set_overlay_enabled(not self.overlay_enabled)

    def action_toggle_direct_cut_mode(self) -> None:
        self.direct_cut_mode = not self.direct_cut_mode

    def _available_ai_transition_modes(self) -> tuple[str, ...]:
        return AI_TRANSITION_MODES

    def action_toggle_ai_transition_mode(self) -> None:
        modes = self._available_ai_transition_modes()
        try:
            index = modes.index(self.auto_transition_mode)
        except ValueError:
            self._set_auto_transition_settings(transition_mode=modes[0])
            return
        next_mode = modes[(index + 1) % len(modes)]
        if next_mode == "wipe" and not self.wipe_file:
            self.push_screen(
                self._wipe_modal(),
                callback=self._on_ai_wipe_path_selected,
            )
            return
        self._set_auto_transition_settings(transition_mode=next_mode)

    def _on_ai_wipe_path_selected(self, path: Optional[str]) -> None:
        if not path:
            return
        self.wipe_file = path
        self._update_wipe_button_tooltip()
        self._set_auto_transition_settings(transition_mode="wipe")

    @work(thread=False)
    async def _set_auto_transition_settings(self, transition_mode: Optional[str] = None) -> None:
        if not self._conn.connected:
            self.auto_control_connected = False
            self.auto_transition_mode = "n/a"
            self.notify(
                f"Auto-switch control not available on {self.avp_host}:{self.avp_port}",
                severity="warning",
            )
            return

        mode = transition_mode or self.auto_transition_mode
        if not self.wipe_file and mode == "wipe":
            self.notify("Select a media wipe file first", severity="warning")
            return
        cmd = auto_switch_set_command(
            fade_duration_s=self._duration(),
            transition_mode=transition_mode,
            wipe_file=self.wipe_file,
        )
        resp = await self._conn.command(cmd, raise_for_status=False)
        if resp is None:
            self.auto_control_connected = False
            self.auto_transition_mode = "n/a"
            self.notify("Auto-switch control connection lost", severity="warning")
            return
        if resp.code >= 400:
            self.notify(f"Auto-switch update failed: {resp.code} {resp.status}", severity="error")
            return
        if resp.code == 201 and resp.content:
            try:
                status = parse_auto_switch_status(resp.content)
                if status.transition_mode is not None:
                    self.auto_transition_mode = status.transition_mode
                if status.fade_duration_s is not None:
                    self._set_duration(status.fade_duration_s)
                    self._auto_duration_initialized = True
                if status.wipe_file:
                    self.wipe_file = status.wipe_file
                    self._update_wipe_button_tooltip()
                self.auto_control_connected = True
            except Exception:
                pass

    @work(thread=False)
    async def _set_overlay_enabled(self, enabled: bool) -> None:
        try:
            if not self._conn.connected:
                self.notify("Connection lost", severity="error")
                return

            commands = overlay_toggle_commands(
                enabled=enabled,
                mixer_name=self.mixer_name,
                overlay_source_otm_name=self.overlay_source_otm_name,
                overlay_otm_name=self.overlay_otm_name,
                overlay_selector_name=self.overlay_selector_name,
            )

            for i, cmd in enumerate(commands):
                if i > 0:
                    await asyncio.sleep(OVERLAY_SETTLE_SECONDS)
                resp = await self._conn.command(cmd, raise_for_status=False)
                if resp is None:
                    self.notify("Connection lost", severity="error")
                    return
                if resp.code >= 400:
                    self.notify(f"Overlay toggle failed: {resp.code} {resp.status}", severity="error")
                    return

            self.overlay_enabled = enabled
        finally:
            self._overlay_toggle_in_progress = False

    @work(thread=False)
    async def _do_command(self, cmd: str) -> None:
        is_take = cmd.startswith(TAKE_COMMAND_PREFIXES)
        if is_take and self._mixer_take_in_progress():
            return
        old_pgm = self.pgm_scene
        resp = await self._conn.command(cmd, raise_for_status=False)
        if resp is None:
            self.notify("Connection lost", severity="error")
        elif resp.code >= 400:
            self.notify(f"Error {resp.code}: {resp.status}", severity="error")
        else:
            if cmd.startswith(TAKE_COMMAND_PREFIXES):
                await self._wait_transition_idle(self._idle_wait_budget(cmd))
                self._apply_pgm_pvw_swap(old_pgm)
            else:
                await asyncio.sleep(0.3)
                await self._refresh_once()

    async def _refresh_once(self) -> None:
        resp = await self._conn.command(f"mixer.status {self.mixer_name}", raise_for_status=False)
        if resp and resp.code == 201 and resp.content:
            try:
                status = parse_mixer_status(resp.content)
                self.pgm_scene = status.pgm_scene
                self.pvw_scene_remote = status.pvw_scene
                self.transition_mode = status.transition
            except Exception:
                pass

    # ── Key events ───────────────────────────────────────────────────────────

    def on_key(self, event) -> None:
        if isinstance(getattr(self, "focused", None), Input):
            return
        key = event.key
        # 1-9: select preview, or direct cut if direct mode is enabled.
        if key.isdigit() and key != "0":
            idx = int(key) - 1
            self._select_or_cut_scene(idx)
            event.stop()
            return
        # F1-F9: direct cut (skip preview)
        if key.startswith("f") and key[1:].isdigit():
            fnum = int(key[1:])
            if 1 <= fnum <= 9:
                idx = fnum - 1
                if 0 <= idx < len(self.scenes):
                    self._do_command(self._mixer_command("cut", scene=self.scenes[idx]))
                    event.stop()
                    return

    # ── Button presses ────────────────────────────────────────────────────────

    @on(Button.Pressed, "#btn_cut")
    def on_btn_cut(self) -> None:
        self.action_cut()

    @on(Button.Pressed, "#btn_auto")
    def on_btn_auto(self) -> None:
        self.action_auto_transition()

    @on(Button.Pressed, "#btn_wipe")
    def on_btn_wipe(self) -> None:
        self.action_wipe()

    @on(Button.Pressed, "#btn_change_wipe")
    def on_btn_change_wipe(self) -> None:
        self.action_change_wipe_path()

    @on(Button.Pressed, "#btn_overlay")
    def on_btn_overlay(self) -> None:
        self.action_toggle_overlay()

    @on(Button.Pressed, "#btn_direct")
    def on_btn_direct(self) -> None:
        self.action_toggle_direct_cut_mode()

    @on(Button.Pressed, "#btn_ai_transition")
    def on_btn_ai_transition(self) -> None:
        self.action_toggle_ai_transition_mode()

    @on(Input.Submitted, "#dur_input")
    def on_duration_submitted(self) -> None:
        self._set_fade_frames(self._fade_frames())
        if self.auto_transition_mode in ("fade", "wipe"):
            self._set_auto_transition_settings()

    def on_scene_button_selected(self, event: SceneButton.Selected) -> None:
        self._select_or_cut_scene(event.index)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avplumber Mixer TUI — live video switcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="localhost", help="avplumber host (default: localhost)")
    parser.add_argument("--port", type=int, required=True, help="avplumber TCP control port")
    parser.add_argument("--mixer", default="mixer", help="mixer instance name (default: mixer)")
    parser.add_argument("--overlay-otm", default="otm_html_overlay", help="overlay one_to_many node name")
    parser.add_argument("--overlay-source-otm", default="otm_html_overlay_src", help="overlay source one_to_many node name")
    parser.add_argument("--overlay-selector", default="overlay_sel", help="overlay source_switcher node name")
    parser.add_argument("--wipe-file", default=None, metavar="PATH",
                        help="default wipe file path; skips the prompt when set")
    parser.add_argument("--wipe-dir", default=os.environ.get("AVP_WIPE_DIR"), metavar="PATH",
                        help="directory of selectable wipe .mov files; may also be set with AVP_WIPE_DIR")
    parser.add_argument("--fade-fps", type=float, default=DEFAULT_FADE_FPS,
                        help=f"frame rate used to convert fade frames to seconds (default: {DEFAULT_FADE_FPS:g})")
    parser.add_argument("--fade-frames", type=int, default=DEFAULT_FADE_FRAMES,
                        help=f"default fade/wipe transition frame count (default: {DEFAULT_FADE_FRAMES})")
    args = parser.parse_args()

    app = MixerTUI(
        host=args.host,
        port=args.port,
        mixer=args.mixer,
        overlay_otm=args.overlay_otm,
        overlay_source_otm=args.overlay_source_otm,
        overlay_selector=args.overlay_selector,
        wipe_file=args.wipe_file,
        wipe_dir=args.wipe_dir,
        fade_fps=args.fade_fps,
        fade_frames=args.fade_frames,
    )
    app.run()


if __name__ == "__main__":
    main()
