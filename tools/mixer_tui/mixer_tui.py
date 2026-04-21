#!/usr/bin/env python3
"""
Avplumber Mixer TUI -- live production video switcher.

Scene composition (graphs, PiP positions) is defined on the avplumber side with
``mixer.scene`` JSON using a ``sources`` map: logical camera name -> ``graph``
plus ``dst_x`` / ``dst_y`` (see doc/mixer_orchestrator.md). This UI only refers
to scenes by name.

Usage:
    python tools/mixer_tui.py --host localhost --port 5555 --mixer mixer \\
        fullcam1 pip sidebyside

Keyboard shortcuts:
    1-9        Select scene N on Preview bus
    F1-F9      Direct CUT to scene N (skips preview step)
    c          CUT  (take preview to program, hard cut)
    x          X-FADE (crossfade preview to program at set duration)
    w          WIPE (prompt for wipe file path, then execute)
    d          Focus the duration input field
    s          Force an immediate mixer.status refresh
    q / ctrl+c Quit

The Textual keys / shortcuts panel opens on startup.
"""

import argparse
import asyncio
import json
import sys
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Static,
)


# ---------------------------------------------------------------------------
# avplumber TCP client (ported from msesm/src/stream_monitor.py)
# ---------------------------------------------------------------------------

class AvpResponse:
    def __init__(self, code: int, status: str, content: Optional[str]):
        self.code = code
        self.status = status
        self.content = content

    def json(self):
        return json.loads(self.content) if self.content else None


class AvpError(RuntimeError):
    pass


class AvpClient:
    """Async TCP client for avplumber's text control protocol."""

    def __init__(self):
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None

    async def connect(self, host: str, port: int) -> None:
        self.reader, self.writer = await asyncio.open_connection(host, port)
        response = await self.read()
        if response.code != 100:
            raise AvpError(f"Unexpected greeting from avplumber: {response.code} {response.status}")

    async def disconnect(self) -> None:
        if self.writer:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass
            self.writer = None
            self.reader = None

    async def read(self, raise_for_status: bool = False) -> AvpResponse:
        line = await self.reader.readline()
        line = line.decode("utf-8")
        if not line:
            raise EOFError("Connection closed by avplumber")
        parts = line.split(maxsplit=1)
        code = int(parts[0].strip())
        status = parts[1].strip() if len(parts) > 1 else ""
        content = None
        if code == 201:
            content = ""
            while True:
                line = await self.reader.readline()
                line = line.decode("utf-8")
                if line.rstrip("\n") == "":
                    break
                content += line
        if raise_for_status and not (200 <= code < 300):
            raise AvpError(f"avplumber error {code} {status}")
        return AvpResponse(code=code, status=status, content=content)

    async def write(self, cmd: str) -> None:
        self.writer.write(cmd.encode("utf-8") + b"\n")
        await self.writer.drain()

    async def command(self, cmd: str, raise_for_status: bool = True) -> AvpResponse:
        await self.write(cmd)
        return await self.read(raise_for_status=raise_for_status)


class AvpConnection:
    """Manages an AvpClient with automatic reconnection."""

    RECONNECT_INTERVAL = 2.0

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._client: Optional[AvpClient] = None
        self._lock = asyncio.Lock()
        self.connected = False

    async def get_client(self) -> Optional[AvpClient]:
        if self._client is not None and self.connected:
            return self._client
        return None

    async def ensure_connected(self) -> bool:
        async with self._lock:
            if self.connected and self._client is not None:
                return True
            client = AvpClient()
            try:
                await client.connect(self.host, self.port)
                self._client = client
                self.connected = True
                return True
            except Exception:
                self._client = None
                self.connected = False
                return False

    async def command(self, cmd: str, raise_for_status: bool = True) -> Optional[AvpResponse]:
        client = await self.get_client()
        if client is None:
            return None
        try:
            return await client.command(cmd, raise_for_status=raise_for_status)
        except Exception:
            self.connected = False
            self._client = None
            return None

    async def disconnect(self) -> None:
        if self._client:
            await self._client.disconnect()
        self._client = None
        self.connected = False


# ---------------------------------------------------------------------------
# Wipe file prompt modal
# ---------------------------------------------------------------------------

class WipeModal(ModalScreen[Optional[str]]):
    """Modal dialog to enter the wipe file path."""

    CSS = """
    WipeModal {
        align: center middle;
    }
    WipeModal > Vertical {
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    WipeModal Label {
        margin-bottom: 1;
    }
    WipeModal Input {
        margin-bottom: 1;
    }
    WipeModal Horizontal {
        height: auto;
        align-horizontal: right;
    }
    WipeModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Enter wipe file path:")
            yield Input(placeholder="/path/to/wipe.mov", id="wipe_path")
            with Horizontal():
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("WIPE", variant="primary", id="ok")

    def on_mount(self) -> None:
        self.query_one("#wipe_path", Input).focus()

    @on(Button.Pressed, "#ok")
    def do_ok(self) -> None:
        path = self.query_one("#wipe_path", Input).value.strip()
        self.dismiss(path if path else None)

    @on(Button.Pressed, "#cancel")
    def do_cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted)
    def on_submitted(self) -> None:
        self.do_ok()


# ---------------------------------------------------------------------------
# Scene button widget
# ---------------------------------------------------------------------------

class SceneButton(Static):
    """A scene selection button showing its index, name, and bus status."""

    class Selected(Message):
        """Posted when the user clicks this scene button."""
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    DEFAULT_CSS = """
    SceneButton {
        width: auto;
        min-width: 16;
        height: 5;
        border: solid $primary-darken-2;
        padding: 0 1;
        margin: 0 1;
        content-align: center middle;
        text-align: center;
    }
    SceneButton.pgm {
        border: solid $error;
        background: $error-darken-3;
        color: $error-lighten-2;
    }
    SceneButton.pvw {
        border: solid $success;
        background: $success-darken-3;
        color: $success-lighten-2;
    }
    SceneButton:hover {
        border: solid $accent;
        background: $accent-darken-3;
    }
    """

    def __init__(self, index: int, name: str) -> None:
        super().__init__()
        self.scene_index = index
        self.scene_name = name
        self._refresh_content()

    def _refresh_content(self) -> None:
        idx = self.scene_index + 1
        name = self.scene_name
        classes = self.classes
        if "pgm" in classes:
            bus = "◉ PGM"
        elif "pvw" in classes:
            bus = "● PVW"
        else:
            bus = f"  [{idx}]"
        self.update(f"{bus}\n{name}")

    def on_click(self) -> None:
        self.post_message(SceneButton.Selected(self.scene_index))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

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

/* ── Transition bar ── */
#transition_bar {
    height: 5;
    background: $panel;
    border: solid $primary-darken-2;
    align: left middle;
    padding: 0 1;
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
    width: 8;
    margin-right: 1;
}

#btn_cut {
    margin-right: 1;
}

#btn_auto {
    margin-right: 1;
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
        Binding("d", "focus_duration", "Duration", show=False),
        Binding("s", "refresh_status", "Refresh", show=False),
        # F1-F9: direct cut to scene (bound dynamically in on_key)
    ]

    # Reactive state from mixer.status
    pgm_scene: reactive[str] = reactive("", layout=True)
    pvw_scene_remote: reactive[str] = reactive("")
    transition_mode: reactive[str] = reactive("idle")

    # Local PVW selection (TD's intent)
    pvw_selected: reactive[int] = reactive(-1, layout=True)

    connected: reactive[bool] = reactive(False)

    def __init__(self, host: str, port: int, mixer: str, scenes: list[str]) -> None:
        super().__init__()
        self.avp_host = host
        self.avp_port = port
        self.mixer_name = mixer
        self.scenes = scenes
        self._conn = AvpConnection(host, port)
        self._pending_action: Optional[str] = None  # "cut" | "auto" | "wipe:<path>"

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
            for i, name in enumerate(self.scenes):
                yield SceneButton(i, name)

        with Horizontal(id="transition_bar"):
            yield Static("Mode: idle", id="trans_mode_label")
            yield Button("✂ CUT", id="btn_cut", variant="error")
            yield Button("⟿ X-FADE", id="btn_auto", variant="success")
            yield Static("Duration:", id="dur_label")
            yield Input("2.0", id="dur_input")
            yield Button("▶ WIPE", id="btn_wipe", variant="primary")

        yield Footer()

    # ── On mount ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        if self.scenes:
            self.pvw_selected = 0
        self._start_connection_loop()
        self._poll_status()
        self.call_later(self.action_show_help_panel)

    # ── Connection & polling workers ────────────────────────────────────────

    # Use a dedicated worker group so this is not cancelled when `_poll_status`
    # starts (both used exclusive=True on the default group, which only allows
    # one worker — the poll worker was killing the connection loop on startup).
    @work(exclusive=True, thread=False, group="mixer_avp_connect")
    async def _start_connection_loop(self) -> None:
        """Keep trying to connect; on success hand off to polling."""
        while True:
            if not self._conn.connected:
                ok = await self._conn.ensure_connected()
                self.connected = ok
                if ok:
                    self._set_status_bar(False)
                else:
                    self._set_status_bar(True, f"Connecting to {self.avp_host}:{self.avp_port}…")
                    await asyncio.sleep(self._conn.RECONNECT_INTERVAL)
            else:
                await asyncio.sleep(0.5)

    @work(exclusive=True, thread=False, group="mixer_avp_poll")
    async def _poll_status(self) -> None:
        """Poll mixer.status every 500 ms and update reactive state."""
        while True:
            await asyncio.sleep(0.5)
            if not self._conn.connected:
                continue
            resp = await self._conn.command(f"mixer.status {self.mixer_name}", raise_for_status=False)
            if resp is None:
                self.connected = False
                self._set_status_bar(True, f"Lost connection to {self.avp_host}:{self.avp_port}")
                continue
            if resp.code == 201 and resp.content:
                try:
                    data = json.loads(resp.content.strip())
                    self.pgm_scene = data.get("pgm_scene", "")
                    self.pvw_scene_remote = data.get("pvw_scene", "")
                    self.transition_mode = data.get("transition", "idle")
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
        try:
            return float(self.query_one("#dur_input", Input).value)
        except (ValueError, NoMatches):
            return 2.0

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

    def _mixer_take_in_progress(self) -> bool:
        return self.transition_mode != "idle"

    # ── Actions ──────────────────────────────────────────────────────────────

    def action_cut(self) -> None:
        scene = self._pvw_scene_name()
        if scene:
            self._do_command(f"mixer.cut {self.mixer_name} {scene}")

    def action_auto_transition(self) -> None:
        scene = self._pvw_scene_name()
        if scene:
            dur = self._duration()
            self._do_command(f"mixer.fade {self.mixer_name} {scene} {dur}")

    def action_wipe(self) -> None:
        if self._mixer_take_in_progress():
            return
        scene = self._pvw_scene_name()
        if scene:
            self.push_screen(WipeModal(), callback=lambda path: self._do_wipe(scene, path))

    def _do_wipe(self, scene: str, path: Optional[str]) -> None:
        if path:
            self._do_command(f"mixer.wipe {self.mixer_name} {scene} {path}")

    def action_focus_duration(self) -> None:
        try:
            self.query_one("#dur_input", Input).focus()
        except NoMatches:
            pass

    def action_refresh_status(self) -> None:
        self._poll_status()

    @work(thread=False)
    async def _do_command(self, cmd: str) -> None:
        is_take = cmd.startswith(("mixer.cut", "mixer.fade", "mixer.wipe"))
        if is_take and self._mixer_take_in_progress():
            return
        old_pgm = self.pgm_scene
        resp = await self._conn.command(cmd, raise_for_status=False)
        if resp is None:
            self.notify("Connection lost", severity="error")
        elif resp.code >= 400:
            self.notify(f"Error {resp.code}: {resp.status}", severity="error")
        else:
            if cmd.startswith(("mixer.cut", "mixer.fade", "mixer.wipe")):
                await self._wait_transition_idle(self._idle_wait_budget(cmd))
                self._apply_pgm_pvw_swap(old_pgm)
            else:
                await asyncio.sleep(0.3)
                await self._refresh_once()

    async def _refresh_once(self) -> None:
        resp = await self._conn.command(f"mixer.status {self.mixer_name}", raise_for_status=False)
        if resp and resp.code == 201 and resp.content:
            try:
                data = json.loads(resp.content.strip())
                self.pgm_scene = data.get("pgm_scene", "")
                self.pvw_scene_remote = data.get("pvw_scene", "")
                self.transition_mode = data.get("transition", "idle")
            except Exception:
                pass

    # ── Key events ───────────────────────────────────────────────────────────

    def on_key(self, event) -> None:
        key = event.key
        # 1-9: select preview
        if key.isdigit() and key != "0":
            idx = int(key) - 1
            self._select_pvw(idx)
            event.stop()
            return
        # F1-F9: direct cut (skip preview)
        if key.startswith("f") and key[1:].isdigit():
            fnum = int(key[1:])
            if 1 <= fnum <= 9:
                idx = fnum - 1
                if 0 <= idx < len(self.scenes):
                    self._do_command(f"mixer.cut {self.mixer_name} {self.scenes[idx]}")
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

    def on_scene_button_selected(self, event: SceneButton.Selected) -> None:
        self._select_pvw(event.index)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avplumber Mixer TUI — live video switcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="127.0.0.1", help="avplumber host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, required=True, help="avplumber TCP control port")
    parser.add_argument("--mixer", default="mixer", help="mixer instance name (default: mixer)")
    parser.add_argument(
        "scenes",
        nargs="+",
        metavar="SCENE",
        help="Scene names (must match mixer.scene definitions on the server)",
    )
    args = parser.parse_args()

    if len(args.scenes) > 9:
        print("Warning: only the first 9 scenes will be mapped to number keys", file=sys.stderr)

    app = MixerTUI(
        host=args.host,
        port=args.port,
        mixer=args.mixer,
        scenes=args.scenes,
    )
    app.run()


if __name__ == "__main__":
    main()
