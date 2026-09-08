"""Textual manual control surface for the generic mixer demo."""

from __future__ import annotations

import argparse
import json
import asyncio

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.scrollbar import ScrollBarRender
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static

from avpmixer.control import AvpConnection

try:
    from .control import (
        mixer_command,
        parse_mixer_status,
        parse_scene_list,
    )
except ImportError:
    from control import (
        mixer_command,
        parse_mixer_status,
        parse_scene_list,
    )


class SceneScrollBarRender(ScrollBarRender):
    # Fractional block glyphs leave end-cap artifacts in browser terminals.
    HORIZONTAL_BARS = [" "]


class SceneButton(Static):
    """Scene tile showing its keyboard index and current bus assignment."""

    class Selected(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    DEFAULT_CSS = """
    SceneButton {
        width: auto;
        min-width: 18;
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
        self.refresh_content()

    def refresh_content(self) -> None:
        if "pgm" in self.classes:
            bus = "◉ PGM"
        elif "pvw" in self.classes:
            bus = "● PVW"
        else:
            bus = f"  [{self.scene_index + 1}]"
        self.update(f"{bus}\n{self.scene_name}")

    def on_click(self) -> None:
        self.post_message(self.Selected(self.scene_index))


class MixerTui(App):
    TITLE = "AVPlumber Generic Mixer"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("c", "cut", "Cut"),
        Binding("f", "fade", "Fade"),
        Binding("w", "wipe", "Wipe"),
        Binding("t", "toggle_direct", "Direct"),
        Binding("r", "reconnect", "Reconnect"),
    ]
    CSS = """
    Screen { background: $surface-darken-1; }
    #connection {
        height: 1;
        background: $error-darken-2;
        color: $error-lighten-1;
        content-align: center middle;
        text-align: center;
    }
    #connection.connected {
        background: $success-darken-3;
        color: $success-lighten-1;
    }
    #buses { height: 9; margin-bottom: 1; }
    .bus {
        width: 1fr;
        height: 100%;
        align: center middle;
        content-align: center middle;
        padding: 0 2;
    }
    #program {
        border: heavy $error;
        background: $error-darken-3;
    }
    #preview {
        border: heavy $success;
        background: $success-darken-3;
    }
    .bus.transitioning {
        border: heavy $warning;
        background: $warning-darken-3;
    }
    .bus Label, .bus Static {
        width: 100%;
        content-align: center middle;
        text-align: center;
    }
    .bus Label { text-style: bold; }
    #program Label { color: $error-lighten-1; }
    #preview Label { color: $success-lighten-1; }
    #program_scene, #preview_scene { text-style: bold; }
    #scenes {
        height: 7;
        background: $panel;
        border: solid $primary-darken-2;
        align: left middle;
        padding: 0 1;
        margin-bottom: 1;
        overflow-x: auto;
    }
    #grids, #takes, #settings {
        height: 5;
        background: $panel;
        border: solid $primary-darken-2;
        align: left middle;
        padding: 0 1;
    }
    #grids { margin-bottom: 1; }
    Button { margin-right: 1; }
    Input { width: 24; margin-right: 1; }
    #wipe_file { width: 1fr; min-width: 24; }
    #wipe_choice { width: 26; margin-right: 1; }
    #direct_transition { width: 22; margin-right: 1; }
    #transition_status { width: auto; min-width: 20; color: $text-muted; }
    #transition_status.busy { color: $warning; text-style: bold; }
    """

    def __init__(
        self,
        host: str,
        port: int,
        mixer: str,
        *,
        fade_duration: float,
        wipe_file: str = "",
        direct: bool = True,
    ) -> None:
        super().__init__()
        self.connection = AvpConnection(host, port)
        self.mixer_name = mixer
        self.default_fade_duration = fade_duration
        self.default_wipe_file = wipe_file
        self._wipes: dict[str, dict] = {}
        self.scenes: list[str] = []
        self.selected_scene = ""
        self.pgm_scene = ""
        self.pvw_scene = ""
        self.transition = "idle"
        self.direct_mode = direct
        self._poll_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Disconnected", id="connection")
        with Horizontal(id="buses"):
            with Vertical(classes="bus", id="program"):
                yield Label("◉  PROGRAM  ◉")
                yield Static("(none)", id="program_scene")
                yield Static("ON AIR")
            with Vertical(classes="bus", id="preview"):
                yield Label("●  PREVIEW  ●")
                yield Static("(none)", id="preview_scene")
                yield Static("READY")
        with Horizontal(id="scenes"):
            pass
        with Horizontal(id="grids"):
            yield Button("Fullscreen", id="fullscreen", variant="primary")
            yield Button("2-box", id="grid_2")
            yield Button("4-box", id="grid_4")
            yield Button("8-box", id="grid_8")
            yield Button("16-box", id="grid_16")
            yield Button("◀ Page", id="page_previous")
            yield Button("Page ▶", id="page_next")
        with Horizontal(id="settings"):
            yield Static("Mode: idle", id="transition_status")
            yield Label("Fade seconds:")
            yield Input(str(self.default_fade_duration), id="fade_duration")
            yield Label("Wipe:")
            # A library published by the mixer fills the picker; without one the
            # path field stays the way to name a clip.
            yield Select([], id="wipe_choice", allow_blank=True, prompt="(path below)")
            yield Input(self.default_wipe_file, id="wipe_file",
                        placeholder="/path/on/mixer/host/wipe.mov")
        with Horizontal(id="takes"):
            yield Button("✂ CUT", id="cut", variant="error")
            yield Button("⟿ FADE", id="fade", variant="success")
            yield Button("▶ MEDIA WIPE", id="wipe", variant="primary")
            yield Button("Direct: ON" if self.direct_mode else "Direct: OFF", id="direct",
                         variant="warning" if self.direct_mode else "default")
            yield Select([("Cut", "cut"), ("Fade", "fade"), ("Media Wipe", "wipe")],
                         value="cut", allow_blank=False, id="direct_transition",
                         tooltip="Current transition, shared by take buttons and Direct mode")
            yield Button("↻ RECONNECT", id="reconnect")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#scenes").horizontal_scrollbar.renderer = SceneScrollBarRender
        self._poll_timer = self.set_interval(0.5, self._poll_status)
        self.action_reconnect()

    async def on_unmount(self) -> None:
        self._poll_timer.stop()
        await self.connection.disconnect()

    def _set_connection_text(self, text: str) -> None:
        connection = self.query_one("#connection", Static)
        connection.update(text)
        connection.set_class(self.connection.connected, "connected")

    def _refresh_scene_buttons(self) -> None:
        for button in self.query(SceneButton):
            is_program = button.scene_name == self.pgm_scene
            is_preview = button.scene_name == self.pvw_scene
            button.set_class(is_program, "pgm")
            button.set_class(is_preview and not is_program, "pvw")
            button.refresh_content()

    async def _fetch_scenes(self) -> None:
        content = await self.connection.command(f"mixer.scenes {self.mixer_name}")
        if content is None:
            raise RuntimeError("mixer.scenes returned no content")
        scenes = parse_scene_list(content)
        self.scenes = scenes
        scene_list = self.query_one("#scenes", Horizontal)
        for button in list(scene_list.query(SceneButton)):
            await button.remove()
        await scene_list.mount(
            *[SceneButton(index, scene) for index, scene in enumerate(scenes)]
        )
        if scenes:
            self.selected_scene = scenes[0]
        self._refresh_scene_buttons()

    async def _read_status(self) -> None:
        content = await self.connection.command(f"mixer.status {self.mixer_name}")
        if content is None:
            raise RuntimeError("mixer.status returned no content")
        status = parse_mixer_status(content)
        # A reply can arrive after Textual has detached the closing screen.
        if not self.query("#program_scene"):
            return
        self.pgm_scene = status.pgm_scene
        self.pvw_scene = status.pvw_scene
        self.transition = status.transition
        self.query_one("#program_scene", Static).update(self.pgm_scene or "(none)")
        self.query_one("#preview_scene", Static).update(self.pvw_scene or "(none)")
        transition_status = self.query_one("#transition_status", Static)
        transition_status.update(f"Mode: {self.transition}")
        transition_status.set_class(self.transition != "idle", "busy")
        for selector in ("#program", "#preview"):
            self.query_one(selector).set_class(
                self.transition != "idle", "transitioning"
            )
        self._refresh_scene_buttons()

    async def _poll_status(self) -> None:
        if not self.connection.connected or not self.query("#program_scene"):
            return
        try:
            await self._read_status()
        except Exception as exc:
            await self.connection.disconnect()
            if self.query("#connection"):
                self._set_connection_text(f"Connection lost: {exc}")

    @work(exclusive=True, group="connection")
    async def _reconnect(self) -> None:
        self._set_connection_text(
            f"Connecting to {self.connection.host}:{self.connection.port}..."
        )
        try:
            await self.connection.connect()
            await self._fetch_scenes()
            await self._apply_settings()
            await self._read_status()
        except Exception as exc:
            await self.connection.disconnect()
            self._set_connection_text(f"Connection failed: {exc}")
            return
        self._set_connection_text(
            f"Connected to {self.connection.host}:{self.connection.port}; "
            f"transition={self.transition}"
        )

    def action_reconnect(self) -> None:
        self._reconnect()

    async def _preview_and_wait(self, scene: str) -> None:
        if self.transition != "idle":
            raise RuntimeError(f"transition is busy: {self.transition}")
        if self.pvw_scene != scene:
            await self.connection.command(
                mixer_command("preview", self.mixer_name, scene=scene)
            )
        for _ in range(80):
            await self._read_status()
            if self.pvw_scene == scene and self.transition == "idle":
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(f"preview did not become ready: {scene}")

    @work(exclusive=True, group="take")
    async def _take(self, transition: str, scene: str | None = None) -> None:
        direct = scene is not None
        scene = scene if direct else self.selected_scene
        if not scene:
            self.notify("Select a scene first", severity="warning")
            return
        try:
            # Every take reaches the native mixer, including a correction back
            # to Program while its previous transition is still in flight.
            # Preview is optional preparation, never a prerequisite for a take.
            payload = {"scene": scene}
            if transition == "fade":
                duration = float(self.query_one("#fade_duration", Input).value)
                if duration <= 0:
                    raise ValueError("transition duration must be positive")
                payload["duration_sec"] = duration
            elif transition == "wipe":
                chosen = self.query_one("#wipe_choice", Select).value
                wipe = self._wipes.get(chosen) if chosen is not Select.BLANK else None
                wipe_file = wipe["path"] if wipe else self.query_one("#wipe_file", Input).value.strip()
                if wipe and wipe.get("duration_seconds"):
                    payload["duration_sec"] = wipe["duration_seconds"]
                if not wipe_file:
                    raise ValueError("Enter a transparent wipe file path on the mixer host")
                # The backend probes the clip's duration. Its filesystem may
                # differ from the machine running this control interface.
                payload["wipe_file"] = wipe_file
            await self.connection.command(
                mixer_command(transition, self.mixer_name, **payload)
            )
            await self._read_status()
        except Exception as exc:
            self.notify(str(exc), severity="error")

    def _use_transition(self, transition: str) -> None:
        self.query_one("#direct_transition", Select).value = transition
        self._take(transition)

    def action_cut(self) -> None:
        self._use_transition("cut")

    def action_fade(self) -> None:
        self._use_transition("fade")

    def action_wipe(self) -> None:
        self._use_transition("wipe")

    @work(exclusive=True, group="preview")
    async def _preview_selected(self) -> None:
        if not self.selected_scene:
            return
        try:
            await self._preview_and_wait(self.selected_scene)
        except Exception as exc:
            self.notify(str(exc), severity="error")

    def _select_scene(self, scene: str) -> None:
        try:
            self.scenes.index(scene)
        except ValueError:
            self.notify(f"Scene is unavailable: {scene}", severity="warning")
            return
        self.selected_scene = scene
        if self.direct_mode:
            self._direct_take(scene)
        else:
            self._preview_selected()

    def _selected_grid(self) -> tuple[int, int] | None:
        parts = self.selected_scene.split("_")
        if len(parts) == 4 and parts[0] == "grid" and parts[2] == "page":
            return int(parts[1]), int(parts[3])
        return None

    def _select_grid_page(self, capacity: int, page: int) -> None:
        available = sorted(
            int(scene.rsplit("_", 1)[1])
            for scene in self.scenes
            if scene.startswith(f"grid_{capacity}_page_")
        )
        if not available:
            self.notify(f"No {capacity}-box pages", severity="warning")
            return
        page = min(max(page, available[0]), available[-1])
        self._select_scene(f"grid_{capacity}_page_{page}")

    def _direct_take(self, scene: str) -> None:
        transition = str(self.query_one("#direct_transition", Select).value)
        self._take(transition, scene)

    def action_toggle_direct(self) -> None:
        self._set_direct(not self.direct_mode)

    async def _apply_settings(self) -> None:
        """Defaults published by a config-driven mixer; older mixers have none."""
        try:
            content = await self.connection.command(f"mixer.settings {self.mixer_name}")
            settings = json.loads(content or "{}")
        except Exception:
            return
        if "direct" in settings:
            self._set_direct(bool(settings["direct"]))
        if "fade_seconds" in settings:
            self.query_one("#fade_duration", Input).value = str(settings["fade_seconds"])
        library = settings.get("wipes") or []
        if isinstance(library, list) and library != list(self._wipes.values()):
            self._wipes = {str(w["id"]): w for w in library}
            picker = self.query_one("#wipe_choice", Select)
            picker.set_options([(str(w.get("name") or w["id"]), str(w["id"])) for w in library])
            default = settings.get("default_wipe")
            if default in self._wipes:
                picker.value = default
        if settings.get("wipe_file") and not self._wipes:
            self.query_one("#wipe_file", Input).value = str(settings["wipe_file"])

    def _set_direct(self, enabled: bool) -> None:
        self.direct_mode = enabled
        button = self.query_one("#direct", Button)
        button.label = "Direct: ON" if self.direct_mode else "Direct: OFF"
        button.variant = "warning" if self.direct_mode else "default"
        button.tooltip = (
            "Scene and layout selections use the chosen transition to program"
            if self.direct_mode
            else "Scene and layout selections load preview"
        )

    def on_scene_button_selected(self, event: SceneButton.Selected) -> None:
        if 0 <= event.index < len(self.scenes):
            self._select_scene(self.scenes[event.index])

    def on_key(self, event) -> None:
        if isinstance(self.focused, (Input, Select)):
            return
        key = event.key
        if key.isdigit() and key != "0":
            index = int(key) - 1
            if 0 <= index < min(9, len(self.scenes)):
                self._select_scene(self.scenes[index])
                event.stop()
            return
        if key.startswith("f") and key[1:].isdigit():
            index = int(key[1:]) - 1
            if 0 <= index < min(9, len(self.scenes)):
                self._direct_take(self.scenes[index])
                event.stop()

    @on(Button.Pressed)
    def on_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "fullscreen":
            scene = (
                self.selected_scene
                if self.selected_scene.startswith("fullscreen_")
                else "fullscreen_0"
            )
            self._select_scene(scene)
        elif button_id.startswith("grid_"):
            self._select_grid_page(int(button_id.split("_", 1)[1]), 0)
        elif button_id in ("page_previous", "page_next"):
            selected = self._selected_grid()
            if selected:
                capacity, page = selected
                self._select_grid_page(
                    capacity,
                    page + (-1 if button_id == "page_previous" else 1),
                )
        elif button_id == "cut":
            self.action_cut()
        elif button_id == "fade":
            self.action_fade()
        elif button_id == "wipe":
            self.action_wipe()
        elif button_id == "direct":
            self.action_toggle_direct()
        elif button_id == "reconnect":
            self.action_reconnect()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--mixer", default="mixer")
    parser.add_argument("--fade-duration", type=float, default=0.5)
    parser.add_argument("--wipe-file", default="",
                        help="Transparent media wipe path on the mixer host")
    parser.add_argument("--no-direct", action="store_true",
                        help="Start with scene picks loading preview instead of taking to program")
    args = parser.parse_args(argv)
    MixerTui(
        args.host,
        args.port,
        args.mixer,
        fade_duration=args.fade_duration,
        wipe_file=args.wipe_file,
        direct=not args.no_direct,
    ).run()


if __name__ == "__main__":
    main()
