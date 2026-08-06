"""Textual manual control surface for the generic mixer demo."""

from __future__ import annotations

import argparse
import asyncio

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Static

try:
    from .control import AvpConnection, mixer_command, parse_mixer_status, parse_scene_list
except ImportError:
    from control import AvpConnection, mixer_command, parse_mixer_status, parse_scene_list


class MixerTui(App):
    TITLE = "AVPlumber Generic Mixer"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("c", "cut", "Cut"),
        Binding("f", "fade", "Fade"),
        Binding("w", "wipe", "Wipe"),
        Binding("r", "reconnect", "Reconnect"),
    ]
    CSS = """
    Screen { layout: vertical; }
    #connection { height: 1; padding: 0 1; }
    #buses { height: 7; }
    .bus { width: 1fr; border: solid $primary; padding: 1 2; }
    #program { border: solid $error; }
    #preview { border: solid $success; }
    #scenes { height: 1fr; border: solid $primary; }
    #grids, #takes, #settings { height: 3; align-vertical: middle; }
    Button { margin-right: 1; }
    Input { width: 24; margin-right: 1; }
    """

    def __init__(
        self,
        host: str,
        port: int,
        mixer: str,
        *,
        fade_duration: float,
        wipe_file: str,
    ) -> None:
        super().__init__()
        self.connection = AvpConnection(host, port)
        self.mixer_name = mixer
        self.default_fade_duration = fade_duration
        self.default_wipe_file = wipe_file
        self.scenes: list[str] = []
        self.selected_scene = ""
        self.pgm_scene = ""
        self.pvw_scene = ""
        self.transition = "idle"
        self._poll_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Disconnected", id="connection")
        with Horizontal(id="buses"):
            with Vertical(classes="bus", id="program"):
                yield Label("PROGRAM")
                yield Static("(none)", id="program_scene")
            with Vertical(classes="bus", id="preview"):
                yield Label("PREVIEW")
                yield Static("(none)", id="preview_scene")
        yield ListView(id="scenes")
        with Horizontal(id="grids"):
            yield Button("2-box", id="grid_2")
            yield Button("4-box", id="grid_4")
            yield Button("8-box", id="grid_8")
            yield Button("16-box", id="grid_16")
            yield Button("Previous page", id="page_previous")
            yield Button("Next page", id="page_next")
        with Horizontal(id="settings"):
            yield Label("Fade seconds:")
            yield Input(str(self.default_fade_duration), id="fade_duration")
            yield Label("Wipe file:")
            yield Input(self.default_wipe_file, id="wipe_file")
        with Horizontal(id="takes"):
            yield Button("PREVIEW", id="preview_take", variant="primary")
            yield Button("CUT", id="cut", variant="error")
            yield Button("FADE", id="fade", variant="success")
            yield Button("WIPE", id="wipe", variant="warning")
            yield Button("RECONNECT", id="reconnect")
        yield Footer()

    def on_mount(self) -> None:
        self._poll_timer = self.set_interval(0.5, self._poll_status)
        self.action_reconnect()

    def _set_connection_text(self, text: str) -> None:
        self.query_one("#connection", Static).update(text)

    async def _fetch_scenes(self) -> None:
        content = await self.connection.command(f"mixer.scenes {self.mixer_name}")
        if content is None:
            raise RuntimeError("mixer.scenes returned no content")
        scenes = parse_scene_list(content)
        self.scenes = scenes
        scene_list = self.query_one("#scenes", ListView)
        await scene_list.clear()
        await scene_list.extend(ListItem(Label(scene)) for scene in scenes)
        if scenes:
            scene_list.index = 0
            self.selected_scene = scenes[0]

    async def _read_status(self) -> None:
        content = await self.connection.command(f"mixer.status {self.mixer_name}")
        if content is None:
            raise RuntimeError("mixer.status returned no content")
        status = parse_mixer_status(content)
        self.pgm_scene = status.pgm_scene
        self.pvw_scene = status.pvw_scene
        self.transition = status.transition
        self.query_one("#program_scene", Static).update(self.pgm_scene or "(none)")
        self.query_one("#preview_scene", Static).update(self.pvw_scene or "(none)")

    async def _poll_status(self) -> None:
        if not self.connection.connected:
            return
        try:
            await self._read_status()
        except Exception as exc:
            await self.connection.disconnect()
            self._set_connection_text(f"Connection lost: {exc}")

    @work(exclusive=True, group="connection")
    async def _reconnect(self) -> None:
        self._set_connection_text(
            f"Connecting to {self.connection.host}:{self.connection.port}..."
        )
        try:
            await self.connection.connect()
            await self._fetch_scenes()
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
    async def _take(self, transition: str) -> None:
        scene = self.selected_scene
        if not scene:
            self.notify("Select a scene first", severity="warning")
            return
        try:
            await self._preview_and_wait(scene)
            payload = {"scene": scene}
            if transition == "fade":
                duration = float(self.query_one("#fade_duration", Input).value)
                if duration <= 0:
                    raise ValueError("fade duration must be positive")
                payload["duration_sec"] = duration
            elif transition == "wipe":
                wipe_file = self.query_one("#wipe_file", Input).value.strip()
                if not wipe_file:
                    raise ValueError("wipe file is required")
                payload["wipe_file"] = wipe_file
            await self.connection.command(
                mixer_command(transition, self.mixer_name, **payload)
            )
            await self._read_status()
        except Exception as exc:
            self.notify(str(exc), severity="error")

    def action_cut(self) -> None:
        self._take("cut")

    def action_fade(self) -> None:
        self._take("fade")

    def action_wipe(self) -> None:
        self._take("wipe")

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
            index = self.scenes.index(scene)
        except ValueError:
            self.notify(f"Scene is unavailable: {scene}", severity="warning")
            return
        self.selected_scene = scene
        self.query_one("#scenes", ListView).index = index
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

    @on(ListView.Selected, "#scenes")
    def on_scene_selected(self, event: ListView.Selected) -> None:
        if 0 <= event.index < len(self.scenes):
            self.selected_scene = self.scenes[event.index]
            self._preview_selected()

    @on(Button.Pressed)
    def on_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("grid_"):
            self._select_grid_page(int(button_id.split("_", 1)[1]), 0)
        elif button_id in ("page_previous", "page_next"):
            selected = self._selected_grid()
            if selected:
                capacity, page = selected
                self._select_grid_page(
                    capacity,
                    page + (-1 if button_id == "page_previous" else 1),
                )
        elif button_id == "preview_take":
            self._preview_selected()
        elif button_id == "cut":
            self.action_cut()
        elif button_id == "fade":
            self.action_fade()
        elif button_id == "wipe":
            self.action_wipe()
        elif button_id == "reconnect":
            self.action_reconnect()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--mixer", default="mixer")
    parser.add_argument("--fade-duration", type=float, default=0.5)
    parser.add_argument("--wipe-file", default="")
    args = parser.parse_args(argv)
    MixerTui(
        args.host,
        args.port,
        args.mixer,
        fade_duration=args.fade_duration,
        wipe_file=args.wipe_file,
    ).run()


if __name__ == "__main__":
    main()
