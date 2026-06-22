import os
from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static


class WipeModal(ModalScreen[Optional[str]]):
    """Modal dialog to select or enter a wipe file path."""

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
    WipeModal ListView {
        height: 12;
        margin-bottom: 1;
        border: solid $primary;
    }
    WipeModal ListItem {
        height: 1;
    }
    WipeModal #wipe_empty {
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

    def __init__(
        self,
        current_path: Optional[str] = None,
        *,
        wipe_dir: Optional[str] = None,
        wipe_files: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self._current_path = current_path or ""
        self._wipe_dir = wipe_dir or ""
        self._wipe_files = wipe_files or []

    def compose(self) -> ComposeResult:
        with Vertical():
            if self._wipe_files:
                label = f"Wipe files in {self._wipe_dir}:" if self._wipe_dir else "Wipe files:"
                yield Label(label)
                yield ListView(
                    *[
                        ListItem(Label(os.path.basename(path)))
                        for path in self._wipe_files
                    ],
                    id="wipe_list",
                )
                yield Label("Path:")
            else:
                yield Label("Wipe file path:")
                if self._wipe_dir:
                    yield Label(f"No .mov files found in {self._wipe_dir}", id="wipe_empty")
                else:
                    yield Label("No wipe directory configured.", id="wipe_empty")
                    yield Label("Use --wipe-dir PATH or AVP_WIPE_DIR.", id="wipe_empty_hint")
            yield Input(
                value=self._current_path,
                placeholder="/path/to/wipe.mov",
                id="wipe_path",
            )
            with Horizontal():
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("OK", variant="primary", id="ok")

    def on_mount(self) -> None:
        if self._wipe_files:
            self.query_one("#wipe_list", ListView).focus()
            return
        inp = self.query_one("#wipe_path", Input)
        inp.focus()
        inp.cursor_position = len(inp.value)

    @on(ListView.Selected, "#wipe_list")
    def on_wipe_selected(self, event: ListView.Selected) -> None:
        if event.index < 0 or event.index >= len(self._wipe_files):
            return
        path = self._wipe_files[event.index]
        inp = self.query_one("#wipe_path", Input)
        inp.value = path
        inp.cursor_position = len(path)
        inp.focus()

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


class FadeFramesInput(Input):
    """Fade-frame editor that preserves the AI transition shortcut."""

    def on_key(self, event) -> None:
        if event.key == "a":
            self.app.action_toggle_ai_transition_mode()
            event.stop()
