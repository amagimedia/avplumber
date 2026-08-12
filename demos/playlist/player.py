#!/usr/bin/env python3
"""Clickable single-playlist regression UI with no terminal log panel."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import threading
import time
from pathlib import Path

from playlist import (Clip, ElementMode, InMemoryBackend, PlaylistController,
                      PlaylistMode, TransportState)

PLAYLIST_MODES = list(PlaylistMode)
ELEMENT_MODES = list(ElementMode)
FIXTURE_NAMES = (
    "01-testsrc2.mp4",
    "02-smpte.mp4",
    "03-smpte-hd.mp4",
    "04-rgb.mp4",
    "05-yuv.mp4",
)


def demo_clips(media_dir: Path | None = None) -> list[Clip]:
    directory = media_dir or Path(__file__).with_name("test-media")
    modes = (
        (ElementMode.PLAY_TO_END, None),
        (ElementMode.TIMED, 7000),
        (ElementMode.LOOP_SELF, None),
        (ElementMode.PLAY_TO_END, None),
        (ElementMode.TIMED, 10_000),
    )
    return [
        Clip(
            url=str((directory / filename).resolve()),
            name=filename,
            item_id=f"fixture-{index + 1}",
            element_mode=mode,
            duration_ms=duration,
        )
        for index, (filename, (mode, duration)) in enumerate(zip(FIXTURE_NAMES, modes))
    ]


def stop_application_bounded(application, timeout: float) -> bool:
    error = []

    def stop():
        try:
            application.stop()
        except Exception as exc:  # re-raised on the caller's thread
            error.append(exc)

    worker = threading.Thread(target=stop, name="playlist-stop", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return False
    if error:
        raise error[0]
    return True


class FrameworkStdoutRedirect:
    """Keep embedded command replies off Textual's terminal without C++ changes."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self._saved_fd = None
        self._saved_stdout = None
        self._stream = None

    def __enter__(self):
        path = Path(self.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout.flush()
        self._stream = path.open("a", buffering=1)
        self._saved_fd = os.dup(1)
        os.dup2(self._stream.fileno(), 1)
        self._saved_stdout = sys.stdout
        sys.stdout = self._stream
        return self

    def __exit__(self, _exception_type, _exception, _traceback):
        if self._saved_fd is None:
            return
        sys.stdout.flush()
        # AVPlumber writes command replies through std::cout. With the default
        # synchronized iostreams, flushing libc drains those bytes before fd 1
        # is restored to Textual's terminal.
        ctypes.CDLL(None).fflush(None)
        os.dup2(self._saved_fd, 1)
        os.close(self._saved_fd)
        self._saved_fd = None
        sys.stdout = self._saved_stdout
        self._saved_stdout = None
        self._stream.close()
        self._stream = None


try:
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Button, DataTable, Header, Input, Static
except ImportError:
    PlaylistTui = None
else:
    class ClipModal(ModalScreen):
        """Add or edit one element without exposing backend command text."""

        def __init__(self, callback, clip: Clip | None = None):
            super().__init__()
            self._callback = callback
            self._clip = clip
            self._mode = clip.element_mode if clip else ElementMode.PLAY_TO_END

        def compose(self) -> ComposeResult:
            clip = self._clip
            with Vertical(id="clip-modal"):
                yield Static("EDIT ELEMENT" if clip else "ADD ELEMENT", id="modal-title")
                yield Input(value=clip.url if clip else "", placeholder="media path", id="edit-url")
                yield Input(value=clip.name if clip else "", placeholder="display name", id="edit-name")
                yield Input(value=str(clip.play_from_ms) if clip else "0",
                            placeholder="cue-in ms", id="edit-from")
                yield Input(value="" if clip is None or clip.play_to_ms is None
                            else str(clip.play_to_ms), placeholder="cue-out ms", id="edit-to")
                yield Input(value="" if clip is None or clip.duration_ms is None
                            else str(clip.duration_ms), placeholder="Timed duration ms", id="edit-duration")
                yield Input(value=str(clip.speed) if clip else "1.0",
                            placeholder="speed", id="edit-speed")
                with Horizontal(classes="control-row"):
                    for mode in ELEMENT_MODES:
                        yield Button(
                            mode.value, id=f"edit_mode_{mode.name}",
                            classes="edit-mode" + (" active" if mode is self._mode else ""))
                with Horizontal(classes="control-row"):
                    yield Button("SAVE", id="edit-save", variant="success")
                    yield Button("CANCEL", id="edit-cancel")

        @on(Button.Pressed, ".edit-mode")
        def choose_mode(self, event):
            self._mode = ElementMode[event.button.id[len("edit_mode_"):]]
            for button in self.query(".edit-mode"):
                button.set_class(button is event.button, "active")

        @on(Button.Pressed, "#edit-cancel")
        def cancel(self):
            self.dismiss(None)

        @on(Button.Pressed, "#edit-save")
        def save(self):
            def text(selector):
                return self.query_one(selector, Input).value.strip()

            def optional_int(selector):
                value = text(selector)
                return None if not value else int(value)

            try:
                url = text("#edit-url")
                if not url:
                    raise ValueError("media path is required")
                duration = optional_int("#edit-duration")
                if self._mode is ElementMode.TIMED and duration is None:
                    raise ValueError("Timed element requires duration")
                item_id = self._clip.item_id if self._clip else ""
                values = dict(
                    url=url,
                    name=text("#edit-name"),
                    element_mode=self._mode,
                    play_from_ms=int(text("#edit-from") or "0"),
                    play_to_ms=optional_int("#edit-to"),
                    duration_ms=duration,
                    speed=float(text("#edit-speed") or "1"),
                )
                if item_id:
                    values["item_id"] = item_id
                result = Clip(**values)
            except ValueError as exc:
                self.app.notify(str(exc), severity="error")
                return
            self._callback(result)
            self.dismiss(result)


    class PlaylistTui(App):
        TITLE = "AVPlumber Playlist → Janus"
        CSS = """
        Screen { layout: vertical; }
        #state { height: 4; border: heavy $success; padding: 0 1; }
        #state.stopped { border: heavy $warning; }
        #state.error { border: heavy $error; }
        #clips { height: 1fr; }
        #controls { height: 12; border: round $primary; padding: 0 1; }
        .section-label { height: 1; color: $text-muted; }
        .control-row { height: 3; }
        Button { min-width: 10; margin-right: 1; }
        Button.active { background: $primary; }
        #clip-modal { width: 72; height: auto; border: heavy $primary;
                      background: $surface; padding: 1; }
        """
        BINDINGS = [
            Binding("q", "quit", "", show=False),
            Binding("space", "list_toggle", "", show=False),
            Binding("s", "list_stop", "", show=False),
            Binding("n", "list_next", "", show=False),
            Binding("p", "list_prev", "", show=False),
            Binding("enter", "item_play", "", show=False),
            Binding("u", "item_pause", "", show=False),
            Binding("x", "item_stop", "", show=False),
            Binding("m", "item_mode", "", show=False),
            Binding("e", "item_enable", "", show=False),
            Binding("a", "item_add", "", show=False),
            Binding("delete", "item_remove", "", show=False),
            Binding("1", "list_mode('PLAY_ALL')", "", show=False),
            Binding("2", "list_mode('PLAY_CURRENT')", "", show=False),
            Binding("3", "list_mode('LOOP_ALL')", "", show=False),
            Binding("4", "list_mode('LOOP_CURRENT')", "", show=False),
        ]

        def __init__(self, controller: PlaylistController):
            super().__init__()
            self.controller = controller

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static(id="state")
            yield DataTable(id="clips")
            with Vertical(id="controls"):
                yield Static("PLAYLIST", classes="section-label")
                with Horizontal(classes="control-row"):
                    for button_id, label in (
                        ("list-play", "LIST PLAY"),
                        ("list-pause", "LIST PAUSE"),
                        ("list-stop", "LIST STOP"),
                        ("list-prev", "LIST PREV"),
                        ("list-next", "LIST NEXT"),
                    ):
                        yield Button(label, id=button_id)
                with Horizontal(classes="control-row"):
                    for mode in PLAYLIST_MODES:
                        yield Button(
                            mode.value, id=f"list-mode-{mode.name}", classes="list-mode")
                yield Static("SELECTED ELEMENT", classes="section-label")
                with Horizontal(classes="control-row"):
                    for button_id, label in (
                        ("item-play", "ITEM PLAY"),
                        ("item-pause", "ITEM PAUSE"),
                        ("item-stop", "ITEM STOP"),
                        ("item-mode", "ITEM MODE"),
                        ("item-edit", "EDIT"),
                        ("item-enable", "ENABLE"),
                        ("item-add", "ADD"),
                        ("item-remove", "REMOVE"),
                        ("item-up", "UP"),
                        ("item-down", "DOWN"),
                    ):
                        yield Button(label, id=button_id)

        def on_mount(self):
            table = self.query_one("#clips", DataTable)
            table.add_columns("state", "#", "name", "element mode",
                              "cue-in", "cue-out/duration", "speed")
            table.cursor_type = "row"
            self._poll_timer = self.set_interval(0.05, self._poll)
            self.refresh_state()

        def on_unmount(self):
            if hasattr(self, "_poll_timer"):
                self._poll_timer.pause()

        def _poll(self):
            self.controller.poll(int(time.monotonic() * 1000))
            if self.is_mounted and list(self.query("#state")):
                self.refresh_state()

        def _selected(self) -> int:
            table = self.query_one("#clips", DataTable)
            row = table.cursor_row
            return 0 if row is None else min(row, len(self.controller.clips) - 1)

        def _select_controller_row(self) -> int:
            index = self._selected()
            self.controller.select(index)
            return index

        def _do(self, operation, *args):
            try:
                operation(*args)
            except Exception as exc:
                self.controller.set_error(str(exc))
                self.notify(str(exc), severity="error")
            self.refresh_state()

        def refresh_state(self):
            status = self.controller.status()
            state = self.query_one("#state", Static)
            active = "-" if status.active_index is None else str(status.active_index + 1)
            selected = "-" if status.selected_index is None else str(status.selected_index + 1)
            pending = "-" if status.pending_index is None else str(status.pending_index + 1)
            janus = "ALIVE" if status.output_alive else "NOT READY"
            state.update(
                f"PLAYLIST {status.mode.value}  {status.transport.value.upper()}  "
                f"JANUS {janus}\nselected={selected} active={active} pending={pending}"
                + (f"  ERROR: {status.error}" if status.error else "")
            )
            state.set_class(bool(status.error), "error")
            state.set_class(
                status.transport in (TransportState.STOPPED, TransportState.PAUSED)
                and not status.error, "stopped")
            for button in self.query(".list-mode"):
                button.set_class(
                    button.id == f"list-mode-{status.mode.name}", "active")
            self._fill_table(status)

        def _fill_table(self, status):
            table = self.query_one("#clips", DataTable)
            cursor = table.cursor_row
            table.clear()
            for index, clip in enumerate(self.controller.clips):
                item_state = self.controller.element_state(index)
                marker = {
                    TransportState.PLAYING: "PLAY",
                    TransportState.PAUSED: "PAUSE",
                    TransportState.LOADING: "LOAD",
                    TransportState.STOPPED: "STOP",
                }[item_state]
                if clip.disabled:
                    marker = "OFF"
                value = (f"{clip.duration_ms}ms" if clip.element_mode is ElementMode.TIMED
                         else ("end" if clip.play_to_ms is None else f"{clip.play_to_ms}ms"))
                table.add_row(marker, str(index + 1), clip.name,
                              clip.element_mode.value, f"{clip.play_from_ms}ms",
                              value, f"{clip.speed:g}")
            if table.row_count:
                desired = status.selected_index if cursor is None else cursor
                table.move_cursor(row=min(desired or 0, table.row_count - 1))

        @on(DataTable.RowHighlighted, "#clips")
        def row_highlighted(self, event):
            if self.controller.clips and 0 <= event.cursor_row < len(self.controller.clips):
                self.controller.select(event.cursor_row)

        # Playlist actions.
        def action_list_toggle(self):
            self._do(self.controller.toggle)

        def action_list_stop(self):
            self._do(self.controller.stop)

        def action_list_next(self):
            self._do(self.controller.next)

        def action_list_prev(self):
            self._do(self.controller.prev)

        def action_list_mode(self, mode_name: str):
            self._do(self.controller.set_mode, PlaylistMode[mode_name])

        # Selected-item actions.
        def action_item_play(self):
            self._do(self.controller.element_play, self._select_controller_row())

        def action_item_pause(self):
            self._do(self.controller.element_pause, self._select_controller_row())

        def action_item_stop(self):
            self._do(self.controller.element_stop, self._select_controller_row())

        def action_item_mode(self):
            index = self._select_controller_row()
            current = self.controller.clips[index].element_mode
            mode = ELEMENT_MODES[(ELEMENT_MODES.index(current) + 1) % len(ELEMENT_MODES)]
            duration = self.controller.clips[index].duration_ms
            if mode is ElementMode.TIMED and duration is None:
                duration = 5000
            self._do(self.controller.set_element_mode, index, mode, duration)

        def action_item_enable(self):
            index = self._select_controller_row()
            self._do(self.controller.set_disabled, index,
                     not self.controller.clips[index].disabled)

        def action_item_add(self):
            self.push_screen(ClipModal(self.controller.append_clip))

        def action_item_edit(self):
            index = self._select_controller_row()

            def save(clip):
                self.controller.replace_clip(index, clip)

            self.push_screen(ClipModal(save, self.controller.clips[index]))

        def action_item_remove(self):
            self._do(self.controller.remove_clip, self._select_controller_row())

        def action_item_up(self):
            index = self._select_controller_row()
            if index > 0:
                self._do(self.controller.reorder_clip, index, index - 1)

        def action_item_down(self):
            index = self._select_controller_row()
            if index + 1 < len(self.controller.clips):
                self._do(self.controller.reorder_clip, index, index + 1)

        @on(Button.Pressed)
        def button_pressed(self, event):
            button_id = event.button.id or ""
            actions = {
                "list-play": lambda: self._do(self.controller.play),
                "list-pause": lambda: self._do(self.controller.pause),
                "list-stop": self.action_list_stop,
                "list-prev": self.action_list_prev,
                "list-next": self.action_list_next,
                "item-play": self.action_item_play,
                "item-pause": self.action_item_pause,
                "item-stop": self.action_item_stop,
                "item-mode": self.action_item_mode,
                "item-edit": self.action_item_edit,
                "item-enable": self.action_item_enable,
                "item-add": self.action_item_add,
                "item-remove": self.action_item_remove,
                "item-up": self.action_item_up,
                "item-down": self.action_item_down,
            }
            if button_id in actions:
                actions[button_id]()
            elif button_id.startswith("list-mode-"):
                self.action_list_mode(button_id[len("list-mode-"):])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="run the non-printing in-memory backend")
    parser.add_argument("--no-tui", action="store_true",
                        help="perform a short headless control smoke")
    parser.add_argument("--media-dir", type=Path,
                        default=Path(__file__).with_name("test-media"))
    parser.add_argument("--janus-host", default="127.0.0.1")
    parser.add_argument("--janus-video-port", type=int, default=5004)
    parser.add_argument("--janus-video-pt", type=int, default=96)
    parser.add_argument("--janus-video-ssrc", type=lambda value: int(value, 0),
                        default=0x41565001)
    parser.add_argument("--janus-rtcp-bind", default="0.0.0.0")
    parser.add_argument("--janus-rtcp-port", type=int, default=0)
    parser.add_argument("--log-file", default="playlist-demo.log")
    parser.add_argument("--control-timeout", type=float, default=10.0)
    return parser.parse_args(argv)


def make_backend(args):
    clips = demo_clips(args.media_dir)
    if args.dry_run:
        backend = InMemoryBackend()
        controller = PlaylistController(backend, clips)
        controller.play()
        controller.poll(int(time.monotonic() * 1000))
        return controller, None

    from playlist_app import (JanusVideoConfig, PlaylistConfig,
                              build_playlist_application)
    config = PlaylistConfig(
        clips=clips,
        janus=JanusVideoConfig(
            host=args.janus_host,
            video_port=args.janus_video_port,
            payload_type=args.janus_video_pt,
            ssrc=args.janus_video_ssrc,
            rtcp_bind=args.janus_rtcp_bind,
            rtcp_port=args.janus_rtcp_port,
        ),
        control_timeout=args.control_timeout,
        log_file=args.log_file,
    )
    application = build_playlist_application(config)
    application.start()
    return application.controller, application


def main(argv=None) -> int:
    args = parse_args(argv)
    redirect = None
    if not args.dry_run:
        redirect = FrameworkStdoutRedirect(args.log_file)
        redirect.__enter__()
    application = None
    try:
        controller, application = make_backend(args)
        if args.no_tui:
            controller.next()
            controller.poll(int(time.monotonic() * 1000))
            controller.pause()
            controller.play()
            controller.stop()
            return 0
        if PlaylistTui is None:
            raise RuntimeError("Textual is required; install demos/playlist/requirements.txt")
        PlaylistTui(controller).run()
        return 0
    finally:
        try:
            if application is not None and not stop_application_bounded(
                    application, args.control_timeout):
                raise TimeoutError("playlist application shutdown timed out")
        finally:
            if redirect is not None:
                redirect.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
