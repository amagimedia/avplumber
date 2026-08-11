#!/usr/bin/env python3
"""Textual TUI for the single-playlist player.  Mirrors demos/replay/player.py:
a status header, a DataTable of clips, transport/mode/edit button rows, and an
ADD modal.  Runs against a live backend, or with --dry-run to print the command
stream (handy for eyeballing the AVP commands without a GPU)."""
from __future__ import annotations

import argparse

from playlist import (Clip, ElementMode, PlaylistController, PlaylistMode)

MODES = list(PlaylistMode)
EMODES = list(ElementMode)


def demo_clips() -> list[Clip]:
    return [
        Clip(url="/media/bumper.mp4", name="bumper.mp4", element_mode=ElementMode.TIMED,
             duration_ms=3000),
        Clip(url="/media/intro.mp4", name="intro.mp4"),
        Clip(url="/media/ad_break.mp4", name="ad_break.mp4", disabled=True),
        Clip(url="/media/sports.mp4", name="sports.mp4",
             element_mode=ElementMode.LOOP_SELF),
        Clip(url="/media/outro.mp4", name="outro.mp4"),
    ]


def make_backend(args) -> tuple[PlaylistController, object]:
    """Return (controller, app).  `app` is None in dry-run (no live graph)."""
    if args.dry_run:
        def sink(cmd: str) -> None:
            print(cmd)
        return PlaylistController(sink, demo_clips()), None
    # Live backend wiring is done by build_playlist_application (GPU host).
    from playlist_app import (JanusVideoConfig, PlaylistConfig,  # deferred import
                              build_playlist_application)
    config = PlaylistConfig(
        clips=demo_clips(),
        janus=JanusVideoConfig(host=args.janus_host, video_port=args.janus_video_port))
    app = build_playlist_application(config)
    app.start()
    return app.controller, app


try:
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import (Button, DataTable, Footer, Header, Input,
                                 Static)
    from textual.screen import ModalScreen
except ImportError:
    PlaylistTui = None
else:
    class AddClipModal(ModalScreen):
        def __init__(self, on_add):
            super().__init__()
            self._on_add = on_add

        def compose(self) -> ComposeResult:
            with Vertical(id="modal"):
                yield Static("+ ADD CLIP", id="modal-title")
                yield Input(placeholder="/media/file.mp4 or rtsp://…", id="url")
                yield Input(placeholder="from ms (0)", id="from")
                yield Input(placeholder="duration ms (Timed only)", id="dur")
                yield Input(placeholder="speed (1.0)", id="speed")
                with Horizontal():
                    for em in EMODES:
                        yield Button(em.value, id=f"em_{em.name}", classes="emode")
                with Horizontal():
                    yield Button("ADD", id="confirm", variant="success")
                    yield Button("CANCEL", id="cancel")
            self._emode = ElementMode.PLAY_TO_END

        @on(Button.Pressed, ".emode")
        def _pick(self, event):
            self._emode = ElementMode[event.button.id[3:]]
            for b in self.query(".emode"):
                b.variant = "primary" if b is event.button else "default"

        @on(Button.Pressed, "#cancel")
        def _cancel(self, _):
            self.dismiss(None)

        @on(Button.Pressed, "#confirm")
        def _confirm(self, _):
            url = self.query_one("#url", Input).value.strip()
            if not url:
                self.dismiss(None)
                return
            def val(sel, cast, default):
                raw = self.query_one(sel, Input).value.strip()
                return cast(raw) if raw else default
            try:
                clip = Clip(
                    url=url, element_mode=self._emode,
                    play_from_ms=val("#from", int, 0),
                    duration_ms=val("#dur", int, None),
                    speed=val("#speed", float, 1.0))
            except ValueError as exc:
                self.app.notify(str(exc), severity="error")
                return
            self._on_add(clip)
            self.dismiss(clip)

    class PlaylistTui(App):
        TITLE = "AVPlumber Playlist → Janus"
        CSS = """
        #state { height: 5; border: heavy $success; padding: 0 1; }
        #state.paused { border: heavy $warning; }
        #state.error { border: heavy $error; }
        .row { height: 3; }
        Button { margin-right: 1; }
        Button.active { background: $primary; }
        #modal { width: 60; height: auto; border: heavy $primary;
                 background: $surface; padding: 1; }
        DataTable { height: 1fr; }
        """
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("space", "toggle", "Play/Pause"),
            Binding("n", "next", "Next"),
            Binding("p", "prev", "Prev"),
            Binding("m", "emode", "Elem mode"),
            Binding("a", "add", "Add"),
            Binding("delete", "remove", "Remove"),
        ]

        def __init__(self, controller: PlaylistController):
            super().__init__()
            self.controller = controller

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static(id="state")
            yield DataTable(id="clips")
            with Vertical():
                with Horizontal(classes="row"):
                    for bid, label in (("toggle", "⏯ TOGGLE"), ("stop", "■ STOP"),
                                       ("prev", "|< PREV"), ("next", ">| NEXT")):
                        yield Button(label, id=bid)
                with Horizontal(classes="row"):
                    for mode in MODES:
                        yield Button(mode.value, id=f"mode_{mode.name}", classes="mode")
                with Horizontal(classes="row"):
                    for bid, label in (("add", "+ ADD"), ("remove", "- REMOVE"),
                                       ("up", "^ UP"), ("down", "v DOWN"),
                                       ("disable", "o EN/DIS"), ("emode", "<> MODE"),
                                       ("play", ">> PLAY")):
                        yield Button(label, id=bid)
            yield Footer()

        def on_mount(self):
            table = self.query_one("#clips", DataTable)
            table.add_columns("st", "#", "name", "element mode", "from", "to/dur", "spd")
            table.cursor_type = "row"
            self.set_interval(0.2, self.refresh_state)
            self.refresh_state()

        def refresh_state(self):
            st = self.controller.status()
            state = self.query_one("#state", Static)
            state.update(
                f"MODE={st.mode.value}  {'▶ PLAY' if st.playing else '⏸ PAUSE'}  "
                f"clip {('-' if st.current_index is None else st.current_index + 1)}"
                f"/{st.clip_count}  worker={st.active_worker}\n"
                f"{st.error or ('last: ' + st.last_command)}")
            state.set_class(bool(st.error), "error")
            state.set_class(not st.playing and not st.error, "paused")
            for b in self.query(".mode"):
                b.set_class(b.id == f"mode_{st.mode.name}", "active")
            self._fill_table(st)

        def _fill_table(self, st):
            table = self.query_one("#clips", DataTable)
            row = table.cursor_row
            table.clear()
            nxt = st.next_index
            for i, c in enumerate(self.controller.clips):
                marker = "··" if c.disabled else "ok"
                if i == nxt:
                    marker = "->"
                if i == st.current_index:
                    marker = ">>"
                to = (f"{c.duration_ms}ms" if c.element_mode == ElementMode.TIMED
                      else ("end" if c.play_to_ms is None else f"{c.play_to_ms}"))
                table.add_row(marker, str(i + 1), c.name, c.element_mode.value,
                              str(c.play_from_ms), to, f"{c.speed:g}")
            if row is not None and row < table.row_count:
                table.move_cursor(row=row)

        # -- helpers --------------------------------------------------------
        def _sel(self) -> int:
            return self.query_one("#clips", DataTable).cursor_row or 0

        def _do(self, fn, *a):
            try:
                fn(*a)
            except Exception as exc:
                self.notify(str(exc), severity="error")
            self.refresh_state()

        # -- actions --------------------------------------------------------
        def action_toggle(self): self._do(self.controller.toggle)
        def action_next(self): self._do(self.controller.next)
        def action_prev(self): self._do(self.controller.prev)
        def action_remove(self): self._do(self.controller.remove_clip, self._sel())

        def action_emode(self):
            i = self._sel()
            cur = self.controller.clips[i].element_mode
            nxt = EMODES[(EMODES.index(cur) + 1) % len(EMODES)]
            dur = 5000 if nxt == ElementMode.TIMED else None
            self._do(self.controller.set_element_mode, i, nxt, dur)

        def action_add(self):
            self.push_screen(AddClipModal(self.controller.append_clip))

        @on(Button.Pressed)
        def _button(self, event):
            bid = event.button.id or ""
            simple = {"toggle": self.controller.toggle, "stop": self.controller.stop,
                      "prev": self.controller.prev, "next": self.controller.next}
            if bid in simple:
                self._do(simple[bid])
            elif bid.startswith("mode_"):
                self._do(self.controller.set_mode, PlaylistMode[bid[5:]])
            elif bid == "add":
                self.action_add()
            elif bid == "remove":
                self.action_remove()
            elif bid == "up":
                i = self._sel(); self._do(self.controller.reorder_clip, i, max(0, i - 1))
            elif bid == "down":
                i = self._sel()
                self._do(self.controller.reorder_clip, i,
                         min(len(self.controller.clips) - 1, i + 1))
            elif bid == "disable":
                i = self._sel()
                self._do(self.controller.set_disabled, i,
                         not self.controller.clips[i].disabled)
            elif bid == "emode":
                self.action_emode()
            elif bid == "play":
                self._do(self.controller.goto, self._sel())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="print the AVP command stream instead of driving a backend")
    p.add_argument("--no-tui", action="store_true")
    p.add_argument("--janus-host", default="127.0.0.1")
    p.add_argument("--janus-video-port", type=int, default=5004)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    controller, app = make_backend(args)
    try:
        if args.no_tui or PlaylistTui is None:
            # non-interactive smoke: exercise the verbs, print commands
            controller.next(); controller.next(); controller.prev()
            controller.set_mode(PlaylistMode.LOOP_CURRENT)
            return 0
        PlaylistTui(controller).run()
        return 0
    finally:
        if app is not None:
            app.stop()


if __name__ == "__main__":
    raise SystemExit(main())
