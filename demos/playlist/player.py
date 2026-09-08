#!/usr/bin/env python3
"""Playlist terminal UI.

Attaches to a running ``server.py`` over AVPlumber's control port, or runs a
``--dry-run`` against an in-memory controller with no video.  Two action bars:
one for the playlist, one for the selected element.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, Select, Static

from control import LocalClient, RemoteClient, Snapshot
from playlist import ElementMode, InMemoryBackend, PlaylistController, PlaylistMode, Transition

MODES = [(m.value, m.value) for m in PlaylistMode]
TRANSITIONS = [(t.value, t.value) for t in Transition]
ENDS = [m.value for m in ElementMode]
END_LABEL = {"PlayToEnd": "Play", "Timed": "Timed", "LoopSelf": "Loop"}
COLUMNS = (" ", "#", "Name", "Cue in", "Cue out", "Length", "End", "Speed")


def clock(ms: Optional[int], blank: str = "") -> str:
    if ms is None:
        return blank
    minutes, rest = divmod(max(0, int(ms)), 60_000)
    return f"{minutes}:{rest // 1000:02d}.{rest % 1000:03d}"


def bar(position: Optional[int], start: int, end: Optional[int], width: int = 16) -> str:
    if position is None or end is None or end <= start:
        return "─" * width
    filled = round(min(1.0, max(0.0, (position - start) / (end - start))) * (width - 1))
    return "━" * filled + "●" + "─" * (width - 1 - filled)


class EditScreen(ModalScreen):
    """Element settings; on Save posts ``edit`` (existing) or ``add`` (new)."""

    DEFAULT_CSS = """
    EditScreen { align: center middle; }
    #dialog { width: 76; height: auto; border: round $primary; background: $surface; padding: 1 2; }
    #dialog .field { height: 3; }
    #dialog .field Label { width: 26; margin-top: 1; color: $text-muted; }
    #dialog .field Input { width: 1fr; }
    #dialog-actions { height: auto; margin-top: 1; }
    #dialog-actions Button { margin-right: 1; }
    #dialog-error { color: $error; height: auto; }
    """

    FIELDS = (("name", "Name", ""), ("url", "Path", ""), ("cue_in", "Cue in (ms)", "0"),
              ("cue_out", "Cue out (ms, blank = end)", ""), ("duration", "Timed length (ms)", ""),
              ("speed", "Speed", "1.0"))

    def __init__(self, clip: Optional[dict], index: Optional[int]):
        super().__init__()
        self.clip, self.index = clip, index

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Edit element" if self.clip else "Add element", classes="title")
            for key, label, default in self.FIELDS:
                value = default if self.clip is None else self.clip.get(key)
                with Horizontal(classes="field"):
                    yield Label(label)
                    yield Input(value="" if value is None else str(value), id=f"field-{key}")
            yield Static("", id="dialog-error")
            with Horizontal(id="dialog-actions"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    def values(self) -> dict:
        data = {key: self.query_one(f"#field-{key}", Input).value.strip() for key, _, _ in self.FIELDS}
        if not data["url"]:
            raise ValueError("path is required")
        for key in ("cue_in", "cue_out", "duration"):
            data[key] = int(data[key]) if data[key] else None
        data["cue_in"] = data["cue_in"] or 0
        data["speed"] = float(data["speed"] or 1.0)
        if self.clip:
            data["end"] = self.clip["end"]
            data["disabled"] = self.clip["disabled"]
        return data

    @on(Button.Pressed, "#save")
    def save(self) -> None:
        try:
            self.dismiss(self.values())
        except ValueError as exc:
            self.query_one("#dialog-error", Static).update(str(exc))

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(None)


class PlaylistTui(App):
    TITLE = "Playlist"
    CSS = """
    Screen { layout: vertical; }
    #top { height: 1fr; }
    #clips { width: 1fr; }
    #onair { width: 38; border: round $primary-darken-2; padding: 0 1; }
    #onair .big { text-style: bold; }
    #onair .live { color: $success; }
    #onair .dead { color: $error; }
    .bar { height: auto; border: round $primary-darken-2; padding: 0 1; }
    .bar Button { width: 11; min-width: 11; margin-right: 1; }
    #el-up, #el-down { width: 5; min-width: 5; }
    .bar Select { width: 14; margin-right: 1; }
    .bar Input { width: 9; }
    .bar Label { margin: 1 1 0 0; color: $text-muted; }
    #el-take { background: $error-darken-2; }
    #error { height: 1; color: $error; padding: 0 1; }
    #hints { height: 1; color: $text-muted; padding: 0 1; }
    """
    BINDINGS = [
        Binding("space", "toggle", "play/pause", show=False),
        Binding("s", "pl('stop')", show=False),
        Binding("n", "pl('next')", show=False),
        Binding("p", "pl('prev')", show=False),
        Binding("u", "el('hold')", show=False),
        Binding("x", "el('park')", show=False),
        Binding("e", "edit", show=False),
        Binding("m", "cycle_end", show=False),
        Binding("o", "toggle_enabled", show=False),
        Binding("a", "add", show=False),
        Binding("delete", "el('remove')", show=False),
        Binding("q", "quit", show=False),
    ]

    def __init__(self, client, poll_interval: float = 0.1):
        super().__init__()
        self.client = client
        self.poll_interval = poll_interval
        self.snapshot: Optional[Snapshot] = None
        self._row_keys: list = []
        self._shown_selected: Optional[int] = None

    # ---- layout --------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            yield DataTable(id="clips", cursor_type="row", zebra_stripes=True)
            with Vertical(id="onair"):
                yield Static("", id="oa-name", classes="big")
                yield Static("", id="oa-bar")
                yield Static("", id="oa-next")
                yield Static("", id="oa-transition")
                yield Static("")
                yield Static("", id="oa-mode")
                yield Static("", id="oa-janus")
        with Horizontal(id="bar-playlist", classes="bar"):
            yield Button("▶ Play", id="pl-play")
            yield Button("⏸ Pause", id="pl-pause")
            yield Button("■ Stop", id="pl-stop")
            yield Button("⏮ Prev", id="pl-prev")
            yield Button("⏭ Next", id="pl-next")
            yield Label("Mode")
            yield Select(MODES, value=PlaylistMode.LOOP_ALL.value, allow_blank=False, id="pl-mode")
            yield Label("Transition")
            yield Select(TRANSITIONS, value=Transition.CUT.value, allow_blank=False, id="pl-transition")
            yield Input("500", id="pl-transition-ms", type="integer", tooltip="ms")
        with Horizontal(id="bar-element", classes="bar"):
            yield Button("▶ Take", id="el-take")
            yield Button("⏸ Pause", id="el-hold")
            yield Button("■ Stop", id="el-park")
            yield Button("✎ Edit", id="el-edit")
            yield Button("End: Play", id="el-end")
            yield Button("Off", id="el-onoff")
            yield Button("↑", id="el-up")
            yield Button("↓", id="el-down")
            yield Button("+ Add", id="el-add")
            yield Button("− Remove", id="el-remove")
        yield Static("", id="error")
        yield Static("space play/pause · s stop · n/p prev/next · enter take · u/x hold/park · e edit"
                     " · a add · del remove · q quit", id="hints")

    def on_mount(self) -> None:
        self.query_one("#bar-playlist").border_title = "PLAYLIST"
        self.query_one("#onair").border_title = "ON AIR"
        table = self.query_one("#clips", DataTable)
        table.add_columns(*COLUMNS)
        table.focus()
        self.set_interval(self.poll_interval, self.refresh_status)
        self.refresh_status()

    # ---- polling -------------------------------------------------------
    @work(exclusive=True, group="status")
    async def refresh_status(self) -> None:
        try:
            if not self.client.connected:
                await self.client.connect()
            snapshot = await self.client.status()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#error", Static).update(f"backend unreachable: {exc}")
            return
        self.snapshot = snapshot
        try:
            self.render_snapshot(snapshot)
        except NoMatches:
            pass                                      # screen torn down mid-refresh

    def render_snapshot(self, s: Snapshot) -> None:
        clips = s.clips
        table = self.query_one("#clips", DataTable)
        keys = [c["id"] for c in clips]
        rebuilt = keys != self._row_keys
        if rebuilt:
            table.clear()
            for c in clips:
                table.add_row(*self.row(c, s), key=c["id"])
            self._row_keys = keys
        else:
            for c in clips:
                for column, value in zip(table.columns, self.row(c, s)):
                    table.update_cell(c["id"], column, value)
        if s.selected is not None and (rebuilt or s.selected != self._shown_selected):
            table.move_cursor(row=s.selected)      # backend-side selection change or table rebuild
        self._shown_selected = s.selected

        active = clips[s.active] if s.active is not None else None
        self.query_one("#oa-name", Static).update(
            f"▶ {active['name']}" if active and s.playing else
            f"⏸ {active['name']}" if active and s.transport == "Paused" else
            f"… {clips[s.pending]['name']}" if s.pending is not None else "■ stopped")
        if active:
            end = active["cue_out"] if active["end"] != "Timed" else None
            if end is None and s.end_ms is not None and s.position_ms is not None:
                end = s.position_ms + int((s.end_ms - s.now_ms) * active["speed"])
            self.query_one("#oa-bar", Static).update(
                f"{clock(s.position_ms, '-:--.---')} {bar(s.position_ms, active['cue_in'], end)} {clock(end, 'loop')}")
        else:
            self.query_one("#oa-bar", Static).update("")
        nxt = clips[s.next] if s.next is not None else None
        wait = s.seconds_to_next()
        self.query_one("#oa-next", Static).update(
            f"next   {nxt['name']}" + (f"  in {wait:4.1f} s" if wait is not None else "") if nxt else "next   —")
        self.query_one("#oa-transition", Static).update(
            f"{s.transition.lower():6} {s.transition_ms} ms" if s.transition != "Cut" else "cut    armed natively")
        self.query_one("#oa-mode", Static).update(f"mode   {s.mode}")
        janus = self.query_one("#oa-janus", Static)
        janus.update(f"janus  {'● live' if s.output_alive else '○ no output'}")
        janus.set_class(s.output_alive, "live")
        janus.set_class(not s.output_alive, "dead")

        sel = clips[s.selected] if s.selected is not None else None
        self.query_one("#bar-element").border_title = sel["name"].upper() if sel else "ELEMENT"
        self.query_one("#el-end", Button).label = f"End: {END_LABEL[sel['end']]}" if sel else "End"
        self.query_one("#el-onoff", Button).label = ("On" if sel["disabled"] else "Off") if sel else "Off"
        self.query_one("#pl-play", Button).label = "▶ Resume" if s.transport == "Paused" else "▶ Play"
        for widget_id, value in (("pl-mode", s.mode), ("pl-transition", s.transition)):
            select = self.query_one(f"#{widget_id}", Select)
            if select.value != value:
                with select.prevent(Select.Changed):
                    select.value = value
        error = s.error
        if error and s.error_index is not None:
            error = f"{clips[s.error_index]['name']}: {error}"
        self.query_one("#error", Static).update(error)

    @staticmethod
    def row(c: dict, s: Snapshot) -> tuple:
        index = s.clips.index(c)
        mark = "▶" if index == s.active and s.transport != "Stopped" else \
               "…" if index == s.pending else "›" if index == s.selected else "·" if c["disabled"] else " "
        name = c["name"] + ("  off" if c["disabled"] else "")
        length = c["duration"] if c["end"] == "Timed" else (
            None if c["cue_out"] is None else int((c["cue_out"] - c["cue_in"]) / c["speed"]))
        return (mark, str(index + 1), name, clock(c["cue_in"]), clock(c["cue_out"], "end"),
                clock(length, "media"), END_LABEL[c["end"]], f"{c['speed']:g}×")

    # ---- actions -------------------------------------------------------
    @property
    def selected(self) -> Optional[int]:
        return None if self.snapshot is None else self.snapshot.selected

    @work(group="send")
    async def send(self, verb: str, **arg) -> None:
        try:
            await self.client.send(verb, **arg)
        except Exception as exc:  # noqa: BLE001
            self.query_one("#error", Static).update(str(exc))
        self.refresh_status()

    def action_toggle(self) -> None:
        self.send("toggle")

    def action_pl(self, verb: str) -> None:
        self.send(verb)

    def action_el(self, verb: str) -> None:
        if self.selected is not None:
            self.send(verb, index=self.selected)

    def action_cycle_end(self) -> None:
        if self.snapshot is None or self.selected is None:
            return
        clip = self.snapshot.clips[self.selected]
        end = ENDS[(ENDS.index(clip["end"]) + 1) % len(ENDS)]
        duration = clip["duration"] or (clip["cue_out"] or 10_000) - clip["cue_in"]
        self.send("end_mode", index=self.selected, end=end, duration=duration)

    def action_toggle_enabled(self) -> None:
        if self.snapshot is not None and self.selected is not None:
            self.send("enable", index=self.selected, enabled=self.snapshot.clips[self.selected]["disabled"])

    def action_edit(self) -> None:
        if self.snapshot is None or self.selected is None:
            return
        index = self.selected

        def done(values):
            if values:
                self.send("edit", index=index, clip=values)
        self.push_screen(EditScreen(self.snapshot.clips[index], index), done)

    def action_add(self) -> None:
        def done(values):
            if values:
                self.send("add", clip=values, index=None if self.selected is None else self.selected + 1)
        self.push_screen(EditScreen(None, None), done)

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if not bid.startswith(("pl-", "el-")):
            return                                    # dialog buttons handle themselves
        self.screen_stack[0].query_one("#clips", DataTable).focus()   # keep arrow keys on the table
        if bid.startswith("pl-"):
            self.send(bid[3:])
        elif bid == "el-edit":
            self.action_edit()
        elif bid == "el-add":
            self.action_add()
        elif bid == "el-end":
            self.action_cycle_end()
        elif bid == "el-onoff":
            self.action_toggle_enabled()
        elif bid in ("el-up", "el-down") and self.selected is not None:
            to = self.selected + (-1 if bid == "el-up" else 1)
            if 0 <= to < len(self.snapshot.clips):
                self.send("move", index=self.selected, to=to)
        elif bid.startswith("el-"):
            self.action_el(bid[3:])

    @on(Select.Changed, "#pl-mode")
    def mode_changed(self, event: Select.Changed) -> None:
        if self.snapshot and event.value != self.snapshot.mode:
            self.send("mode", mode=event.value)

    @on(Select.Changed, "#pl-transition")
    def transition_changed(self, event: Select.Changed) -> None:
        if self.snapshot and event.value != self.snapshot.transition:
            self.send("transition", transition=event.value)

    @on(Input.Submitted, "#pl-transition-ms")
    def transition_ms_changed(self, event: Input.Submitted) -> None:
        if event.value.isdigit() and self.snapshot:
            self.send("transition", transition=self.snapshot.transition, duration_ms=int(event.value))

    @on(DataTable.RowHighlighted, "#clips")
    def highlighted(self, event: DataTable.RowHighlighted) -> None:
        stale = event.cursor_row != event.data_table.cursor_row     # left over from a table rebuild
        if self.snapshot is not None and not stale and event.cursor_row != self.snapshot.selected:
            self.send("select", index=event.cursor_row)

    @on(DataTable.RowSelected, "#clips")
    def row_selected(self, event: DataTable.RowSelected) -> None:
        self.send("take", index=event.cursor_row)


def dry_run_client(media_dir: Path) -> LocalClient:
    from server import default_clips
    now = lambda: time.monotonic_ns() // 1_000_000  # noqa: E731
    ctl = PlaylistController(InMemoryBackend(clock=now), default_clips(media_dir))
    return LocalClient(ctl)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7778)
    p.add_argument("--dry-run", action="store_true", help="in-memory controller, no backend")
    p.add_argument("--media-dir", type=Path, default=Path(__file__).resolve().parent / "test-media")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    client = dry_run_client(args.media_dir) if args.dry_run else RemoteClient(args.host, args.port)
    PlaylistTui(client).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
