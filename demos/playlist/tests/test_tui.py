"""Rendered Textual surface against the in-memory controller."""

import asyncio

import pytest

pytest.importorskip("textual")

from textual.widgets import Button, DataTable, Select, Static  # noqa: E402

from control import LocalClient  # noqa: E402
from helpers import Clock, clips  # noqa: E402
from player import EditScreen, PlaylistTui, bar, clock  # noqa: E402
from playlist import InMemoryBackend, PlaylistController  # noqa: E402


def make_app():
    tick = Clock()
    backend = InMemoryBackend(clock=tick)
    ctl = PlaylistController(backend, clips("a", "b", "c", "d", "e"))
    return PlaylistTui(LocalClient(ctl, clock=tick), poll_interval=0.02), ctl, backend, tick


def run(coro):
    return asyncio.run(coro)


def test_formatting_helpers():
    assert clock(61_005) == "1:01.005" and clock(None, "end") == "end"
    assert bar(5000, 0, 10_000, width=11) == "━━━━━●─────"
    assert bar(None, 0, 10_000, width=4) == "────"


def test_two_action_bars_group_playlist_and_element_controls():
    async def scenario():
        app, *_ = make_app()
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause()
            ids = {b.id for b in app.query(Button)}
            assert ids == {"pl-play", "pl-pause", "pl-stop", "pl-prev", "pl-next",
                           "el-take", "el-hold", "el-park", "el-edit", "el-end", "el-onoff",
                           "el-up", "el-down", "el-add", "el-remove"}
            labels = " ".join(str(b.label) for b in app.query(Button)).lower()
            assert "item" not in labels and "list" not in labels
            assert app.query_one("#bar-playlist").border_title == "PLAYLIST"
            assert app.query_one("#bar-element").border_title == "A"
            assert all(b.region.width > 0 for b in app.query(Button))
    run(scenario())


def test_play_take_and_navigation_drive_the_controller():
    async def scenario():
        app, ctl, backend, tick = make_app()
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause()
            await pilot.click("#pl-play")
            await pilot.pause(0.1)
            assert ctl.status().active_index == 0
            assert "▶ a" in str(app.query_one("#oa-name", Static).render())
            assert "next   b" in str(app.query_one("#oa-next", Static).render())
            await pilot.press("down", "down")
            await pilot.pause(0.1)
            assert ctl.status().selected_index == 2
            assert app.query_one("#bar-element").border_title == "C"
            await pilot.press("enter")                 # Enter on the table takes the highlighted row
            await pilot.pause(0.1)
            assert ctl.status().active_index == 2
            await pilot.press("n")
            await pilot.pause(0.1)
            assert ctl.status().active_index == 3
            await pilot.press("space")
            await pilot.pause(0.1)
            assert ctl.status().transport.value == "Paused"
            assert str(app.query_one("#pl-play", Button).label) == "▶ Resume"
            await pilot.press("s")
            await pilot.pause(0.1)
            assert ctl.status().transport.value == "Stopped"
    run(scenario())


def test_mode_transition_end_and_enable_controls():
    async def scenario():
        app, ctl, backend, tick = make_app()
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause()
            app.query_one("#pl-mode", Select).value = "PlayAll"
            app.query_one("#pl-transition", Select).value = "Fade"
            await pilot.pause(0.1)
            assert ctl.mode.value == "PlayAll" and ctl.transition.value == "Fade"
            await pilot.press("m")
            await pilot.pause(0.1)
            assert ctl.clips[0].element_mode.value == "Timed"
            assert str(app.query_one("#el-end", Button).label) == "End: Timed"
            await pilot.press("o")
            await pilot.pause(0.1)
            assert ctl.clips[0].disabled and ctl.status().selected_index == 1
            table = app.query_one("#clips", DataTable)
            assert "off" in str(table.get_row_at(0)[2])
    run(scenario())


def test_edit_dialog_saves_cue_points_and_add_inserts_after_selection():
    async def scenario():
        app, ctl, backend, tick = make_app()
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause()
            await pilot.press("e")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, EditScreen)
            screen.query_one("#field-cue_in").value = "1000"
            screen.query_one("#field-cue_out").value = "4000"
            await pilot.click("#save")
            await pilot.pause(0.1)
            assert (ctl.clips[0].play_from_ms, ctl.clips[0].play_to_ms) == (1000, 4000)
            await pilot.press("a")
            await pilot.pause()
            app.screen.query_one("#field-url").value = "/media/new.mp4"
            await pilot.click("#save")
            await pilot.pause(0.1)
            assert ctl.clips[1].url == "/media/new.mp4"
            await pilot.press("e")
            await pilot.pause()
            app.screen.query_one("#field-url").value = ""
            await pilot.click("#save")
            await pilot.pause()
            assert isinstance(app.screen, EditScreen)   # rejected, dialog stays open
            await pilot.click("#cancel")
    run(scenario())


def test_scheduled_advance_shows_countdown_and_switches_rows():
    async def scenario():
        app, ctl, backend, tick = make_app()
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause()
            ctl.play(); ctl.poll(0)
            tick.now = 9_000
            await pilot.pause(0.1)
            assert "in  1.0 s" in str(app.query_one("#oa-next", Static).render())
            tick.now = 10_000
            await pilot.pause(0.1)
            assert ctl.status().active_index == 1
            table = app.query_one("#clips", DataTable)
            assert str(table.get_row_at(1)[0]) == "▶"
    run(scenario())


def test_enter_inside_the_edit_dialog_does_not_take_and_dialog_fits_120x32():
    async def scenario():
        app, ctl, backend, tick = make_app()
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause()
            await pilot.press("e")
            await pilot.pause()
            assert isinstance(app.screen, EditScreen)
            save = app.screen.query_one("#save", Button)
            assert save.region.y + save.region.height <= 32
            app.screen.query_one("#field-url").focus()
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert ctl.status().transport.value == "Stopped"   # no take from inside the dialog
            await pilot.click("#cancel")
    run(scenario())


def test_selection_survives_add_and_remove():
    async def scenario():
        app, ctl, backend, tick = make_app()
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause()
            await pilot.press("down", "down")
            await pilot.pause(0.1)
            assert ctl.status().selected_index == 2
            ctl.insert_clip(3, clips("x")[0])          # backend-side change rebuilds the table
            await pilot.pause(0.2)
            assert ctl.status().selected_index == 2
            assert app.query_one("#clips", DataTable).cursor_row == 2
    run(scenario())
