"""Rendered Textual controls: one surface, every action, no terminal logs."""

import asyncio
import os
import sys
import time

import pytest

pytest.importorskip("textual")

from textual.widgets import Button, DataTable, Footer

from helpers import clips
from player import FrameworkStdoutRedirect, PlaylistTui
from playlist import (Clip, ElementMode, InMemoryBackend, PlaylistController,
                      PlaylistMode, TransportState)


def running_controller(auto_ready=True):
    backend = InMemoryBackend(auto_ready=auto_ready)
    ctl = PlaylistController(backend, clips("a", "b", "c", "d", "e"))
    ctl.play()
    if auto_ready:
        ctl.poll(0)
    else:
        request = backend.calls[-1][1]
        ctl.notify_source_ready(ctl.clips[0].item_id, request)
    backend.clear()
    return ctl, backend


def run(coro):
    return asyncio.run(coro)


def test_one_visible_control_surface_has_unique_ids_and_no_footer():
    async def scenario():
        ctl, _ = running_controller()
        app = PlaylistTui(ctl)
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            buttons = list(app.query(Button))
            ids = [button.id for button in buttons]
            assert len(ids) == len(set(ids))
            assert list(app.query(Footer)) == []
            assert set(ids) == {
                "list-play", "list-pause", "list-stop", "list-prev", "list-next",
                "list-mode-PLAY_ALL", "list-mode-PLAY_CURRENT",
                "list-mode-LOOP_ALL", "list-mode-LOOP_CURRENT",
                "item-play", "item-pause", "item-stop", "item-mode", "item-edit",
                "item-enable", "item-add", "item-remove", "item-up", "item-down",
            }
    run(scenario())


def test_every_control_is_visible_in_an_80_by_24_terminal():
    async def scenario():
        ctl, _ = running_controller()
        app = PlaylistTui(ctl)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            for button in app.query(Button):
                assert button.region.x >= 0
                assert button.region.y >= 0
                assert button.region.right <= app.size.width
                assert button.region.bottom <= app.size.height
    run(scenario())


def test_every_edit_control_is_visible_in_an_80_by_24_terminal():
    async def scenario():
        ctl, _ = running_controller()
        app = PlaylistTui(ctl)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#item-edit")
            await pilot.pause()
            for control in app.screen.query("Input, Button"):
                assert control.region.x >= 0
                assert control.region.y >= 0
                assert control.region.right <= app.size.width
                assert control.region.bottom <= app.size.height
    run(scenario())


def test_every_playlist_button_executes_and_modes_are_independent():
    async def scenario():
        ctl, backend = running_controller()
        app = PlaylistTui(ctl)
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.click("#list-pause")
            assert ctl.status().transport is TransportState.PAUSED
            await pilot.click("#list-play")
            assert ctl.status().transport is TransportState.PLAYING
            await pilot.click("#list-next")
            await pilot.pause(0.1)
            assert ctl.status().active_index == 1
            await pilot.click("#list-prev")
            await pilot.pause(0.1)
            assert ctl.status().active_index == 0
            for mode in PlaylistMode:
                await pilot.click(f"#list-mode-{mode.name}")
                assert ctl.status().mode is mode
            await pilot.click("#list-stop")
            assert ctl.status().transport is TransportState.STOPPED
            assert not [call for call in backend.calls if call[0] == "shutdown"]
    run(scenario())


def test_selected_item_transport_mode_enable_and_edit_buttons():
    async def scenario():
        ctl, backend = running_controller()
        app = PlaylistTui(ctl)
        async with app.run_test(size=(180, 50)) as pilot:
            table = app.query_one("#clips", DataTable)
            table.move_cursor(row=2)
            await pilot.pause()
            await pilot.click("#item-play")
            await pilot.pause(0.1)
            assert ctl.status().active_index == 2
            await pilot.click("#item-pause")
            assert ctl.status().transport is TransportState.PAUSED
            await pilot.click("#item-stop")
            assert ctl.status().transport is TransportState.STOPPED
            old_mode = ctl.clips[2].element_mode
            await pilot.click("#item-mode")
            assert ctl.clips[2].element_mode is not old_mode
            await pilot.click("#item-enable")
            assert ctl.clips[2].disabled is True

            # Edit is modal and preserves item identity.
            table.move_cursor(row=1)
            await pilot.click("#item-edit")
            await pilot.pause()
            app.screen.query_one("#edit-name").value = "renamed"
            await pilot.click("#edit-save")
            await pilot.pause()
            assert ctl.clips[1].name == "renamed"
            assert backend.calls
    run(scenario())


def test_add_remove_and_reorder_buttons_target_highlighted_row():
    async def scenario():
        ctl, _ = running_controller()
        app = PlaylistTui(ctl)
        async with app.run_test(size=(180, 50)) as pilot:
            table = app.query_one("#clips", DataTable)
            table.move_cursor(row=1)
            await pilot.pause()
            name = ctl.clips[1].name
            await pilot.click("#item-down")
            assert ctl.clips[2].name == name
            table.move_cursor(row=2)
            await pilot.click("#item-up")
            assert ctl.clips[1].name == name
            await pilot.click("#item-remove")
            assert len(ctl.clips) == 4

            await pilot.click("#item-add")
            await pilot.pause()
            app.screen.query_one("#edit-url").value = "/media/new"
            app.screen.query_one("#edit-name").value = "new"
            await pilot.click("#edit-save")
            await pilot.pause()
            assert len(ctl.clips) == 5
            assert ctl.clips[-1].name == "new"
    run(scenario())


def test_keyboard_bindings_share_action_methods_and_remain_hidden():
    async def scenario():
        ctl, _ = running_controller()
        app = PlaylistTui(ctl)
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.press("space")
            assert ctl.status().transport is TransportState.PAUSED
            await pilot.press("space")
            assert ctl.status().transport is TransportState.PLAYING
            await pilot.press("n")
            await pilot.pause(0.1)
            assert ctl.status().active_index == 1
            await pilot.press("p")
            await pilot.pause(0.1)
            assert ctl.status().active_index == 0
            await pilot.press("4")
            assert ctl.status().mode is PlaylistMode.LOOP_CURRENT
            await pilot.press("m")
            assert ctl.clips[0].element_mode is ElementMode.TIMED
            await pilot.press("s")
            assert ctl.status().transport is TransportState.STOPPED
    run(scenario())


def test_pending_backend_operation_does_not_block_more_ui_actions():
    async def scenario():
        ctl, backend = running_controller(auto_ready=False)
        app = PlaylistTui(ctl)
        async with app.run_test(size=(180, 50)) as pilot:
            table = app.query_one("#clips", DataTable)
            table.move_cursor(row=3)
            await pilot.pause()
            started = time.monotonic()
            app.action_item_play()
            app.action_list_mode("PLAY_CURRENT")
            elapsed = time.monotonic() - started
            assert elapsed < 0.05
            assert ctl.status().pending_index == 3
            assert ctl.status().mode is PlaylistMode.PLAY_CURRENT
            assert backend.calls[-1][0] == "play_item"
    run(scenario())


def test_tui_emits_no_backend_logs_or_commands(capsys):
    async def scenario():
        ctl, backend = running_controller()
        app = PlaylistTui(ctl)
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.click("#list-pause")
            await pilot.click("#list-play")
            await pilot.click("#list-stop")
            state = str(app.query_one("#state").render())
            assert "pause_item" not in state
            assert "stop_item" not in state
            assert "node.object.set" not in state
            assert backend.calls
    run(scenario())
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_live_framework_stdout_is_redirected_to_log(tmp_path, capfd):
    log_file = tmp_path / "playlist.log"
    with FrameworkStdoutRedirect(str(log_file)):
        print("python command reply", flush=True)
        print("textual frame", file=sys.__stderr__, flush=True)
        os.write(1, b"native command reply\n")
        print("python diagnostic", file=sys.stderr, flush=True)
        os.write(2, b"native diagnostic\n")
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == "textual frame\n"
    assert log_file.read_text().splitlines() == [
        "python command reply", "native command reply",
        "python diagnostic", "native diagnostic"]
