import asyncio
import json

import pytest

pytest.importorskip("textual")

from tui import MixerTui, SceneButton
from textual.widgets import Input, Select


class FakeConnection:
    host = "127.0.0.1"
    port = 7777
    connected = True

    def __init__(self):
        self.program = "fullscreen_0"
        self.preview = "grid_2_page_0"
        self.commands = []

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def command(self, command):
        self.commands.append(command)
        if command.startswith("mixer.scenes"):
            return json.dumps(["fullscreen_0", "grid_2_page_0", "grid_4_page_0"])
        if command.startswith("mixer.status"):
            return json.dumps(
                {
                    "pgm_scene": self.program,
                    "pvw_scene": self.preview,
                    "transition": "idle",
                }
            )
        if command.startswith("mixer.cut"):
            self.program = json.loads(command.partition(" ")[2])["scene"]
            return None
        if command.startswith("mixer.preview"):
            self.preview = json.loads(command.partition(" ")[2])["scene"]
            return None
        if command.startswith("mixer.settings"):
            return "{}"
        raise AssertionError(command)


def test_direct_mode_scene_tile_cuts_to_program():
    async def exercise():
        app = MixerTui(
            "127.0.0.1",
            7777,
            "mixer",
            fade_duration=0.5,
            wipe_file="",
            direct=False,
            transition="cut",
        )
        connection = FakeConnection()
        app.connection = connection

        async with app.run_test(size=(160, 45)) as pilot:
            await pilot.pause()
            assert len(app.query(SceneButton)) == 3
            controls = app.query(
                "Button, #transition_status, #fade_duration, #wipe_file"
            )
            for control in controls:
                assert control.region.x >= 0
                assert control.region.right <= app.size.width

            await pilot.press("t")
            app.query(SceneButton)[1].on_click()
            await pilot.pause()

            cuts = [
                command
                for command in connection.commands
                if command.startswith("mixer.cut")
            ]
            assert cuts
            assert json.loads(cuts[-1].partition(" ")[2])["scene"] == "grid_2_page_0"
            assert app.pgm_scene == "grid_2_page_0"
            assert str(app.query_one("#direct").label) == "Direct: ON"

    asyncio.run(exercise())


@pytest.mark.parametrize("transition", ["cut", "fade", "wipe"])
@pytest.mark.parametrize("selection", ["tile", "layout", "function_key"])
def test_direct_selection_honors_transition(transition, selection):
    class TransitionConnection(FakeConnection):
        async def command(self, command):
            if command.startswith(("mixer.fade ", "mixer.wipe ")):
                self.commands.append(command)
                self.program = json.loads(command.partition(" ")[2])["scene"]
                return None
            return await super().command(command)

    async def exercise():
        app = MixerTui("127.0.0.1", 7777, "mixer", fade_duration=0.25, direct=False,
                       wipe_file="/media/My wipe.mov")
        connection = TransitionConnection()
        app.connection = connection
        async with app.run_test(size=(160, 45)) as pilot:
            await pilot.pause()
            app.query_one("#direct_transition", Select).value = transition
            await pilot.press("t")
            if selection == "tile":
                app.query(SceneButton)[2].on_click()
            elif selection == "layout":
                await pilot.click("#grid_4")
            else:
                await pilot.press("f3")
            await pilot.pause()
            takes = [command for command in connection.commands
                     if command.startswith(("mixer.cut ", "mixer.fade ", "mixer.wipe "))]
            payload = {"mixer": "mixer", "scene": "grid_4_page_0"}
            if transition == "fade":
                payload["duration_sec"] = 0.25
            elif transition == "wipe":
                payload["wipe_file"] = "/media/My wipe.mov"
            assert len(takes) == 1
            assert takes[0].partition(" ")[0] == f"mixer.{transition}"
            assert json.loads(takes[0].partition(" ")[2]) == payload
            assert app.pgm_scene == "grid_4_page_0"

    asyncio.run(exercise())


@pytest.mark.parametrize("transition", ["cut", "fade", "wipe"])
def test_direct_mode_reuses_current_transition_button(transition):
    class TransitionConnection(FakeConnection):
        async def command(self, command):
            if command.startswith(("mixer.fade ", "mixer.wipe ")):
                self.commands.append(command)
                self.program = json.loads(command.partition(" ")[2])["scene"]
                return None
            return await super().command(command)

    async def exercise():
        app = MixerTui("127.0.0.1", 7777, "mixer", fade_duration=0.25, direct=False,
                       wipe_file="/media/My wipe.mov")
        connection = TransitionConnection()
        app.connection = connection
        async with app.run_test(size=(160, 45)) as pilot:
            await pilot.pause()
            app.query(SceneButton)[1].on_click()
            await pilot.pause()
            await pilot.click(f"#{transition}")
            await pilot.pause()
            assert app.pgm_scene == "grid_2_page_0"
            await pilot.click("#direct")
            await pilot.click("#grid_4")
            await pilot.pause()
            takes = [command for command in connection.commands
                     if command.startswith(("mixer.cut ", "mixer.fade ", "mixer.wipe "))]
            assert [command.partition(" ")[0] for command in takes] == [
                f"mixer.{transition}", f"mixer.{transition}"]
            assert json.loads(takes[-1].partition(" ")[2])["scene"] == "grid_4_page_0"
            assert app.pgm_scene == "grid_4_page_0"

    asyncio.run(exercise())


def test_rapid_tui_cuts_keep_tcp_replies_aligned():
    """A second keypress must survive cancellation of the first TUI worker."""
    async def exercise():
        first_cut = asyncio.Event()
        release_first = asyncio.Event()
        second_cut = asyncio.Event()
        program = "fullscreen_0"

        async def serve(reader, writer):
            nonlocal program
            writer.write(b"100 ready\n")
            await writer.drain()
            try:
                while line := await reader.readline():
                    command = line.decode().strip()
                    content = None
                    if command.startswith("mixer.scenes"):
                        content = ["fullscreen_0", "grid_2_page_0", "grid_4_page_0"]
                    elif command.startswith("mixer.status"):
                        content = {"pgm_scene": program, "pvw_scene": "", "transition": "idle"}
                    elif command.startswith("mixer.cut"):
                        program = json.loads(command.partition(" ")[2])["scene"]
                        if program == "grid_2_page_0":
                            first_cut.set()
                            await release_first.wait()
                        else:
                            second_cut.set()
                    elif command.startswith("mixer.settings"):
                        content = {}
                    else:
                        raise AssertionError(command)
                    writer.write(("200 OK\n" if content is None else
                                  "201 content\n" + json.dumps(content) + "\n\n").encode())
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        app = MixerTui("127.0.0.1", server.sockets[0].getsockname()[1], "mixer",
                       fade_duration=0.5, wipe_file="", direct=False, transition="cut")
        try:
            async with app.run_test(size=(160, 45)) as pilot:
                await pilot.pause()
                await pilot.press("f2")
                await asyncio.wait_for(first_cut.wait(), 1)
                await pilot.press("f3")
                release_first.set()
                await asyncio.wait_for(second_cut.wait(), 1)
                await pilot.pause()
                assert app.connection.connected
                assert program == "grid_4_page_0"
        finally:
            release_first.set()
            await app.connection.disconnect()
            server.close()
            await server.wait_closed()

    asyncio.run(exercise())


def test_media_wipe_uses_edited_backend_path_and_clip_duration():
    """The clip lives on the mixer host; the TUI must not require a local file."""
    class MediaConnection(FakeConnection):
        async def command(self, command):
            if command.startswith("mixer.wipe "):
                self.commands.append(command)
                self.program = json.loads(command.partition(" ")[2])["scene"]
                return None
            return await super().command(command)

    async def exercise():
        app = MixerTui("127.0.0.1", 7777, "mixer",
                       fade_duration=0.5, wipe_file="/media/initial.mov", direct=False)
        connection = MediaConnection()
        app.connection = connection
        async with app.run_test(size=(160, 45)) as pilot:
            await pilot.pause()
            app.query_one("#wipe_file", Input).value = "/media/My transparent wipe.mov"
            app.query(SceneButton)[1].on_click()
            await pilot.pause()
            await pilot.click("#wipe")
            await pilot.pause()
            wipes = [command for command in connection.commands
                     if command.startswith("mixer.wipe ")]
            assert len(wipes) == 1
            assert json.loads(wipes[0].partition(" ")[2]) == {
                "mixer": "mixer", "scene": "grid_2_page_0",
                "wipe_file": "/media/My transparent wipe.mov",
            }
            assert app.pgm_scene == "grid_2_page_0"

    asyncio.run(exercise())


@pytest.mark.parametrize('current', ['cut', 'crossfade', 'wipe'])
@pytest.mark.parametrize('requested', ['cut', 'fade', 'wipe'])
def test_new_direct_selection_reaches_backend_while_busy(current, requested):
    class BusyConnection(FakeConnection):
        async def command(self, command):
            if command.startswith('mixer.status'):
                self.commands.append(command)
                return json.dumps({'pgm_scene': self.program,
                    'pvw_scene': self.preview, 'transition': current})
            if command.startswith(('mixer.cut ', 'mixer.fade ', 'mixer.wipe ')):
                self.commands.append(command)
                return None
            return await super().command(command)
    async def exercise():
        app = MixerTui('127.0.0.1', 7777, 'mixer', fade_duration=.5,
                       wipe_file='/media/wipe.mov', direct=False)
        app.connection = connection = BusyConnection()
        async with app.run_test(size=(160,45)) as pilot:
            await pilot.pause()
            app.query_one('#direct_transition', Select).value = requested
            await pilot.click('#direct')
            await pilot.click('#grid_2')
            await pilot.pause()
            assert any(c.startswith('mixer.' + requested + ' ') for c in connection.commands), connection.commands
    asyncio.run(exercise())

def test_return_to_current_program_reaches_backend_during_pending_cut():
    class PendingConnection(FakeConnection):
        async def command(self, command):
            if command.startswith('mixer.status'):
                self.commands.append(command)
                return json.dumps({'pgm_scene': 'fullscreen_0',
                    'pvw_scene': 'grid_2_page_0', 'transition': 'cut'})
            return await super().command(command)
    async def exercise():
        app = MixerTui('127.0.0.1', 7777, 'mixer', fade_duration=.5, wipe_file='', direct=False,
                       transition='cut')
        app.connection = connection = PendingConnection()
        async with app.run_test(size=(160,45)) as pilot:
            await pilot.pause()
            await pilot.press('f1')
            await pilot.pause()
            assert any(c.startswith('mixer.cut ') for c in connection.commands), connection.commands
    asyncio.run(exercise())


@pytest.mark.asyncio
async def test_direct_is_on_by_default_and_mixer_settings_apply():
    class SettingsConnection(FakeConnection):
        async def command(self, command):
            if command.startswith("mixer.settings"):
                self.commands.append(command)
                return json.dumps({"direct": False, "fade_seconds": 1.5, "transition": "wipe",
                                   "wipe_file": "/media/w.mov"})
            return await super().command(command)

    app = MixerTui("127.0.0.1", 7777, "mixer", fade_duration=0.5, wipe_file="")
    assert app.direct_mode is True and app.default_transition == "cut"
    app.connection = SettingsConnection()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.direct_mode is False
        assert app.query_one("#fade_duration").value == "1.5"
        assert app.query_one("#direct_transition").value == "wipe"
        assert app.query_one("#wipe_file").value == "/media/w.mov"
        assert str(app.query_one("#direct").label) == "Direct: OFF"
