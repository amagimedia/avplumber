import asyncio
import json

import pytest

pytest.importorskip("textual")

from tui import MixerTui, SceneButton


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
        raise AssertionError(command)


def test_direct_mode_scene_tile_cuts_to_program():
    async def exercise():
        app = MixerTui(
            "127.0.0.1",
            7777,
            "mixer",
            fade_duration=0.5,
            wipe_style="wipe_left",
        )
        connection = FakeConnection()
        app.connection = connection

        async with app.run_test(size=(160, 45)) as pilot:
            await pilot.pause()
            assert len(app.query(SceneButton)) == 3
            controls = app.query(
                "Button, #transition_status, #fade_duration, #wipe_style"
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
