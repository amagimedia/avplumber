import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("textual")

from player import ReplayTui
from replay import (
    JanusVideoConfig,
    PlaybackController,
    PlayerConfig,
    ReplayArtifact,
    ReplaySlotConfig,
    SeekTableEntry,
    TimestampHistory,
    TimestampHistoryEntry,
)


def make_application(tmp_path):
    recording = tmp_path / "clip.ts"
    artifact = ReplayArtifact(
        recording,
        tuple(SeekTableEntry(1_260 + index * 40, index * 188) for index in range(101)),
        TimestampHistory((TimestampHistoryEntry(0, 0, -1_786_363_198_740, 0),)),
        25,
    )
    commands = []
    controller = PlaybackController(commands.append, artifact)
    controller.observe(frame_number=50, media_timestamp_ms=3_260)
    config = PlayerConfig(ReplaySlotConfig(recording), JanusVideoConfig())
    return SimpleNamespace(
        artifact=artifact,
        controller=controller,
        config=config,
    ), commands


def test_tui_exposes_every_frame_and_second_nudge(tmp_path):
    async def exercise():
        application, commands = make_application(tmp_path)
        app = ReplayTui(application)
        async with app.run_test(size=(180, 42)) as pilot:
            await pilot.pause()
            for unit, operation in (("frame", "frame"), ("second", "now")):
                for value in (-30, -5, -1, 1, 5, 30):
                    app.query_one(f"#{unit}_{value}").press()
            await pilot.pause()
            assert sum(command.startswith("seek replay_sync") for command in commands) == 12
            assert "LOOP=ON" in str(app.query_one("#state").render())

    asyncio.run(exercise())


def test_tui_dispatches_transport_speed_and_absolute_seeks(tmp_path):
    async def exercise():
        application, commands = make_application(tmp_path)
        app = ReplayTui(application)
        async with app.run_test(size=(180, 42)) as pilot:
            await pilot.pause()
            for button_id in (
                "pause", "play", "toggle", "play", "reverse",
                "scrub_-200", "scrub_0", "speed_25", "tail",
            ):
                app.query_one(f"#{button_id}").press()
                await pilot.pause()

            app.query_one("#media_time").value = "00:02.000"
            app.query_one("#seek_media").press()
            app.query_one("#utc_time").value = "2026-08-10T12:00:02Z"
            app.query_one("#seek_utc").press()
            await pilot.pause()

            assert "pause replay_pause now" in commands
            assert "resume replay_pause" in commands
            assert "speed.set replay_speed -1" in commands
            assert "speed.set replay_speed -2" in commands
            assert "speed.set replay_speed -0.25" in commands
            assert "seek replay_sync now 2260" in commands
            assert "seek replay_sync now 3260" in commands
            assert "seek replay_sync now 2026-08-10T12:00:02.000" in commands
            assert app.query_one("#exercise") is not None

    asyncio.run(exercise())
