from datetime import datetime, timezone
import pytest

from replay import PlaybackController, PlaybackOperation, ReplayArtifact, SeekTableEntry, TimestampHistory, TimestampHistoryEntry


class Commands:
    def __init__(self):
        self.values = []

    def __call__(self, value):
        self.values.append(value)


class Clock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        return self.value


@pytest.fixture
def controller(tmp_path):
    commands = Commands()
    clock = Clock()
    artifact = ReplayArtifact(
        tmp_path / "clip.ts",
        tuple(SeekTableEntry(1_260 + index * 40, index * 188) for index in range(101)),
        TimestampHistory((TimestampHistoryEntry(0, 0, -1_786_314_598_740, 0),)),
        25,
    )
    return PlaybackController(commands, artifact, clock=clock), commands, clock


def test_pause_play_toggle_and_zero_speed_are_truthful(controller):
    control, commands, _ = controller

    assert control.execute(PlaybackOperation.PAUSE).playing is False
    assert commands.values == ["pause replay now"]
    assert control.execute(PlaybackOperation.PLAY).playing is True
    assert commands.values[-1] == "resume replay"
    assert control.execute(PlaybackOperation.TOGGLE).playing is False

    status = control.execute(PlaybackOperation.SPEED, 0)
    assert status.playing is False
    assert status.speed_percent == 0
    assert control.execute(PlaybackOperation.PLAY).playing is False
    assert "speed is 0%" in control.status().message


@pytest.mark.parametrize(
    "operation,value,command",
    [
        (PlaybackOperation.SEEK_MS, 2_500, "seek replay now 3760"),
        (PlaybackOperation.SEEK_FRAMES, -30, "seek replay frame -30"),
        (PlaybackOperation.SEEK_FRAMES, 5, "seek replay frame +5"),
        (PlaybackOperation.SEEK_SECONDS, -30, "seek replay now -30000"),
        (PlaybackOperation.SEEK_SECONDS, 1, "seek replay now +1000"),
    ],
)
def test_seek_operations_translate_to_complete_commands(controller, operation, value, command):
    control, commands, _ = controller
    control.execute(operation, value)
    assert commands.values[-1] == command


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1", None])
def test_invalid_numeric_values_issue_no_command(controller, value):
    control, commands, _ = controller
    with pytest.raises((TypeError, ValueError)):
        control.execute(PlaybackOperation.SEEK_SECONDS, value)
    assert commands.values == []


def test_speed_and_reverse_retain_direction_and_play_turns_forward(controller):
    control, commands, _ = controller

    status = control.execute(PlaybackOperation.REVERSE)
    assert status.direction == "reverse"
    assert status.playing is True
    assert commands.values[-1] == "speed.set replay -1"

    status = control.execute(PlaybackOperation.SPEED, 50)
    assert status.speed_percent == 50
    assert status.direction == "reverse"
    assert commands.values[-1] == "speed.set replay -0.5"

    # The toggle resumes in the current direction.
    control.execute(PlaybackOperation.PAUSE)
    status = control.execute(PlaybackOperation.TOGGLE)
    assert status.direction == "reverse" and status.playing
    assert commands.values[-1] == "resume replay"

    # PLAY is the forward direction: it turns playback around, in one command.
    before = len(commands.values)
    status = control.execute(PlaybackOperation.PLAY)
    assert status.direction == "forward" and status.playing
    assert commands.values[before:] == ["speed.set replay 0.5"]

    # And REVERSE turns it back, at the configured speed.
    status = control.execute(PlaybackOperation.REVERSE)
    assert status.direction == "reverse"
    assert commands.values[-1] == "speed.set replay -0.5"

    # PLAY while paused and reversing: resumes *and* turns forward.
    control.execute(PlaybackOperation.PAUSE)
    before = len(commands.values)
    status = control.execute(PlaybackOperation.PLAY)
    assert status.direction == "forward" and status.playing
    assert commands.values[before:] == ["resume replay", "speed.set replay 0.5"]

    # PLAY when already playing forward sends nothing.
    before = len(commands.values)
    control.execute(PlaybackOperation.PLAY)
    assert commands.values[before:] == []


@pytest.mark.parametrize("initial,target", [(200, 100), (200, 50), (50, 100), (100, 50)])
def test_active_speed_change_is_one_command(controller, initial, target):
    """No transition gate: a rate change on the Rust core touches nothing in
    flight, so nothing has to be drained first."""
    control, commands, _ = controller
    control.observe(frame_number=50, media_timestamp_ms=3_260)
    control.execute(PlaybackOperation.SPEED, initial)
    commands.values.clear()

    status = control.execute(PlaybackOperation.SPEED, target)

    assert commands.values == [f"speed.set replay {target / 100:g}"]
    assert status.playing is True
    assert status.speed_percent == target
    assert status.frame_number == 50


def test_failed_command_is_reported_and_raised(controller):
    original, _, _ = controller
    commands = []

    def command(value):
        commands.append(value)
        if value == "speed.set replay 1":
            raise RuntimeError("speed command failed")

    control = PlaybackController(command, original.artifact)
    control.execute(PlaybackOperation.SPEED, 200)
    with pytest.raises(RuntimeError, match="speed command failed"):
        control.execute(PlaybackOperation.SPEED, 100)
    assert control.status().speed_percent == 200
    assert "speed command failed" in control.status().error
    assert control.status().last_command == "speed.set replay 1"


@pytest.mark.parametrize("value", [-1, 401, float("-inf")])
def test_speed_range_is_enforced(controller, value):
    control, commands, _ = controller
    with pytest.raises(ValueError):
        control.execute(PlaybackOperation.SPEED, value)
    assert commands.values == []


def test_scrub_dead_zone_throttle_and_restore_paused_reverse(controller):
    control, commands, clock = controller
    control.execute(PlaybackOperation.REVERSE)
    control.execute(PlaybackOperation.PAUSE)
    commands.values.clear()

    assert control.execute(PlaybackOperation.SCRUB, 20).scrubbing_percent == 20
    assert commands.values == []

    control.execute(PlaybackOperation.SCRUB, 200)
    assert commands.values == [
        "speed.set replay 2",
        "resume replay",
    ]
    clock.value += 0.05
    control.execute(PlaybackOperation.SCRUB, -300)
    assert len(commands.values) == 2
    clock.value += 0.06
    control.execute(PlaybackOperation.SCRUB, -300)
    assert commands.values[-1] == "speed.set replay -3"

    status = control.execute(PlaybackOperation.SCRUB, 0)
    assert commands.values[-2:] == [
        "speed.set replay -1",
        "pause replay now",
    ]
    assert status.playing is False
    assert status.direction == "reverse"
    assert status.speed_percent == 100
    assert status.scrubbing_percent == 0

    commands.values.clear()
    control.execute(PlaybackOperation.SCRUB, -200)
    assert commands.values == [
        "speed.set replay -2",
        "resume replay",
    ]


def test_tail_and_utc_seek(controller):
    control, commands, _ = controller

    control.execute(PlaybackOperation.TAIL)
    assert commands.values[-1] == "seek replay now 2260"

    target = datetime(2026, 8, 10, 12, 0, 2, tzinfo=timezone.utc)
    control.execute(PlaybackOperation.SEEK_UTC, target)
    assert commands.values[-1] == "seek replay now 2026-08-10T12:00:02.000"


def test_observation_reports_zero_based_timeline_and_wallclock(controller):
    control, _, _ = controller
    status = control.observe(frame_number=25, media_timestamp_ms=2_260)

    assert status.ready is True
    assert status.frame_number == 25
    assert status.position_ms == 1_000
    assert status.wallclock_ms == 1_786_314_601_000
    assert status.duration_ms == 4_000


def test_status_documents_are_folded_in_by_serial(controller):
    control, _, _ = controller
    first = control.observe_status({"serial": 3, "frame": 25, "media_ms": 2_260, "at_end": False})
    assert first.ready is True and first.frame_number == 25

    # A repeated read of the same serial is not a new frame.
    assert control.observe_status({"serial": 3, "frame": 25, "media_ms": 2_260}) == first
    assert control.observation_marker() == 1
    # Garbage keeps the last observation.
    assert control.observe_status({"serial": "bad"}) == first
    assert control.observe_status({"serial": 4, "frame": None, "media_ms": None}) == first

    later = control.observe_status({"serial": 5, "frame": 26, "media_ms": 2_300, "at_end": True})
    assert later.frame_number == 26
    assert later.at_end is True
    assert control.observed_frames_since(1) == (26,)


def test_observation_marker_returns_every_later_frame(controller):
    control, _, _ = controller
    control.observe(frame_number=25, media_timestamp_ms=2_260)
    marker = control.observation_marker()

    control.observe(frame_number=26, media_timestamp_ms=2_300)
    control.observe(frame_number=27, media_timestamp_ms=2_340)

    assert control.observed_frames_since(marker) == (26, 27)
