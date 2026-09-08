from datetime import timezone

import pytest

from types import SimpleNamespace

import player
from player import exercise_v2, parse_args, parse_media_time, parse_utc, stop_bounded
from replay import PlaybackOperation as Op, SeekTableEntry, TimestampHistory, TimestampHistoryEntry


@pytest.mark.parametrize(
    "value, expected",
    [("1250", 1250), ("01:02.500", 62_500), ("01:01:02", 3_662_000)],
)
def test_media_time_parser(value, expected):
    assert parse_media_time(value) == expected


def test_utc_parser_requires_timezone_and_normalizes_offset():
    assert parse_utc("2026-08-10T14:00:00+02:00").hour == 12
    assert parse_utc("2026-08-10T14:00:00+02:00").tzinfo is timezone.utc
    with pytest.raises(ValueError, match="explicit offset"):
        parse_utc("2026-08-10T12:00:00")


def test_player_cli_is_one_recording_one_janus_output(tmp_path):
    config, no_tui, exercise, backend = parse_args([
        "--recording", str(tmp_path / "clip.ts"),
        "--janus-host", "127.0.0.1",
        "--janus-video-port", "6000",
        "--janus-video-pt", "97",
        "--janus-video-ssrc", "0x1234",
        "--no-loop", "--no-tui", "--exercise-v2",
    ])
    assert config.slot.recording == (tmp_path / "clip.ts").resolve()
    assert config.slot.loop is False
    assert config.janus.video_port == 6000
    assert config.janus.payload_type == 97
    assert config.janus.ssrc == 0x1234
    assert no_tui is True
    assert exercise is True
    assert backend.binary is None and backend.connect is None


def test_player_cli_names_the_backend(tmp_path):
    args = ["--recording", str(tmp_path / "clip.ts")]
    assert parse_args(args + ["--avplumber", "/opt/avp"])[3].binary == "/opt/avp"
    connected = parse_args(args + ["--connect", "10.0.0.2:20300", "--avplumber-log", "avp.log"])[3]
    assert connected.connect == ("10.0.0.2", 20300)
    assert str(connected.log) == "avp.log"
    assert parse_args(args + ["--connect", ":20300"])[3].connect == ("127.0.0.1", 20300)


def test_shutdown_is_bounded():
    class SlowApplication:
        def stop(self):
            import time
            time.sleep(0.2)

    assert stop_bounded(SlowApplication(), 0.01) is False


def test_shutdown_exception_is_not_hidden():
    class BrokenApplication:
        def stop(self):
            raise RuntimeError("probe stop failed")

    with pytest.raises(RuntimeError, match="probe stop failed"):
        stop_bounded(BrokenApplication(), 1)


@pytest.mark.parametrize("nudge_fault", [None, "stale", "off_by_one"])
def test_headless_exercise_covers_v2_controls_from_observed_positions(monkeypatch, nudge_fault):
    class SimController:
        def __init__(self):
            self.artifact = SimpleNamespace(
                fps=25,
                frame_count=2_001,
                duration_ms=80_000,
                start_ms=1_260,
                seek_entries=tuple(
                    SeekTableEntry(1_260 + frame * 40, frame * 188)
                    for frame in range(2_001)
                ),
                history=TimestampHistory((TimestampHistoryEntry(0, 0, -1_786_363_198_740, 0),)),
            )
            self.state = SimpleNamespace(
                ready=True, position_ms=40_000, frame_number=1_000,
                playing=True, direction="forward", speed_percent=100,
                scrubbing_percent=0,
            )
            self.restore = None
            self.serial = 0
            self.markers = {0: self.state.frame_number}

        def status(self):
            return SimpleNamespace(**vars(self.state))

        def observation_marker(self):
            return self.serial

        def observed_frames_since(self, marker):
            return tuple(range(self.markers[marker] + 1, self.state.frame_number + 1))

        def position(self, milliseconds):
            self.state.position_ms = min(max(round(milliseconds), 0), self.artifact.duration_ms)
            self.state.frame_number = round(self.state.position_ms / 40)
            self.serial += 1
            self.markers[self.serial] = self.state.frame_number

        def execute(self, operation, value=None):
            if operation is Op.PAUSE:
                self.state.playing = False
            elif operation is Op.PLAY:
                self.state.playing = self.state.speed_percent > 0
            elif operation is Op.SEEK_MS:
                self.position(value)
            elif operation is Op.SEEK_FRAMES:
                if value == -1 and nudge_fault == "stale":
                    return self.state
                if value == -1 and nudge_fault == "off_by_one":
                    value -= 1
                self.position(self.state.position_ms + value * 40)
            elif operation is Op.SEEK_SECONDS:
                self.position(self.state.position_ms + value * 1000)
            elif operation is Op.SPEED:
                self.state.speed_percent = value
                if value == 0:
                    self.state.playing = False
            elif operation is Op.REVERSE:
                self.state.direction = "reverse"
                self.state.playing = True
            elif operation is Op.SCRUB:
                if value:
                    self.restore = self.restore or (self.state.playing, self.state.direction)
                    self.state.scrubbing_percent = value
                    self.state.direction = "reverse" if value < 0 else "forward"
                    self.state.playing = True
                else:
                    self.state.scrubbing_percent = 0
                    self.state.playing, self.state.direction = self.restore
                    self.restore = None
            elif operation is Op.TAIL:
                self.position(self.artifact.duration_ms - 3_000)
            elif operation is Op.SEEK_UTC:
                utc_ms = round(value.timestamp() * 1000)
                media_ms = self.artifact.history.wallclock_to_media_ms(utc_ms)
                self.position(media_ms - self.artifact.start_ms)
            return self.state

    controller = SimController()
    clock = SimpleNamespace(value=10.0)

    def sleep(seconds):
        if controller.state.playing:
            speed = controller.state.scrubbing_percent or controller.state.speed_percent
            direction = -1 if controller.state.direction == "reverse" else 1
            controller.position(controller.state.position_ms + seconds * 1000 * abs(speed) / 100 * direction)
        clock.value += seconds

    monkeypatch.setattr(player.time, "sleep", sleep)
    monkeypatch.setattr(player.time, "monotonic", lambda: clock.value)

    results = exercise_v2(controller, 1.0)
    failures = [result.name for result in results if result.outcome == "FAIL"]
    assert failures == (["nudge -1f"] if nudge_fault else [])
    assert {result.name for result in results} >= {
        "play advances", "pause stable", "absolute seek", "nudge -30f",
        "nudge +30s", "speed 50/100/200%",
        "speed 200->100% frame continuity", "reverse play",
        "scrub forward", "scrub reverse", "scrub restore", "tail -3s",
        "UTC seek", "rapid paused seeks",
    }
