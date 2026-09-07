"""Native seek/pause/speed regressions, shared by software H.264 and NVDEC."""

import random
import socket
import time
from datetime import datetime, timezone

import pytest

from player import exercise_v2
from replay import (
    JanusVideoConfig, PlaybackOperation as Op, PlayerConfig, ReplaySlotConfig,
    build_player_application,
)
from playback_backend import backend_api
from test_transcode_integration import _transcode_test_source


@pytest.fixture(scope="module", params=[24, 25, 30, 60], ids=lambda fps: f"{fps}fps")
def recording(request, tmp_path_factory, playback_backend):
    return _transcode_test_source(
        tmp_path_factory.mktemp(f"playback-{request.param}"),
        request.param * 8, fps=request.param, backend=playback_backend,
    )


@pytest.fixture
def application(recording, request, playback_backend):
    options = getattr(request, "param", {})
    if isinstance(options, str):
        options = {"timestamp_source": options}
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    app = None
    try:
        app = build_player_application(PlayerConfig(
            ReplaySlotConfig(recording, loop=options.get("loop", True)),
            JanusVideoConfig(video_port=receiver.getsockname()[1]),
        ), api=backend_api(playback_backend))
        if "timestamp_source" in options:
            app.avp.executeCommandsFromString(
                f'node.param.set replay_input timestamp_source "{options["timestamp_source"]}"'
            )
        if options.get("buffered_decoder") and playback_backend == "nvidia":
            app.avp.executeCommandsFromString("node.param.set replay_decode options {}")
        app.start()
        app.controller.execute(Op.PAUSE)
        _assert_stable(app.controller)
        yield app
    finally:
        try:
            if app is not None:
                app.stop()
        finally:
            receiver.close()


def _wait(control, predicate, description, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = control.status()
        assert not status.error, status.error
        if predicate(status):
            return status
        time.sleep(.005)
    pytest.fail(f"{description}: {control.status()}")


def _assert_stable(control):
    # Pause seeks to the held frame. Decoder restart after EOF can deliver that
    # same picture later; verify every picture, rather than probe silence.
    time.sleep(3 / control.artifact.fps)
    marker = control.observation_marker()
    before = control.status()
    assert not before.playing
    time.sleep(3 / control.artifact.fps)
    after = control.status()
    assert not after.playing
    assert after.frame_number == before.frame_number
    assert all(frame == before.frame_number for frame in control.observed_frames_since(marker))


def _nearest_frame(artifact, target):
    # The seek contract selects the nearest indexed frame; ties select later.
    return min(range(artifact.frame_count), key=lambda i: (
        abs(artifact.seek_entries[i].timestamp_ms - artifact.start_ms - target), -i,
    ))


def _seek_and_observe(app, target, kind="media"):
    control = app.controller
    artifact = control.artifact
    expected = _nearest_frame(artifact, target)
    marker = control.observation_marker()
    if kind == "media":
        command = f"seek replay_sync now {artifact.start_ms + target}"
    elif kind == "relative":
        command = f"seek replay_sync now {target - control.status().position_ms:+d}"
    elif kind == "utc":
        wallclock = artifact.history.media_to_wallclock_ms(artifact.start_ms + target)
        timestamp = datetime.fromtimestamp(wallclock / 1000, timezone.utc)
        command = f"seek replay_sync now {timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}"
    else:
        raise AssertionError(kind)
    # Send raw AVP commands: rounding in the Python controller cannot mask a bug.
    app.avp.executeCommandsFromString(command)
    status = _wait(control, lambda s: (
        control.observation_marker() > marker and s.frame_number == expected
    ), f"{command}, expected frame {expected}")
    assert status.position_ms == artifact.seek_entries[expected].timestamp_ms - artifact.start_ms
    assert not status.playing
    return status


@pytest.mark.parametrize("kind", ["media", "relative", "utc"])
@pytest.mark.parametrize("speed", [25, 50, 100, 200, -25, -50, -100, -200])
def test_paused_seek_frame_boundaries(application, kind, speed):
    app = application
    control = app.controller
    if speed < 0:
        control.execute(Op.REVERSE)
        control.execute(Op.PAUSE)
    control.execute(Op.SPEED, abs(speed))
    artifact = control.artifact
    lo = artifact.seek_entries[artifact.fps - 1].timestamp_ms - artifact.start_ms
    hi = artifact.seek_entries[artifact.fps].timestamp_ms - artifact.start_ms
    # Includes the old 7-ms discard cutoff and both sides of nearest-frame ties.
    targets = sorted({lo, lo + 1, lo + 6, lo + 7, lo + 8,
                      (lo + hi) // 2 - 1, (lo + hi) // 2,
                      (lo + hi) // 2 + 1, hi - 1, hi})
    for target in targets:
        for origin in (0, 2000):
            _seek_and_observe(app, origin)
            _seek_and_observe(app, target, kind)
    _assert_stable(control)
    before = control.status().frame_number
    control.execute(Op.PLAY)
    _wait(control, lambda s: (
        s.frame_number < before if speed < 0 else s.frame_number > before
    ), "resume after paused seek")
    control.execute(Op.PAUSE)
    _assert_stable(control)


def test_paused_seek_endpoints_repeated_and_seeded_targets(application):
    app = application
    control = app.controller
    artifact = control.artifact
    rng = random.Random(0x35983)
    targets = [0, 1, artifact.duration_ms - 1, artifact.duration_ms,
               983, 983, 984, 983, 0]
    targets += [rng.randrange(artifact.duration_ms + 1) for _ in range(40)]
    for index, target in enumerate(targets):
        _seek_and_observe(app, target, ("media", "relative", "utc")[index % 3])
    # Queue commands without waiting; the last command must determine the hold.
    for target in targets:
        app.avp.executeCommandsFromString(f"seek replay_sync now {target}")
    _seek_and_observe(app, 983)
    _assert_stable(control)


def test_frame_steps_after_unaligned_seek_and_speed_changes(application):
    app = application
    control = app.controller
    for speed in (200, 50, 100, 0, 100):
        control.execute(Op.SPEED, speed)
        for delta in (-30, -5, -1, 0, 1, 5, 30):
            base = _seek_and_observe(app, 3983).frame_number
            marker = control.observation_marker()
            control.execute(Op.SEEK_FRAMES, delta)
            _wait(control, lambda s: (
                control.observation_marker() > marker and s.frame_number == base + delta
            ), f"exact frame step {delta} at speed {speed}")
    _assert_stable(control)


def test_absolute_frame_seeks_clamp_at_recording_end(application):
    app = application
    control = app.controller
    count = control.artifact.frame_count
    for target in (0, count - 1, count, count + 1, 0):
        marker = control.observation_marker()
        app.avp.executeCommandsFromString(f"seek replay_sync frame {target}")
        _wait(control, lambda s: (
            control.observation_marker() > marker and s.frame_number == min(target, count - 1)
        ), f"absolute frame seek {target}")
    _assert_stable(control)


def test_relative_seeks_clamp_at_both_recording_edges(application):
    app = application
    control = app.controller
    for target, sign in ((0, -1), (control.artifact.duration_ms, 1)):
        expected = _nearest_frame(control.artifact, target)
        for operation, amount in ((Op.SEEK_FRAMES, 1), (Op.SEEK_FRAMES, 30),
                                  (Op.SEEK_SECONDS, 1), (Op.SEEK_SECONDS, 30)):
            _seek_and_observe(app, target)
            marker = control.observation_marker()
            control.execute(operation, sign * amount)
            _wait(control, lambda s: (
                control.observation_marker() > marker and s.frame_number == expected
            ), f"{operation} {sign * amount} at recording edge {target}")
    _assert_stable(control)


@pytest.mark.parametrize("application", ["input", "none"], indirect=True)
def test_seek_with_other_packet_timestamp_sources(application):
    for kind in ("media", "relative", "utc"):
        _seek_and_observe(application, 0)
        _seek_and_observe(application, 983, kind)
        _seek_and_observe(application, 2000)
        _seek_and_observe(application, 983, kind)
    _assert_stable(application.controller)


@pytest.mark.parametrize("application", [{"buffered_decoder": True}], indirect=True)
def test_unaligned_seek_with_nvdec_display_buffering(application):
    # Keep the original failing decoder configuration covered as well as the
    # interactive low-delay graph; low_delay must not hide a native seek defect.
    for target in (0, 974, 0, 983, 2000, 983, 0, 984, 3983):
        _seek_and_observe(application, target)
    _assert_stable(application.controller)


def test_playing_seeks_pause_and_speed_transitions(application):
    app = application
    control = app.controller
    for speed in (50, 100, 200, 100, 50, 200):
        control.execute(Op.SPEED, speed)
        control.execute(Op.PAUSE)
        _seek_and_observe(app, 1983)
        control.execute(Op.PLAY)
        _wait(control, lambda s: s.position_ms > 2200, "play advances")
        marker = control.observation_marker()
        control.execute(Op.SEEK_MS, 4983)
        _wait(control, lambda s: (
            control.observation_marker() > marker and 4900 <= s.position_ms <= 5500
        ), "seek while playing")
        control.execute(Op.PAUSE)
        _assert_stable(control)
        _seek_and_observe(app, 983)
    results = exercise_v2(control, timeout=5)
    assert not [result for result in results if result.outcome == "FAIL"], results


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_repeated_active_speed_changes_keep_frame_continuity(application, reverse):
    app = application
    control = app.controller
    _seek_and_observe(app, 6000 if reverse else 1000)
    control.execute(Op.REVERSE if reverse else Op.PLAY)
    for speed in (200, 100, 25, 100, 200, 50, 25, 50, 100, 50, 200, 100):
        control.execute(Op.SPEED, speed)
        marker = control.observation_marker()
        _wait(control, lambda _s: len(control.observed_frames_since(marker)) >= 10,
              f"output at speed {speed}")
        frames = control.observed_frames_since(marker)[:10]
        deltas = [(b - a) * (-1 if reverse else 1) for a, b in zip(frames, frames[1:])]
        assert all(delta >= 0 for delta in deltas), (speed, frames)
        if speed == 100:
            assert all(delta == 1 for delta in deltas), (speed, frames)
        elif speed in (25, 50):
            assert set(deltas) == {0, 1}, (speed, frames)
        else:
            assert sum(deltas) >= len(deltas) * 1.5, (speed, frames)
        assert control.status().playing
    control.execute(Op.PAUSE)
    _assert_stable(control)
    _seek_and_observe(app, 3983)


@pytest.mark.parametrize("application", [{"loop": True}, {"loop": False}],
                         indirect=True, ids=["loop", "no-loop"])
def test_pause_and_seek_after_recording_end(application):
    app = application
    control = app.controller
    artifact = control.artifact
    _seek_and_observe(app, artifact.seek_entries[-5].timestamp_ms - artifact.start_ms)
    marker = control.observation_marker()
    control.execute(Op.PLAY)
    if control.status().loop:
        _wait(control, lambda s: control.observation_marker() > marker and s.position_ms < 1000,
              "recording loops", timeout=6)
    else:
        _wait(control, lambda s: s.frame_number == artifact.frame_count - 1,
              "recording reaches its final frame")
        time.sleep(.2)
    control.execute(Op.PAUSE)
    _assert_stable(control)
    _seek_and_observe(app, 1000)
    _seek_and_observe(app, 983)
    control.execute(Op.PLAY)
    _wait(control, lambda s: s.position_ms > 1100, "resume after EOF seek")


@pytest.mark.parametrize("application", [{"loop": False}], indirect=True)
def test_seek_while_eof_frames_are_draining(application):
    app = application
    control = app.controller
    artifact = control.artifact
    for _ in range(8):
        control.execute(Op.PAUSE)
        _seek_and_observe(app, artifact.seek_entries[-4].timestamp_ms - artifact.start_ms)
        control.execute(Op.PLAY)
        _wait(control, lambda s: s.frame_number >= artifact.frame_count - 3,
              "final frames start draining")
        control.execute(Op.PAUSE)
        _seek_and_observe(app, 0)
        _assert_stable(control)
