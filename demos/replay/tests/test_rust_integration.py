"""End to end on the Rust avplumber: transcode a generated clip, play it, run
the V2 exercise, and see RTP arrive. Needs ``AVPLUMBER_BIN`` and the
``ffmpeg``/``ffprobe`` CLI with libx264; skipped otherwise."""

import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

import transcode
from player import exercise_v2
from replay import (Backend, JanusVideoConfig, PlaybackOperation as Op, PlayerConfig,
                    ReplaySlotConfig, TranscodeConfig, build_player_application,
                    detect_backend, read_seek_table, validate_recording)


pytestmark = pytest.mark.skipif(
    not os.environ.get("AVPLUMBER_BIN") or shutil.which("ffmpeg") is None,
    reason="needs AVPLUMBER_BIN (the Rust executable) and the ffmpeg CLI",
)

FPS = 30


@pytest.fixture(params=[Backend.CPU, Backend.NVIDIA], ids=["cpu", "nvidia"])
def backend(request):
    """Both codec backends, where the machine supports them. The NVIDIA one
    keeps every frame on the GPU."""
    if request.param is Backend.NVIDIA and detect_backend() is not Backend.NVIDIA:
        pytest.skip("no working NVIDIA driver with h264_cuvid and h264_nvenc")
    return request.param
# Long enough for a five-second nudge from the middle; the thirty-second
# nudges stay SKIPped, as on any short recording.
SECONDS = 14


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    directory = tmp_path_factory.mktemp("replay-rust")
    source = directory / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc2=size=160x120:rate={FPS}:duration={SECONDS}",
         "-c:v", "libx264", "-preset", "ultrafast", "-g", "15", "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )
    output = directory / "replay.ts"
    origin = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)
    transcode.run(TranscodeConfig(source, output, FPS, origin))
    return output


def test_transcode_produces_a_valid_indexed_recording(recording):
    artifact = validate_recording(recording)
    assert artifact.fps == FPS
    assert artifact.frame_count == FPS * SECONDS
    assert artifact.wallclock_start_ms == 1_786_363_200_000
    assert Path(f"{recording}+txt").read_text().count("\n") == artifact.frame_count


def test_player_passes_the_v2_exercise_and_sends_rtp(recording, tmp_path):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    port = receiver.getsockname()[1]
    config = PlayerConfig(
        ReplaySlotConfig(recording, loop=True, control_timeout=5.0),
        JanusVideoConfig(video_port=port, rtcp_port=0),
    )
    application = build_player_application(config, avplumber_log=tmp_path / "avplumber.log")
    application.start()
    try:
        results = exercise_v2(application.controller, config.slot.control_timeout)
        failures = [f"{r.name}: {r.detail}" for r in results if r.outcome == "FAIL"]
        assert not failures, "\n".join(failures)
        assert {r.name for r in results if r.outcome == "PASS"} >= {
            "source ready", "play advances", "pause stable", "absolute seek",
            "nudge +1f", "nudge -30f", "nudge +5s", "speed 50/100/200%",
            "reverse play", "scrub forward", "scrub reverse", "scrub restore",
            "tail -3s", "UTC seek", "rapid paused seeks",
        }
        application.controller.execute(Op.PLAY)
        packet, _ = receiver.recvfrom(2048)
        assert packet[0] >> 6 == 2, "RTP version 2"
        assert packet[1] & 0x7F == config.janus.payload_type
        assert int.from_bytes(packet[8:12], "big") == config.janus.ssrc
    finally:
        application.stop()
        receiver.close()


# ------------------------------------------------------------------ oracle

def _assert_stable(control):
    """The paused-picture oracle of the old native suite: after a pause the
    frame number holds, and every frame observed in the meantime is that one
    (a decoder restart could otherwise deliver the same picture later)."""
    time.sleep(3 / control.artifact.fps)
    marker = control.observation_marker()
    before = control.status()
    assert not before.playing
    time.sleep(3 / control.artifact.fps)
    after = control.status()
    assert not after.playing
    assert after.frame_number == before.frame_number
    assert all(frame == before.frame_number for frame in control.observed_frames_since(marker))


def _wait(control, predicate, description, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = control.status()
        assert not status.error, status.error
        if predicate(status):
            return status
        time.sleep(0.005)
    pytest.fail(f"{description}: {control.status()}")


# ------------------------------------------------- transcode, every packet

def _source_clip(directory, frame_count, fps, source_gop):
    """A testsrc2 clip, all-intra or with B-frames, `frame_count` frames long."""
    source = directory / f"source-{frame_count}-{source_gop}.mp4"
    codec_options = (["-g", "1", "-bf", "0"] if source_gop == "intra" else
                     ["-g", "60", "-bf", "3", "-x264-params", "b-adapt=0"])
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc2=size=160x120:rate={fps}", "-frames:v", str(frame_count),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", *codec_options, str(source)],
        check=True,
    )
    return source


def _probe(path, entries):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", entries, "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs the ffprobe CLI")
@pytest.mark.parametrize("frame_count", [30, 60, 73])
@pytest.mark.parametrize("source_gop", ["intra", "interframe"])
def test_finite_transcode_publishes_every_all_intra_packet(tmp_path, frame_count, source_gop):
    """Old `test_transcode_integration.py`: whatever the source's GOP, the
    recording has one keyframe packet and one seek entry per source frame,
    the odd count included, i.e. the graph drains its tail."""
    source = _source_clip(tmp_path, frame_count, FPS, source_gop)
    frame_types = {frame["pict_type"] for frame in _probe(source, "frame=pict_type")["frames"]}
    if source_gop == "intra":
        assert frame_types == {"I"}
    else:
        assert "B" in frame_types
    output = tmp_path / f"replay-{frame_count}-{source_gop}.ts"
    transcode.run(TranscodeConfig(source, output, FPS, datetime(2026, 8, 10, 12, tzinfo=timezone.utc)))

    packets = _probe(output, "packet=flags")["packets"]
    seek_entries = read_seek_table(Path(f"{output}+seek"))
    assert len(packets) == frame_count
    assert len(seek_entries) == len(packets)
    assert all("K" in packet["flags"] for packet in packets)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs the ffprobe CLI")
def test_transcode_on_each_backend_produces_the_same_kind_of_recording(tmp_path, backend):
    """The transcode leg on both backends. On NVIDIA the frames go straight
    from NVDEC to NVENC; either way the recording is all-intra with one seek
    entry per frame, which is what playback depends on."""
    source = _source_clip(tmp_path, 45, FPS, "interframe")
    output = tmp_path / f"replay-{backend.value}.ts"
    transcode.run(TranscodeConfig(source, output, FPS,
                                  datetime(2026, 8, 10, 12, tzinfo=timezone.utc),
                                  backend=backend))

    packets = _probe(output, "packet=flags")["packets"]
    assert len(packets) == 45
    assert all("K" in packet["flags"] for packet in packets)
    assert len(read_seek_table(Path(f"{output}+seek"))) == 45
    stream = _probe(output, "stream=codec_name,width,height")["streams"][0]
    assert stream["codec_name"] == "h264"
    assert (stream["width"], stream["height"]) == (160, 120)


# ------------------------------------------------------- RTP across seeks

def _fresh_packets(receiver, seconds):
    """Drains the socket for `seconds` and returns the RTP packets that arrived."""
    packets = []
    deadline = time.monotonic() + seconds
    receiver.settimeout(0.2)
    while time.monotonic() < deadline:
        try:
            packet, _ = receiver.recvfrom(65535)
        except socket.timeout:
            continue
        packets.append(packet)
    return packets


def _player(recording, receiver, tmp_path, *, loop=True, payload_type=96, ssrc=0x41565001,
            backend=Backend.CPU):
    config = PlayerConfig(
        ReplaySlotConfig(recording, loop=loop, control_timeout=5.0),
        JanusVideoConfig(video_port=receiver.getsockname()[1], rtcp_port=0,
                         payload_type=payload_type, ssrc=ssrc),
        backend,
    )
    return build_player_application(config, avplumber_log=tmp_path / "avplumber.log")


@pytest.fixture
def receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    try:
        yield sock
    finally:
        sock.close()


def test_rtp_keeps_flowing_across_seeks_reverse_and_scrubbing(recording, receiver, tmp_path,
                                                              backend):
    """The browser's view of the player: encoded video must keep arriving
    after every kind of discontinuity, with the configured payload type and
    SSRC. Regression: the encoder used to die at the first seek (the
    picture froze on REVERSE), while playback status carried on."""
    application = _player(recording, receiver, tmp_path, payload_type=97, ssrc=0x12345678,
                          backend=backend)
    application.start()
    control = application.controller
    try:
        control.execute(Op.PLAY)
        assert len(_fresh_packets(receiver, 1.0)) >= 10, "RTP while playing"
        # (step, what it does, the direction playback must be going afterwards)
        steps = [
            ("seek while playing", lambda: control.execute(Op.SEEK_MS, 9000), "forward"),
            ("reverse", lambda: control.execute(Op.REVERSE), "reverse"),
            ("2x reverse", lambda: control.execute(Op.SPEED, 200), "reverse"),
            ("1x reverse", lambda: control.execute(Op.SPEED, 100), "reverse"),
            ("pause and toggle", lambda: (control.execute(Op.PAUSE), control.execute(Op.TOGGLE)), "reverse"),
            ("play turns forward", lambda: control.execute(Op.PLAY), "forward"),
            ("scrub back", lambda: control.execute(Op.SCRUB, -200), "reverse"),
            ("scrub released restores forward", lambda: control.execute(Op.SCRUB, 0), "forward"),
            ("frame nudges", lambda: [control.execute(Op.SEEK_FRAMES, d) for d in (-30, 5, 1, -1)], "forward"),
            ("reverse again", lambda: control.execute(Op.REVERSE), "reverse"),
        ]
        for name, act, expected in steps:
            act()
            packets = _fresh_packets(receiver, 1.0)
            assert len(packets) >= 10, f"RTP stopped after {name}: {len(packets)} packets in a second"
            assert all(p[1] & 0x7F == 97 for p in packets), name
            assert all(int.from_bytes(p[8:12], "big") == 0x12345678 for p in packets), name
            _wait(control, lambda s: s.direction == expected and s.playing, f"{name}: {expected} play")
        # Paused: one frame's packets, then silence; a paused seek sends that frame.
        control.execute(Op.PAUSE)
        _fresh_packets(receiver, 0.5)
        assert len(_fresh_packets(receiver, 0.5)) == 0, "no RTP while paused"
        control.execute(Op.SEEK_MS, 2000)
        assert 1 <= len(_fresh_packets(receiver, 0.5)) <= 20, "a paused seek sends one frame"
        _assert_stable(control)
        assert not control.status().error
    finally:
        application.stop()


# -------------------------------------------------------- prompt shutdown

@pytest.mark.parametrize("state", ["paused", "playing", "at_end", "mid_seeks", "reverse"])
def test_player_stops_promptly_in_every_state(recording, receiver, tmp_path, state):
    """Old `test_demux_shutdown_integration.py` bounded native shutdown with
    subprocess deadlines; here the whole application, Rust process included,
    must stop well within the control timeout whatever it was doing."""
    application = _player(recording, receiver, tmp_path, loop=state != "at_end")
    application.start()
    control = application.controller
    if state == "playing":
        control.execute(Op.PLAY)
        _wait(control, lambda s: s.position_ms > 200, "playback advances")
    elif state == "at_end":
        control.execute(Op.PAUSE)
        control.execute(Op.TAIL)
        control.execute(Op.PLAY)
        _wait(control, lambda s: s.at_end, "the recording ends", timeout=8.0)
    elif state == "mid_seeks":
        control.execute(Op.PAUSE)
        for _ in range(10):
            control.execute(Op.SEEK_MS, 1000)
            control.execute(Op.SEEK_MS, 9000)
    elif state == "reverse":
        control.execute(Op.REVERSE)
        _wait(control, lambda s: s.direction == "reverse" and s.playing, "reverse play")
    else:
        control.execute(Op.PAUSE)
        _assert_stable(control)
    started = time.monotonic()
    application.stop()
    took = time.monotonic() - started
    assert took < 3.0, f"stopping while {state} took {took:.1f} s"
    assert application.process._process is None
