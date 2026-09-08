"""Shared recording, graph, and control logic for the replay demos.

The graphs run on the Rust avplumber, reached over its control protocol
(`control_client.py`); nothing here runs inside the media process. Seek, rate
and pause address one playback group, `GROUP`, which is the `sync_group` of the
seekable `input` and of the `realtime` pacing node
(`doc/specs/rust-refactor/rust_refactor_playback.md`).
"""

from __future__ import annotations

import json
import struct
import math
import tempfile
import threading
import time
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, NamedTuple

from control_client import AvplumberProcess, ControlClient, find_avplumber


SEEK_STRUCT = struct.Struct("=qQ")
HISTORY_STRUCT = struct.Struct("=qqqq")
SEEK_TABLE_SUFFIX = "+seek"
TEXT_SEEK_TABLE_SUFFIX = "+txt"
HISTORY_SUFFIX = "+history"
GROUP = "replay"
PLAYER_NODE_GROUP = "player"
TRANSCODE_NODE_GROUP = "transcode"
JANUS_FORCE_KEYFRAME_NODE = "janus_force_keyframe"
JANUS_FORCE_KEYFRAME_COMMAND = (
    f"node.object.set {JANUS_FORCE_KEYFRAME_NODE} trigger true"
)
STATUS_POLL_INTERVAL = 0.01


class SeekTableEntry(NamedTuple):
    timestamp_ms: int
    byte_offset: int


class TimestampHistoryEntry(NamedTuple):
    changed_at: int
    input_offset: int
    wallclock_offset: int
    output_offset: int


class TimestampHistory(tuple[TimestampHistoryEntry, ...]):
    def __new__(cls, entries):
        return super().__new__(cls, entries)

    def _at_media(self, timestamp_ms: int) -> TimestampHistoryEntry:
        index = bisect_right([entry.changed_at for entry in self], timestamp_ms) - 1
        return self[max(index, 0)]

    def media_to_wallclock_ms(self, timestamp_ms: int) -> int:
        return timestamp_ms - self._at_media(timestamp_ms).wallclock_offset

    def wallclock_to_media_ms(self, timestamp_ms: int) -> int:
        candidates = sorted(
            self,
            key=lambda entry: entry.changed_at - entry.wallclock_offset,
        )
        starts = [entry.changed_at - entry.wallclock_offset for entry in candidates]
        entry = candidates[max(bisect_right(starts, timestamp_ms) - 1, 0)]
        return timestamp_ms + entry.wallclock_offset


@dataclass(frozen=True)
class ReplayArtifact:
    path: Path
    seek_entries: tuple[SeekTableEntry, ...]
    history: TimestampHistory
    fps: int

    @property
    def frame_count(self) -> int:
        return len(self.seek_entries)

    @property
    def start_ms(self) -> int:
        return self.seek_entries[0].timestamp_ms

    @property
    def duration_ms(self) -> int:
        return self.seek_entries[-1].timestamp_ms - self.start_ms

    @property
    def wallclock_start_ms(self) -> int:
        return self.history.media_to_wallclock_ms(self.start_ms)


def _read_records(path: Path, record: struct.Struct, label: str) -> bytes:
    data = path.read_bytes()
    if not data:
        raise ValueError(f"{label} is empty: {path}")
    if len(data) % record.size:
        raise ValueError(
            f"{label} size must be a multiple of {record.size} bytes: {path}"
        )
    return data


def read_seek_table(path: str | Path) -> tuple[SeekTableEntry, ...]:
    path = Path(path)
    data = _read_records(path, SEEK_STRUCT, "seek table")
    entries = tuple(SeekTableEntry(*values) for values in SEEK_STRUCT.iter_unpack(data))
    if any(a.timestamp_ms > b.timestamp_ms for a, b in zip(entries, entries[1:])):
        raise ValueError(f"seek table timestamps decrease: {path}")
    if any(a.byte_offset > b.byte_offset for a, b in zip(entries, entries[1:])):
        raise ValueError(f"seek table byte offsets decrease: {path}")
    return entries


def read_timestamp_history(path: str | Path) -> TimestampHistory:
    path = Path(path)
    data = _read_records(path, HISTORY_STRUCT, "timestamp history")
    entries = TimestampHistory(
        TimestampHistoryEntry(*values) for values in HISTORY_STRUCT.iter_unpack(data)
    )
    if entries[0].changed_at != 0:
        raise ValueError(f"timestamp history must map frame zero: {path}")
    if any(a.changed_at > b.changed_at for a, b in zip(entries, entries[1:])):
        raise ValueError(f"timestamp history changed_at values decrease: {path}")
    return entries


def _infer_fps(entries: tuple[SeekTableEntry, ...]) -> int:
    if len(entries) < 2:
        raise ValueError("at least two seek entries are required to infer frame cadence")
    span_ms = entries[-1].timestamp_ms - entries[0].timestamp_ms
    if span_ms <= 0:
        raise ValueError("seek table has no positive frame cadence")
    fps = round((len(entries) - 1) * 1000 / span_ms)
    if not 1 <= fps <= 240:
        raise ValueError(f"inferred frame cadence is outside 1..240 fps: {fps}")
    tolerance_ms = max(2.0, 1000 / fps * 0.08)
    for index, entry in enumerate(entries):
        expected = entries[0].timestamp_ms + index * 1000 / fps
        if abs(entry.timestamp_ms - expected) > tolerance_ms:
            raise ValueError(
                f"inconsistent frame cadence at frame {index}: "
                f"{entry.timestamp_ms} ms, expected {expected:.1f} ms"
            )
    return fps


def validate_recording(path: str | Path) -> ReplayArtifact:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    entries = read_seek_table(Path(f"{path}{SEEK_TABLE_SUFFIX}"))
    history = read_timestamp_history(Path(f"{path}{HISTORY_SUFFIX}"))
    return ReplayArtifact(path, entries, history, _infer_fps(entries))


class PlaybackOperation(str, Enum):
    PLAY = "play"
    PAUSE = "pause"
    TOGGLE = "toggle"
    REVERSE = "reverse"
    SEEK_MS = "seek_ms"
    SEEK_FRAMES = "seek_frames"
    SEEK_SECONDS = "seek_seconds"
    SEEK_UTC = "seek_utc"
    SPEED = "speed"
    SCRUB = "scrub"
    TAIL = "tail"


@dataclass(frozen=True)
class PlaybackStatus:
    playing: bool
    direction: str
    speed_percent: float
    scrubbing_percent: float
    position_ms: int
    wallclock_ms: int | None
    frame_number: int | None
    duration_ms: int
    ready: bool
    loop: bool
    at_end: bool = False
    message: str = ""
    error: str = ""
    last_command: str = ""


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _format_number(value: float | int) -> str:
    return f"{value:g}"


class PlaybackController:
    """Translate SSGW v2 single-source operations into avplumber commands.

    Rate and direction are one signed rate on the playback group; the
    transition gate the C++ core needed for a clean speed change is gone,
    because a rate change on the Rust core touches nothing in flight.
    """

    def __init__(self, command, artifact: ReplayArtifact, *, loop=True,
                 clock=time.monotonic, control_timeout=5.0):
        self._command = command
        self.artifact = artifact
        self._clock = clock
        self._control_timeout = control_timeout
        self._lock = threading.RLock()
        self._observed = threading.Condition(self._lock)
        self._observation_serial = 0
        self._observations = deque(maxlen=256)
        self._last_server_serial = 0
        self._status = PlaybackStatus(
            playing=True,
            direction="forward",
            speed_percent=100,
            scrubbing_percent=0,
            position_ms=0,
            wallclock_ms=artifact.wallclock_start_ms,
            frame_number=None,
            duration_ms=artifact.duration_ms,
            ready=False,
            loop=loop,
        )
        self._scrub_restore: tuple[bool, str] | None = None
        self._last_scrub_command = float("-inf")

    def status(self) -> PlaybackStatus:
        with self._observed:
            return self._status

    def set_error(self, error: str) -> PlaybackStatus:
        with self._lock:
            self._status = replace(self._status, error=error)
            return self._status

    def observe(self, *, frame_number: int, media_timestamp_ms: int) -> PlaybackStatus:
        with self._lock:
            position = min(
                max(media_timestamp_ms - self.artifact.start_ms, 0),
                self.artifact.duration_ms,
            )
            self._status = replace(
                self._status,
                ready=True,
                frame_number=frame_number,
                position_ms=position,
                wallclock_ms=self.artifact.history.media_to_wallclock_ms(media_timestamp_ms),
            )
            self._observation_serial += 1
            self._observations.append((self._observation_serial, frame_number))
            self._observed.notify_all()
            return self._status

    def observation_marker(self) -> int:
        with self._lock:
            return self._observation_serial

    def observed_frames_since(self, marker: int) -> tuple[int, ...]:
        with self._lock:
            return tuple(frame for serial, frame in self._observations if serial > marker)

    def observe_status(self, status: dict) -> PlaybackStatus:
        """Folds one `playback.status` document in: a new `serial` is a newly
        released frame, `frame`/`media_ms` say which."""
        try:
            serial = int(status.get("serial", 0))
            frame = status.get("frame")
            media_ms = status.get("media_ms")
            at_end = bool(status.get("at_end", False))
        except (TypeError, ValueError, AttributeError):
            return self.status()
        with self._lock:
            if at_end != self._status.at_end:
                self._status = replace(self._status, at_end=at_end)
            if serial <= self._last_server_serial or frame is None or media_ms is None:
                return self._status
            self._last_server_serial = serial
            return self.observe(frame_number=int(frame), media_timestamp_ms=int(media_ms))

    def _send(self, command: str) -> None:
        try:
            self._command(command)
        except Exception as exc:
            self._status = replace(self._status, error=str(exc), last_command=command)
            raise
        self._status = replace(self._status, error="", last_command=command)

    def _signed_rate(self, speed_percent: float, direction: str | None = None) -> float:
        direction = direction or self._status.direction
        return speed_percent / 100 * (-1 if direction == "reverse" else 1)

    def _pause(self) -> None:
        if self._status.playing:
            self._send(f"pause {GROUP} now")
        self._status = replace(self._status, playing=False, message="")

    def _play(self) -> None:
        if self._status.speed_percent == 0:
            self._status = replace(
                self._status,
                playing=False,
                message="Playback speed is 0%; choose a nonzero speed first",
            )
            return
        if not self._status.playing:
            self._send(f"resume {GROUP}")
        self._status = replace(self._status, playing=True, message="")

    def _play_towards(self, direction: str) -> None:
        """PLAY and REVERSE are the two directions of one control: each plays,
        and turns playback around when it was going the other way. The toggle
        (Space) is the one that resumes in whatever direction was current."""
        turned = self._status.direction != direction
        self._status = replace(self._status, direction=direction)
        self._play()
        if turned and self._status.playing:
            self._send(
                f"speed.set {GROUP} "
                f"{_format_number(self._signed_rate(self._status.speed_percent))}"
            )

    def _set_speed(self, value) -> None:
        speed = _number(value, "speed")
        if not 0 <= speed <= 400:
            raise ValueError("speed must be between 0 and 400 percent")
        if speed == 0:
            self._pause()
        else:
            self._send(f"speed.set {GROUP} {_format_number(self._signed_rate(speed))}")
        self._status = replace(self._status, speed_percent=speed)

    def _scrub(self, value) -> None:
        speed = _number(value, "scrubbing speed")
        if not -500 <= speed <= 500:
            raise ValueError("scrubbing speed must be between -500 and 500 percent")
        if speed == 0:
            if self._scrub_restore is not None:
                was_playing, direction = self._scrub_restore
                signed_speed = self._signed_rate(self._status.speed_percent, direction)
                if signed_speed:
                    self._send(f"speed.set {GROUP} {_format_number(signed_speed)}")
                if not was_playing or not signed_speed:
                    self._send(f"pause {GROUP} now")
                elif not self._status.playing:
                    self._send(f"resume {GROUP}")
                self._status = replace(
                    self._status,
                    playing=was_playing and bool(signed_speed),
                    direction=direction,
                )
            self._scrub_restore = None
            self._last_scrub_command = float("-inf")
            self._status = replace(self._status, scrubbing_percent=0)
            return
        if self._scrub_restore is None:
            self._scrub_restore = (self._status.playing, self._status.direction)
        self._status = replace(self._status, scrubbing_percent=speed)
        if abs(speed) <= 20 or self._clock() - self._last_scrub_command < 0.1:
            return
        self._send(f"speed.set {GROUP} {_format_number(speed / 100)}")
        if not self._status.playing:
            self._send(f"resume {GROUP}")
        self._status = replace(
            self._status,
            playing=True,
            direction="reverse" if speed < 0 else "forward",
        )
        self._last_scrub_command = self._clock()

    def execute(self, operation: PlaybackOperation, value=None) -> PlaybackStatus:
        operation = PlaybackOperation(operation)
        with self._lock:
            if operation is PlaybackOperation.PAUSE:
                self._pause()
            elif operation is PlaybackOperation.PLAY:
                self._play_towards("forward")
            elif operation is PlaybackOperation.TOGGLE:
                self._pause() if self._status.playing else self._play()
            elif operation is PlaybackOperation.REVERSE:
                self._play_towards("reverse")
            elif operation is PlaybackOperation.SPEED:
                self._set_speed(value)
            elif operation is PlaybackOperation.SCRUB:
                self._scrub(value)
            elif operation is PlaybackOperation.SEEK_MS:
                target = _number(value, "seek timestamp")
                if not 0 <= target <= self.artifact.duration_ms:
                    raise ValueError("seek timestamp is outside the recording")
                self._send(
                    f"seek {GROUP} now "
                    f"{_format_number(self.artifact.start_ms + target)}"
                )
            elif operation is PlaybackOperation.SEEK_FRAMES:
                frames = _number(value, "frames")
                if not frames.is_integer() or abs(frames) > 2**53 - 1:
                    raise ValueError("frames must be a safe integer")
                self._send(f"seek {GROUP} frame {int(frames):+d}")
            elif operation is PlaybackOperation.SEEK_SECONDS:
                seconds = _number(value, "seconds")
                milliseconds = seconds * 1000
                if abs(milliseconds) > 2**53 - 1:
                    raise ValueError("seconds are outside the supported range")
                self._send(
                    f"seek {GROUP} now "
                    f"{_format_number(milliseconds) if milliseconds < 0 else '+' + _format_number(milliseconds)}"
                )
            elif operation is PlaybackOperation.SEEK_UTC:
                if not isinstance(value, datetime) or value.tzinfo is None:
                    raise ValueError("UTC seek requires a timezone-qualified datetime")
                target = value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                self._send(f"seek {GROUP} now {target}")
            elif operation is PlaybackOperation.TAIL:
                target = self.artifact.start_ms + max(self.artifact.duration_ms - 3_000, 0)
                self._send(f"seek {GROUP} now {target}")
            return self._status


@dataclass(frozen=True)
class TranscodeConfig:
    source: Path
    output: Path
    fps: int
    wallclock_start: datetime
    force: bool = False

    def __post_init__(self):
        if not Path(self.source).is_file():
            raise FileNotFoundError(self.source)
        if isinstance(self.fps, bool) or not isinstance(self.fps, int) or not 1 <= self.fps <= 240:
            raise ValueError("fps must be an integer between 1 and 240")
        if self.wallclock_start.tzinfo is None:
            raise ValueError("wallclock_start must include a timezone")


@dataclass(frozen=True)
class ReplaySlotConfig:
    recording: Path
    loop: bool = True
    control_timeout: float = 5.0

    def __post_init__(self):
        if not isinstance(self.loop, bool):
            raise ValueError("loop must be true or false")
        if not math.isfinite(self.control_timeout) or self.control_timeout <= 0:
            raise ValueError("control_timeout must be positive")


@dataclass(frozen=True)
class JanusVideoConfig:
    host: str = "127.0.0.1"
    video_port: int = 5004
    payload_type: int = 96
    ssrc: int = 0x41565001
    rtcp_bind: str = "0.0.0.0"
    rtcp_port: int = 0

    def __post_init__(self):
        if not self.host:
            raise ValueError("Janus host is required")
        if not 1 <= self.video_port < 65535:
            raise ValueError("Janus video and paired RTCP ports must be valid")
        if not 0 <= self.payload_type <= 127:
            raise ValueError("RTP payload type must be between 0 and 127")
        if not 0 <= self.ssrc <= 0xFFFFFFFF:
            raise ValueError("RTP SSRC must be a 32-bit unsigned integer")
        if not 0 <= self.rtcp_port <= 65535:
            raise ValueError("RTCP bind port must be between 0 and 65535")


@dataclass(frozen=True)
class PlayerConfig:
    slot: ReplaySlotConfig
    janus: JanusVideoConfig


def node_add(node_type: str, name: str, group: str, **parameters) -> str:
    """One `node.add` line. Option values go out as strings: that is what the
    libav dictionaries take, whatever the script author typed."""
    params = {"type": node_type, "name": name, "group": group}
    for key, value in parameters.items():
        if key == "options" and isinstance(value, dict):
            value = {k: str(v) for k, v in value.items()}
        params[key] = value
    return "node.add " + json.dumps(params, separators=(",", ":"))


def _stringify(options: dict) -> dict:
    return {key: str(value) for key, value in options.items()}


def transcode_script(config: TranscodeConfig) -> list[str]:
    """The transcode graph, video only, all-intra H.264 with a seek table.
    CPU codecs: software decode and libx264."""
    group = TRANSCODE_NODE_GROUP
    fps = config.fps
    output = str(config.output)
    return [
        "queue.plan_capacity * 4",
        node_add("input", "replay_input", group,
                 url=str(config.source), dst="transcode_packets", eof_mode="drain"),
        node_add("demux", "replay_demux", group,
                 src="transcode_packets", routing={"?v:0": "transcode_video_packets"},
                 wait_for_keyframe=False),
        node_add("dec_video", "replay_decode", group,
                 src="transcode_video_packets", dst="transcode_decoded",
                 codec_map={"h264": "h264", "hevc": "hevc"}),
        node_add("force_fps", "replay_fps", group,
                 src="transcode_decoded", dst="transcode_fps", fps=f"{fps}/1"),
        node_add("force_keyframe", "replay_keyframes", group,
                 src="transcode_fps", dst="transcode_keyframes", interval_sec=f"1/{fps}"),
        node_add("enc_video", "replay_encoder", group,
                 src="transcode_keyframes", dst="transcode_encoded", codec="libx264",
                 options={"g": 1, "bf": 0, "profile": "baseline", "preset": "ultrafast",
                          "tune": "zerolatency", "crf": 17,
                          "x264-params": "keyint=1:scenecut=0"}),
        node_add("mux", "replay_mux", group,
                 src=["transcode_encoded"], dst="transcode_muxed", ts_sort_wait=0),
        node_add("output", "replay_output", group,
                 src="transcode_muxed", url=output, format="mpegts",
                 seek_table=f"{output}{SEEK_TABLE_SUFFIX}",
                 seek_table_text=f"{output}{TEXT_SEEK_TABLE_SUFFIX}"),
        f"group.start {group}",
    ]


def _rtp_url(config: JanusVideoConfig) -> str:
    return (
        f"rtp://{config.host}:{config.video_port}?pkt_size=1200"
        f"&rtcp_port={config.video_port + 1}"
    )


def player_script(config: PlayerConfig, artifact: ReplayArtifact) -> list[str]:
    """The player graph: a seekable input paced by the `replay` group, then the
    Janus leg with a forced keyframe every second, libx264, SPS/PPS repeated
    in band, RTP out."""
    group = PLAYER_NODE_GROUP
    fps = artifact.fps
    bitrate = "4000k"
    return [
        "queue.plan_capacity * 1",
        node_add("input", "replay_input", group, sync_group=GROUP,
                 url=str(artifact.path), dst="player_packets", loop=config.slot.loop),
        node_add("demux", "replay_demux", group,
                 src="player_packets", routing={"v:0": "player_video_packets"}),
        node_add("dec_video", "replay_decode", group,
                 src="player_video_packets", dst="player_decoded",
                 codec_map={"h264": "h264"},
                 options={"threads": 1, "flags": "low_delay"}),
        node_add("realtime", "replay_realtime", group, sync_group=GROUP,
                 src="player_decoded", dst="player_realtime", tick_period=f"1/{fps}"),
        node_add("force_keyframe", JANUS_FORCE_KEYFRAME_NODE, group,
                 src="player_realtime", dst="janus_keyframes", interval_sec="1/1"),
        # A live output: a seek must not touch the encoder (`flush: keep`), the
        # paced frames keep monotonic timestamps across it anyway.
        node_add("enc_video", "janus_encoder", group,
                 src="janus_keyframes", dst="janus_encoded", codec="libx264", flush="keep",
                 options={"b": bitrate, "maxrate": bitrate, "bufsize": bitrate, "g": fps,
                          "bf": 0, "preset": "ultrafast", "profile": "baseline",
                          "tune": "zerolatency", "x264-params": "aud=1:scenecut=0"}),
        node_add("bsf", "janus_headers", group,
                 src="janus_encoded", dst="janus_headers", bsf="dump_extra=freq=keyframe"),
        node_add("mux", "janus_mux", group,
                 src=["janus_headers"], dst="janus_muxed", ts_sort_wait=0),
        node_add("output", "janus_rtp_output", group,
                 src="janus_muxed", url=_rtp_url(config.janus), format="rtp",
                 options={"payload_type": config.janus.payload_type,
                          "rtpflags": "skip_rtcp", "ssrc": config.janus.ssrc}),
    ]


@dataclass
class TranscodeApplication:
    config: TranscodeConfig
    binary: Path
    group: str = TRANSCODE_NODE_GROUP
    log_level: str = "info"

    def run(self) -> None:
        """Runs the transcode to completion in a batch avplumber process."""
        with tempfile.NamedTemporaryFile("w", suffix=".avplumber", prefix="replay-transcode-",
                                         delete=False) as script:
            script.write("\n".join(transcode_script(self.config)) + "\n")
            script_path = Path(script.name)
        try:
            result = AvplumberProcess(self.binary, log_level=self.log_level).run_batch(script_path)
        finally:
            script_path.unlink(missing_ok=True)
        if result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-8:])
            raise RuntimeError(f"avplumber exited with status {result.returncode}:\n{tail}")


def build_transcode_application(config: TranscodeConfig, binary: str | Path | None = None) -> TranscodeApplication:
    return TranscodeApplication(config, find_avplumber(binary))


class StatusPoller:
    """Feeds `playback.status` into the controller a few times per frame, so
    every released frame is observed and a fresh one can be told from a
    repeated read."""

    def __init__(self, client: ControlClient, controller: PlaybackController,
                 *, interval: float = STATUS_POLL_INTERVAL, node_group: str = PLAYER_NODE_GROUP):
        self.client = client
        self.controller = controller
        self.interval = interval
        self.node_group = node_group
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="replay-status", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        polls = 0
        while not self._stop.is_set():
            try:
                self.controller.observe_status(self.client.status(GROUP))
                polls += 1
                if polls % 50 == 0:
                    for outcome in self.client.group_status(self.node_group).get("outcomes", []):
                        if outcome.startswith(("failed:", "panicked:")):
                            self.controller.set_error(outcome)
            except Exception as exc:  # the connection or the server went away
                if self._stop.is_set():
                    return
                self.controller.set_error(str(exc))
                time.sleep(0.5)
                continue
            self._stop.wait(self.interval)


@dataclass
class PlayerApplication:
    config: PlayerConfig
    artifact: ReplayArtifact
    controller: PlaybackController
    client: ControlClient
    rtcp_feedback_listener: object
    process: AvplumberProcess | None = None
    poller: StatusPoller | None = None
    node_group: str = PLAYER_NODE_GROUP
    _stopped: bool = False

    def _wait_for(self, predicate, description: str) -> None:
        deadline = time.monotonic() + self.config.slot.control_timeout
        while not predicate():
            error = self.controller.status().error
            if error:
                raise RuntimeError(error)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {description}")
            time.sleep(0.01)

    def start(self) -> None:
        try:
            if self.process is not None:
                self.process.start()
                self.client = self.process.connect(self.config.slot.control_timeout)
            elif not self.client.connected:
                self.client.connect()
            for line in player_script(self.config, self.artifact):
                self.client.command(line)
            self.client.command(f"group.start {self.node_group}")
            self.poller = StatusPoller(self.client, self.controller, node_group=self.node_group)
            self.poller.start()
            self._wait_for(lambda: self.controller.status().ready, "the first source frame")
            self.rtcp_feedback_listener.start()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.rtcp_feedback_listener.stop()
        if self.poller is not None:
            self.poller.stop()
        if self.client.connected:
            try:
                self.client.command(f"group.stop {self.node_group}")
            except Exception:
                pass
            self.client.close()
        if self.process is not None:
            self.process.stop(self.config.slot.control_timeout)


def load_rtcp_listener():
    from rtcp_feedback import RtcpFeedbackListener
    return RtcpFeedbackListener


def build_player_application(config: PlayerConfig, *, binary: str | Path | None = None,
                             connect: tuple[str, int] | None = None,
                             listener_factory: Callable | None = None,
                             avplumber_log: Path | None = None) -> PlayerApplication:
    """One replay slot and one Janus output on a Rust avplumber: spawned from
    `binary` (or `$AVPLUMBER_BIN`), or an already running one at `connect`."""
    artifact = validate_recording(config.slot.recording)
    if connect is not None:
        process = None
        client = ControlClient(*connect, timeout=config.slot.control_timeout)
    else:
        process = AvplumberProcess(find_avplumber(binary), log_path=avplumber_log)
        client = ControlClient(process.host, process.port, timeout=config.slot.control_timeout)

    # Both closures go through the application, whose client a spawned process
    # replaces on start (and a test replaces with a fake).
    def command(value: str) -> None:
        application.client.command(value)
        if value.startswith("seek "):
            application.client.command(JANUS_FORCE_KEYFRAME_COMMAND)

    controller = PlaybackController(
        command,
        artifact,
        loop=config.slot.loop,
        control_timeout=config.slot.control_timeout,
    )
    listener_factory = listener_factory or load_rtcp_listener()
    listener = listener_factory(
        bind_host=config.janus.rtcp_bind,
        bind_port=config.janus.rtcp_port,
        janus_host=config.janus.host,
        janus_rtcp_port=config.janus.video_port + 1,
        media_ssrc=config.janus.ssrc,
        on_keyframe_request=lambda _request: application.client.command(JANUS_FORCE_KEYFRAME_COMMAND),
    )
    application = PlayerApplication(config, artifact, controller, client, listener, process=process)
    return application
