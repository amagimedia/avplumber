"""Shared recording, graph, and control logic for the replay demos."""

from __future__ import annotations

import struct
import math
import threading
import time
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple


SEEK_STRUCT = struct.Struct("=qQ")
HISTORY_STRUCT = struct.Struct("=qqqq")
SEEK_TABLE_SUFFIX = "+seek"
TEXT_SEEK_TABLE_SUFFIX = "+txt"
HISTORY_SUFFIX = "+history"
JANUS_FORCE_KEYFRAME_NODE = "janus_force_keyframe"
JANUS_FORCE_KEYFRAME_COMMAND = (
    f"node.object.set {JANUS_FORCE_KEYFRAME_NODE} trigger true"
)
GPU_DEVICE = "replay_gpu"


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
    """Translate SSGW v2 single-source operations into AVPlumber commands."""

    PAUSE_TEAM = "replay_pause"
    SPEED_TEAM = "replay_speed"
    SYNC_TEAM = "replay_sync"
    TRANSITION_TEAM = "replay_transition"

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

    def observe_metadata(self, metadata) -> PlaybackStatus:
        try:
            if "frame_no" not in metadata or "frame_ts" not in metadata:
                return self.status()
            return self.observe(
                frame_number=int(metadata["frame_no"]),
                media_timestamp_ms=int(metadata["frame_ts"]),
            )
        except (TypeError, ValueError):
            return self.status()

    def _send(self, command: str) -> None:
        try:
            self._command(command)
        except Exception as exc:
            self._status = replace(self._status, error=str(exc), last_command=command)
            raise
        self._status = replace(self._status, error="", last_command=command)

    def _pause(self) -> None:
        if self._status.playing:
            self._send(f"pause {self.PAUSE_TEAM} now")
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
            self._send(f"resume {self.PAUSE_TEAM}")
        self._status = replace(self._status, playing=True, message="")

    def _wait_for_quiet_output(self, deadline: float) -> None:
        quiet_period = 2 / self.artifact.fps
        serial = self._observation_serial
        quiet_since = time.monotonic()
        while True:
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("timed out draining the speed transition")
            if now - quiet_since >= quiet_period:
                return
            self._observed.wait(min(quiet_period - (now - quiet_since), deadline - now))
            if self._observation_serial != serial:
                serial = self._observation_serial
                quiet_since = time.monotonic()

    def _decrease_speed_cleanly(self, speed: float, sign: int) -> None:
        transition_error = None
        gate_closed = False
        try:
            self._send(f"pause {self.TRANSITION_TEAM} now")
            gate_closed = True
            self._wait_for_quiet_output(time.monotonic() + self._control_timeout)
            self._send(
                f"speed.set {self.SPEED_TEAM} "
                f"{_format_number(sign * speed / 100)}"
            )
            self._status = replace(self._status, speed_percent=speed)
            self._send(JANUS_FORCE_KEYFRAME_COMMAND)
        except Exception as exc:
            transition_error = exc
        if gate_closed:
            try:
                self._send(f"resume {self.TRANSITION_TEAM}")
            except Exception as exc:
                if transition_error is None:
                    transition_error = exc
        if transition_error is not None:
            self._status = replace(
                self._status,
                message="",
                error=f"speed transition failed: {transition_error}",
            )
            raise transition_error

    def _set_speed(self, value) -> None:
        speed = _number(value, "speed")
        if not 0 <= speed <= 400:
            raise ValueError("speed must be between 0 and 400 percent")
        if speed == 0:
            self._pause()
        else:
            sign = -1 if self._status.direction == "reverse" else 1
            if (self._status.ready and self._status.playing
                    and self._status.speed_percent > 100 and speed <= 100):
                self._decrease_speed_cleanly(speed, sign)
            else:
                self._send(f"speed.set {self.SPEED_TEAM} {_format_number(sign * speed / 100)}")
        self._status = replace(self._status, speed_percent=speed)

    def _scrub(self, value) -> None:
        speed = _number(value, "scrubbing speed")
        if not -500 <= speed <= 500:
            raise ValueError("scrubbing speed must be between -500 and 500 percent")
        if speed == 0:
            if self._scrub_restore is not None:
                was_playing, direction = self._scrub_restore
                signed_speed = self._status.speed_percent / 100 * (-1 if direction == "reverse" else 1)
                if signed_speed:
                    self._send(f"speed.set {self.SPEED_TEAM} {_format_number(signed_speed)}")
                if not was_playing or not signed_speed:
                    self._send(f"pause {self.PAUSE_TEAM} now")
                elif not self._status.playing:
                    self._send(f"resume {self.PAUSE_TEAM}")
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
        self._send(f"speed.set {self.SPEED_TEAM} {_format_number(speed / 100)}")
        if not self._status.playing:
            self._send(f"resume {self.PAUSE_TEAM}")
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
                self._play()
            elif operation is PlaybackOperation.TOGGLE:
                self._pause() if self._status.playing else self._play()
            elif operation is PlaybackOperation.REVERSE:
                self._status = replace(self._status, direction="reverse")
                self._play()
                if self._status.playing:
                    self._send(
                        f"speed.set {self.SPEED_TEAM} "
                        f"{_format_number(-self._status.speed_percent / 100)}"
                    )
            elif operation is PlaybackOperation.SPEED:
                self._set_speed(value)
            elif operation is PlaybackOperation.SCRUB:
                self._scrub(value)
            elif operation is PlaybackOperation.SEEK_MS:
                target = _number(value, "seek timestamp")
                if not 0 <= target <= self.artifact.duration_ms:
                    raise ValueError("seek timestamp is outside the recording")
                self._send(
                    f"seek {self.SYNC_TEAM} now "
                    f"{_format_number(self.artifact.start_ms + target)}"
                )
            elif operation is PlaybackOperation.SEEK_FRAMES:
                frames = _number(value, "frames")
                if not frames.is_integer() or abs(frames) > 2**53 - 1:
                    raise ValueError("frames must be a safe integer")
                self._send(
                    f"seek {self.SYNC_TEAM} frame "
                    f"{int(frames):+d}"
                )
            elif operation is PlaybackOperation.SEEK_SECONDS:
                seconds = _number(value, "seconds")
                milliseconds = seconds * 1000
                if abs(milliseconds) > 2**53 - 1:
                    raise ValueError("seconds are outside the supported range")
                self._send(
                    f"seek {self.SYNC_TEAM} now "
                    f"{_format_number(milliseconds) if milliseconds < 0 else '+' + _format_number(milliseconds)}"
                )
            elif operation is PlaybackOperation.SEEK_UTC:
                if not isinstance(value, datetime) or value.tzinfo is None:
                    raise ValueError("UTC seek requires a timezone-qualified datetime")
                target = value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                self._send(f"seek {self.SYNC_TEAM} now {target}")
            elif operation is PlaybackOperation.TAIL:
                target = self.artifact.start_ms + max(self.artifact.duration_ms - 3_000, 0)
                self._send(f"seek {self.SYNC_TEAM} now {target}")
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


@dataclass
class TranscodeApplication:
    avp: object
    config: TranscodeConfig
    group: str = "transcode"

    def run(self) -> None:
        self.avp.executeCommandsFromString(
            "event.on.node.finished transcode_finished replay_output"
        )
        try:
            self.avp.group(self.group).startNodes()
            self.avp.executeCommandsFromString("event.wait transcode_finished")
        finally:
            self.avp.setExceptionCallback(None)
            self.avp.shutdown()


def load_avp_api():
    from pyplumber import AVPlumber
    from pyplumber.node import (
        DecVideo,
        Bsf,
        Demux,
        EncVideo,
        ForceFPS,
        ForceKeyFrame,
        Input,
        InputRec,
        Mux,
        Output,
        Pause,
        PythonNode,
        RealtimeVideoFrame,
        SpeedVideo,
    )
    from pyplumber.rtcp_feedback import RtcpFeedbackListener

    class PositionProbe(PythonNode):
        def __init__(self, parameters, controller):
            super().__init__(parameters)
            self.controller = controller

        def process(self):
            frame = self._src.get()
            if frame:
                self.controller.observe_metadata(frame.metadata)
                self._dst.enqueue(frame)

        @property
        def worker_running(self):
            return self._wrapper is not None and self._wrapper.isWorking

        def request_stop(self):
            self._wrapper.stop(False)

        def detach(self):
            if self.worker_running:
                raise RuntimeError("position probe is still running")
            self._src = None
            self._dst = None
            self._wrapper = None
            self._avplumber = None

    return SimpleNamespace(
        AVPlumber=AVPlumber,
        Bsf=Bsf,
        Input=Input,
        InputRec=InputRec,
        Demux=Demux,
        DecVideo=DecVideo,
        ForceFPS=ForceFPS,
        ForceKeyFrame=ForceKeyFrame,
        EncVideo=EncVideo,
        Mux=Mux,
        Output=Output,
        Pause=Pause,
        PositionProbe=PositionProbe,
        RealtimeVideoFrame=RealtimeVideoFrame,
        RtcpFeedbackListener=RtcpFeedbackListener,
        SpeedVideo=SpeedVideo,
    )


def _add_node(avp, node_type, name: str, group: str, **parameters):
    node = node_type({"name": name, "group": group, **parameters})
    avp.addNode(node)
    return node


def _init_cuda(avp) -> None:
    avp.executeCommandsFromString(
        f'hwaccel.init {{ "name": "{GPU_DEVICE}", "type": "cuda" }}'
    )


def build_transcode_application(config: TranscodeConfig, api=None) -> TranscodeApplication:
    api = api or load_avp_api()
    avp = api.AVPlumber()
    _init_cuda(avp)
    avp.edges.planCapacity("*", 4)
    group = "transcode"
    _add_node(avp, api.Input, "replay_input", group,
              url=str(config.source), dst="transcode_packets", eof_mode="drain")
    _add_node(avp, api.Demux, "replay_demux", group,
              src="transcode_packets", routing={"?v:0": "transcode_video_packets"},
              wait_for_keyframe=False)
    _add_node(avp, api.DecVideo, "replay_decode", group,
              src="transcode_video_packets", dst="transcode_decoded",
              pixel_format="cuda", hwaccel=GPU_DEVICE,
              codec_map={"h264": "h264_cuvid", "hevc": "hevc_cuvid"},
              hwaccel_only_for_codecs=["h264", "hevc"])
    _add_node(avp, api.ForceFPS, "replay_fps", group,
              src="transcode_decoded", dst="transcode_fps", fps=f"{config.fps}/1")
    _add_node(avp, api.ForceKeyFrame, "replay_keyframes", group,
              src="transcode_fps", dst="transcode_keyframes",
              interval_sec=f"1/{config.fps}")
    _add_node(avp, api.EncVideo, "replay_encoder", group,
              src="transcode_keyframes", dst="transcode_encoded",
              codec="h264_nvenc", hwaccel=GPU_DEVICE,
              options={"g": 1, "bf": 0, "profile": "baseline", "rc": "vbr",
                       "cq": 17, "b": 0, "tune": "ull", "rc-lookahead": 0,
                       "zerolatency": 1, "delay": 0})
    _add_node(avp, api.Mux, "replay_mux", group,
              src=["transcode_encoded"], dst="transcode_muxed", ts_sort_wait=0)
    _add_node(avp, api.Output, "replay_output", group,
              src="transcode_muxed", url=str(config.output), format="mpegts",
              seek_table=f"{config.output}{SEEK_TABLE_SUFFIX}",
              seek_table_text=f"{config.output}{TEXT_SEEK_TABLE_SUFFIX}")
    return TranscodeApplication(avp, config, group)


@dataclass
class PlayerApplication:
    avp: object
    config: PlayerConfig
    artifact: ReplayArtifact
    controller: PlaybackController
    position_probe: object
    rtcp_feedback_listener: object
    video_edge: str = "player_observed"
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
            self.avp.group("player").startNodes()
            self._wait_for(lambda: self.controller.status().ready, "the first source frame")
            self.avp.group("output").startNodes()
            self._wait_for(
                lambda: self.avp.node("janus_rtp_output").isWorking,
                "the Janus RTP output",
            )
            self.rtcp_feedback_listener.start()
            self.avp.setReady()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._stopped:
            return
        self.avp.setExceptionCallback(None)
        self.rtcp_feedback_listener.stop()
        if self.position_probe.worker_running:
            self.position_probe.request_stop()
            deadline = time.monotonic() + self.config.slot.control_timeout
            while self.position_probe.worker_running:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out stopping the position probe")
                time.sleep(0.01)
        self.position_probe.detach()
        self.avp.group("output").stopNodes()
        self.avp.group("player").stopNodes()
        self.avp.shutdown()
        self._stopped = True


def _rtp_url(config: JanusVideoConfig) -> str:
    return (
        f"rtp://{config.host}:{config.video_port}?pkt_size=1200"
        f"&rtcp_port={config.video_port + 1}"
    )


def build_player_application(config: PlayerConfig, api=None) -> PlayerApplication:
    artifact = validate_recording(config.slot.recording)
    api = api or load_avp_api()
    avp = api.AVPlumber()
    _init_cuda(avp)
    avp.edges.planCapacity("*", 1)
    def command(value: str) -> None:
        avp.executeCommandsFromString(value)
        if value.startswith("seek "):
            avp.executeCommandsFromString(JANUS_FORCE_KEYFRAME_COMMAND)

    controller = PlaybackController(
        command,
        artifact,
        loop=config.slot.loop,
        control_timeout=config.slot.control_timeout,
    )
    avp.on_exception = lambda name, node_type, message: controller.set_error(
        f"{name} ({node_type}): {message}"
    )
    fps = artifact.fps

    _add_node(avp, api.InputRec, "replay_input", "player",
              url=str(artifact.path), dst="player_packets", timeout=-1, preseek=0,
              seek_table="", ts_offsets="", team="replay_input_seek",
              timestamp_source="wallclock", pause_team=PlaybackController.PAUSE_TEAM,
              stop_delay=3000, loop=config.slot.loop)
    _add_node(avp, api.Demux, "replay_demux", "player",
              src="player_packets", routing={"v:0": "player_video_packets"})
    _add_node(avp, api.DecVideo, "replay_decode", "player",
              src="player_video_packets", dst="player_decoded", pixel_format="cuda",
              hwaccel=GPU_DEVICE, codec_map={"h264": "h264_cuvid"},
              hwaccel_only_for_codecs=["h264"], flush_magic=True)
    _add_node(avp, api.SpeedVideo, "replay_speed", "player",
              src="player_decoded", dst="player_speed_raw", team=PlaybackController.SPEED_TEAM,
              sync_team=PlaybackController.SYNC_TEAM, sync_node="replay_realtime", speed=1)
    _add_node(avp, api.Pause, "replay_transition_gate", "player",
              src="player_speed_raw", dst="player_speed",
              team=PlaybackController.TRANSITION_TEAM)
    _add_node(avp, api.ForceFPS, "replay_fps", "player",
              src="player_speed", dst="player_fps", fps=f"{fps}/1")
    _add_node(avp, api.Pause, "replay_pause", "player",
              src="player_fps", dst="player_paused", team=PlaybackController.PAUSE_TEAM,
              sync_team=PlaybackController.SYNC_TEAM)
    _add_node(avp, api.RealtimeVideoFrame, "replay_realtime", "player",
              src="player_paused", dst="player_realtime", team=PlaybackController.SYNC_TEAM,
              set_pts=True, tick_period=f"1/{fps}", negative_time_tolerance=1 / fps,
              negative_time_discard=1 / fps, discontinuity_threshold=3)
    probe = api.PositionProbe({
        "name": "replay_position",
        "src": "player_realtime",
        "dst": "player_observed",
        "data_type": "VideoFrame",
        "group": "player",
    }, controller)
    avp.addNode(probe)

    bitrate = "4000k"
    _add_node(avp, api.ForceKeyFrame, JANUS_FORCE_KEYFRAME_NODE, "output",
              src="player_observed", dst="janus_keyframes", interval_sec="1/1")
    _add_node(avp, api.EncVideo, "janus_encoder", "output",
              src="janus_keyframes", dst="janus_encoded", codec="h264_nvenc",
              hwaccel=GPU_DEVICE, options={
                  "b": bitrate, "maxrate": bitrate, "bufsize": bitrate, "g": fps,
                  "bf": 0, "preset": "p6", "profile": "baseline", "tune": "ull",
                  "rc": "cbr", "rc-lookahead": 0, "zerolatency": 1, "delay": 0,
                  "forced-idr": 1, "no-scenecut": 1, "strict_gop": 1, "aud": 1,
                  "spatial-aq": 1, "temporal-aq": 0,
              })
    _add_node(avp, api.Bsf, "janus_headers", "output",
              src="janus_encoded", dst="janus_headers", bsf="dump_extra=freq=keyframe")
    _add_node(avp, api.Mux, "janus_mux", "output",
              src=["janus_headers"], dst="janus_muxed", ts_sort_wait=0)
    _add_node(avp, api.Output, "janus_rtp_output", "output",
              src="janus_muxed", url=_rtp_url(config.janus), format="rtp",
              options={"payload_type": config.janus.payload_type,
                       "rtpflags": "skip_rtcp", "ssrc": config.janus.ssrc})

    listener = api.RtcpFeedbackListener(
        bind_host=config.janus.rtcp_bind,
        bind_port=config.janus.rtcp_port,
        janus_host=config.janus.host,
        janus_rtcp_port=config.janus.video_port + 1,
        media_ssrc=config.janus.ssrc,
        on_keyframe_request=lambda _request: avp.executeCommandsFromString(
            JANUS_FORCE_KEYFRAME_COMMAND
        ),
    )
    return PlayerApplication(avp, config, artifact, controller, probe, listener)
