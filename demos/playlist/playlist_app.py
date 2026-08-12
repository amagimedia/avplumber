#!/usr/bin/env python3
"""Asynchronous live backend for the playlist regression harness.

Playlist elements occupy stable inputs on AVPlumber's existing C++ source
switcher. The replay Janus output group is permanent; item and playlist Stop
operations only stop item groups. All blocking graph operations are serialized
off the Textual thread.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from playlist import (BackendEvent, DEFAULT_SLOT_CAPACITY, GPU_DEVICE,
                      SWITCHER_NAME, SWITCHER_TYPE, Clip, NodeSpec,
                      PlaylistController, PlaylistMode, item_edge, item_group,
                      item_node_names, item_pause_team, plan_item_nodes,
                      plan_switch_nodes)

JANUS_FORCE_KEYFRAME_NODE = "janus_force_keyframe"
JANUS_FORCE_KEYFRAME_COMMAND = (
    f"node.object.set {JANUS_FORCE_KEYFRAME_NODE} trigger true")


@dataclass(frozen=True)
class JanusVideoConfig:
    host: str = "127.0.0.1"
    video_port: int = 5004
    payload_type: int = 96
    ssrc: int = 0x41565001
    rtcp_bind: str = "0.0.0.0"
    rtcp_port: int = 0

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("Janus host is required")
        if not 1 <= self.video_port < 65535:
            raise ValueError("Janus video and paired RTCP ports must be valid")
        if not 0 <= self.payload_type <= 127:
            raise ValueError("RTP payload type must be between 0 and 127")
        if not 0 <= self.ssrc <= 0xFFFFFFFF:
            raise ValueError("RTP SSRC must be a 32-bit unsigned integer")


@dataclass(frozen=True)
class PlaylistConfig:
    clips: List[Clip]
    mode: PlaylistMode = PlaylistMode.LOOP_ALL
    fps: int = 30
    width: int = 1920
    height: int = 1080
    slot_capacity: int = DEFAULT_SLOT_CAPACITY
    janus: JanusVideoConfig = field(default_factory=JanusVideoConfig)
    control_timeout: float = 10.0
    log_file: str = "playlist-demo.log"

    def __post_init__(self) -> None:
        if not self.clips:
            raise ValueError("playlist must contain at least one element")
        if self.fps <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("output format must be positive")
        if len(self.clips) > self.slot_capacity:
            raise ValueError("playlist has more elements than switcher slots")
        if self.control_timeout <= 0:
            raise ValueError("control timeout must be positive")


def load_avp_api():
    from pyplumber import AVPlumber
    from pyplumber.node import (Bsf, DecVideo, Demux, EncVideo, ForceFPS,
                                ForceKeyFrame, InputRec, Mux, Output,
                                PythonNode, RealtimeVideoFrame, SourceSwitcher,
                                SpeedVideo)
    from pyplumber.rtcp_feedback import RtcpFeedbackListener

    class SourceSwitcherVideoFrame(SourceSwitcher):
        TYPE = SWITCHER_TYPE

    by_type = {
        "input_rec": InputRec,
        "demux": Demux,
        "dec_video": DecVideo,
        "speed_video": SpeedVideo,
        "force_fps": ForceFPS,
        SWITCHER_TYPE: SourceSwitcherVideoFrame,
        "realtime<av::VideoFrame>": RealtimeVideoFrame,
    }
    return SimpleNamespace(
        AVPlumber=AVPlumber,
        Bsf=Bsf,
        EncVideo=EncVideo,
        ForceKeyFrame=ForceKeyFrame,
        Mux=Mux,
        Output=Output,
        PythonNode=PythonNode,
        RtcpFeedbackListener=RtcpFeedbackListener,
        by_type=by_type,
    )


def _add_node(avp, node_type, name: str, group: str, **params):
    node = node_type({"name": name, "group": group, **params})
    avp.addNode(node)
    return node


def _add_spec(avp, api, spec: NodeSpec):
    return _add_node(
        avp, api.by_type[spec.type], spec.name, spec.group, **spec.params)


def _spec_command(spec: NodeSpec) -> str:
    return "node.add " + json.dumps({
        "type": spec.type,
        "name": spec.name,
        "group": spec.group,
        **spec.params,
    }, separators=(",", ":"))


def _rtp_url(config: JanusVideoConfig) -> str:
    return (
        f"rtp://{config.host}:{config.video_port}?pkt_size=1200"
        f"&rtcp_port={config.video_port + 1}"
    )


def _clip_fingerprint(clip: Clip) -> Tuple:
    return (
        clip.url, clip.element_mode.value, clip.play_from_ms, clip.play_to_ms,
        clip.duration_ms, clip.speed,
    )


@dataclass(frozen=True)
class _Task:
    kind: str
    request_id: Optional[int] = None
    item_id: Optional[str] = None
    clip: Optional[Clip] = None


class AsyncPlaylistBackend:
    """Quick public methods plus one serialized AVPlumber worker."""

    def __init__(self, avp, api, config: PlaylistConfig):
        self.avp = avp
        self.api = api
        self.config = config
        self._tasks: queue.Queue[_Task] = queue.Queue()
        self._events: queue.SimpleQueue[BackendEvent] = queue.SimpleQueue()
        self._state_lock = threading.Lock()
        self._frame_condition = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._closing = threading.Event()
        self._latest_request: Optional[int] = None
        self._active_item: Optional[str] = None
        self._active_slot: Optional[int] = None
        self._probe_item: Optional[str] = None
        self._frame_sequence = 0
        self._item_frame_sequences: Dict[int, int] = {}
        self._slot_errors: Dict[int, str] = {}
        self._switch_started = False
        self._output_started = False
        self._output_alive = False
        self._listener = None
        self._slot_items: Dict[int, str] = {}
        self._item_slots: Dict[str, int] = {}
        self._slot_fingerprints: Dict[int, Tuple] = {}
        self._slot_generations: Dict[int, int] = {}
        self._next_generations: Dict[int, int] = {}
        self._all_item_groups = set()
        self._built_slots = set()
        self._watched_slots = set()
        self._running_slots = set()

    def bind_listener(self, listener) -> None:
        self._listener = listener

    def register_built_item(self, item_id: str, slot: int, clip: Clip) -> None:
        self._slot_items[slot] = item_id
        self._item_slots[item_id] = slot
        self._slot_fingerprints[slot] = _clip_fingerprint(clip)
        self._slot_generations[slot] = 0
        self._next_generations[slot] = 1
        self._all_item_groups.add(item_group(slot))
        self._built_slots.add(slot)
        self._watch_slot(slot)

    # Public methods never wait for AVPlumber.
    def play_item(self, request_id: int, item_id: str, clip: Clip) -> None:
        with self._state_lock:
            self._latest_request = request_id
        self._tasks.put(_Task("play", request_id, item_id, clip))

    def resume_item(self, item_id: str) -> None:
        self._tasks.put(_Task("resume", item_id=item_id))

    def pause_item(self, item_id: str) -> None:
        self._tasks.put(_Task("pause", item_id=item_id))

    def stop_item(self, item_id: str) -> None:
        self._tasks.put(_Task("stop", item_id=item_id))

    def cancel_activation(self) -> None:
        with self._state_lock:
            self._latest_request = None

    def remove_item(self, item_id: str) -> None:
        self._tasks.put(_Task("remove", item_id=item_id))

    def output_alive(self) -> bool:
        with self._state_lock:
            return self._output_alive

    def poll_events(self) -> List[BackendEvent]:
        result = []
        while True:
            try:
                result.append(self._events.get_nowait())
            except queue.Empty:
                return result

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="playlist-backend", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.cancel_activation()
        self._closing.set()
        self._tasks.put(_Task("shutdown"))
        if self._thread is not None:
            self._thread.join(self.config.control_timeout)
            if self._thread.is_alive():
                raise TimeoutError("playlist backend did not stop")
            self._thread = None

    def _emit(self, event: BackendEvent) -> None:
        self._events.put(event)

    def _execute(self, command: str) -> None:
        self.avp.executeCommandsFromString(command)

    def _is_current(self, request_id: int) -> bool:
        with self._state_lock:
            return self._latest_request == request_id and not self._closing.is_set()

    def _set_output_alive(self, value: bool) -> None:
        with self._state_lock:
            changed = self._output_alive != value
            self._output_alive = value
        if changed:
            self._emit(BackendEvent("health", value=value))

    def _free_slot(self, exclude: Optional[int] = None) -> int:
        for slot in range(self.config.slot_capacity):
            if slot != exclude and slot not in self._slot_items:
                return slot
        raise RuntimeError("no free source-switcher slot")

    def _stop_slot(self, slot: int) -> None:
        if slot not in self._running_slots:
            return
        generation = self._slot_generations[slot]
        self._execute(f"group.stop {item_group(slot, generation)}")
        deadline = time.monotonic() + self.config.control_timeout
        while any(self.avp.node(name).isWorking
                  for name in item_node_names(slot, generation)):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"playlist slot {slot} did not stop")
            time.sleep(0.01)
        self._running_slots.discard(slot)

    def _release_slot(self, slot: int) -> None:
        """Stop the source generation before reusing its fixed switch edge."""
        self._stop_slot(slot)
        self._built_slots.discard(slot)
        self._slot_fingerprints.pop(slot, None)
        self._slot_generations.pop(slot, None)
        old_item = self._slot_items.pop(slot, None)
        if old_item is not None and self._item_slots.get(old_item) == slot:
            self._item_slots.pop(old_item, None)
        with self._frame_condition:
            self._slot_errors.pop(slot, None)

    def _build_slot(self, slot: int, item_id: str, clip: Clip) -> None:
        if slot in self._built_slots:
            raise RuntimeError(f"playlist slot {slot} is already built")
        generation = self._next_generations.get(slot, 0)
        self._next_generations[slot] = generation + 1
        for spec in plan_item_nodes(
                slot, clip, self.config.fps, generation=generation):
            self._execute(_spec_command(spec))
        self._slot_items[slot] = item_id
        self._item_slots[item_id] = slot
        self._slot_fingerprints[slot] = _clip_fingerprint(clip)
        self._slot_generations[slot] = generation
        self._all_item_groups.add(item_group(slot, generation))
        self._built_slots.add(slot)
        self._watch_slot(slot)

    def _watch_slot(self, slot: int) -> None:
        if slot in self._watched_slots:
            return
        self._watched_slots.add(slot)
        self._item_frame_sequences.setdefault(slot, 0)
        self.avp.getEdge(item_edge(slot, "normalized")).addWiretapCallback(
            lambda _frame, watched_slot=slot: self.observe_item_frame(watched_slot))

    def _start_and_ready_slot(self, slot: int) -> None:
        self._stop_slot(slot)
        generation = self._slot_generations[slot]
        with self._frame_condition:
            self._slot_errors.pop(slot, None)
        self._execute(f"pause {item_pause_team(slot, generation)} now")
        self._execute(f"group.start {item_group(slot, generation)}")
        self._running_slots.add(slot)

    def _wait_for_item_frame(self, slot: int, baseline: int) -> None:
        deadline = time.monotonic() + self.config.control_timeout
        with self._frame_condition:
            while self._item_frame_sequences.get(slot, 0) <= baseline:
                if slot in self._slot_errors:
                    raise RuntimeError(self._slot_errors[slot])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"item slot {slot} produced no frame")
                self._frame_condition.wait(min(remaining, 0.1))

    def _prepare_slot(self, item_id: str, clip: Clip) -> Tuple[int, Optional[int]]:
        existing = self._item_slots.get(item_id)
        fingerprint = _clip_fingerprint(clip)
        replaced_slot = None

        if existing is None:
            slot = self._free_slot()
            self._build_slot(slot, item_id, clip)
        elif self._slot_fingerprints.get(existing) != fingerprint:
            slot = self._free_slot(exclude=existing)
            self._build_slot(slot, item_id, clip)
            replaced_slot = existing
        else:
            slot = existing
        self._start_and_ready_slot(slot)
        return slot, replaced_slot

    def _wait_for_frames(self, baseline: int, count: int = 2) -> None:
        deadline = time.monotonic() + self.config.control_timeout
        with self._frame_condition:
            while self._frame_sequence < baseline + count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("selected item did not reach the shared output")
                self._frame_condition.wait(min(remaining, 0.1))

    def _ensure_switch_started(self) -> None:
        if not self._switch_started:
            self._execute("group.start switch")
            self._switch_started = True

    def _ensure_output_started(self) -> None:
        if self._output_started:
            return
        self._execute("group.start output")
        deadline = time.monotonic() + self.config.control_timeout
        while not self.avp.node("janus_rtp_output").isWorking:
            if time.monotonic() >= deadline:
                raise TimeoutError("Janus RTP output did not start")
            time.sleep(0.01)
        if self._listener is not None:
            self._listener.start()
        self._output_started = True
        self._set_output_alive(True)

    def _activate(self, task: _Task) -> None:
        assert task.request_id is not None and task.item_id is not None
        assert task.clip is not None
        slot = None
        replaced_slot = None
        previous_item, previous_slot = self._active_item, self._active_slot
        previous_generation = (None if previous_slot is None else
                               self._slot_generations.get(previous_slot))
        try:
            # Permanent consumers must exist before a paused source starts.
            # Otherwise its first ready frame can be stranded on an edge that
            # had no consumer when the source group was created.
            self._ensure_switch_started()
            slot, replaced_slot = self._prepare_slot(task.item_id, task.clip)
            if not self._is_current(task.request_id):
                self._stop_slot(slot)
                return

            with self._frame_condition:
                item_baseline = self._item_frame_sequences.get(slot, 0)
            generation = self._slot_generations[slot]
            self._execute(f"resume {item_pause_team(slot, generation)}")
            self._wait_for_item_frame(slot, item_baseline)
            self._execute(f"pause {item_pause_team(slot, generation)} now")
            if not self._is_current(task.request_id):
                self._stop_slot(slot)
                return

            # The encoder/output needs the preheated source's timebase, but it
            # must start before the source is resumed and selected so the
            # position-probe edge cannot fill while readiness is measured.
            self._ensure_output_started()

            with self._frame_condition:
                baseline = self._frame_sequence
            self._probe_item = task.item_id
            self._active_item, self._active_slot = task.item_id, slot
            self._execute(f"resume {item_pause_team(slot, generation)}")
            self._execute(f"node.object.set {SWITCHER_NAME} active {slot}")
            if self._output_started:
                self._execute(JANUS_FORCE_KEYFRAME_COMMAND)
            self._wait_for_frames(baseline)

            if not self._is_current(task.request_id):
                return
            if previous_slot is not None and previous_slot != slot:
                self._stop_slot(previous_slot)
            if replaced_slot is not None and replaced_slot != slot:
                self._release_slot(replaced_slot)
            self._emit(BackendEvent(
                "ready", task.item_id, task.request_id))
        except Exception as exc:
            if (slot is not None and self._active_slot == slot
                    and previous_slot is not None and previous_slot != slot
                    and previous_generation is not None):
                try:
                    self._execute(
                        f"resume {item_pause_team(previous_slot, previous_generation)}")
                    self._execute(
                        f"node.object.set {SWITCHER_NAME} active {previous_slot}")
                    self._probe_item = previous_item
                    self._active_item, self._active_slot = previous_item, previous_slot
                except Exception as rollback_error:
                    self._emit(BackendEvent(
                        "error", task.item_id,
                        message=f"source rollback failed: {rollback_error}"))
            if (slot is not None
                    and self._item_slots.get(task.item_id) == slot):
                if replaced_slot is None:
                    self._item_slots.pop(task.item_id, None)
                else:
                    self._item_slots[task.item_id] = replaced_slot
            if slot is not None and slot != self._active_slot:
                self._release_slot(slot)
            self._emit(BackendEvent(
                "failed", task.item_id, task.request_id, message=str(exc)))

    def _handle(self, task: _Task) -> None:
        if task.kind == "play":
            self._activate(task)
            return
        item_id = task.item_id
        slot = None if item_id is None else self._item_slots.get(item_id)
        generation = (None if slot is None else
                      self._slot_generations.get(slot))
        if task.kind == "pause" and slot is not None:
            self._execute(f"pause {item_pause_team(slot, generation)} now")
        elif task.kind == "resume" and slot is not None:
            self._execute(f"resume {item_pause_team(slot, generation)}")
        elif task.kind == "stop" and slot is not None:
            self._stop_slot(slot)
        elif task.kind == "remove" and slot is not None:
            self._release_slot(slot)

    def _run(self) -> None:
        if self.config.log_file:
            self.avp.setLogFile(self.config.log_file)
        while not self._closing.is_set():
            try:
                task = self._tasks.get(timeout=0.1)
            except queue.Empty:
                if self._output_started:
                    try:
                        self._set_output_alive(
                            bool(self.avp.node("janus_rtp_output").isWorking))
                    except Exception as exc:
                        self._set_output_alive(False)
                        self._emit(BackendEvent("error", message=str(exc)))
                continue
            if task.kind == "shutdown":
                break
            try:
                self._handle(task)
            except Exception as exc:
                self._emit(BackendEvent("error", task.item_id, message=str(exc)))

    # Called by the Python probe on its graph worker thread.
    def report_graph_exception(self, name: str, node_type: str,
                               message: str) -> None:
        detail = f"{name} ({node_type}): {message}"
        slot = next((candidate for candidate, generation
                     in self._slot_generations.items()
                     if (name == item_group(candidate, generation)
                         or name in item_node_names(candidate, generation))), None)
        if slot is None:
            if any(name == group or name.startswith(f"{group}_")
                   for group in self._all_item_groups):
                return
            self._emit(BackendEvent("error", message=detail))
            return
        with self._frame_condition:
            self._slot_errors[slot] = detail
            self._frame_condition.notify_all()
        if slot == self._active_slot:
            self._emit(BackendEvent(
                "error", self._slot_items.get(slot), message=detail))

    def observe_item_frame(self, slot: int) -> None:
        with self._frame_condition:
            self._item_frame_sequences[slot] = (
                self._item_frame_sequences.get(slot, 0) + 1)
            self._frame_condition.notify_all()

    def observe_switched_frame(self, frame) -> None:
        try:
            if frame.width == 0:
                self.observe_eof()
        except (AttributeError, RuntimeError):
            return

    def observe_frame(self, metadata) -> None:
        position_ms = None
        try:
            if "frame_ts" in metadata:
                position_ms = int(metadata["frame_ts"])
        except (KeyError, TypeError, ValueError):
            pass
        with self._frame_condition:
            self._frame_sequence += 1
            self._frame_condition.notify_all()
        if position_ms is not None:
            self._emit(BackendEvent(
                "position", self._probe_item, position_ms=position_ms))

    def observe_eof(self) -> None:
        self._emit(BackendEvent("eof", self._probe_item))

    def item_groups(self) -> List[str]:
        return sorted(self._all_item_groups)


@dataclass
class PlaylistApplication:
    avp: object
    config: PlaylistConfig
    controller: PlaylistController
    backend: AsyncPlaylistBackend
    position_probe: object
    rtcp_feedback_listener: object
    _stopped: bool = False

    def start(self) -> None:
        self.backend.start()
        try:
            if not self.controller.play():
                raise RuntimeError("initial playlist element did not start")
            deadline = time.monotonic() + self.config.control_timeout
            while True:
                self.controller.poll(int(time.monotonic() * 1000))
                status = self.controller.status()
                if status.playing and status.output_alive:
                    self.avp.setReady()
                    return
                if status.error:
                    raise RuntimeError(status.error)
                if time.monotonic() >= deadline:
                    raise TimeoutError("playlist/Janus startup timed out")
                time.sleep(0.01)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._stopped:
            return
        self.avp.setExceptionCallback(None)
        self.backend.close()
        try:
            self.rtcp_feedback_listener.stop()
        except Exception:
            pass
        if self.position_probe.worker_running:
            self.position_probe.request_stop()
            deadline = time.monotonic() + self.config.control_timeout
            while self.position_probe.worker_running:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out stopping playlist position probe")
                time.sleep(0.01)
        self.position_probe.detach()
        for group in ("output", "switch"):
            try:
                self.avp.group(group).stopNodes()
            except Exception:
                pass
        for group in self.backend.item_groups():
            try:
                self.avp.group(group).stopNodes()
            except Exception:
                pass
        self.avp.shutdown()
        self.backend._set_output_alive(False)
        self._stopped = True


def build_playlist_application(config: PlaylistConfig, api=None) -> PlaylistApplication:
    api = api or load_avp_api()
    avp = api.AVPlumber()
    if config.log_file:
        Path(config.log_file).parent.mkdir(parents=True, exist_ok=True)
        avp.setLogFile(config.log_file)
    avp.executeCommandsFromString(
        f'hwaccel.init {{"name":"{GPU_DEVICE}","type":"cuda"}}')
    avp.edges.planCapacity("*", 1)

    backend = AsyncPlaylistBackend(avp, api, config)
    controller = PlaylistController(backend, config.clips, config.mode)
    avp.on_exception = backend.report_graph_exception

    first_index = next(
        index for index, clip in enumerate(config.clips) if not clip.disabled)
    first_clip = config.clips[first_index]
    for spec in plan_item_nodes(0, first_clip, config.fps):
        _add_spec(avp, api, spec)
    backend.register_built_item(first_clip.item_id, 0, first_clip)

    for spec in plan_switch_nodes(config.slot_capacity, config.fps):
        _add_spec(avp, api, spec)
    avp.getEdge("pl_switched").addWiretapCallback(
        backend.observe_switched_frame)

    class PositionProbe(api.PythonNode):
        def __init__(self, parameters):
            super().__init__(parameters)

        def process(self):
            frame = self._src.get()
            if frame is None or frame.width == 0:
                backend.observe_eof()
                return
            backend.observe_frame(frame.metadata)
            self._dst.enqueue(frame)

        @property
        def worker_running(self):
            return self._wrapper is not None and self._wrapper.isWorking

        def request_stop(self):
            self._wrapper.stop(False)

        def detach(self):
            if self.worker_running:
                raise RuntimeError("position probe is still running")
            self._src = self._dst = self._wrapper = self._avplumber = None

    probe = PositionProbe({
        "name": "pl_position",
        "src": "pl_realtime_out",
        "dst": "pl_observed",
        "data_type": "VideoFrame",
        "group": "switch",
    })
    avp.addNode(probe)

    bitrate = "4000k"
    _add_node(avp, api.ForceKeyFrame, JANUS_FORCE_KEYFRAME_NODE, "output",
              src="pl_observed", dst="janus_keyframes", interval_sec="1/1")
    _add_node(avp, api.EncVideo, "janus_encoder", "output",
              src="janus_keyframes", dst="janus_encoded", codec="h264_nvenc",
              hwaccel=GPU_DEVICE, options={
                  "b": bitrate, "maxrate": bitrate, "bufsize": bitrate,
                  "g": config.fps, "bf": 0, "preset": "p6",
                  "profile": "baseline", "tune": "ull", "rc": "cbr",
                  "rc-lookahead": 0, "zerolatency": 1, "delay": 0,
                  "forced-idr": 1, "no-scenecut": 1, "strict_gop": 1,
                  "aud": 1, "spatial-aq": 1, "temporal-aq": 0,
              })
    _add_node(avp, api.Bsf, "janus_headers", "output",
              src="janus_encoded", dst="janus_headers",
              bsf="dump_extra=freq=keyframe")
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
            JANUS_FORCE_KEYFRAME_COMMAND),
    )
    backend.bind_listener(listener)
    return PlaylistApplication(
        avp, config, controller, backend, probe, listener)
