#!/usr/bin/env python3
"""Mixer-backed playlist engine.

Every playlist element owns one of sixteen fixed mixer sources.  Its decode
chain (``avpmixer.inputs.build_input`` with a pause node and a realtime sync
team, the replay demo's wiring) lives in group ``pl_item_<slot>`` and feeds
scene ``item_<slot>``, a fullscreen layout on the native two-slot mixer.

A transition is *armed* natively shortly before its start: ``mixer.cut`` /
``fade`` / ``wipe`` with a wallclock ``start_pts_ms``, then ``resume <team>
at`` so the incoming chain runs just ahead of the cut.  The engine confirms
the switch from ``mixer.status`` over the local control port.  Python never
sits on the frame path.

Elements that leave air are *parked*: paused and seeked back to cue-in with
their decoder resident, so the next take is warm.  Every chain loops between
its cue points and never reaches EOF; the schedule alone ends an element.
"""

from __future__ import annotations

import json
import queue
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from avpmixer.inputs import build_input
from avpmixer.janus import JanusVideoConfig, build_janus_output
from playlist import SLOT_CAPACITY, BackendEvent, Clip, Transition, now_ms

MIXER = "mixer"
OUTPUT_GROUP = "output"
HWACCEL = "@gpu"


def hms(ms: int) -> str:
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def slot_group(slot: int) -> str:
    return f"pl_item_{slot}"


def slot_tag(slot: int) -> str:
    return f"pl{slot}"


def slot_scene(slot: int) -> str:
    return f"item_{slot}"


def slot_pause_team(slot: int) -> str:
    return f"pl_item_{slot}_pause"


def slot_sync_team(slot: int) -> str:
    return f"pl_item_{slot}_sync"


def chain_nodes(slot: int) -> List[str]:
    tag = slot_tag(slot)
    return [f"{kind}_{tag}" for kind in ("input", "demux", "decode", "speed", "pause", "realtime", "fps")]


def chain_edge(slot: int) -> str:
    return f"input_{slot_tag(slot)}_fps"


@dataclass(frozen=True)
class PlaylistConfig:
    fps: int = 30
    width: int = 1920
    height: int = 1080
    janus: JanusVideoConfig = field(default_factory=JanusVideoConfig)
    control_port: int = 7778
    control_timeout: float = 10.0
    log_file: str = "playlist-demo.log"
    preroll_ms: int = 50            # resume the incoming chain this early
    switch_margin_ms: int = 100     # mixer's minimum lead for a scheduled cut
    arm_lead_ms: int = 600          # issue the native transition this early
    wipe_file: Optional[str] = None
    record: Optional[str] = None    # also write the program to this MP4/TS (verification)

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("output format must be positive")
        if self.control_timeout <= 0:
            raise ValueError("control timeout must be positive")
        if min(self.preroll_ms, self.switch_margin_ms) < 0 or self.arm_lead_ms <= self.switch_margin_ms:
            raise ValueError("preroll >= 0, switch margin >= 0 and arm lead > switch margin required")


def load_avp_api():
    from pyplumber import AVPlumber
    from pyplumber.node import (AssumeVideoFormat, Bsf, DecVideo, Demux, EncVideo, ForceFPS,
                                ForceKeyFrame, InputRec, Mux, Output, Pause, Realtime, SpeedVideo,
                                Split)
    from pyplumber.rtcp_feedback import RtcpFeedbackListener
    from avpmixer import MixerGraphBuilder
    from types import SimpleNamespace
    return SimpleNamespace(
        AVPlumber=AVPlumber, MixerGraphBuilder=MixerGraphBuilder,
        AssumeVideoFormat=AssumeVideoFormat, Bsf=Bsf, DecVideo=DecVideo, Demux=Demux,
        EncVideo=EncVideo, ForceFPS=ForceFPS, ForceKeyFrame=ForceKeyFrame, InputRec=InputRec,
        Mux=Mux, Output=Output, Pause=Pause, Realtime=Realtime, SpeedVideo=SpeedVideo, Split=Split,
        RtcpFeedbackListener=RtcpFeedbackListener,
    )


def probe_length_ms(url: str) -> Optional[int]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", url],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return int(float(out) * 1000)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _fingerprint(clip: Clip) -> Tuple:
    return (clip.url, clip.play_from_ms, clip.play_to_ms, clip.speed)


class ControlClient:
    """Blocking client for AVPlumber's line protocol on the local control port."""

    def __init__(self, port: int, timeout: float):
        self.port, self.timeout, self._sock, self._file = port, timeout, None, None

    def command(self, line: str) -> str:
        for attempt in (0, 1):
            try:
                return self._exchange(line)
            except OSError:
                self._close()
                if attempt:
                    raise
        raise AssertionError("unreachable")

    def _exchange(self, line: str) -> str:
        if self._sock is None:
            self._sock = socket.create_connection(("127.0.0.1", self.port), timeout=self.timeout)
            self._file = self._sock.makefile("rb")
            if not self._file.readline().startswith(b"100"):
                raise OSError("unexpected control greeting")
        self._sock.sendall(line.encode() + b"\n")
        head = self._file.readline().decode().rstrip("\n")
        code = int(head.split(" ", 1)[0])
        content = ""
        if code == 201:
            while True:
                row = self._file.readline()
                if row in (b"", b"\n", b"\r\n"):
                    break
                content += row.decode()
        if not 200 <= code < 300:
            raise RuntimeError(f"{line.split()[0]}: {head}")
        return content

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = self._file = None


@dataclass(frozen=True)
class _Task:
    kind: str
    item_id: Optional[str] = None
    clip: Optional[Clip] = None
    request_id: Optional[int] = None
    at_ms: Optional[int] = None
    transition: Transition = Transition.CUT
    transition_ms: int = 0
    message: str = ""


@dataclass
class _Armed:
    request_id: int
    item_id: str
    slot: int
    start: int                       # incoming picture visible from here (fade: fade start)
    end: int                         # switch complete
    transition: Transition
    transition_ms: int
    issued: bool = False

    @property
    def duration_ms(self) -> int:
        return 0 if self.transition is Transition.CUT else self.transition_ms


class PlaylistEngine:
    """Backend for ``PlaylistController``; public methods never block on AVPlumber."""

    def __init__(self, avp, api, config: PlaylistConfig):
        self.avp, self.api, self.config = avp, api, config
        self.mixer = None
        self.listener = None
        self._tasks: "queue.Queue[_Task]" = queue.Queue()
        self._events: "queue.SimpleQueue[BackendEvent]" = queue.SimpleQueue()
        self._thread: Optional[threading.Thread] = None
        self._closing = threading.Event()
        self._lock = threading.Lock()
        self._control = ControlClient(config.control_port, config.control_timeout) if config.control_port else None
        self._slot_of: Dict[str, int] = {}
        self._bound: Dict[int, Tuple] = {}         # slot -> fingerprint
        self._orphaned: set = set()                # program slots whose element was removed
        self._lengths: Dict[str, Optional[int]] = {}
        self._pgm_slot: Optional[int] = None
        self._armed: Optional[_Armed] = None
        self._output_alive = False

    # ---- graph -----------------------------------------------------------
    def build(self, first_clip: Clip) -> None:
        avp, api, cfg = self.avp, self.api, self.config
        if cfg.log_file:
            Path(cfg.log_file).parent.mkdir(parents=True, exist_ok=True)
            avp.setLogFile(cfg.log_file)
        if cfg.control_port:
            avp.enableControlServer(cfg.control_port)
        avp.executeCommandsFromString(f'hwaccel.init {{"name":"{HWACCEL}","type":"cuda"}}')
        avp.edges.planCapacity("*", 4)
        self.mixer = api.MixerGraphBuilder(
            avp, name=MIXER, canvas=(cfg.width, cfg.height), fps=(cfg.fps, 1), hwaccel=HWACCEL,
            enable_wipe=True, switch_margin_ms=cfg.switch_margin_ms,
            defer_initial_routes=True, defer_output=True)
        full = {"dst_x": 0, "dst_y": 0, "dst_w": cfg.width, "dst_h": cfg.height, "fit": "contain"}
        for slot in range(SLOT_CAPACITY):
            self.mixer.add_source(f"source_{slot}", pre_otm_edge=chain_edge(slot),
                                  input_group=slot_group(slot), default_graph="")
            self.mixer.add_scene(slot_scene(slot), {f"source_{slot}": full})
        self.mixer.set_initial_scene(slot_scene(0), slot="A")
        self._add_chain(0, first_clip)
        self._pgm_slot = 0
        program = self.mixer.build()
        if cfg.record:
            avp.addNode(api.Split({"name": "program_split", "src": program,
                                   "dst": ["program_janus", "program_record"], "group": OUTPUT_GROUP}))
            self._add_recorder("program_record", cfg.record)
            program = "program_janus"
        self.listener = build_janus_output(
            avp, api, program, cfg.janus, fps=cfg.fps, width=cfg.width, height=cfg.height,
            hwaccel=HWACCEL, group=OUTPUT_GROUP)
        avp.on_exception = self._graph_exception

    def _add_recorder(self, src: str, path: str) -> None:
        """Near-lossless NVENC record of the program for tests/verify_recording.py."""
        avp, api, cfg = self.avp, self.api, self.config
        fmt = "mpegts" if path.endswith(".ts") else "mp4"
        avp.addNode(api.ForceFPS({"name": "record_fps", "src": src, "dst": "record_fps",
                                  "fps": f"{cfg.fps}/1", "group": OUTPUT_GROUP}))
        avp.addNode(api.AssumeVideoFormat({
            "name": "record_format", "src": "record_fps", "dst": "record_video", "width": cfg.width,
            "height": cfg.height, "pixel_format": "cuda", "real_pixel_format": "nv12", "group": OUTPUT_GROUP}))
        avp.addNode(api.EncVideo({
            "name": "record_encoder", "src": "record_video", "dst": "record_encoded", "codec": "h264_nvenc",
            "hwaccel": HWACCEL, "options": {"preset": "p4", "rc": "constqp", "qp": 12, "bf": 0,
                                            "g": cfg.fps}, "group": OUTPUT_GROUP}))
        avp.addNode(api.Mux({"name": "record_mux", "src": ["record_encoded"], "dst": "record_muxed",
                             "ts_sort_wait": 0, "group": OUTPUT_GROUP}))
        avp.addNode(api.Output({"name": "record_output", "src": "record_muxed", "url": path,
                                "format": fmt, "group": OUTPUT_GROUP}))

    def _add_chain(self, slot: int, clip: Clip) -> None:
        fps = self.config.fps
        params = {"pause_team": slot_pause_team(slot), "start_ts": hms(clip.play_from_ms),
                  "timeout": -1, "preseek": 0}
        if clip.play_to_ms is not None:
            params["stop_ts"] = hms(clip.play_to_ms)
        build_input(self.avp, self.api, slot_tag(slot), clip.url, group=slot_group(slot), fps=fps,
                    hwaccel=HWACCEL, loop=True, input_params=params, auto_restart=None,
                    speed_team=f"pl_item_{slot}_speed", speed=clip.speed,
                    pause_team=slot_pause_team(slot), sync_team=slot_sync_team(slot),
                    realtime_params={"tick_period": f"1/{fps}", "negative_time_tolerance": 1 / fps,
                                     "negative_time_discard": 1 / fps, "discontinuity_threshold": 3})
        self._slot_of[clip.item_id] = slot
        self._bound[slot] = _fingerprint(clip)
        if clip.url not in self._lengths:
            self._lengths[clip.url] = probe_length_ms(clip.url)

    def _remove_chain(self, slot: int) -> None:
        self._exec(f"group.stop {slot_group(slot)}")
        self._wait(lambda: not any(self.avp.node(n).isWorking for n in chain_nodes(slot)),
                   f"slot {slot} did not stop")
        self._exec("\n".join(f"node.delete {n}" for n in chain_nodes(slot)))
        self._bound.pop(slot, None)
        self._orphaned.discard(slot)
        for item, s in list(self._slot_of.items()):
            if s == slot:
                del self._slot_of[item]

    def _wait(self, predicate, what: str) -> None:
        deadline = time.monotonic() + self.config.control_timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError(what)
            time.sleep(0.02)

    def _wait_edge(self, edge: str, what: str, baseline: int = 0) -> None:
        self._wait(lambda: self.avp.getEdge(edge).enqueued_total > baseline, what)

    def start(self) -> None:
        """Mixer-style preheat.  The first element runs meanwhile; the controller's
        first ``cue`` re-parks it on cue-in and resumes it, so air starts there."""
        avp, m = self.avp, self.mixer
        avp.group(slot_group(0)).startNodes()
        self._wait_edge(chain_edge(0), "first element produced no frame")
        m.initialize_routes()
        m.start_groups()
        for node in (f"{MIXER}_comp_a", f"{MIXER}_comp_b", f"{MIXER}_otm_scene_a",
                     f"{MIXER}_otm_scene_b", f"{MIXER}_out_sel_transition"):
            self._wait(lambda n=node: avp.node(n).isWorking, f"{node} did not start")
        m.begin_transition_preheat()
        try:
            self._wait_edge(f"{MIXER}_trans_out", "transition warm-up produced no frame")
        finally:
            m.finish_transition_preheat()
        m.start_output()
        avp.group(OUTPUT_GROUP).startNodes()
        self._wait_edge(f"{MIXER}_final_out", "program output produced no frame")
        self.listener.start()
        self._set_alive(True)
        self._thread = threading.Thread(target=self._run, name="playlist-engine", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closing.set()
        self._tasks.put(_Task("shutdown"))
        if self._thread is not None:
            self._thread.join(self.config.control_timeout)
        try:
            self.listener.stop()
        except Exception:  # noqa: BLE001
            pass
        self.avp.shutdown()
        self._set_alive(False)

    # ---- backend protocol (non-blocking) -------------------------------------
    def cue(self, request_id: int, clip: Clip, at_ms: Optional[int], transition: Transition,
            transition_ms: int) -> None:
        self._tasks.put(_Task("cue", clip.item_id, clip, request_id, at_ms, transition, transition_ms))

    def pause(self, item_id: str) -> None:
        self._tasks.put(_Task("pause", item_id))

    def resume(self, item_id: str) -> None:
        self._tasks.put(_Task("resume", item_id))

    def park(self, item_id: str) -> None:
        self._tasks.put(_Task("park", item_id))

    def remove(self, item_id: str) -> None:
        self._tasks.put(_Task("remove", item_id))

    def media_length_ms(self, clip: Clip) -> Optional[int]:
        if clip.url not in self._lengths:
            self._lengths[clip.url] = probe_length_ms(clip.url)
        return self._lengths[clip.url]

    def poll_events(self) -> List[BackendEvent]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def output_alive(self) -> bool:
        with self._lock:
            return self._output_alive

    # ---- worker ----------------------------------------------------------------
    def _exec(self, commands: str) -> None:
        self.avp.executeCommandsFromString(commands)

    def _emit(self, event: BackendEvent) -> None:
        self._events.put(event)

    def _set_alive(self, value: bool) -> None:
        with self._lock:
            changed, self._output_alive = self._output_alive != value, value
        if changed:
            self._emit(BackendEvent("health", value=value))

    def _run(self) -> None:
        while not self._closing.is_set():
            try:
                task = self._tasks.get(timeout=0.02)
            except queue.Empty:
                self._tick()
                continue
            if task.kind == "shutdown":
                return
            try:
                self._handle(task)
            except Exception as exc:  # noqa: BLE001
                if task.kind == "cue":
                    self._fail(task.request_id, task.item_id, str(exc))
                else:
                    self._emit(BackendEvent("error", task.item_id, message=str(exc)))

    def _tick(self) -> None:
        armed, now = self._armed, now_ms()
        if armed is not None:
            try:
                if not armed.issued and now >= armed.start - self.config.arm_lead_ms:
                    self._issue(armed)
                elif armed.issued and now >= armed.start:
                    self._confirm(armed, now)
            except Exception as exc:  # noqa: BLE001
                self._fail(armed.request_id, armed.item_id, str(exc))
        for slot in list(self._orphaned):
            if slot != self._pgm_slot:
                self._remove_chain(slot)
        try:
            self._set_alive(bool(self.avp.node("janus_rtp_output").isWorking))
        except Exception as exc:  # noqa: BLE001
            self._set_alive(False)
            self._emit(BackendEvent("error", message=str(exc)))

    def _handle(self, task: _Task) -> None:
        if task.kind == "cue":
            self._cue(task)
            return
        if task.kind == "graph_error":
            self._graph_error(task.item_id, task.message)
            return
        slot = self._slot_of.get(task.item_id)
        if slot is None:
            return
        team = slot_pause_team(slot)
        if task.kind == "pause":
            self._exec(f"pause {team} now")
        elif task.kind == "resume":
            self._exec(f"resume {team}")
        elif task.kind == "park":
            self._disarm(task.item_id)
            self._park(slot)
        elif task.kind == "remove":
            self._disarm(task.item_id)
            if slot == self._pgm_slot:
                self._orphaned.add(slot)          # freed once another element is on air
            else:
                self._remove_chain(slot)

    def _park(self, slot: int) -> None:
        """Pause, then seek the chain back to cue-in through its realtime sync team."""
        cue_in = self._bound[slot][1]
        self._exec(f"pause {slot_pause_team(slot)} now\nseek {slot_sync_team(slot)} now {hms(cue_in)}")

    def _disarm(self, item_id: str) -> None:
        armed = self._armed
        if armed is None or armed.item_id != item_id:
            return
        self._armed = None
        if armed.issued:
            self._exec(f'mixer.interrupt {{"mixer":"{MIXER}"}}')

    def _fail(self, request_id: Optional[int], item_id: Optional[str], message: str) -> None:
        if self._armed is not None and self._armed.request_id == request_id:
            armed, self._armed = self._armed, None
            try:
                if armed.issued:
                    self._exec(f'mixer.interrupt {{"mixer":"{MIXER}"}}')
                if armed.slot != self._pgm_slot:
                    self._park(armed.slot)
            except Exception as exc:  # noqa: BLE001
                message = f"{message}; cleanup: {exc}"
        self._emit(BackendEvent("failed", item_id, request_id, message=message))

    def _bind(self, clip: Clip) -> int:
        """Ensure a running, parked chain for the clip; return its slot."""
        slot = self._slot_of.get(clip.item_id)
        if slot is not None and self._bound.get(slot) == _fingerprint(clip):
            return slot
        if slot is None:
            free = [s for s in range(SLOT_CAPACITY) if s not in self._bound]
            if not free:
                raise RuntimeError(f"all {SLOT_CAPACITY} mixer sources are in use")
            slot = free[0]
        else:
            self._remove_chain(slot)
        self._add_chain(slot, clip)
        # Prime the decoder: run until the first frame reaches the fps edge,
        # then park exactly on cue-in.  A chain that starts paused decodes nothing.
        baseline = self.avp.getEdge(chain_edge(slot)).enqueued_total
        self.avp.group(slot_group(slot)).startNodes()
        self._wait_edge(chain_edge(slot), f"{clip.name}: no frame decoded", baseline)
        self._park(slot)
        return slot

    def _cue(self, task: _Task) -> None:
        clip = task.clip
        previous = self._armed
        if previous is not None:
            self._disarm(previous.item_id)
            if previous.item_id != clip.item_id:
                self._park(previous.slot)
        slot = self._bind(clip)
        now = now_ms()
        if slot == self._pgm_slot:
            # Already the program (startup, edit of the active element, Stop then
            # Play): restart it from cue-in.  The seek is the only discontinuity.
            self._park(slot)
            self._exec(f"resume {slot_pause_team(slot)}")
            self._emit(BackendEvent("on_air", clip.item_id, task.request_id, at_ms=now))
            return
        if task.transition is Transition.WIPE and not self.config.wipe_file:
            raise RuntimeError("no --wipe-file configured; choose Cut or Fade")
        duration = task.transition_ms if task.transition is not Transition.CUT else 0
        end = task.at_ms if task.at_ms is not None else now + self.config.switch_margin_ms + duration
        self._armed = _Armed(task.request_id, clip.item_id, slot, end - duration, end,
                             task.transition, task.transition_ms)
        if task.at_ms is None or self._armed.start - now <= self.config.arm_lead_ms:
            self._issue(self._armed)

    def _issue(self, armed: _Armed) -> None:
        """Arm the native transition, then schedule the incoming chain's resume."""
        cfg, now = self.config, now_ms()
        scene, seconds = slot_scene(armed.slot), armed.transition_ms / 1000
        immediate = armed.start - now < cfg.switch_margin_ms + 50
        start_pts = -1 if immediate else armed.start           # -1: the mixer picks now + margin
        if armed.transition is Transition.CUT:
            self.mixer.cut(scene, start_pts_ms=start_pts)
        elif armed.transition is Transition.FADE:
            self.mixer.fade(scene, duration_sec=seconds, start_pts_ms=start_pts)
        else:
            self.mixer.wipe(scene, cfg.wipe_file, duration_sec=seconds, start_pts_ms=start_pts)
        if immediate:
            armed.start = now + cfg.switch_margin_ms
            armed.end = armed.start + armed.duration_ms
        self._exec(f"resume {slot_pause_team(armed.slot)} at {max(now, armed.start - cfg.preroll_ms)}")
        armed.issued = True

    def _confirm(self, armed: _Armed, now: int) -> None:
        """Report on_air once the mixer shows the scene as program and idle."""
        if self._control is not None:
            status = json.loads(self._control.command(f"mixer.status {MIXER}") or "{}")
            done = status.get("pgm_scene") == slot_scene(armed.slot) and status.get("transition") == "idle"
        else:
            done = now >= armed.end + 1000 // self.config.fps
        if done:
            self._armed = None
            self._pgm_slot = armed.slot
            self._emit(BackendEvent("on_air", armed.item_id, armed.request_id, at_ms=armed.start))
        elif now > armed.end + 3000:
            raise TimeoutError("mixer did not switch to the armed element")

    # ---- graph errors (called on a C++ thread; handed to the worker) -----------------
    def _graph_exception(self, name: str, node_type: str, message: str) -> None:
        self._tasks.put(_Task("graph_error", name, message=f"{name} ({node_type}): {message}"))

    def _graph_error(self, name: str, detail: str) -> None:
        slot = next((s for s in self._bound if name in chain_nodes(s) or name == slot_group(s)), None)
        item = next((i for i, s in self._slot_of.items() if s == slot), None)
        if self._armed is not None and item == self._armed.item_id:
            self._fail(self._armed.request_id, item, detail)
        else:
            self._emit(BackendEvent("error", item, message=detail))
