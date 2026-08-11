"""Live backend for the playlist player.  Builds N replay-style worker chains ->
source_switcher -> one realtime reclock -> the verbatim replay Janus H.264 RTP
output group, and hands the controller a command sink wired to AVPlumber.

Everything here is glue over EXISTING AVPlumber nodes -- there are NO C++
changes.  The worker chain and Janus output group are copied from
demos/replay/replay.py so the two demos stay bug-for-bug comparable.  The pure
control logic lives in playlist.py and is fully unit-tested without a GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List

from playlist import (GPU_DEVICE, Clip, PlaylistController, PlaylistMode,
                      plan_output_nodes, plan_worker_nodes, worker_group)

JANUS_FORCE_KEYFRAME_NODE = "janus_force_keyframe"
JANUS_FORCE_KEYFRAME_COMMAND = (
    f"node.object.set {JANUS_FORCE_KEYFRAME_NODE} trigger true")


# --------------------------------------------------------------------------- #
# Config (mirrors demos/replay JanusVideoConfig / PlayerConfig)
# --------------------------------------------------------------------------- #
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


@dataclass(frozen=True)
class PlaylistConfig:
    clips: List[Clip]
    mode: PlaylistMode = PlaylistMode.LOOP_ALL
    worker_count: int = 2
    fps: int = 30
    width: int = 1920
    height: int = 1080
    janus: JanusVideoConfig = field(default_factory=JanusVideoConfig)
    control_timeout: float = 10.0


# --------------------------------------------------------------------------- #
# AVP API loader (same node set as demos/replay + PositionProbe)
# --------------------------------------------------------------------------- #
def load_avp_api():
    from pyplumber import AVPlumber
    from pyplumber.node import (Bsf, DecVideo, Demux, EncVideo, ForceFPS,
                                ForceKeyFrame, InputRec, Mux, Output, Pause,
                                PythonNode, RealtimeVideoFrame, RescaleVideo,
                                SourceSwitcher, SpeedVideo)
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
            self._src = self._dst = self._wrapper = self._avplumber = None

    # map the plan's node-type strings to the pyplumber node classes
    by_type = {
        "input_rec": InputRec, "demux": Demux, "dec_video": DecVideo,
        "speed_video": SpeedVideo, "rescale_video": RescaleVideo,
        "force_fps": ForceFPS, "pause": Pause, "source_switcher": SourceSwitcher,
        "realtime<av::VideoFrame>": RealtimeVideoFrame,
    }
    return SimpleNamespace(
        AVPlumber=AVPlumber, Bsf=Bsf, EncVideo=EncVideo, ForceKeyFrame=ForceKeyFrame,
        Mux=Mux, Output=Output, PositionProbe=PositionProbe,
        RtcpFeedbackListener=RtcpFeedbackListener, by_type=by_type)


def _add_node(avp, node_type, name, group, **params):
    node = node_type({"name": name, "group": group, **params})
    avp.addNode(node)
    return node


def _add_spec(avp, api, spec):
    node_type = api.by_type[spec.type]
    return _add_node(avp, node_type, spec.name, spec.group, **spec.params)


def _init_cuda(avp) -> None:
    avp.executeCommandsFromString(
        f'hwaccel.init {{ "name": "{GPU_DEVICE}", "type": "cuda" }}')


def _rtp_url(cfg: JanusVideoConfig) -> str:
    return (f"rtp://{cfg.host}:{cfg.video_port}?pkt_size=1200"
            f"&rtcp_port={cfg.video_port + 1}")


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
@dataclass
class PlaylistApplication:
    avp: object
    config: PlaylistConfig
    controller: PlaylistController
    position_probe: object
    rtcp_feedback_listener: object
    _stopped: bool = False

    def _wait_for(self, predicate, description: str) -> None:
        deadline = time.monotonic() + self.config.control_timeout
        while not predicate():
            if self.controller.error:
                raise RuntimeError(self.controller.error)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {description}")
            time.sleep(0.01)

    def start(self) -> None:
        try:
            self.avp.group(worker_group(0)).startNodes()
            self.avp.group("switch").startNodes()
            # worker 0 was added frozen (paused=True); release it so it produces
            # frames, otherwise the readiness wait below never completes.
            self.controller.play()
            self._wait_for(lambda: self.controller.ready, "the first source frame")
            # arm the next clip on the idle worker (best-effort, never blocks)
            self.controller._rebuild_idle_for_next()
            self.avp.group("output").startNodes()
            self._wait_for(lambda: self.avp.node("janus_rtp_output").isWorking,
                           "the Janus RTP output")
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
            deadline = time.monotonic() + self.config.control_timeout
            while self.position_probe.worker_running:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out stopping the position probe")
                time.sleep(0.01)
        self.position_probe.detach()
        self.avp.group("output").stopNodes()
        self.avp.group("switch").stopNodes()
        for i in range(self.config.worker_count):
            self.avp.group(worker_group(i)).stopNodes()
        self.avp.shutdown()
        self._stopped = True


def build_playlist_application(config: PlaylistConfig, api=None) -> PlaylistApplication:
    api = api or load_avp_api()
    avp = api.AVPlumber()
    _init_cuda(avp)
    avp.edges.planCapacity("*", 1)

    def command(value: str) -> None:
        avp.executeCommandsFromString(value)

    controller = PlaylistController(
        command, config.clips, mode=config.mode, worker_count=config.worker_count,
        fps=config.fps, width=config.width, height=config.height)
    avp.on_exception = lambda name, node_type, message: controller.set_error(
        f"{name} ({node_type}): {message}")

    fps = config.fps

    # worker 0 (the initially-active clip) -- the rest are built on demand by
    # the controller's preload path.  worker 0 must NOT start paused, so we
    # release it immediately after building.
    first_clip = config.clips[controller.current_index]
    for spec in plan_worker_nodes(0, first_clip, fps, config.width, config.height):
        _add_spec(avp, api, spec)

    # switcher + single shared realtime reclock
    for spec in plan_output_nodes(config.worker_count, fps, active=0):
        _add_spec(avp, api, spec)

    # position probe on the shared output edge (feeds observe_metadata)
    probe = api.PositionProbe({
        "name": "pl_position", "src": "pl_realtime_out", "dst": "pl_observed",
        "data_type": "VideoFrame", "group": "switch"}, controller)
    avp.addNode(probe)

    # verbatim replay Janus output group, downstream of pl_observed
    bitrate = "4000k"
    _add_node(avp, api.ForceKeyFrame, JANUS_FORCE_KEYFRAME_NODE, "output",
              src="pl_observed", dst="janus_keyframes", interval_sec="1/1")
    _add_node(avp, api.EncVideo, "janus_encoder", "output",
              src="janus_keyframes", dst="janus_encoded", codec="h264_nvenc",
              hwaccel=GPU_DEVICE, options={
                  "b": bitrate, "maxrate": bitrate, "bufsize": bitrate, "g": fps,
                  "bf": 0, "preset": "p6", "profile": "baseline", "tune": "ull",
                  "rc": "cbr", "rc-lookahead": 0, "zerolatency": 1, "delay": 0,
                  "forced-idr": 1, "no-scenecut": 1, "strict_gop": 1, "aud": 1,
                  "spatial-aq": 1, "temporal-aq": 0})
    _add_node(avp, api.Bsf, "janus_headers", "output",
              src="janus_encoded", dst="janus_headers", bsf="dump_extra=freq=keyframe")
    _add_node(avp, api.Mux, "janus_mux", "output",
              src=["janus_headers"], dst="janus_muxed", ts_sort_wait=0)
    _add_node(avp, api.Output, "janus_rtp_output", "output",
              src="janus_muxed", url=_rtp_url(config.janus), format="rtp",
              options={"payload_type": config.janus.payload_type,
                       "rtpflags": "skip_rtcp", "ssrc": config.janus.ssrc})

    listener = api.RtcpFeedbackListener(
        bind_host=config.janus.rtcp_bind, bind_port=config.janus.rtcp_port,
        janus_host=config.janus.host, janus_rtcp_port=config.janus.video_port + 1,
        media_ssrc=config.janus.ssrc,
        on_keyframe_request=lambda _r: avp.executeCommandsFromString(
            JANUS_FORCE_KEYFRAME_COMMAND))

    return PlaylistApplication(avp, config, controller, probe, listener)
