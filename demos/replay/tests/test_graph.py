from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import struct
import threading

import pytest

from replay import (
    JanusVideoConfig,
    PlayerConfig,
    ReplaySlotConfig,
    TranscodeConfig,
    build_player_application,
    build_transcode_application,
)


class FakeNode:
    node_type = "unknown"

    def __init__(self, parameters):
        self.parameters = {"type": self.node_type, **parameters}
        self.isWorking = False

    def stopAndWait(self):
        if not self.isWorking:
            raise RuntimeError("node no longer has a stopping interface")
        self.avp.group_calls.append(f"stop:{self.parameters['name']}")
        self.isWorking = False


def node_type(name):
    return type(name, (FakeNode,), {"node_type": name})


class FakeGroup:
    def __init__(self, avp, name):
        self.avp = avp
        self.name = name

    def startNodes(self):
        self.avp.group_calls.append("start")
        if self.avp.on_group_start:
            self.avp.on_group_start(self.name)
        for node in self.avp.nodes:
            if node.parameters["group"] == self.name:
                node.isWorking = True

    def stopNodes(self):
        self.avp.group_calls.append(f"stop-group:{self.name}")
        if not self.avp.on_group_stop or not self.avp.on_group_stop(self.name):
            for node in self.sortedNodes:
                node.isWorking = False

    @property
    def sortedNodes(self):
        return [node for node in self.avp.nodes if node.parameters["group"] == self.name]


class FakeEdges:
    def __init__(self):
        self.plans = []

    def planCapacity(self, pattern, capacity):
        self.plans.append((pattern, capacity))


class FakeAvp:
    def __init__(self):
        self.nodes = []
        self.commands = []
        self.group_calls = []
        self.edges = FakeEdges()
        self.ready = False
        self.on_group_start = None
        self.on_group_stop = None
        self.shutdown_while_working = False
        self.exception_callback_active = True

    def addNode(self, node):
        node.avp = self
        self.nodes.append(node)

    def executeCommandsFromString(self, command):
        self.commands.append(command)

    def group(self, name):
        return FakeGroup(self, name)

    def node(self, name):
        return next(node for node in self.nodes if node.parameters["name"] == name)

    def setReady(self):
        self.ready = True

    def setExceptionCallback(self, callback):
        self.exception_callback_active = callback is not None

    def shutdown(self):
        self.shutdown_while_working = any(node.isWorking for node in self.nodes)
        self.group_calls.append("shutdown")


class FakePositionProbe(FakeNode):
    node_type = "python_node_siso"

    def __init__(self, parameters, controller):
        super().__init__(parameters)
        self.controller = controller
        self.attached = True

    def request_stop(self):
        self.avp.group_calls.append("probe-stop")
        self.isWorking = False

    @property
    def worker_running(self):
        return self.isWorking

    def detach(self):
        assert not self.worker_running
        self.avp.group_calls.append("probe-detach")
        self.attached = False


class FakeRtcpFeedbackListener:
    def __init__(self, **parameters):
        self.parameters = parameters
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


def fake_api():
    classes = {
        name: node_type(node)
        for name, node in {
            "Input": "input",
            "Demux": "demux",
            "DecVideo": "dec_video",
            "ForceFPS": "force_fps",
            "ForceKeyFrame": "force_keyframe",
            "EncVideo": "enc_video",
            "Mux": "mux",
            "Output": "output",
            "InputRec": "input_rec",
            "SpeedVideo": "speed_video",
            "Pause": "pause",
            "RealtimeVideoFrame": "realtime<av::VideoFrame>",
            "Bsf": "bsf",
        }.items()
    }
    return SimpleNamespace(
        AVPlumber=FakeAvp,
        PositionProbe=FakePositionProbe,
        RtcpFeedbackListener=FakeRtcpFeedbackListener,
        **classes,
    )


def replay_file(tmp_path, fps=25):
    recording = tmp_path / "clip.ts"
    recording.write_bytes(b"mpegts")
    timestamps = tuple(1_260 + index * (1000 // fps) for index in range(101))
    Path(f"{recording}+seek").write_bytes(
        b"".join(struct.pack("=qQ", ts, 188 * index) for index, ts in enumerate(timestamps))
    )
    Path(f"{recording}+history").write_bytes(
        struct.pack("=qqqq", 0, 0, timestamps[0] - 1_786_363_200_000, 0)
    )
    return recording


def test_transcode_graph_is_video_only_cuda_all_intra(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"vod")
    config = TranscodeConfig(
        source=source,
        output=tmp_path / "replay.ts",
        fps=30,
        wallclock_start=datetime(2026, 8, 10, tzinfo=timezone.utc),
    )

    application = build_transcode_application(config, api=fake_api())
    nodes = {node.parameters["name"]: node.parameters for node in application.avp.nodes}
    serialized = repr(nodes).lower()

    assert nodes["replay_input"]["eof_mode"] == "drain"
    assert nodes["replay_demux"]["routing"] == {"?v:0": "transcode_video_packets"}
    assert nodes["replay_decode"]["src"] == "transcode_video_packets"
    assert nodes["replay_decode"]["pixel_format"] == "cuda"
    assert nodes["replay_decode"]["hwaccel"] == "replay_gpu"
    assert nodes["replay_decode"]["codec_map"] == {
        "h264": "h264_cuvid",
        "hevc": "hevc_cuvid",
    }
    assert nodes["replay_fps"]["src"] == "transcode_decoded"
    assert nodes["replay_fps"]["fps"] == "30/1"
    assert nodes["replay_keyframes"]["interval_sec"] == "1/30"
    assert nodes["replay_encoder"]["codec"] == "h264_nvenc"
    assert nodes["replay_encoder"]["options"] == {
        "g": 1,
        "bf": 0,
        "profile": "baseline",
        "rc": "vbr",
        "cq": 17,
        "b": 0,
        "tune": "ull",
        "rc-lookahead": 0,
        "zerolatency": 1,
        "delay": 0,
    }
    assert nodes["replay_mux"]["src"] == ["transcode_encoded"]
    assert nodes["replay_mux"]["ts_sort_wait"] == 0
    assert nodes["replay_output"]["format"] == "mpegts"
    assert nodes["replay_output"]["seek_table"] == f"{config.output}+seek"
    assert nodes["replay_output"]["seek_table_text"] == f"{config.output}+txt"
    assert "audio" not in serialized
    assert "realtime" not in serialized
    assert "hwdownload" not in serialized
    assert "hwupload" not in serialized
    assert application.avp.commands == [
        'hwaccel.init { "name": "replay_gpu", "type": "cuda" }'
    ]


def test_transcode_config_rejects_missing_input_and_invalid_fps(tmp_path):
    common = {
        "source": tmp_path / "missing.mp4",
        "output": tmp_path / "replay.ts",
        "wallclock_start": datetime.now(timezone.utc),
    }
    try:
        TranscodeConfig(fps=30, **common)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing input accepted")

    source = tmp_path / "source.mp4"
    source.write_bytes(b"vod")
    try:
        TranscodeConfig(source, tmp_path / "replay.ts", 0, common["wallclock_start"])
    except ValueError:
        pass
    else:
        raise AssertionError("invalid fps accepted")


def test_transcode_run_shuts_down_graph_before_returning(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"vod")
    application = build_transcode_application(
        TranscodeConfig(
            source=source,
            output=tmp_path / "replay.ts",
            fps=30,
            wallclock_start=datetime(2026, 8, 10, tzinfo=timezone.utc),
        ),
        api=fake_api(),
    )

    application.run()

    assert application.avp.commands[-2:] == [
        "event.on.node.finished transcode_finished replay_output",
        "event.wait transcode_finished",
    ]
    assert application.avp.group_calls == ["start", "shutdown"]
    assert application.avp.exception_callback_active is False


def test_player_graph_is_single_source_cuda_and_janus_video_only(tmp_path):
    recording = replay_file(tmp_path)
    config = PlayerConfig(
        ReplaySlotConfig(recording, loop=True),
        JanusVideoConfig(host="127.0.0.1", video_port=5004, payload_type=96, ssrc=0x41565001),
    )

    application = build_player_application(config, api=fake_api())
    nodes = {node.parameters["name"]: node.parameters for node in application.avp.nodes}
    serialized = repr(nodes).lower()

    assert application.artifact.fps == 25
    assert application.video_edge == "player_observed"
    assert application.avp.edges.plans == [("*", 1)]
    assert nodes["replay_input"]["seek_table"] == ""
    assert nodes["replay_input"]["ts_offsets"] == ""
    assert nodes["replay_input"]["timestamp_source"] == "wallclock"
    assert nodes["replay_input"]["preseek"] == 0
    assert nodes["replay_input"]["loop"] is True
    assert nodes["replay_input"]["pause_team"] == "replay_pause"
    assert "speed_team" not in nodes["replay_input"]
    assert nodes["replay_demux"]["routing"] == {"v:0": "player_video_packets"}
    assert nodes["replay_decode"]["pixel_format"] == "cuda"
    assert nodes["replay_decode"]["hwaccel"] == "replay_gpu"
    assert nodes["replay_decode"]["flush_magic"] is True
    assert nodes["replay_speed"]["team"] == "replay_speed"
    assert nodes["replay_transition_gate"]["type"] == "pause"
    assert nodes["replay_transition_gate"]["src"] == "player_speed_raw"
    assert nodes["replay_transition_gate"]["dst"] == "player_speed"
    assert nodes["replay_transition_gate"]["team"] == "replay_transition"
    assert nodes["replay_pause"]["team"] == "replay_pause"
    assert nodes["replay_realtime"]["team"] == "replay_sync"
    assert nodes["replay_fps"]["fps"] == "25/1"
    assert nodes["replay_position"]["dst"] == "player_observed"

    encoder = nodes["janus_encoder"]
    assert encoder["codec"] == "h264_nvenc"
    assert encoder["hwaccel"] == "replay_gpu"
    assert encoder["options"] == {
        "b": "4000k",
        "maxrate": "4000k",
        "bufsize": "4000k",
        "g": 25,
        "bf": 0,
        "preset": "p6",
        "profile": "baseline",
        "tune": "ull",
        "rc": "cbr",
        "rc-lookahead": 0,
        "zerolatency": 1,
        "delay": 0,
        "forced-idr": 1,
        "no-scenecut": 1,
        "strict_gop": 1,
        "aud": 1,
        "spatial-aq": 1,
        "temporal-aq": 0,
    }
    assert nodes["janus_mux"]["src"] == ["janus_headers"]
    assert nodes["janus_rtp_output"]["url"] == (
        "rtp://127.0.0.1:5004?pkt_size=1200&rtcp_port=5005"
    )
    assert nodes["janus_rtp_output"]["options"] == {
        "payload_type": 96,
        "rtpflags": "skip_rtcp",
        "ssrc": 0x41565001,
    }
    assert application.rtcp_feedback_listener.parameters["janus_rtcp_port"] == 5005
    assert "audio" not in serialized
    assert not any(node["type"].startswith("obs_") for node in nodes.values())
    assert "hwdownload" not in serialized
    assert "hwupload" not in serialized
    assert application.avp.commands == [
        'hwaccel.init { "name": "replay_gpu", "type": "cuda" }'
    ]


def test_janus_config_validates_port_pair_payload_and_ssrc():
    for kwargs in (
        {"video_port": 65535},
        {"payload_type": 128},
        {"ssrc": -1},
    ):
        try:
            JanusVideoConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid Janus config accepted: {kwargs}")


def test_player_lifecycle_starts_listener_and_stops_once(tmp_path):
    application = build_player_application(
        PlayerConfig(ReplaySlotConfig(replay_file(tmp_path)), JanusVideoConfig()),
        api=fake_api(),
    )
    application.controller.observe(frame_number=0, media_timestamp_ms=application.artifact.start_ms)
    application.start()

    assert application.avp.group_calls == ["start", "start"]
    assert application.rtcp_feedback_listener.started is True
    assert application.avp.ready is True

    application.stop()
    application.stop()
    assert application.avp.group_calls == [
        "start", "start",
        "probe-stop", "probe-detach",
        "stop-group:output", "stop-group:player", "shutdown",
    ]
    assert application.position_probe.attached is False
    assert application.rtcp_feedback_listener.started is False
    assert application.avp.exception_callback_active is False


def test_player_probe_timeout_does_not_start_native_teardown_and_can_retry(tmp_path):
    application = build_player_application(
        PlayerConfig(
            ReplaySlotConfig(replay_file(tmp_path), control_timeout=0.001),
            JanusVideoConfig(),
        ),
        api=fake_api(),
    )
    application.controller.observe(
        frame_number=0,
        media_timestamp_ms=application.artifact.start_ms,
    )
    application.start()

    def request_stop_without_finishing():
        application.avp.group_calls.append("probe-stop")

    application.position_probe.request_stop = request_stop_without_finishing
    with pytest.raises(TimeoutError, match="position probe"):
        application.stop()

    assert application.avp.group_calls == ["start", "start", "probe-stop"]
    assert application.position_probe.attached is True
    assert application._stopped is False

    application.position_probe.isWorking = False
    application.stop()
    assert application.avp.group_calls[-1] == "shutdown"
    assert application.position_probe.attached is False
    assert application._stopped is True


def test_player_stops_all_nodes_before_shutdown(tmp_path):
    application = build_player_application(
        PlayerConfig(
            ReplaySlotConfig(replay_file(tmp_path), control_timeout=0.5),
            JanusVideoConfig(),
        ),
        api=fake_api(),
    )
    application.controller.observe(frame_number=0, media_timestamp_ms=application.artifact.start_ms)
    application.start()
    application.stop()

    assert application.avp.shutdown_while_working is False


def test_player_waits_for_source_metadata_before_starting_output(tmp_path):
    application = build_player_application(
        PlayerConfig(
            ReplaySlotConfig(replay_file(tmp_path), control_timeout=0.5),
            JanusVideoConfig(),
        ),
        api=fake_api(),
    )
    timer = None

    def on_group_start(name):
        nonlocal timer
        if name == "player":
            timer = threading.Timer(
                0.02,
                lambda: application.controller.observe(
                    frame_number=0,
                    media_timestamp_ms=application.artifact.start_ms,
                ),
            )
            timer.start()
        elif name == "output":
            assert application.controller.status().ready

    application.avp.on_group_start = on_group_start
    try:
        application.start()
    finally:
        if timer:
            timer.join()
        application.stop()
