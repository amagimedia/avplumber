"""The graphs the demos send to avplumber, and the player's lifecycle over a
fake control connection."""

from datetime import datetime, timezone
from pathlib import Path
import json
import struct

import pytest

from replay import (
    GROUP,
    Backend,
    JANUS_FORCE_KEYFRAME_COMMAND,
    JanusVideoConfig,
    PlayerApplication,
    PlayerConfig,
    ReplaySlotConfig,
    TranscodeConfig,
    build_player_application,
    player_script,
    transcode_script,
    validate_recording,
)


def nodes_of(script):
    """`name -> parameters` of every `node.add` line, plus the other lines."""
    nodes, others = {}, []
    for line in script:
        if line.startswith("node.add "):
            params = json.loads(line[len("node.add "):])
            nodes[params["name"]] = params
        else:
            others.append(line)
    return nodes, others


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


def test_transcode_graph_is_video_only_all_intra_with_seek_tables(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"vod")
    config = TranscodeConfig(
        source=source,
        output=tmp_path / "replay.ts",
        fps=30,
        wallclock_start=datetime(2026, 8, 10, tzinfo=timezone.utc),
    )

    nodes, others = nodes_of(transcode_script(config))

    assert others == ["queue.plan_capacity * 4", "group.start transcode"]
    assert nodes["replay_input"]["eof_mode"] == "drain"
    assert nodes["replay_input"]["url"] == str(source)
    assert nodes["replay_demux"]["routing"] == {"?v:0": "transcode_video_packets"}
    assert nodes["replay_decode"]["src"] == "transcode_video_packets"
    assert "hwaccel" not in nodes["replay_decode"]
    assert nodes["replay_decode"]["codec_map"] == {"h264": "h264", "hevc": "hevc"}
    assert nodes["replay_fps"]["fps"] == "30/1"
    assert nodes["replay_keyframes"]["interval_sec"] == "1/30"
    encoder = nodes["replay_encoder"]
    assert encoder["codec"] == "libx264"
    assert encoder["options"]["g"] == "1" and encoder["options"]["bf"] == "0"
    assert "keyint=1" in encoder["options"]["x264-params"]
    assert all(isinstance(value, str) for value in encoder["options"].values())
    assert nodes["replay_mux"]["src"] == ["transcode_encoded"]
    output = nodes["replay_output"]
    assert output["format"] == "mpegts"
    assert output["seek_table"] == str(tmp_path / "replay.ts+seek")
    assert output["seek_table_text"] == str(tmp_path / "replay.ts+txt")
    assert {params["group"] for params in nodes.values()} == {"transcode"}


def test_nvidia_transcode_graph_decodes_and_encodes_on_the_gpu(tmp_path):
    """The NVIDIA backend: one device, cuvid in, NVENC out, and nothing that
    could move a frame to host memory in between."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"vod")
    config = TranscodeConfig(
        source=source,
        output=tmp_path / "replay.ts",
        fps=30,
        wallclock_start=datetime(2026, 8, 10, tzinfo=timezone.utc),
        backend=Backend.NVIDIA,
    )

    nodes, others = nodes_of(transcode_script(config))

    assert others[0] == 'hwaccel.init {"name":"replay_gpu","type":"cuda"}', \
        "the device opens before any node names it"
    decoder = nodes["replay_decode"]
    assert decoder["hwaccel"] == "replay_gpu"
    assert decoder["pixel_format"] == "cuda"
    assert decoder["codec_map"] == {"h264": "h264_cuvid", "hevc": "hevc_cuvid"}
    assert decoder["hwaccel_only_for_codecs"] == ["h264", "hevc"]
    encoder = nodes["replay_encoder"]
    assert encoder["codec"] == "h264_nvenc" and encoder["hwaccel"] == "replay_gpu"
    assert encoder["options"]["cq"] == "17" and encoder["options"]["rc"] == "vbr"
    assert encoder["options"]["delay"] == "0"
    # No node between them may download, upload or convert.
    between = ["replay_fps", "replay_keyframes"]
    assert [name for name in nodes if name not in
            {"replay_input", "replay_demux", "replay_decode", "replay_encoder",
             "replay_mux", "replay_output"}] == between
    for name in between:
        assert "pixel_format" not in nodes[name], name
        assert "hwaccel" not in nodes[name], name


def test_player_graph_is_one_seekable_slot_paced_into_a_janus_rtp_leg(tmp_path):
    recording = replay_file(tmp_path)
    config = PlayerConfig(
        ReplaySlotConfig(recording, loop=False),
        JanusVideoConfig("10.0.0.5", 6000, 97, 0x1234),
    )

    nodes, others = nodes_of(player_script(config, validate_recording(recording)))

    assert others == ["queue.plan_capacity * 1"], "the client starts the group itself"
    source = nodes["replay_input"]
    assert source["sync_group"] == GROUP and source["loop"] is False
    assert source["url"] == str(recording)
    assert "eof_mode" not in source and "stop_delay" not in source
    assert nodes["replay_demux"]["routing"] == {"v:0": "player_video_packets"}
    assert nodes["replay_decode"]["options"] == {"threads": "1", "flags": "low_delay"}
    pacing = nodes["replay_realtime"]
    assert pacing["type"] == "realtime" and pacing["sync_group"] == GROUP
    assert pacing["tick_period"] == "1/25"
    assert nodes["janus_force_keyframe"]["interval_sec"] == "1/1"
    assert nodes["janus_force_keyframe"]["src"] == "player_realtime"
    encoder = nodes["janus_encoder"]
    assert encoder["codec"] == "libx264" and encoder["options"]["g"] == "25"
    assert encoder["flush"] == "keep", "a live encoder is never flushed by a seek"
    assert nodes["janus_headers"]["bsf"] == "dump_extra=freq=keyframe"
    output = nodes["janus_rtp_output"]
    assert output["format"] == "rtp"
    assert output["url"] == "rtp://10.0.0.5:6000?pkt_size=1200&rtcp_port=6001"
    assert output["options"] == {"payload_type": "97", "rtpflags": "skip_rtcp", "ssrc": "4660"}
    assert {params["group"] for params in nodes.values()} == {"player"}
    assert "hwaccel" not in nodes["replay_decode"]
    # One chain, every edge produced once and consumed once.
    produced = [params["dst"] for params in nodes.values() if "dst" in params]
    produced += [edge for params in nodes.values() for edge in params.get("routing", {}).values()]
    consumed = [edge for params in nodes.values() if "src" in params
                for edge in (params["src"] if isinstance(params["src"], list) else [params["src"]])]
    assert sorted(produced) == sorted(consumed)


class FakeClient:
    """Answers commands the way avplumber does, and remembers them."""

    def __init__(self, *, frames=(), fail=None):
        self.commands = []
        self.connected = False
        self.closed = False
        self.frames = list(frames)
        self.serial = 0
        self.fail = fail

    def connect(self):
        self.connected = True

    def command(self, line):
        self.commands.append(line)
        if self.fail and line.startswith(self.fail):
            raise RuntimeError(f"refused: {line}")
        if line.startswith("playback.status"):
            if self.frames:
                self.serial += 1
                frame = self.frames.pop(0)
                return json.dumps({"serial": self.serial, "frame": frame,
                                   "media_ms": 1_260 + frame * 40, "at_end": False})
            return json.dumps({"serial": self.serial, "frame": None, "media_ms": None})
        if line.startswith("group.status"):
            return json.dumps({"state": "running", "outcomes": []})
        return ""

    def status(self, group):
        return json.loads(self.command(f"playback.status {group}"))

    def group_status(self, group):
        return json.loads(self.command(f"group.status {group}"))

    def close(self):
        self.closed = True
        self.connected = False


class FakeListener:
    def __init__(self, **parameters):
        self.parameters = parameters
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


def build(tmp_path, client, **slot):
    recording = replay_file(tmp_path)
    config = PlayerConfig(ReplaySlotConfig(recording, **slot), JanusVideoConfig(video_port=6000))
    application = build_player_application(
        config, connect=("127.0.0.1", 1), listener_factory=FakeListener,
    )
    application.client = client
    return application


def test_nvidia_player_graph_keeps_the_live_leg_on_the_gpu(tmp_path):
    """The player on NVIDIA: NVDEC surfaces are paced, given a keyframe and
    encoded by NVENC in place. The encoder still refuses to be flushed by a
    seek, which is what a live output needs whatever the codec."""
    recording = replay_file(tmp_path)
    config = PlayerConfig(
        ReplaySlotConfig(recording, loop=False),
        JanusVideoConfig("10.0.0.5", 6000, 97, 0x1234),
        Backend.NVIDIA,
    )

    nodes, others = nodes_of(player_script(config, validate_recording(recording)))

    assert others == ['hwaccel.init {"name":"replay_gpu","type":"cuda"}',
                      "queue.plan_capacity * 1"]
    decoder = nodes["replay_decode"]
    assert decoder["hwaccel"] == "replay_gpu" and decoder["pixel_format"] == "cuda"
    assert decoder["codec_map"] == {"h264": "h264_cuvid"}
    assert decoder["hwaccel_only_for_codecs"] == ["h264"]
    assert decoder["options"] == {"threads": "1", "flags": "low_delay"}
    encoder = nodes["janus_encoder"]
    assert encoder["codec"] == "h264_nvenc" and encoder["hwaccel"] == "replay_gpu"
    assert encoder["flush"] == "keep"
    assert encoder["options"]["tune"] == "ull" and encoder["options"]["delay"] == "0"
    assert encoder["options"]["forced-idr"] == "1"
    assert encoder["options"]["b"] == "4000k" and encoder["options"]["g"] == "25"


def test_the_output_bitrate_follows_the_configuration(tmp_path):
    """A bigger picture needs a bigger number, so the encoder takes it from the
    Janus configuration rather than a constant."""
    recording = replay_file(tmp_path)
    nodes, _ = nodes_of(player_script(
        PlayerConfig(ReplaySlotConfig(recording), JanusVideoConfig(bitrate="40M"),
                     Backend.NVIDIA),
        validate_recording(recording),
    ))
    options = nodes["janus_encoder"]["options"]
    assert options["b"] == "40M" and options["maxrate"] == "40M"
    assert options["bufsize"] == "40M"
    # Between decoder and encoder, only pacing and the keyframe request: no
    # node that would touch pixels or move them off the device.
    for name in ("replay_realtime", "janus_force_keyframe"):
        assert "hwaccel" not in nodes[name] and "pixel_format" not in nodes[name], name


@pytest.mark.parametrize("backend", [Backend.CPU, Backend.NVIDIA])
def test_only_the_codecs_differ_between_backends(tmp_path, backend):
    """Everything that is not a codec is the same graph on both backends: same
    nodes, same edges, same order."""
    recording = replay_file(tmp_path)

    def shape(chosen):
        nodes, _ = nodes_of(player_script(
            PlayerConfig(ReplaySlotConfig(recording), JanusVideoConfig(), chosen),
            validate_recording(recording),
        ))
        return [(name, params["type"], params.get("src"), params.get("dst"))
                for name, params in nodes.items()]

    assert shape(backend) == shape(Backend.CPU)


def test_player_start_sends_the_graph_waits_for_a_frame_then_listens(tmp_path):
    client = FakeClient(frames=[0, 1, 2])
    application = build(tmp_path, client)

    application.start()
    try:
        node_adds = [c for c in client.commands if c.startswith("node.add ")]
        assert len(node_adds) == 9
        assert client.commands.index("group.start player") > client.commands.index(node_adds[-1])
        assert application.controller.status().ready is True
        assert application.controller.status().frame_number == 0
        assert application.rtcp_feedback_listener.started is True
        listener = application.rtcp_feedback_listener.parameters
        assert listener["janus_rtcp_port"] == 6001 and listener["media_ssrc"] == 0x41565001

        # Seeks are followed by a keyframe request for Janus; other commands not.
        application.controller.execute("seek_frames", 5)
        assert client.commands[-2:] == ["seek replay frame +5", JANUS_FORCE_KEYFRAME_COMMAND]
        application.controller.execute("pause")
        assert client.commands[-1] == "pause replay now"
    finally:
        application.stop()

    assert application.rtcp_feedback_listener.started is False
    assert "group.stop player" in client.commands
    assert client.closed is True
    application.stop()  # idempotent


def test_player_start_failure_stops_everything(tmp_path):
    client = FakeClient(fail="group.start")
    application = build(tmp_path, client)
    with pytest.raises(RuntimeError, match="refused: group.start"):
        application.start()
    assert application.rtcp_feedback_listener.started is False
    assert client.closed is True


def test_player_start_times_out_without_a_first_frame(tmp_path):
    client = FakeClient(frames=[])
    application = build(tmp_path, client, control_timeout=0.2)
    with pytest.raises(TimeoutError, match="first source frame"):
        application.start()
    assert client.closed is True


def test_player_needs_a_binary_or_an_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("AVPLUMBER_BIN", raising=False)
    recording = replay_file(tmp_path)
    config = PlayerConfig(ReplaySlotConfig(recording), JanusVideoConfig())
    with pytest.raises(FileNotFoundError, match="AVPLUMBER_BIN"):
        build_player_application(config, listener_factory=FakeListener)
