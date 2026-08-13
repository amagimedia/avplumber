from types import SimpleNamespace

import pytest

from mixer import GraphOptions, build_application, infer_output_format, parse_args


class FakeNode:
    node_type = "unknown"

    def __init__(self, parameters):
        self.parameters = {"type": self.node_type, **parameters}


def node_type(name):
    return type(name, (FakeNode,), {"node_type": name})


class FakeGroup:
    def startNodes(self):
        pass

    def stopNodes(self):
        pass


class FakeEdges:
    def __init__(self):
        self.plans = []

    def planCapacity(self, pattern, capacity):
        self.plans.append((pattern, capacity))


class FakeAvp:
    def __init__(self):
        self.nodes = []
        self.commands = []
        self.control_port = None
        self.ready = False
        self.edges = FakeEdges()

    def addNode(self, node):
        self.nodes.append(node)

    def executeCommandsFromString(self, commands):
        self.commands.append(commands)

    def enableControlServer(self, port):
        self.control_port = port

    def setReady(self):
        self.ready = True

    def group(self, _name):
        return FakeGroup()


class FakeMixer:
    instances = []

    def __init__(self, avp, **parameters):
        self.avp = avp
        self.parameters = parameters
        self.timeline = "mixer_timeline"
        self.routed_sources = []
        self.scenes = {}
        self.initial_scene = None
        self.instances.append(self)

    def add_routed_source(self, name, **parameters):
        self.routed_sources.append((name, parameters))

    def add_scene(self, name, sources, *, routes):
        self.scenes[name] = {"sources": sources, "routes": routes}

    def set_initial_scene(self, name, slot):
        self.initial_scene = (name, slot)

    def build(self):
        return "mixer_final_out"

    def start_groups(self):
        pass


class FakeRtcpFeedbackListener:
    def __init__(self, **parameters):
        self.parameters = parameters
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


def fake_api():
    names = (
        "assume_video_format",
        "bsf",
        "dec_video",
        "demux",
        "enc_video",
        "filter_video",
        "force_fps",
        "force_key_frame",
        "input_rec",
        "mux",
        "output",
        "preheat_video_router",
        "realtime",
        "split",
    )
    api = {
        "AVPlumber": FakeAvp,
        "MixerGraphBuilder": FakeMixer,
        "RtcpFeedbackListener": FakeRtcpFeedbackListener,
    }
    api.update({
        {
            "assume_video_format": "AssumeVideoFormat",
            "bsf": "Bsf",
            "dec_video": "DecVideo",
            "demux": "Demux",
            "enc_video": "EncVideo",
            "filter_video": "FilterVideo",
            "force_fps": "ForceFPS",
            "force_key_frame": "ForceKeyFrame",
            "input_rec": "InputRec",
            "mux": "Mux",
            "output": "Output",
            "preheat_video_router": "PreheatVideoRouter",
            "realtime": "Realtime",
            "split": "Split",
        }[name]: node_type(name)
        for name in names
    })
    return SimpleNamespace(**api)


def test_graph_is_video_only_and_always_preheated():
    FakeMixer.instances.clear()
    application = build_application(
        GraphOptions(
            inputs=tuple(f"input-{index}.mp4" for index in range(17)),
            output="program.mp4",
        ),
        api=fake_api(),
    )
    mixer = FakeMixer.instances[-1]
    node_types = [node.parameters["type"] for node in application.avp.nodes]

    assert node_types.count("preheat_video_router") == 1
    assert len(mixer.routed_sources) == 1 + 2 + 4 + 8 + 16
    assert mixer.parameters["defer_initial_routes"] is True
    router = next(
        node.parameters
        for node in application.avp.nodes
        if node.parameters["type"] == "preheat_video_router"
    )
    assert len(router["routes"]) == 2 * (1 + 2 + 4 + 8 + 16)
    assert set(router["routes"]) == set(range(17))
    assert len(application.preheated_output_edges) == len(router["routes"])
    assert mixer.parameters["enable_wipe"] is False
    assert not hasattr(mixer, "add_source")
    assert not any(
        forbidden in node_type_name
        for node_type_name in node_types
        for forbidden in ("audio", "vad", "infer", "face", "speaker")
    )
    assert mixer.initial_scene == ("fullscreen_0", "A")
    normalizers = [
        node.parameters
        for node in application.avp.nodes
        if node.parameters["type"] == "filter_video"
        and node.parameters["name"].startswith("normalize_")
    ]
    assert normalizers
    assert {node["dst_frame_rate"] for node in normalizers} == {"30/1"}

    serialized_graph = repr([node.parameters for node in application.avp.nodes]).lower()
    assert "hwdownload" not in serialized_graph
    assert "hwupload" not in serialized_graph


def test_graph_has_stable_fullscreen_and_paged_scenes():
    FakeMixer.instances.clear()
    build_application(
        GraphOptions(inputs=("one", "two", "three"), output="program.flv"),
        api=fake_api(),
    )
    scenes = FakeMixer.instances[-1].scenes

    assert {f"fullscreen_{index}" for index in range(3)} <= scenes.keys()
    assert "grid_2_page_0" in scenes
    assert "grid_2_page_1" in scenes
    assert scenes["grid_2_page_1"]["routes"] == {"layout_2_slot_0": 2}


def test_output_format_inference_is_explicit_when_ambiguous():
    assert infer_output_format("rtmp://example.invalid/live") == "flv"
    assert infer_output_format("srt://example.invalid:9000") == "mpegts"
    assert infer_output_format("program.mp4") == "mp4"
    assert infer_output_format("anything", "nut") == "nut"


def test_cli_accepts_ordered_repeatable_input_paths():
    options = parse_args([
        "--input",
        "/media/first clip.mp4",
        "--input",
        "/media/second.mp4",
        "--output",
        "program.mp4",
        "--fps",
        "60",
        "--loop-inputs",
    ])

    assert options.inputs == ("/media/first clip.mp4", "/media/second.mp4")
    assert options.fps == 60
    assert options.loop_inputs is True


def test_configured_fps_reaches_decode_normalization_mixer_and_outputs():
    FakeMixer.instances.clear()
    application = build_application(
        GraphOptions(
            inputs=("input.mp4",),
            output="program.ts",
            janus_output=True,
            fps=60,
        ),
        api=fake_api(),
    )
    nodes = {node.parameters["name"]: node.parameters for node in application.avp.nodes}

    assert nodes["fps_0"]["fps"] == "60/1"
    assert nodes["normalize_0"]["dst_frame_rate"] == "60/1"
    assert nodes["layout_preheat_router"]["frame_rate"] == "60/1"
    assert nodes["program_fps"]["fps"] == "60/1"
    assert nodes["program_encoder"]["options"]["g"] == 120
    assert nodes["janus_fps"]["fps"] == "60/1"
    assert nodes["janus_encoder"]["options"]["g"] == 60
    assert "level" not in nodes["janus_encoder"]["options"]
    assert FakeMixer.instances[-1].parameters["fps"] == (60, 1)


def test_janus_only_output_builds_video_rtp_and_feedback():
    application = build_application(
        GraphOptions(inputs=("input.mp4",), janus_output=True),
        api=fake_api(),
    )
    nodes = {node.parameters["name"]: node.parameters for node in application.avp.nodes}

    assert nodes["janus_encoder"]["codec"] == "h264_nvenc"
    assert nodes["janus_rtp_output"]["format"] == "rtp"
    assert nodes["janus_rtp_output"]["options"]["payload_type"] == 96
    assert application.rtcp_feedback_listener is not None


def test_record_and_janus_outputs_split_program_video():
    application = build_application(
        GraphOptions(
            inputs=("input.mp4",),
            output="program.mp4",
            janus_output=True,
        ),
        api=fake_api(),
    )
    nodes = {node.parameters["name"]: node.parameters for node in application.avp.nodes}

    assert nodes["split_program_video_output"]["dst"] == [
        "program_video_record",
        "program_video_janus",
    ]


def test_output_target_is_required():
    with pytest.raises(ValueError, match="--output or --janus-output"):
        build_application(GraphOptions(inputs=("input.mp4",)), api=fake_api())


def test_cpu_encoder_is_rejected():
    with pytest.raises(ValueError, match="NVENC"):
        build_application(
            GraphOptions(inputs=("input.mp4",), output="program.mp4", codec="libx264"),
            api=fake_api(),
        )
