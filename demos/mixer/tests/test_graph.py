from types import SimpleNamespace

from mixer import GraphOptions, build_application, infer_output_format


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
        self.edges = FakeEdges()

    def addNode(self, node):
        self.nodes.append(node)

    def executeCommandsFromString(self, commands):
        self.commands.append(commands)

    def enableControlServer(self, port):
        self.control_port = port

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


def fake_api():
    names = (
        "assume_video_format",
        "dec_video",
        "demux",
        "enc_video",
        "filter_video",
        "force_fps",
        "input_rec",
        "mux",
        "output",
        "preheat_video_router",
        "realtime",
    )
    api = {
        "AVPlumber": FakeAvp,
        "MixerGraphBuilder": FakeMixer,
    }
    api.update({
        {
            "assume_video_format": "AssumeVideoFormat",
            "dec_video": "DecVideo",
            "demux": "Demux",
            "enc_video": "EncVideo",
            "filter_video": "FilterVideo",
            "force_fps": "ForceFPS",
            "input_rec": "InputRec",
            "mux": "Mux",
            "output": "Output",
            "preheat_video_router": "PreheatVideoRouter",
            "realtime": "Realtime",
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
    assert not hasattr(mixer, "add_source")
    assert not any(
        forbidden in node_type_name
        for node_type_name in node_types
        for forbidden in ("audio", "vad", "infer", "face", "speaker")
    )
    assert mixer.initial_scene == ("fullscreen_0", "A")


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

