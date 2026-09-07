"""Real Python graph/startup logic, with only the native engine substituted.

Frame correctness is tested by the native playout tests and GPU recordings;
these tests ensure READY cannot be published before the graph is warmed.
"""
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from mixer import GraphOptions, build_application


@pytest.fixture
def native_boundary(monkeypatch):
    before = set(sys.modules)
    monkeypatch.setitem(sys.modules, "_avplumber", SimpleNamespace(AVPlumber=object))
    nodes = importlib.import_module("pyplumber.node")
    builder = importlib.import_module("avpmixer").MixerGraphBuilder
    yield nodes, builder
    for name in set(sys.modules) - before:
        if name == "avpmixer" or name.startswith(("avpmixer.", "pyplumber")):
            sys.modules.pop(name, None)


class NativeEngine:
    """The engine exposes node/edge readiness and records public commands."""
    def __init__(self):
        self.nodes = {}
        self.started = set()
        self.events = []
        self.missing = set()
        self.drained = set()
        self.ready = False
        self.shutdown_complete = False
        self.edges = SimpleNamespace(planCapacity=lambda *_: None)

    def addNode(self, node):
        self.nodes[node.parameters["name"]] = node.parameters

    def executeCommandsFromString(self, commands):
        self.events.extend(commands.splitlines())

    def group(self, name):
        def start():
            self.started.add(name)
            self.events.append("start " + name)
        return SimpleNamespace(startNodes=start)

    def node(self, name):
        return SimpleNamespace(isWorking=self.nodes[name]["group"] in self.started)

    def getEdge(self, name):
        # Readiness is supplied by the engine boundary, never by the builder.
        self.events.append("inspect " + name)
        return SimpleNamespace(occupied=0 if name in self.missing | self.drained else 1,
                               enqueued_total=0 if name in self.missing else 1)

    def setReady(self):
        self.ready = True
        self.events.append("READY")

    def enableControlServer(self, _port):
        pass

    def shutdown(self):
        self.shutdown_complete = True


def application(native_boundary, **kwargs):
    nodes, builder = native_boundary
    api = SimpleNamespace(**{
        name: getattr(nodes, name) for name in (
            "AssumeVideoFormat", "Bsf", "DecVideo", "Demux", "EncVideo",
            "FilterVideo", "ForceFPS", "ForceKeyFrame", "InputRec", "Mux",
            "Output", "PreheatVideoRouter", "Realtime", "Split",
        )
    }, AVPlumber=NativeEngine, MixerGraphBuilder=builder)
    return build_application(GraphOptions(
        inputs=("camera-a.mp4", "camera-b.mp4"), output="program.ts",
        fps=60, **kwargs,
    ), api=api)


def test_ready_requires_inputs_compositors_and_transition_prewarm(native_boundary):
    app = application(native_boundary)
    engine = app.avp
    assert not engine.ready
    app.start()
    events = engine.events
    routes = next(i for i, event in enumerate(events) if event.startswith("mixer.init_routes "))
    for edge in app.input_edges:
        assert events.index("inspect " + edge) < routes
    transition_ready = events.index("inspect mixer_trans_out")
    assert events.index("node.object.set mixer_otm_scene_a outputs 2") < transition_ready
    assert events.index("node.object.set mixer_otm_scene_b outputs 2") < transition_ready
    for slot in ("a", "b"):
        restored = events.index(f"node.object.set mixer_otm_scene_{slot} outputs 1")
        assert transition_ready < restored < events.index("start output")
    assert events[-1] == "READY"
    assert engine.ready


@pytest.mark.parametrize("phase", ["input", "transition"])
def test_missing_prewarm_frame_never_publishes_ready(native_boundary, monkeypatch, phase):
    app = application(native_boundary, preheat_timeout_sec=0.01)
    edge = {
        "input": app.input_edges[-1],
        "transition": "mixer_trans_out",
    }[phase]
    app.avp.missing.add(edge)
    # Advance only the clock boundary, without sleeping or native GPU work.
    ticks = iter(range(10000))
    monkeypatch.setattr("mixer.time.monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="preheat timed out"):
        app.start()
    assert not app.avp.ready
    assert "output" not in app.avp.started
    if phase == "transition":
        for slot in ("a", "b"):
            assert f"node.object.set mixer_otm_scene_{slot} outputs 1" in app.avp.events


def test_both_slots_share_rate_and_delay_without_second_resampler(native_boundary):
    app = application(native_boundary, mixer_latency_ms=50)
    for slot in ("a", "b"):
        node = app.avp.nodes[f"mixer_comp_{slot}"]
        assert node["fps"] == "60/1"
        assert node["latency_ms"] == 50
        snapshot = app.avp.nodes[f"mixer_snapshot_{slot}"]
        assert snapshot["src"] == node["dst"]
        assert snapshot["fps"] == "60/1"
        assert snapshot["latency_ms"] == 50
        assert app.avp.nodes[f"mixer_otm_scene_{slot}"]["src"] == snapshot["dst"]
    output = app.avp.nodes["mixer_snapshot_output"]
    assert output["src"] == app.avp.nodes["mixer_wipe_sel"]["dst"]
    assert output["snapshot"] == snapshot["snapshot"]
    assert output["slot"] == -1


def test_prewarm_does_not_rebuild_graph_or_change_requested_program(native_boundary):
    app = application(native_boundary)
    app.mixer.initialize_routes()
    initial_nodes = dict(app.avp.nodes)
    app.mixer.begin_transition_preheat()
    app.mixer.finish_transition_preheat()
    assert app.avp.nodes == initial_nodes
    assert app.mixer.current_scene == "fullscreen_0"
    preview = [event for event in app.avp.events if event.startswith("mixer.preview ")]
    assert json.loads(preview[-1].partition(" ")[2])["scene"] == "fullscreen_0"


def test_media_wipe_path_is_registered_without_starting_an_empty_clip(native_boundary):
    app = application(native_boundary)
    app.start()
    engine = app.avp
    init = next(event for event in engine.events if event.startswith("mixer.init "))
    config = json.loads(init.split(" ", 2)[2])
    assert config["wipe_group"] == "mixer_wipe"
    assert config["wipe_input_node"] == "mixer_wipe_input"
    assert engine.nodes["mixer_wipe_input"]["group"] == "mixer_wipe"
    assert "mixer_wipe" not in engine.started
    upload = engine.nodes["mixer_wipe_fmt"]
    assert upload["hwaccel"] == config["hwaccel"]
    assert upload["graph"].split(",")[-1] == "hwupload"


def test_compatibility_import_keeps_the_real_builder(native_boundary):
    _, builder = native_boundary
    assert importlib.import_module("pyplumber.mixer").MixerGraphBuilder is builder


def test_stop_waits_for_the_engine_shutdown(native_boundary):
    app = application(native_boundary)
    app.start()
    app.stop()
    assert app.avp.shutdown_complete


def test_consumed_prewarm_frame_still_proves_readiness(native_boundary, monkeypatch):
    app = application(native_boundary, preheat_timeout_sec=0.01)
    app.avp.drained.add("mixer_trans_out")
    ticks = iter(range(10000))
    monkeypatch.setattr("mixer.time.monotonic", lambda: next(ticks))
    app.start()
    assert app.avp.ready


def test_program_excludes_frames_from_before_prewarm_finished(native_boundary, monkeypatch):
    monkeypatch.setattr("time.monotonic_ns", lambda: 123456789000)
    app = application(native_boundary)
    assert app.avp.nodes["mixer_otm_final"]["outputs"] == 0
    app.start()
    gates = [json.loads(event.partition(" ")[2]) for event in app.avp.events
             if event.startswith("timeline.set ")]
    assert {"name": "mixer_tl", "ch": "mixer_otm_final", "key": "outputs",
            "at": 123457, "val": 1} in gates
    assert app.avp.events.index("inspect mixer_final_out") < app.avp.events.index("READY")


def test_geometry_is_resolved_by_two_compositors_without_filter_branches(native_boundary):
    app = application(native_boundary)
    nodes = app.avp.nodes
    assert not any(name.startswith(("mixer_cs_", "layout_preheat_")) for name in nodes)
    for slot in ("a", "b"):
        compositor = nodes[f"mixer_comp_{slot}"]
        assert compositor["scale"] is True
        assert compositor["src"] == [f"mixer_source_{i}_{slot}" for i in range(2)]
        for i in range(2):
            assert f"mixer_source_{i}_{slot}" in nodes[f"mixer_otm_source_{i}"]["dst"]
    layer = nodes["mixer_comp_a"]["layers"][0]
    assert layer == {"dst_x": 0, "dst_y": 0, "dst_w": 1080, "dst_h": 1920, "fit": "contain", "source_canvas": {"w": 1920, "h": 1080}}
    commands = [event for event in app.avp.events if event.startswith("mixer.source ")]
    assert commands == [f"mixer.source mixer source_{i} mixer_otm_source_{i} {i}" for i in range(2)]


def test_existing_explicit_filter_sources_still_create_slot_filters(native_boundary):
    _, builder = native_boundary
    engine = NativeEngine()
    mixer = builder(engine)
    graph = "scale_cuda=w=640:h=360"
    mixer.add_source("camera", "decoded", "input", default_graph=graph)
    mixer.add_scene("program", {"camera": {"graph": graph, "dst_x": 100}})
    mixer.set_initial_scene("program")
    mixer.build()
    for slot in ("a", "b"):
        assert engine.nodes[f"mixer_cs_camera_{slot}"]["graph"] == graph
        assert engine.nodes[f"mixer_comp_{slot}"]["src"] == [f"mixer_camera_scaled_{slot}"]
