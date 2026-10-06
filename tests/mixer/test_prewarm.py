"""Real Python graph/startup logic, with only the native engine substituted.

Frame correctness is tested by the native playout tests and GPU recordings;
these tests ensure READY cannot be published before the graph is warmed.
"""
import importlib
import json
from types import SimpleNamespace

import pytest

from pyplumber.mixer.cli import GraphOptions, build_application
from pyplumber.mixer import clipcache
from wipe_loader_sim import WipeLoaderSim




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
        self.wipes = WipeLoaderSim()
        self.loaded_clips = self.wipes.loads   # what the running clip cache was told to load
        self.edges = SimpleNamespace(planCapacity=lambda *_: None)
        self.manager = SimpleNamespace(shouldWork=True)

    def addNode(self, node):
        self.nodes[node.parameters["name"]] = node.parameters

    def on_exception(self, name, kind, message):
        pass

    def executeCommandsFromString(self, commands):
        self.events.extend(commands.splitlines())
        self.wipes.executeCommandsFromString(commands)

    def group(self, name):
        loader = self.wipes.group(name)   # the wipe loader chain runs, ends and stops in the simulation
        def start():
            self.started.add(name)
            self.events.append("start " + name)
            if loader:
                loader.startNodes()
        def stop():
            self.events.append("stop " + name)
            if loader:
                loader.stopNodes()
        return SimpleNamespace(startNodes=start, stopNodes=stop)

    def node(self, name):
        if name in self.wipes.loader_nodes:
            return self.wipes.node(name)
        return SimpleNamespace(isWorking=self.nodes[name]["group"] in self.started, getObject=self.wipes.status)

    def getEdge(self, name, data_type=None):
        wipe_edge = self.wipes.getEdge(name)
        if wipe_edge:
            return wipe_edge
        # Readiness is supplied by the engine boundary, never by the builder.
        if name.endswith("_encoded"):
            assert data_type == "packet", "readiness must not create a VideoFrame edge before the encoder"
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
        self.events.append("shutdown")


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
def test_native_startup_error_aborts_without_waiting_for_frames(native_boundary, monkeypatch, phase):
    app = application(native_boundary)
    engine = app.avp
    edge = app.input_edges[0] if phase == "input" else "mixer_trans_out"
    engine.missing.add(edge)
    received = []
    original = engine.on_exception = lambda *args: received.append(args)
    failure = ("mixer_color_camera0", "filter_video", "unsupported CUDA storage")
    inspect = engine.getEdge
    def get_edge(name, data_type=None):
        if name == edge:
            engine.on_exception(*failure)
        return inspect(name, data_type)
    monkeypatch.setattr(engine, "getEdge", get_edge)
    monkeypatch.setattr("pyplumber.mixer.cli.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="mixer_color_camera0.*unsupported CUDA storage"):
        app.start()
    assert received == [failure]
    assert engine.on_exception is original
    assert not engine.ready


def test_native_error_handler_restored_after_successful_start(native_boundary):
    app = application(native_boundary)
    received = []
    original = app.avp.on_exception = lambda *args: received.append(args)
    app.start()
    assert app.avp.on_exception is original
    failure = ("decoder", "dec_video", "later failure")
    app.avp.on_exception(*failure)
    assert received == [failure]
    assert app._startup_error is None


@pytest.mark.parametrize("failure", [False, True])
def test_ready_waits_for_aux_encoder_and_keeps_startup_error_handler(native_boundary, monkeypatch, failure):
    app = application(native_boundary)
    class Aux:
        def __iter__(self):
            return iter([SimpleNamespace(prefix="aux_test")])

        def start(self):
            app.avp.events.append("start aux_test")
    app.aux_buses = Aux()
    original = app.avp.on_exception
    inspect = app.avp.getEdge
    def get_edge(name, data_type=None):
        if name == "aux_test_encoded" and failure:
            app.avp.missing.add(name)
            app.avp.on_exception("aux_test_encoder", "enc_video", "Cannot allocate memory")
        return inspect(name, data_type)
    monkeypatch.setattr(app.avp, "getEdge", get_edge)
    monkeypatch.setattr("pyplumber.mixer.cli.time.sleep", lambda _: None)
    if failure:
        with pytest.raises(RuntimeError, match="aux_test_encoder.*Cannot allocate memory"):
            app.start()
        assert not app.avp.ready
    else:
        app.start()
        assert app.avp.events.index("inspect aux_test_encoded") < app.avp.events.index("READY")
    assert app.avp.on_exception == original


def test_cut_measurements_are_opt_in_and_enabled_after_encoder_start(native_boundary):
    app = application(native_boundary, cut_latency_encoder="program_encoder")
    app.start()
    command = 'mixer.measurements {"mixer": "mixer", "encoder": "program_encoder"}'
    assert app.avp.events.index("start output") < app.avp.events.index(command)
    assert app.avp.events.index(command) < app.avp.events.index("READY")
    ordinary = application(native_boundary)
    ordinary.start()
    assert not any(event.startswith("mixer.measurements ") for event in ordinary.avp.events)


def test_direct_cut_prewarm_is_opt_in_and_enabled_before_ready(native_boundary):
    app = application(native_boundary, prewarm_cut_scenes=("fullscreen_0", "fullscreen_1"))
    app.start()
    command = 'mixer.prewarm {"mixer": "mixer", "scenes": ["fullscreen_0", "fullscreen_1"]}'
    assert app.avp.events.index("start output") < app.avp.events.index(command)
    assert app.avp.events.index(command) < app.avp.events.index("READY")
    ordinary = application(native_boundary)
    ordinary.start()
    assert not any(event.startswith("mixer.prewarm ") for event in ordinary.avp.events)


def test_direct_cut_prewarm_wildcard_uses_the_current_catalogue(native_boundary):
    app = application(native_boundary, prewarm_cut_scenes=("*",))
    app.start()
    command = next(event for event in app.avp.events if event.startswith("mixer.prewarm "))
    assert json.loads(command.partition(" ")[2])["scenes"] == app.mixer.scenes()


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
    monkeypatch.setattr("pyplumber.mixer.cli.time.monotonic", lambda: next(ticks))
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
        assert "warmup_timeout_ms" not in node   # a cold load must not flip to a partial canvas
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
    app = application(native_boundary, wipe_cache_mb=0)   # the decode-per-take chain
    app.start()
    engine = app.avp
    init = next(event for event in engine.events if event.startswith("mixer.init "))
    config = json.loads(init.split(" ", 2)[2])
    assert config["wipe_group"] == "mixer_wipe"
    assert config["wipe_input_node"] == "mixer_wipe_input"
    assert "wipe_overlay" not in config and "wipe_cache_store" not in config
    assert "mixer_wipe_cache" not in engine.nodes
    assert engine.nodes["mixer_wipe_input"]["group"] == "mixer_wipe"
    # The whole chain, clip conforming included, is started with the group, per take.
    assert engine.nodes["mixer_wipe_rt_fps"]["group"] == "mixer_wipe"
    assert engine.nodes["mixer_wipe_overlay"]["src"] == ["mixer_final_wipe_in", "mixer_wipe_rt_fps_out"]
    assert engine.nodes["mixer_wipe_overlay"]["active_inputs"] == 3
    pacer = engine.nodes["mixer_wipe_rt"]   # stamps the clip as it plays; a marker would end the compositor's input
    assert pacer["set_pts"] is True and not pacer.get("forward_eof")
    assert "mixer_wipe" not in engine.started and "mixer_wipe_load" not in engine.started
    upload = engine.nodes["mixer_wipe_fmt"]
    assert upload["hwaccel"] == config["hwaccel"]
    assert upload["graph"].split(",")[-1] == "hwupload_cuda=pinned=1"


def test_cached_wipe_chain_runs_from_startup_parked_and_is_never_stopped(native_boundary):
    # A take arms the resident player and compositor instead of creating, starting and
    # stopping nodes: nothing churns CUDA allocations or threads under the program.
    app = application(native_boundary, wipe_cache_mb=256, wipe_file="/media/wipe.mov")
    app.start()
    engine = app.avp
    events = engine.events
    init = next(event for event in events if event.startswith("mixer.init "))
    config = json.loads(init.split(" ", 2)[2])
    assert config["wipe_input_node"] == "mixer_wipe_cache"
    assert config["wipe_overlay"] == "mixer_wipe_overlay"
    assert config["wipe_cache_store"] == "clips"
    assert "wipe_flush_edges" not in config
    # The player feeds the compositor directly, stamped on the output grid; the
    # compositor starts parked.
    assert "mixer_wipe_rt_fps" not in engine.nodes
    assert engine.nodes["mixer_wipe_cache"]["group"] == "mixer_wipe"
    assert engine.nodes["mixer_wipe_overlay"]["src"] == ["mixer_final_wipe_in", "mixer_wipe_cached"]
    assert engine.nodes["mixer_wipe_overlay"]["active_inputs"] == 0
    # Started with the mixer's own groups; the preload only starts the decode chain and
    # tells the running player which clip it delivers.
    player = events.index("start mixer_wipe")
    assert events.index("start mixer") < player < events.index("start output")
    load = events.index('node.object.set mixer_wipe_cache load "/media/wipe.mov"')
    assert player < load < events.index("start mixer_wipe_load") < events.index("stop mixer_wipe_load")
    assert events.index("stop mixer_wipe_load") < events.index("READY")
    assert "stop mixer_wipe" not in events
    assert not any(event.startswith("mixer.wipe.warmup") for event in events)
    assert engine.loaded_clips == ["/media/wipe.mov"]
    # The loader paces the clip but keeps its own timestamps, and passes the end-of-stream
    # marker on: the cache ends a load on that marker and on nothing else.
    pacer = engine.nodes["mixer_wipe_rt"]
    assert pacer["forward_eof"] is True and pacer["set_pts"] is False
    # clipcache.preload() stops, inspects and clears this chain by these names.
    loader = [name for name, node in engine.nodes.items() if node["group"] == "mixer_wipe_load"]
    assert loader == ["mixer_" + name for name in clipcache.LOADER_NODES]
    assert [engine.nodes[name].get("dst") or engine.nodes[name]["routing"]["v:0"] for name in loader] == [
        "mixer_" + edge for edge in (*clipcache.LOADER_EDGES, clipcache.CACHE_INPUT_EDGE)]
    assert engine.nodes["mixer_wipe_dec"]["dst"] == "mixer_" + clipcache.DECODED_EDGE
    assert engine.nodes["mixer_" + clipcache.CACHE_NODE]["src"] == "mixer_" + clipcache.CACHE_INPUT_EDGE


def test_a_wipe_evicted_by_a_later_wipe_fails_the_start(native_boundary):
    app = application(native_boundary, wipe_cache_mb=256, wipe_file="/media/wipe.mov")
    app.wipe_files = ("/media/second.mov",)
    app.avp.wipes.budget_frames = 200   # either clip of 120 frames fits, both do not
    with pytest.raises(RuntimeError, match=r"/media/wipe\.mov.* 120 MiB.* 200 MiB"):
        app.start()
    assert app.avp.loaded_clips == ["/media/wipe.mov", "/media/second.mov"] and not app.avp.ready


@pytest.mark.parametrize("cache_mb, store", [(0, None), (256, "clips")])
def test_mixer_init_names_the_wipe_cache_store_only_when_caching(native_boundary, cache_mb, store):
    # mixer.status reports the store it is given; without one it has no wipe_cache.
    app = application(native_boundary, wipe_cache_mb=cache_mb)
    app.start()
    init = next(event for event in app.avp.events if event.startswith("mixer.init "))
    assert json.loads(init.split(" ", 2)[2]).get("wipe_cache_store") == store


def test_mixer_init_names_the_canvas_a_dip_colour_is_converted_for(native_boundary):
    app = application(native_boundary)
    app.start()
    init = next(event for event in app.avp.events if event.startswith("mixer.init "))
    assert json.loads(init.split(" ", 2)[2])["color"] == "sdr"


def test_compatibility_import_keeps_the_real_builder(native_boundary):
    _, builder = native_boundary
    assert importlib.import_module("pyplumber.mixer").MixerGraphBuilder is builder


def test_stop_waits_for_the_engine_shutdown(native_boundary):
    app = application(native_boundary)
    app.start()
    app.stop()
    assert app.avp.shutdown_complete


def test_stop_asks_every_input_group_before_the_serial_shutdown(native_boundary):
    app = application(native_boundary)
    app.start()
    app.stop()
    events = app.avp.events
    stops = [events.index("stop " + group) for group in app.input_groups]
    assert len(stops) == 2 and max(stops) < events.index("shutdown") == len(events) - 1


def test_stop_after_a_panic_leaves_groups_to_the_running_shutdown(native_boundary):
    app = application(native_boundary)
    app.start()
    app.avp.manager.shouldWork = False
    app.stop()
    assert not any(event.startswith("stop ") for event in app.avp.events)
    assert app.avp.events[-1] == "shutdown"


def test_panic_ends_the_run_loop_and_the_process_fails(native_boundary, monkeypatch):
    import pyplumber.mixer.cli as mixer
    app = application(native_boundary)
    app.start = lambda: None
    sleeps = []
    def sleep(_seconds):
        sleeps.append(_seconds)
        app.avp.manager.shouldWork = len(sleeps) < 3
    monkeypatch.setattr(mixer.time, "sleep", sleep)
    monkeypatch.setattr(mixer, "build_application", lambda _options: app)
    with pytest.raises(SystemExit, match="auto_restart panic") as exit_:
        mixer.main(["--input", "a.mp4", "--output", "p.ts"])
    assert len(sleeps) == 3 and exit_.value.code
    assert app.avp.events[-1] == "shutdown"


def test_interrupt_stops_cleanly_without_failing(native_boundary, monkeypatch):
    import pyplumber.mixer.cli as mixer
    app = application(native_boundary)
    def interrupted(*_):
        raise KeyboardInterrupt
    monkeypatch.setattr(mixer, "build_application", lambda _options: app)
    app.start = interrupted
    mixer.main(["--input", "a.mp4", "--output", "p.ts"])
    assert app.avp.events[-1] == "shutdown"


def test_consumed_prewarm_frame_still_proves_readiness(native_boundary, monkeypatch):
    app = application(native_boundary, preheat_timeout_sec=0.01)
    app.avp.drained.add("mixer_trans_out")
    ticks = iter(range(10000))
    monkeypatch.setattr("pyplumber.mixer.cli.time.monotonic", lambda: next(ticks))
    app.start()
    assert app.avp.ready


def test_program_excludes_frames_from_before_prewarm_finished(native_boundary, monkeypatch):
    monkeypatch.setattr("time.monotonic_ns", lambda: 123456789000)
    app = application(native_boundary)
    assert app.avp.nodes["mixer_otm_final"]["outputs"] == 0
    app.start()
    assert "node.object.set mixer_otm_final enable_from 123457" in app.avp.events
    assert app.avp.events.index("inspect mixer_final_out") < app.avp.events.index("READY")


@pytest.mark.parametrize("count, lit", [(70, [0, 63, 64, 69]), (150, [0, 63, 64, 127, 128, 149])])
def test_pad_masks_above_64_travel_as_bit_strings(native_boundary, tmp_path, count, lit):
    """A show wider than 64 pads cannot put active_inputs in a JSON number, so the builder sends
    the least-significant-bit-first bit string mixer_compositor also parses."""
    doc = {
        "canvas": {"width": 320, "height": 180, "fps": 60},
        "sources": [{"id": f"s{i}", "kind": "video", "path": f"/m/{i}.mp4", "width": 320, "height": 180}
                    for i in range(count)],
        "scenes": [{"id": "wide", "items": [{"source": f"s{i}", "dst": {"x": 0, "y": 0, "w": 320, "h": 180}}
                                            for i in lit]}],
        "initial_scene": "wide",
    }
    path = tmp_path / "wide.json"
    path.write_text(json.dumps(doc))
    class ConfigEngine(NativeEngine):   # a config-driven show also registers demo commands
        def registerControlCommand(self, *args, **kwargs):
            pass

    nodes, builder = native_boundary
    api = SimpleNamespace(**{name: getattr(nodes, name) for name in (
        "AssumeVideoFormat", "Bsf", "DecVideo", "Demux", "EncVideo", "FilterVideo", "ForceFPS",
        "ForceKeyFrame", "InputRec", "Mux", "Output", "PreheatVideoRouter", "Realtime", "Split")},
        AVPlumber=ConfigEngine, MixerGraphBuilder=builder)
    app = build_application(GraphOptions(config=str(path), output="program.ts"), api=api)

    masks = {slot: app.avp.nodes[f"mixer_comp_{slot}"]["active_inputs"] for slot in ("a", "b")}
    program = [m for m in masks.values() if m != 0]
    assert len(program) == 1
    mask = program[0]
    assert isinstance(mask, str), f"mask past bit 63 must not be a JSON number: {mask!r}"
    assert [i for i, bit in enumerate(mask) if bit == "1"] == lit
    assert len(app.avp.nodes["mixer_comp_a"]["src"]) == count


def test_geometry_is_resolved_by_two_compositors_without_filter_branches(native_boundary):
    app = application(native_boundary)
    nodes = app.avp.nodes
    assert not any(name.startswith(("mixer_cs_", "layout_preheat_")) for name in nodes)
    for slot in ("a", "b"):
        compositor = nodes[f"mixer_comp_{slot}"]
        assert "scale" not in compositor   # kernels load at init; the parameter is gone
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
        assert engine.nodes[f"mixer_cs_camera_{slot}"]["graph"] == graph + ",scale_cuda=format=nv12"
        assert engine.nodes[f"mixer_comp_{slot}"]["src"] == [f"mixer_camera_scaled_{slot}"]


def test_startup_failure_is_reported_before_native_cleanup(monkeypatch):
    import os
    import pyplumber.mixer.cli as mixer
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    monkeypatch.setenv("AVP_MIXER_STARTUP_FD", str(write_fd))
    def start():
        raise RuntimeError("aux_test_encoder: invalid preset")
    def stop():
        # A real native stop can hang here; the parent must already have the failure.
        assert os.read(read_fd, 4000) == b"aux_test_encoder: invalid preset"
        assert "AVP_MIXER_STARTUP_FD" not in os.environ
    app = SimpleNamespace(start=start, stop=stop)
    monkeypatch.setattr(mixer, "parse_args", lambda _: SimpleNamespace(webui_url=""))
    monkeypatch.setattr(mixer, "build_application", lambda _: app)
    try:
        with pytest.raises(RuntimeError, match="aux_test_encoder: invalid preset"):
            mixer.main([])
    finally:
        os.close(read_fd)


def test_successful_start_closes_private_setup_pipe(monkeypatch):
    import os
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    monkeypatch.setenv("AVP_MIXER_STARTUP_FD", str(write_fd))
    from pyplumber.mixer.application import _startup_result
    _startup_result()
    try:
        assert os.read(read_fd, 4000) == b""
        assert "AVP_MIXER_STARTUP_FD" not in os.environ
    finally:
        os.close(read_fd)
