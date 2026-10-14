"""Application API usable from another repository, without importing demo code."""
from dataclasses import replace
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from types import SimpleNamespace

from pyplumber.mixer import MixerOptions, SourceContext, build_application
from pyplumber.mixer import application as mixer_application
from pyplumber.mixer import config
from pyplumber.mixer.janus import JANUS_KEYFRAME_NODE
from test_graph import CONFIG, FakeAvp, fake_api
from test_prewarm import NativeEngine


def show(storage="cuda", **source_changes):
    camera = {**CONFIG["sources"][0], "decode_storage": storage, **source_changes}
    return config.parse({
        "canvas": CONFIG["canvas"],
        "sources": [camera, {**camera, "id": "idle", "path": "idle.mp4"}],
        "scenes": [{"id": "two", "items": CONFIG["scenes"][1]["items"][1:]}],
        "renditions": [{"id": "record", "target": "program.ts"}],
    })


def test_public_import_needs_neither_demo_nor_native_engine(tmp_path):
    source = Path(__file__).resolve().parents[2] / "pyplumber"
    shutil.copytree(source, tmp_path / "pyplumber", ignore=shutil.ignore_patterns("__pycache__"))
    root = str(tmp_path)
    script = f"""
import sys
sys.path.insert(0, {root!r})
from pyplumber.mixer import MixerOptions, MixerApplication, SourceContext, build_application
from pyplumber.mixer import run_application
from pyplumber.mixer.gui import MixerBridge, serve
from pyplumber.mixer.gui.web import PAGE, WALL, page
from pyplumber.mixer.gui.setup_runtime import SetupRuntime
assert callable(build_application) and callable(run_application)
assert b"AVPlumber mixer" in page({{}}) and b"AVPlumber outputs" in page({{}}, WALL)
assert PAGE.with_name("outputs.js").is_file() and PAGE.with_name("setup.js").is_file()
server = serve(MixerBridge("127.0.0.1", 7777, "ai"), "127.0.0.1", 0)
server.server_close()
assert not {{'_avplumber', 'pyplumber.core', 'pyplumber.node', 'mixer', 'layouts'}} & set(sys.modules)
assert not any(name.startswith('demos.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path, check=True, capture_output=True, text=True)


def test_a_declared_decoded_format_lets_the_fanout_tag_the_frames(native_boundary):
    """A decoded source whose colour and storage are declared and match the canvas has no colour
    filter: the fan-out stamps the tags. Undeclared storage keeps the filter, which converts."""
    def nodes_of(**source):
        api = fake_api()
        api.MixerGraphBuilder = native_boundary[1]   # the real graph builder on the recording engine
        app = build_application(show(**source), api=api)
        return {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    # The scene shows the camera twice, so its fan-out is the one that feeds both aliases.
    undeclared = nodes_of(color="sdr")
    assert undeclared["mixer_color_cam"]["graph"].endswith("scale_cuda=format=nv12")
    assert undeclared["mixer_color_alias_cam"]["src"] == undeclared["mixer_color_cam"]["dst"]
    assert "frame_color" not in undeclared["mixer_color_alias_cam"]
    declared = nodes_of(color="sdr", decoded_format="nv12")
    assert "mixer_color_cam" not in declared
    assert declared["mixer_color_alias_cam"]["src"] == undeclared["mixer_color_cam"]["src"] == "input_0_realtime"
    assert declared["mixer_color_alias_cam"]["frame_color"] == {
        "color_trc": "bt709", "color_primaries": "bt709", "colorspace": "bt709", "color_range": "tv"}
    # Ten-bit frames on this eight-bit canvas still need converting.
    assert "mixer_color_cam" in nodes_of(color="sdr", decoded_format="p010le")
    for bad in ({"decoded_format": "yuv420p"}, {"decoded_format": "nv12", "filter": "hflip_cuda", "filter_output_format": "nv12"}):
        with pytest.raises(config.ConfigError, match="decoded_format"):
            show(**bad)
    with pytest.raises(config.ConfigError, match="decoded_format"):
        config.parse({**CONFIG, "sources": [CONFIG["sources"][0], {**CONFIG["sources"][1], "decoded_format": "nv12"}]})


def test_config_renditions_do_not_need_cli_output_flags(tmp_path):
    cfg = show()
    app = build_application(cfg, api=fake_api())
    nodes = {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    assert nodes["record_output"]["url"] == "program.ts"
    assert nodes["record_fps"]["fps"] == "60/1"
    assert app.mixer.parameters["fps"] == (60, 1)
    assert json.loads(app.avp.commands_registered["mixer.settings"]("")) == cfg.settings()
    from pyplumber.mixer.cli import GraphOptions, build_application as build_cli
    path = tmp_path / "show.json"
    doc = {**CONFIG, "sources": CONFIG["sources"][:1], "scenes": CONFIG["scenes"][:1],
           "initial_scene": "full", "renditions": [{"id": "record", "target": "program.ts"}]}
    path.write_text(json.dumps(doc))
    assert build_cli(GraphOptions(config=str(path)), api=fake_api()).mixer



def test_config_janus_enables_keyframe_control_and_validates_transport():
    cfg = replace(show(), renditions=(config.Rendition("preview", "janus", 1920, 1080, 60, 4500),))
    app = build_application(cfg, api=fake_api())
    assert app.mixer.keyframe_node == JANUS_KEYFRAME_NODE
    with pytest.raises(ValueError, match="janus_host"):
        build_application(cfg, MixerOptions(janus_host=""), api=fake_api())


def test_missing_output_fails_before_creating_engine():
    api = fake_api()
    api.AVPlumber = lambda: pytest.fail("created engine for a show with no output")
    with pytest.raises(ValueError, match="rendition or output"):
        build_application(replace(show(), renditions=()), api=api)


@pytest.mark.parametrize("storage", ["cuda", "cuarray"])
def test_processing_is_once_per_camera_before_alias_fanout(storage):
    contexts = []
    def process(ctx):
        assert isinstance(ctx, SourceContext)
        contexts.append(ctx)
        edge = f"processed_{ctx.source.id}"
        ctx.avp.addNode(ctx.api.FilterVideo({
            "name": edge, "src": ctx.edge, "dst": edge,
            "graph": "null", "group": ctx.group,
        }))
        return edge

    app = build_application(show(storage), process_source=process, api=fake_api())
    assert [c.source.id for c in contexts] == ["cam", "idle"]
    assert [c.group for c in contexts] == list(app.input_groups) == ["input_0", "input_1"]
    assert all(c.hwaccel == "mixer_gpu" and c.fps == 60 for c in contexts)
    assert app.input_edges == ("processed_cam", "processed_idle")
    sources = dict(app.mixer.sources)
    assert sources["cam"]["pre_otm_edge"] == sources["cam#2"]["pre_otm_edge"] == "processed_cam"
    nodes = {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    assert nodes["decode_0"]["pixel_format"] == ("cuarray" if storage == "cuarray" else "?cuda")
    assert nodes["processed_cam"]["src"] == contexts[0].edge


def test_processing_follows_configured_transform_and_filter():
    contexts = []
    cfg = show(transform={"width": 1920, "height": 1080, "sw_format": "nv12"},
               filter="scale_cuda=format=nv12", filter_output_format="nv12")
    app = build_application(cfg, process_source=lambda ctx: contexts.append(ctx) or ctx.edge, api=fake_api())
    assert [ctx.edge for ctx in contexts] == ["input_0_filtered", "input_1_filtered"]
    assert app.input_edges == tuple(ctx.edge for ctx in contexts)


@pytest.mark.parametrize("failure", [RuntimeError("inference setup failed"), KeyboardInterrupt()])
def test_processing_failure_shuts_down_partial_engine(failure):
    engine = FakeAvp()
    api = fake_api()
    api.AVPlumber = lambda: engine
    def fail(ctx):
        raise failure
    with pytest.raises(type(failure)):
        build_application(show(), process_source=fail, api=api)
    assert engine.shut_down


@pytest.mark.parametrize("edge", [None, "", "  ", 42])
def test_invalid_processing_edge_shuts_down_partial_engine(edge):
    engine = FakeAvp()
    api = fake_api()
    api.AVPlumber = lambda: engine
    with pytest.raises(ValueError, match="process_source for 'cam'.*video edge"):
        build_application(show(), process_source=lambda ctx: edge, api=api)
    assert engine.shut_down


def test_going_on_air_releases_idle_browser_imports_before_the_outputs():
    from pyplumber.mixer.application import MixerApplication
    commands = []
    def expired(name):   # what an importer has released after the passes asked of it so far
        passes = commands.count(f"node.object.set {name} purge_idle 500")
        return (0, 14, 15, 15)[passes] if name == "input_3_to_cuda" else 0
    avp = SimpleNamespace(executeCommandsFromString=lambda text: commands.extend(text.splitlines()),
                          node=lambda name: SimpleNamespace(getObject=lambda key: {
                              "instance_cleanup_pending": 0, "expired": expired(name)}))
    app = MixerApplication(avp=avp, mixer=None, input_groups=(), input_edges=(), routed_inputs=False,
                           preheat_timeout_sec=1, held_outputs=("janus",),
                           browser_importers=("input_3_to_cuda", "input_7_to_cuda"))
    app._go_on_air()
    purged = [command for command in commands if " purge_idle " in command]
    # Three passes: the first two released buffers (14, then 1), the third found nothing left.
    assert purged == ["node.object.set input_3_to_cuda purge_idle 500", "node.object.set input_7_to_cuda purge_idle 500"] * 3
    # The releases stall draws: they have to be over while the outputs are still held.
    assert commands.index(purged[-1]) < commands.index("node.object.set janus_rtp_output hold false")
    # The counters start from zero before the output is let go: nothing on air is reset away.
    assert commands[-4:] == ["node.object.set mixer_comp_a reset_counters true", "node.object.set mixer_comp_b reset_counters true",
                             "node.object.set janus_rtp_output hold false", "node.object.set janus_force_keyframe trigger true"]


@pytest.mark.parametrize("missing_frame", [False, True])
def test_startup_waits_for_processed_frames_and_aux_shares_them(native_boundary, monkeypatch, missing_frame):
    class Engine(NativeEngine):
        def addNode(self, node, **kwargs):
            super().addNode(node)

        def registerControlCommand(self, *args):
            pass

    cfg = show()
    cfg = replace(cfg, aux_buses=config.parse_aux_buses([{
        "id": "cameras", "layout": {"preset": "source_pages"},
        "renditions": [{"id": "monitor", "port": 5012}],
    }], cfg))
    api = fake_api()
    api.AVPlumber = Engine
    api.MixerGraphBuilder = native_boundary[1]
    calls = []
    def process(ctx):
        calls.append(ctx.source.id)
        edge = f"processed_{ctx.source.id}"
        ctx.avp.addNode(ctx.api.FilterVideo({"name": edge, "src": ctx.edge, "dst": edge,
                                            "graph": "null", "group": ctx.group}))
        return edge

    app = build_application(cfg, MixerOptions(preheat_timeout_sec=.01), api=api, process_source=process)
    assert calls == ["cam", "idle"]
    # One consumer of the processed camera feeds the shared conversion/fan-out;
    # adding an alias and a source-page AUX bus must not duplicate inference.
    readers = [node for node in app.avp.nodes.values() if node.get("src") == "processed_cam"]
    assert len(readers) == 1
    if missing_frame:
        app.avp.missing.add("processed_cam")
        ticks = iter(range(10000))
        monkeypatch.setattr("time.monotonic", lambda: next(ticks))
    try:
        if missing_frame:
            with pytest.raises(RuntimeError, match="input readiness.*processed_cam"):
                app.start()
            assert not app.avp.ready
        else:
            app.start()
            assert app.avp.ready
            events = app.avp.events
            assert events.index("start input_0") < events.index("inspect processed_cam") < events.index("READY")
            # Every Janus output is built held and released, with a key frame asked for, only after
            # the last bus has produced encoded output; the compositors then count from zero.
            assert app.held_outputs == ()   # this show records its program; only its bus goes to Janus
            outputs = ["aux_cameras_rtp_output"]
            assert all(app.avp.nodes[name]["hold"] is True for name in outputs)
            released = [event for event in events if event.endswith(" hold false")]
            assert released == [f"node.object.set {name} hold false" for name in outputs]
            assert [event for event in events if event.endswith("_force_keyframe trigger true")] == [
                "node.object.set aux_cameras_force_keyframe trigger true"]
            assert [event.split()[1] for event in events if event.endswith(" reset_counters true")] == [
                "mixer_comp_a", "mixer_comp_b", "aux_cameras_comp"]
            last_inspected = max(i for i, event in enumerate(events) if event == "inspect aux_cameras_encoded")
            assert last_inspected < events.index(released[0]) < events.index("READY")
    finally:
        app.stop()
    assert app.avp.shutdown_complete


def test_shared_setup_accepts_an_explicit_show_and_custom_command(tmp_path, monkeypatch):
    from pyplumber.mixer.gui.setup_runtime import SetupRuntime
    runtime = SetupRuntime(tmp_path, tmp_path / "settings.json", None, {},
                           mixer_command=(sys.executable, "-m", "ai_mixer"))
    monkeypatch.setattr(runtime, "_start_recovering", lambda *args: None)
    doc = {**CONFIG, "renditions": [{"id": "record", "target": "program.ts"}]}
    runtime.apply(doc)
    runtime.worker.join(3)
    assert runtime.status()["phase"] == "running", runtime.status()
    assert json.loads(runtime.config_path.read_text()) == doc
    assert runtime.mixer_command[-1] == "ai_mixer"
    runtime.close()


# ---------------------------------------------------------------------------------------------
# Going on air: the outputs stay held until the show has been stable (no missed deadline, no
# output drop, every bus drawing) for start_stable_sec. The clock below is the only time these
# tests know: sleep() advances it, so a wait of a minute takes none.

class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def time(self):
        return 1_700_000_000.0 + self.now


class CountingEngine:
    """An engine whose compositors publish counters: *counters* maps a node to a function of the
    clock giving (frames, missed_deadlines, output_drops), as node.object.get status would after
    the frame count's publication every 60 frames. Every command is recorded with the time."""

    def __init__(self, clock, counters):
        self.clock, self.counters, self.commands = clock, counters, []

    def executeCommandsFromString(self, text):
        self.commands.extend((self.clock.now, line) for line in text.splitlines())

    def node(self, name):
        if name not in self.counters:
            return None
        def status(key):
            assert key == "status"
            frames, missed, drops = self.counters[name](self.clock.now)
            return {"output_drops": drops, "playout": {"frames": frames, "missed_deadlines": missed}}
        return SimpleNamespace(getObject=status)

    def registerControlCommand(self, name, handler, _payload, _quiet=False):
        pass

    def at(self, suffix):
        """When the commands ending in *suffix* were given."""
        return [when for when, line in self.commands if line.endswith(suffix)]


def published(fps, every=60):
    """A compositor's frame count as it is published: every 60 frames drawn at *fps*."""
    return lambda now: int(now * fps) // every * every


def idle(_now):
    return 0, 0, 0


def go_on_air(monkeypatch, counters, buses=("aux_1",), **options):
    clock = Clock()
    monkeypatch.setattr(mixer_application, "time", clock)
    engine = CountingEngine(clock, counters)
    app = mixer_application.MixerApplication(
        avp=engine, mixer=None, input_groups=(), input_edges=(), routed_inputs=False, preheat_timeout_sec=1,
        held_outputs=("janus",), aux_buses=[SimpleNamespace(prefix=prefix) for prefix in buses], **options)
    app._go_on_air()
    return app, engine, clock


def test_outputs_are_released_after_a_stable_window_with_the_counters_reset_first(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="mixer")
    program = published(30)
    bus = published(15)
    app, engine, clock = go_on_air(monkeypatch, {
        "mixer_comp_a": lambda now: (program(now), 0, 0),
        "mixer_comp_b": idle,                                                  # the PVW slot: no scene loaded, never ticks
        "aux_1_comp": lambda now: (bus(now), (now >= 2) + (now >= 5), 0),      # late twice while the show comes up
    })
    # The default window is ten seconds; the last miss was at 5 s, so nothing was let go before 15 s.
    assert engine.at(" hold false") == [15.0, 15.0]
    assert clock.now == 15.0
    lines = [line for _, line in engine.commands]
    # The counters start from zero before the outputs are let go, which start on a key frame.
    assert lines == ["node.object.set mixer_comp_a reset_counters true", "node.object.set mixer_comp_b reset_counters true",
                     "node.object.set aux_1_comp reset_counters true",
                     "node.object.set janus_rtp_output hold false", "node.object.set janus_force_keyframe trigger true",
                     "node.object.set aux_1_rtp_output hold false", "node.object.set aux_1_force_keyframe trigger true"]
    # What `mixer.startup` answers with: the start's misses, kept apart from the counters on air.
    assert app.startup["settled"] is True and app.startup["reason"] == ""
    assert app.startup["waited_sec"] == 15.0 and app.startup["outputs"] == 2
    assert app.startup["warm_up"] == {"missed_deadlines": 2, "output_drops": 0}
    assert app.startup["counters"]["aux_1_comp"] == {"missed_deadlines": 2, "output_drops": 0, "frames": 180}
    assert app.startup["stable_sec"] == 10.0 and app.startup["released_at"] == 1_700_000_015.0
    assert json.loads(json.dumps(app.startup)) == app.startup
    on_air = [r for r in caplog.records if r.getMessage().startswith("On air")]
    assert len(on_air) == 1 and on_air[0].levelno == logging.INFO
    assert "after 15.0 s: 2 outputs released; stable for 10 s; the start's 2 missed deadlines" in on_air[0].getMessage()


def test_a_show_that_never_settles_is_released_at_the_timeout_and_the_log_says_which_bus(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="mixer")
    app, engine, clock = go_on_air(monkeypatch, {
        "mixer_comp_a": lambda now: (published(30)(now), 0, 0),
        "mixer_comp_b": idle,
        "aux_1_comp": lambda now: (published(15)(now), 0, 0),
        "aux_2_comp": lambda now: (published(15)(now), int(now), int(now) // 2),   # a miss a second, a drop every other
    }, buses=("aux_1", "aux_2"), start_stable_sec=5, start_stable_timeout_sec=20)
    assert engine.at(" hold false") == [20.0, 20.0, 20.0]     # released anyway, after the timeout
    assert app.startup["settled"] is False and app.startup["waited_sec"] == 20.0
    # What the bus counted since the window last began, at 19 s: the miss of 20 s and the drop of 20 s.
    assert app.startup["reason"] == "aux_2_comp missed 1 deadlines and dropped 1 frames"
    assert app.startup["warm_up"] == {"missed_deadlines": 20, "output_drops": 10}
    warning = next(r for r in caplog.records if r.getMessage().startswith("On air"))
    assert warning.levelno == logging.WARNING
    assert "after 20.0 s without a stable 5 s (aux_2_comp missed 1 deadlines and dropped 1 frames): 3 outputs released" \
        in warning.getMessage()
    assert "the show may keep counting" in warning.getMessage()


@pytest.mark.parametrize("stuck", ["program", "bus"])
def test_a_compositor_that_draws_nothing_holds_the_outputs_until_the_timeout(monkeypatch, stuck):
    counters = {
        "mixer_comp_a": idle if stuck == "program" else lambda now: (published(30)(now), 0, 0),
        "mixer_comp_b": idle,
        "aux_1_comp": idle if stuck == "bus" else lambda now: (published(15)(now), 0, 0),
    }
    app, engine, clock = go_on_air(monkeypatch, counters, start_stable_sec=3, start_stable_timeout_sec=12)
    # No miss anywhere (a compositor that cannot tick misses nothing), yet nothing drew: not stable.
    assert engine.at(" hold false") == [12.0, 12.0]
    assert app.startup["settled"] is False
    assert app.startup["reason"] == {"program": "mixer_comp_a or mixer_comp_b drew no frame in 12 s",
                                     "bus": "aux_1_comp drew no frame in 12 s"}[stuck]


def test_a_window_shorter_than_a_frame_count_publication_waits_for_the_count_not_the_timeout(monkeypatch):
    # A half-rate bus publishes its frame count every 60 frames, 4 s at 15 fps. With a 2 s window
    # the bus has drawn nothing the gate can see at 2 s; the first publication lets the show go.
    app, engine, clock = go_on_air(monkeypatch, {
        "mixer_comp_a": lambda now: (published(30)(now), 0, 0),
        "mixer_comp_b": idle,
        "aux_1_comp": lambda now: (published(15)(now), 0, 0),
    }, start_stable_sec=2, start_stable_timeout_sec=60)
    assert engine.at(" hold false") == [4.0, 4.0]
    assert app.startup["settled"] is True and app.startup["waited_sec"] == 4.0


def test_a_miss_restarts_the_window_from_the_poll_that_saw_it(monkeypatch):
    # A program miss at 1 s and a bus drop at 5 s: the window that began at 1 s is cut short at
    # 5 s, and the one from 5 s ends at 11 s; a drop counts the same as a missed deadline.
    app, engine, clock = go_on_air(monkeypatch, {
        "mixer_comp_a": lambda now: (published(30)(now), int(now >= 1), 0),
        "mixer_comp_b": idle,
        "aux_1_comp": lambda now: (published(15)(now), 0, int(now >= 5)),
    }, start_stable_sec=6, start_stable_timeout_sec=60)
    assert engine.at(" hold false") == [11.0, 11.0]
    assert app.startup["warm_up"] == {"missed_deadlines": 1, "output_drops": 1}


def test_an_engine_without_playout_counters_has_nothing_to_wait_for(monkeypatch):
    # The fakes of the other tests report no `playout`: the gate releases at once, as before.
    app, engine, clock = go_on_air(monkeypatch, {})
    assert engine.at(" hold false") == [0.0, 0.0]
    assert app.startup["settled"] is True and app.startup["counters"] == {}
    assert app.startup["warm_up"] == {"missed_deadlines": 0, "output_drops": 0}


def test_browser_measurement_baseline_is_saved_at_output_release(monkeypatch, caplog):
    status = {"windows": [{"id": "web_1", "stats": {
        "paintCount": 900, "txFrameCount": 880, "droppedFrames": 20,
        "droppedReasons": {"no_consumer": 20}}}]}
    requests = []

    def rest(base_url, method, path, body=None, timeout=60):
        requests.append((base_url, method, path, timeout))
        if isinstance(status, Exception):
            raise status
        return status

    monkeypatch.setattr(mixer_application, "rest_request", rest)
    app, engine, _ = go_on_air(monkeypatch, {}, browser_windows=("web_1",), dmabuf_rest="http://browser:9009")
    assert requests == [("http://browser:9009", "GET", "/status", 5)]
    assert app.startup["browser_windows"] == {"web_1": status["windows"][0]["stats"]}
    assert engine.at(" hold false") == [0.0, 0.0]

    status = ConnectionRefusedError(111, "refused")
    app, engine, _ = go_on_air(monkeypatch, {}, browser_windows=("web_1",), dmabuf_rest="http://browser:9009")
    assert app.startup["browser_windows"] is None
    assert engine.at(" hold false") == [0.0, 0.0]
    assert "Browser counters at on air unavailable" in caplog.text

    requests.clear()
    app, _, _ = go_on_air(monkeypatch, {})
    assert requests == [] and app.startup["browser_windows"] is None


def test_start_stable_options_default_validate_and_reach_the_application():
    from pyplumber.mixer.cli import GraphOptions, parse_args
    assert (MixerOptions().start_stable_sec, MixerOptions().start_stable_timeout_sec) == (10.0, 60.0)
    assert build_application(show(), api=fake_api()).start_stable_sec == 10.0
    app = build_application(show(), MixerOptions(start_stable_sec=3, start_stable_timeout_sec=30), api=fake_api())
    assert (app.start_stable_sec, app.start_stable_timeout_sec) == (3, 30)
    for bad in ({"start_stable_sec": -1}, {"start_stable_sec": 5, "start_stable_timeout_sec": 4},
                {"start_stable_sec": 0, "start_stable_timeout_sec": 0}):
        with pytest.raises(ValueError, match="start_stable"):
            MixerOptions(**bad).validate()
    MixerOptions(start_stable_sec=0, start_stable_timeout_sec=1).validate()   # 0: no clean stretch asked, only a frame drawn
    parsed = parse_args(["--input", "a.mp4", "--output", "p.ts", "--start-stable-sec", "3", "--start-stable-timeout-sec", "30"])
    assert isinstance(parsed, GraphOptions) and (parsed.start_stable_sec, parsed.start_stable_timeout_sec) == (3.0, 30.0)
    assert parse_args(["--input", "a.mp4", "--output", "p.ts"]).start_stable_sec == 10.0


def test_the_startup_report_is_a_control_command_once_the_show_is_on_air(native_boundary):
    from test_prewarm import application
    app = application(native_boundary)
    app.start()
    report = json.loads(app.avp.commands_registered["mixer.startup"](""))
    # This show records its program and has no bus: nothing held, nothing counted, released at once.
    assert report["settled"] is True and report["outputs"] == 0
    assert report["warm_up"] == {"missed_deadlines": 0, "output_drops": 0}
    assert report["stable_sec"] == 10.0 and report["timeout_sec"] == 60.0
