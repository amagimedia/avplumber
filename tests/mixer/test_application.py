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


def test_going_on_air_releases_idle_browser_imports_before_the_encoders_and_the_outputs():
    from pyplumber.mixer.application import MixerApplication
    commands = []
    def expired(name):   # what an importer has released after the passes asked of it so far
        passes = commands.count(f"node.object.set {name} purge_idle 500")
        return (0, 14, 15, 15)[passes] if name == "input_3_to_cuda" else 0
    avp = SimpleNamespace(executeCommandsFromString=lambda text: commands.extend(text.splitlines()),
                          node=lambda name: SimpleNamespace(isWorking=True, getObject=lambda key: {
                              "instance_cleanup_pending": 0, "expired": expired(name)}),
                          group=lambda name: SimpleNamespace(startNodes=lambda: commands.append("start " + name)),
                          getEdge=lambda name, data_type=None: SimpleNamespace(enqueued_total=1))
    app = MixerApplication(avp=avp, mixer=None, input_groups=(), input_edges=(), routed_inputs=False,
                           preheat_timeout_sec=1, held_outputs=("janus",),
                           browser_importers=("input_3_to_cuda", "input_7_to_cuda"))
    app._go_on_air()
    purged = [command for command in commands if " purge_idle " in command]
    # Three passes: the first two released buffers (14, then 1), the third found nothing left.
    assert purged == ["node.object.set input_3_to_cuda purge_idle 500", "node.object.set input_7_to_cuda purge_idle 500"] * 3
    # The releases stall draws: they have to be over before the encoder opens, let alone the output goes.
    assert commands.index(purged[-1]) < commands.index("start janus_encode")
    # The encoder's group starts and its gate opens; the counters start from zero before the output
    # is let go: nothing on air is reset away.
    assert commands[-6:] == ["start janus_encode", "node.object.set janus_gate outputs 1",
                             "node.object.set mixer_comp_a reset_counters true", "node.object.set mixer_comp_b reset_counters true",
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
            # Every Janus chain is built behind a closed gate with its encoder's group unstarted and
            # its output held. The bus starts without its encoder and is waited for up to the gate;
            # the encoder's group starts and the gate opens once the show is stable; the output is
            # released, with a key frame asked for, after the chain's first packet and the
            # compositors counting from zero.
            assert app.held_outputs == ()   # this show records its program; only its bus goes to Janus
            nodes = app.avp.nodes
            assert nodes["aux_cameras_rtp_output"]["hold"] is True
            assert (nodes["aux_cameras_gate"]["type"], nodes["aux_cameras_gate"]["outputs"], nodes["aux_cameras_gate"]["group"]) \
                == ("one_to_many", 0, "aux_cameras")
            assert nodes["aux_cameras_gate"]["src"] == nodes["aux_cameras_sdr"]["dst"] == "aux_cameras_sdr"
            assert nodes["aux_cameras_fps"]["src"] == nodes["aux_cameras_gate"]["dst"][0] == "aux_cameras_gated"
            assert {nodes[f"aux_cameras_{n}"]["group"] for n in ("fps", "force_keyframe", "format", "encoder", "repeat_headers", "mux", "rtp_output")} \
                == {"aux_cameras_encode"}
            assert nodes["record_encoder"]["group"] == "output"   # a recorded program is not held: it encodes from the start
            order = ["start aux_cameras", "inspect aux_cameras_sdr", "start aux_cameras_encode",
                     "node.object.set aux_cameras_gate outputs 1", "inspect aux_cameras_encoded",
                     "node.object.set mixer_comp_a reset_counters true", "node.object.set mixer_comp_b reset_counters true",
                     "node.object.set aux_cameras_comp reset_counters true", "node.object.set aux_cameras_rtp_output hold false",
                     "node.object.set aux_cameras_force_keyframe trigger true", "READY"]
            assert [event for event in events if event in order] == order
            assert "inspect aux_cameras_encoded" not in events[:events.index("start aux_cameras_encode")]
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
    the frame count's publication every 60 frames. Every command is recorded with the time, and so
    is a group start, as "start <group>". An encoder runs as soon as its group has started and its
    chain has a packet at once."""

    def __init__(self, clock, counters):
        self.clock, self.counters, self.commands, self.inspected = clock, counters, [], []

    def executeCommandsFromString(self, text):
        self.commands.extend((self.clock.now, line) for line in text.splitlines())

    def group(self, name):
        return SimpleNamespace(startNodes=lambda: self.commands.append((self.clock.now, "start " + name)))

    def getEdge(self, name, data_type=None):
        assert data_type == "packet" and name.endswith("_encoded")
        assert any(line == f"start {name[:-len('_encoded')]}_encode" for _, line in self.commands), "waited for a packet before the group started"
        self.inspected.append((self.clock.now, name))
        return SimpleNamespace(enqueued_total=1)

    def node(self, name):
        if name.endswith("_encoder"):
            started = any(line == f"start {name[:-len('_encoder')]}_encode" for _, line in self.commands)
            return SimpleNamespace(isWorking=started)
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


def test_encoders_start_after_a_stable_window_and_the_outputs_are_released_after_another_with_the_counters_reset_first(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="mixer")
    program = published(30)
    bus = published(15)
    app, engine, clock = go_on_air(monkeypatch, {
        "mixer_comp_a": lambda now: (program(now), 0, 0),
        "mixer_comp_b": idle,                                                  # the PVW slot: no scene loaded, never ticks
        "aux_1_comp": lambda now: (bus(now), (now >= 2) + (now >= 5), 0),      # late twice while the show comes up
    })
    # The default window is ten seconds; the last miss was at 5 s, so no encoder started before 15 s.
    # Each encoder's group starts and its gate opens once it runs, one after another; the outputs
    # are let go after the default 3 s more without a miss, the counters reset first, on a key frame.
    assert engine.at("_encode") == [15.0, 15.0] and engine.at(" hold false") == [18.0, 18.0]
    assert clock.now == 18.0
    lines = [line for _, line in engine.commands]
    assert lines == ["start janus_encode", "node.object.set janus_gate outputs 1",
                     "start aux_1_encode", "node.object.set aux_1_gate outputs 1",
                     "node.object.set mixer_comp_a reset_counters true", "node.object.set mixer_comp_b reset_counters true",
                     "node.object.set aux_1_comp reset_counters true",
                     "node.object.set janus_rtp_output hold false", "node.object.set janus_force_keyframe trigger true",
                     "node.object.set aux_1_rtp_output hold false", "node.object.set aux_1_force_keyframe trigger true"]
    assert engine.inspected == [(15.0, "janus_encoded"), (15.0, "aux_1_encoded")]   # each chain's first packet, before the next starts
    # What `mixer.startup` answers with: the start's misses, kept apart from the counters on air.
    assert app.startup["settled"] is True and app.startup["reason"] == ""
    assert app.startup["waited_sec"] == 15.0 and app.startup["outputs"] == 2 and app.startup["on_air_sec"] == 18.0
    assert app.startup["warm_up"] == {"missed_deadlines": 2, "output_drops": 0, "repeats": 0, "discarded": 0, "browser_dropped": None}
    assert app.startup["counters"]["aux_1_comp"] == {"missed_deadlines": 2, "output_drops": 0, "frames": 180, "repeats": 0, "discarded": 0}
    assert app.startup["encoders"] == {"count": 2, "opened_sec": 0.0, "first_packet_ms": {"janus": 0, "aux_1": 0},
                                       "settled": True, "waited_sec": 3.0, "reason": "",
                                       "counters": {"mixer_comp_a": {"missed_deadlines": 0, "output_drops": 0, "frames": 540, "repeats": 0, "discarded": 0},
                                                    "mixer_comp_b": {"missed_deadlines": 0, "output_drops": 0, "frames": 0, "repeats": 0, "discarded": 0},
                                                    "aux_1_comp": {"missed_deadlines": 2, "output_drops": 0, "frames": 240, "repeats": 0, "discarded": 0}}}
    assert app.startup["stable_sec"] == 10.0 and app.startup["encoder_settle_sec"] == 3.0
    assert app.startup["released_at"] == 1_700_000_018.0 and app.startup["browser_windows"] is None
    assert json.loads(json.dumps(app.startup)) == app.startup
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert "Stable for 10 s, 15.0 s after the compositors started: starting 2 encoders" in messages
    assert "Encoders open: 2 in 0.0 s, first packets 0-0 ms after each start; settling for 3 s" in messages
    on_air = [r for r in caplog.records if r.getMessage().startswith("On air")]
    assert len(on_air) == 1 and on_air[0].levelno == logging.INFO
    assert on_air[0].getMessage() == ("On air after 18.0 s: 2 outputs released 3.0 s after their encoders opened, stable for 10 s "
                                      "before and 3 s after; the start's 2 missed deadlines, 0 output drops, 0 repeats and 0 discards are not counted")


def test_a_show_that_never_settles_is_released_at_the_timeout_and_the_log_says_which_bus(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="mixer")
    app, engine, clock = go_on_air(monkeypatch, {
        "mixer_comp_a": lambda now: (published(30)(now), 0, 0),
        "mixer_comp_b": idle,
        "aux_1_comp": lambda now: (published(15)(now), 0, 0),
        "aux_2_comp": lambda now: (published(15)(now), int(now), int(now) // 2),   # a miss a second, a drop every other
    }, buses=("aux_1", "aux_2"), start_stable_sec=5, start_stable_timeout_sec=20)
    # The encoders start anyway after the timeout; the window after them never comes either, and
    # its own timeout (20 s) releases the outputs.
    assert engine.at("_encode") == [20.0, 20.0, 20.0] and engine.at(" hold false") == [40.0, 40.0, 40.0]
    assert app.startup["settled"] is False and app.startup["waited_sec"] == 20.0
    # What the bus counted since the window last began, at 19 s: the miss of 20 s and the drop of 20 s.
    assert app.startup["reason"] == "aux_2_comp missed 1 deadlines and dropped 1 frames"
    assert app.startup["encoders"]["settled"] is False and app.startup["encoders"]["waited_sec"] == 20.0
    assert app.startup["warm_up"] == {"missed_deadlines": 40, "output_drops": 20, "repeats": 0, "discarded": 0, "browser_dropped": None}
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "Not stable for 5 s in 20.0 s (aux_2_comp missed 1 deadlines and dropped 1 frames): starting 3 encoders anyway" in warnings
    on_air = next(message for message in warnings if message.startswith("On air"))
    assert "after 40.0 s without a stable 5 s before the encoders opened (aux_2_comp missed 1 deadlines and dropped 1 frames): " \
        "3 outputs released" in on_air
    assert "the start's 40 missed deadlines, 20 output drops, 0 repeats and 0 discards are not counted, and the show may keep counting" in on_air


@pytest.mark.parametrize("stuck", ["program", "bus"])
def test_a_compositor_that_draws_nothing_holds_the_outputs_until_the_timeout(monkeypatch, stuck):
    counters = {
        "mixer_comp_a": idle if stuck == "program" else lambda now: (published(30)(now), 0, 0),
        "mixer_comp_b": idle,
        "aux_1_comp": idle if stuck == "bus" else lambda now: (published(15)(now), 0, 0),
    }
    app, engine, clock = go_on_air(monkeypatch, counters, start_stable_sec=3, start_stable_timeout_sec=12)
    # No miss anywhere (a compositor that cannot tick misses nothing), yet nothing drew: not stable,
    # before the encoders (12 s) or after them (20 s more).
    assert engine.at("_encode") == [12.0, 12.0] and engine.at(" hold false") == [32.0, 32.0]
    assert app.startup["settled"] is False and app.startup["encoders"]["settled"] is False
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
    assert engine.at("_encode") == [4.0, 4.0]
    assert app.startup["settled"] is True and app.startup["waited_sec"] == 4.0
    # The same after the encoders: the 3 s window is up at 7 s, the bus's next count comes at 8 s.
    assert engine.at(" hold false") == [8.0, 8.0] and app.startup["encoders"]["waited_sec"] == 4.0


def test_a_miss_restarts_the_window_from_the_poll_that_saw_it(monkeypatch):
    # A program miss at 1 s and a bus drop at 5 s: the window that began at 1 s is cut short at
    # 5 s, and the one from 5 s ends at 11 s; a drop counts the same as a missed deadline.
    app, engine, clock = go_on_air(monkeypatch, {
        "mixer_comp_a": lambda now: (published(30)(now), int(now >= 1), 0),
        "mixer_comp_b": idle,
        "aux_1_comp": lambda now: (published(15)(now), 0, int(now >= 5)),
    }, start_stable_sec=6, start_stable_timeout_sec=60)
    assert engine.at("_encode") == [11.0, 11.0] and engine.at(" hold false") == [14.0, 14.0]
    assert app.startup["warm_up"] == {"missed_deadlines": 1, "output_drops": 1, "repeats": 0, "discarded": 0, "browser_dropped": None}


def test_an_engine_without_playout_counters_has_nothing_to_wait_for(monkeypatch):
    # The fakes of the other tests report no `playout`: the gate releases at once, as before.
    app, engine, clock = go_on_air(monkeypatch, {})
    assert engine.at("_encode") == [0.0, 0.0] and engine.at(" hold false") == [0.0, 0.0]
    assert app.startup["settled"] is True and app.startup["counters"] == {}
    assert app.startup["warm_up"] == {"missed_deadlines": 0, "output_drops": 0, "repeats": 0, "discarded": 0, "browser_dropped": None}


def test_a_short_encoder_window_is_the_first_quiet_poll_and_the_option_reaches_the_application(monkeypatch):
    app, engine, clock = go_on_air(monkeypatch, {
        "mixer_comp_a": lambda now: (published(30)(now), int(now >= 1), 0),
        "mixer_comp_b": idle,
        "aux_1_comp": lambda now: (published(15)(now), 0, 0),
    }, start_stable_sec=4, start_stable_timeout_sec=60, encoder_settle_sec=0)
    # The miss at 1 s: stable from 5 s. With no window asked after the encoders, the first poll
    # without a miss and with a frame drawn since (the bus's next count, 8 s) lets the outputs go.
    assert engine.at("_encode") == [5.0, 5.0] and engine.at(" hold false") == [8.0, 8.0]
    assert app.startup["encoder_settle_sec"] == 0 and app.startup["encoders"]["settled"] is True
    assert build_application(show(), MixerOptions(start_encoder_settle_sec=1.5), api=fake_api()).encoder_settle_sec == 1.5
    for bad in ({"start_encoder_settle_sec": -1}, {"start_encoder_settle_sec": 21}):
        with pytest.raises(ValueError, match="start_encoder_settle_sec"):
            MixerOptions(**bad).validate()


def test_the_browser_counters_are_read_as_the_show_goes_on_air(monkeypatch, caplog):
    """The browser service counts from before the mixer exists; the web UI's drop window starts
    from what it counted at this moment (telemetry.BrowserCpuStats), as the compositors' counters
    start from zero. A service that does not answer leaves None, logged, and the start goes on."""
    caplog.set_level(logging.INFO, logger="mixer")
    status = {"windows": [{"id": "web_1", "stats": {"paintCount": 900, "txFrameCount": 880, "droppedFrames": 20,
                                                   "droppedReasons": {"no_consumer": 20}}},
                          {"id": "web_2", "stats": {"paintCount": 300, "txFrameCount": 299, "droppedFrames": 1, "droppedReasons": {}}}]}
    requests = []
    def rest(base_url, method, path, body=None, timeout=60):
        requests.append((base_url, method, path, timeout))
        if isinstance(status, Exception):
            raise status
        return status
    monkeypatch.setattr(mixer_application, "rest_request", rest)
    app, engine, clock = go_on_air(monkeypatch, {}, browser_windows=("web_1", "web_2"), dmabuf_rest="http://browser:9009")
    assert requests == [("http://browser:9009", "GET", "/status", 5)]
    assert app.startup["browser_windows"] == {"web_1": status["windows"][0]["stats"], "web_2": status["windows"][1]["stats"]}
    assert app.startup["warm_up"]["browser_dropped"] == 21
    status = ConnectionRefusedError(111, "refused")
    app, engine, clock = go_on_air(monkeypatch, {}, browser_windows=("web_1",), dmabuf_rest="http://browser:9009")
    assert app.startup["browser_windows"] is None and app.startup["warm_up"]["browser_dropped"] is None
    assert engine.at(" hold false") == [0.0, 0.0]   # the start went on
    assert any(r.levelno == logging.WARNING and r.getMessage().startswith("Browser counters at on air unavailable")
               for r in caplog.records)
    # A show without browsers asks nothing.
    requests.clear()
    go_on_air(monkeypatch, {}, dmabuf_rest="http://browser:9009")
    assert requests == []


def test_inputs_start_in_stages_browsers_first_then_raw_readers_then_decoders(monkeypatch, caplog):
    """Each stage has its frames before the next starts: the browser imports register buffers with
    CUDA while nothing else uses the device, the raw readers page-lock their files, the decoders
    open last. A show from --input alone is all decoders: one stage, as before."""
    from pyplumber.mixer.application import MixerApplication
    caplog.set_level(logging.INFO, logger="mixer")
    events = []
    refreshed = []
    monkeypatch.setattr(mixer_application, "refresh_windows", lambda rest, ids: refreshed.append((rest, ids, list(events))))
    avp = SimpleNamespace(group=lambda name: SimpleNamespace(startNodes=lambda: events.append("start " + name)),
                          getEdge=lambda name, data_type=None: events.append("inspect " + name) or SimpleNamespace(enqueued_total=1))
    kinds = ("video", "browser", "nv12", "video", "browser", "v210", "p010")
    app = MixerApplication(avp=avp, mixer=None, routed_inputs=False, preheat_timeout_sec=1,
                           input_groups=tuple(f"input_{i}" for i in range(7)), input_edges=tuple(f"edge_{i}" for i in range(7)),
                           input_kinds=kinds, browser_windows=("web_1", "web_2"), dmabuf_rest="http://browser:9009")
    app._start_inputs()
    assert events == ["start input_1", "start input_4", "inspect edge_1", "inspect edge_4",
                      "start input_2", "start input_5", "start input_6", "inspect edge_2", "inspect edge_5", "inspect edge_6",
                      "start input_0", "start input_3", "inspect edge_0", "inspect edge_3"]
    # The pages are reloaded once their chains are connected, before their frames are waited for.
    assert refreshed == [("http://browser:9009", ["web_1", "web_2"], ["start input_1", "start input_4"])]
    messages = [r.getMessage() for r in caplog.records]
    assert [m.split(" in ")[0] for m in messages] == ["Inputs ready: 2 browser", "Inputs ready: 3 raw", "Inputs ready: 2 decoded", "Inputs ready: 7"]
    # Without kinds (a --input show), everything is one stage of decoders.
    events.clear()
    MixerApplication(avp=avp, mixer=None, routed_inputs=False, preheat_timeout_sec=1,
                     input_groups=("input_0", "input_1"), input_edges=("edge_0", "edge_1"))._start_inputs()
    assert events == ["start input_0", "start input_1", "inspect edge_0", "inspect edge_1"]
    # A built show names its kinds in source order.
    cfg = show()
    assert build_application(cfg, api=fake_api()).input_kinds == tuple(s.kind for s in cfg.sources) == ("video", "video")


def test_start_stable_options_default_validate_and_reach_the_application():
    from pyplumber.mixer.cli import GraphOptions, parse_args
    assert (MixerOptions().start_stable_sec, MixerOptions().start_stable_timeout_sec) == (10.0, 60.0)
    assert MixerOptions().start_encoder_settle_sec == 3.0
    assert build_application(show(), api=fake_api()).start_stable_sec == 10.0
    app = build_application(show(), MixerOptions(start_stable_sec=3, start_stable_timeout_sec=30), api=fake_api())
    assert (app.start_stable_sec, app.start_stable_timeout_sec) == (3, 30)
    for bad in ({"start_stable_sec": -1}, {"start_stable_sec": 5, "start_stable_timeout_sec": 4},
                {"start_stable_sec": 0, "start_stable_timeout_sec": 0}):
        with pytest.raises(ValueError, match="start_stable"):
            MixerOptions(**bad).validate()
    MixerOptions(start_stable_sec=0, start_stable_timeout_sec=1).validate()   # 0: no clean stretch asked, only a frame drawn
    parsed = parse_args(["--input", "a.mp4", "--output", "p.ts", "--start-stable-sec", "3", "--start-stable-timeout-sec", "30",
                         "--start-encoder-settle-sec", "2"])
    assert isinstance(parsed, GraphOptions) and (parsed.start_stable_sec, parsed.start_stable_timeout_sec) == (3.0, 30.0)
    assert parsed.start_encoder_settle_sec == 2.0
    assert parse_args(["--input", "a.mp4", "--output", "p.ts"]).start_stable_sec == 10.0


def test_the_startup_report_is_a_control_command_once_the_show_is_on_air(native_boundary):
    from test_prewarm import application
    app = application(native_boundary)
    app.start()
    report = json.loads(app.avp.commands_registered["mixer.startup"](""))
    # This show records its program and has no bus: nothing held, nothing counted, released at once.
    assert report["settled"] is True and report["outputs"] == 0 and report["encoders"]["count"] == 0
    assert report["warm_up"] == {"missed_deadlines": 0, "output_drops": 0, "repeats": 0, "discarded": 0, "browser_dropped": None}
    assert report["stable_sec"] == 10.0 and report["timeout_sec"] == 60.0 and report["encoder_settle_sec"] == 3.0
