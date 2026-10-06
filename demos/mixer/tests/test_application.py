"""Application API usable from another repository, without importing demo code."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pyplumber.mixer import MixerOptions, SourceContext, build_application
from pyplumber.mixer import config
from pyplumber.mixer.janus import JANUS_KEYFRAME_NODE
from test_graph import CONFIG, FakeAvp, fake_api
from test_prewarm import NativeEngine, native_boundary


def show(storage="cuda", **source_changes):
    camera = {**CONFIG["sources"][0], "decode_storage": storage, **source_changes}
    return config.parse({
        "canvas": CONFIG["canvas"],
        "sources": [camera, {**camera, "id": "idle", "path": "idle.mp4"}],
        "scenes": [{"id": "two", "items": CONFIG["scenes"][1]["items"][1:]}],
        "renditions": [{"id": "record", "target": "program.ts"}],
    })


def test_public_import_needs_neither_demo_nor_native_engine(tmp_path):
    root = str(Path(__file__).resolve().parents[3])
    script = f"""
import sys
sys.path.insert(0, {root!r})
from pyplumber.mixer import MixerOptions, MixerApplication, SourceContext, build_application
assert callable(build_application)
assert not {{'_avplumber', 'pyplumber.core', 'pyplumber.node', 'mixer', 'layouts'}} & set(sys.modules)
assert not any(name.startswith('demos.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path, check=True, capture_output=True, text=True)


def test_config_renditions_do_not_need_cli_output_flags():
    cfg = show()
    app = build_application(cfg, api=fake_api())
    nodes = {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    assert nodes["record_output"]["url"] == "program.ts"
    assert nodes["record_fps"]["fps"] == "60/1"
    assert app.mixer.parameters["fps"] == (60, 1)
    assert json.loads(app.avp.commands_registered["mixer.settings"]("")) == cfg.settings()


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
    finally:
        app.stop()
    assert app.avp.shutdown_complete
