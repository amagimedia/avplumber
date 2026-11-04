"""mixer.init carries the audio_router_node field MixerOrchestrator's C++ side
consumes (src/avplumber.cpp's mixer.init handler,
src/mixer/orchestrator/{scene,fade,wipe}.cpp's emitTransitionEnded/
emitTransitionProgress) -- those can't run here (no native engine on this
machine), so this is the boundary: everything on the Python side that
produces the commands the C++ side is contracted to understand.
"""
from types import SimpleNamespace

import pytest

from pyplumber.mixer import config as mixer_config
from pyplumber.mixer.application import MixerOptions, build_application
from test_graph import CONFIG, fake_api
from test_prewarm import NativeEngine


class _Engine(NativeEngine):
    """application.py's config-driven build path registers a settings command
    NativeEngine (built for cli.py's build path) doesn't stub."""
    def registerControlCommand(self, *args):
        pass


def _api(nodes, builder):
    from pyplumber.rtcp_feedback import RtcpFeedbackListener
    # Every node class the graph tests build from (test_graph.fake_api), from the native boundary.
    engine = {"AVPlumber", "MixerGraphBuilder", "RtcpFeedbackListener"}
    return SimpleNamespace(**{
        name: getattr(nodes, name) for name in vars(fake_api()) if name not in engine
    }, AVPlumber=_Engine, MixerGraphBuilder=builder, RtcpFeedbackListener=RtcpFeedbackListener)


def _audio_show():
    camera = {**CONFIG["sources"][0], "has_audio": True}
    return mixer_config.parse({
        "canvas": CONFIG["canvas"],
        "sources": [camera],
        "scenes": [{"id": "full", "items": CONFIG["scenes"][0]["items"]}],
        "renditions": [{"id": "record", "target": "program.ts"}],
    })


def test_mixer_init_carries_audio_router_node(native_boundary):
    nodes, builder = native_boundary
    app = build_application(_audio_show(), MixerOptions(audio_router_node="audio_mixer"),
                            api=_api(nodes, builder))
    init_events = [e for e in app.avp.events if e.startswith("mixer.init ")]
    assert len(init_events) == 1
    assert '"audio_router_node": "audio_mixer"' in init_events[0]
    assert "audio_bus" not in init_events[0]


def test_mixer_scene_carries_no_audio_fields(native_boundary):
    nodes, builder = native_boundary
    app = build_application(_audio_show(), MixerOptions(audio_router_node="audio_mixer"),
                            api=_api(nodes, builder))
    scene_events = [e for e in app.avp.events if e.startswith("mixer.scene ")]
    assert len(scene_events) == 1
    assert "audio_source" not in scene_events[0]
    assert "audio_follow" not in scene_events[0]


def test_no_cut_before_nodes_exist_then_initial_gains_without_a_take(native_boundary):
    nodes, builder = native_boundary
    app = build_application(_audio_show(), MixerOptions(audio_router_node="audio_mixer"),
                            api=_api(nodes, builder))
    # Regression guard: build() alone must never fire a take -- live, an early cut
    # raced ahead of mixer_comp_a/b and audio_mixer construction and failed with
    # "node has no runtime controls", leaving AudioMixer's gains at their
    # all-zero default forever. raise_initial_audio_gains sends transition_ended
    # directly (no mixer.cut), so it does not swap slots or force a keyframe.
    assert not [e for e in app.avp.events if e.startswith("mixer.cut ")]
    before = len(app.avp.events)
    app.mixer.raise_initial_audio_gains()
    assert not [e for e in app.avp.events[before:] if e.startswith("mixer.cut ")]
    ended = [e for e in app.avp.events[before:] if "transition_ended" in e]
    assert len(ended) == 1
    assert ended[0].startswith("node.object.set audio_mixer transition_ended ")
    assert '"output": "program"' in ended[0]
    assert '"sources"' in ended[0]


def test_sources_with_audio_bring_the_audio_mixer(native_boundary):
    # Unread, a source's audio edge would fill and hold its demux, and the video with it.
    nodes, builder = native_boundary
    app = build_application(_audio_show(), MixerOptions(), api=_api(nodes, builder))
    init_events = [e for e in app.avp.events if e.startswith("mixer.init ")]
    assert '"audio_router_node": "audio_mixer"' in init_events[0]
    assert app.avp.nodes["audio_mixer"]["source_ids"] == ["cam"]


def test_the_program_audio_is_on_one_janus_output_and_in_its_settings():
    import pyplumber.mixer.tools.janus_mountpoints as janus_mountpoints
    show = {"canvas": CONFIG["canvas"], "sources": [{**CONFIG["sources"][0], "has_audio": True}],
            "scenes": [{"id": "full", "items": CONFIG["scenes"][0]["items"]}],
            "renditions": [{"id": "sdr", "target": "janus", "port": 5004},
                           {"id": "hd", "target": "janus", "port": 5006, "codec": "hevc_nvenc"}]}
    specs = janus_mountpoints.outputs(show)
    assert (specs["sdr"]["audio"], specs["sdr"]["audioport"], specs["sdr"]["audiocodec"]) == (True, 5002, "opus")
    assert specs["hd"]["audio"] is False
    programs = mixer_config.parse(show).settings()["program_outputs"]
    assert [o.get("audio", False) for o in programs] == [True, False]
    show["sources"][0]["has_audio"] = False
    assert not any(spec["audio"] for spec in janus_mountpoints.outputs(show).values())
    assert not any("audio" in o for o in mixer_config.parse(show).settings()["program_outputs"])


def test_audio_port_duplicates_program_audio_onto_another_rendition():
    """HDR wants the same audio SDR already has, not an independent mix: audio_port on a
    top-level rendition (unlike on an AUX bus's) attaches a second Opus leg at its own port,
    alongside -- not instead of -- the legacy audio_rendition's."""
    import pyplumber.mixer.tools.janus_mountpoints as janus_mountpoints
    show = {"canvas": CONFIG["canvas"], "sources": [{**CONFIG["sources"][0], "has_audio": True}],
            "scenes": [{"id": "full", "items": CONFIG["scenes"][0]["items"]}],
            "renditions": [{"id": "sdr", "target": "janus", "port": 5004},
                           {"id": "hdr", "target": "janus", "port": 5006, "codec": "hevc_nvenc",
                            "audio_port": 5012}]}
    specs = janus_mountpoints.outputs(show)
    assert (specs["sdr"]["audio"], specs["sdr"]["audioport"]) == (True, 5002)
    assert (specs["hdr"]["audio"], specs["hdr"]["audioport"]) == (True, 5012)
    programs = mixer_config.parse(show).settings()["program_outputs"]
    assert [o.get("audio", False) for o in programs] == [True, True]


def test_audio_port_rendition_duplicates_the_program_bus_via_a_split(native_boundary):
    nodes, builder = native_boundary
    cfg = mixer_config.parse({
        "canvas": CONFIG["canvas"], "sources": [{**CONFIG["sources"][0], "has_audio": True}],
        "scenes": [{"id": "full", "items": CONFIG["scenes"][0]["items"]}],
        "renditions": [{"id": "sdr", "target": "janus", "port": 5004},
                       {"id": "hdr", "target": "janus", "port": 5006, "codec": "hevc_nvenc",
                        "audio_port": 5012}],
    })
    app = build_application(cfg, MixerOptions(janus_output=True), api=_api(nodes, builder))
    # The Split reads the bus edge itself: audio_mixer mirrors its "outputs" into "dst", which is
    # what the dependency sort reads (NodeGroupUtils::offers), so the sort knows the Split has to
    # wait for audio_mixer. Before that mirror this needed a no-op anull relay in between, whose
    # plain "dst" was the only thing making the dependency visible.
    assert "program_audio_relay" not in app.avp.nodes
    assert set(app.avp.nodes["audio_mixer"]["dst"]) == {"audio_mixer_program"}
    split = app.avp.nodes["split_program_audio"]
    assert split["src"] == "audio_mixer_program"
    assert set(split["dst"]) == {"audio_mixer_program_split", "janus_audio_hdr_src"}
    # The legacy leg now reads the split's own branch, not the raw bus edge.
    assert app.avp.nodes["janus_audio_limiter"]["src"] == "audio_mixer_program_split"
    assert app.avp.nodes["janus_audio_hdr_limiter"]["src"] == "janus_audio_hdr_src"


def test_aux_bus_audio_port_is_surfaced_in_preview_outputs():
    """The webui's mute icon (outputs.js previewOption/programOptions) is keyed off an "audio"
    flag it reads straight from mixer.settings; an AUX bus with audio_port must carry it in
    preview_outputs the same way a program rendition carries it in program_outputs, or the icon
    never shows even though the bus has a real, independent Opus leg."""
    cfg = mixer_config.parse({
        "canvas": CONFIG["canvas"], "sources": [{**CONFIG["sources"][0], "has_audio": True}],
        "scenes": [{"id": "full", "items": CONFIG["scenes"][0]["items"]}],
        "renditions": [{"id": "record", "target": "program.ts"}],
        "aux_buses": [
            {"id": "aux1", "renditions": [{"id": "aux1_sdr", "port": 5010, "audio_port": 5016}]},
            {"id": "aux2", "renditions": [{"id": "aux2_sdr", "port": 5012}]},
        ],
    })
    previews = {o["bus"]: o for o in cfg.settings()["preview_outputs"]}
    assert previews["aux1"]["audio"] is True
    assert "audio" not in previews["aux2"]


def test_only_a_buss_first_rendition_can_claim_its_audio():
    """A bus's audio leg, its audio_mixer bus and its port check are all renditions[0]'s
    (aux.py, application.py), so a later rendition's audio_port must not paint a mute button
    the operator's click could never reach. parse rejects a second aux rendition today, so
    this pins the settings rule rather than a reachable config."""
    from dataclasses import replace as dc_replace
    cfg = mixer_config.parse({
        "canvas": CONFIG["canvas"], "sources": [{**CONFIG["sources"][0], "has_audio": True}],
        "scenes": [{"id": "full", "items": CONFIG["scenes"][0]["items"]}],
        "renditions": [{"id": "record", "target": "program.ts"}],
        "aux_buses": [{"id": "aux1", "renditions": [{"id": "aux1_sdr", "port": 5010}]}],
    })
    bus = cfg.aux_buses[0]
    second = dc_replace(bus.renditions[0], id="aux1_extra", port=5020, audio_port=5022)
    cfg = dc_replace(cfg, aux_buses=(dc_replace(bus, renditions=(bus.renditions[0], second)),))
    previews = {o["rendition"]: o for o in cfg.settings()["preview_outputs"]}
    assert "audio" not in previews["aux1_sdr"]     # renditions[0], no audio_port of its own
    assert "audio" not in previews["aux1_extra"]   # has one, but nothing is wired to it


def test_a_show_without_audio_leaves_the_legacy_audio_ports_free():
    """5002/5003 are reserved for the legacy single audio leg because any has_audio source
    claims them at runtime -- but a show with no audio never builds that leg, and had no
    reason to refuse a video rendition on those ports."""
    def show(port, has_audio):
        return mixer_config.parse({
            "canvas": CONFIG["canvas"], "sources": [{**CONFIG["sources"][0], "has_audio": has_audio}],
            "scenes": [{"id": "full", "items": CONFIG["scenes"][0]["items"]}],
            "renditions": [{"id": "sdr", "target": "janus", "port": port}],
            "aux_buses": [],
        })
    assert show(5002, False).renditions[0].port == 5002
    with pytest.raises(mixer_config.ConfigError, match="distinct RTP/RTCP"):
        show(5002, True)


def test_aux_bus_with_audio_port_gets_its_own_outputs_entry(native_boundary, monkeypatch):
    """The AuxBus/AuxBuses themselves (their own audio leg) are covered in test_aux.py; this
    isolates the one line in application.py that decides audio_mixer's outputs map, by faking
    out the aux compositor build (native_boundary's engine stub predates aux support)."""
    import pyplumber.mixer.aux as aux_module

    class FakeAuxBus:
        def __init__(self, avp, api, mixer, cfg, bus):
            self.bus, self.pgm_edge = bus, None

        def build(self, options):
            pass

    class FakeAuxBuses:
        def __init__(self, avp, buses):
            self._buses = buses

        def __iter__(self):
            return iter(self._buses)

        def __len__(self):
            return len(self._buses)

        def register_commands(self):
            pass

    monkeypatch.setattr(aux_module, "AuxBus", FakeAuxBus)
    monkeypatch.setattr(aux_module, "AuxBuses", FakeAuxBuses)
    nodes, builder = native_boundary
    cfg = mixer_config.parse({
        "canvas": CONFIG["canvas"], "sources": [{**CONFIG["sources"][0], "has_audio": True}],
        "scenes": [{"id": "full", "items": CONFIG["scenes"][0]["items"]}],
        "renditions": [{"id": "record", "target": "program.ts"}],
        "aux_buses": [{"id": "aux1", "renditions": [{"id": "aux1_sdr", "port": 5010, "audio_port": 5016}]}],
    })
    app = build_application(cfg, MixerOptions(audio_router_node="audio_mixer"), api=_api(nodes, builder))
    outputs = app.avp.nodes["audio_mixer"]["outputs"]
    assert outputs == {"program": "audio_mixer_program", "aux1": "audio_mixer_aux1"}
    # Every bus edge is mirrored into "dst" so the dependency sort knows who produces it: the aux
    # leg reads audio_mixer_aux1 with nothing in between, so the sort is all that orders them.
    assert set(app.avp.nodes["audio_mixer"]["dst"]) == set(outputs.values())


def test_video_only_show_is_unaffected(native_boundary):
    """has_audio default off: a plain video-only show must not change behavior."""
    nodes, builder = native_boundary
    cfg = mixer_config.parse({
        "canvas": CONFIG["canvas"], "sources": [CONFIG["sources"][0]],
        "scenes": [{"id": "full", "items": CONFIG["scenes"][0]["items"]}],
        "renditions": [{"id": "record", "target": "program.ts"}],
    })
    app = build_application(cfg, MixerOptions(), api=_api(nodes, builder))
    assert app.audio_edges == ()
    assert not [e for e in app.avp.events if e.startswith("mixer.cut ")]
    assert not any("audio" in name for name in app.avp.nodes)


def test_audio_mixer_is_wired_when_router_and_has_audio(native_boundary):
    nodes, builder = native_boundary
    app = build_application(_audio_show(), MixerOptions(audio_router_node="audio_mixer"),
                            api=_api(nodes, builder))
    assert app.audio_edges
    # NativeEngine.addNode stores parameters by name.
    params = app.avp.nodes.get("audio_mixer")
    assert params is not None
    assert params["type"] == "audio_mixer"
    assert params["source_ids"]
    assert "program" in params["outputs"]


def test_unbuilt_audio_router_node_is_cleared_not_left_dangling(native_boundary):
    """A router name with no has_audio sources behind it must not survive into
    mixer.init: app.mixer.audio_router_node is what MixerApplication._warm_outputs
    waits on, and a node that is never built would hang preheat for
    preheat_timeout_sec (60s by default) before failing -- see application.py's
    `elif options.audio_router_node and not audio_edges` branch. options is cleared
    too so aux.build does not wire Janus audio to an edge nothing feeds."""
    nodes, builder = native_boundary
    cfg = mixer_config.parse({
        "canvas": CONFIG["canvas"], "sources": [CONFIG["sources"][0]],
        "scenes": [{"id": "full", "items": CONFIG["scenes"][0]["items"]}],
        "renditions": [{"id": "record", "target": "program.ts"}],
    })
    app = build_application(cfg, MixerOptions(audio_router_node="custom_router"),
                            api=_api(nodes, builder))
    assert app.audio_edges == ()
    assert app.mixer.audio_router_node == ""
    init_events = [e for e in app.avp.events if e.startswith("mixer.init ")]
    assert "audio_router_node" not in init_events[0]


def test_build_audio_mixer_false_leaves_a_custom_router_name_alone(native_boundary):
    """With build_audio_mixer=False, build_application must not build its own
    audio_mixer node (or anything else) under the caller's router name -- that
    name is reserved for an IInputsObjects impl the caller builds itself (e.g.
    dyremix_mixer in avplumber-nodes-2), and a competing audio_mixer there would
    defeat the whole point of supplying a custom node."""
    nodes, builder = native_boundary
    app = build_application(_audio_show(),
                            MixerOptions(audio_router_node="dyremix", build_audio_mixer=False),
                            api=_api(nodes, builder))
    assert app.audio_edges
    assert app.mixer.audio_router_node == "dyremix"
    assert app.avp.nodes.get("dyremix") is None
    assert not any(params.get("type") == "audio_mixer" for params in app.avp.nodes.values())
    init_events = [e for e in app.avp.events if e.startswith("mixer.init ")]
    assert '"audio_router_node": "dyremix"' in init_events[0]


def test_build_audio_mixer_false_without_a_router_name_builds_nothing(native_boundary):
    """build_audio_mixer=False with no caller-supplied audio_router_node must not
    fall back to the in-tree "audio_mixer" name -- that name would then reach
    _wait_for_node with nothing ever built under it (build_audio_mixer=False skips
    wire_audio_mixer too), hanging the preheat for preheat_timeout_sec."""
    nodes, builder = native_boundary
    app = build_application(_audio_show(), MixerOptions(build_audio_mixer=False),
                            api=_api(nodes, builder))
    assert app.audio_edges
    assert not app.mixer.audio_router_node
    assert app.avp.nodes.get("audio_mixer") is None
    init_events = [e for e in app.avp.events if e.startswith("mixer.init ")]
    assert "audio_router_node" not in init_events[0]


def test_audio_router_node_is_refused_without_a_config():
    """--input mode never builds an audio router: _build_input drops the demuxer's audio leg
    and only the config path calls wire_audio_mixer. The name still reached mixer.init and
    _warm_outputs, where avp.node() raises for a node that was never created -- the show died
    at the first preheat poll instead of at the argument it came from."""
    from pyplumber.mixer.cli import GraphOptions
    with pytest.raises(ValueError, match="audio-router-node"):
        GraphOptions(inputs=("camera.mp4",), output="program.ts",
                     audio_router_node="audio_mixer").validate()
    # A config declares has_audio sources, so there the name is the caller's to use.
    GraphOptions(config="show.json", audio_router_node="audio_mixer").validate()


class _GainAvp:
    """The node is unreachable, as it is mid-startup: _rendered() degrades to the cache."""
    def __init__(self):
        self.sent = []

    def node(self, _name):
        raise RuntimeError("no node yet")

    def executeCommandsFromString(self, command):
        self.sent.append(command)


def test_a_rejected_gain_leaves_every_other_source_alone():
    """The values were checked one at a time as they were sent, so a request carrying one bad
    gain put the sources named before it on air at their new volume and still answered with an
    error -- which half of it applied depended on the order the keys arrived in."""
    from pyplumber.mixer.audio_gain import AudioGainController
    avp = _GainAvp()
    controller = AudioGainController(avp, "audio_mixer", ["cam", "idle"])
    with pytest.raises(mixer_config.ConfigError, match="finite number"):
        controller.set({"gains": {"cam": -6.0, "idle": "loud"}})
    assert avp.sent == []
    assert controller.state() == [{"id": "cam", "gain_db": 0.0}, {"id": "idle", "gain_db": 0.0}]


@pytest.mark.parametrize("bad", (True, False, float("nan"), float("inf")))
def test_a_gain_must_be_a_finite_number(bad):
    """A bool passes isinstance(x, int), so true would have been accepted as +1 dB; a
    non-finite one clamps to itself and reaches the node as bare NaN, which is not JSON."""
    from pyplumber.mixer.audio_gain import AudioGainController
    avp = _GainAvp()
    controller = AudioGainController(avp, "audio_mixer", ["cam"])
    with pytest.raises(mixer_config.ConfigError, match="finite number"):
        controller.set({"gains": {"cam": bad}})
    assert avp.sent == []


def test_a_valid_gain_is_clamped_and_sent_once():
    from pyplumber.mixer.audio_gain import AudioGainController
    avp = _GainAvp()
    controller = AudioGainController(avp, "audio_mixer", ["cam"])
    assert controller.set({"gains": {"cam": -200.0}}) == [{"id": "cam", "gain_db": -70.0}]
    assert len(avp.sent) == 1 and "set_source_gain" in avp.sent[0]
    controller.set({"gains": {"cam": -70.0}})   # already there: nothing to send
    assert len(avp.sent) == 1


def test_janus_audio_limits_float_mix_before_s16_conversion():
    from pyplumber.mixer.janus import JanusAudioConfig, build_janus_audio_output

    nodes = []
    avp = SimpleNamespace(addNode=nodes.append)
    build_janus_audio_output(avp, fake_api(), "mixed", JanusAudioConfig())
    params = {node.parameters["name"]: node.parameters for node in nodes}
    assert params["janus_audio_limiter"]["src"] == "mixed"
    assert params["janus_audio_limiter"]["dst"] == "janus_audio_limited"
    assert params["janus_audio_limiter"]["graph"].startswith("alimiter=")
    assert params["janus_audio_resample"]["src"] == "janus_audio_limited"
    assert params["janus_audio_resample"]["dst_sample_format"] == "s16"


def test_janus_audio_limiter_can_be_disabled():
    from pyplumber.mixer.janus import JanusAudioConfig, build_janus_audio_output

    nodes = []
    avp = SimpleNamespace(addNode=nodes.append)
    build_janus_audio_output(avp, fake_api(), "mixed", JanusAudioConfig(limiter=False))
    params = {node.parameters["name"]: node.parameters for node in nodes}
    assert "janus_audio_limiter" not in params
    assert params["janus_audio_resample"]["src"] == "mixed"
