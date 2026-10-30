"""mixer.init carries the audio_router_node field MixerOrchestrator's C++ side
consumes (src/avplumber.cpp's mixer.init handler,
src/mixer/orchestrator/{scene,fade,wipe}.cpp's emitTransitionEnded/
emitTransitionProgress) -- those can't run here (no native engine on this
machine), so this is the boundary: everything on the Python side that
produces the commands the C++ side is contracted to understand.
"""
from types import SimpleNamespace

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
    # audio_mixer declares its bus edges under "outputs", not "dst", so the graph's dependency sort
    # never learns it produces audio_mixer_program; a Split reading that edge directly can be built
    # before audio_mixer exists and fail. The no-op relay has a plain "dst", so Split's dependency
    # on it (not on audio_mixer) is one the sort does track.
    relay = app.avp.nodes["program_audio_relay"]
    assert relay["src"] == "audio_mixer_program"
    split = app.avp.nodes["split_program_audio"]
    assert split["src"] == "audio_mixer_program_relay"
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
