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
from test_graph import CONFIG
from test_prewarm import NativeEngine


class _Engine(NativeEngine):
    """application.py's config-driven build path registers a settings command
    NativeEngine (built for cli.py's build path) doesn't stub."""
    def registerControlCommand(self, *args):
        pass


def _api(nodes, builder):
    from pyplumber.rtcp_feedback import RtcpFeedbackListener
    return SimpleNamespace(**{
        name: getattr(nodes, name) for name in (
            "AssumeVideoFormat", "AudioMixer", "Bsf", "DecAudio", "DecVideo", "Demux", "EncAudio",
            "EncVideo", "FilterVideo", "ForceFPS", "ForceKeyFrame", "InputRec", "Mux",
            "Output", "PreheatVideoRouter", "Realtime", "ResampleAudio",
            "ResampleAudioForwardChannels", "Split",
        )
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


def test_initial_scene_gets_a_startup_cut_when_audio_router_set(native_boundary):
    nodes, builder = native_boundary
    app = build_application(_audio_show(), MixerOptions(audio_router_node="audio_mixer"),
                            api=_api(nodes, builder))
    cut_events = [e for e in app.avp.events if e.startswith("mixer.cut ")]
    assert len(cut_events) == 1
    assert '"scene": "full"' in cut_events[0]


def test_no_audio_router_means_no_startup_cut_and_no_audio_fields(native_boundary):
    nodes, builder = native_boundary
    app = build_application(_audio_show(), MixerOptions(), api=_api(nodes, builder))
    assert not [e for e in app.avp.events if e.startswith("mixer.cut ")]
    init_events = [e for e in app.avp.events if e.startswith("mixer.init ")]
    assert "audio_router_node" not in init_events[0]
    assert "audio_bus" not in init_events[0]


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
