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


def test_local_monitor_taps_program_before_encoding(tmp_path):
    app = build_application(show(), MixerOptions(dmabuf_monitor_dir=str(tmp_path)), api=fake_api())
    nodes = {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    tap, sink = nodes["monitor_program_tap"], nodes["monitor_program"]
    assert tap["src"] == "program_measured" and tap["outputs"] == 1
    primary, monitor = tap["dst"]
    assert tap["subscribed_outputs"] == {monitor: monitor}
    assert sink["subscription"] == monitor and sink["socket"] == str(tmp_path / "program.sock")
    assert any(n.get("src") == primary for n in nodes.values())
    assert sink["type"] == "ipc_dmabuf_sink"


def test_local_aux_previews_replace_the_bus_encoder_with_its_monitor(native_boundary, tmp_path):
    """The desktop app shows an AUX bus from the GPU, so the bus has no encoder, Janus stream or
    monitor tap: the monitor is the only reader. Without the option, or without a monitor
    directory, the bus is encoded as before."""
    class Engine(NativeEngine):
        def addNode(self, node, **kwargs):
            super().addNode(node)

        def registerControlCommand(self, *args):
            pass

    cfg = show()
    cfg = replace(cfg, aux_buses=config.parse_aux_buses([{
        "id": "mv", "renditions": [{"id": "monitor", "port": 5012}]}], cfg))
    def build(**options):
        api = fake_api()
        api.AVPlumber, api.MixerGraphBuilder = Engine, native_boundary[1]
        app = build_application(cfg, MixerOptions(**options), api=api)
        return app, app.avp.nodes
    app, local = build(dmabuf_monitor_dir=str(tmp_path), local_aux_previews=True)
    bus = next(iter(app.aux_buses))
    assert bus.listener is None and bus.local
    sink = local["monitor_aux_mv"]
    assert sink["src"] == bus.output_edge and "subscription" not in sink and "monitor_aux_mv_tap" not in local
    assert not [name for name in local if name.startswith("aux_mv") and "enc" in name]
    assert "monitor_program_tap" in local and any(n.get("type") == "enc_video" for n in local.values())   # the program is encoded
    app, encoded = build(dmabuf_monitor_dir=str(tmp_path))
    assert not next(iter(app.aux_buses)).local and "monitor_aux_mv_tap" in encoded
    assert set(local) < set(encoded)
    _, plain = build(local_aux_previews=True)
    assert set(plain) == {name for name in encoded if not name.startswith("monitor_")}


def test_processed_cuda_ipc_input_uses_final_edge_for_optional_startup(native_boundary):
    options = MixerOptions(cuda_ipc_inputs=(("cam", "/tmp/playout.sock"),))
    app = build_application(show(), options, api=fake_api(),
                            process_source=lambda ctx: f"processed_{ctx.source.id}")
    assert app.optional_input_edges == ("processed_cam",)
    app.absent_inputs = app.optional_input_edges
    app._release_absent_inputs()


def test_cuda_ipc_override_only_opens_remaining_browser(native_boundary, monkeypatch):
    cfg = show()
    cfg = replace(cfg, sources=tuple(replace(s, kind="browser", location="about:blank") for s in cfg.sources))
    opened, sockets = [], []
    monkeypatch.setattr(mixer_application, "open_windows", lambda url, windows: opened.extend(windows))
    monkeypatch.setattr(mixer_application, "wait_for_sockets", lambda paths, timeout: sockets.extend(paths))
    app = build_application(cfg, MixerOptions(cuda_ipc_inputs=(("cam", "/tmp/playout.sock"),)), api=fake_api())
    assert [window["id"] for window in opened] == ["idle"]
    assert [Path(path).name for path in sockets] == ["idle.sock"]
    assert app.browser_windows == ("idle",)
    assert app.browser_importers == ("input_1_to_cuda",)
    assert app.browser_input_edges == (app.input_edges[1],)


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


@pytest.mark.parametrize("reader", config.MXL_READERS)
@pytest.mark.parametrize("color, storage", [("sdr", "nv12"), ("sdr", "nv16"), ("hlg", "p210le")])
def test_an_mxl_writer_restart_never_ends_the_program(native_boundary, color, storage, reader):
    """A writer coming back restarts the MXL input group, which then sends an EOF that would end a
    program showing only that source. A format declaration that ignores EOF, restarting with the
    group, drops it before the fan-out. The unpack writes the canvas's storage itself when the
    colour is the canvas's (no colour stage, no second pass); a conversion reads all of P210.
    The native reader is one node that restarts its group itself; the demuxer chain is the fallback."""
    api = fake_api()
    api.MixerGraphBuilder = native_boundary[1]
    cfg = config.parse({"canvas": {**CONFIG["canvas"], "mxl_reader": reader,
                                   "working_format": storage if color == "sdr" else "nv12"},
                        "renditions": [{"id": "record", "target": "program.ts"}],
                        "sources": [{"id": "cam", "kind": "mxl", "url": "/dev/shm/mxl", "flow": "f",
                                     "width": 1920, "height": 1080, "color": color}],
                        "scenes": [{"id": "one", "items": [{"source": "cam", "dst": {"x": 0, "y": 0, "w": 1920, "h": 1080}}]}]})
    app = build_application(cfg, api=api)
    nodes = {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    live = next(n for n in nodes.values() if n.get("dst") == nodes["mixer_otm_cam"]["src"])
    if reader == "native":
        assert not {"unpack_0", "decode_0"} & nodes.keys() and nodes["realtime_0"]["src"] == nodes["input_0"]["dst"]
        assert {k: v for k, v in nodes["input_0"].items() if k not in ("name", "dst", "hwaccel", "group")} == {
            "type": "raw_video_file_cuda", "url": "mxl:///dev/shm/mxl?id=f", "pixel_format": "v210", "storage": storage,
            "video_size": "1920x1080", "framerate": "60/1", "loop": False, "loop_continuous_ts": False,
            "frame_color": cfg.sources[0].color.tags, "auto_restart": "group"}
        # Released 40 ms after each grain's own time (TAI) at any rate, on a team of its own.
        assert (nodes["realtime_0"]["team"], nodes["realtime_0"]["tai_delay"]) == ("input_0_tai", 0.040)
        assert "jitter_margin" not in nodes["realtime_0"]
    else:
        # The demuxer's timestamps run on across a stall: a frame of jitter buffer after arrival.
        assert nodes["realtime_0"]["jitter_margin"] == 1 / 60 and "tai_delay" not in nodes["realtime_0"]
        assert nodes["unpack_0"]["graph"].startswith(f"hwupload_cuda=v210_width=1920:v210_format={storage},")
        # Live: never looped; a recreated flow restarts the source's chain.
        assert (nodes["input_0"]["url"], nodes["input_0"]["format"]) == ("mxl:///dev/shm/mxl?id=f", "mxl")
        assert nodes["input_0"]["options"] == {"blocking": "1", "grain_index_init": "head", "on_too_late": "reset",
                                               "reset_on_drop": "1"}
        assert "loop" not in nodes["input_0"] and nodes["input_0"]["auto_restart"] == nodes["decode_0"]["auto_restart"] == "group"
        assert nodes["decode_0"]["options"] == {"video_size": "5120x1080", "pixel_format": "gray"}
    if color == "hlg":
        assert "tonemap_cuda" in nodes["mixer_color_cam"]["graph"] and live["src"] == nodes["mixer_color_cam"]["dst"]
    else:
        assert "mixer_color_cam" not in nodes
    assert {k: live[k] for k in ("type", "ignore_eof", "group", "pixel_format", "real_pixel_format")} == {
        "type": "assume_video_format", "ignore_eof": True, "group": nodes["mixer_otm_cam"]["group"], "pixel_format": "cuda",
        "real_pixel_format": cfg.working_format}


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


def test_an_alpha_source_reaches_the_canvas_as_straight_rgba(native_boundary):
    """NVDEC and the NV12 colour stage both drop alpha. An alpha source decodes in software, is
    converted to RGBA and uploaded, and is composited as packed RGB, so a blended item draws it with
    its alpha. Its alpha is straight, unlike a browser page's."""
    api = fake_api()
    api.MixerGraphBuilder = native_boundary[1]
    app = build_application(show(alpha=True), api=api)
    nodes = {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    assert nodes["decode_0"]["hwaccel_only_for_codecs"] == []
    # Uploaded after the chain's pacing, so seeks still reach the decoder.
    upload = nodes["alpha_upload_0"]
    assert upload["graph"] == "format=rgba,hwupload_cuda" and upload["hwaccel"] == "mixer_gpu"
    assert upload["src"] == "input_0_realtime" and "filter_0" not in nodes
    assert app.input_edges[0] == "input_0_rgba"
    (cam,) = [s for s in app.mixer._sources if s.name == "cam"]
    assert cam.packed_rgb and not cam.premultiplied_alpha and cam.pixel_format == "rgba"
    # No colour filter: the fan-out tags the frames, and nothing converts them to NV12.
    assert "mixer_color_cam" not in nodes
    assert nodes["mixer_color_alias_cam"]["frame_color"]["colorspace"] == "bt709"
    for bad in ({"alpha": "yes"}, {"alpha": True, "decode_storage": "cuarray"},
                {"alpha": True, "filter": "hflip_cuda", "filter_output_format": "nv12"}):
        with pytest.raises(config.ConfigError, match="alpha"):
            show(**bad)
    with pytest.raises(config.ConfigError, match="SDR only"):
        show(alpha=True, color="hlg")


def test_an_ndi_source_is_received_converted_to_nv12_and_uploaded(native_boundary):
    """FFmpeg's libndi_newtek device receives the named source (found at extra_ips besides mDNS);
    its packed 4:2:2 is wrapped in software, uploaded after pacing and unpacked to tagged NV12 there, and
    its audio joins the show's like a video file's. NDI never ends: no loop, the group restarts."""
    def fake_api_with(boundary):
        api = fake_api()
        api.MixerGraphBuilder = boundary[1]
        return api
    api = fake_api_with(native_boundary)
    ndi = {"id": "cam", "kind": "ndi", "name": "HOST (avp-smoke)", "extra_ips": "127.0.0.1", "color": "sdr",
           "width": 1920, "height": 1080}
    def ndi_show(source):
        return config.parse({"canvas": CONFIG["canvas"], "renditions": [{"id": "record", "target": "program.ts"}],
                             "sources": [source, {**CONFIG["sources"][0], "id": "idle", "path": "idle.mp4"}],
                             "scenes": [{"id": "two", "items": CONFIG["scenes"][1]["items"][1:]}]})
    cfg = ndi_show({**ndi, "has_audio": True})
    app = build_application(cfg, api=api)
    nodes = {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    received = nodes["input_0"]
    # The plain input node, whose reads time out: a silent receiver fails it and its group restarts.
    assert (received["type"], received["url"], received["format"]) == ("input", "HOST (avp-smoke)", "libndi_newtek")
    assert received["options"] == {"wait_sources": "5", "extra_ips": "127.0.0.1"} and received["timeout"] == 2
    assert "loop" not in received and nodes["demux_0"]["src"] == received["dst"] and received["auto_restart"] == "group"
    pacing = nodes["realtime_0"]
    # Arrival-paced with a frame of jitter buffer, resynced when frames run ahead (a stale anchor).
    assert pacing["discontinuity_threshold"] == 0.1 and pacing["jitter_margin"] == pacing["initial_jitter_margin"] > 0
    fps_num, fps_den = config.rate_pair(cfg.fps)
    period = fps_den / fps_num
    assert pacing["jitter_margin"] == period
    # A backlog in front of pacing (what arrived against what is being paced) is dropped, not kept,
    # for a source without audio: with it the audio node would share the rule and the team.
    assert pacing["team"] == "input_0_av_sync" and "input_ts_queue" not in pacing
    silent = {n.parameters["name"]: n.parameters for n in build_application(ndi_show(ndi), api=fake_api_with(native_boundary)).avp.nodes}["realtime_0"]
    assert (silent["team"], silent["input_ts_queue"], silent["intermediate_queues"]) == (
        "input_0_av_sync", received["dst"], ["input_0_video_packets"])
    assert (silent["max_buffered"], silent["min_buffered"]) == (3.5 * period, 1.5 * period)
    # Its restart while the show starts does not fail the start.
    assert app.live_inputs == ("input_0",)
    assert nodes["decode_0"]["auto_restart"] == "group" and nodes["decode_0"]["hwaccel_only_for_codecs"] == []
    upload = nodes["ndi_upload_0"]
    assert upload["src"] == "input_0_realtime" and app.input_edges[0] == "input_0_nv12"
    # Unpacked into NV12 by the device, not converted on the CPU: no filter graph.
    assert upload["type"] == "packed_to_cuda" and upload["frame_color"] == cfg.sources[0].color.tags
    assert (upload["hwaccel"], upload["auto_restart"]) == ("mixer_gpu", "group") and "graph" not in upload
    # A sender lost on program restarts the chain: its EOF stops at the guard before the fan-out.
    live = next(n for n in nodes.values() if n.get("dst") == nodes["mixer_otm_cam"]["src"])
    assert live["type"] == "assume_video_format" and live["ignore_eof"] and (live["width"], live["height"]) == (1920, 1080)
    (cam,) = [s for s in app.mixer._sources if s.name == "cam"]
    assert cam.pixel_format == "nv12" and not cam.packed_rgb
    assert "realtime_audio_0" in nodes
    assert upload["storage"] == "nv12" and live["real_pixel_format"] == "nv12"
    # With alpha (graphics): BGRA on the GPU whatever the sender delivers, straight alpha, composited
    # as packed RGB like a browser page, the guard declaring that format.
    keyed = build_application(ndi_show({**ndi, "alpha": True}), api=fake_api_with(native_boundary))
    nodes = {n.parameters["name"]: n.parameters for n in keyed.avp.nodes}
    upload = nodes["ndi_upload_0"]
    assert (upload["type"], upload["storage"], upload["dst"]) == ("packed_to_cuda", "bgra", "input_0_bgra")
    live = next(n for n in nodes.values() if n.get("dst") == nodes["mixer_otm_cam"]["src"])
    assert live["type"] == "assume_video_format" and live["ignore_eof"] and live["real_pixel_format"] == "bgra"
    (cam,) = [s for s in keyed.mixer._sources if s.name == "cam"]
    assert cam.packed_rgb and not cam.premultiplied_alpha and cam.pixel_format == "bgra" and not cam.color_tagged
    for bad, match in (({"color": "hlg"}, "SDR only"), ({"name": None}, "needs name"), ({"width": None}, "width"),
                       ({"extra_ips": 1}, "extra_ips")):
        with pytest.raises(config.ConfigError, match=match):
            ndi_show({k: v for k, v in {**ndi, **bad}.items() if v is not None})
    with pytest.raises(config.ConfigError, match="extra_ips"):
        show(extra_ips="127.0.0.1")


def test_config_renditions_do_not_need_cli_output_flags(tmp_path):
    cfg = show()
    app = build_application(cfg, api=fake_api())
    nodes = {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    assert nodes["record_output"]["url"] == "program.ts"
    assert nodes["record_fps"]["fps"] == "60/1"
    assert app.mixer.parameters["fps"] == (60, 1)
    # The show's settings, with the NVENC load of its outputs; its media cannot be probed here: no decode load.
    assert json.loads(app.avp.commands_registered["mixer.settings"]("")) == {
        **cfg.settings(), "encode_by_codec": [{"codec": "h264", "sessions": 1, "fps": 60}]}
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


def test_settings_report_what_every_decoder_and_encoder_of_the_show_does(native_boundary, monkeypatch, caplog):
    class Engine(FakeAvp):
        def addNode(self, node, **kwargs):
            super().addNode(node)

    cfg = show()
    hevc = replace(cfg.renditions[0], id="hevc", target="hevc.ts", codec="hevc_nvenc", fps=30)
    cfg = replace(cfg, renditions=(*cfg.renditions, hevc), aux_buses=config.parse_aux_buses([{
        "id": "cameras", "layout": {"preset": "source_pages"}, "renditions": [{"id": "monitor", "port": 5012}],
    }], cfg))
    api = fake_api()
    api.AVPlumber, api.MixerGraphBuilder = Engine, native_boundary[1]

    def settings():
        return json.loads(build_application(cfg, api=api).avp.commands_registered["mixer.settings"](""))
    # The program record at 60 fps and the aux monitor at the bus's 30 share H.264; the second record is HEVC.
    encode = [{"codec": "h264", "sessions": 2, "fps": 90}, {"codec": "hevc", "sessions": 1, "fps": 30}]
    streams = {"/media/cam.mp4": {"codec": "hevc", "width": 1920, "height": 1080, "fps": 30.0},
               "idle.mp4": {"codec": "h264", "width": 1280, "height": 720, "fps": 25.0}}
    monkeypatch.setattr(config, "probe_video_stream", streams.__getitem__)
    reported = settings()
    assert reported["encode_by_codec"] == encode
    assert reported["decode"] == {"count": 2, "fps": 55.0, "mpix_s": 85.25, "by_codec": [
        {"codec": "h264", "sources": 1, "fps": 25.0}, {"codec": "hevc", "sources": 1, "fps": 30.0}]}
    # A source that cannot be probed leaves the decode load out, never a total without it.
    del streams["idle.mp4"]
    reported = settings()
    assert "decode" not in reported and reported["encode_by_codec"] == encode
    assert "Decode load unknown" in caplog.text


def test_audio_sources_are_reported_only_where_their_gain_can_be_set(native_boundary):
    """audio_sources is what makes an operator page poll mixer.audio_gain_status, and that
    command is registered only alongside the in-tree audio_mixer. A show handing its audio to
    a router the caller builds has the sources but not the command, and the poll's failure
    costs /api/state its whole answer -- every page loses the show, not just its faders."""
    def settings(options):
        api = fake_api()
        api.MixerGraphBuilder = native_boundary[1]
        return json.loads(build_application(show(has_audio=True), options,
                                            api=api).avp.commands_registered["mixer.settings"](""))
    assert settings(MixerOptions(audio_router_node="audio_mixer"))["audio_sources"] == ["cam", "idle"]
    assert "audio_sources" not in settings(
        MixerOptions(audio_router_node="dyremix", build_audio_mixer=False))
    # A show with no audio at all keeps reporting the empty list it always did.
    assert json.loads(build_application(show(), api=fake_api())
                      .avp.commands_registered["mixer.settings"](""))["audio_sources"] == []


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
            app.start()   # the show goes on without that source
            assert app.avp.ready
            assert app.absent_inputs == ("processed_cam",)
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


def go_on_air(monkeypatch, counters, buses=("aux_1",), *, start=True, **options):
    clock = Clock()
    monkeypatch.setattr(mixer_application, "time", clock)
    engine = CountingEngine(clock, counters)
    app = mixer_application.MixerApplication(
        avp=engine, mixer=None, input_groups=(), input_edges=(), routed_inputs=False, preheat_timeout_sec=1,
        held_outputs=("janus",), aux_buses=[SimpleNamespace(prefix=prefix, listener=True) for prefix in buses], **options)
    if start:
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


def test_a_show_that_never_settles_goes_on_air_at_the_timeout_flagged(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="mixer")
    app, engine, clock = go_on_air(monkeypatch, {
        "mixer_comp_a": lambda now: (published(30)(now), 0, 0),
        "mixer_comp_b": idle,
        "aux_1_comp": lambda now: (published(15)(now), 0, 0),
        "aux_2_comp": lambda now: (published(15)(now), int(now), int(now) // 2),   # a miss a second, a drop every other
    }, buses=("aux_1", "aux_2"), start_stable_sec=5, start_stable_timeout_sec=20)
    # Released as a settled show is, the counters reset first; the report and a warning say it never settled.
    assert engine.at(" reset_counters true") == [20.0] * 4 and engine.at(" hold false") == [20.0] * 3
    reason = "aux_2_comp missed 1 deadlines and dropped 1 frames"
    assert app.startup["settled"] is False and app.startup["reason"] == reason and app.startup["waited_sec"] == 20.0
    on_air = [r for r in caplog.records if r.getMessage().startswith("On air")]
    assert len(on_air) == 1 and on_air[0].levelno == logging.WARNING
    assert f"after 20.0 s: 3 outputs released; never stable for 5 s ({reason})" in on_air[0].getMessage()


@pytest.mark.parametrize("stuck", ["program", "bus"])
def test_a_compositor_that_draws_nothing_holds_the_outputs_until_the_timeout(monkeypatch, stuck):
    counters = {
        "mixer_comp_a": idle if stuck == "program" else lambda now: (published(30)(now), 0, 0),
        "mixer_comp_b": idle,
        "aux_1_comp": idle if stuck == "bus" else lambda now: (published(15)(now), 0, 0),
    }
    app, engine, clock = go_on_air(monkeypatch, counters, start_stable_sec=3, start_stable_timeout_sec=12)
    # No miss anywhere (a compositor that cannot tick misses nothing), yet nothing drew: not stable.
    reason = {"program": "mixer_comp_a or mixer_comp_b drew no frame in 12 s",
              "bus": "aux_1_comp drew no frame in 12 s"}[stuck]
    assert engine.at(" hold false") == [12.0, 12.0]
    assert app.startup["settled"] is False and app.startup["reason"] == reason


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
    # A program miss at 1 s and a bus drop at 5 s: the window starts again at 1 s and ends at 7 s.
    # The drop is counted but does not restart it (a GPU time-sharing episode drops on every bus).
    app, engine, clock = go_on_air(monkeypatch, {
        "mixer_comp_a": lambda now: (published(30)(now), int(now >= 1), 0),
        "mixer_comp_b": idle,
        "aux_1_comp": lambda now: (published(15)(now), 0, int(now >= 5)),
    }, start_stable_sec=6, start_stable_timeout_sec=60)
    assert engine.at(" hold false") == [7.0, 7.0]
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
    assert "Browser counters unavailable" in caplog.text

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


def test_program_delivery_and_explicit_static_cadence():
    cfg = show()
    app = build_application(cfg, api=fake_api())
    nodes = {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    assert app.delivery_monitor
    assert nodes["mixer_delivery"]["on_change_sources"] == []
    assert nodes["program_fps"]["src"] == "mixer_final_out"
    assert nodes["program_fps"]["dst"] == nodes["mixer_delivery"]["src"] == "program_clocked"
    assert nodes["scale_record"]["src"] == "program_measured"
    doc = {**CONFIG, "sources": [CONFIG["sources"][0], {**CONFIG["sources"][1], "capture_cadence": "on_change"}]}
    assert config.parse(doc).sources[1].capture_cadence == "on_change"
    doc["sources"][1]["capture_cadence"] = "guess"
    with pytest.raises(config.ConfigError, match="capture_cadence"):
        config.parse(doc)


def test_production_api_exposes_program_delivery_node(native_boundary):
    from pyplumber.mixer.application import load_avp_api
    assert load_avp_api().MixerDelivery.TYPE == "mixer_delivery"


def test_options_the_show_overrides_are_refused_instead_of_ignored():
    cfg = show()   # 60 fps canvas, a recording rendition
    with pytest.raises(ValueError) as error:
        build_application(cfg, MixerOptions(fps=25, max_compositor_layers=7, output="other.mp4",
                                            codec="hevc_nvenc", bitrate="20M"), api=fake_api())
    for option in ("fps=25", "max_compositor_layers=7", "output='other.mp4'", "codec='hevc_nvenc'", "bitrate='20M'"):
        assert option in str(error.value)
    # Defaults, and values equal to the show's, are not conflicts.
    build_application(cfg, MixerOptions(fps=cfg.fps, max_compositor_layers=cfg.max_compositor_layers), api=fake_api())
    build_application(cfg, MixerOptions(), api=fake_api())


class _Group:
    def __init__(self, avp, name):
        self.avp, self.name = avp, name

    def startNodes(self):
        if self.name in self.avp.broken:
            raise RuntimeError("decoder did not open")
        self.avp.started.append(self.name)


class _SourcesAvp:
    """Sources that start, sources that refuse to, and edges that see frames or do not."""

    def __init__(self, broken=(), silent=()):
        self.broken, self.silent, self.started, self.commands = set(broken), set(silent), [], []

    def group(self, name):
        return _Group(self, name)

    def getEdge(self, edge, data_type=None):
        return type("Edge", (), {"enqueued_total": 0 if edge in self.silent else 1})()

    def executeCommandsFromString(self, text):
        self.commands.extend(text.splitlines())


def _sources_app(avp, sources=3, aliases=1):
    return mixer_application.MixerApplication(
        avp=avp, mixer=None, routed_inputs=False, preheat_timeout_sec=0.01,
        input_groups=tuple(f"input_{i}" for i in range(sources)), input_edges=tuple(f"edge_{i}" for i in range(sources)),
        compositor_inputs=tuple(tuple(range(i * aliases, (i + 1) * aliases)) for i in range(sources)))


def test_a_source_that_cannot_start_stays_out_and_the_others_start():
    avp = _SourcesAvp(broken={"input_1"})
    app = _sources_app(avp)
    assert app._start_inputs() == ("edge_1",)
    assert avp.started == ["input_0", "input_2"]


def test_a_show_with_no_source_left_does_not_start():
    with pytest.raises(RuntimeError, match="no source started"):
        _sources_app(_SourcesAvp(broken={"input_0", "input_1", "input_2"}))._start_inputs()


def test_a_silent_source_is_given_up_at_the_timeout_when_others_gave_frames():
    avp = _SourcesAvp(silent={"edge_2"})
    assert _sources_app(avp)._wait_for_edges(("edge_0", "edge_2"), "input readiness", required=False) == ("edge_2",)


def test_every_edge_silent_is_still_a_failed_start():
    avp = _SourcesAvp(silent={"edge_0", "edge_1"})
    with pytest.raises(RuntimeError, match="preheat timed out"):
        _sources_app(avp)._wait_for_edges(("edge_0", "edge_1"), "input readiness", required=False)


def test_a_required_edge_still_fails_the_start():
    with pytest.raises(RuntimeError, match="preheat timed out"):
        _sources_app(_SourcesAvp(silent={"edge_1"}))._wait_for_edges(("edge_0", "edge_1"), "warm-up")


def test_no_tick_waits_for_an_absent_source_on_any_of_its_aliases():
    avp = _SourcesAvp()
    app = _sources_app(avp, aliases=2)
    app.absent_inputs = ("edge_1",)
    app._release_absent_inputs()
    assert avp.commands == [f"node.object.set mixer_comp_{slot} release_input {i}" for slot in "ab" for i in (2, 3)]


@pytest.mark.parametrize("fault", ["missed", "dropped", "missing"])
def test_managed_browser_start_keeps_outputs_held_on_unstable_graph(monkeypatch, fault):
    monkeypatch.setenv("AVP_MIXER_STARTUP_FD", "7")
    counters = {
        "mixer_comp_a": lambda now: (published(30)(now), 0, 0),
        "mixer_comp_b": idle,
        "aux_1_comp": lambda now: (published(30)(now), int(now) if fault == "missed" else 0,
                                   int(now) if fault == "dropped" else 0),
    }
    if fault == "missing":
        del counters["aux_1_comp"]
    app, engine, _ = go_on_air(monkeypatch, counters, start=False, browser_windows=("browser",),
                               start_stable_sec=3, start_stable_timeout_sec=10)
    with pytest.raises(mixer_application.BrowserStartupUnsettled):
        app._go_on_air()
    assert not engine.at(" hold false")
    assert not engine.at(" reset_counters true")


def test_the_v210_unpack_kernels_of_the_node_and_of_the_ffmpeg_patch_are_the_same(tmp_path):
    """hwupload_cuda=v210_width (FFmpeg patch 0017: the demuxer chain, v210 files) and
    raw_video_file_cuda unpack with the same kernels, kept in two files; a change belongs in both."""
    repo = Path(__file__).resolve().parents[2]
    source = "libavfilter/vf_hwupload_cuda.cu"
    patches = sorted((repo / "deps/ffmpeg/9").glob("*.patch"))
    subprocess.run(["git", "apply", f"--include={source}", *patches], cwd=tmp_path, check=True)
    kernels = (tmp_path / source).read_text()
    start = "#define INLINE"
    # The node's file goes on with the UYVY and RGB unpacks (packed_to_cuda), which the patch does not have.
    node = (repo / "src/nodes/hwaccel/v210_pack.cu").read_text().split("\n// ---- Unpack: UYVY rows")[0]
    assert node[node.index(start):].strip() == kernels[kernels.index(start):].strip()
