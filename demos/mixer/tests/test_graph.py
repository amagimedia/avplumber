import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyplumber.mixer import config as mixer_config
from pyplumber.mixer.color import Color
from mixer import GraphOptions, build_application, infer_output_format, parse_args


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
        self.armed_clips = []
        self.control_port = None
        self.ready = False
        self.edges = FakeEdges()

    def addNode(self, node):
        self.nodes.append(node)

    def executeCommandsFromString(self, commands):
        self.commands.append(commands)
        for line in commands.splitlines():
            parts = line.split()
            if parts[:1] == ["node.param.set"] and parts[2:3] == ["url"]:
                self.armed_clips.append(json.loads(parts[3]))

    def enableControlServer(self, port):
        self.control_port = port

    def node(self, name):
        # The clip cache reports every requested clip as already held, so start()
        # exercises the preload path without a decoder.
        return SimpleNamespace(getObject=lambda key: {
            "clips": [{"path": path, "frames": 30, "bytes": 1 << 20, "complete": True}
                      for path in self.armed_clips],
            "bytes": 1 << 20, "budget_bytes": 768 << 20})

    def registerControlCommand(self, name, handler, _payload):
        self.commands_registered = getattr(self, "commands_registered", {})
        self.commands_registered[name] = handler

    def setReady(self):
        self.ready = True

    def group(self, _name):
        return FakeGroup()


class FakeMixer:
    instances = []

    def __init__(self, avp, **parameters):
        self.avp = avp
        self.parameters = parameters
        self.timeline = "mixer_timeline"
        self.routed_sources = []
        self.sources = []
        self.scenes = {}
        self.initial_scene = None
        self.instances.append(self)

    def add_source(self, name, **parameters):
        self.sources.append((name, parameters))

    def add_routed_source(self, name, **parameters):
        self.routed_sources.append((name, parameters))

    def add_scene(self, name, sources, *, routes=None):
        self.scenes[name] = {"sources": sources, "routes": routes}

    def set_initial_scene(self, name, slot):
        self.initial_scene = (name, slot)

    def build(self):
        return "mixer_final_out"

    def initialize_routes(self):
        pass

    def begin_transition_preheat(self):
        pass

    def finish_transition_preheat(self):
        pass

    def start_output(self):
        pass

    def warmup_wipe(self, wipe_file, timeout_ms=30000):
        self.warmed_wipe = (wipe_file, timeout_ms)

    def start_groups(self):
        pass


class FakeRtcpFeedbackListener:
    def __init__(self, **parameters):
        self.parameters = parameters
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


def fake_api():
    names = (
        "assume_video_format",
        "bsf",
        "clip_cache",
        "dec_video",
        "demux",
        "drm_prime_to_cuda",
        "enc_video",
        "filter_video",
        "force_fps",
        "force_key_frame",
        "input_rec",
        "ipc_dmabuf_source",
        "mux",
        "one_to_many",
        "output",
        "preheat_video_router",
        "realtime",
        "repeat_last_frame",
        "smooth_timestamps",
        "split",
        "v210_to_cuda",
    )
    api = {
        "AVPlumber": FakeAvp,
        "MixerGraphBuilder": FakeMixer,
        "RtcpFeedbackListener": FakeRtcpFeedbackListener,
    }
    api.update({
        {
            "assume_video_format": "AssumeVideoFormat",
            "bsf": "Bsf",
            "clip_cache": "ClipCache",
            "dec_video": "DecVideo",
            "demux": "Demux",
            "drm_prime_to_cuda": "DrmPrimeToCuda",
            "enc_video": "EncVideo",
            "filter_video": "FilterVideo",
            "force_fps": "ForceFPS",
            "force_key_frame": "ForceKeyFrame",
            "input_rec": "InputRec",
            "ipc_dmabuf_source": "IpcDmabufSource",
            "mux": "Mux",
            "one_to_many": "OneToMany",
            "output": "Output",
            "preheat_video_router": "PreheatVideoRouter",
            "realtime": "Realtime",
            "repeat_last_frame": "RepeatLastFrame",
            "smooth_timestamps": "SmoothTimestamps",
            "split": "Split",
            "v210_to_cuda": "V210ToCuda",
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

    assert "preheat_video_router" not in node_types
    assert len(mixer.sources) == 17
    assert not mixer.routed_sources
    assert all(params["default_graph"] == "" for _, params in mixer.sources)
    assert mixer.parameters["defer_initial_routes"] is True
    assert mixer.parameters["enable_wipe"] is True
    assert not any(
        forbidden in node_type_name
        for node_type_name in node_types
        for forbidden in ("audio", "vad", "infer", "face", "speaker")
    )
    assert mixer.initial_scene == ("fullscreen_0", "A")
    normalizers = [
        node.parameters
        for node in application.avp.nodes
        if node.parameters["type"] == "filter_video"
        and node.parameters["name"].startswith("normalize_")
    ]
    assert normalizers == []

    serialized_graph = repr([node.parameters for node in application.avp.nodes]).lower()
    assert "hwdownload" not in serialized_graph
    assert "hwupload" not in serialized_graph


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
    assert set(scenes["grid_2_page_1"]["sources"]) == {"source_2"}
    assert scenes["grid_2_page_1"]["routes"] == {}


def test_output_format_inference_is_explicit_when_ambiguous():
    assert infer_output_format("rtmp://example.invalid/live") == "flv"
    assert infer_output_format("srt://example.invalid:9000") == "mpegts"
    assert infer_output_format("program.mp4") == "mp4"
    assert infer_output_format("anything", "nut") == "nut"


def test_cli_accepts_ordered_repeatable_input_paths():
    options = parse_args([
        "--input",
        "/media/first clip.mp4",
        "--input",
        "/media/second.mp4",
        "--output",
        "program.mp4",
        "--fps",
        "60",
        "--loop-inputs",
    ])

    assert options.inputs == ("/media/first clip.mp4", "/media/second.mp4")
    assert options.fps == 60
    assert options.loop_inputs is True


def test_configured_fps_reaches_input_mixer_and_outputs():
    FakeMixer.instances.clear()
    application = build_application(
        GraphOptions(
            inputs=("input.mp4",),
            output="program.ts",
            janus_output=True,
            fps=60,
        ),
        api=fake_api(),
    )
    nodes = {node.parameters["name"]: node.parameters for node in application.avp.nodes}

    assert nodes["fps_0"]["fps"] == "60/1"
    assert "normalize_0" not in nodes
    assert nodes["program_fps"]["fps"] == "60/1"
    assert nodes["program_encoder"]["options"]["g"] == 120
    assert nodes["janus_fps"]["fps"] == "60/1"
    assert nodes["janus_encoder"]["options"]["g"] == 60
    assert "level" not in nodes["janus_encoder"]["options"]
    assert FakeMixer.instances[-1].parameters["fps"] == (60, 1)


def test_janus_only_output_builds_video_rtp_and_feedback():
    application = build_application(
        GraphOptions(inputs=("input.mp4",), janus_output=True),
        api=fake_api(),
    )
    nodes = {node.parameters["name"]: node.parameters for node in application.avp.nodes}

    assert nodes["janus_encoder"]["codec"] == "h264_nvenc"
    assert nodes["janus_rtp_output"]["format"] == "rtp"
    assert nodes["janus_rtp_output"]["options"]["payload_type"] == 96
    assert application.rtcp_feedback_listener is not None


def test_record_and_janus_outputs_split_program_video():
    application = build_application(
        GraphOptions(
            inputs=("input.mp4",),
            output="program.mp4",
            janus_output=True,
        ),
        api=fake_api(),
    )
    nodes = {node.parameters["name"]: node.parameters for node in application.avp.nodes}

    # Both flags become renditions of the one program, like a config document's.
    assert nodes["split_renditions"]["dst"] == ["program_rendition_program", "program_rendition_janus"]
    assert nodes["program_encoder"]["codec"] == "h264_nvenc" and nodes["program_encoder"]["options"]["b"] == "8000k"


def test_output_target_is_required():
    with pytest.raises(ValueError, match="--output or --janus-output"):
        build_application(GraphOptions(inputs=("input.mp4",)), api=fake_api())


def test_cpu_encoder_is_rejected():
    with pytest.raises(ValueError, match="NVENC"):
        build_application(
            GraphOptions(inputs=("input.mp4",), output="program.mp4", codec="libx264"),
            api=fake_api(),
        )


def test_large_catalogue_keeps_paging_without_geometry_filters():
    app = build_application(GraphOptions(inputs=tuple(f"source-{i}" for i in range(65)),
                                          output="program.ts"), api=fake_api())
    mixer = FakeMixer.instances[-1]
    assert not mixer.sources
    assert len(mixer.routed_sources) == 16
    assert app.routed_inputs
    assert len([n for n in app.avp.nodes if n.parameters["name"].startswith("normalize_")]) == 65
    assert mixer.scenes["fullscreen_64"]["routes"] == {"source_0": 64}
    assert mixer.scenes["grid_16_page_4"]["routes"] == {"source_0": 64}
    assert all(params["default_graph"] == "" for _, params in mixer.routed_sources)


def test_dmabuf_input_builds_browser_chain_next_to_files(tmp_path):
    FakeMixer.instances.clear()
    (tmp_path / "page_00.sock").touch()
    application = build_application(
        GraphOptions(inputs=("camera.mp4", "dmabuf://page_00"), output="program.mp4",
                     dmabuf_socket_dir=str(tmp_path), dmabuf_size=(480, 270), fps=60),
        api=fake_api(),
    )
    nodes = {node.parameters.get("name"): node.parameters for node in application.avp.nodes}

    assert not any("drm" in c for c in application.avp.commands)   # DRM frames are imported by CUDA directly
    receive = nodes["input_1_receive"]
    assert receive["type"] == "ipc_dmabuf_source" and "hwaccel" not in receive
    assert receive["socket"] == str(tmp_path / "page_00.sock")
    assert receive["fps"] == "60/1" and receive["group"] == "input_1"
    assert nodes["input_1_to_cuda"]["type"] == "drm_prime_to_cuda"
    smooth = nodes["input_1_smooth"]
    assert (smooth["type"], smooth["src"], smooth["dst"]) == ("smooth_timestamps", "input_1_cuda_raw", "input_1_cuda_smooth")
    assert smooth["fps"] == "60/1" and smooth["discontinuity_threshold"] == 0.1
    stamp = nodes["input_1_timestamp"]
    assert (stamp["dst_width"], stamp["dst_height"], stamp["dst_frame_rate"]) == (480, 270, "60/1")
    assert (stamp["src"], stamp["dst"]) == ("input_1_cuda_smooth", "input_1_cuda")
    assert "round(PTS" not in stamp["graph"]   # arrival times are numbered, not rounded
    hold = nodes["input_1_hold"]
    assert (hold["type"], hold["src"], hold["dst"], hold["fps"]) == ("repeat_last_frame", "input_1_cuda", "input_1_held", "60/1")
    assert "decode_1" not in nodes and "decode_0" in nodes
    sources = dict(FakeMixer.instances[-1].sources)
    assert sources["source_1"]["pre_otm_edge"] == "input_1_held"
    assert sources["source_0"]["pre_otm_edge"] == "input_0_fps"


def test_file_inputs_do_not_touch_drm_or_sockets(tmp_path):
    FakeMixer.instances.clear()
    application = build_application(
        GraphOptions(inputs=("a.mp4",), output="p.mp4", dmabuf_socket_dir=str(tmp_path / "missing")),
        api=fake_api(),
    )
    assert not any("drm" in c for c in application.avp.commands)


def test_cli_parses_dmabuf_options():
    options = parse_args(["--input", "dmabuf://page_03", "--janus-output", "--dmabuf-size", "480x270",
                          "--dmabuf-open", "http://pages/smoke.html"])
    assert options.dmabuf_inputs == ["page_03"]
    assert options.dmabuf_size == (480, 270)
    assert options.dmabuf_open == "http://pages/smoke.html"
    with pytest.raises(ValueError):
        parse_args(["--input", "dmabuf://x", "--janus-output", "--dmabuf-size", "wide"])
    with pytest.raises(ValueError):
        GraphOptions(inputs=("dmabuf://", ), output="p.mp4").validate()


def test_dmabuf_windows_are_closed_before_reopening(monkeypatch):
    from pyplumber.mixer import dmabuf_inputs

    calls = []

    def fake_rest(base_url, method, path, body=None):
        calls.append((method, path, body))
        if path == "/status":
            return {"windows": [{"id": "page_00"}]}
        return {"ok": True}

    monkeypatch.setattr(dmabuf_inputs, "rest_request", fake_rest)
    dmabuf_inputs.open_browser_windows("http://b", ["page_00", "page_01"], "http://p", 480, 270, 60)

    assert [c[:2] for c in calls] == [
        ("GET", "/status"), ("POST", "/window/close"), ("POST", "/window/open"), ("POST", "/window/open")]
    assert calls[1][2] == {"id": "page_00"}
    assert calls[3][2] == {"id": "page_01", "url": "http://p", "width": 480, "height": 270, "fps": 60,
                           "audio": False}


def test_unchanged_browser_windows_survive_reconfiguration(monkeypatch):
    from pyplumber.mixer import dmabuf_inputs

    unchanged = dict(id="page_00", url="http://p", width=480, height=270, fps=60, audio=False)
    calls = []

    def fake_rest(base_url, method, path, body=None):
        calls.append((method, path, body))
        if path == "/status":
            return {"windows": [unchanged, {**unchanged, "id": "page_01", "fps": 30}]}

    monkeypatch.setattr(dmabuf_inputs, "rest_request", fake_rest)
    dmabuf_inputs.open_browser_windows("http://b", ["page_00", "page_01", "page_02"], "http://p", 480, 270, 60)
    assert calls == [
        ("GET", "/status", None),
        ("POST", "/window/close", {"id": "page_01"}),
        ("POST", "/window/open", {**unchanged, "id": "page_01"}),
        ("POST", "/window/open", {**unchanged, "id": "page_02"}),
    ]


def test_quarantined_browser_is_not_reused_or_recreated(monkeypatch):
    from pyplumber.mixer import dmabuf_inputs

    calls = []
    def fake_rest(base_url, method, path, body=None):
        calls.append(path)
        return {"windows": [{"id": "page_00", "stats": {"quarantinedFrameCount": 1}}]}

    monkeypatch.setattr(dmabuf_inputs, "rest_request", fake_rest)
    with pytest.raises(RuntimeError, match="quarantined DMA-BUF"):
        dmabuf_inputs.open_browser_windows("http://b", ["page_00"], "http://p", 480, 270, 60)
    assert calls == ["/status"]


def test_browser_failure_precedes_native_initialization(monkeypatch):
    import mixer

    def fail(*args):
        raise RuntimeError("browser unavailable")

    monkeypatch.setattr(mixer, "open_browser_windows", fail)
    monkeypatch.setattr(mixer, "_init_avp", lambda *args: pytest.fail("native threads started before browser preparation"))
    with pytest.raises(RuntimeError, match="browser unavailable"):
        build_application(GraphOptions(inputs=("dmabuf://page_00",), output="p.mp4", dmabuf_open="http://p"), api=fake_api())
    monkeypatch.setattr(mixer, "open_windows", fail)
    cfg = SimpleNamespace(fps=60, latency_ms=None, sources=[SimpleNamespace(
        kind="browser", id="page_00", location="http://p", width=480, height=270, fps=60)])
    with pytest.raises(RuntimeError, match="browser unavailable"):
        mixer._build_from_config(GraphOptions(output="p.mp4"), cfg, fake_api())


def test_startup_failure_shuts_down_application(monkeypatch):
    import mixer

    stopped = []
    app = SimpleNamespace(stop=lambda: stopped.append(True))
    monkeypatch.setattr(mixer, "build_application", lambda options: app)
    def fail(*args):
        raise RuntimeError("preheat failed")
    monkeypatch.setattr(mixer, "_run_application", fail)
    with pytest.raises(RuntimeError, match="preheat failed"):
        mixer.main(["--input", "a.mp4", "--output", "out.mp4"])
    assert stopped == [True]


def test_browser_http_error_includes_worker_reason(monkeypatch):
    import io
    import urllib.error
    from pyplumber.mixer import dmabuf_inputs

    def unavailable(*args, **kwargs):
        raise urllib.error.HTTPError("http://b/window/open", 503, "Service Unavailable", {},
                                     io.BytesIO(b'{"error":"Electron worker 0 is not ready"}'))
    monkeypatch.setattr(dmabuf_inputs.urllib.request, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="HTTP 503.*Electron worker 0 is not ready"):
        dmabuf_inputs.rest_request("http://b", "POST", "/window/open", {"id": "page_00"})


def test_wipe_file_preloads_into_the_clip_cache_at_start(monkeypatch):
    FakeMixer.instances.clear()
    application = build_application(
        GraphOptions(inputs=("a.mp4",), output="p.mp4", wipe_file="/media/wipe.mov", wipe_cache_mb=256), api=fake_api())
    monkeypatch.setattr(application, "_wait_for_edges", lambda *a, **k: None)
    monkeypatch.setattr(application, "_wait_for_node", lambda *a, **k: None)
    application.start()
    # Caching opted in: the clip is armed on the loader and decoded once, so
    # mixer.wipe.warmup (which only compiles the filter) is not used.
    armed = "\n".join(application.avp.commands)
    assert 'node.param.set mixer_wipe_input url "/media/wipe.mov"' in armed
    assert 'node.param.set mixer_wipe_cache url "/media/wipe.mov"' in armed
    assert not hasattr(FakeMixer.instances[-1], "warmed_wipe")
    assert application.avp.ready

    FakeMixer.instances.clear()
    plain = build_application(GraphOptions(inputs=("a.mp4",), output="p.mp4"), api=fake_api())
    monkeypatch.setattr(plain, "_wait_for_edges", lambda *a, **k: None)
    monkeypatch.setattr(plain, "_wait_for_node", lambda *a, **k: None)
    plain.start()
    assert not any("node.param.set" in c for c in plain.avp.commands)


CONFIG = {
    "canvas": {"width": 1920, "height": 1080, "fps": 60},
    "sources": [
        {"id": "cam", "kind": "video", "path": "/media/cam.mp4", "width": 1920, "height": 1080},
        {"id": "page", "kind": "browser", "color": "sdr", "url": "https://example.org/", "width": 1280, "height": 720},
    ],
    "wipes": [{"id": "swoosh", "path": "/media/swoosh.mov"}],
    "control": {"direct": False, "fade_seconds": 0.8},
    "scenes": [
        {"id": "full", "items": [{"source": "cam", "dst": {"x": 0, "y": 0, "w": 1920, "h": 1080}, "fit": "cover"}]},
        {"id": "pip", "items": [
            {"source": "page", "dst": {"x": 0, "y": 0, "w": 1920, "h": 1080}},
            {"source": "cam", "dst": {"x": 1440, "y": 60, "w": 420, "h": 236}},
            {"source": "cam", "dst": {"x": 60, "y": 60, "w": 420, "h": 236},
             "crop": {"x": 480, "y": 270, "w": 960, "h": 540}}]},
    ],
    "initial_scene": "pip",
}


def test_config_scene_layers_carry_z_cover_and_aliases():
    from pyplumber.mixer import config as mc
    cfg = mc.parse(CONFIG)
    assert cfg.alias_counts == {"cam": 2, "page": 1}
    full = mc.scene_layers(cfg, cfg.scenes[0])
    assert full["cam"] == {"dst_x": 0, "dst_y": 0, "dst_w": 1920, "dst_h": 1080, "z": 0, "fit": "stretch",
                           "crop": {"x": 0, "y": 0, "w": 1920, "h": 1080}}
    pip = mc.scene_layers(cfg, cfg.scenes[1])
    assert list(pip) == ["page", "cam", "cam#2"]
    assert [layer["z"] for layer in pip.values()] == [0, 1, 2]
    assert pip["cam#2"]["crop"] == {"x": 480, "y": 270, "w": 960, "h": 540} and pip["cam#2"]["fit"] == "contain"
    # cover of a 16:9 source into a 9:16 box keeps the full height and a centred slice
    tall = mc.cover_crop(1920, 1080, mc.Rect(0, 0, 1080, 1920), None)
    assert (tall.w, tall.h, tall.x) == (606, 1080, 657)


def test_config_rejects_duplicate_locations_and_bad_references():
    from pyplumber.mixer import config as mc
    import copy
    dup = copy.deepcopy(CONFIG)
    dup["sources"].append({"id": "cam2", "kind": "video", "path": "/media/cam.mp4"})
    with pytest.raises(mc.ConfigError, match="already declared"):
        mc.parse(dup)
    bad = copy.deepcopy(CONFIG)
    bad["scenes"][0]["items"][0]["source"] = "nope"
    with pytest.raises(mc.ConfigError, match="unknown source"):
        mc.parse(bad)
    bad_filter = copy.deepcopy(CONFIG)
    bad_filter["sources"][0]["filter"] = {"transfer_in": "sdr"}
    with pytest.raises(mc.ConfigError, match="filter must be a CUDA filter graph string"):
        mc.parse(bad_filter)
    nocover = copy.deepcopy(CONFIG)
    del nocover["sources"][0]["width"]
    cfg = mc.parse(nocover)
    with pytest.raises(mc.ConfigError, match="needs its size"):
        mc.scene_layers(cfg, cfg.scenes[0])
    probed = mc.with_probed_sizes(cfg, probe=lambda path: (640, 360))
    assert (probed.source("cam").width, probed.source("cam").height) == (640, 360)
    assert probed.source("page").width == 1280          # declared sizes are kept
    # cover of a 16:9 clip into the full 16:9 box keeps the whole frame
    assert mc.scene_layers(probed, probed.scenes[0])["cam"]["crop"] == {"x": 0, "y": 0, "w": 640, "h": 360}


@pytest.mark.parametrize("flags", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_sources_require_explicit_opt_in_on_each_declaration(flags):
    sources = [{**CONFIG["sources"][0], "id": sid, "independent": independent}
               for sid, independent in zip(("cam", "cam2"), flags)]
    doc = {**CONFIG, "sources": sources, "scenes": CONFIG["scenes"][:1], "initial_scene": "full"}
    if all(flags):
        assert len(mixer_config.parse(doc).sources) == 2
    else:
        with pytest.raises(mixer_config.ConfigError, match="already declared"):
            mixer_config.parse(doc)


def test_independent_sources_build_separate_decoders(tmp_path):
    sources = [{**CONFIG["sources"][0], "id": sid, "independent": True} for sid in ("cam", "cam2")]
    doc = {**CONFIG, "sources": sources, "scenes": CONFIG["scenes"][:1], "initial_scene": "full"}
    path = tmp_path / "mixer.json"
    path.write_text(json.dumps(doc))
    application = build_application(GraphOptions(config=str(path), output="p.mp4"), api=fake_api())
    nodes = {node.parameters.get("name"): node.parameters for node in application.avp.nodes}
    assert {n for n in nodes if n and n.startswith("decode_")} == {"decode_0", "decode_1"}
    sources = dict(FakeMixer.instances[-1].sources)
    assert sources["cam"]["pre_otm_edge"] != sources["cam2"]["pre_otm_edge"]


def test_recipe_builds_all_independent_input_chains_and_renditions(tmp_path):
    pytest.importorskip("numpy")
    from prepare_demo import plan
    recipe = json.loads((Path(__file__).resolve().parents[1] / "demo.example.json").read_text())
    doc, _, _ = plan(recipe, tmp_path)
    path = tmp_path / "show.json"
    path.write_text(json.dumps(doc))
    application = build_application(GraphOptions(config=str(path), janus_output=True), api=fake_api())
    nodes = {node.parameters.get("name"): node.parameters for node in application.avp.nodes}
    assert len([n for n in nodes if n and n.startswith("decode_")]) == 12
    assert len([n for n in nodes if n and n.startswith("unpack_")]) == 4
    mixer = FakeMixer.instances[-1]
    assert len(mixer.sources) == 16 and len(mixer.scenes) == 24
    assert len({params["pre_otm_edge"] for _, params in mixer.sources}) == 16
    assert nodes["janus_encoder"]["options"]["profile"] == "baseline"
    assert nodes["janus_hdr_encoder"]["options"]["profile"] == "main10"


@pytest.mark.parametrize("value", [0.5, "true", 1, None])
def test_scene_blend_rejects_non_boolean_values(value):
    scene = {"id": "blend", "items": [{**CONFIG["scenes"][0]["items"][0], "blend": value}]}
    with pytest.raises(mixer_config.ConfigError, match="blend must be a boolean"):
        mixer_config.parse({**CONFIG, "scenes": [scene], "initial_scene": "blend"})


@pytest.mark.parametrize("blend", [False, True])
def test_browser_alpha_preservation_follows_scene_blending(tmp_path, monkeypatch, blend):
    from pyplumber.mixer import dmabuf_inputs
    (tmp_path / "page.sock").touch()
    monkeypatch.setattr(dmabuf_inputs, "rest_request", lambda *args, **kwargs: {"windows": []})
    scene = {"id": "overlay", "items": [CONFIG["scenes"][0]["items"][0],
             {**CONFIG["scenes"][1]["items"][0], "blend": blend}]}
    path = tmp_path / "show.json"
    path.write_text(json.dumps({**CONFIG, "scenes": [scene], "initial_scene": "overlay"}))
    app = build_application(GraphOptions(config=str(path), output="p.mp4", dmabuf_socket_dir=str(tmp_path)),
                            api=fake_api())
    nodes = {n.parameters.get("name"): n.parameters for n in app.avp.nodes}
    assert nodes["input_1_to_cuda"]["drop_alpha"] is not blend
    layer = FakeMixer.instances[-1].scenes["overlay"]["sources"]["page"]
    assert layer.get("blend", False) is blend


@pytest.mark.parametrize("source_filter", ["", "tonemap_cuda=transfer_in=sdr:transfer_out=hlg,scale_cuda=format=p210le"])
def test_config_builds_one_chain_per_source_with_alias_fanout(tmp_path, monkeypatch, source_filter):
    import json as _json
    from pyplumber.mixer import dmabuf_inputs
    (tmp_path / "page.sock").touch()
    path = tmp_path / "mixer.json"
    doc = {**CONFIG, "sources": [{**CONFIG["sources"][0], "filter": source_filter, "filter_output_format": "p210le" if source_filter else ""},
                                *CONFIG["sources"][1:]]}
    path.write_text(_json.dumps(doc))
    opened = []
    monkeypatch.setattr(dmabuf_inputs, "rest_request",
                        lambda base, method, p, body=None: opened.append((method, p, body)) or {"windows": []})
    from pyplumber.mixer import config as mc
    monkeypatch.setattr(mc, "probe_video_size", lambda path: (_ for _ in ()).throw(AssertionError("declared sizes must not be probed")))
    FakeMixer.instances.clear()
    application = build_application(
        GraphOptions(config=str(path), output="p.mp4", dmabuf_socket_dir=str(tmp_path)), api=fake_api())
    nodes = {node.parameters.get("name"): node.parameters for node in application.avp.nodes}
    mixer = FakeMixer.instances[-1]

    assert [name for name in nodes if name and name.startswith("decode_")] == ["decode_0"]
    source_edge = "input_0_filtered" if source_filter else "input_0_fps"
    assert dict(mixer.sources)["cam"]["pre_otm_edge"] == source_edge
    assert dict(mixer.sources)["cam#2"]["pre_otm_edge"] == source_edge
    if source_filter:
        assert nodes["source_filter_0"]["graph"] == source_filter
        assert nodes["source_filter_0"]["src"] == "input_0_fps"
        assert nodes["source_filter_0"]["hwaccel"] == "@gpu"
        assert nodes["source_filter_0"]["group"] == "input_0"
        assert [n for n in nodes if n and n.startswith("source_filter_")] == ["source_filter_0"]
    else:
        assert "source_filter_0" not in nodes
    assert "alias_0" not in nodes  # shared fan-out belongs to the reusable builder
    assert [name for name, _ in mixer.sources] == ["cam", "cam#2", "page"]
    assert dict(mixer.sources)["page"]["pre_otm_edge"] == "input_1_held"
    assert opened[-1][2] == {"id": "page", "url": "https://example.org/", "width": 1280, "height": 720,
                             "fps": 60, "audio": False}
    assert mixer.initial_scene == ("pip", "A") and set(mixer.scenes) == {"full", "pip"}
    assert mixer.parameters["canvas"] == (1920, 1080)
    assert application.wipe_files == ("/media/swoosh.mov",)
    assert application.browser_windows == ("page",)
    monkeypatch.setattr(application, "_wait_for_edges", lambda *a, **k: None)
    monkeypatch.setattr(application, "_wait_for_node", lambda *a, **k: None)
    del opened[:]
    application.start()
    assert ("POST", "/window/refresh", {"id": "page"}) in opened      # static pages repaint into the live chain
    assert json.loads(application.avp.commands_registered["mixer.settings"]("")) == {
        "source_count": 2,
        "preview_codecs": [],
        "canvas": {"width": 1920, "height": 1080, "fps": 60, "working_format": "nv12"},
        "source_counts": {"video": 1, "browser": 1, "v210": 0},
        "direct": False, "fade_seconds": 0.8, "transition": "cut",
        "wipe_file": "/media/swoosh.mov",
        "default_wipe": "swoosh",
        "wipes": [{"id": "swoosh", "name": "swoosh", "path": "/media/swoosh.mov",
                   "duration_seconds": 0.0}]}
    assert nodes["program_format"]["width"] == 1920
    assert nodes["program_fps"]["fps"] == "60/1"      # outputs follow the document's fps, not the CLI default


def test_example_configuration_uses_placeholder_locations_and_valid_layers():
    from pyplumber.mixer import config as mc

    path = Path(__file__).resolve().parents[1] / "config.example.json"
    cfg = mc.load(path)
    for source in cfg.sources:
        if source.kind == "browser":
            assert source.location == f"https://<host>/{source.id}"
        else:
            assert source.location.startswith("<path>/")
    assert all(w.path.startswith("<path>/") for w in cfg.wipes)
    cfg = mc.with_probed_sizes(cfg, probe=lambda _location: (1920, 1080))
    for scene in cfg.scenes:
        assert len(mc.scene_layers(cfg, scene)) == len(scene.items)


def test_cli_passes_the_web_ui_url_through_to_the_options():
    assert parse_args(["--config", "m.json", "--janus-output",
                       "--webui-url", "http://ui:22222"]).webui_url == "http://ui:22222"
    assert parse_args(["--config", "m.json", "--janus-output"]).webui_url == ""


def test_cli_requires_inputs_or_config():
    with pytest.raises(SystemExit):
        parse_args(["--output", "p.mp4"])
    assert parse_args(["--config", "m.json", "--janus-output"]).config == "m.json"


def test_transitions_trigger_a_keyframe_only_when_streaming():
    FakeMixer.instances.clear()
    application = build_application(GraphOptions(inputs=("a.mp4",), janus_output=True), api=fake_api())
    assert FakeMixer.instances[-1].parameters["keyframe_node"] == "janus_force_keyframe"
    node = next(n for n in application.avp.nodes if n.parameters.get("name") == "janus_force_keyframe")
    assert node.parameters["min_interval_ms"] == 150
    assert node.parameters["interval_sec"] == "1/1"

    FakeMixer.instances.clear()
    build_application(GraphOptions(inputs=("a.mp4",), output="p.mp4"), api=fake_api())
    assert FakeMixer.instances[-1].parameters["keyframe_node"] is None


@pytest.mark.parametrize("minimum", [0, 100, 150, 200, 500])
def test_janus_keyframe_limit_is_configurable(minimum):
    from pyplumber.mixer.janus import JanusVideoConfig, build_janus_output
    avp = FakeAvp()
    build_janus_output(avp, fake_api(), "program", JanusVideoConfig(keyframe_min_interval_ms=minimum),
                       fps=60, width=1920, height=1080)
    node = next(n for n in avp.nodes if n.parameters.get("name") == "janus_force_keyframe")
    assert node.parameters["min_interval_ms"] == minimum


@pytest.mark.parametrize("minimum", [-1, 0.2, True, "200", 2**31])
def test_janus_rejects_invalid_keyframe_limit(minimum):
    from pyplumber.mixer.janus import JanusVideoConfig
    with pytest.raises(ValueError, match="keyframe_min_interval_ms"):
        JanusVideoConfig(keyframe_min_interval_ms=minimum)


@pytest.mark.parametrize("bitrate,pacing", [(3000, "9000000"), (8000, "24000000")])
def test_janus_paces_rtp_without_reducing_encoding_quality(bitrate, pacing):
    from urllib.parse import parse_qs, urlsplit
    from pyplumber.mixer.janus import JanusVideoConfig, build_janus_output
    avp = FakeAvp()
    build_janus_output(avp, fake_api(), "program", JanusVideoConfig(bitrate_kbps=bitrate),
                       fps=30, width=1080, height=1920)
    nodes = {n.parameters["name"]: n.parameters for n in avp.nodes}
    output = nodes["janus_rtp_output"]
    url = urlsplit(output["url"])
    assert output["format"] == "rtp" and url.scheme == "udp"
    assert output["options"]["rtpflags"] == "skip_rtcp"
    assert url.port == 5004
    query = parse_qs(url.query)
    assert query["bitrate"] == [pacing]
    assert query["pkt_size"] == ["1200"]
    assert query["burst_bits"] == ["38400"]
    assert nodes["janus_encoder"]["options"]["b"] == f"{bitrate}k"
    assert nodes["janus_encoder"]["options"]["bufsize"] == f"{bitrate}k"


@pytest.mark.parametrize("configured", [False, True])
def test_mixer_keyframe_option_reaches_each_output_path(tmp_path, configured):
    if configured:
        path = tmp_path / "mixer.json"
        path.write_text(json.dumps({
            **CONFIG, "sources": CONFIG["sources"][:1], "scenes": CONFIG["scenes"][:1],
            "initial_scene": "full", "wipes": [],
            "renditions": [{"id": "program", "target": "janus"}],
        }))
        options = GraphOptions(config=str(path), janus_output=True, keyframe_min_interval_ms=200)
    else:
        options = parse_args(["--input", "a.mp4", "--janus-output", "--keyframe-min-interval-ms", "200"])
    application = build_application(options, api=fake_api())
    node = next(n for n in application.avp.nodes if n.parameters.get("name") == "janus_force_keyframe")
    assert node.parameters["min_interval_ms"] == 200
    assert parse_args(["--input", "a.mp4", "--janus-output"]).keyframe_min_interval_ms == 150


def test_wipe_dir_scans_a_library_and_explicit_entries_win(tmp_path):
    from pyplumber.mixer import config as mc
    for name in ("b_swoosh.mov", "a_dip.webm", "notes.txt", "c_star.mp4"):
        (tmp_path / name).write_bytes(b"x")
    doc = {**CONFIG, "wipe_dir": str(tmp_path),
           "wipes": [{"id": "a_dip", "path": str(tmp_path / "a_dip.webm"),
                      "duration_seconds": 1.2, "name": "Dip"}]}
    cfg = mc.parse(doc)

    assert [(w.id, w.label, w.duration_seconds) for w in cfg.wipes] == [
        ("a_dip", "Dip", 1.2), ("b_swoosh", "b_swoosh", 0.0), ("c_star", "c_star", 0.0)]
    assert cfg.settings()["default_wipe"] == "a_dip"
    assert [w["id"] for w in cfg.settings()["wipes"]] == ["a_dip", "b_swoosh", "c_star"]
    with pytest.raises(mc.ConfigError, match="not a directory"):
        mc.parse({**CONFIG, "wipe_dir": str(tmp_path / "nope")})


def test_wipe_cache_is_optional_and_splits_the_chain(tmp_path):
    from pyplumber.mixer import clipcache

    FakeMixer.instances.clear()
    default = build_application(GraphOptions(inputs=("a.mp4",), output="p.mp4"), api=fake_api())
    assert default.wipe_cache_mb == 0
    assert FakeMixer.instances[-1].parameters["cache_wipes_mb"] is None

    FakeMixer.instances.clear()
    cached = build_application(
        GraphOptions(inputs=("a.mp4",), output="p.mp4", wipe_cache_mb=256), api=fake_api())
    assert cached.wipe_cache_mb == 256
    assert FakeMixer.instances[-1].parameters["cache_wipes_mb"] == 256

    assert parse_args(["--input", "a.mp4", "--janus-output"]).wipe_cache_mb == 0
    assert parse_args(["--input", "a.mp4", "--janus-output", "--wipe-cache-mb", "256"]).wipe_cache_mb == 256
    assert clipcache.loader_group("mixer") == "mixer_wipe_load"


def test_clip_cache_node_parameters_name_the_clip_by_url():
    from pyplumber.mixer import clipcache

    node = clipcache.cache_node(name="mixer_wipe_cache", src="in", dst="out", group="g",
                                fps="60/1", budget_mb=256, url="/media/w.mov")
    assert node == {"name": "mixer_wipe_cache", "src": "in", "dst": "out", "group": "g",
                    "fps": "60/1", "cache": "clips", "url": "/media/w.mov", "budget_mb": 256}
    assert "budget_mb" not in clipcache.cache_node(name="n", src="a", dst="b", group="g", fps="60/1")


RENDITION_CONFIG = {**CONFIG, "canvas": {"width": 1080, "height": 1920, "fps": 30},
                    "renditions": [{"id": "program", "target": "janus", "width": 1080,
                                    "height": 1920, "fps": 30, "aspect": "9:16",
                                    "bitrate_kbps": 3000, "profile": "baseline", "preset": "p7"}]}


def test_renditions_encode_the_one_composited_program(tmp_path, monkeypatch):
    import json as _json
    from pyplumber.mixer import dmabuf_inputs
    from pyplumber.mixer import dmabuf_inputs
    (tmp_path / "page.sock").touch()
    monkeypatch.setattr(dmabuf_inputs, "rest_request", lambda *a, **k: {"windows": []})
    doc = {**RENDITION_CONFIG,
           "renditions": [RENDITION_CONFIG["renditions"][0],
                          {"id": "square", "target": "/rec/square.mp4", "width": 1080,
                           "height": 1080, "fps": 30, "bitrate_kbps": 2000}]}
    path = tmp_path / "m.json"
    path.write_text(_json.dumps(doc))
    FakeMixer.instances.clear()
    app = build_application(GraphOptions(config=str(path), janus_output=True,
                                         dmabuf_socket_dir=str(tmp_path)), api=fake_api())
    nodes = {n.parameters.get("name"): n.parameters for n in app.avp.nodes}

    assert nodes["split_renditions"]["dst"] == ["program_rendition_program", "program_rendition_square"]
    assert nodes["scale_program"]["graph"] == Color().setparams   # SDR canvas to SDR output: tags only
    assert nodes["scale_square"]["graph"].startswith("scale_cuda=w=1080:h=1080,")
    assert nodes["janus_encoder"]["options"]["preset"] == "p7"
    assert nodes["janus_encoder"]["options"]["profile"] == "baseline"
    assert nodes["janus_fps"]["fps"] == "30/1"
    assert FakeMixer.instances[-1].parameters["fps"] == (30, 1)


def test_a_rendition_may_not_ask_for_more_than_the_composer_renders():
    from pyplumber.mixer import config as mc
    with pytest.raises(mc.ConfigError, match="exceeds the canvas rate"):
        mc.parse({**RENDITION_CONFIG,
                  "renditions": [{**RENDITION_CONFIG["renditions"][0], "fps": 60}]})
    with pytest.raises(mc.ConfigError, match="is 9:16, not 16:9"):
        mc.parse({**RENDITION_CONFIG,
                  "renditions": [{**RENDITION_CONFIG["renditions"][0], "aspect": "16:9"}]})
    cfg = mc.parse(RENDITION_CONFIG)
    assert cfg.renditions[0].aspect == "9:16" and cfg.renditions[0].fps == 30


def test_hdr_and_sdr_janus_renditions_have_independent_feedback(tmp_path):
    doc = {**CONFIG, "sources": CONFIG["sources"][:1], "scenes": CONFIG["scenes"][:1],
           "initial_scene": "full", "wipes": [],
           "canvas": {"width": 1920, "height": 1080, "fps": 60, "working_format": "p010le",
                      "color": "hlg"},
           "renditions": [
               {"id": "hdr", "target": "janus", "port": 5006, "codec": "hevc_nvenc", "profile": "main10"},
               {"id": "sdr", "target": "janus", "port": 5004, "codec": "h264_nvenc",
                "profile": "baseline", "tonemap": "mobius", "tonemap_param": 0.9}]}
    path = tmp_path / "dual.json"
    path.write_text(json.dumps(doc))
    app = build_application(GraphOptions(config=str(path), janus_output=True), api=fake_api())
    nodes = {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    assert len(nodes) == len(app.avp.nodes)
    assert nodes["janus_encoder"]["options"]["profile"] == "main10"
    assert nodes["janus_sdr_encoder"]["options"]["profile"] == "baseline"
    assert nodes["janus_sdr_encoder"]["options"]["color_trc"] == "bt709"
    assert nodes["janus_sdr_format"]["real_pixel_format"] == "nv12"
    assert "tonemap_cuda=transfer_in=auto:transfer_out=sdr" in nodes["scale_sdr"]["graph"]
    assert ":tonemap=mobius:sdr_white=203:hdr_peak=1000:desat=0:param=0.9" in nodes["scale_sdr"]["graph"]
    assert ":5006?" in nodes["janus_rtp_output"]["url"]
    assert ":5004?" in nodes["janus_sdr_rtp_output"]["url"]
    feedback = app.rtcp_feedback_listener
    feedback.start()
    assert all(listener.started for listener in feedback.listeners)
    for listener in feedback.listeners:
        listener.parameters["on_keyframe_request"](None)
    assert app.avp.commands[-2:] == [
        "node.object.set janus_force_keyframe trigger true",
        "node.object.set janus_sdr_force_keyframe trigger true"]
    feedback.stop()
    assert not any(listener.started for listener in feedback.listeners)


def test_v210_sources_keep_422_through_a_p210_canvas(tmp_path):
    """NVDEC only yields 4:2:0; generated v210 content is the 4:2:2 path. It must
    reach the P210 canvas untouched and be subsampled once, at the encoder."""
    doc = {**CONFIG, "wipes": [], "initial_scene": "full",
           "sources": [{"id": "gen", "kind": "v210", "path": "/media/gen.v210", "width": 1920,
                        "height": 1080, "color": "hlg"}, CONFIG["sources"][0]],
           "scenes": [{"id": "full", "items": [
               {"source": "gen", "dst": {"x": 0, "y": 0, "w": 1920, "h": 1080}},
               {"source": "cam", "dst": {"x": 0, "y": 0, "w": 960, "h": 540}}]}],
           "canvas": {"width": 1920, "height": 1080, "fps": 60, "working_format": "p210le", "color": "hlg"},
           "renditions": [{"id": "hdr", "target": "janus"},
                          {"id": "sdr", "target": "/rec/sdr.mp4", "codec": "h264_nvenc", "tonemap": "hable"}]}
    path = tmp_path / "gen.json"
    path.write_text(json.dumps(doc))
    FakeMixer.instances.clear()
    app = build_application(GraphOptions(config=str(path), janus_output=True), api=fake_api())
    nodes = {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    sources = dict(FakeMixer.instances[-1].sources)
    assert nodes["unpack_0"]["sw_format"] == "p210le" and nodes["unpack_0"]["color_trc"] == "arib-std-b67"
    assert sources["gen"]["pixel_format"] == "p210le" and sources["gen"]["color"] == Color("hlg")
    assert sources["cam"]["pixel_format"] is None and sources["cam"]["color"] is None   # NVDEC, tags from the frames
    # HDR out: one chroma subsample to P010 for NVENC, no tone-map pass.
    assert nodes["scale_hdr"]["graph"] == Color("hlg").setparams + ",scale_cuda=format=p010le"
    assert nodes["janus_format"]["real_pixel_format"] == "p010le"
    assert nodes["scale_sdr"]["graph"].startswith(Color("hlg").setparams + ",tonemap_cuda=transfer_in=auto:transfer_out=sdr:format=nv12")


def test_cli_inputs_declare_browser_rgb_and_optional_file_color(tmp_path):
    (tmp_path / "page_00.sock").touch()
    FakeMixer.instances.clear()
    build_application(GraphOptions(inputs=("camera.mp4", "dmabuf://page_00"), output="program.ts",
                                   dmabuf_socket_dir=str(tmp_path), input_color="hlg"), api=fake_api())
    sources = dict(FakeMixer.instances[-1].sources)
    assert sources["source_0"]["color"] == "hlg" and not sources["source_0"]["packed_rgb"]
    assert sources["source_1"]["color"] == "sdr" and sources["source_1"]["packed_rgb"]
    FakeMixer.instances.clear()
    build_application(GraphOptions(inputs=("camera.mp4",), output="program.ts"), api=fake_api())
    assert dict(FakeMixer.instances[-1].sources)["source_0"]["color"] is None   # frame tags decide
    with pytest.raises(ValueError, match="input-color"):
        GraphOptions(inputs=("a.mp4",), output="o.ts", input_color="rec2020").validate()


@pytest.mark.parametrize("input_count", (32, 33))
@pytest.mark.parametrize("color", ("", "sdr", "hlg", "pq"))
def test_cli_input_color_survives_routing_threshold(input_count, color):
    app = build_application(GraphOptions(inputs=tuple(f"camera-{i}.mp4" for i in range(input_count)),
                                         output="program.ts", input_color=color), api=fake_api())
    sources = app.mixer.routed_sources if app.routed_inputs else app.mixer.sources
    assert app.routed_inputs == (input_count > 32)
    assert all(params["color"] == (color or None) for _, params in sources)


@pytest.mark.parametrize("wipe_color", ("", "sdr"))
def test_cli_wipe_color_reaches_builder(wipe_color):
    argv = ["--input", "camera.mp4", "--output", "program.ts", "--wipe-file", "wipe.mov"]
    if wipe_color:
        argv += ["--wipe-color", wipe_color]
    app = build_application(parse_args(argv), api=fake_api())
    assert app.mixer.parameters["wipe_color"] == (wipe_color or None)
    with pytest.raises(ValueError, match="HDR alpha"):
        GraphOptions(inputs=("camera.mp4",), output="program.ts", wipe_color="hlg").validate()


def test_cli_wipe_color_overrides_config(tmp_path):
    doc = {**CONFIG, "sources": CONFIG["sources"][:1], "scenes": CONFIG["scenes"][:1],
           "initial_scene": "full", "wipe_color": "hlg"}
    path = tmp_path / "mixer.json"
    path.write_text(json.dumps(doc))
    app = build_application(GraphOptions(config=str(path), output="program.ts", wipe_color="sdr"), api=fake_api())
    assert app.mixer.parameters["wipe_color"] == "sdr"


def test_pq_renditions_carry_hdr10_static_metadata(tmp_path):
    doc = {**CONFIG, "sources": CONFIG["sources"][:1], "scenes": CONFIG["scenes"][:1], "wipes": [],
           "initial_scene": "full",
           "canvas": {"width": 1920, "height": 1080, "fps": 60, "working_format": "p010le", "color": "hlg"},
           "renditions": [
               {"id": "pq", "target": "janus", "codec": "hevc_nvenc", "color": "pq", "max_fall": 300},
               {"id": "hlg", "target": "/rec/hlg.ts", "codec": "hevc_nvenc"},
               {"id": "pqfile", "target": "/rec/pq.ts", "codec": "hevc_nvenc", "color": "pq", "tonemap_peak": 40}]}
    path = tmp_path / "pq.json"
    path.write_text(json.dumps(doc))
    app = build_application(GraphOptions(config=str(path), janus_output=True), api=fake_api())
    nodes = {n.parameters["name"]: n.parameters for n in app.avp.nodes}
    md = nodes["janus_encoder"]["hdr_metadata"]
    assert md["primaries"][0] == [0.708, 0.292] and md["white_point"] == [0.3127, 0.3290]   # BT.2020 / D65
    assert (md["max_luminance"], md["min_luminance"], md["max_cll"], md["max_fall"]) == (1000, 0.0001, 1000, 300)
    assert nodes["janus_encoder"]["options"]["color_trc"] == "smpte2084"
    assert "hdr_metadata" not in nodes["hlg_encoder"]            # HLG signals nothing static
    assert nodes["pqfile_encoder"]["hdr_metadata"]["max_luminance"] == 4000
    assert nodes["pqfile_encoder"]["hdr_metadata"]["max_fall"] == 1600
    with pytest.raises(mixer_config.ConfigError, match="max_fall"):
        mixer_config.parse({**doc, "renditions": [{"id": "x", "codec": "hevc_nvenc", "color": "pq", "max_fall": 5000}]})
    with pytest.raises(mixer_config.ConfigError, match="wipe_color"):
        mixer_config.parse({**CONFIG, "wipe_color": "rec2020"})


def test_hdr_example_config_parses_and_builds(tmp_path, monkeypatch):
    """The shipped HDR example must stay valid: P210 HLG canvas, v210 + video + browser sources,
    HLG Janus, mobius SDR Janus and a PQ archive with HDR10 metadata."""
    path = Path(__file__).resolve().parents[1] / "config.example.hdr.json"
    cfg = mixer_config.parse(json.loads(path.read_text()))
    assert cfg.working_format == "p210le" and cfg.out_color == Color("hlg")
    assert [r.id for r in cfg.renditions] == ["hdr", "sdr", "archive"]
    (tmp_path / "page.sock").touch()
    from pyplumber.mixer import dmabuf_inputs
    monkeypatch.setattr(dmabuf_inputs, "rest_request", lambda *a, **k: {"windows": []})
    FakeMixer.instances.clear()
    app = build_application(GraphOptions(config=str(path), janus_output=True, dmabuf_socket_dir=str(tmp_path)),
                            api=fake_api())
    nodes = {n.parameters.get("name"): n.parameters for n in app.avp.nodes}
    assert nodes["unpack_0"]["sw_format"] == "p210le"
    assert nodes["scale_hdr"]["graph"] == Color("hlg").setparams + ",scale_cuda=format=p010le"
    assert ":tonemap=mobius:" in nodes["scale_sdr"]["graph"] and ":param=0.9" in nodes["scale_sdr"]["graph"]
    assert nodes["archive_encoder"]["hdr_metadata"]["max_cll"] == 1000
    assert nodes["archive_encoder"]["options"]["color_trc"] == "smpte2084"


def test_generated_hdr_show_has_hdr_and_sdr_renditions():
    import make_config
    import io
    import contextlib
    args = ["--canvas", "1920x1080", "--fps", "60", "--color", "hlg", "--working-format", "p210le",
            "--program-port", "5006", "--sdr-port", "5004", "movie=/m/hdr.mp4", "clip=/m/hlg.mp4:hlg", "pat=/f/p.v210@1920x1080:hlg",
            "bunny=/m/bunny.mp4:sdr", "page=https://example.org/a@1920x1080"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        make_config.main(args)
    doc = json.loads(out.getvalue())
    cfg = mixer_config.parse(doc)
    assert cfg.working_format == "p210le" and cfg.out_color == Color("hlg")
    assert doc["canvas"]["color"] == "hlg"
    program, sdr = doc["renditions"]
    assert (program["codec"], program["profile"]) == ("hevc_nvenc", "main10") and "tonemap" not in program
    assert program["port"] == 5006
    assert (sdr["codec"], sdr["port"], sdr["tonemap"], sdr["tonemap_param"]) == ("h264_nvenc", 5004, "mobius", 0.9)
    kinds = {s["id"]: (s["kind"], s.get("color")) for s in doc["sources"]}
    assert kinds == {"movie": ("video", None), "clip": ("video", "hlg"), "pat": ("v210", "hlg"),
                     "bunny": ("video", "sdr"), "page": ("browser", "sdr")}
    assert next(s for s in doc["sources"] if s["id"] == "pat")["width"] == 1920
    with pytest.raises(SystemExit, match="v210 clips need"):
        make_config.source_spec("raw=/f/p.v210@1920x1080")
    with pytest.raises(SystemExit, match="--sdr-port needs"):
        make_config.main(["--sdr-port", "5004", "clip=/m/c.mp4"])
    with pytest.raises(SystemExit, match="distinct RTP ports"):
        make_config.generate(["--color", "hlg", "--working-format", "p210le",
                              "--sdr-port", "5004", "clip=/m/c.mp4"])


def test_generated_ten_bit_sdr_show_uses_hevc_profile():
    from make_config import generate
    doc = generate(["--working-format", "p210le", "clip=/media/clip.mp4:sdr"])
    rendition = doc["renditions"][0]
    assert (rendition["codec"], rendition["profile"]) == ("hevc_nvenc", "main10")
    assert mixer_config.parse(doc).out_color == Color("sdr")


def test_canvas_latency_reaches_the_builder_unless_the_cli_overrides_it(tmp_path, monkeypatch):
    from pyplumber.mixer import dmabuf_inputs
    monkeypatch.setattr(dmabuf_inputs, "rest_request", lambda *a, **k: {"windows": []})
    (tmp_path / "page.sock").touch()
    doc = {**CONFIG, "canvas": {**CONFIG["canvas"], "latency_ms": 50}}
    assert mixer_config.parse(doc).latency_ms == 50.0
    assert mixer_config.parse(CONFIG).latency_ms is None
    with pytest.raises(mixer_config.ConfigError, match="six frames"):
        mixer_config.parse({**CONFIG, "canvas": {**CONFIG["canvas"], "fps": 60, "latency_ms": 100}})
    with pytest.raises(mixer_config.ConfigError, match="latency_ms"):
        mixer_config.parse({**CONFIG, "canvas": {**CONFIG["canvas"], "latency_ms": "soon"}})
    path = tmp_path / "show.json"
    path.write_text(json.dumps(doc))
    FakeMixer.instances.clear()
    build_application(GraphOptions(config=str(path), janus_output=True, dmabuf_socket_dir=str(tmp_path)),
                      api=fake_api())
    assert FakeMixer.instances[-1].parameters["latency_ms"] == 50.0
    FakeMixer.instances.clear()
    build_application(GraphOptions(config=str(path), janus_output=True, dmabuf_socket_dir=str(tmp_path),
                                   mixer_latency_ms=20.0), api=fake_api())
    assert FakeMixer.instances[-1].parameters["latency_ms"] == 20.0


@pytest.mark.parametrize("count", [33, 45, 64, 65, 96, 128, 129])
def test_source_mask_capacity(count):
    sources = [{"id": f"s{i}", "kind": "video", "path": f"/m/{i}.mp4", "width": 16, "height": 16} for i in range(count)]
    doc = {**CONFIG, "initial_scene": "s", "sources": sources, "scenes": [{"id": "s", "items": [{"source": "s0", "dst": {"x": 0, "y": 0, "w": 16, "h": 16}}]}]}
    if count > mixer_config.MAX_SOURCES:
        with pytest.raises(mixer_config.ConfigError, match="at most 128 sources"):
            mixer_config.parse(doc)
    else:
        assert len(mixer_config.parse(doc).sources) == count


def test_mobius_knee_must_leave_shoulder_room():
    doc = {**CONFIG, "renditions": [{"id": "sdr", "codec": "h264_nvenc", "tonemap": "mobius", "tonemap_param": 1.0}]}
    with pytest.raises(mixer_config.ConfigError, match="below 1.0"):
        mixer_config.parse(doc)


@pytest.mark.parametrize("fmt", ["yuv420p", "yuv444p10le", "yuv422p10le"])
def test_planar_working_formats_are_rejected_up_front(fmt):
    # The compositor cannot promote 8-bit sources or draw the RGBA wipe onto planar canvases.
    with pytest.raises(ValueError, match="working-format"):
        GraphOptions(inputs=("a.mp4",), output="o.ts", working_format=fmt).validate()
    with pytest.raises(mixer_config.ConfigError, match="working_format"):
        mixer_config.parse({**CONFIG, "canvas": {**CONFIG["canvas"], "working_format": fmt}})


def test_fractional_janus_rate_uses_one_second_gop():
    from pyplumber.mixer.janus import JanusVideoConfig, build_janus_output
    avp = FakeAvp()
    build_janus_output(avp, fake_api(), "program", JanusVideoConfig(), fps=60000, fps_den=1001,
                       width=1920, height=1080)
    nodes = {n.parameters["name"]: n.parameters for n in avp.nodes}
    assert nodes["janus_fps"]["fps"] == "60000/1001"
    assert nodes["janus_encoder"]["options"]["g"] == 60


def test_control_section_carries_the_defaults_the_surfaces_start_from():
    from pyplumber.mixer import config as mc
    import copy
    doc = copy.deepcopy(CONFIG)
    del doc["control"]
    del doc["canvas"]["fps"]
    cfg = mc.parse(doc)
    # An undeclared canvas rate is 30, and a pick cuts directly to program.
    assert (cfg.fps, cfg.direct, cfg.fade_seconds, cfg.transition) == (30, True, 0.5, "cut")
    assert cfg.settings()["transition"] == "cut"
    chosen = mc.parse({**doc, "control": {"transition": "wipe", "fade_seconds": 1.5, "direct": False}})
    assert (chosen.transition, chosen.fade_seconds, chosen.direct) == ("wipe", 1.5, False)
    with pytest.raises(mc.ConfigError, match="control.transition must be"):
        mc.parse({**doc, "control": {"transition": "dissolve"}})
    with pytest.raises(mc.ConfigError, match="needs a wipe library"):
        mc.parse({**{k: v for k, v in doc.items() if k != "wipes"}, "control": {"transition": "wipe"}})
    with pytest.raises(mc.ConfigError, match="fade_seconds must be positive"):
        mc.parse({**doc, "control": {"fade_seconds": 0}})


def test_generated_grids_show_every_distinct_source_before_any_repeat():
    import make_config
    import io
    import contextlib
    args = [f"clip{i}=/media/clip{i}.mp4" for i in range(4)] + ["dup=/media/clip0.mp4"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        make_config.main(args)
    doc = json.loads(out.getvalue())
    assert [s["id"] for s in doc["sources"]] == ["clip0", "clip1", "clip2", "clip3"]
    four_box = next(s for s in doc["scenes"] if s["id"] == "grid_4_page_0")
    assert [i["source"] for i in four_box["items"]] == ["clip0", "clip1", "clip2", "clip3"]
    assert doc["canvas"]["fps"] == 30 and doc["control"]["transition"] == "cut"
    assert doc["renditions"][0]["bitrate_kbps"] == 2700 and doc["renditions"][0]["aspect"] == "9:16"
