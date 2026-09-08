from types import SimpleNamespace

import pytest

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
        self.control_port = None
        self.ready = False
        self.edges = FakeEdges()

    def addNode(self, node):
        self.nodes.append(node)

    def executeCommandsFromString(self, commands):
        self.commands.append(commands)

    def enableControlServer(self, port):
        self.control_port = port

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
        "split",
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
            "split": "Split",
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

    assert nodes["split_program_video_output"]["dst"] == [
        "program_video_record",
        "program_video_janus",
    ]


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
    stamp = nodes["input_1_timestamp"]
    assert (stamp["dst_width"], stamp["dst_height"], stamp["dst_frame_rate"]) == (480, 270, "60/1")
    assert stamp["dst"] == "input_1_cuda"
    assert "decode_1" not in nodes and "decode_0" in nodes
    sources = dict(FakeMixer.instances[-1].sources)
    assert sources["source_1"]["pre_otm_edge"] == "input_1_cuda"
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
    from avpmixer import dmabuf_inputs

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


def test_wipe_file_warms_the_wipe_chain_up_at_start(monkeypatch):
    FakeMixer.instances.clear()
    application = build_application(
        GraphOptions(inputs=("a.mp4",), output="p.mp4", wipe_file="/media/wipe.mov"), api=fake_api())
    monkeypatch.setattr(application, "_wait_for_edges", lambda *a, **k: None)
    monkeypatch.setattr(application, "_wait_for_node", lambda *a, **k: None)
    application.start()
    assert FakeMixer.instances[-1].warmed_wipe == ("/media/wipe.mov", 30000)
    assert application.avp.ready

    FakeMixer.instances.clear()
    plain = build_application(GraphOptions(inputs=("a.mp4",), output="p.mp4"), api=fake_api())
    monkeypatch.setattr(plain, "_wait_for_edges", lambda *a, **k: None)
    monkeypatch.setattr(plain, "_wait_for_node", lambda *a, **k: None)
    plain.start()
    assert not hasattr(FakeMixer.instances[-1], "warmed_wipe")


CONFIG = {
    "canvas": {"width": 1920, "height": 1080, "fps": 60},
    "sources": [
        {"id": "cam", "kind": "video", "path": "/media/cam.mp4", "width": 1920, "height": 1080},
        {"id": "page", "kind": "browser", "url": "https://example.org/", "width": 1280, "height": 720},
    ],
    "wipes": [{"id": "swoosh", "path": "/media/swoosh.mov"}],
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
    from avpmixer import config as mc
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
    from avpmixer import config as mc
    import copy
    dup = copy.deepcopy(CONFIG)
    dup["sources"].append({"id": "cam2", "kind": "video", "path": "/media/cam.mp4"})
    with pytest.raises(mc.ConfigError, match="already declared"):
        mc.parse(dup)
    bad = copy.deepcopy(CONFIG)
    bad["scenes"][0]["items"][0]["source"] = "nope"
    with pytest.raises(mc.ConfigError, match="unknown source"):
        mc.parse(bad)
    nocover = copy.deepcopy(CONFIG)
    del nocover["sources"][0]["width"]
    with pytest.raises(mc.ConfigError, match="cover needs"):
        mc.parse(nocover)


def test_config_builds_one_chain_per_source_with_alias_fanout(tmp_path, monkeypatch):
    import json as _json
    from avpmixer import dmabuf_inputs
    (tmp_path / "page.sock").touch()
    path = tmp_path / "mixer.json"
    path.write_text(_json.dumps(CONFIG))
    opened = []
    monkeypatch.setattr(dmabuf_inputs, "rest_request",
                        lambda base, method, p, body=None: opened.append((method, p, body)) or {"windows": []})
    FakeMixer.instances.clear()
    application = build_application(
        GraphOptions(config=str(path), output="p.mp4", dmabuf_socket_dir=str(tmp_path)), api=fake_api())
    nodes = {node.parameters.get("name"): node.parameters for node in application.avp.nodes}
    mixer = FakeMixer.instances[-1]

    assert [name for name in nodes if name and name.startswith("decode_")] == ["decode_0"]
    assert nodes["alias_0"]["dst"] == ["input_0_fps_alias1", "input_0_fps_alias2"]
    assert nodes["alias_0"]["outputs"] == 3
    assert [name for name, _ in mixer.sources] == ["cam", "cam#2", "page"]
    assert dict(mixer.sources)["page"]["pre_otm_edge"] == "input_1_cuda"
    assert opened[-1][2] == {"id": "page", "url": "https://example.org/", "width": 1280, "height": 720,
                             "fps": 60, "audio": False}
    assert mixer.initial_scene == ("pip", "A") and set(mixer.scenes) == {"full", "pip"}
    assert mixer.parameters["canvas"] == (1920, 1080)
    assert application.wipe_files == ("/media/swoosh.mov",)
    assert nodes["program_format"]["width"] == 1920
    assert nodes["program_fps"]["fps"] == "60/1"      # outputs follow the document's fps, not the CLI default


def test_cli_requires_inputs_or_config():
    with pytest.raises(SystemExit):
        parse_args(["--output", "p.mp4"])
    assert parse_args(["--config", "m.json", "--janus-output"]).config == "m.json"
