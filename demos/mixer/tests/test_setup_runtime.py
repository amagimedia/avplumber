"""Setup changes validate before touching the live process and recover on failure."""
import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import prepare_demo
from setup_runtime import DEFAULT_SETTINGS, SetupRuntime, recipe_for, source_counts
from webui import serve


@pytest.mark.parametrize('count', [1, 8, 16, 32, 42, 48, 64, 96])
@pytest.mark.parametrize('fps', [25, 30, 50, 60])
def test_generic_setups_expand(tmp_path, count, fps):
    if fps >= 50 and count > 48:
        with pytest.raises(ValueError, match="source_count must be an integer from 1 to 48"):
            recipe_for({**DEFAULT_SETTINGS, "source_count": count, "fps": fps})
        return
    recipe = recipe_for({**DEFAULT_SETTINGS, 'source_count': count, 'fps': fps})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    assert len(show['sources']) == count
    assert len(show['scenes']) == DEFAULT_SETTINGS['scene_count']
    assert show['canvas']['fps'] == fps


@pytest.mark.parametrize('changes', [{'source_count': 97}, {'scene_count': 0}, {'scene_count': 129}, {'fps': 24},
    {'weights': [0] * 5}, {'weights': [True] * 5}, {'resolution': '../../file'}, {'command': 'id'}])
def test_reject_unbounded_settings(changes):
    with pytest.raises(ValueError):
        recipe_for({**DEFAULT_SETTINGS, **changes})


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777))
    manager.process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(manager, '_stop', lambda: None)
    monkeypatch.setattr(manager, '_close_removed_browsers', lambda *_: None)
    monkeypatch.setattr(manager, '_start', lambda _: None)
    return manager


def test_prepare_failure_keeps_old_show_and_process(runtime, monkeypatch):
    config = runtime.media_dir / 'mixer.demo.json'
    config.write_text('{"old": true}')
    def fail(*args):
        raise ValueError('disk full')
    monkeypatch.setattr(prepare_demo, 'prepare', fail)
    runtime.apply(DEFAULT_SETTINGS)
    runtime.worker.join(3)
    assert runtime.status()['phase'] == 'error'
    assert 'disk full' in runtime.status()['message']
    assert json.loads(config.read_text()) == {'old': True}
    assert runtime.process.poll() is None


def test_failed_start_restores_previous_show(runtime, monkeypatch):
    config = runtime.media_dir / 'mixer.demo.json'
    config.write_text('{"old": true}')
    monkeypatch.setattr(prepare_demo, 'prepare', lambda *_: config.write_text('{"new": true}'))
    starts = []
    def start(path):
        doc = json.loads(path.read_text())
        starts.append(doc)
        if 'new' in doc:
            raise RuntimeError('GPU out of memory')
    monkeypatch.setattr(runtime, '_start', start)
    runtime.apply(DEFAULT_SETTINGS)
    runtime.worker.join(3)
    assert starts == [{'new': True}, {'old': True}]
    assert 'previous setup restored' in runtime.status()['message']
    assert not runtime.recipe_path.exists()


def test_single_job_and_persist_only_after_ready(runtime, monkeypatch):
    revision = runtime.revision
    entered, release = threading.Event(), threading.Event()
    def prepare(*_):
        entered.set()
        assert release.wait(3)
        (runtime.media_dir / 'mixer.demo.json').write_text('{}')
    monkeypatch.setattr(prepare_demo, 'prepare', prepare)
    runtime.apply(DEFAULT_SETTINGS)
    assert entered.wait(3)
    try:
        assert not runtime.recipe_path.exists()
        with pytest.raises(RuntimeError, match='already in progress'):
            runtime.apply(DEFAULT_SETTINGS)
    finally:
        release.set()
        runtime.worker.join(3)
    assert runtime.status()['phase'] == 'running'
    assert runtime.status()['settings'] == DEFAULT_SETTINGS
    assert runtime.status()['revision'] == revision + 1
    assert json.loads(runtime.recipe_path.read_text())['setup'] == DEFAULT_SETTINGS


def test_api_rejects_cross_origin_and_oversized_requests(runtime):
    server = serve(SimpleNamespace(), '127.0.0.1', 0, setup=runtime)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}/api/setup'
    try:
        for headers, data, code in [({'Origin': 'http://elsewhere.invalid'}, b'{}', 403),
                                    ({}, b' ' * 4097, 400), ({}, b'{}', 400)]:
            request = Request(url, data=data, headers={'Content-Type': 'application/json', **headers})
            with pytest.raises(HTTPError) as error:
                urlopen(request, timeout=3)
            assert error.value.code == code
        assert runtime.worker is None
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("bit_depth", [8, 10])
def test_bit_depth_controls_canvas_sources_and_encoders(tmp_path, bit_depth):
    settings = {**DEFAULT_SETTINGS, "bit_depth": bit_depth, "chroma": "420" if bit_depth == 8 else "422",
                "weights": [4, 0, 0, 0, 1]}
    show, jobs, _ = prepare_demo.plan(recipe_for(settings), tmp_path)
    from pyplumber.mixer.config import parse
    cfg = parse(show)
    if bit_depth == 8:
        assert show["canvas"]["working_format"] == "nv12"
        assert show["canvas"]["color"] == "sdr"
        assert [r["codec"] for r in show["renditions"]] == ["h264_nvenc"]
        assert cfg.settings()["preview_codecs"] == ["h264"]
        assert not any("hlg" in str(path) or path.suffix == ".v210" for path in jobs)
    else:
        assert show["canvas"]["working_format"] == "p210le"
        assert cfg.settings()["preview_codecs"] == ["h264", "h265"]


def test_8bit_rejects_ten_bit_sources():
    with pytest.raises(ValueError, match="SDR 4:2:0 and browser"):
        recipe_for({**DEFAULT_SETTINGS, "bit_depth": 8})


def test_old_setup_defaults_to_ten_bit():
    old = {k: v for k, v in DEFAULT_SETTINGS.items() if k not in ("bit_depth", "chroma")}
    assert recipe_for(old)["canvas"]["working_format"] == "p210le"
    old.update(bit_depth=8, weights=[4, 0, 0, 0, 1])
    assert recipe_for(old)["setup"]["chroma"] == "420"


def test_bitrate_is_configurable_and_scales_every_rendition():
    """One control sets the SDR bitrate; other renditions keep their ratio to it."""
    base = recipe_for(DEFAULT_SETTINGS)["renditions"]
    assert [r["bitrate_kbps"] for r in base] == [6000, 8000]

    halved = recipe_for({**DEFAULT_SETTINGS, "bitrate_kbps": 3000})["renditions"]
    assert [r["bitrate_kbps"] for r in halved] == [3000, 4000]

    # The setting is optional: an older stored setup still expands, at the recipe's own numbers.
    legacy = {k: v for k, v in DEFAULT_SETTINGS.items() if k != "bitrate_kbps"}
    assert recipe_for(legacy)["renditions"][0]["bitrate_kbps"] == 6000
    assert recipe_for(legacy)["setup"]["bitrate_kbps"] == 6000


@pytest.mark.parametrize("value", [499, 40001, 0, -1, 6000.0, "6000", True])
def test_bitrate_outside_the_range_is_rejected(value):
    with pytest.raises(ValueError, match="bitrate_kbps"):
        recipe_for({**DEFAULT_SETTINGS, "bitrate_kbps": value})


@pytest.mark.parametrize("bit_depth", [8, 10])
@pytest.mark.parametrize("fps, maximum", [(25, 96), (30, 96), (50, 48), (60, 48)])
def test_setup_limits_in_both_modes(bit_depth, fps, maximum):
    settings = {**DEFAULT_SETTINGS, "bit_depth": bit_depth, "fps": fps,
                "chroma": "420" if bit_depth == 8 else "422",
                "source_count": maximum, "scene_count": 128, "weights": [1, 0, 0, 0, 0]}
    assert recipe_for(settings)["source_count"] == maximum
    with pytest.raises(ValueError, match="source_count"):
        recipe_for({**settings, "source_count": maximum + 1})
    with pytest.raises(ValueError, match="scene_count"):
        recipe_for({**settings, "scene_count": 129})


def test_hdr_420_canvas_and_assets(tmp_path):
    recipe = recipe_for({**DEFAULT_SETTINGS, "chroma": "420", "weights": [8, 6, 0, 0, 2]})
    show, jobs, _ = prepare_demo.plan(recipe, tmp_path)
    assert show["canvas"]["working_format"] == "p010le"
    assert show["canvas"]["color"] == "hlg"
    assert len(show["renditions"]) == 2
    assert any(s["color"] == "hlg" for s in show["sources"] if s["kind"] == "video")
    assert not any(p.suffix == ".v210" for p in jobs)


@pytest.mark.parametrize("changes, message", [
    ({"chroma": "444"}, "Unsupported chroma"),
    ({"chroma": "420"}, "4:2:0 mode"),
    ({"bit_depth": 8, "weights": [1, 0, 0, 0, 0]}, "4:2:0 canvas"),
    ({"weights": [0, 0, 1, 0, 0]}, "enable another source type"),
])
def test_reject_incompatible_chroma(changes, message):
    with pytest.raises(ValueError, match=message):
        recipe_for({**DEFAULT_SETTINGS, **changes})


@pytest.mark.parametrize("total", [8, 16, 24, 32, 42, 64])
@pytest.mark.parametrize("weights", [[8, 4, 2, 0, 2], [1, 1, 100, 0, 1]])
def test_hdr_422_cap_preserves_total_and_disabled_types(tmp_path, total, weights):
    recipe = recipe_for({**DEFAULT_SETTINGS, "fps": 25, "source_count": total, "weights": weights})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    counts = source_counts(total, weights)
    assert len(show["sources"]) == sum(counts) == total
    assert 0 < counts[2] <= 4
    assert counts[3] == 0
    assert sum(s["kind"] == "v210" for s in show["sources"]) == counts[2]
    assert recipe["setup"]["weights"] == weights


def test_four_hdr_422_inputs_can_be_used_alone():
    assert source_counts(4, [0, 0, 1, 0, 0]) == [0, 0, 4, 0, 0]


@pytest.mark.parametrize('weights, expected', [
    ([1, 0, 0, 0, 100], [32, 0, 0, 0, 32]),
    ([1, 0, 100, 0, 100], [28, 0, 4, 0, 32]),
    ([1, 0, 1, 0, 100], [28, 0, 4, 0, 32]),
])
def test_browser_cap_redistributes_without_exceeding_other_caps(weights, expected):
    assert source_counts(64, weights) == expected
    recipe = recipe_for({**DEFAULT_SETTINGS, 'source_count': 64, 'fps': 25, 'weights': weights})
    assert [source['weight'] for source in recipe['inputs']] == expected


def test_browser_only_limit():
    settings = {**DEFAULT_SETTINGS, 'fps': 30, 'source_count': 32, 'weights': [0, 0, 0, 0, 1]}
    assert recipe_for(settings)['inputs'][-1]['weight'] == 32
    with pytest.raises(ValueError, match='Browser is limited to 32'):
        recipe_for({**settings, 'source_count': 33})
    with pytest.raises(ValueError, match='enable another source type'):
        source_counts(37, [0, 0, 1, 0, 1])


def test_failed_first_start_does_not_leave_show_for_resume(runtime, monkeypatch):
    config = runtime.media_dir / 'mixer.demo.json'
    monkeypatch.setattr(prepare_demo, 'prepare', lambda *_: config.write_text('{"sources": []}'))
    def fail(*_):
        raise RuntimeError('encoder unavailable')
    monkeypatch.setattr(runtime, '_start', fail)
    runtime.apply(DEFAULT_SETTINGS)
    runtime.worker.join(3)
    assert 'encoder unavailable' in runtime.status()['message']
    assert not config.exists()
    assert not runtime.recipe_path.exists()


@pytest.mark.parametrize('previous', [None, b'{"old": true}'])
def test_shutdown_during_prepare_restores_committed_show(runtime, monkeypatch, previous):
    config = runtime.media_dir / 'mixer.demo.json'
    if previous is not None:
        config.write_bytes(previous)
    def prepare(*_):
        config.write_text('{"new": true}')
        runtime.closing.set()
    monkeypatch.setattr(prepare_demo, 'prepare', prepare)
    monkeypatch.setattr(runtime, '_start', lambda _: pytest.fail('must not start while closing'))
    runtime.apply(DEFAULT_SETTINGS)
    runtime.worker.join(3)
    assert (config.read_bytes() if config.exists() else None) == previous
    assert not runtime.recipe_path.exists()


def test_restart_changes_revision(tmp_path, monkeypatch):
    monkeypatch.setattr('setup_runtime.time.time_ns', lambda: 1_000_000_000_000)
    first = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777))
    monkeypatch.setattr('setup_runtime.time.time_ns', lambda: 2_000_000_000_000)
    restarted = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777))
    assert first.status()['revision'] != restarted.status()['revision']
    assert restarted.status()['revision'] < 2 ** 53


@pytest.mark.parametrize("before,after", [(8, 10), (10, 8)])
def test_setup_preserves_live_aux_through_color_change_and_resume(runtime, monkeypatch, before, after):
    def settings(depth):
        return {**DEFAULT_SETTINGS, "bit_depth": depth, "chroma": "420",
                "weights": [4, 0, 0, 0, 1]}
    config = runtime.media_dir / "mixer.demo.json"
    show, _, _ = prepare_demo.plan(recipe_for(settings(before)), runtime.media_dir)
    scene_ids = [s["id"] for s in show["scenes"]]
    show["aux_buses"] = [{"id": "mv", "scenes": [scene_ids[0]] * 8,
                          "renditions": [{"id": "monitor", "port": 5008}]}]
    config.write_text(json.dumps(show))
    live = [scene_ids[1], scene_ids[-1], *([None] * 6)]
    runtime.bridge.command = lambda cmd: json.dumps([{"id": "mv", "scenes": live}])
    def prepare(recipe, directory):
        generated, _, _ = prepare_demo.plan(recipe, directory)
        config.write_text(json.dumps(generated))
    monkeypatch.setattr(prepare_demo, "prepare", prepare)
    runtime.apply(settings(after))
    runtime.worker.join(3)
    assert runtime.status()["phase"] == "running", runtime.status()
    changed = json.loads(config.read_text())
    assert changed["canvas"]["color"] == ("sdr" if after == 8 else "hlg")
    assert changed["aux_buses"][0]["scenes"] == live
    from pyplumber.mixer.config import parse
    rendition = parse(changed).aux_buses[0].renditions[0]
    assert (rendition.codec, rendition.color, rendition.port) == ("h264_nvenc", "sdr", 5008)
    assert json.loads(runtime.recipe_path.read_text())["aux_buses"] == changed["aux_buses"]
    runtime.apply()  # same path as resuming the saved recipe after server restart
    runtime.worker.join(3)
    assert runtime.status()["phase"] == "running", runtime.status()
    assert json.loads(config.read_text())["aux_buses"] == changed["aux_buses"]


def test_setup_reconciles_aux_geometry_rate_and_removed_scenes(runtime):
    from pyplumber.mixer.config import parse
    old, _, _ = prepare_demo.plan(recipe_for(DEFAULT_SETTINGS), runtime.media_dir)
    old["aux_buses"] = [{"id": "mv", "scenes": [old["scenes"][0]["id"], "removed", *([None] * 6)],
                         "renditions": [{"id": "monitor", "port": 5008, "width": 1080,
                                         "height": 1920, "fps": 30, "bitrate_kbps": 4500}]}]
    (runtime.media_dir / "mixer.demo.json").write_text(json.dumps(old))
    runtime.process = None
    recipe = recipe_for({**DEFAULT_SETTINGS, "resolution": "1280x720", "orientation": "landscape",
                         "fps": 50, "scene_count": 1, "layout": "fullscreen"})
    show, _, _ = prepare_demo.plan(recipe, runtime.media_dir)
    runtime._preserve_aux(recipe, show)
    cfg = parse(prepare_demo.plan(recipe, runtime.media_dir)[0])
    assert cfg.aux_buses[0].scenes == (old["scenes"][0]["id"], *([None] * 7))
    r = cfg.aux_buses[0].renditions[0]
    assert (r.width, r.height, r.fps, r.bitrate_kbps) == (1280, 720, 25, 4500)


def test_setup_clears_aux_tiles_that_exceed_new_draw_budget(runtime):
    recipe = recipe_for({**DEFAULT_SETTINGS, "fps": 30, "source_count": 64,
                         "weights": [1, 0, 0, 0, 0], "layout": "grids"})
    show, _, _ = prepare_demo.plan(recipe, runtime.media_dir)
    grid = next(s["id"] for s in show["scenes"] if s["id"].startswith("grid_64_"))
    show["aux_buses"] = [{"id": "mv", "scenes": [grid] * 8,
                          "renditions": [{"id": "monitor", "port": 5008}]}]
    (runtime.media_dir / "mixer.demo.json").write_text(json.dumps(show))
    runtime.process = None
    del show["aux_buses"]
    runtime._preserve_aux(recipe, show)
    result, _, _ = prepare_demo.plan(recipe, runtime.media_dir)
    assert result["aux_buses"][0]["scenes"] == [grid, grid, *([None] * 6)]


def test_invalid_preserved_aux_rejected_before_preparation(runtime, monkeypatch):
    config = runtime.media_dir / "mixer.demo.json"
    config.write_text(json.dumps({"aux_buses": [{"id": "mv", "scenes": [None] * 8,
                                              "renditions": [{"id": "monitor", "port": 5004}]}]}))
    runtime.process = None
    monkeypatch.setattr(prepare_demo, "prepare", lambda *_: pytest.fail("must validate first"))
    with pytest.raises(ValueError, match="port"):
        runtime.apply(DEFAULT_SETTINGS)
    assert runtime.worker is None


@pytest.mark.parametrize("hdr", [False, True])
def test_start_waits_for_program_and_aux_encoders(runtime, monkeypatch, hdr):
    monkeypatch.setattr("setup_runtime.subprocess.Popen", lambda *a, **kw: SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(runtime.closing, "wait", lambda _: False)
    runtime.bridge.state = lambda **kw: {"status": {"pgm_scene": "full"},
        "settings": {"preview_codecs": ["h264", "h265"] if hdr else ["h264"], "aux_buses": ["mv"]}}
    expected = ["janus_encoded", "aux_mv_encoded", *(["janus_hdr_encoded"] if hdr else [])]
    calls = []
    def queues(command, **kw):
        calls.append(command)
        assert len(calls) <= len(expected)
        return json.dumps([{"name": name, "enqueued_total": int(i < len(calls))}
                           for i, name in enumerate(expected)])
    runtime.bridge.command = queues
    SetupRuntime._start(runtime, runtime.media_dir / "mixer.demo.json")
    assert len(calls) == len(expected)
