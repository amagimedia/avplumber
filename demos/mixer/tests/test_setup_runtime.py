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


@pytest.mark.parametrize('count', [1, 8, 16, 32, 41, 42, 48, 50, 64, 83, 96, 100, 110])
@pytest.mark.parametrize('fps', [25, 30, 50, 60])
def test_generic_setups_expand(tmp_path, count, fps):
    maximum = {25: 110, 30: 83, 50: 50, 60: 41}[fps]
    if count > maximum:
        with pytest.raises(ValueError, match=f"source_count must be an integer from 1 to {maximum}"):
            recipe_for({**DEFAULT_SETTINGS, "source_count": count, "fps": fps})
        return
    if count > (40 if fps <= 30 else 20) + 32 + 4:
        with pytest.raises(ValueError, match="enable another source type"):
            recipe_for({**DEFAULT_SETTINGS, "source_count": count, "fps": fps})
        return
    recipe = recipe_for({**DEFAULT_SETTINGS, 'source_count': count, 'fps': fps})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    assert len(show['sources']) == count
    assert len(show['scenes']) == DEFAULT_SETTINGS['scene_count']
    assert show['canvas']['fps'] == fps


@pytest.mark.parametrize('changes', [{'source_count': 97}, {'scene_count': 0}, {'scene_count': 193}, {'fps': 24},
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
    monkeypatch.setattr(manager, '_recover_browsers', lambda *_: False)
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


@pytest.mark.parametrize("fps, size", [(25, 6), (30, 6), (50, 9), (60, 9)])
def test_browser_ring_default_tracks_fps(tmp_path, fps, size):
    settings = {k: v for k, v in DEFAULT_SETTINGS.items() if k != "browser_ring_size"}
    recipe = recipe_for({**settings, "fps": fps})
    assert recipe["browser_ring_size"] == size
    del recipe["browser_ring_size"]
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    from pyplumber.mixer.config import parse
    from mixer import GraphOptions
    assert show["browser_ring_size"] == size
    del show["browser_ring_size"]
    assert parse(show).browser_ring_size == size
    assert GraphOptions(fps=fps).browser_ring_size == size


@pytest.mark.parametrize("size", [6, 9, 11])
def test_browser_ring_limit_reaches_show(tmp_path, size):
    recipe = recipe_for({**DEFAULT_SETTINGS, "browser_ring_size": size})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    from pyplumber.mixer.config import parse
    assert parse(show).browser_ring_size == size
    assert parse(show).settings()["browser_ring_size"] == size


@pytest.mark.parametrize("size", [0, 65, 6.5, True, "6"])
def test_browser_ring_limit_validation(size):
    with pytest.raises(ValueError, match="browser_ring_size"):
        recipe_for({**DEFAULT_SETTINGS, "browser_ring_size": size})


@pytest.mark.parametrize("value", [499, 40001, 0, -1, 6000.0, "6000", True])
def test_bitrate_outside_the_range_is_rejected(value):
    with pytest.raises(ValueError, match="bitrate_kbps"):
        recipe_for({**DEFAULT_SETTINGS, "bitrate_kbps": value})


@pytest.mark.parametrize("bit_depth", [8, 10])
@pytest.mark.parametrize("fps, maximum", [(25, 110), (30, 83), (50, 50), (60, 41)])
def test_setup_limits_in_both_modes(bit_depth, fps, maximum):
    settings = {**DEFAULT_SETTINGS, "bit_depth": bit_depth, "fps": fps,
                "chroma": "420" if bit_depth == 8 else "422",
                "source_count": maximum, "scene_count": 192,
                "weights": [1, 0, 0, 0 if bit_depth == 8 else 1, 1, 1, 0]}
    feasible = min(maximum, (40 if fps <= 30 else 20) + 32 + min(28, 700 // fps))
    assert recipe_for({**settings, "source_count": feasible})["source_count"] == feasible
    with pytest.raises(ValueError, match="source_count"):
        recipe_for({**settings, "source_count": maximum + 1})
    with pytest.raises(ValueError, match="scene_count"):
        recipe_for({**settings, "scene_count": 193})


def test_hdr_420_canvas_and_assets(tmp_path):
    recipe = recipe_for({**DEFAULT_SETTINGS, "chroma": "420", "weights": [8, 6, 0, 0, 2]})
    show, jobs, _ = prepare_demo.plan(recipe, tmp_path)
    assert show["canvas"]["working_format"] == "p010le"
    assert show["canvas"]["color"] == "hlg"
    assert len(show["renditions"]) == 2
    assert any(s["color"] == "hlg" for s in show["sources"] if s["kind"] == "video")
    assert not any(p.suffix == ".v210" for p in jobs)


@pytest.mark.parametrize("bit_depth,chroma", [(8, "420"), (10, "420"), (10, "422")])
def test_raw_sdr_420_mix_in_every_canvas_mode(tmp_path, bit_depth, chroma):
    settings = {**DEFAULT_SETTINGS, "source_count": 6, "weights": [2, 0, 0, 0, 1, 3],
                "bit_depth": bit_depth, "chroma": chroma}
    show, jobs, _ = prepare_demo.plan(recipe_for(settings), tmp_path)
    from collections import Counter
    assert Counter(s["kind"] for s in show["sources"]) == {"video": 2, "browser": 1, "nv12": 3}
    assert len({s["id"] for s in show["sources"]}) == 6
    assert all(s["color"] == "sdr" for s in show["sources"] if s["kind"] == "nv12")
    assert sum(p.suffix == ".nv12" for p in jobs) == 3


def test_legacy_source_weights_get_zero_raw_uploads():
    settings = recipe_for({**DEFAULT_SETTINGS, "weights": [8, 4, 2, 0, 2]})["setup"]
    assert settings["weights"] == [8, 4, 2, 0, 2, 0, 0]


@pytest.mark.parametrize("chroma", ["420", "422"])
def test_hdr_raw_420_uses_p010_without_nvdec(tmp_path, chroma):
    settings = {**DEFAULT_SETTINGS, "fps": 30, "chroma": chroma, "source_count": 6,
                "weights": [1, 1, 0, 0, 1, 1, 2]}
    show, jobs, _ = prepare_demo.plan(recipe_for(settings), tmp_path)
    raw = [s for s in show["sources"] if s["kind"] == "p010"]
    assert len(raw) == 2 and all(s["color"] == "hlg" for s in raw)
    assert sum(s["kind"] == "video" for s in show["sources"]) == 2
    assert sum(path.suffix == ".p010" for path in jobs) == 2
    from pyplumber.mixer.config import parse
    assert parse(show).settings()["source_counts"]["p010"] == 2
    with pytest.raises(ValueError, match="8-bit mode"):
        recipe_for({**settings, "bit_depth": 8, "chroma": "420"})


@pytest.mark.parametrize("fps,limit", [(25, 14), (30, 11), (50, 7), (60, 5)])
def test_hdr_raw_upload_counts_twice_toward_byte_budget(fps, limit):
    weights = [0, 0, 0, 0, 0, 0, 1]
    assert source_counts(limit, weights, fps)[6] == limit
    with pytest.raises(ValueError, match="upload units"):
        source_counts(limit + 1, weights, fps)
    mixed = source_counts(32, [1, 1, 0, 0, 1, 1, 1], fps)
    assert mixed[5] + 2 * mixed[6] <= min(28, 700 // fps)


@pytest.mark.parametrize("fps,limit", [(25, 40), (30, 40), (50, 20), (60, 20)])
def test_sdr_and_hdr_share_nvdec_budget(fps, limit):
    counts = source_counts(48, [100, 100, 0, 0, 1, 0, 0], fps)
    assert counts[0] + counts[1] == limit
    assert sum(counts) == 48


def test_raw_only_mix_keeps_a_steady_alpha_background(tmp_path):
    recipe = recipe_for({**DEFAULT_SETTINGS, "source_count": 2, "weights": [0, 0, 0, 0, 1, 1]})
    show, jobs, _ = prepare_demo.plan(recipe, tmp_path)
    assert recipe["alpha_background"] == "sdr420_raw_000"
    assert any(path.name == "sdr_420_bars_nv12.nv12" for path in jobs)
    assert len(show["sources"]) == 2


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
    assert recipe["setup"]["weights"] == weights + [0, 0]


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
    assert [source['weight'] for source in recipe['inputs']] == expected + [0, 0]


def test_browser_only_limit():
    settings = {**DEFAULT_SETTINGS, 'fps': 30, 'source_count': 32, 'weights': [0, 0, 0, 0, 1]}
    assert next(s for s in recipe_for(settings)['inputs'] if s['kind'] == 'browser')['weight'] == 32
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


@pytest.mark.parametrize("limit,tiles", [(256, 2), (512, 6)])
def test_setup_clears_aux_tiles_that_exceed_new_draw_budget(runtime, limit, tiles):
    recipe = recipe_for({**DEFAULT_SETTINGS, "fps": 30, "source_count": 64,
                         "weights": [1, 0, 0, 0, 1], "layout": "grids"})
    show, _, _ = prepare_demo.plan(recipe, runtime.media_dir)
    grid = next(s["id"] for s in show["scenes"] if s["id"].startswith("grid_64_"))
    show["aux_buses"] = [{"id": "mv", "scenes": [grid] * 8,
                          "renditions": [{"id": "monitor", "port": 5008}]}]
    show["max_compositor_layers"] = limit
    (runtime.media_dir / "mixer.demo.json").write_text(json.dumps(show))
    runtime.process = None
    del show["aux_buses"]
    runtime._preserve_aux(recipe, show)
    result, _, _ = prepare_demo.plan(recipe, runtime.media_dir)
    assert result["max_compositor_layers"] == limit
    assert result["aux_buses"][0]["scenes"] == [grid] * tiles + [None] * (8 - tiles)


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


@pytest.mark.parametrize("fps, maximum", [(25, 28), (30, 23), (50, 14), (60, 11)])
def test_raw_upload_budget(fps, maximum):
    settings = {**DEFAULT_SETTINGS, "fps": fps, "source_count": maximum,
                "weights": [0, 0, 0, 0, 0, 1]}
    assert recipe_for(settings)["inputs"][5]["weight"] == maximum
    with pytest.raises(ValueError, match=f"Raw 4:2:0 upload units is limited to {maximum}"):
        recipe_for({**settings, "source_count": maximum + 1})
    counts = source_counts(maximum + 4, [1, 0, 0, 0, 0, 100], fps)
    assert counts == [4, 0, 0, 0, 0, maximum]


def test_combined_source_caps():
    assert source_counts(100, [1, 0, 100, 0, 100, 100]) == [36, 0, 4, 0, 32, 28]
    with pytest.raises(ValueError, match="enable another source type"):
        source_counts(65, [0, 0, 1, 0, 1, 1])


def test_192_scenes_expand(tmp_path):
    recipe = recipe_for({**DEFAULT_SETTINGS, "scene_count": 192})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    assert len(show["scenes"]) == 192


def test_shutdown_reaps_killed_child_before_recovery(tmp_path, monkeypatch):
    import signal
    import subprocess
    from unittest.mock import Mock
    events = []
    process = Mock(pid=1234, returncode=None)
    process.poll.return_value = None
    def wait(timeout):
        events.append(('wait', timeout))
        if timeout == 120:
            raise subprocess.TimeoutExpired('mixer', timeout)
        process.returncode = -signal.SIGKILL
    process.wait.side_effect = wait
    monkeypatch.setattr('setup_runtime.os.killpg', lambda pid, sig: events.append(('signal', sig)))
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777))
    manager.process = process
    manager._stop()
    assert events == [('signal', signal.SIGINT), ('wait', 120), ('signal', signal.SIGKILL), ('wait', 10)]
    assert manager.process is None


def test_browser_recovery_requires_dead_consumer(tmp_path, monkeypatch):
    calls = []
    def request(url, method, path, body=None):
        calls.append((method, path, body))
        return {'windows': [{'id': 'browser', 'stats': {'quarantinedFrameCount': 6}}]}
    monkeypatch.setattr('pyplumber.mixer.dmabuf_inputs.rest_request', request)
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777))
    shows = [{'sources': [{'id': 'browser', 'kind': 'browser'}]}]
    manager.process = SimpleNamespace(poll=lambda: None)
    with pytest.raises(RuntimeError, match='while the mixer is running'):
        manager._recover_browsers(shows)
    assert calls == []
    manager.process = SimpleNamespace(poll=lambda: -9)
    assert manager._recover_browsers(shows)
    assert calls[-1] == ('POST', '/workers/recover', {'ids': ['browser']})


def test_late_quarantine_retries_requested_setup_once(runtime, monkeypatch):
    config = runtime.media_dir / 'mixer.demo.json'
    config.write_text('{"sources": []}')
    events = []
    checks = iter([False, True])
    monkeypatch.setattr(runtime, '_recover_browsers', lambda _: next(checks))
    monkeypatch.setattr(runtime, '_stop', lambda: events.append('reaped'))
    def start(path):
        events.append('start')
        if events.count('start') == 1:
            raise RuntimeError('quarantined buffers')
    monkeypatch.setattr(runtime, '_start', start)
    runtime._start_recovering(config, None)
    assert events == ['start', 'reaped', 'start']


def test_permanent_start_failure_is_not_retried(runtime, monkeypatch):
    config = runtime.media_dir / 'mixer.demo.json'
    config.write_text('{"sources": []}')
    from unittest.mock import Mock
    start = Mock(side_effect=RuntimeError('invalid encoder'))
    monkeypatch.setattr(runtime, '_start', start)
    with pytest.raises(RuntimeError, match='invalid encoder'):
        runtime._start_recovering(config, None)
    assert start.call_count == 1


def test_recovery_precedes_removing_quarantined_windows(runtime, monkeypatch):
    config = runtime.media_dir / 'mixer.demo.json'
    current = {'sources': [{'id': 'current', 'kind': 'browser'}]}
    previous = {'sources': [{'id': 'removed', 'kind': 'browser'}]}
    config.write_text(json.dumps(current))
    events = []
    monkeypatch.setattr(runtime, '_recover_browsers', lambda shows: events.append(('recover', shows)))
    monkeypatch.setattr(runtime, '_close_removed_browsers', lambda *shows: events.append(('close', shows)))
    monkeypatch.setattr(runtime, '_start', lambda _: events.append(('start', None)))
    runtime._start_recovering(config, previous)
    assert events == [('recover', [current, previous]), ('close', (previous, current)), ('start', None)]


def test_unreaped_process_blocks_browser_recovery(tmp_path, monkeypatch):
    import subprocess
    from unittest.mock import Mock
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777))
    process = Mock(pid=1234)
    process.poll.return_value = None
    process.wait.side_effect = subprocess.TimeoutExpired('mixer', 10)
    manager.process = process
    monkeypatch.setattr('setup_runtime.os.killpg', lambda *_: None)
    with pytest.raises(subprocess.TimeoutExpired):
        manager._stop()
    assert manager.process is process
    with pytest.raises(RuntimeError, match='while the mixer is running'):
        manager._recover_browsers([])


def test_healthy_browsers_are_not_restarted(tmp_path, monkeypatch):
    calls = []
    def request(url, method, path, body=None):
        calls.append(path)
        return {'windows': [{'id': 'browser', 'stats': {'quarantinedFrameCount': 0}}]}
    monkeypatch.setattr('pyplumber.mixer.dmabuf_inputs.rest_request', request)
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777))
    assert not manager._recover_browsers([{'sources': [{'id': 'browser', 'kind': 'browser'}]}])
    assert calls == ['/status']
