"""Setup changes validate before touching the live process and recover on failure."""
import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from instance_profiles import INSTANCE_PROFILES, InstanceType
import prepare_demo
from pyplumber.mixer.aux_layout import draws_program
from pyplumber.mixer.config import parse
from setup_runtime import (DEFAULT_SETTINGS, HEALTHY_RUN_SEC, RECOVER_TIMEOUT_SEC, RETRY_DELAYS_SEC, STOP_TIMEOUT_SEC,
                           SetupRuntime, extra_aux_limit, recipe_for, source_counts, source_limit)
from webui import serve

T4 = INSTANCE_PROFILES[InstanceType.TESLA_T4]


@pytest.mark.parametrize('count', [1, 8, 16, 32, 41, 42, 48, 50, 64, 83, 96, 100, 110])
@pytest.mark.parametrize('fps', [25, 30, 50, 60])
def test_generic_setups_expand(tmp_path, count, fps):
    # DEFAULT_SETTINGS is a 10-bit 4:2:2 canvas: above its capacity the setup scales the show down.
    maximum = source_limit(T4, fps, DEFAULT_SETTINGS["bit_depth"], DEFAULT_SETTINGS["chroma"])
    count = min(count, maximum)
    if count > T4["nvdec_decodes"][fps] + T4["browser_windows"] + T4["hlg_v210"]:
        with pytest.raises(ValueError, match="enable another source type"):
            recipe_for(T4, {**DEFAULT_SETTINGS, "source_count": count, "fps": fps})
        return
    recipe = recipe_for(T4, {**DEFAULT_SETTINGS, 'source_count': count, 'fps': fps})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    assert len(show['sources']) == count
    assert len(show['scenes']) == DEFAULT_SETTINGS['scene_count']
    assert show['canvas']['fps'] == fps


@pytest.mark.parametrize('changes', [{'source_count': 0}, {'scene_count': 0}, {'scene_count': 193}, {'fps': 24},
    {'weights': [0] * 5}, {'weights': [True] * 5}, {'resolution': '../../file'}, {'command': 'id'}])
def test_reject_unbounded_settings(changes):
    with pytest.raises(ValueError):
        recipe_for(T4, {**DEFAULT_SETTINGS, **changes})


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
    manager.process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(manager, '_stop', lambda: None)
    monkeypatch.setattr(manager, '_close_removed_browsers', lambda *_: None)
    monkeypatch.setattr(manager, '_recover_browsers', lambda *_: False)
    monkeypatch.setattr(manager, '_start', lambda _: setattr(manager, 'process', SimpleNamespace(poll=lambda: None)))
    monkeypatch.setattr(manager, '_watch', lambda _: None)
    return manager


class FakeTimer:
    def __init__(self, delay, function):
        self.delay, self.function, self.cancelled, self.daemon = delay, function, False, False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def timers(monkeypatch):
    created = []
    monkeypatch.setattr('setup_runtime.threading.Timer', lambda *a: created.append(FakeTimer(*a)) or created[-1])
    return created


def crash(runtime, code=-11):
    """The running mixer exits on its own; the real watcher sees it."""
    process = SimpleNamespace(wait=lambda: code, poll=lambda: code)
    runtime.process = process
    SetupRuntime._watch(runtime, process)


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
                "weights": [4, 0, 0, 0, 1, 0, 0]}
    show, jobs, _ = prepare_demo.plan(recipe_for(T4, settings), tmp_path)
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
        recipe_for(T4, {**DEFAULT_SETTINGS, "bit_depth": 8})


def test_bitrate_is_configurable_and_scales_every_rendition():
    """One control sets the SDR bitrate; other renditions keep their ratio to it."""
    base = recipe_for(T4, DEFAULT_SETTINGS)["renditions"]
    assert [r["bitrate_kbps"] for r in base] == [6000, 8000]

    halved = recipe_for(T4, {**DEFAULT_SETTINGS, "bitrate_kbps": 3000})["renditions"]
    assert [r["bitrate_kbps"] for r in halved] == [3000, 4000]


@pytest.mark.parametrize("fps, size", [(25, 6), (30, 6), (50, 9), (60, 9)])
def test_browser_ring_default_tracks_fps(tmp_path, fps, size):
    # Settings saved with a ring size load; the setup leaves it to the frame rate's default.
    recipe = recipe_for(T4, {**DEFAULT_SETTINGS, "fps": fps, "browser_ring_size": 11})
    assert "browser_ring_size" not in recipe
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    from pyplumber.mixer.config import parse
    from mixer import GraphOptions
    assert show["browser_ring_size"] == size
    del show["browser_ring_size"]
    assert parse(show).browser_ring_size == size
    assert GraphOptions(fps=fps).browser_ring_size == size


@pytest.mark.parametrize("size", [6, 9, 11])
def test_browser_ring_limit_reaches_show(tmp_path, size):
    recipe = {**recipe_for(T4, DEFAULT_SETTINGS), "browser_ring_size": size}   # a hand-written recipe
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    from pyplumber.mixer.config import parse
    assert parse(show).browser_ring_size == size
    assert parse(show).settings()["browser_ring_size"] == size


@pytest.mark.parametrize("value", [499, 40001, 0, -1, 6000.0, "6000", True])
def test_bitrate_outside_the_range_is_rejected(value):
    with pytest.raises(ValueError, match="bitrate_kbps"):
        recipe_for(T4, {**DEFAULT_SETTINGS, "bitrate_kbps": value})


@pytest.mark.parametrize("bit_depth", [8, 10])
@pytest.mark.parametrize("fps, maximum", [(25, 110), (30, 110), (50, 90), (60, 75)])
def test_setup_limits_in_both_modes(bit_depth, fps, maximum):
    chroma = "420" if bit_depth == 8 else "422"
    maximum = source_limit(T4, fps, bit_depth, chroma)   # the SDR rate limit, scaled for a 10-bit canvas
    settings = {**DEFAULT_SETTINGS, "bit_depth": bit_depth, "fps": fps, "chroma": chroma,
                "source_count": maximum, "scene_count": 192,
                "weights": [1, 0, 0, 0 if bit_depth == 8 else 1, 1, 1, 0]}
    feasible = min(maximum, T4["nvdec_decodes"][fps] + T4["browser_windows"] + T4["raw_upload_units"][fps])
    assert recipe_for(T4, {**settings, "source_count": feasible})["source_count"] == feasible
    if feasible == maximum:   # above the limit the show is scaled down to it, not refused
        assert recipe_for(T4, {**settings, "source_count": maximum + 1})["source_count"] == maximum
    with pytest.raises(ValueError, match="scene_count"):
        recipe_for(T4, {**settings, "scene_count": 193})


def test_hdr_420_canvas_and_assets(tmp_path):
    recipe = recipe_for(T4, {**DEFAULT_SETTINGS, "chroma": "420", "weights": [8, 6, 0, 0, 2, 0, 0]})
    show, jobs, _ = prepare_demo.plan(recipe, tmp_path)
    assert show["canvas"]["working_format"] == "p010le"
    assert show["canvas"]["color"] == "hlg"
    assert len(show["renditions"]) == 2
    assert any(s["color"] == "hlg" for s in show["sources"] if s["kind"] == "video")
    assert not any(p.suffix == ".v210" for p in jobs)


@pytest.mark.parametrize("bit_depth,chroma", [(8, "420"), (10, "420"), (10, "422")])
def test_raw_sdr_420_mix_in_every_canvas_mode(tmp_path, bit_depth, chroma):
    settings = {**DEFAULT_SETTINGS, "source_count": 6, "weights": [2, 0, 0, 0, 1, 3, 0],
                "bit_depth": bit_depth, "chroma": chroma}
    show, jobs, _ = prepare_demo.plan(recipe_for(T4, settings), tmp_path)
    from collections import Counter
    assert Counter(s["kind"] for s in show["sources"]) == {"video": 2, "browser": 1, "nv12": 3}
    assert len({s["id"] for s in show["sources"]}) == 6
    assert all(s["color"] == "sdr" for s in show["sources"] if s["kind"] == "nv12")
    assert sum(p.suffix == ".nv12" for p in jobs) == 3


@pytest.mark.parametrize("chroma", ["420", "422"])
def test_hdr_raw_420_uses_p010_without_nvdec(tmp_path, chroma):
    settings = {**DEFAULT_SETTINGS, "fps": 30, "chroma": chroma, "source_count": 6,
                "weights": [1, 1, 0, 0, 1, 1, 2]}
    show, jobs, _ = prepare_demo.plan(recipe_for(T4, settings), tmp_path)
    raw = [s for s in show["sources"] if s["kind"] == "p010"]
    assert len(raw) == 2 and all(s["color"] == "hlg" for s in raw)
    assert sum(s["kind"] == "video" for s in show["sources"]) == 2
    assert sum(path.suffix == ".p010" for path in jobs) == 2
    from pyplumber.mixer.config import parse
    assert parse(show).settings()["source_counts"]["p010"] == 2
    with pytest.raises(ValueError, match="8-bit mode"):
        recipe_for(T4, {**settings, "bit_depth": 8, "chroma": "420"})


@pytest.mark.parametrize("fps,limit", [(25, 15), (30, 17), (50, 10), (60, 8)])
def test_hdr_raw_upload_counts_twice_toward_byte_budget(fps, limit):
    weights = [0, 0, 0, 0, 0, 0, 1]
    assert source_counts(T4, limit, weights, fps)[6] == limit
    with pytest.raises(ValueError, match="upload units"):
        source_counts(T4, limit + 1, weights, fps)
    mixed = source_counts(T4, 32, [1, 1, 0, 0, 1, 1, 1], fps)
    assert mixed[5] + 2 * mixed[6] <= T4["raw_upload_units"][fps]


@pytest.mark.parametrize("fps,limit", [(25, 40), (30, 36), (50, 22), (60, 18)])
def test_sdr_and_hdr_share_nvdec_budget(fps, limit):
    counts = source_counts(T4, 48, [100, 100, 0, 0, 1, 0, 0], fps)
    assert counts[0] + counts[1] == limit
    assert sum(counts) == 48


def test_raw_only_mix_keeps_a_steady_alpha_background(tmp_path):
    recipe = recipe_for(T4, {**DEFAULT_SETTINGS, "source_count": 2, "weights": [0, 0, 0, 0, 1, 1, 0]})
    show, jobs, _ = prepare_demo.plan(recipe, tmp_path)
    assert recipe["alpha_background"] == "sdr420_raw_000"
    assert any(path.name == "sdr420_raw_000_sdr_420_bars_nv12.nv12" for path in jobs)
    assert len(show["sources"]) == 2


@pytest.mark.parametrize("changes, message", [
    ({"chroma": "444"}, "Unsupported chroma"),
    ({"chroma": "420"}, "4:2:0 mode"),
    ({"bit_depth": 8, "weights": [1, 0, 0, 0, 0, 0, 0]}, "4:2:0 canvas"),
    ({"weights": [0, 0, 1, 0, 0, 0, 0]}, "enable another source type"),
])
def test_reject_incompatible_chroma(changes, message):
    with pytest.raises(ValueError, match=message):
        recipe_for(T4, {**DEFAULT_SETTINGS, **changes})


@pytest.mark.parametrize("total", [8, 16, 24, 32, 42, 60])
@pytest.mark.parametrize("weights", [[8, 4, 2, 0, 2], [1, 1, 100, 0, 1]])
def test_hdr_422_cap_preserves_total_and_disabled_types(tmp_path, total, weights):
    recipe = recipe_for(T4, {**DEFAULT_SETTINGS, "fps": 25, "source_count": total, "weights": weights + [0, 0]})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    counts = source_counts(T4, total, weights)
    assert len(show["sources"]) == sum(counts) == total
    assert 0 < counts[2] <= 4
    assert counts[3] == 0
    assert sum(s["kind"] == "v210" for s in show["sources"]) == counts[2]
    assert recipe["setup"]["weights"] == weights + [0, 0]


def test_four_hdr_422_inputs_can_be_used_alone():
    assert source_counts(T4, 4, [0, 0, 1, 0, 0]) == [0, 0, 4, 0, 0]


@pytest.mark.parametrize('weights, expected', [
    ([1, 0, 0, 0, 100], [24, 0, 0, 0, 40]),
    ([1, 0, 100, 0, 100], [20, 0, 4, 0, 40]),
    ([1, 0, 1, 0, 100], [20, 0, 4, 0, 40]),
])
def test_browser_cap_redistributes_without_exceeding_other_caps(weights, expected):
    assert source_counts(T4, 64, weights) == expected
    # Above the capacity of a 10-bit 4:2:2 canvas at 25 fps the show is scaled down first.
    recipe = recipe_for(T4, {**DEFAULT_SETTINGS, 'source_count': 64, 'fps': 25, 'weights': weights + [0, 0]})
    capacity = min(64, source_limit(T4, 25, DEFAULT_SETTINGS['bit_depth'], DEFAULT_SETTINGS['chroma']))
    assert [source['weight'] for source in recipe['inputs']] == source_counts(T4, capacity, weights) + [0, 0]


def test_browser_only_limit():
    settings = {**DEFAULT_SETTINGS, 'fps': 50, 'bit_depth': 8, 'chroma': '420', 'source_count': 40, 'weights': [0, 0, 0, 0, 1, 0, 0]}
    assert next(s for s in recipe_for(T4, settings)['inputs'] if s['kind'] == 'browser')['weight'] == 40
    with pytest.raises(ValueError, match='Browser is limited to 40'):
        recipe_for(T4, {**settings, 'source_count': 41})
    with pytest.raises(ValueError, match='enable another source type'):
        source_counts(T4, 45, [0, 0, 1, 0, 1], 50)


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
    first = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
    monkeypatch.setattr('setup_runtime.time.time_ns', lambda: 2_000_000_000_000)
    restarted = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
    assert first.status()['revision'] != restarted.status()['revision']
    assert restarted.status()['revision'] < 2 ** 53


@pytest.mark.parametrize("before,after", [(8, 10), (10, 8)])
def test_setup_preserves_live_aux_through_color_change_and_resume(runtime, monkeypatch, before, after):
    def settings(depth):
        return {**DEFAULT_SETTINGS, "bit_depth": depth, "chroma": "420",
                "weights": [4, 0, 0, 0, 1, 0, 0]}
    config = runtime.media_dir / "mixer.demo.json"
    show, _, _ = prepare_demo.plan(recipe_for(T4, settings(before)), runtime.media_dir)
    scene_ids = [s["id"] for s in show["scenes"]]
    show["aux_buses"] = [{"id": "mv", "scenes": [scene_ids[0]] * 8,
                          "renditions": [{"id": "monitor", "port": 5008}]}]
    config.write_text(json.dumps(show))
    live = [scene_ids[1], scene_ids[-1], *([None] * 6)]
    # The live layout and layouts carry over, the grid's PGM pad with them; pages follow the new
    # source list from page 0.
    layout = {"preset": "source_pages", "page": 1}
    layouts = [{"preset": "pgm_pvw_grid"}, layout]
    runtime.bridge.command = lambda cmd: json.dumps([{"id": "mv", "layout": layout, "layouts": layouts, "scenes": live}])
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
    assert changed["aux_buses"][0]["layout"] == {"preset": "source_pages"}
    assert changed["aux_buses"][0]["layouts"] == [{"preset": "pgm_pvw_grid"}, {"preset": "source_pages"}]
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
    old, _, _ = prepare_demo.plan(recipe_for(T4, DEFAULT_SETTINGS), runtime.media_dir)
    old["aux_buses"] = [{"id": "mv", "scenes": [old["scenes"][0]["id"], "removed", *([None] * 6)],
                         "renditions": [{"id": "monitor", "port": 5008, "width": 1080,
                                         "height": 1920, "fps": 30, "bitrate_kbps": 4500}]},
                        {"id": "mv2", "layout": {"preset": "source_pages"},
                         "renditions": [{"id": "monitor", "port": 5012, "width": 1080, "height": 1920, "fps": 30}]},
                        {"id": "mv3", "full_rate": True, "pvw_align": "pgm_tile",
                         "renditions": [{"id": "monitor", "port": 5016, "width": 1080, "height": 1920, "fps": 60}]}]
    (runtime.media_dir / "mixer.demo.json").write_text(json.dumps(old))
    runtime.process = None
    recipe = recipe_for(T4, {**DEFAULT_SETTINGS, "orientation": "landscape",
                         "fps": 50, "scene_count": 1, "layout": "fullscreen"})
    show, _, _ = prepare_demo.plan(recipe, runtime.media_dir)
    runtime._preserve_aux(recipe, show)
    cfg = parse(prepare_demo.plan(recipe, runtime.media_dir)[0])
    assert cfg.aux_buses[0].scenes == (old["scenes"][0]["id"], *([None] * 7))
    r = cfg.aux_buses[0].renditions[0]
    assert (r.width, r.height, r.fps, r.bitrate_kbps) == (1920, 1080, 25, 4500)
    pages = cfg.aux_buses[1]
    assert (pages.layout, pages.scenes) == ({"preset": "source_pages", "page": 0}, ())
    assert (pages.renditions[0].width, pages.renditions[0].height, pages.renditions[0].fps) == (1920, 1080, 25)
    full = cfg.aux_buses[2]   # a full-rate bus follows the new canvas rate, not half of it
    assert (full.full_rate, full.pvw_align, full.renditions[0].fps) == (True, "pgm_tile", 50)


@pytest.mark.parametrize("limit,tiles", [(256, 2), (512, 6)])
def test_setup_clears_aux_tiles_that_exceed_new_draw_budget(runtime, limit, tiles):
    recipe = recipe_for(T4, {**DEFAULT_SETTINGS, "fps": 30, "source_count": 64, "bit_depth": 8, "chroma": "420",
                         "weights": [1, 0, 0, 0, 1, 0, 0], "layout": "grids"})
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
    monkeypatch.setattr("setup_runtime.subprocess.Popen", lambda *a, **kw: SimpleNamespace(pid=1234, poll=lambda: None))
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


@pytest.mark.parametrize("fps, maximum", [(25, 30), (30, 34), (50, 20), (60, 17)])
def test_raw_upload_budget(fps, maximum):
    settings = {**DEFAULT_SETTINGS, "fps": fps, "source_count": maximum,
                "weights": [0, 0, 0, 0, 0, 1, 0]}
    assert recipe_for(T4, settings)["inputs"][5]["weight"] == maximum
    with pytest.raises(ValueError, match=f"Raw 4:2:0 upload units is limited to {maximum}"):
        recipe_for(T4, {**settings, "source_count": maximum + 1})
    counts = source_counts(T4, maximum + 4, [1, 0, 0, 0, 0, 100], fps)
    assert counts == [4, 0, 0, 0, 0, maximum]


def test_combined_source_caps():
    assert source_counts(T4, 100, [1, 0, 100, 0, 100, 100]) == [26, 0, 4, 0, 40, 30]
    with pytest.raises(ValueError, match="enable another source type"):
        source_counts(T4, 75, [0, 0, 1, 0, 1, 1])


def test_192_scenes_expand(tmp_path):
    recipe = recipe_for(T4, {**DEFAULT_SETTINGS, "scene_count": 192})
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
        if timeout == STOP_TIMEOUT_SEC:
            raise subprocess.TimeoutExpired('mixer', timeout)
        process.returncode = -signal.SIGKILL
    process.wait.side_effect = wait
    monkeypatch.setattr('setup_runtime.os.killpg', lambda pid, sig: events.append(('signal', sig)))
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
    manager.process = process
    manager._stop()
    assert events == [('signal', signal.SIGINT), ('wait', STOP_TIMEOUT_SEC), ('signal', signal.SIGKILL), ('wait', 10)]
    assert manager.process is None


def exiting_process(events):
    """A mixer that exits once `exited` is set after its SIGINT."""
    from unittest.mock import Mock
    exited = threading.Event()
    process = Mock(pid=1234, returncode=0)
    process.poll.side_effect = lambda: 0 if exited.is_set() else None
    process.wait.side_effect = lambda timeout: events.append('exited') if exited.wait(3) else None
    return process, exited


def test_close_stops_the_mixer_before_waiting_for_setup_work(tmp_path, monkeypatch):
    import signal
    events = []
    process, exited = exiting_process(events)
    monkeypatch.setattr('setup_runtime.os.killpg', lambda pid, sig: (events.append(sig), exited.set()))
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
    manager.process = process
    manager.worker = SimpleNamespace(join=lambda timeout: events.append('joined'))
    manager.close()
    assert events == [signal.SIGINT, 'exited', 'joined']


def test_concurrent_stops_signal_the_mixer_once(tmp_path, monkeypatch):
    import signal
    import time
    events = []
    process, exited = exiting_process(events)
    monkeypatch.setattr('setup_runtime.os.killpg', lambda pid, sig: events.append(sig))
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
    manager.process = process
    stops = [threading.Thread(target=manager._stop) for _ in range(2)]
    for stop in stops:
        stop.start()
        time.sleep(0.05)   # the second one waits for the first
    exited.set()
    for stop in stops:
        stop.join(3)
    assert events == [signal.SIGINT, 'exited']
    assert manager.process is None


def test_browser_recovery_requires_dead_consumer(tmp_path, monkeypatch):
    calls = []
    def request(url, method, path, body=None, timeout=60):
        calls.append((method, path, body, timeout))
        return {'windows': [{'id': 'browser', 'stats': {'quarantinedFrameCount': 6}}]}
    monkeypatch.setattr('pyplumber.mixer.dmabuf_inputs.rest_request', request)
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
    shows = [{'sources': [{'id': 'browser', 'kind': 'browser'}]}]
    manager.process = SimpleNamespace(poll=lambda: None)
    with pytest.raises(RuntimeError, match='while the mixer is running'):
        manager._recover_browsers(shows)
    assert calls == []
    manager.process = SimpleNamespace(poll=lambda: -9)
    assert manager._recover_browsers(shows)
    assert calls[-1] == ('POST', '/workers/recover', {'ids': ['browser']}, RECOVER_TIMEOUT_SEC)


def test_recovery_timeout_is_not_recovered_again_by_the_rollback(runtime, monkeypatch):
    config = runtime.media_dir / 'mixer.demo.json'
    config.write_text('{"old": true}')
    monkeypatch.setattr(prepare_demo, 'prepare', lambda *_: config.write_text('{"new": true}'))
    recoveries = []
    def recover(shows):
        recoveries.append(shows)
        raise TimeoutError('Browser POST /workers/recover: no answer within 180 s')
    monkeypatch.setattr(runtime, '_recover_browsers', recover)
    monkeypatch.setattr(runtime, '_start', lambda _: pytest.fail('must not start while workers restart'))
    runtime.apply(DEFAULT_SETTINGS)
    runtime.worker.join(3)
    assert len(recoveries) == 1
    assert 'still busy' in runtime.status()['message']
    assert json.loads(config.read_text()) == {'old': True}


def test_unexpected_exit_restarts_after_a_growing_capped_backoff(runtime, timers):
    config = runtime.media_dir / 'mixer.demo.json'
    config.write_text('{"sources": []}')
    starts = []
    started = runtime._start
    runtime._start = lambda path: (starts.append(path), started(path))
    revision = runtime.revision
    for attempt, delay in enumerate([*RETRY_DELAYS_SEC, RETRY_DELAYS_SEC[-1]]):
        crash(runtime)
        assert timers[-1].delay == delay
        assert runtime.status()['phase'] == 'error'
        assert f'Mixer exited (-11); restarting in {delay} s' in runtime.status()['message']
        timers[-1].function()
        runtime.worker.join(3)
        assert len(starts) == attempt + 1
        assert runtime.status()['phase'] == 'running'
    assert runtime.status()['revision'] == revision + len(starts)


def test_a_healthy_run_resets_the_backoff(runtime, timers, monkeypatch):
    runtime.retries = 3
    clock = iter([0.0, HEALTHY_RUN_SEC])
    monkeypatch.setattr('setup_runtime.time.monotonic', lambda: next(clock))
    crash(runtime)
    assert timers[-1].delay == RETRY_DELAYS_SEC[0]


def test_failed_restart_backs_off_further(runtime, timers):
    (runtime.media_dir / 'mixer.demo.json').write_text('{"sources": []}')
    def fail(_):
        raise RuntimeError('GPU lost')
    runtime._start = fail
    crash(runtime)
    timers[-1].function()
    runtime.worker.join(3)
    assert timers[-1].delay == RETRY_DELAYS_SEC[1]
    assert 'Mixer restart failed: GPU lost; restarting in 5 s' in runtime.status()['message']


def test_deliberate_stops_and_setup_changes_are_not_retried(runtime, timers):
    process = SimpleNamespace(wait=lambda: 0)
    runtime.process = None   # _stop() replaced it
    SetupRuntime._watch(runtime, process)
    runtime.worker = SimpleNamespace(is_alive=lambda: True)   # an Apply owns the mixer
    crash(runtime)
    runtime.worker = None
    runtime.closing.set()
    crash(runtime)
    assert timers == []


def test_real_process_stop_is_not_a_crash_but_a_kill_is(tmp_path, timers):
    import subprocess
    import sys
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
    def spawn_watched():
        manager.process = subprocess.Popen(
            [sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGINT, lambda *_: exit(0)); '
                                   'print(flush=True); time.sleep(30)'],
            stdout=subprocess.PIPE, start_new_session=True)
        manager.process.stdout.readline()   # its SIGINT handler is installed
        watcher = threading.Thread(target=manager._watch, args=(manager.process,))
        watcher.start()
        return watcher
    watcher = spawn_watched()
    manager._stop()   # two threads wait on one Popen: the watcher must still see a deliberate stop
    watcher.join(5)
    assert not watcher.is_alive() and timers == []
    watcher = spawn_watched()
    manager.process.kill()
    watcher.join(5)
    assert [t.delay for t in timers] == [RETRY_DELAYS_SEC[0]]
    manager.process.stdout.close()


def test_apply_cancels_a_pending_restart(runtime, timers, monkeypatch):
    crash(runtime)
    pending = timers[-1]
    monkeypatch.setattr(prepare_demo, 'prepare', lambda *_: (runtime.media_dir / 'mixer.demo.json').write_text('{}'))
    runtime.apply(DEFAULT_SETTINGS)
    runtime.worker.join(3)
    assert pending.cancelled and runtime.retries == 0
    applied = runtime.worker
    pending.function()   # fired just before it was cancelled
    assert runtime.worker is applied and runtime.status()['message'] == 'Mixer ready.'


def test_api_answers_409_to_a_concurrent_apply_without_asking_the_mixer(runtime, monkeypatch):
    runtime.worker = SimpleNamespace(is_alive=lambda: True)   # the first Apply is stopping the mixer
    monkeypatch.setattr(runtime, '_preserve_aux', lambda *_: pytest.fail('must not query the live mixer'))
    server = serve(SimpleNamespace(), '127.0.0.1', 0, setup=runtime)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        request = Request(f'http://127.0.0.1:{server.server_port}/api/setup', data=json.dumps(DEFAULT_SETTINGS).encode(),
                          headers={'Content-Type': 'application/json'})
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=3)
        assert error.value.code == 409
    finally:
        server.shutdown()
        server.server_close()


def test_a_torn_write_never_reaches_the_show(tmp_path, monkeypatch):
    from pathlib import Path
    from demo_recipe import write_atomic
    path = tmp_path / 'mixer.demo.json'
    path.write_text('{"old": true}')
    def torn(self, text, **_):
        Path.write_bytes(self, text[:4].encode())
        raise OSError('disk full')
    monkeypatch.setattr(Path, 'write_text', torn)
    with pytest.raises(OSError, match='disk full'):
        write_atomic(path, '{"new": true}')
    assert path.read_text() == '{"old": true}'


def test_live_aux_assignments_survive_resume(runtime, monkeypatch):
    settings = {**DEFAULT_SETTINGS, "bit_depth": 8, "chroma": "420", "weights": [4, 0, 0, 0, 1, 0, 0]}
    config = runtime.media_dir / "mixer.demo.json"
    recipe = recipe_for(T4, settings)
    show, _, _ = prepare_demo.plan(recipe, runtime.media_dir)
    scene_ids = [s["id"] for s in show["scenes"]]
    recipe["aux_buses"] = show["aux_buses"] = [{"id": "mv", "scenes": [scene_ids[0]] * 8,
                                                "renditions": [{"id": "monitor", "port": 5008}]}]
    config.write_text(json.dumps(show))
    runtime.recipe_path.write_text(json.dumps(recipe))
    live = [scene_ids[1], None, scene_ids[-1], *([None] * 5)]
    layout = {"preset": "source_pages", "page": 0}
    layouts = [{"preset": "pgm_pvw_grid"}, layout]   # a grid bus switched to its pages keeps the grid
    runtime.remember_aux("mv", {"scenes": live, "revision": "r"})
    runtime.remember_aux("mv", {"layout": layout, "layouts": layouts, "page": 0})
    runtime.remember_aux("unknown", {"scenes": live})
    assert json.loads(config.read_text())["aux_buses"][0] == {
        "id": "mv", "scenes": live, "layout": layout, "layouts": layouts, "renditions": [{"id": "monitor", "port": 5008}]}
    def prepare(recipe, directory):
        generated, _, _ = prepare_demo.plan(recipe, directory)
        config.write_text(json.dumps(generated))
    monkeypatch.setattr(prepare_demo, "prepare", prepare)
    runtime.process = None   # a container restart: nothing is running
    runtime.apply()          # resume
    runtime.worker.join(3)
    assert runtime.status()["phase"] == "running", runtime.status()
    from pyplumber.mixer.config import parse
    bus = parse(json.loads(config.read_text())).aux_buses[0]
    assert (bus.layout, list(bus.scenes), bus.layouts) == (layout, live, (layout, layouts[0]))


def test_stale_preparation_directories_are_removed_on_start(tmp_path):
    for stale in (tmp_path / ".prepare-abc", tmp_path / "assets" / "clips" / ".prepare-def"):
        stale.mkdir(parents=True)
        (stale / "clip.nv12").write_bytes(b"partial")
    (tmp_path / "assets" / "clips" / "kept.nv12").write_bytes(b"complete")
    SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["assets", "clips", "kept.nv12"]


def test_apply_logs_timed_phases_and_ready(runtime, monkeypatch, caplog):
    import logging
    config = runtime.media_dir / 'mixer.demo.json'
    monkeypatch.setattr(prepare_demo, 'prepare', lambda *_: config.write_text('{}'))
    with caplog.at_level(logging.INFO, logger='setup'):
        runtime.apply(DEFAULT_SETTINGS)
        runtime.worker.join(3)
    messages = [r.getMessage() for r in caplog.records if r.name == 'setup']
    assert messages[0] == 'Preparing assets'
    assert messages[1].startswith('Assets ready in ') and messages[-1].startswith('Mixer ready: setup applied in ')


def test_rest_timeout_names_the_request(monkeypatch):
    from pyplumber.mixer import dmabuf_inputs
    def urlopen(request, timeout):
        assert timeout == 7
        raise TimeoutError('timed out')
    monkeypatch.setattr(dmabuf_inputs.urllib.request, 'urlopen', urlopen)
    with pytest.raises(TimeoutError, match='POST /workers/recover: no answer within 7 s'):
        dmabuf_inputs.rest_request('http://browser', 'POST', '/workers/recover', {}, timeout=7)


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
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
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
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
    assert not manager._recover_browsers([{'sources': [{'id': 'browser', 'kind': 'browser'}]}])
    assert calls == ['/status']


@pytest.mark.parametrize("bit_depth", [8, 10])
def test_dsk_pages_are_browser_sources_with_clean_copies_of_each_output(tmp_path, bit_depth):
    settings = {**DEFAULT_SETTINGS, "bit_depth": bit_depth, "chroma": "420" if bit_depth == 8 else "422",
                "weights": [4, 0, 0, 0, 1, 0, 0], "dsk": ["lower_third", "bug_left", "bug_right"], "clean_feed": True}
    show, _, _ = prepare_demo.plan(recipe_for(T4, settings), tmp_path)
    from pyplumber.mixer.config import parse
    cfg = parse(show)
    keys = {k.id: k for k in cfg.dsk_keys}
    assert list(keys) == ["lower_third", "bug_left", "bug_right"]
    for key in keys.values():
        source = cfg.source(key.source)
        # The window is the graphic's own size (an allowlisted browser size), never a
        # transparent full canvas; a 1080-wide canvas maps it 1:1.
        assert source.kind == "browser" and (source.width, source.height) == prepare_demo.DSK_WINDOWS[key.id]
        assert (key.dst.w, key.dst.h) == (source.width, source.height)
        assert key.dst.x + key.dst.w <= cfg.canvas_w and key.dst.y + key.dst.h <= cfg.canvas_h
        assert key.on is False
    # Generated scenes do not scatter the key pages into grids.
    assert not any(item.source.startswith("dsk_") for scene in cfg.scenes for item in scene.items)
    feeds = [(r.id, r.feed, r.port) for r in cfg.renditions]
    # Clean is SDR only, to spare the encoder: keyed SDR (+ HDR) plus one clean H.264.
    if bit_depth == 8:
        assert feeds == [("sdr", "dirty", 5004), ("sdr_clean", "clean", 5010)]
    else:
        assert feeds == [("sdr", "dirty", 5004), ("hdr", "dirty", 5006), ("sdr_clean", "clean", 5010)]
    assert cfg.renditions[-1].codec == "h264_nvenc"


def test_dsk_pages_take_their_share_of_the_source_budget():
    # 60 fps, 10-bit 4:2:2; SDR 4:2:2 v210 (no budget of its own) fills what the capped types leave
    limit = source_limit(T4, 60, DEFAULT_SETTINGS["bit_depth"], DEFAULT_SETTINGS["chroma"])
    keys = {"dsk": ["lower_third", "bug_left", "bug_right"], "weights": [8, 4, 2, 8, 2, 0, 0]}
    assert recipe_for(T4, {**DEFAULT_SETTINGS, "source_count": limit - 3, **keys})["source_count"] == limit - 3
    assert recipe_for(T4, {**DEFAULT_SETTINGS, "source_count": limit - 2, **keys})["source_count"] == limit - 3
    assert source_counts(T4, 45, [1, 0, 0, 0, 10], 25, reserved_browsers=3)[4] == 37   # 40 browsers at 25 fps minus 3 keys


@pytest.mark.parametrize("changes", [{"dsk": ["nope"]}, {"dsk": ["ticker", "ticker"]}, {"dsk": "ticker"},
                                     {"clean_feed": True}, {"dsk": ["ticker"], "clean_feed": 1}])
def test_dsk_settings_are_bounded(changes):
    with pytest.raises(ValueError):
        recipe_for(T4, {**DEFAULT_SETTINGS, **changes})


@pytest.mark.parametrize('fps, mode, expected', [
    (30, (8, "420"), 110), (30, (10, "420"), 90), (30, (10, "422"), 81),
    (50, (8, "420"), 82), (50, (10, "420"), 73), (50, (10, "422"), 66),
    (60, (8, "420"), 75), (60, (10, "420"), 61), (60, (10, "422"), 55)])
def test_a_show_above_the_modes_capacity_is_scaled_down(tmp_path, fps, mode, expected):
    """Switching 110 SDR inputs at 30 fps to a 10-bit canvas keeps the show within that canvas's capacity."""
    bit_depth, chroma = mode
    weights = [36, 0, 0, 0, 36, 34, 0] if bit_depth == 8 else [18, 18, 0, 0, 36, 17, 17]
    recipe = recipe_for(T4, {**DEFAULT_SETTINGS, "fps": fps, "bit_depth": bit_depth, "chroma": chroma,
                         "source_count": 110, "weights": weights})
    assert source_limit(T4, fps, bit_depth, chroma) == expected
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    assert len(show["sources"]) == expected


@pytest.mark.parametrize("instance_type", list(InstanceType))
def test_every_profile_covers_every_rate_and_canvas(tmp_path, instance_type):
    """A new instance type fails here, not in a live Apply."""
    profile = INSTANCE_PROFILES[instance_type]
    assert set(profile["mode_share"]) == {"8:420", "10:420", "10:422"}
    for fps in (25, 30, 50, 60):
        recipe_for(profile, {**DEFAULT_SETTINGS, "fps": fps})
    # setup.html indexes the served tables by frame rate, which JSON turns into string keys.
    status = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), instance_type).status()
    assert status["instance_type"] == instance_type.value
    assert json.loads(json.dumps(status))["profile"]["nvdec_decodes"].keys() == {"25", "30", "50", "60"}
    assert profile["nvenc"].keys() == {"budget_pct", "h264_pct_per_fps", "hevc_cost", "preset"}


def own_aux():
    """The live show's own buses: Program preview and Multiviewer."""
    return [{"id": "mv", "renditions": [{"id": "monitor", "port": 5008}]},
            {"id": "mv2", "layout": {"preset": "source_pages"}, "renditions": [{"id": "monitor", "port": 5012}]}]


# A key page and the clean feed, on four NVDEC sources and a browser page.
KEYED = {**DEFAULT_SETTINGS, "dsk": ["lower_third"], "clean_feed": True, "chroma": "420", "weights": [4, 0, 0, 0, 1, 0, 0]}


@pytest.mark.parametrize("bit_depth, limits", [(8, [9, 6, 7, 4]), (10, [7, 4, 3, 0])])
def test_extra_aux_fill_nvenc_beside_the_program_clean_feed_and_own_buses(tmp_path, bit_depth, limits):
    """The live show's encodes (docs/capacity.md): the H.264 program and clean feed, the HEVC HLG
    program on a 10-bit canvas and two own buses, at the program rate at 25/30 fps, half at 50/60."""
    for fps, limit in zip((25, 30, 50, 60), limits):
        for settings in (KEYED, {**KEYED, "dsk": [], "clean_feed": False}):   # the clean feed counts even while off
            recipe = recipe_for(T4, {**settings, "fps": fps, "bit_depth": bit_depth})
            show, _, _ = prepare_demo.plan(recipe, tmp_path)
            assert extra_aux_limit(T4, parse({**show, "aux_buses": own_aux()})) == limit, (fps, settings["clean_feed"])


def test_extra_aux_buses_follow_the_own_ones_and_keep_their_live_layouts(runtime, monkeypatch):
    runtime.janus_api = "http://127.0.0.1:8088/janus"
    syncs = []
    monkeypatch.setattr("setup_runtime.janus_mountpoints.sync", lambda api, ports, prune=False: syncs.append((ports, prune)))
    config = runtime.media_dir / "mixer.demo.json"
    monkeypatch.setattr(prepare_demo, "prepare", lambda recipe, directory: config.write_text(
        json.dumps(prepare_demo.plan(recipe, directory)[0])))
    settings = {**KEYED, "fps": 30, "bit_depth": 8}
    show, _, _ = prepare_demo.plan(recipe_for(T4, settings), runtime.media_dir)
    config.write_text(json.dumps({**show, "aux_buses": own_aux()}))
    live = {}
    runtime.bridge.command = lambda command: json.dumps(list(live.values()))

    def apply(count, **changes):
        runtime.apply({**settings, **changes, "extra_aux": count})
        runtime.worker.join(3)
        assert runtime.status()["phase"] == "running", runtime.status()
        return parse(json.loads(config.read_text()))

    cfg = apply(3)
    ports = {"aux1": 5016, "aux2": 5020, "aux3": 5024}
    # Each is an output of the control page, its mountpoint created before the mixer starts.
    assert [(o["bus"], o["label"], o["mountpoint"]) for o in cfg.settings()["preview_outputs"]][-3:] == [
        ("aux1", "Aux 1", 5016), ("aux2", "Aux 2", 5020), ("aux3", "Aux 3", 5024)]
    assert syncs == [(ports, False), (ports, True)]
    assert runtime.status()["aux_buses"] == [{"id": "mv", "full_rate": False}, {"id": "mv2", "full_rate": False}]
    first = cfg.aux_buses
    for bus in first[2:]:
        assert len(bus.layouts) == 5 and bus.layout == bus.layouts[0] and not draws_program(cfg, bus.layouts)
        assert bus.renditions[0].bitrate_kbps == first[0].renditions[0].bitrate_kbps
        assert bus.renditions[0].preset == T4["nvenc"]["preset"]   # not the renditions' p7 default
    # Fewer drops the last ones; the operator's pick on aux2 stays.
    live["aux2"] = {"id": "aux2", "layout": first[3].layouts[2], "layouts": list(first[3].layouts), "scenes": []}
    cfg = apply(2)
    assert [b.id for b in cfg.aux_buses] == ["mv", "mv2", "aux1", "aux2"]
    assert cfg.aux_buses[3].layout == first[3].layouts[2]
    assert syncs[-1] == ({"aux1": 5016, "aux2": 5020}, True)
    # More adds them back: the same ids, ports and layouts.
    cfg = apply(4)
    assert [(b.id, b.label, b.renditions[0].port) for b in cfg.aux_buses[4:]] == [("aux3", "Aux 3", 5024), ("aux4", "Aux 4", 5028)]
    assert cfg.aux_buses[4].layouts == first[4].layouts
    # Another orientation draws them again, on the same outputs.
    cfg = apply(4, orientation="landscape")
    assert [b.renditions[0].port for b in cfg.aux_buses[2:]] == [5016, 5020, 5024, 5028]
    assert cfg.aux_buses[2].layouts != first[2].layouts and (cfg.canvas_w, cfg.canvas_h) == (1920, 1080)
    # None removes their mountpoints; with none before or after, Janus is not asked.
    cfg = apply(0)
    assert [b.id for b in cfg.aux_buses] == ["mv", "mv2"] and syncs[-1] == ({}, True)
    calls = len(syncs)
    apply(0)
    assert len(syncs) == calls
    with pytest.raises(ValueError, match="At most 6 extra aux outputs fit this setup's NVENC budget on tesla_t4"):
        runtime.apply({**settings, "extra_aux": 7})
    runtime.janus_api = None
    with pytest.raises(ValueError, match="--janus-api"):
        runtime.apply({**settings, "extra_aux": 1})


def test_a_janus_failure_leaves_the_mixer_running_and_shows_in_the_status(runtime, monkeypatch):
    runtime.janus_api = "http://127.0.0.1:8088/janus"
    def refuse(*_):
        raise OSError("Connection refused")
    monkeypatch.setattr("setup_runtime.janus_mountpoints.sync", refuse)
    monkeypatch.setattr(prepare_demo, "prepare", lambda recipe, directory: (directory / "mixer.demo.json").write_text(
        json.dumps(prepare_demo.plan(recipe, directory)[0])))
    runtime.apply({**KEYED, "fps": 30, "bit_depth": 8, "extra_aux": 1})
    runtime.worker.join(3)
    status = runtime.status()
    assert (status["phase"], status["message"]) == ("running", "Mixer ready. Extra aux mountpoints failed: Connection refused.")
