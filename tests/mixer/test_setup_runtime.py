"""Setup changes validate before touching the live process and recover on failure."""
import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from demos.mixer.instance_profiles import INSTANCE_PROFILES, InstanceType
import demos.mixer.prepare_demo as prepare_demo
from pyplumber.mixer.aux_layout import draws_program
from pyplumber.mixer.config import MAX_SOURCES, parse
from pyplumber.mixer.gui.setup_runtime import ( FAILED_START_STOP_TIMEOUT_SEC, HEALTHY_RUN_SEC, RECOVER_TIMEOUT_SEC, RETRY_DELAYS_SEC, STOP_TIMEOUT_SEC,
                           )
from demos.mixer.setup_runtime import DEFAULT_SETTINGS, SetupRuntime, current_settings, extra_aux_limit, recipe_for, source_counts, source_limit
from pyplumber.mixer.gui.web import serve

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
    {'weights': [0] * 5}, {'weights': [True] * 5}, {'resolution': '../../file'}, {'command': 'id'},
    {'fps': 25.0}, {'bit_depth': 10.0}])
def test_reject_unbounded_settings(changes):
    with pytest.raises(ValueError):
        recipe_for(T4, {**DEFAULT_SETTINGS, **changes})


def prepare_without_media(recipe, directory, **kwargs):
    (directory / "mixer.demo.json").write_text(json.dumps(prepare_demo.plan(recipe, directory)[0]))


def apply_and_wait(runtime, settings=None):
    runtime.apply(settings)
    runtime.worker.join(3)
    assert not runtime.worker.is_alive(), "setup worker did not finish"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    manager = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
    manager.process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(manager, '_stop', lambda **kwargs: None)
    monkeypatch.setattr(manager, '_close_removed_browsers', lambda *_: None)
    monkeypatch.setattr(manager, '_recover_browsers', lambda *_: False)
    monkeypatch.setattr(manager, '_start', lambda _: setattr(manager, 'process', SimpleNamespace(poll=lambda: None)))
    monkeypatch.setattr(manager, '_watch', lambda _: None)
    monkeypatch.setattr(manager, '_validate_capacity', lambda *args: None)
    monkeypatch.setattr(prepare_demo, "prepare", prepare_without_media)
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
    monkeypatch.setattr('pyplumber.mixer.gui.setup_runtime.threading.Timer', lambda *a: created.append(FakeTimer(*a)) or created[-1])
    return created


def crash(runtime, code=-11):
    """The running mixer exits on its own; the real watcher sees it."""
    process = SimpleNamespace(wait=lambda: code, poll=lambda: code)
    runtime.process = process
    SetupRuntime._watch(runtime, process)


def test_prepare_failure_keeps_old_show_and_process(runtime, monkeypatch):
    config = runtime.media_dir / 'mixer.demo.json'
    config.write_text('{"old": true}')
    def fail(*args, **kwargs):
        raise ValueError('disk full')
    monkeypatch.setattr(prepare_demo, 'prepare', fail)
    apply_and_wait(runtime, DEFAULT_SETTINGS)
    assert runtime.status()['phase'] == 'error'
    assert 'disk full' in runtime.status()['message']
    assert json.loads(config.read_text()) == {'old': True}
    assert runtime.process.poll() is None


def test_failed_start_restores_previous_show(runtime, monkeypatch):
    config = runtime.media_dir / 'mixer.demo.json'
    config.write_text('{"old": true}')
    monkeypatch.setattr(prepare_demo, 'prepare', lambda *_, **kwargs: config.write_text('{"new": true}'))
    starts = []
    def start(path):
        doc = json.loads(path.read_text())
        starts.append(doc)
        if 'new' in doc:
            raise RuntimeError('GPU out of memory')
    monkeypatch.setattr(runtime, '_start', start)
    apply_and_wait(runtime, DEFAULT_SETTINGS)
    assert starts == [{'new': True}, {'old': True}]
    assert 'previous setup restored' in runtime.status()['message']
    assert not runtime.recipe_path.exists()


def test_single_job_and_persist_only_after_ready(runtime, monkeypatch):
    revision = runtime.revision
    entered, release = threading.Event(), threading.Event()
    def prepare(*_, **kwargs):
        kwargs["progress"](2, 8)
        entered.set()
        assert release.wait(3)
        (runtime.media_dir / 'mixer.demo.json').write_text('{}')
    monkeypatch.setattr(prepare_demo, 'prepare', prepare)
    runtime.apply(DEFAULT_SETTINGS)
    assert entered.wait(3)
    try:
        assert runtime.status()["message"] == "Preparing assets… 2/8 ready."
        assert not runtime.recipe_path.exists()
        with pytest.raises(RuntimeError, match='already in progress'):
            runtime.apply(DEFAULT_SETTINGS)
    finally:
        release.set()
        runtime.worker.join(3)
    assert runtime.status()['phase'] == 'running'
    saved = current_settings(T4, DEFAULT_SETTINGS)   # every program encode filled in
    assert runtime.status()['settings'] == saved
    assert runtime.status()['revision'] == revision + 1
    assert json.loads(runtime.recipe_path.read_text())['setup'] == saved


def test_api_rejects_cross_origin_and_oversized_requests(runtime):
    server = serve(SimpleNamespace(), '127.0.0.1', 0, setup=runtime)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}/api/setup'
    try:
        for headers, data, code in [({'Origin': 'http://elsewhere.invalid'}, b'{}', 403),
                                    ({}, b' ' * 4097, 400), ({}, b'{}', 400),
                                    ({}, b'null', 400), ({}, b'[]', 400), ({}, b'"resume"', 400)]:
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


def encode(preset="p3", kbps=3000):
    return {"preset": preset, "bitrate_kbps": kbps}


def test_every_program_encode_takes_its_own_preset_and_bitrate(tmp_path):
    """By default the profile's: the recipe's bitrates at p3, the clean copy at the SDR program's."""
    def encodes(settings):
        show, _, _ = prepare_demo.plan(recipe_for(T4, {**KEYED, "bit_depth": 10, **settings}), tmp_path)
        return [(r["id"], r["preset"], r["bitrate_kbps"]) for r in show["renditions"]]
    assert encodes({}) == [("sdr", "p3", 6000), ("hdr", "p3", 8000), ("sdr_clean", "p3", 6000)]
    recipe = json.loads((prepare_demo.DEMO_DIR / "demo.example.json").read_text())
    assert [T4["nvenc"]["defaults"][r["id"]]["bitrate_kbps"] for r in recipe["renditions"]] == [r["bitrate_kbps"] for r in recipe["renditions"]]
    assert encodes({"encodes": {"sdr": encode("p1", 2000), "hdr": encode("p5", 20000), "sdr_clean": encode("p5", 4500)}}) == [
        ("sdr", "p1", 2000), ("hdr", "p5", 20000), ("sdr_clean", "p5", 4500)]


@pytest.mark.parametrize("legacy, kbps", [(6000, [6000, 8000, 6000]), (3000, [3000, 4000, 3000]),
                                        (10**1000, [20000, 20000, 20000]), (-10**1000, [250, 250, 250]),
                                          (16000, [16000, 20000, 16000]), (500, [500, 667, 500])])
def test_settings_saved_with_one_program_bitrate_load_per_output(legacy, kbps):
    """The SDR program and its clean copy take it, the HLG one the defaults' ratio, within the range."""
    older = {**{k: v for k, v in DEFAULT_SETTINGS.items() if k != "encodes"}, "bitrate_kbps": legacy}
    encodes = recipe_for(T4, older)["setup"]["encodes"]
    assert [encodes[o] for o in ("sdr", "hdr", "sdr_clean")] == [{**encode("p3", n), "codec": codec} for n, codec in zip(kbps, ("h264_nvenc", "hevc_nvenc", "h264_nvenc"))]
    assert encodes["extra"] == {**encode("p1", 4000), "codec": "h264_nvenc"}


@pytest.mark.parametrize("fps, size", [(25, 6), (30, 6), (50, 9), (60, 9)])
def test_browser_ring_default_tracks_fps(tmp_path, fps, size):
    # Settings saved with a ring size load; the setup leaves it to the frame rate's default.
    recipe = recipe_for(T4, {**DEFAULT_SETTINGS, "fps": fps, "browser_ring_size": 11})
    assert "browser_ring_size" not in recipe
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    from pyplumber.mixer.config import parse
    from pyplumber.mixer.cli import GraphOptions
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


def test_a_setup_saved_with_one_program_bitrate_resumes_with_per_output_encodes(runtime, monkeypatch):
    """The page loads the status's settings: per output, from the old bitrate or the defaults."""
    older = {k: v for k, v in DEFAULT_SETTINGS.items() if k not in ("encodes", "extra_aux")}
    for saved, kbps in (({**older, "bitrate_kbps": 3000}, [3000, 4000, 3000]), (older, [6000, 8000, 6000])):
        runtime.recipe_path.write_text(json.dumps({**recipe_for(T4, DEFAULT_SETTINGS), "setup": saved}))
        apply_and_wait(runtime)
        encodes = runtime.status()["settings"]["encodes"]
        assert [encodes[o] for o in ("sdr", "hdr", "sdr_clean", "extra")] == [{**e, "codec": "hevc_nvenc" if i == 1 else "h264_nvenc"}
            for i, e in enumerate([*map(encode, ["p3"] * 3, kbps), encode("p1", 4000)])]


@pytest.mark.parametrize("encodes, message", [
    *(({"sdr": encode("p3", kbps)}, "encodes.sdr: bitrate_kbps must be an integer from 250 to 20000")
      for kbps in (249, 20001, 0, 6000.0, "6000", True)),
    ({"mv": encode("p3", 249)}, "encodes.mv: bitrate_kbps"),
    *(({"extra": value}, "encodes.extra must have a preset of p1, p3, p5 and a bitrate_kbps")
      for value in (encode("p7"), encode("p4"), {"preset": "p3"}, {**encode(), "tune": "ll"})),
    ({"extra": 3000}, "encodes.extra: codec must be"),
    ([], "encodes must map"), ({f"aux{i}": encode() for i in range(65)}, "encodes must map")])
def test_encodes_outside_the_profile_are_rejected(encodes, message):
    with pytest.raises(ValueError, match=message):
        recipe_for(T4, {**DEFAULT_SETTINGS, "encodes": encodes})


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
        recipe_for(T4, {**settings, "scene_count": 257})


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


@pytest.mark.parametrize("weights", [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0]])
def test_sdr_and_hdr_v210_share_upload_limit(weights):
    assert sum(source_counts(T4, 4, weights)[2:4]) == 4
    with pytest.raises(ValueError, match="4:2:2 upload is limited to 4"):
        recipe_for(T4, {**DEFAULT_SETTINGS, "source_count": 5, "weights": weights})
    counts = source_counts(T4, 64, [1, 1, weights[2], weights[3], 100, 0, 0])
    assert sum(counts) == 64 and sum(counts[2:4]) <= 4 and counts[4] <= 40


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
    monkeypatch.setattr(prepare_demo, 'prepare', lambda *_, **kwargs: config.write_text('{"sources": []}'))
    def fail(*_):
        raise RuntimeError('encoder unavailable')
    monkeypatch.setattr(runtime, '_start', fail)
    apply_and_wait(runtime, DEFAULT_SETTINGS)
    assert 'encoder unavailable' in runtime.status()['message']
    assert not config.exists()
    assert not runtime.recipe_path.exists()


@pytest.mark.parametrize('previous', [None, b'{"old": true}'])
def test_shutdown_during_prepare_restores_committed_show(runtime, monkeypatch, previous):
    config = runtime.media_dir / 'mixer.demo.json'
    if previous is not None:
        config.write_bytes(previous)
    def prepare(*_, **kwargs):
        config.write_text('{"new": true}')
        runtime.closing.set()
    monkeypatch.setattr(prepare_demo, 'prepare', prepare)
    monkeypatch.setattr(runtime, '_start', lambda _: pytest.fail('must not start while closing'))
    apply_and_wait(runtime, DEFAULT_SETTINGS)
    assert (config.read_bytes() if config.exists() else None) == previous
    assert not runtime.recipe_path.exists()


def test_restart_changes_revision(tmp_path, monkeypatch):
    monkeypatch.setattr('pyplumber.mixer.gui.setup_runtime.time.time_ns', lambda: 1_000_000_000_000)
    first = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), InstanceType.TESLA_T4)
    monkeypatch.setattr('pyplumber.mixer.gui.setup_runtime.time.time_ns', lambda: 2_000_000_000_000)
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
    apply_and_wait(runtime, settings(after))
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


def test_setup_drops_saved_aux_layouts_the_new_sources_no_longer_have(runtime, monkeypatch):
    """Fewer sources (e.g. 110 at 30 fps, 75 at 60) must not refuse an Apply over a cells layout."""
    settings = {**DEFAULT_SETTINGS, "chroma": "420", "weights": [4, 0, 0, 0, 1, 0, 0]}
    config = runtime.media_dir / "mixer.demo.json"
    show, _, _ = prepare_demo.plan(recipe_for(T4, settings), runtime.media_dir)
    show["aux_buses"] = [{"id": "mv", "renditions": [{"id": "monitor", "port": 5008}]}]
    config.write_text(json.dumps(show))
    cell = lambda source: {"role": "source", "source": source, "x": 0, "y": 0, "w": 960, "h": 540}
    gone, kept = {"cells": [cell(99)]}, {"cells": [cell(0)]}
    runtime.bridge.command = lambda cmd: json.dumps([{"id": "mv", "layout": gone, "layouts": [gone, kept], "scenes": []}])
    apply_and_wait(runtime, settings)
    assert runtime.status()["phase"] == "running", runtime.status()
    bus = json.loads(config.read_text())["aux_buses"][0]
    assert bus["layouts"] == [kept] and bus["layout"] == kept


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
    # The setup sets every bus's encode: one it omits takes the profile's aux default.
    assert (r.width, r.height, r.fps, r.preset, r.bitrate_kbps) == (1920, 1080, 25, "p1", 4000)
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
    monkeypatch.setattr(prepare_demo, "prepare", lambda *_, **kwargs: pytest.fail("must validate first"))
    with pytest.raises(ValueError, match="port"):
        runtime.apply(DEFAULT_SETTINGS)
    assert runtime.worker is None


@pytest.mark.parametrize("hdr", [False, True])
@pytest.mark.parametrize("sdr_codec", ["h264", "h265"])
def test_start_waits_for_program_and_aux_encoders(runtime, monkeypatch, hdr, sdr_codec):
    monkeypatch.setattr("pyplumber.mixer.gui.setup_runtime.subprocess.Popen", lambda *a, **kw: SimpleNamespace(pid=1234, poll=lambda: None))
    monkeypatch.setattr(runtime.closing, "wait", lambda _: False)
    runtime.bridge.state = lambda **kw: {"status": {"pgm_scene": "full"},
        "settings": {"preview_codecs": [sdr_codec, "h265"] if hdr else [sdr_codec], "aux_buses": ["mv"],
                     "program_outputs": [{"rendition": "sdr"}, *([{"rendition": "hdr"}] if hdr else [])]}}
    expected = ["janus_encoded", "aux_mv_encoded", *(["janus_hdr_encoded"] if hdr else [])]
    config = runtime.media_dir / "mixer.demo.json"
    config.write_text(json.dumps({"sources": [{"id": "visible"}, {"id": "unused"}]}))
    calls = []
    def queues(command, **kw):
        if command == "nodes.json":
            # An unused source can wait for consumers; readiness must not require its frames.
            return json.dumps([{"name": "mixer_color_" + name, "type": "filter_video", "working": True}
                               for name in ("visible", "unused")])
        calls.append(command)
        assert len(calls) <= len(expected)
        return json.dumps([{"name": name, "enqueued_total": int(i < len(calls))}
                           for i, name in enumerate(expected)])
    runtime.bridge.command = queues
    SetupRuntime._start(runtime, config)
    assert len(calls) == len(expected)


@pytest.mark.parametrize("nodes", [[], [{"name": "mixer_color_hdr", "type": "filter_video", "working": False}]])
def test_start_rejects_missing_or_failed_source_normalization(runtime, monkeypatch, nodes):
    monkeypatch.setattr("pyplumber.mixer.gui.setup_runtime.subprocess.Popen", lambda *a, **kw: SimpleNamespace(pid=1234, poll=lambda: None))
    monkeypatch.setattr(runtime.closing, "wait", lambda _: False)
    config = runtime.media_dir / "mixer.demo.json"
    config.write_text(json.dumps({"sources": [{"id": "hdr"}]}))
    runtime.bridge.state = lambda **kw: {"status": {"pgm_scene": "full"}}
    runtime.bridge.command = lambda command, **kw: json.dumps(nodes if command == "nodes.json" else
        [{"name": "janus_encoded", "enqueued_total": 100}])
    with pytest.raises(RuntimeError, match="Source normalization not running: mixer_color_hdr"):
        SetupRuntime._start(runtime, config)


def test_start_reports_child_failure_while_native_cleanup_is_still_running(runtime, monkeypatch):
    import os
    from unittest.mock import Mock
    config = runtime.media_dir / "mixer.demo.json"
    config.write_text('{"sources": []}')
    def launch(*args, pass_fds, env, **kwargs):
        assert int(env["AVP_MIXER_STARTUP_FD"]) == pass_fds[0]
        os.write(pass_fds[0], b"aux_test_encoder: invalid preset")
        return SimpleNamespace(pid=1234, poll=lambda: None)
    monkeypatch.setattr("pyplumber.mixer.gui.setup_runtime.subprocess.Popen", launch)
    monkeypatch.setattr(runtime.closing, "wait", lambda _: False)
    monkeypatch.setattr(runtime, "_start", lambda path: SetupRuntime._start(runtime, path))
    stop = Mock()
    monkeypatch.setattr(runtime, "_stop", stop)
    runtime.bridge.state = lambda **kwargs: pytest.fail("startup failure precedes control readiness")
    with pytest.raises(RuntimeError, match="Mixer startup failed: aux_test_encoder: invalid preset"):
        runtime._start_recovering(config, None)
    stop.assert_called_once_with(timeout=FAILED_START_STOP_TIMEOUT_SEC)


def test_start_waits_for_child_pipe_before_blocking_on_control_greeting(runtime, monkeypatch):
    import os
    config = runtime.media_dir / "mixer.demo.json"
    config.write_text('{"sources": []}')
    writer = None
    def launch(*args, pass_fds, **kwargs):
        nonlocal writer
        writer = os.dup(pass_fds[0])
        return SimpleNamespace(pid=1234, poll=lambda: None)
    monkeypatch.setattr("pyplumber.mixer.gui.setup_runtime.subprocess.Popen", launch)
    waits = 0
    def wait(_):
        nonlocal waits
        waits += 1
        assert waits <= 2
        if waits == 2:
            os.write(writer, b"encoder initialization failed")
            os.close(writer)
        return False
    monkeypatch.setattr(runtime.closing, "wait", wait)
    runtime.bridge.state = lambda **kwargs: pytest.fail("control greeting waits for successful startup")
    with pytest.raises(RuntimeError, match="Mixer startup failed: encoder initialization failed"):
        SetupRuntime._start(runtime, config)
    assert waits == 2


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


def test_256_scenes_expand(tmp_path):
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    recipe = recipe_for(profile, {**DEFAULT_SETTINGS, "scene_count": 256})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    assert len(show["scenes"]) == 256
    with pytest.raises(ValueError, match="scene_count must be an integer from 1 to 256"):
        recipe_for(profile, {**DEFAULT_SETTINGS, "scene_count": 257})


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
    monkeypatch.setattr('pyplumber.mixer.gui.setup_runtime.os.killpg', lambda pid, sig: events.append(('signal', sig)))
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
    monkeypatch.setattr('pyplumber.mixer.gui.setup_runtime.os.killpg', lambda pid, sig: (events.append(sig), exited.set()))
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
    monkeypatch.setattr('pyplumber.mixer.gui.setup_runtime.os.killpg', lambda pid, sig: events.append(sig))
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
    monkeypatch.setattr(prepare_demo, 'prepare', lambda *_, **kwargs: config.write_text('{"new": true}'))
    recoveries = []
    def recover(shows):
        recoveries.append(shows)
        raise TimeoutError('Browser POST /workers/recover: no answer within 180 s')
    monkeypatch.setattr(runtime, '_recover_browsers', recover)
    monkeypatch.setattr(runtime, '_start', lambda _: pytest.fail('must not start while workers restart'))
    apply_and_wait(runtime, DEFAULT_SETTINGS)
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
    monkeypatch.setattr('pyplumber.mixer.gui.setup_runtime.time.monotonic', lambda: next(clock))
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
    monkeypatch.setattr(prepare_demo, 'prepare', lambda *_, **kwargs: (runtime.media_dir / 'mixer.demo.json').write_text('{}'))
    apply_and_wait(runtime, DEFAULT_SETTINGS)
    assert pending.cancelled and runtime.retries == 0
    applied = runtime.worker
    pending.function()   # fired just before it was cancelled
    assert runtime.worker is applied and runtime.status()['message'] == 'Mixer ready.'


def test_api_answers_409_to_a_concurrent_apply_without_asking_the_mixer(runtime, monkeypatch, http_server):
    runtime.worker = SimpleNamespace(is_alive=lambda: True)   # the first Apply is stopping the mixer
    monkeypatch.setattr(runtime, '_preserve_aux', lambda *_: pytest.fail('must not query the live mixer'))
    server = http_server(SimpleNamespace(), setup=runtime)
    request = Request(f'http://127.0.0.1:{server.server_port}/api/setup', data=json.dumps(DEFAULT_SETTINGS).encode(),
                      headers={'Content-Type': 'application/json'})
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=3)
    assert error.value.code == 409




def test_a_torn_write_never_reaches_the_show(tmp_path, monkeypatch):
    from pathlib import Path
    from demos.mixer.demo_recipe import write_atomic
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
    runtime.process = None   # a container restart: nothing is running
    apply_and_wait(runtime)
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
    monkeypatch.setattr(prepare_demo, 'prepare', lambda *_, **kwargs: config.write_text('{}'))
    with caplog.at_level(logging.INFO, logger='setup'):
        apply_and_wait(runtime, DEFAULT_SETTINGS)
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
    monkeypatch.setattr(runtime, '_stop', lambda **kwargs: events.append('reaped'))
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
    monkeypatch.setattr('pyplumber.mixer.gui.setup_runtime.os.killpg', lambda *_: None)
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
    # Keys consume browser slots; SDR and HDR v210 share the same upload budget.
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
        assert source_limit(profile, fps) < MAX_SOURCES   # room for a bus's PGM pad in the pad mask
    # setup.html indexes the served tables by frame rate, which JSON turns into string keys.
    status = SetupRuntime(tmp_path, tmp_path / 'demo.json', SimpleNamespace(port=7777), instance_type).status()
    assert status["instance_type"] == instance_type.value
    assert json.loads(json.dumps(status))["profile"]["nvdec_decodes"].keys() == {"25", "30", "50", "60"}
    nvenc = profile["nvenc"]
    assert nvenc.keys() == {"budget_pct", "pct_per_fps", "bitrate_kbps", "defaults"} | (
        {"max_outputs"} if instance_type in (InstanceType.NVIDIA_L4, InstanceType.NVIDIA_L4_CUARRAY) else set())
    assert nvenc["defaults"].keys() == {"sdr", "hdr", "sdr_clean", "aux"}
    assert nvenc["pct_per_fps"]["hevc"].keys() == nvenc["pct_per_fps"]["h264"].keys() >= {e["preset"] for e in nvenc["defaults"].values()}


@pytest.mark.parametrize("fps", [25, 30])
def test_l4_setup_takes_its_limit_above_128_sources(tmp_path, fps):
    """The L4's SDR limits pass the former 128-pad mask: every source and the program preview's
    PGM pad fit, so its maximum builds a show."""
    l4 = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    maximum = source_limit(l4, fps)
    assert maximum > 128
    recipe = recipe_for(l4, {**DEFAULT_SETTINGS, "fps": fps, "bit_depth": 8, "chroma": "420",
                             "source_count": maximum, "weights": [1, 0, 0, 0, 1, 1, 0]})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    assert len(show["sources"]) == maximum
    cfg = parse({**show, "aux_buses": own_aux()})
    assert draws_program(cfg, cfg.aux_buses[0].layouts)


@pytest.mark.parametrize('fps, mode, expected', [
    (30, (8, "420"), {"video": 83, "browser": 40, "nv12": 47}), (60, (8, "420"), {"video": 40, "browser": 40, "nv12": 19}),
    (30, (10, "420"), {"video": 96, "browser": 40}), (60, (10, "420"), {"video": 48, "browser": 40}),
    (30, (10, "422"), {"video": 88, "browser": 40, "v210": 4}), (60, (10, "422"), {"video": 44, "browser": 40, "v210": 4})])
def test_l4_largest_show_is_the_measured_mix(tmp_path, fps, mode, expected):
    """A canvas's own limits (mode_limits) bound the L4's HLG shows: NVDEC and the browser windows
    full, no raw 4:2:0 upload beside them, the v210 inputs counted only on a 4:2:2 canvas."""
    l4, (bit_depth, chroma) = INSTANCE_PROFILES[InstanceType.NVIDIA_L4], mode
    weights = [36, 0, 0, 0, 36, 34, 0] if bit_depth == 8 else [18, 18, 4 * (chroma == "422"), 0, 36, 17, 17]
    recipe = recipe_for(l4, {**DEFAULT_SETTINGS, "fps": fps, "bit_depth": bit_depth, "chroma": chroma,
                             "source_count": 191, "weights": weights})
    assert source_limit(l4, fps, bit_depth, chroma) == sum(expected.values())
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    kinds = [s["kind"] for s in show["sources"]]
    assert {kind: kinds.count(kind) for kind in set(kinds)} == expected


def own_aux(**rendition):
    """The live show's own buses: Program preview and Multiviewer."""
    return [{"id": "mv", "renditions": [{"id": "monitor", "port": 5008, **rendition}]},
            {"id": "mv2", "layout": {"preset": "source_pages"}, "renditions": [{"id": "monitor", "port": 5012, **rendition}]}]


# A key page and the clean feed, on four NVDEC sources and a browser page.
KEYED = {**DEFAULT_SETTINGS, "dsk": ["lower_third"], "clean_feed": True, "chroma": "420", "weights": [4, 0, 0, 0, 1, 0, 0]}


def nvenc_limits(tmp_path, bit_depth, own_preset="p1", **encodes):
    """extra_aux_limit at 25, 30, 50 and 60 fps beside two own buses at *own_preset*, the reason
    where the encodes exceed the budget; the same with the clean feed on and off."""
    def limit(settings):
        recipe = recipe_for(T4, {**settings, "bit_depth": bit_depth, "encodes": encodes})
        show, _, _ = prepare_demo.plan(recipe, tmp_path)
        try:
            return extra_aux_limit(T4, parse({**show, "aux_buses": own_aux(preset=own_preset)}), recipe["setup"]["encodes"])
        except ValueError as exc:
            return str(exc)
    limits = [limit({**KEYED, "fps": fps}) for fps in (25, 30, 50, 60)]
    assert limits == [limit({**KEYED, "fps": fps, "dsk": [], "clean_feed": False}) for fps in (25, 30, 50, 60)]
    return limits


@pytest.mark.parametrize("bit_depth, limits", [(8, [11, 8, 9, 6]), (10, [10, 7, 6, 3])])
def test_extra_aux_fill_nvenc_beside_the_program_clean_feed_and_own_buses(tmp_path, bit_depth, limits):
    """The live show's encodes at the defaults (docs/capacity.md): the H.264 program and clean feed
    (counted even while off) and the HEVC HLG program on a 10-bit canvas at p3, two own buses and the
    extra ones at p1, at the program rate at 25/30 fps, half at 50/60."""
    assert nvenc_limits(tmp_path, bit_depth) == limits


OVER_BUDGET = "The encodes need 94.7% of NVENC, above its 80% budget: choose faster presets"
# The programs and the clean feed at p5.
P5 = {"sdr": encode("p5", 6000), "hdr": encode("p5", 8000), "sdr_clean": encode("p5", 6000)}


@pytest.mark.parametrize("bit_depth, own_preset, encodes, limits", [
    (8, "p3", {"extra": encode("p3")}, [10, 8, 8, 6]),   # every encode at p3
    (10, "p3", {"extra": encode("p3")}, [9, 6, 5, 3]),
    (8, "p1", P5, [9, 6, 4, 1]),                          # the monitors at p1
    (10, "p1", P5, [6, 4, 0, OVER_BUDGET])])
def test_each_encode_costs_its_preset(tmp_path, bit_depth, own_preset, encodes, limits):
    assert nvenc_limits(tmp_path, bit_depth, own_preset, **encodes) == limits


def test_encodes_above_the_nvenc_budget_are_refused_without_extra_aux(runtime, monkeypatch):
    """No Janus API and no extra aux outputs: the show's own encodes alone exceed the budget. At
    60 fps the programs and the clean feed at p5 do not fit beside two aux buses at p1, nor the SDR
    program and the clean feed at p5 beside the HLG program at p3."""
    monkeypatch.setattr(prepare_demo, "prepare", lambda *_, **kwargs: pytest.fail("must validate first"))
    settings = {**KEYED, "fps": 60, "bit_depth": 10}
    show, _, _ = prepare_demo.plan(recipe_for(T4, settings), runtime.media_dir)
    (runtime.media_dir / "mixer.demo.json").write_text(json.dumps({**show, "aux_buses": own_aux()}))
    runtime.process = None
    for encodes, reason in ((P5, OVER_BUDGET), ({**P5, "hdr": encode("p3", 8000)}, "The encodes need 86.5% of NVENC")):
        with pytest.raises(ValueError, match=reason):
            runtime.apply({**settings, "encodes": encodes})


def test_extra_aux_buses_follow_the_own_ones_and_keep_their_live_layouts(runtime, monkeypatch):
    runtime.janus_api = "http://127.0.0.1:8088/janus"
    syncs = []
    def sync(api, specs, prune=False, replace=False):
        extra = {key.split(":")[0]: value["videoport"] for key, value in specs.items()
                 if value["description"].startswith("avplumber extra aux")}
        assert replace != prune
        syncs.append((extra, prune))
    monkeypatch.setattr("pyplumber.mixer.gui.setup_runtime.janus_mountpoints.sync", sync)
    config = runtime.media_dir / "mixer.demo.json"
    settings = {**KEYED, "fps": 30, "bit_depth": 8}
    show, _, _ = prepare_demo.plan(recipe_for(T4, settings), runtime.media_dir)
    config.write_text(json.dumps({**show, "aux_buses": own_aux()}))
    live = {}
    runtime.bridge.command = lambda command: json.dumps(list(live.values()))

    def apply(count, **changes):
        apply_and_wait(runtime, {**settings, **changes, "extra_aux": count})
        assert runtime.status()["phase"] == "running", runtime.status()
        return parse(json.loads(config.read_text()))

    cfg = apply(3)
    ports = {"aux0": 5016, "aux1": 5020, "aux2": 5024}
    # Each is an output of the control page, its mountpoint created before the mixer starts.
    assert [(o["bus"], o["label"], o["mountpoint"]) for o in cfg.settings()["preview_outputs"]][-3:] == [
        ("aux0", "Aux 0", 5016), ("aux1", "Aux 1", 5020), ("aux2", "Aux 2", 5024)]
    assert syncs == [(ports, False), (ports, True)]
    # Every encode the setup generates names its preset and bitrate: never the renditions' p7 default.
    raw = json.loads(config.read_text())
    encodes = [*raw["renditions"], *(r for bus in raw["aux_buses"] for r in bus["renditions"])]
    assert len(encodes) == 7 and all(r["preset"] in ("p1", "p3", "p5") and "bitrate_kbps" in r for r in encodes)
    # The page names the own buses; like the extra ones, they start at the profile's aux default.
    assert runtime.status()["aux_buses"] == [{"id": "mv", "label": "Program preview", "full_rate": False},
                                             {"id": "mv2", "label": "Multiviewer", "full_rate": False}]
    first = cfg.aux_buses
    for bus in first[2:]:
        assert len(bus.layouts) == 5 and bus.layout == bus.layouts[0] and not draws_program(cfg, bus.layouts)
        assert (bus.renditions[0].preset, bus.renditions[0].bitrate_kbps) == ("p1", 4000)   # the aux default
    # Fewer drops the last ones; the operator's pick on aux1 stays. Every extra bus takes the extra
    # encode, an own bus its own or the aux default.
    live["aux1"] = {"id": "aux1", "layout": first[3].layouts[2], "layouts": list(first[3].layouts), "scenes": []}
    cfg = apply(2, encodes={"extra": encode("p1", 2500), "mv": encode("p5", 4500)})
    assert [b.id for b in cfg.aux_buses] == ["mv", "mv2", "aux0", "aux1"]
    assert cfg.aux_buses[3].layout == first[3].layouts[2]
    assert [(b.renditions[0].preset, b.renditions[0].bitrate_kbps) for b in cfg.aux_buses] == [
        ("p5", 4500), ("p1", 4000), ("p1", 2500), ("p1", 2500)]
    assert syncs[-1] == ({"aux0": 5016, "aux1": 5020}, True)
    # More adds them back: the same ids, ports and layouts.
    cfg = apply(4)
    assert [(b.id, b.label, b.renditions[0].port) for b in cfg.aux_buses[4:]] == [("aux2", "Aux 2", 5024), ("aux3", "Aux 3", 5028)]
    assert cfg.aux_buses[4].layouts == first[4].layouts
    # Another orientation draws them again, on the same outputs.
    cfg = apply(4, orientation="landscape")
    assert [b.renditions[0].port for b in cfg.aux_buses[2:]] == [5016, 5020, 5024, 5028]
    assert cfg.aux_buses[2].layouts != first[2].layouts and (cfg.canvas_w, cfg.canvas_h) == (1920, 1080)
    # None removes extra mountpoints; fixed output contracts are still synchronized.
    cfg = apply(0)
    assert [b.id for b in cfg.aux_buses] == ["mv", "mv2"] and syncs[-1] == ({}, True)
    calls = len(syncs)
    apply(0)
    assert len(syncs) == calls + 2
    assert len(apply(9).aux_buses) == 10
    assert runtime.status()["settings"]["extra_aux"] == 8
    assert json.loads(runtime.recipe_path.read_text())["setup"]["extra_aux"] == 8
    runtime.janus_api = None
    with pytest.raises(ValueError, match="--janus-api"):
        runtime.apply({**settings, "extra_aux": 1})


def test_adopted_setup_metadata_keeps_managed_aux_scalable(tmp_path):
    from pyplumber.mixer.tools.extra_aux import extra_buses
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    encodes = {"sdr": encode("p5", 6000), "mv": encode("p3"), "mv2": encode("p3"), "extra": encode("p3")}
    settings = {**KEYED, "fps": 25, "bit_depth": 8, "extra_aux": 26, "encodes": encodes}
    recipe = recipe_for(profile, settings)
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    show["aux_buses"] = own_aux(preset="p3")
    show["aux_buses"] += extra_buses(parse(show), [], 26, prepare_demo.CLEAN_PORT, encode("p3"))
    show["setup"] = recipe["setup"]
    (tmp_path / "mixer.demo.json").write_text(json.dumps(show))
    manager = SetupRuntime(tmp_path, tmp_path / "recipe.json", None, InstanceType.NVIDIA_L4, janus_api="http://127.0.0.1")
    assert manager.status()["settings"]["extra_aux"] == 26
    assert [b["id"] for b in manager.status()["aux_buses"]] == ["mv", "mv2"]
    assert len(manager.extra_ports) == 26
    recipe = recipe_for(profile, {**settings, "fps": 60, "bit_depth": 10})
    changed, _, _ = prepare_demo.plan(recipe, tmp_path)
    manager._preserve_aux(recipe, changed)
    assert recipe["setup"]["extra_aux"] == 13
    assert [b["id"] for b in recipe["aux_buses"]] == ["mv", "mv2", *(f"aux{i}" for i in range(13))]
    assert all(b["renditions"][0]["fps"] == 30 for b in recipe["aux_buses"])
    assert not manager.recipe_path.exists(), "planning does not alter the running show"
    # Without explicit metadata, named instance outputs must never be silently discarded.
    del show["setup"]
    (tmp_path / "mixer.demo.json").write_text(json.dumps(show))
    manager = SetupRuntime(tmp_path, tmp_path / "recipe.json", None, InstanceType.NVIDIA_L4)
    assert manager.settings is None and len(manager.aux_buses) == 28


def test_l4_extra_aux_limit_reserves_programs_and_own_buses_in_output_cap(tmp_path):
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    recipe = recipe_for(profile, {**KEYED, "fps": 25, "bit_depth": 8})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    assert extra_aux_limit(profile, parse({**show, "aux_buses": own_aux(preset="p1")}), recipe["setup"]["encodes"]) == 22


def test_l4_fixed_outputs_above_memory_capacity_are_rejected(tmp_path):
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    recipe = recipe_for(profile, {**KEYED, "fps": 25, "bit_depth": 8})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    fixed = [{"id": f"fixed{i}", "layout": {"preset": "source_pages"},
              "renditions": [{"id": "monitor", "port": 6000 + i * 4, **encode("p1")}]} for i in range(25)]
    with pytest.raises(ValueError, match="at most 26 encoded outputs, including programs"):
        extra_aux_limit(profile, parse({**show, "aux_buses": fixed}), recipe["setup"]["encodes"])


@pytest.mark.parametrize("fps,expected", [(25, 22), (30, 22), (50, 18), (60, 18)])
def test_sdr_output_count_ceiling_cannot_be_bypassed_by_fast_presets(tmp_path, fps, expected):
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    recipe = recipe_for(profile, {**KEYED, "fps": fps, "bit_depth": 8,
        "encodes": {output: encode("p1") for output in ("sdr", "sdr_clean", "extra")}})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    assert extra_aux_limit(profile, parse({**show, "aux_buses": own_aux(preset="p1")}), recipe["setup"]["encodes"]) == expected


@pytest.mark.parametrize("fps,expected", [(25, 15), (60, 13)])
def test_mode_output_count_limit_overrides_only_its_rates(tmp_path, fps, expected):
    base = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    profile = {**base, "mode_limits": {**base["mode_limits"], "10:420": {
        **base["mode_limits"]["10:420"], "nvenc_max_outputs": {25: 20}}}}
    recipe = recipe_for(profile, {**KEYED, "fps": fps,
        "encodes": {"sdr": encode("p5", 6000), "extra": encode("p3")}})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    assert extra_aux_limit(profile, parse({**show, "aux_buses": own_aux(preset="p3")}), recipe["setup"]["encodes"]) == expected


def test_resume_clamps_saved_outputs_without_regenerating_custom_inputs(runtime, monkeypatch):
    from pyplumber.mixer.tools.extra_aux import extra_buses
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    settings = {**KEYED, "fps": 25, "extra_aux": 25, "encodes": {"extra": encode("p1")}}
    recipe = recipe_for(profile, settings)
    recipe["inputs"][0].update(codec="hevc", decode_storage="cuarray", extra_hw_frames=12)
    show, _, _ = prepare_demo.plan(recipe, runtime.media_dir)
    own = own_aux(preset="p1")
    cfg = parse({**show, "aux_buses": own})
    recipe["aux_buses"] = own + extra_buses(cfg, [], 25, prepare_demo.CLEAN_PORT, encode("p1"))
    show, _, _ = prepare_demo.plan(recipe, runtime.media_dir)
    config = runtime.media_dir / "mixer.demo.json"
    config.write_text(json.dumps(show))
    runtime.recipe_path.write_text(json.dumps(recipe))
    runtime.profile = {**profile, "mode_limits": {**profile["mode_limits"], "10:420": {
        **profile["mode_limits"]["10:420"], "nvenc_max_outputs": {25: 15}}}}
    runtime.process = None
    runtime.janus_api = "http://127.0.0.1"
    monkeypatch.setattr(runtime, "_mountpoints", lambda *args, **kwargs: None)
    monkeypatch.setattr("demos.mixer.setup_runtime.recipe_for", lambda *args: pytest.fail("resume must preserve its saved recipe"))
    apply_and_wait(runtime)
    assert runtime.status()["phase"] == "running", runtime.status()
    saved = json.loads(runtime.recipe_path.read_text())
    assert saved["inputs"] == recipe["inputs"]
    assert saved["setup"]["extra_aux"] == runtime.status()["settings"]["extra_aux"] == 10
    assert len(saved["aux_buses"]) == len(json.loads(config.read_text())["aux_buses"]) == 12
    assert [bus["id"] for bus in runtime._previous_show()[1]] == ["mv", "mv2"]
    assert all(source["decode_storage"] == "cuarray" and source["extra_hw_frames"] == 12
               for source in json.loads(config.read_text())["sources"] if source["kind"] == "video")


@pytest.mark.parametrize("fps,bit_depth,chroma,expected", [
    (60, 10, "422", 12), (60, 10, "420", 13), (60, 8, "420", 18),
    (50, 10, "422", 12), (30, 10, "422", 15), (25, 10, "422", 15)])
def test_l4_nvenc_margin_applies_only_to_the_measured_mode_and_rate(tmp_path, fps, bit_depth, chroma, expected):
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    encodes = {"sdr": encode("p5", 6000), "extra": encode("p3")}
    recipe = recipe_for(profile, {**KEYED, "fps": fps, "bit_depth": bit_depth, "chroma": chroma, "encodes": encodes})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    cfg = parse({**show, "aux_buses": own_aux(preset="p3")})
    assert extra_aux_limit(profile, cfg, recipe["setup"]["encodes"]) == expected
    assert recipe["setup"]["encodes"]["sdr"]["preset"] == "p5"


def test_mode_nvenc_margin_also_validates_fixed_outputs(tmp_path):
    base = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    profile = {**base, "mode_limits": {**base["mode_limits"], "10:422": {
        **base["mode_limits"]["10:422"], "nvenc_max_outputs": {60: 30}}}}
    recipe = recipe_for(profile, {**KEYED, "fps": 60, "chroma": "422", "encodes": {"sdr": encode("p5", 6000)}})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    fixed = [{"id": f"fixed{i}", "layout": {"preset": "source_pages"},
              "renditions": [{"id": "monitor", "port": 6000 + i * 4, **encode("p3")}]} for i in range(15)]
    cfg = parse({**show, "aux_buses": fixed})
    with pytest.raises(ValueError, match="72.1% of NVENC, above its 70% budget"):
        extra_aux_limit(profile, cfg, recipe["setup"]["encodes"])


@pytest.mark.parametrize("fps,maximum", [(25, 53), (30, 53), (50, 32), (60, 27)])
def test_l4_hdr_decodes_have_a_vram_cap(fps, maximum):
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    settings = {**DEFAULT_SETTINGS, "fps": fps, "chroma": "420", "source_count": maximum,
                "weights": [0, 1, 0, 0, 0, 0, 0]}
    assert recipe_for(profile, settings)["inputs"][1]["weight"] == maximum
    with pytest.raises(ValueError, match=f"HDR NVDEC inputs are limited to {maximum}"):
        recipe_for(profile, {**settings, "source_count": maximum + 1})


def test_resume_rejects_saved_hdr_decodes_above_current_capacity(runtime, monkeypatch):
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    recipe = recipe_for({**profile, "nvdec_hdr_decodes": {}}, {**DEFAULT_SETTINGS, "fps": 25,
        "chroma": "420", "source_count": 96, "weights": [0, 1, 0, 0, 0, 0, 0]})
    runtime.profile = profile
    runtime.process = None
    runtime.recipe_path.write_text(json.dumps(recipe))
    monkeypatch.setattr(runtime, "_validate_capacity", lambda *args: SetupRuntime._validate_capacity(runtime, *args))
    monkeypatch.setattr(prepare_demo, "prepare", lambda *args, **kwargs: pytest.fail("must reject before preparation"))
    with pytest.raises(ValueError, match="HDR NVDEC inputs are limited to 53"):
        runtime.apply()
    assert runtime.worker is None
    assert json.loads(runtime.recipe_path.read_text()) == recipe


def test_failed_start_does_not_restart_previous_outputs_above_current_capacity(runtime, monkeypatch):
    from pyplumber.mixer.tools.extra_aux import extra_buses
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    runtime.profile = profile
    runtime.process = None
    recipe = recipe_for(profile, {**KEYED, "fps": 25})
    previous, _, _ = prepare_demo.plan(recipe, runtime.media_dir)
    previous["aux_buses"] = extra_buses(parse(previous), [], 20, prepare_demo.CLEAN_PORT, encode("p1"))
    # Treat these as the old managed tail, so the requested setup can remove them first.
    runtime.recipe_path.write_text(json.dumps({**recipe, "setup": {**recipe["setup"], "extra_aux": 20}}))
    config = runtime.media_dir / "mixer.demo.json"
    original = json.dumps(previous)
    config.write_text(original)
    monkeypatch.setattr(runtime, "_validate_capacity", lambda *args: SetupRuntime._validate_capacity(runtime, *args))
    monkeypatch.setattr(prepare_demo, "prepare", lambda recipe, directory, **kwargs:
        config.write_text(json.dumps(prepare_demo.plan(recipe, directory)[0])))
    starts = []
    def start(path):
        starts.append(path)
        raise RuntimeError("native startup failed")
    monkeypatch.setattr(runtime, "_start", start)
    apply_and_wait(runtime, {**KEYED, "fps": 25})
    assert len(starts) == 1
    assert config.read_text() == original
    assert "recovery failed: The instance supports at most 20 encoded outputs" in runtime.status()["message"]


def test_a_janus_prune_failure_leaves_the_mixer_running_and_shows_in_the_status(runtime, monkeypatch):
    runtime.janus_api = "http://127.0.0.1:8088/janus"
    def refuse(*_, prune=False, replace=False):
        if prune:
            raise OSError("Connection refused")
    monkeypatch.setattr("pyplumber.mixer.gui.setup_runtime.janus_mountpoints.sync", refuse)
    apply_and_wait(runtime, {**KEYED, "fps": 30, "bit_depth": 8, "extra_aux": 1})
    status = runtime.status()
    assert (status["phase"], status["message"]) == ("running", "Mixer ready. Janus mountpoints failed: Connection refused.")


@pytest.mark.parametrize("codec", ["h264", "av1_nvenc", "libx265", None, 1])
def test_setup_rejects_non_offered_codecs(codec):
    with pytest.raises(ValueError, match="codec must be"):
        recipe_for(T4, {**DEFAULT_SETTINGS, "encodes": {"extra": {**encode(), "codec": codec}}})


def test_hdr_program_cannot_be_changed_to_h264():
    with pytest.raises(ValueError, match="Program HDR requires hevc_nvenc"):
        recipe_for(T4, {**DEFAULT_SETTINGS, "encodes": {"hdr": {**encode(), "codec": "h264_nvenc"}}})


@pytest.mark.parametrize("sdr_codec,clean_codec", [("hevc_nvenc", "h264_nvenc"), ("h264_nvenc", "hevc_nvenc")])
def test_program_and_clean_codec_are_independent_on_hdr_canvas(tmp_path, sdr_codec, clean_codec):
    encodes = {output: {**encode("p1", 250), "codec": codec}
               for output, codec in (("sdr", sdr_codec), ("sdr_clean", clean_codec))}
    show, _, _ = prepare_demo.plan(recipe_for(T4, {**KEYED, "encodes": encodes}), tmp_path)
    got = {r["id"]: (r["codec"], r["profile"], r["color"], r["bitrate_kbps"]) for r in show["renditions"]}
    assert got["sdr"] == (sdr_codec, "main" if sdr_codec == "hevc_nvenc" else "baseline", "sdr", 250)
    assert got["sdr_clean"] == (clean_codec, "main" if clean_codec == "hevc_nvenc" else "baseline", "sdr", 250)
    assert got["hdr"] == ("hevc_nvenc", "main10", "hlg", 8000)


def test_cuarray_profile_only_changes_encoded_generated_inputs():
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4_CUARRAY]
    recipe = recipe_for(profile, DEFAULT_SETTINGS)
    for source in recipe["inputs"][:2]:
        assert all(source[key] == value for key, value in profile["generated_decode"].items())
    assert all("decode_storage" not in source and "extra_hw_frames" not in source for source in recipe["inputs"][2:])
    for instance in (InstanceType.TESLA_T4, InstanceType.NVIDIA_L4):
        assert all("decode_storage" not in source for source in recipe_for(INSTANCE_PROFILES[instance], DEFAULT_SETTINGS)["inputs"])


def test_first_resume_keeps_declared_outputs_and_applies_shared_aux_codec(runtime, monkeypatch):
    runtime.process = None
    runtime.janus_api = "http://127.0.0.1:8088/janus"
    monkeypatch.setattr(runtime, "_mountpoints", lambda *args, **kwargs: None)
    encodes = {key: {**encode("p1", 250), "codec": codec} for key, codec in
               (("mv", "hevc_nvenc"), ("mv2", "h264_nvenc"), ("extra", "hevc_nvenc"))}
    recipe = recipe_for(T4, {**KEYED, "fps": 25, "extra_aux": 2, "encodes": encodes})
    recipe["aux_buses"] = own_aux() + [
        {"id": f"aux{i}", "layout": {"preset": "source_pages"},
         "renditions": [{"id": "monitor", "port": 5016 + i * 4, "codec": "h264_nvenc", "profile": "high"}]}
        for i in range(2)]
    runtime.recipe_path.write_text(json.dumps(recipe))
    runtime.resume()
    runtime.worker.join(3)
    assert runtime.status()["phase"] == "running", runtime.status()
    saved = json.loads(runtime.recipe_path.read_text())
    assert [b["id"] for b in saved["aux_buses"]] == ["mv", "mv2", "aux0", "aux1"]
    assert [(b["renditions"][0]["codec"], b["renditions"][0]["profile"], b["renditions"][0]["bitrate_kbps"])
            for b in saved["aux_buses"]] == [("hevc_nvenc", "main", 250), ("h264_nvenc", "high", 250),
                                            ("hevc_nvenc", "main", 250), ("hevc_nvenc", "main", 250)]
    # Shared extras change back without inheriting the old HEVC profile; own buses stay independent.
    settings = {**saved["setup"], "encodes": {**saved["setup"]["encodes"], "extra": {**encode("p1", 500), "codec": "h264_nvenc"}}}
    runtime.bridge.command = lambda *_: "[]"
    apply_and_wait(runtime, settings)
    assert runtime.status()["phase"] == "running", runtime.status()
    buses = json.loads(runtime.recipe_path.read_text())["aux_buses"]
    assert buses[0]["renditions"][0]["codec"] == "hevc_nvenc"
    assert all(b["renditions"][0]["profile"] == "high" for b in buses[1:])


@pytest.mark.parametrize("extra_codec,expected", [("h264_nvenc", 8), ("hevc_nvenc", 11)])
def test_extra_aux_budget_uses_selected_codec(tmp_path, extra_codec, expected):
    recipe = recipe_for(T4, {**KEYED, "fps": 30, "bit_depth": 8,
                             "encodes": {"extra": {**encode("p1"), "codec": extra_codec}}})
    show, _, _ = prepare_demo.plan(recipe, tmp_path)
    cfg = parse({**show, "aux_buses": own_aux(preset="p1")})
    assert extra_aux_limit(T4, cfg, recipe["setup"]["encodes"]) == expected


def test_janus_codec_replacement_happens_stopped_and_restores_before_rollback(runtime, monkeypatch):
    runtime.janus_api = "http://127.0.0.1:8088/janus"
    config = runtime.media_dir / "mixer.demo.json"
    previous, _, _ = prepare_demo.plan(recipe_for(T4, KEYED), runtime.media_dir)
    config.write_text(json.dumps(previous))
    events = []
    monkeypatch.setattr(runtime, "_stop", lambda **kwargs: events.append("stop"))
    def sync(api, specs, prune=False, replace=False):
        assert events[-1] == "stop"
        assert replace and not prune
        events.append(specs["sdr"]["videocodec"])
    monkeypatch.setattr("pyplumber.mixer.gui.setup_runtime.janus_mountpoints.sync", sync)
    def start(path):
        events.append("start")
        if events[-2] == "h265":
            raise RuntimeError("new encoder failed")
    monkeypatch.setattr(runtime, "_start", start)
    apply_and_wait(runtime, {**KEYED, "encodes": {"sdr": {**encode(), "codec": "hevc_nvenc"}}})
    assert "previous setup restored" in runtime.status()["message"]
    assert events == ["stop", "h265", "start", "stop", "stop", "h264", "start"]
    assert json.loads(config.read_text()) == previous


def test_janus_replacement_failure_does_not_start_a_mismatched_encoder(runtime, monkeypatch):
    runtime.janus_api = "http://127.0.0.1:8088/janus"
    def fail(*args, **kwargs):
        raise RuntimeError("mountpoint protected")
    monkeypatch.setattr("pyplumber.mixer.gui.setup_runtime.janus_mountpoints.sync", fail)
    monkeypatch.setattr(runtime, "_start", lambda *_: pytest.fail("codec replacement must succeed first"))
    apply_and_wait(runtime, {**KEYED, "encodes": {"sdr": {**encode(), "codec": "hevc_nvenc"}}})
    assert runtime.status()["phase"] == "error"
    assert "mountpoint protected" in runtime.status()["message"]
    assert not (runtime.media_dir / "mixer.demo.json").exists()
