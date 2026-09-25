"""Recipe expansion plus real CPU asset preparation; no mixer or GPU required."""
from collections import Counter
import base64
import json
from pathlib import Path
import shutil
import subprocess

import pytest

pytest.importorskip("numpy")
from prepare_demo import ensure_asset, plan, prepare, download, render_wipe, WIPE_NAMES
from demo_recipe import allocate
from pyplumber.mixer.config import load, parse, scene_layers
from v210_fixture import frame_stride


@pytest.fixture
def recipe():
    doc = json.loads((Path(__file__).resolve().parents[1] / "demo.example.json").read_text())
    doc["canvas"].update(width=96, height=64, fps=4)
    doc["generation"].update(seconds=1, sdr_encoder="libx264", hdr_encoder="libx265")
    return doc


def test_failed_generation_does_not_leave_a_cached_asset(tmp_path):
    path = tmp_path / "assets" / "clip.mp4"

    def fail(staged):
        staged.write_bytes(b"incomplete")
        raise RuntimeError("encoder failed")

    with pytest.raises(RuntimeError, match="encoder failed"):
        ensure_asset(path, fail)
    assert not path.exists()
    assert not list(path.parent.iterdir())
    ensure_asset(path, lambda staged: staged.write_bytes(b"complete"))
    ensure_asset(path, fail)
    assert path.read_bytes() == b"complete"


@pytest.mark.parametrize("fps", [25, 30, 50, 60])
@pytest.mark.parametrize("pattern", WIPE_NAMES)
def test_wipes_move_and_cover_the_midpoint(tmp_path, fps, pattern):
    if not shutil.which("ffmpeg"):
        pytest.skip("requires FFmpeg")
    import numpy as np

    path = tmp_path / "wipe.mov"
    render_wipe(path, "96x64", fps, "ffmpeg", pattern)
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 64, 96, 4)
    assert len(frames) == 2 * fps
    assert not frames[0, :, :, 3].any() and not frames[-1, :, :, 3].any()
    assert (frames[fps, :, :, 3] == 255).all()
    # Adjacent opaque frames still move; alpha alone must not hide a frozen graphic.
    assert not np.array_equal(frames[fps, :, :, :3], frames[fps + 1, :, :, :3])
    assert 0 < np.count_nonzero(frames[fps // 2, :, :, 3]) < 64 * 96


def test_source_counts_layout_counts_and_reproducible_geometry(recipe, tmp_path):
    doc, _, allocation = plan(recipe, tmp_path)
    assert allocation == dict(sdr420=8, hlg420=4, hlg422=2, sdr422=2, bunny=0, browser=0)
    assert len(doc["sources"]) == len({s["id"] for s in doc["sources"]}) == 16
    assert len(doc["scenes"]) == 24
    assert Counter(s["id"].rsplit("_", 1)[0] for s in doc["scenes"]) == {
        k: v for k, v in recipe["layouts"].items() if v}
    assert doc == plan(recipe, tmp_path)[0]
    recipe["seed"] += 1
    other = plan(recipe, tmp_path)[0]
    assert doc["scenes"] != other["scenes"]
    assert doc["sources"] == other["sources"]
    assert any(len(s["items"]) == 16 for s in doc["scenes"])
    for scene in doc["scenes"]:
        assert len({i["source"] for i in scene["items"]}) == len(scene["items"])
        for i in scene["items"]:
            r = i["dst"]
            assert all(v % 2 == 0 for v in r.values())
            assert 0 <= r["x"] < r["x"] + r["w"] <= 96
            assert 0 <= r["y"] < r["y"] + r["h"] <= 64
    recipe.update(source_count=3, scene_count=31)
    smaller = plan(recipe, tmp_path)[0]
    assert len(smaller["sources"]) == 3 and len(smaller["scenes"]) == 31
    assert max(len(s["items"]) for s in smaller["scenes"]) <= 3


def test_weights_round_to_exact_total_and_zero_disables():
    assert allocate(7, [1, 1, 1, 0]) == [3, 2, 2, 0]
    assert allocate(32, [1e308, 1e308, 0]) == [16, 16, 0]
    for weights in ([0, 0], [-1, 2], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            allocate(16, weights)


@pytest.mark.parametrize("storage,color,pattern,bytes_per_pixel", [("nv12", "sdr", "testsrc2", 1.5), ("p010", "hlg", "0", 3)])
def test_raw_420_assets_are_animated_exact_size_and_cached(recipe, tmp_path, storage, color, pattern, bytes_per_pixel):
    if not shutil.which("ffmpeg"):
        pytest.skip("requires FFmpeg")
    recipe.update(source_count=2, scene_count=1, layouts={"grid_2": 1})
    recipe["inputs"] = [{"id": "raw", "kind": "generated", "color": color, "chroma": "420",
                          "storage": storage, "pattern": pattern, "weight": 1}]
    show, jobs, _ = plan(recipe, tmp_path)
    assert [s["kind"] for s in show["sources"]] == [storage, storage]
    raw_jobs = [(p, writer) for p, writer in jobs.items() if p.suffix == f".{storage}"]
    assert len(raw_jobs) == 1  # distinct chains can reuse the cached clip
    path, writer = raw_jobs[0]
    ensure_asset(path, writer)
    frame_size = int(96 * 64 * bytes_per_pixel)
    data = path.read_bytes()
    assert len(data) == frame_size * 4
    assert data[:frame_size] != data[frame_size:2 * frame_size]
    ensure_asset(path, lambda _: pytest.fail("cached raw clip must not be regenerated"))


@pytest.mark.parametrize("color,chroma", [("hlg", "420"), ("sdr", "422")])
def test_raw_nv12_storage_rejects_incompatible_color(recipe, tmp_path, color, chroma):
    recipe["inputs"] = [{"id": "raw", "kind": "generated", "color": color, "chroma": chroma,
                          "storage": "nv12", "weight": 1}]
    with pytest.raises(ValueError, match="SDR 4:2:0 only"):
        plan(recipe, tmp_path)


def test_sdr_pattern_pool_keeps_cellauto_opt_in(recipe, tmp_path):
    recipe["inputs"] = [recipe["inputs"][0]]
    recipe.update(source_count=16, scene_count=1, layouts={"grid_16": 1})
    spec = recipe["inputs"][0]
    assert all("cellauto" not in s["path"] for s in plan(recipe, tmp_path)[0]["sources"])
    recipe["source_count"] = 3
    spec["patterns"] = ["bars", "gradients"]
    sources = plan(recipe, tmp_path)[0]["sources"]
    assert sources[0]["path"] == sources[2]["path"] != sources[1]["path"]
    assert len({s["id"] for s in sources}) == 3
    assert all(s["independent"] for s in sources)
    del spec["patterns"]
    spec["pattern"] = "cellauto"
    assert all("cellauto" in s["path"] for s in plan(recipe, tmp_path)[0]["sources"])
    spec["patterns"] = ["bars"]
    with pytest.raises(ValueError, match="not both"):
        plan(recipe, tmp_path)


@pytest.mark.parametrize("pool", [[], "bars", ["unknown"]])
def test_invalid_pattern_pool_fails_before_preparation(recipe, tmp_path, pool):
    recipe["inputs"][0]["patterns"] = pool
    with pytest.raises(ValueError, match="patterns must"):
        plan(recipe, tmp_path)


@pytest.mark.parametrize("capacity", [32, 64])
@pytest.mark.parametrize("width,height", [(1920, 1080), (1080, 1920)])
def test_large_grids_cover_canvas_with_distinct_sources(recipe, tmp_path, capacity, width, height):
    recipe.update(source_count=capacity, scene_count=1, layouts={f"grid_{capacity}": 1})
    recipe["canvas"].update(width=width, height=height)
    doc, _, _ = plan(recipe, tmp_path)
    items = doc["scenes"][0]["items"]
    assert {item["source"] for item in items} == {source["id"] for source in doc["sources"]}
    assert len(items) == capacity
    rectangles = [item["dst"] for item in items]
    assert sum(r["w"] * r["h"] for r in rectangles) == width * height
    for index, r in enumerate(rectangles):
        assert all(value % 2 == 0 for value in r.values())
        assert 0 <= r["x"] < r["x"] + r["w"] <= width
        assert 0 <= r["y"] < r["y"] + r["h"] <= height
        for other in rectangles[index + 1:]:
            assert (r["x"] + r["w"] <= other["x"] or other["x"] + other["w"] <= r["x"] or
                    r["y"] + r["h"] <= other["y"] or other["y"] + other["h"] <= r["y"])
    cfg = parse(doc)
    assert len(scene_layers(cfg, cfg.scenes[0])) == capacity


@pytest.mark.parametrize("count", [1, 3, 16, 32])
def test_layouts_stay_inside_portrait_canvas(recipe, tmp_path, count):
    recipe["canvas"].update(width=720, height=1280)
    recipe["source_count"] = count
    doc = plan(recipe, tmp_path)[0]
    for scene in doc["scenes"]:
        for i in scene["items"]:
            r = i["dst"]
            assert r["w"] > 0 and r["h"] > 0
            assert 0 <= r["x"] < r["x"] + r["w"] <= 720
            assert 0 <= r["y"] < r["y"] + r["h"] <= 1280


def test_alpha_overlay_preserves_blend_and_requires_browser(recipe, tmp_path):
    recipe["layouts"] = {"alpha_overlay": 1}
    with pytest.raises(ValueError, match="browser"):
        plan(recipe, tmp_path)
    recipe["inputs"][-1]["weight"] = 4
    doc = plan(recipe, tmp_path)[0]
    cfg = parse(doc)
    for scene in cfg.scenes:
        layers = list(scene_layers(cfg, scene).values())
        assert layers[0]["z"] == 0 and layers[1]["z"] == 1
        assert layers[1]["blend"] is True
        assert cfg.source(scene.items[1].source).kind == "browser"


def test_browser_alpha_recipe_covers_sdr_and_hdr_with_an_embedded_page(tmp_path):
    directory = Path(__file__).resolve().parents[1]
    recipe = json.loads((directory / "demo.browser-alpha.json").read_text())
    doc, _, _ = plan(recipe, tmp_path)
    cfg = parse(doc)
    overlays = [scene for scene in cfg.scenes if scene.id.startswith("alpha_overlay_")]
    assert {cfg.source(scene.items[0].source).color.transfer for scene in overlays} == {"sdr", "hlg"}
    browser = next(source for source in cfg.sources if source.kind == "browser")
    prefix, payload = browser.location.split(",", 1)
    assert prefix == "data:text/html;base64"
    assert base64.b64decode(payload) == (directory / "browser_alpha.html").read_bytes()


def test_equal_recipe_has_six_source_types_and_32_scenes(tmp_path):
    recipe = json.loads((Path(__file__).resolve().parents[1] / "demo.equal.json").read_text())
    doc, _, allocation = plan(recipe, tmp_path)
    assert list(allocation.values()) == [3, 3, 3, 3, 2, 2]
    assert len(doc["sources"]) == 16 and len(doc["scenes"]) == 32
    assert len([s for s in doc["scenes"] if s["id"].startswith("alpha_overlay_")]) == 8
    assert doc["canvas"]["fps"] == 60
    assert (doc["canvas"]["width"], doc["canvas"]["height"]) == (1080, 1920)
    assert (doc["sources"][0]["width"], doc["sources"][0]["height"]) == (1920, 1080)
    assert {s["items"][0]["source"] for s in doc["scenes"] if s["id"].startswith("alpha_overlay_")} == {"sdr420_001"}


def test_cinematic_recipe_keeps_http_hdr_and_smooth_422_inputs(tmp_path):
    recipe = json.loads((Path(__file__).resolve().parents[1] / "demo.cinematic.json").read_text())
    doc, _, allocation = plan(recipe, tmp_path)
    assert len(doc["sources"]) == 16 and len(doc["scenes"]) == 32
    assert allocation["sol_pq"] == allocation["sol_hlg"] == 1
    assert allocation["hlg422"] == 3
    assert [scene["items"][0]["source"] for scene in doc["scenes"][:2]] == ["sol_pq_000", "sol_hlg_000"]
    assert (doc["canvas"]["width"], doc["canvas"]["height"], doc["canvas"]["fps"]) == (1080, 1920, 60)
    assert all(source["url"].startswith("https://") for source in recipe["inputs"][:2])


@pytest.mark.parametrize("field,value", [("source_count", 129), ("source_count", 0), ("scene_count", 0)])
def test_invalid_counts_fail_before_creating_assets(recipe, tmp_path, field, value):
    recipe[field] = value
    with pytest.raises(ValueError):
        prepare(recipe, tmp_path)
    assert not list(tmp_path.iterdir())


def test_conflicting_rtp_rtcp_pairs_fail_before_creating_assets(recipe, tmp_path):
    recipe["renditions"][1]["port"] = 5005
    with pytest.raises(ValueError, match="port pairs"):
        prepare(recipe, tmp_path)
    assert not list(tmp_path.iterdir())


def test_downloads_are_opt_in_and_reused(recipe, tmp_path, monkeypatch):
    doc, jobs, _ = plan(recipe, tmp_path)
    assert not any("downloads" in p.parts for p in jobs)
    recipe["inputs"] = [recipe["inputs"][-2]]
    recipe["inputs"][0]["weight"] = 1
    doc, jobs, _ = plan(recipe, tmp_path)
    downloads = [p for p in jobs if "downloads" in p.parts]
    assert len(downloads) == 1
    assert len({s["path"] for s in doc["sources"]}) == 1
    assert len(parse(doc).sources) == 16  # Independent decode chains, one cached movie.
    import io
    responses = []

    def response(*args, **kwargs):
        stream = io.BytesIO(b"complete media")
        stream.headers = {"Content-Length": "14"}
        responses.append(stream)
        return stream

    monkeypatch.setattr("prepare_demo.urlopen", response)
    path = downloads[0]
    ensure_asset(path, jobs[path])
    ensure_asset(path, jobs[path])
    assert path.read_bytes() == b"complete media" and len(responses) == 1


def test_download_rejects_truncated_response(tmp_path, monkeypatch):
    import io
    stream = io.BytesIO(b"truncated")
    stream.headers = {"Content-Length": "100"}
    monkeypatch.setattr("prepare_demo.urlopen", lambda *args, **kwargs: stream)
    path = tmp_path / "movie.mp4"
    with pytest.raises(ValueError, match="incomplete download"):
        ensure_asset(path, lambda out: download(out, "https://example.org/movie.mp4"))
    assert not path.exists()


def test_prepare_demo_produces_playable_media_and_mapped_config(recipe, tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("requires FFmpeg and ffprobe")
    encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.DEVNULL)
    if any(encoder not in encoders for encoder in (b"libx264", b"libx265", b"qtrle")):
        pytest.skip("requires libx264, libx265 and qtrle")
    root = tmp_path.resolve()
    path = prepare(recipe, root, runtime_media_dir="/media")
    cfg = load(str(path))
    assert cfg.working_format == "p210le" and cfg.out_color.transfer == "hlg"
    assert len(cfg.sources) == 16 and len(cfg.scenes) == 24

    def local(location):
        return root / Path(location).relative_to("/media")

    def probe(media):
        return json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(media)]))["streams"][0]

    for source in cfg.sources:
        media = local(source.location)
        if source.kind == "v210":
            assert media.stat().st_size == frame_stride(96) * 64 * 4
        else:
            stream = probe(media)
            assert (stream["width"], stream["height"], stream["r_frame_rate"]) == (96, 64, "4/1")
            if source.color.transfer == "hlg":
                assert (stream["codec_name"], stream["pix_fmt"]) == ("hevc", "yuv420p10le")
                assert stream["color_transfer"] == "arib-std-b67"
                assert stream["color_primaries"] == "bt2020"
                assert stream["color_space"] == "bt2020nc"
            else:
                assert (stream["codec_name"], stream["pix_fmt"]) == ("h264", "yuv420p")
    assert local(cfg.source("hlg422_000").location).read_bytes() != local(cfg.source("hlg422_001").location).read_bytes()
    wipe = local(cfg.wipes[0].path)
    assert wipe.parent == root / "media_wipes"
    assert probe(wipe)["pix_fmt"] == "argb"
    alpha = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(wipe), "-vf", "alphaextract", "-pix_fmt", "gray",
        "-f", "rawvideo", "pipe:1"])
    frame_size = 96 * 64
    assert max(alpha[:frame_size]) == 0
    assert min(alpha[4 * frame_size:5 * frame_size]) == 255
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file() and p != path}
    recipe["scene_count"] = 7
    prepare(recipe, root, runtime_media_dir="/media")
    assert all(p.stat().st_mtime_ns == mtime for p, mtime in before.items())
    assert len(load(str(path)).scenes) == 7
