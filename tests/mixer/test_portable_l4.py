"""Portable L4 settings preserve the measured input contract on every Setup change."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.mixer.instance_profiles import INSTANCE_PROFILES, InstanceType
from demos.mixer.prepare_demo import plan
from demos.mixer.setup_runtime import DEFAULT_SETTINGS, SetupRuntime, desktop_preset, extra_aux_limit, recipe_for, source_limit
from pyplumber.mixer.config import parse


PACKAGE = Path(__file__).resolve().parents[2] / "docker-compose/mixer/deploy/l4"
spec = importlib.util.spec_from_file_location("l4_seed", PACKAGE / "seed.py")
l4_seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l4_seed)


def test_seed_generates_portable_recipe_and_preserves_edits(tmp_path):
    l4_seed.seed(PACKAGE / "settings.json", tmp_path)
    path = tmp_path / "demo.json"
    recipe = json.loads(path.read_text())
    show = plan(recipe, tmp_path)[0]
    assert len(show["sources"]) == 192
    assert show["canvas"]["raw_upload"] == "resident"
    assert len(show["scenes"]) == 256
    assert [bus["id"] for bus in show["aux_buses"][:2]] == ["mv", "mv2"]
    assert len(show["aux_buses"]) == 15
    assert [r["id"] for r in show["renditions"]] == ["sdr", "sdr_clean"]
    assert all(b["renditions"][0]["preset"] == "p1" for b in show["aux_buses"])
    assert not (tmp_path / "mixer.demo.json").exists(), "only preparation publishes the generated show"
    videos = [s for s in show["sources"] if s["kind"] == "video"]
    assert len(videos) == 105
    assert all(s["decode_storage"] == "cuarray" and s["extra_hw_frames"] == 12 for s in videos)
    recipe["setup"]["encodes"]["mv"]["bitrate_kbps"] = 500
    path.write_text(json.dumps(recipe))
    before = path.read_bytes()
    l4_seed.seed(PACKAGE / "settings.json", tmp_path)
    assert path.read_bytes() == before


def test_seed_does_not_replace_an_adopted_show(tmp_path):
    path = tmp_path / "mixer.demo.json"
    path.write_text('{"custom": true}')
    l4_seed.seed(PACKAGE / "settings.json", tmp_path)
    assert path.read_text() == '{"custom": true}'
    assert not (tmp_path / "demo.json").exists()


@pytest.mark.parametrize(("size", "total", "decoded", "raw", "browser"), [
    ("low", 96, 60, 16, 20), ("high", 192, 120, 32, 40),
])
def test_desktop_presets_plan_exact_inputs_and_fit_session_limit(tmp_path, size, total, decoded, raw, browser):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(desktop_preset(size)))
    l4_seed.seed("low" if size == "low" else path, tmp_path, "geforce")
    recipe = json.loads((tmp_path / "demo.json").read_text())
    show = plan(recipe, tmp_path)[0]
    assert len(show["sources"]) == total
    assert recipe["setup"]["weights"] == [decoded, 0, 0, 0, browser - 4, raw, 0, 0, 0]
    assert len(recipe["dsk"]) == 4
    assert show["canvas"]["raw_upload"] == "resident"
    assert len(show["scenes"]) == 256
    assert show["canvas"]["fps"] == 25 and show["canvas"]["working_format"] == "nv12"
    assert len(show["renditions"]) == len(show["aux_buses"]) == 2
    profile = INSTANCE_PROFILES[InstanceType.GEFORCE]
    assert extra_aux_limit(profile, parse(show), recipe["setup"]["encodes"]) == 8


@pytest.mark.parametrize(("line", "suggested", "reference"), [
    ("NVIDIA GeForce RTX 5080, 12.0, 16303, 15800", "96", "nvidia_l4_cuarray"),   # 2 NVDEC, 16 GB
    ("NVIDIA L4, 8.9, 23034, 22500", "192", "nvidia_l4_cuarray"),                 # the reference itself
    ("NVIDIA GeForce RTX 4060, 8.9, 8188, 7600", "48", "nvidia_l4_cuarray"),       # 1 NVDEC, 8 GB
    ("NVIDIA GeForce RTX 3060, 8.6, 12288, 11500", "24", "tesla_t4"),              # Ampere: from the T4
    ("", "24", None),                                                               # no GPU found
])
def test_desktop_suggests_the_largest_show_the_gpu_carries(tmp_path, monkeypatch, line, suggested, reference):
    monkeypatch.setenv("MIXER_GPU", line)
    monkeypatch.setattr("demos.mixer.gpu_spec.subprocess.check_output", lambda *a, **kw: "")
    status = SetupRuntime(tmp_path, tmp_path / "demo.json", SimpleNamespace(port=7777), "geforce").status()
    assert status["desktop"]["suggested"] == suggested and status["phase"] == "idle"
    assert list(status["desktop"]["presets"]) == ["24", "48", "96", "192"]
    assert (status["desktop"]["gpu"] or {}).get("reference") == reference
    assert status["desktop"]["fits"][suggested]


def test_scaling_leaves_the_measured_t4_and_l4_profiles_alone(tmp_path, monkeypatch):
    before = copy.deepcopy(INSTANCE_PROFILES)
    for line in ("NVIDIA GeForce RTX 3060, 8.6, 12288, 11500", "NVIDIA GeForce RTX 5090, 12.0, 32607, 32000"):
        monkeypatch.setenv("MIXER_GPU", line)
        SetupRuntime(tmp_path, tmp_path / "demo.json", SimpleNamespace(port=7777), "geforce")
    assert INSTANCE_PROFILES == before


@pytest.mark.parametrize(("line", "expected"), [
    ("Tesla T4, 7.5, 15360, 15000", "tesla_t4"), ("NVIDIA L4, 8.9, 23034, 22500", "nvidia_l4_cuarray"),
    ("NVIDIA GeForce RTX 5080, 12.0, 16303, 15800", "geforce"),
])
def test_auto_keeps_the_t4_and_l4_profiles(tmp_path, monkeypatch, line, expected):
    monkeypatch.setenv("MIXER_GPU", line)
    runtime = SetupRuntime(tmp_path, tmp_path / "demo.json", SimpleNamespace(port=7777), "auto")
    assert runtime.status()["instance_type"] == expected
    if expected != "geforce":
        assert runtime.profile["nvdec_decodes"] == INSTANCE_PROFILES[InstanceType(expected)]["nvdec_decodes"]
        assert "desktop" not in runtime.status()


@pytest.mark.parametrize(("line", "total"), [("NVIDIA GeForce RTX 5080, 12.0, 16303, 15800", 96),
                                             ("NVIDIA GeForce RTX 4060, 8.9, 8188, 7600", 48)])
def test_seed_fit_starts_the_suggested_show(tmp_path, monkeypatch, line, total):
    monkeypatch.setenv("MIXER_GPU", line)
    l4_seed.seed("fit", tmp_path, "geforce")
    recipe = json.loads((tmp_path / "demo.json").read_text())
    assert len(plan(recipe, tmp_path)[0]["sources"]) == total


def test_seed_accepts_low_bandwidth_hevc_monitors_and_scales_extra_count(tmp_path):
    settings = json.loads((PACKAGE / "settings.json").read_text())
    for output in ("mv", "mv2", "extra"):
        settings["encodes"][output].update(codec="hevc_nvenc", preset="p3", bitrate_kbps=500)
    settings["extra_aux"] = 30
    path = tmp_path / "initial.json"
    path.write_text(json.dumps(settings))
    media = tmp_path / "media"
    l4_seed.seed(path, media)
    recipe = json.loads((media / "demo.json").read_text())
    assert 0 < recipe["setup"]["extra_aux"] < settings["extra_aux"]
    buses = plan(recipe, media)[0]["aux_buses"]
    assert len(buses) == recipe["setup"]["extra_aux"] + 2
    assert all(bus["renditions"][0]["codec"] == "hevc_nvenc" and
               bus["renditions"][0]["bitrate_kbps"] == 500 for bus in buses)


@pytest.mark.parametrize("fps,depth,chroma", [(25, 8, "420"), (60, 10, "420"), (60, 10, "422")])
def test_setup_keeps_cuarray_contract_without_changing_legacy_profiles(fps, depth, chroma):
    optimized = INSTANCE_PROFILES[InstanceType.NVIDIA_L4_CUARRAY]
    legacy = INSTANCE_PROFILES[InstanceType.NVIDIA_L4]
    settings = {**DEFAULT_SETTINGS, "fps": fps, "bit_depth": depth, "chroma": chroma,
                "source_count": 16, "weights": [8, 0, 0, 0, 4, 4, 0] if depth == 8 else [6, 6, 0, 0, 4, 0, 0]}
    recipe = recipe_for(optimized, settings)
    encoded = [s for s in recipe["inputs"] if s["kind"] == "generated" and not s.get("storage") and not s.get("mxl") and not s.get("ndi") and s["chroma"] == "420"]
    assert all(s["codec"] == "hevc" and s["decode_storage"] == "cuarray" and s["extra_hw_frames"] == 12 for s in encoded)
    assert all("decode_storage" not in s for s in recipe_for(legacy, settings)["inputs"])
    assert source_limit(optimized, 25) == 192
    assert source_limit(legacy, 25) == 170
    # 30 fps carries the same total as 25 with fewer decodes: 105 NVDEC, 40 browser, 47 raw.
    assert source_limit(optimized, 30) == 192
    assert (optimized["nvdec_decodes"][30], optimized["browser_windows"], optimized["raw_upload_units"][30]) == (105, 40, 47)
    assert source_limit(legacy, 30) == 170


@pytest.mark.parametrize("fps,browsers", [(25, 40), (30, 40), (50, 40), (60, 30)])
def test_setup_browser_limit_includes_keys_and_clamps_saved_mix(fps, browsers):
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4_CUARRAY]
    settings = json.loads((PACKAGE / "settings.json").read_text())
    # A saved larger mix must follow the new rate's limits on Apply.
    settings.update(fps=fps, bit_depth=10, chroma="422", source_count=180, weights=[22, 22, 4, 0, 36, 0, 0])
    recipe = recipe_for(profile, settings)
    show = plan(recipe, Path("/tmp/mixer-test"))[0]
    assert sum(s["kind"] == "browser" for s in show["sources"]) == browsers
    assert len(show["sources"]) <= source_limit(profile, fps, 10, "422")


@pytest.mark.parametrize("counts", [
    [105, 0, 0, 0, 10, 47, 0],  # reduced browsers, unchanged decoder/upload counts
    [105, 0, 0, 0, 10, 0, 0],
    [105, 0, 0, 0, 0, 0, 0],   # NVDEC only, at its budget
    [0, 0, 0, 0, 10, 0, 0],
    [0, 0, 0, 0, 0, 47, 0],
])
def test_manual_source_counts_reach_generated_show_unchanged(tmp_path, counts):
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4_CUARRAY]
    recipe = recipe_for(profile, {**DEFAULT_SETTINGS, "fps": 30, "bit_depth": 8, "chroma": "420",
                                  "source_count": sum(counts), "weights": counts})
    assert [s["weight"] for s in recipe["inputs"]] == counts + [0, 0]   # no MXL or NDI
    show = plan(recipe, tmp_path)[0]
    assert len(show["sources"]) == sum(counts)
    assert sum(s["kind"] == "browser" for s in show["sources"]) == counts[4]
    assert sum(s["kind"] == "video" for s in show["sources"]) == counts[0]
    assert sum(s["kind"] == "nv12" for s in show["sources"]) == counts[5]
