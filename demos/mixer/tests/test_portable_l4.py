"""Portable L4 settings preserve the measured input contract on every Setup change."""

import importlib.util
import json
from pathlib import Path

import pytest

from instance_profiles import INSTANCE_PROFILES, InstanceType
from prepare_demo import plan
from setup_runtime import DEFAULT_SETTINGS, recipe_for, source_limit


PACKAGE = Path(__file__).resolve().parents[1] / "deploy" / "l4"
spec = importlib.util.spec_from_file_location("l4_seed", PACKAGE / "seed.py")
l4_seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l4_seed)


def test_seed_generates_portable_recipe_and_preserves_edits(tmp_path):
    l4_seed.seed(PACKAGE / "settings.json", tmp_path)
    path = tmp_path / "demo.json"
    recipe = json.loads(path.read_text())
    show = plan(recipe, tmp_path)[0]
    assert len(show["sources"]) == 88
    assert len(show["scenes"]) == 256
    assert [bus["id"] for bus in show["aux_buses"][:2]] == ["mv", "mv2"]
    assert len(show["aux_buses"]) == 14
    assert len(show["renditions"]) == 3
    assert not (tmp_path / "mixer.demo.json").exists(), "only preparation publishes the generated show"
    videos = [s for s in show["sources"] if s["kind"] == "video"]
    assert len(videos) == 44
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


def test_seed_accepts_low_bandwidth_hevc_monitors_and_scales_extra_count(tmp_path):
    settings = json.loads((PACKAGE / "settings.json").read_text())
    for output in ("mv", "mv2", "extra"):
        settings["encodes"][output].update(codec="hevc_nvenc", bitrate_kbps=500)
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
    encoded = [s for s in recipe["inputs"] if s["kind"] == "generated" and not s.get("storage") and s["chroma"] == "420"]
    assert all(s["codec"] == "hevc" and s["decode_storage"] == "cuarray" and s["extra_hw_frames"] == 12 for s in encoded)
    assert all("decode_storage" not in s for s in recipe_for(legacy, settings)["inputs"])
    assert source_limit(optimized, 25) == 192
    assert source_limit(legacy, 25) == 170
