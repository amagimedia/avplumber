from dataclasses import replace
import json
from types import SimpleNamespace
import threading

import pytest

from pyplumber.mixer.aux import AuxMultiview, aux_fps, composition, parse_aux_buses, validate_assignments
from pyplumber.mixer.config import AuxBus, ConfigError, Item, MixerConfig, Rect, Rendition, Scene, Source


@pytest.fixture
def cfg():
    sources = tuple(Source(f"s{i}", "video", f"clip{i}.mp4", 1920, 1080) for i in range(64))
    items = tuple(Item(s.id, Rect(i % 8 * 134, i // 8 * 240, 134, 240)) for i, s in enumerate(sources))
    scenes = (Scene("full", (Item("s0", Rect(0, 0, 1080, 1920)),)),
              Scene("repeat", (Item("s0", Rect(0, 0, 540, 960)), Item("s0", Rect(540, 960, 540, 960), blend=True))),
              Scene("grid64", items))
    return MixerConfig(1080, 1920, 60, sources, scenes)


def bus_json(**kwargs):
    return {"id": "multiview", "scenes": ["full"] * 8,
            "renditions": [{"id": "monitor", "port": 5010}], **kwargs}


@pytest.mark.parametrize("program,aux", [(25, 25), (30, 30), (50, 25), (60, 30)])
def test_aux_cadence(program, aux):
    assert aux_fps(program) == aux


def test_repeated_scenes_and_occurrences_share_pads(cfg):
    result = composition(cfg, ["repeat"] * 8, "full")
    assert len(result["layers"]) == 18
    assert {l["input"] for l in result["layers"]} == {0, 64}
    assert result["active_inputs"] == "1" + "0" * 63 + "1"
    assert sum(l.get("blend", False) for l in result["layers"]) == 8
    assert len({l["z"] for l in result["layers"]}) == 18


def test_empty_preview_and_tiles_keep_only_real_pgm(cfg):
    result = composition(cfg, [None] * 8, "")
    assert len(result["layers"]) == 1
    assert result["layers"][0]["input"] == 64


@pytest.mark.parametrize("count", [63, 64, 96, 127])
def test_aux_mask_addresses_high_source_and_program_pads(cfg, count):
    cfg = replace(cfg, sources=tuple(Source(f"s{i}", "video", f"clip{i}.mp4", 1920, 1080)
                                    for i in range(count)),
                  scenes=(Scene("last", (Item(f"s{count - 1}", Rect(0, 0, 1080, 1920)),)),))
    wire = composition(cfg, ["last"] + [None] * 7, "")["active_inputs"]
    assert wire == ((1 << 63) | (1 << 62) if count == 63 else "0" * (count - 1) + "11")


def test_capacity_reserves_any_preview(cfg):
    cfg = replace(cfg, max_compositor_layers=512, scenes=(*cfg.scenes, Scene("grid63", cfg.scenes[-1].items[:-1])))
    assignments = ["grid64"] * 6 + ["grid63", None]
    assert len(composition(cfg, assignments, "grid64")["layers"]) == 512
    with pytest.raises(ConfigError, match="513"):
        validate_assignments(cfg, assignments[:-1] + ["full"])


def test_bus_validation_and_distinct_outputs(cfg):
    buses = parse_aux_buses([bus_json(), bus_json(id="second", renditions=[{"id": "monitor", "port": 5012}])], cfg)
    assert len(buses) == 2
    assert buses[0].renditions[0].fps == 30
    assert buses[0].renditions[0].codec == "h264_nvenc"
    for invalid in (bus_json(id="bad name"), bus_json(scenes=["missing"] * 8),
                    bus_json(renditions=[{"id": "monitor", "port": 5010, "color": "hlg"}])):
        with pytest.raises((ConfigError, ValueError)):
            parse_aux_buses([invalid], cfg)
    with pytest.raises(ConfigError, match="port"):
        parse_aux_buses([bus_json(), bus_json(id="second")], cfg)
    with pytest.raises(ConfigError, match="128"):
        parse_aux_buses([bus_json()], replace(cfg, sources=cfg.sources * 2))


def test_assignment_conflict_does_not_publish_or_overwrite(cfg):
    published = []
    def execute(command):
        prefix, value = command.split(" composition ", 1)
        assert prefix == "node.object.set aux_multiview_comp"
        published.append(json.loads(value))
    avp = SimpleNamespace(executeCommandsFromString=execute)
    mixer = SimpleNamespace(add_aux_destination=lambda *args: None)
    bus = AuxMultiview(avp, None, mixer, cfg, AuxBus("multiview", (None,) * 8, (Rendition("monitor"),)))
    revision = bus.revision
    first = {"expected_revision": revision, "scenes": ["full"] + [None] * 7}
    second = {"expected_revision": revision, "scenes": [None, "repeat"] + [None] * 6}
    results = []
    threads = [threading.Thread(target=lambda req=req: results.append(bus.assign(req))) for req in (first, second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(published) == 1
    assert sum(bool(r.get("conflict")) for r in results) == 1
    assert bus.revision != revision
    before = bus.revision, list(bus.scenes)
    with pytest.raises(ConfigError):
        bus.assign({"expected_revision": bus.revision, "scenes": ["grid64"] * 8})
    assert (bus.revision, bus.scenes) == before
    assert bus.assign({"scenes": [None] * 8})["conflict"]


def test_new_instance_rejects_previous_revision(cfg):
    mixer = SimpleNamespace(add_aux_destination=lambda *args: None)
    spec = AuxBus("multiview", (None,) * 8, ())
    old = AuxMultiview(None, None, mixer, cfg, spec)
    new = AuxMultiview(None, None, mixer, cfg, spec)
    assert new.assign({"expected_revision": old.revision, "scenes": [None] * 8})["conflict"]
