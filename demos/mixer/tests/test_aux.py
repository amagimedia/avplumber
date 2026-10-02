from dataclasses import replace
import json
from types import SimpleNamespace
import threading

import pytest

from pyplumber.mixer.aux import AuxBus, AuxBuses
from pyplumber.mixer.aux_layout import (base_composition, check_assignments, grid_cells, layout_cells, page_grid,
                                        parse_layout, pvw_layouts)
from pyplumber.mixer.config import (aux_fps, default_latency_ms, default_pgm_delay_frames, parse_aux_buses)
from pyplumber.mixer.config import ConfigError, Item, MixerConfig, Rect, Scene, Source


@pytest.fixture
def cfg():
    sources = tuple(Source(f"s{i}", "video", f"clip{i}.mp4", 1920, 1080) for i in range(64))
    items = tuple(Item(s.id, Rect(i % 8 * 134, i // 8 * 240, 134, 240)) for i, s in enumerate(sources))
    scenes = (Scene("full", (Item("s0", Rect(0, 0, 1080, 1920)),)),
              Scene("repeat", (Item("s0", Rect(0, 0, 540, 960)), Item("s0", Rect(540, 960, 540, 960), blend=True))),
              Scene("grid64", items))
    return MixerConfig(1080, 1920, 60, sources, scenes, max_compositor_layers=640)


def bus_json(**kwargs):
    return {"id": "multiview", "scenes": ["full"] * 8, "renditions": [{"id": "monitor", "port": 5010}], **kwargs}


def pages_json(**kwargs):
    return {"id": "sources", "layout": {"preset": "source_pages"}, "renditions": [{"id": "monitor", "port": 5012}], **kwargs}


# PGM bottom left, PVW nowhere, three slots: nothing about the grid preset is assumed.
CELLS = [{"role": "slot", "slot": 2, "x": 0, "y": 0, "w": 540, "h": 480},
         {"role": "slot", "slot": 0, "x": 540, "y": 0, "w": 540, "h": 480},
         {"role": "source", "source": 63, "x": 0, "y": 480, "w": 540, "h": 480},
         {"role": "slot", "slot": 1, "x": 540, "y": 480, "w": 540, "h": 480},
         {"role": "pgm", "x": 0, "y": 960, "w": 1080, "h": 960}]


class FakeAvp:
    """Records the follower's layouts; ``statuses`` holds what each node reports, a missing node
    is unreachable. Like the binding, a command to an unreachable node is dropped, never raised;
    like the follower, a reachable one reports the revision it holds at once."""

    def __init__(self):
        self.layouts, self.statuses, self.handlers = [], {}, {}

    def executeCommandsFromString(self, command):
        node, key, value = command.split(" ", 3)[1:]
        assert key == "layout" and node.endswith("_pvw"), command
        if node in self.statuses:
            self.layouts.append(json.loads(value))
            self.statuses[node]["layout_revision"] = self.layouts[-1]["revision"]

    def node(self, name):
        if name not in self.statuses:
            raise Exception(f"Node {name} doesn't exist.")
        return SimpleNamespace(getObject=lambda _key: dict(self.statuses[name]))

    def registerControlCommand(self, name, handler, _payload):
        self.handlers[name] = handler

    def command(self, name, **request):
        return json.loads(self.handlers[name](json.dumps(request)))


def make_bus(cfg, spec, avp=None, latency_ms=None):
    mixer = SimpleNamespace(add_aux_destination=lambda *args: None, name="mixer",
                            latency_ms=default_latency_ms(cfg.fps) if latency_ms is None else latency_ms)
    return AuxBus(avp or FakeAvp(), None, mixer, cfg, parse_aux_buses([spec], cfg)[0])


def reachable(avp, bus, **compositor):
    """The bus's compositor and follower exist; the compositor draws *compositor*'s revision."""
    avp.statuses[bus.follower] = {"layout_revision": bus.revision, "pvw_scene": ""}
    avp.statuses[bus.node_name] = {"suspended": False, "composition_pending": False,
                                   "composition_revision": bus.revision, **compositor}


@pytest.mark.parametrize("program,aux", [(25, 25), (30, 30), (50, 25), (60, 30)])
def test_aux_cadence(program, aux):
    assert aux_fps(program) == aux
    assert aux_fps(program, full_rate=True) == program   # opt-in: the canvas rate, twice the work at 50/60


def composition(cfg, cells, scenes, preview):
    """What the follower sets: pvw[preview] followed by the base."""
    base = base_composition(cfg, cells, scenes)
    return {**base, "layers": (pvw_layouts(cfg, cells)[preview]["layers"] if preview else []) + base["layers"]}


def test_grid_preset_keeps_the_multiview_geometry_and_order(cfg):
    """pgm_pvw_grid is today's multiview: PVW and PGM on top, eight slots below; the slots draw
    in slot order and the program last, z from above the PVW reserve."""
    cells = layout_cells(cfg, parse_layout(cfg, {"preset": "pgm_pvw_grid"}))
    assert cells == grid_cells(cfg)
    assert [c["role"] for c in cells] == ["pvw", "pgm"] + ["slot"] * 8
    assert cells[:3] == [{"role": "pvw", "x": 0, "y": 0, "w": 540, "h": 960},
                         {"role": "pgm", "x": 540, "y": 0, "w": 540, "h": 960},
                         {"role": "slot", "slot": 0, "x": 0, "y": 960, "w": 270, "h": 480}]
    result = composition(cfg, cells, ["full"] * 8, "full")
    assert [layer["tile"] for layer in result["layers"][:-1]] == [
        {k: c[k] for k in ("x", "y", "w", "h")} for c in (cells[0], *cells[2:])]
    pgm = result["layers"][-1]
    assert (pgm["input"], pgm["z"]) == (64, 64 + 8)
    assert (pgm["dst_x"], pgm["dst_y"], pgm["dst_w"], pgm["dst_h"]) == (540, 0, 540, 960)


def test_repeated_scenes_and_occurrences_share_pads(cfg):
    result = composition(cfg, grid_cells(cfg), ["repeat"] * 8, "full")
    assert len(result["layers"]) == 18
    assert {l["input"] for l in result["layers"]} == {0, 64}
    assert result["active_inputs"] == "1" + "0" * 63 + "1"
    assert sum(l.get("blend", False) for l in result["layers"]) == 8
    assert len({l["z"] for l in result["layers"]}) == 18
    assert base_composition(cfg, grid_cells(cfg), [None] * 8)["layers"][0]["input"] == 64   # PGM only


@pytest.mark.parametrize("count", [63, 64, 96, 127])
def test_aux_mask_addresses_high_source_and_program_pads(cfg, count):
    cfg = replace(cfg, sources=tuple(Source(f"s{i}", "video", f"clip{i}.mp4", 1920, 1080) for i in range(count)),
                  scenes=(Scene("last", (Item(f"s{count - 1}", Rect(0, 0, 1080, 1920)),)),))
    wire = base_composition(cfg, grid_cells(cfg), ["last"] + [None] * 7)["active_inputs"]
    assert wire == ((1 << 63) | (1 << 62) if count == 63 else "0" * (count - 1) + "11")


def test_pvw_layouts_and_base_split_the_composition(cfg):
    cells = grid_cells(cfg)
    layouts = pvw_layouts(cfg, cells)
    assert set(layouts) == {"full", "repeat", "grid64"}
    assert [l["z"] for l in layouts["repeat"]["layers"]] == [0, 1]
    assert layouts["grid64"]["layers"][-1]["z"] == 63
    assert layouts["full"]["layers"][0]["tile"] == {k: cells[0][k] for k in ("x", "y", "w", "h")}
    assert layouts["full"]["active_inputs"] == 1
    base = base_composition(cfg, cells, ["repeat"] * 8)
    assert min(l["z"] for l in base["layers"]) == 64
    assert base["layers"][-1]["input"] == 64 and base["layers"][-1]["z"] == 64 + 16


def test_explicit_cells_put_slots_pgm_and_sources_anywhere(cfg):
    layout = parse_layout(cfg, {"cells": CELLS})
    cells = layout_cells(cfg, layout)
    assert cells == CELLS
    # No PVW cell: every scene's preview layers are empty and the base starts at z 0.
    assert all(l == {"layers": [], "active_inputs": 0} for l in pvw_layouts(cfg, cells).values())
    base = base_composition(cfg, cells, ["full", None, "repeat"])
    # Slot and source cells in list order (slot 2, slot 0, source 63, slot 1), the program last.
    assert [(l["input"], l["z"]) for l in base["layers"]] == [(0, 0), (0, 1), (0, 2), (63, 3), (64, 4)]
    assert base["layers"][0]["tile"] == {"x": 0, "y": 0, "w": 540, "h": 480}       # slot 2: "repeat"
    assert base["layers"][2]["tile"] == {"x": 540, "y": 0, "w": 540, "h": 480}     # slot 0: "full"
    assert (base["layers"][-1]["dst_y"], base["layers"][-1]["dst_h"]) == (960, 960)
    # Two PVW cells: each reserves the largest scene's z values.
    two = layout_cells(cfg, parse_layout(cfg, {"cells": [{"role": "pvw", "x": 0, "y": 0, "w": 540, "h": 960},
                                                         {"role": "pvw", "x": 540, "y": 0, "w": 540, "h": 960}]}))
    assert [l["z"] for l in pvw_layouts(cfg, two)["repeat"]["layers"]] == [0, 1, 64, 65]


@pytest.mark.parametrize("cell,match", [
    ({"role": "pip", "x": 0, "y": 0, "w": 2, "h": 2}, "role"),
    ({"role": "pvw", "x": 1, "y": 0, "w": 2, "h": 2}, "even"),
    ({"role": "pvw", "x": 0, "y": 0, "w": 0, "h": 2}, "positive"),
    ({"role": "pvw", "x": 0, "y": 1900, "w": 2, "h": 22}, "inside"),
    ({"role": "pvw", "x": 0, "y": 0, "w": 2, "h": 2, "slot": 0}, "has role"),
    ({"role": "slot", "x": 0, "y": 0, "w": 2, "h": 2}, "has role"),
    ({"role": "slot", "slot": 1, "x": 0, "y": 0, "w": 2, "h": 2}, "0 to n-1"),
    ({"role": "slot", "slot": True, "x": 0, "y": 0, "w": 2, "h": 2}, "slot index"),
    ({"role": "source", "source": 64, "x": 0, "y": 0, "w": 2, "h": 2}, "0 to 63"),
])
def test_explicit_cells_are_validated(cfg, cell, match):
    with pytest.raises(ConfigError, match=match):
        parse_layout(cfg, {"cells": [cell]})
    with pytest.raises(ConfigError, match="0 to n-1"):
        parse_layout(cfg, {"cells": [*CELLS, {**CELLS[0], "slot": 1}]})


def test_layout_specs_normalize_and_reject_unknown_shapes(cfg):
    assert parse_layout(cfg, {"preset": "source_pages"}) == {"preset": "source_pages", "page": 0}
    assert parse_layout(cfg, {"preset": "source_pages", "page": 5})["page"] == 5
    for invalid in ({"preset": "mosaic"}, {"preset": "pgm_pvw_grid", "rows": 2}, {"preset": "source_pages", "page": 6},
                    {"preset": "source_pages", "page": None}, {"preset": "source_pages", "rotate_s": 5},
                    {"cells": []}, {"cells": CELLS, "preset": "pgm_pvw_grid"}, "pgm_pvw_grid", None):
        with pytest.raises(ConfigError):
            parse_layout(cfg, invalid)


@pytest.mark.parametrize("w,h,cols,rows", [(1080, 1920, 2, 6), (1920, 1080, 4, 3)])
def test_page_grid_fills_the_canvas_with_even_separate_tiles(cfg, w, h, cols, rows):
    grid = page_grid(replace(cfg, canvas_w=w, canvas_h=h))
    assert len(grid) == cols * rows == 12
    assert len({(r["w"], r["h"]) for r in grid}) == 1
    for r in grid:
        assert all(v % 2 == 0 for v in r.values())
        assert r["x"] >= 0 and r["y"] >= 0 and r["x"] + r["w"] <= w and r["y"] + r["h"] <= h
    xs, ys = sorted({r["x"] for r in grid}), sorted({r["y"] for r in grid})
    assert (len(xs), len(ys)) == (cols, rows)
    assert all(b - a > grid[0]["w"] for a, b in zip(xs, xs[1:]))
    assert all(b - a > grid[0]["h"] for a, b in zip(ys, ys[1:]))


def test_source_pages_preset_draws_one_page_of_source_cells(cfg):
    assert page_grid(cfg)[0] == {"x": 2, "y": 10, "w": 536, "h": 300}
    cells = layout_cells(cfg, parse_layout(cfg, {"preset": "source_pages"}), 5)
    assert cells[0] == {"role": "source", "source": 60, **page_grid(cfg)[0]}
    result = base_composition(cfg, cells, [])
    assert [(l["input"], l["z"]) for l in result["layers"]] == [(60, 0), (61, 1), (62, 2), (63, 3)]
    assert result["active_inputs"] == sum(1 << i for i in range(60, 64))


def test_budget_counts_the_largest_preview_and_refuses_beyond_it(cfg):
    cfg = replace(cfg, max_compositor_layers=512, scenes=(*cfg.scenes, Scene("grid63", cfg.scenes[-1].items[:-1])))
    cells, assignments = grid_cells(cfg), ["grid64"] * 6 + ["grid63", None]
    check_assignments(cfg, cells, assignments, 512)
    assert len(composition(cfg, cells, assignments, "grid64")["layers"]) == 512
    with pytest.raises(ConfigError, match="needs 513 layers including 64 reserved for PVW; the bus's max_layers is 512"):
        check_assignments(cfg, cells, assignments[:-1] + ["full"], 512)
    with pytest.raises(ConfigError, match="unknown scene"):
        check_assignments(cfg, cells, ["missing"] + [None] * 7, 512)


def test_bus_budget_defaults_to_max_compositor_layers(cfg):
    # A frame draws only its own layers; the budget costs 128 bytes of layer table per layer.
    assert parse_aux_buses([bus_json()], cfg)[0].max_layers == 640
    assert parse_aux_buses([bus_json()], replace(cfg, max_compositor_layers=256))[0].max_layers == 256
    pages = parse_aux_buses([pages_json()], cfg)[0]
    assert (pages.max_layers, pages.layouts) == (640, (pages.layout,))
    assert parse_aux_buses([bus_json(max_layers=100, scenes=[])], cfg)[0].max_layers == 100
    with pytest.raises(ConfigError, match="max_layers is 100"):
        parse_aux_buses([bus_json(max_layers=100, scenes=["grid64", "grid64"])], cfg)
    for invalid in (0, 1.5, True):
        with pytest.raises(ConfigError, match="max_layers"):
            parse_aux_buses([bus_json(max_layers=invalid)], cfg)


def test_bus_validation_and_distinct_outputs(cfg):
    buses = parse_aux_buses([bus_json(), bus_json(id="second", renditions=[{"id": "monitor", "port": 5012}]),
                             pages_json(id="third", renditions=[{"id": "monitor", "port": 5014}])], cfg)
    assert [b.id for b in buses] == ["multiview", "second", "third"]
    assert buses[0].renditions[0].fps == 30 and buses[0].renditions[0].codec == "h264_nvenc"
    # The presets by default, the initial layout first; the grid only beside a layout with a pgm cell.
    assert [l.get("preset") for l in buses[0].layouts] == ["pgm_pvw_grid", "source_pages"]
    assert [l.get("preset") for l in buses[2].layouts] == ["source_pages"]
    assert buses[2].scenes == (None,) * 0 and buses[0].scenes == ("full",) * 8
    assert parse_aux_buses([bus_json(scenes=["full"])], cfg)[0].scenes == ("full",) + (None,) * 7
    custom = parse_aux_buses([bus_json(layout={"cells": CELLS}, scenes=[None] * 3)], cfg)[0]
    assert (custom.layout, len(custom.layouts)) == ({"cells": CELLS}, 3)
    for invalid in (bus_json(id="bad name"), bus_json(scenes=["missing"] * 8), bus_json(scenes="full"),
                    bus_json(layouts={"preset": "source_pages"}), bus_json(layouts=[{"preset": "mosaic"}]),
                    bus_json(renditions=[{"id": "monitor", "port": 5010, "color": "hlg"}])):
        with pytest.raises((ConfigError, ValueError)):
            parse_aux_buses([invalid], cfg)
    with pytest.raises(ConfigError, match="port"):
        parse_aux_buses([bus_json(), bus_json(id="second")], cfg)
    with pytest.raises(ConfigError, match="unique"):
        parse_aux_buses([bus_json(), bus_json(renditions=[{"id": "monitor", "port": 5012}])], cfg)


def test_program_pad_only_for_buses_whose_layouts_draw_it(cfg):
    many = replace(cfg, sources=tuple(Source(f"s{i}", "video", f"c{i}.mp4", 1920, 1080) for i in range(128)))
    assert parse_aux_buses([pages_json()], many)
    with pytest.raises(ConfigError, match="128"):
        parse_aux_buses([pages_json(layouts=[{"preset": "pgm_pvw_grid"}])], many)   # listed: it needs the PGM pad
    pages, grid = make_bus(cfg, pages_json()), make_bus(cfg, bus_json())
    assert (pages.pgm_edge, pages.inputs()[-1], pages.pgm_delay_frames) == (None, "aux_sources_source_63", 0)
    assert (grid.inputs()[-1], grid.pgm_delay_frames) == ("aux_multiview_pgm", 1)
    with pytest.raises(ConfigError, match="no PGM pad for a pgm cell"):
        pages.set_layout({"layout": {"preset": "pgm_pvw_grid"}})


def test_bus_timing_options_parse_and_validate(cfg):
    bus = parse_aux_buses([bus_json()], cfg)[0]
    assert (bus.pvw_align, bus.latency_ms, bus.pgm_delay_frames, bus.full_rate) == ("program", None, 1, False)
    chosen = parse_aux_buses([bus_json(pvw_align="pgm_tile", latency_ms=66.7, pgm_delay_frames=0, full_rate=True)], cfg)[0]
    assert (chosen.pvw_align, chosen.latency_ms, chosen.pgm_delay_frames, chosen.full_rate) == ("pgm_tile", 66.7, 0, True)
    assert chosen.renditions[0].fps == 60   # full rate: the rendition runs at the canvas rate
    assert parse_aux_buses([bus_json(full_rate=True)], cfg)[0].pgm_delay_frames == 2
    assert [default_pgm_delay_frames(aux_fps(fps, full)) for fps, full in ((60, False), (60, True), (50, True), (30, True), (25, False))] == [1, 2, 2, 1, 1]
    pages = parse_aux_buses([pages_json(full_rate=True, latency_ms=40, pvw_align="pgm_tile")], cfg)[0]
    assert (pages.full_rate, pages.latency_ms, pages.renditions[0].fps, pages.pvw_align) == (True, 40, 60, "pgm_tile")
    for invalid in (bus_json(pvw_align="nearest"), bus_json(latency_ms=0), bus_json(latency_ms="50"),
                    bus_json(latency_ms=True), bus_json(pgm_delay_frames=7), bus_json(pgm_delay_frames=1.0),
                    bus_json(pgm_delay_frames=True), bus_json(full_rate=1),
                    bus_json(renditions=[{"id": "monitor", "port": 5010, "fps": 60}])):   # full rate not asked for
        with pytest.raises(ConfigError):
            parse_aux_buses([invalid], cfg)


def test_aux_monitors_default_to_one_reference_frame(cfg):
    assert parse_aux_buses([bus_json()], cfg)[0].renditions[0].dpb_size == 1
    custom = bus_json(renditions=[{"id": "monitor", "port": 5010, "dpb_size": 0}])
    assert parse_aux_buses([custom], cfg)[0].renditions[0].dpb_size == 0


def test_assignment_conflict_sends_one_layout(cfg):
    avp = FakeAvp()
    bus = make_bus(cfg, bus_json(scenes=[]), avp)
    reachable(avp, bus)
    revision = bus.revision
    first = {"expected_revision": revision, "scenes": ["full"] + [None] * 7}
    second = {"expected_revision": revision, "scenes": [None, "repeat"] + [None] * 6}
    results = []
    threads = [threading.Thread(target=lambda req=req: results.append(bus.assign(req))) for req in (first, second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(avp.layouts) == 1 and avp.layouts[0]["revision"] == bus.revision != revision
    assert sum(bool(r.get("conflict")) for r in results) == 1
    before = bus.revision, list(bus.scenes)
    for invalid in (["missing"] + [None] * 7, ["full"] * 7, "full"):
        with pytest.raises(ConfigError):
            bus.assign({"expected_revision": bus.revision, "scenes": invalid})
    assert (bus.revision, bus.scenes) == before
    assert bus.assign({"scenes": [None] * 8})["conflict"]
    # A new instance never accepts the previous one's revision.
    assert make_bus(cfg, bus_json()).assign({"expected_revision": bus.revision, "scenes": [None] * 8})["conflict"]


def test_follower_gets_preview_layers_and_base_of_the_current_layout(cfg):
    avp = FakeAvp()
    bus = make_bus(cfg, bus_json(), avp)
    reachable(avp, bus)
    scenes = ["repeat"] + [None] * 7
    bus.assign({"expected_revision": bus.revision, "scenes": scenes})
    layout, = avp.layouts
    assert layout == {"revision": bus.revision, "pvw": pvw_layouts(cfg, grid_cells(cfg)),
                      "base": base_composition(cfg, grid_cells(cfg), scenes)}


def test_runtime_layout_switch_keeps_assignments_by_slot(cfg):
    avp = FakeAvp()
    bus = make_bus(cfg, bus_json(scenes=["full", "repeat", "full"] + [None] * 5, layouts=[{"cells": CELLS}]), avp)
    reachable(avp, bus)
    result = bus.set_layout({"layout": {"cells": CELLS}})
    assert result["layout"] == {"cells": CELLS} and result["scenes"] == ["full", "repeat", "full"] + [None] * 5
    sent = avp.layouts[-1]
    assert sent["revision"] == bus.revision
    assert sent["base"] == base_composition(cfg, CELLS, ["full", "repeat", "full"])
    assert all(not l["layers"] for l in sent["pvw"].values())   # no pvw cell: the preview draws nothing
    state = bus.state()
    assert [c["role"] for c in state["cells"]] == ["slot", "slot", "source", "slot", "pgm"]
    assert state["cells"][2] == {**CELLS[2], "id": "s63", "kind": "video"}
    # Three slots now: an assignment still lists every slot kept, the hidden ones too.
    bus.assign({"expected_revision": bus.revision, "scenes": [None, "full", None] + [None] * 5})
    assert avp.layouts[-1]["base"]["layers"][0]["input"] == 63
    # Pages: no slots, the assignments wait for a layout that has them.
    bus.set_layout({"layout": {"preset": "source_pages"}})
    assert bus.state()["page"] == 0 and avp.layouts[-1]["base"] == base_composition(cfg, layout_cells(cfg, bus.layout), [])
    with pytest.raises(ConfigError, match="no scene slots"):
        bus.assign({"expected_revision": bus.revision, "scenes": [None] * 8})
    bus.set_layout({"layout": {"preset": "pgm_pvw_grid"}})
    assert bus.scenes == [None, "full"] + [None] * 6
    for invalid in ({"layout": {"preset": "mosaic"}}, {}, {"layout": {"cells": [{**CELLS[0], "x": 1}]}}):
        with pytest.raises(ConfigError):
            bus.set_layout(invalid)


def test_layout_over_the_budget_is_refused_and_changes_nothing(cfg):
    avp = FakeAvp()
    bus = make_bus(cfg, bus_json(scenes=["grid64"] * 3, max_layers=300), avp)
    reachable(avp, bus)
    pvw = [{"role": "pvw", "x": 0, "y": 0, "w": 540, "h": 960}, {"role": "pvw", "x": 540, "y": 0, "w": 540, "h": 960}]
    # Two PVW cells reserve 128, slots 0 and 1 hold 64 each: 256 fit; slot 2 makes it 320.
    bus.set_layout({"layout": {"cells": [*pvw, CELLS[1], CELLS[3]]}})
    before = bus.layout, bus.revision, len(avp.layouts)
    with pytest.raises(ConfigError, match="needs 320 layers including 128 reserved for PVW; the bus's max_layers is 300"):
        bus.set_layout({"layout": {"cells": [*pvw, CELLS[1], CELLS[3], CELLS[0]]}})
    assert (bus.layout, bus.revision, len(avp.layouts)) == before


def test_status_is_pending_until_the_compositor_draws_the_revision(cfg):
    avp = FakeAvp()
    bus = make_bus(cfg, bus_json(), avp)
    assert bus.state()["composition_pending"] and bus.state()["suspended"]   # nothing reachable yet
    reachable(avp, bus)
    assert not bus.state()["composition_pending"]
    bus.assign({"expected_revision": bus.revision, "scenes": [None] * 8})
    assert bus.state()["composition_pending"]   # the follower has not set it yet
    avp.statuses[bus.node_name]["composition_revision"] = bus.revision
    avp.statuses[bus.node_name]["composition_pending"] = True   # staged until its inputs have frames
    assert bus.state()["composition_pending"]
    avp.statuses[bus.node_name]["composition_pending"] = False
    assert not bus.state()["composition_pending"]
    # A staged composition the compositor dropped keeps the old revision: pending, with its error.
    previous = bus.revision
    bus.assign({"expected_revision": bus.revision, "scenes": ["full"] * 8})
    avp.statuses[bus.node_name].update(composition_revision=previous, composition_error="Aux inputs not ready")
    state = bus.state()
    assert state["composition_pending"] and state["composition_error"] == "Aux inputs not ready"


def test_commands_route_any_number_of_buses_by_id(cfg):
    avp = FakeAvp()
    specs = [bus_json(id=f"bus{i}", renditions=[{"id": "monitor", "port": 5010 + 2 * i}]) for i in range(3)]
    buses = AuxBuses(avp, [AuxBus(avp, None, SimpleNamespace(add_aux_destination=lambda *a: None, latency_ms=50),
                                  cfg, spec) for spec in parse_aux_buses(specs, cfg)])
    for bus in buses:
        reachable(avp, bus)
    buses.register_commands()
    assert set(avp.handlers) == {"mixer.aux", "mixer.aux_layout", "mixer.aux_page", "mixer.aux_status"}
    assert avp.command("mixer.aux_layout", bus="bus2", layout={"preset": "source_pages"})["page"] == 0
    assert avp.command("mixer.aux_page", bus="bus2", page=3)["layout"]["page"] == 3
    with pytest.raises(ConfigError, match="no source pages"):
        avp.command("mixer.aux_page", bus="bus0", page=1)
    with pytest.raises(ConfigError, match="Unknown aux bus"):
        avp.command("mixer.aux", bus="bus3", scenes=[])
    status = json.loads(avp.handlers["mixer.aux_status"](""))
    assert [b["id"] for b in status] == ["bus0", "bus1", "bus2"]
    grid, _, pages = status
    assert (grid["layout"], [c["role"] for c in grid["cells"]][:2]) == ({"preset": "pgm_pvw_grid"}, ["pvw", "pgm"])
    assert [l.get("preset") for l in grid["layouts"]] == ["pgm_pvw_grid", "source_pages"]
    assert (pages["page"], pages["pages"], pages["first"], pages["total"]) == (3, 6, 37, 64)
    assert pages["cells"][0] == {"role": "source", "source": 36, "id": "s36", "kind": "video", **page_grid(cfg)[0]}
    assert (grid["fps"], grid["latency_ms"], grid["pvw_align"], grid["pgm_delay_frames"], grid["max_layers"]) == (30, 50, "program", 1, 640)


def test_page_commands_show_and_step_pages(cfg):
    avp = FakeAvp()
    bus = make_bus(cfg, pages_json(), avp)
    reachable(avp, bus)
    state = bus.turn({"page": 2})
    assert (state["page"], state["first"], state["layout"]["page"]) == (2, 25, 2)
    assert avp.layouts[-1]["base"]["layers"][0]["input"] == 24
    assert bus.turn({"step": -3})["page"] == 5
    assert bus.turn({"step": 1})["page"] == 0
    count = len(avp.layouts)
    for invalid in ({"page": 6}, {"page": True}, {"step": "1"}, {"auto": True}, {}):
        with pytest.raises(ConfigError):
            bus.turn(invalid)
    assert len(avp.layouts) == count


def test_one_scheduler_resends_layouts_for_every_bus(cfg, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("pyplumber.mixer.aux.time.monotonic", lambda: clock[0])
    avp = FakeAvp()
    specs = [pages_json(), bus_json(renditions=[{"id": "monitor", "port": 5016}])]
    mixer = SimpleNamespace(add_aux_destination=lambda *a: None, latency_ms=50)
    buses = AuxBuses(avp, [AuxBus(avp, None, mixer, cfg, spec) for spec in parse_aux_buses(specs, cfg)])
    pages, grid = buses
    for bus in buses:
        reachable(avp, bus)
    assert buses.tick() == 1.0   # nothing due before the follower check
    clock[0] += 50
    assert buses.tick() == pytest.approx(1.0)
    assert pages.layout["page"] == 0 and avp.layouts == []   # pages turn only by hand
    avp.statuses[grid.follower]["layout_revision"] = "built"   # restarted: holds its build-time layout
    clock[0] += 1
    buses.tick()
    assert avp.layouts == [grid.layout_object()]


def test_scheduler_survives_an_unreachable_follower(cfg, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("pyplumber.mixer.aux.time.monotonic", lambda: clock[0])
    avp = FakeAvp()
    buses = AuxBuses(avp, [make_bus(cfg, pages_json(), avp)])
    bus, = buses
    bus.turn({"step": 1})
    clock[0] += 1
    assert 0.05 <= buses.tick() <= 1.0
    assert avp.layouts == []   # dropped by the binding, resent once reachable
    assert bus.state()["follower"]["error"]
    reachable(avp, bus)
    avp.statuses[bus.follower]["layout_revision"] = "built"
    clock[0] += 1
    buses.tick()
    assert avp.layouts[-1]["revision"] == bus.revision


def test_bus_builds_its_nodes_in_one_group(cfg, monkeypatch):
    import pyplumber.mixer.janus as janus
    nodes = []
    monkeypatch.setattr(janus, "build_janus_output", lambda *a, **kw: "listener")
    avp = SimpleNamespace(addNode=lambda node, **kw: nodes.append(node), edges=SimpleNamespace(planCapacity=lambda *a: None))
    api = SimpleNamespace(FilterVideo=lambda params: SimpleNamespace(parameters=params))
    mixer = SimpleNamespace(add_aux_destination=lambda *a: None, latency_ms=50, name="mixer",
                            canvas_compositor=lambda params, api: SimpleNamespace(parameters=params),
                            backend=SimpleNamespace(graph_threads=1, conversion=lambda *a, **kw: "graph"))
    options = SimpleNamespace(janus_host="127.0.0.1", janus_video_ssrc=1, janus_rtcp_bind="")
    bus = AuxBus(avp, api, mixer, cfg, parse_aux_buses([bus_json(scenes=["repeat"])], cfg)[0])
    bus.build(options)
    compositor, converted, follower = (n.parameters for n in nodes)
    assert {p["group"] for p in (compositor, converted, follower)} == {"aux_multiview"}
    assert compositor["src"] == compositor["subscriptions"] == [*bus.edges, "aux_multiview_pgm"]
    assert (compositor["max_layers"], compositor["pgm_delay_frames"]) == (640, 1)
    assert compositor["layers"] == follower["layout"]["base"]["layers"]
    assert (follower["type"], follower["name"], follower["compositor"]) == ("mixer_pvw_follow", "aux_multiview_pvw", "aux_multiview_comp")
    assert (follower["fps"], follower["latency_ms"], follower["main_latency_ms"], follower["align"]) == ("30", 50, 50, "program")
    assert follower["layout"] == bus.layout_object()


class _Built(Exception):
    pass


def _build_until_the_graph(cfg, spec, latency_ms):
    """The checks precede the first graph call, which ends the build here."""
    avp = SimpleNamespace(edges=SimpleNamespace(planCapacity=lambda *_: (_ for _ in ()).throw(_Built())))
    make_bus(cfg, spec, avp, latency_ms).build(None)


def test_program_frame_must_reach_the_bus_before_its_deadline(cfg):
    """At the main latency the program frame leaves its compositor when a pgm_delay_frames 0 bus
    would draw it; the PGM pad needs a tick of delay or a longer bus buffer, at least one program
    frame of margin. A bus that never draws the program is not held to it."""
    for options in ({"pgm_delay_frames": 0}, {"full_rate": True, "pgm_delay_frames": 0},
                    {"pgm_delay_frames": 0, "latency_ms": 51}, {"pgm_delay_frames": 0, "latency_ms": 66}):
        with pytest.raises(ConfigError, match=r"pgm_delay_frames .* by a program frame \(16.7 ms\)"):
            _build_until_the_graph(cfg, bus_json(**options), 50)
    for spec in (bus_json(), bus_json(pgm_delay_frames=0, latency_ms=83.4), bus_json(pgm_delay_frames=0, latency_ms=66.7),
                 bus_json(full_rate=True), bus_json(full_rate=True, pgm_delay_frames=1),
                 pages_json(pgm_delay_frames=0, latency_ms=20)):
        with pytest.raises(_Built):
            _build_until_the_graph(cfg, spec, 50)


@pytest.mark.parametrize("spec,limit", [(bus_json(), 200), (pages_json(), 240)])
def test_latency_budget_counts_the_pgm_delay(cfg, spec, limit):
    """At 25 fps six ticks are 240 ms, and a bus's PGM input is held one tick more."""
    cfg25 = replace(cfg, fps=25)
    for latency, error in ((limit, _Built), (limit + 1, ConfigError)):
        with pytest.raises(error):
            _build_until_the_graph(cfg25, {**spec, "latency_ms": latency}, 50)


def test_bus_latency_defaults_to_the_main_mixer(cfg):
    # 60 fps program, 30 fps aux: the main mixer's buffer (three 60 fps ticks, 1.5 aux ticks), so
    # the pvw cell can leave with the program; the sources reach both at the same time.
    assert make_bus(cfg, bus_json()).latency_ms() == default_latency_ms(60) == 50
    cfg25 = replace(cfg, fps=25)
    for latency, expected in ((default_latency_ms(25), 80), (120, 120), (20, 20)):
        assert make_bus(cfg25, pages_json(), latency_ms=latency).latency_ms() == expected
    assert make_bus(cfg, bus_json(latency_ms=70)).latency_ms() == 70   # the bus's own wins


def test_a_bus_without_the_pgm_pad_draws_any_layout_but_pgm_cells(cfg):
    """No layout of a source_pages bus asks for the program: no PGM pad, yet at runtime it draws any
    layout without a pgm cell and refuses one with it. Listing the grid gives it the pad."""
    avp = FakeAvp()
    bus = make_bus(cfg, pages_json(), avp)
    reachable(avp, bus)
    no_pgm = [c for c in CELLS if c["role"] != "pgm"]
    result = bus.set_layout({"layout": {"cells": no_pgm}})
    assert (result["layout"], result["layouts"], result["scenes"]) == ({"cells": no_pgm}, [bus.bus.layout], [None] * 3)
    bus.assign({"expected_revision": bus.revision, "scenes": [None, "full", None]})
    assert [l["input"] for l in avp.layouts[-1]["base"]["layers"]] == [63, 0]   # cells in list order
    before = bus.layout, bus.revision, len(avp.layouts)
    for layout in ({"preset": "pgm_pvw_grid"}, {"cells": CELLS}):
        with pytest.raises(ConfigError, match="no PGM pad for a pgm cell"):
            bus.set_layout({"layout": layout})
    assert (bus.layout, bus.revision, len(avp.layouts)) == before
    cells = make_bus(cfg, bus_json(layout={"cells": no_pgm}, scenes=[]))
    assert (cells.pgm_edge, [l.get("preset") for l in cells.bus.layouts]) == (None, [None, "source_pages"])
    listed = make_bus(cfg, pages_json(layouts=[{"preset": "pgm_pvw_grid"}]))
    assert (listed.inputs()[-1], listed.pgm_delay_frames) == ("aux_sources_pgm", 1)
    assert [l.get("preset") for l in listed.bus.layouts] == ["source_pages", "pgm_pvw_grid"]
