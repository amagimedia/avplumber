from dataclasses import replace
import json
from types import SimpleNamespace
import threading

import pytest

from pyplumber.mixer.aux import (AuxMultiview, AuxSourcePages, aux_fps, composition, multiview_cells,
                                 page_composition, page_grid, parse_aux_buses, register_aux_commands,
                                 validate_assignments)
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


def pages_json(**kwargs):
    return {"id": "sources", "layout": {"preset": "source_pages"},
            "renditions": [{"id": "monitor", "port": 5012}], **kwargs}


def test_source_pages_bus_validation(cfg):
    bus, = parse_aux_buses([pages_json(rotate_s=8)], cfg)
    assert (bus.layout, bus.scenes, bus.rotate_s) == ("source_pages", (), 8.0)
    assert parse_aux_buses([pages_json()], cfg)[0].rotate_s == 5.0
    for invalid in (pages_json(scenes=["full"] * 8), pages_json(layout={"preset": "source_pages", "cols": 3}),
                    pages_json(rotate_s=0.5), pages_json(rotate_s=True), bus_json(layout={"preset": "mosaic"}),
                    bus_json(rotate_s=5)):
        with pytest.raises(ConfigError):
            parse_aux_buses([invalid], cfg)
    # Without a PGM pad a page view takes one source more than a multiview.
    many = replace(cfg, sources=tuple(Source(f"s{i}", "video", f"c{i}.mp4", 1920, 1080) for i in range(128)))
    assert parse_aux_buses([pages_json()], many)
    with pytest.raises(ConfigError, match="128"):
        parse_aux_buses([bus_json()], many)


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


def test_last_page_draws_only_the_remaining_sources(cfg):
    assert page_grid(cfg)[0] == {"x": 2, "y": 10, "w": 536, "h": 300}
    result = page_composition(cfg, 5)
    assert [layer["input"] for layer in result["layers"]] == [60, 61, 62, 63]
    assert result["active_inputs"] == sum(1 << i for i in range(60, 64))


def test_multiview_cells_match_the_composition(cfg):
    cells = multiview_cells(cfg)
    result = composition(cfg, ["full"] * 8, "full")
    assert [layer["tile"] for layer in result["layers"][:-1]] == [
        {k: c[k] for k in ("x", "y", "w", "h")} for c in (cells[0], *cells[2:])]
    pgm = result["layers"][-1]
    assert (pgm["dst_x"], pgm["dst_y"], pgm["dst_w"], pgm["dst_h"]) == tuple(cells[1][k] for k in ("x", "y", "w", "h"))


def _pages(cfg, published):
    avp = SimpleNamespace(executeCommandsFromString=lambda c: published.append(json.loads(c.split(" composition ", 1)[1])),
                          node=lambda _name: SimpleNamespace(getObject=lambda _key: {"suspended": False}))
    mixer = SimpleNamespace(add_aux_destination=lambda *args: None)
    return AuxSourcePages(avp, None, mixer, cfg, parse_aux_buses([pages_json()], cfg)[0])


def test_page_commands_hold_step_and_resume_rotation(cfg):
    published = []
    bus = _pages(cfg, published)
    assert (bus.pages, bus.auto) == (6, True)
    state = bus.turn({"page": 2})
    assert (state["page"], state["auto"], state["first"], state["total"]) == (2, False, 25, 64)
    assert [t["id"] for t in state["tiles"]] == [f"s{i}" for i in range(24, 36)]
    assert published[-1]["layers"][0]["input"] == 24
    assert bus.turn({"step": -3})["page"] == 5
    assert bus.turn({"step": 1})["page"] == 0
    assert bus.turn({"auto": True})["auto"] is True
    count = len(published)
    for invalid in ({"page": 6}, {"page": True}, {"step": "1"}, {"auto": 1}, {}):
        with pytest.raises(ConfigError):
            bus.turn(invalid)
    assert len(published) == count


def test_rotation_advances_pages_until_held(cfg, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("pyplumber.mixer.aux.time.monotonic", lambda: clock[0])
    published = []
    bus = _pages(cfg, published)
    assert bus._tick() == 0.5 and bus.page == 0
    clock[0] += 5
    bus._tick()
    assert bus.page == 1 and len(published) == 1
    bus.turn({"page": 4})
    clock[0] += 50
    assert bus._tick() == 0.5   # a held page is never due: no 50 ms polling
    assert bus.page == 4
    bus.turn({"auto": True})
    clock[0] += 2.5
    bus._tick()
    assert bus.page == 4
    clock[0] += 2.5
    bus._tick()
    assert bus.page == 5


def test_commands_route_by_bus_kind_and_status_reports_geometry(cfg):
    handlers = {}
    avp = SimpleNamespace(registerControlCommand=lambda name, fn, _payload: handlers.__setitem__(name, fn),
                          executeCommandsFromString=lambda command: None)
    mixer = SimpleNamespace(add_aux_destination=lambda *args: None)
    views = (AuxMultiview(avp, None, mixer, cfg, parse_aux_buses([bus_json()], cfg)[0]),
             AuxSourcePages(avp, None, mixer, cfg, parse_aux_buses([pages_json()], cfg)[0]))
    register_aux_commands(avp, views)
    assert json.loads(handlers["mixer.aux_page"](json.dumps({"bus": "sources", "page": 1})))["page"] == 1
    with pytest.raises(ConfigError, match="source pages"):
        handlers["mixer.aux_page"](json.dumps({"bus": "multiview", "page": 1}))
    with pytest.raises(ConfigError, match="scene multiview"):
        handlers["mixer.aux"](json.dumps({"bus": "sources", "scenes": [None] * 8}))
    grid, pages = json.loads(handlers["mixer.aux_status"](""))
    assert (grid["layout"], pages["layout"]) == ("pgm_pvw_grid", "source_pages")
    assert [c["role"] for c in grid["cells"]] == ["pvw", "pgm"] + ["slot"] * 8
    assert pages["tiles"][0] == {"id": "s12", "kind": "video", "x": 2, "y": 10, "w": 536, "h": 300}
    assert pages["canvas"] == {"w": 1080, "h": 1920} and pages["suspended"] is True


class StatusAvp:
    """Records compositions; the test sets what the compositor's status reports."""

    def __init__(self):
        self.published, self.status = [], {"suspended": False, "pvw_scene": ""}

    def executeCommandsFromString(self, command):
        self.published.append(json.loads(command.split(" composition ", 1)[1]))

    def node(self, _name):
        return SimpleNamespace(getObject=lambda _key: dict(self.status))


def test_encoder_backpressure_suspension_survives_automatic_updates_only(cfg, monkeypatch):
    """Following PVW and rotating pages keep a suspended bus suspended; an operator's
    slot assignment or page turn applies a composition without ``enabled``, which resumes it."""
    mixer = SimpleNamespace(add_aux_destination=lambda *args: None)
    avp = StatusAvp()
    grid = AuxMultiview(avp, None, mixer, cfg, parse_aux_buses([bus_json()], cfg)[0])
    avp.status = {"suspended": True, "pvw_scene": "repeat"}
    waits = iter([False, True])                 # one pass of the PVW follower
    grid.stopped = SimpleNamespace(wait=lambda _s: next(waits))
    grid.run()
    assert avp.published[-1] == {**composition(cfg, ["full"] * 8, "repeat"), "enabled": False}
    grid.assign({"expected_revision": grid.revision, "scenes": [None] * 8})
    assert "enabled" not in avp.published[-1]

    clock = [100.0]
    monkeypatch.setattr("pyplumber.mixer.aux.time.monotonic", lambda: clock[0])
    avp = StatusAvp()
    pages = AuxSourcePages(avp, None, mixer, cfg, parse_aux_buses([pages_json()], cfg)[0])
    avp.status = {"suspended": True}
    clock[0] += 5
    pages._tick()
    assert avp.published[-1] == {**page_composition(cfg, 1), "enabled": False}
    pages.turn({"page": 3})
    assert avp.published[-1] == page_composition(cfg, 3)


def test_multiview_shows_pgm_one_tick_late_at_the_main_latency(cfg):
    mixer = SimpleNamespace(add_aux_destination=lambda *args: None, latency_ms=None)
    view = AuxMultiview(None, None, mixer, cfg, parse_aux_buses([bus_json()], cfg)[0])
    pages = AuxSourcePages(None, None, mixer, cfg, parse_aux_buses([pages_json()], cfg)[0])
    # 60 fps program, 30 fps aux: two aux ticks, not the main default of two 60 fps ticks.
    assert view.latency_ms() == pages.latency_ms() == 2000 / 30
    cfg25 = replace(cfg, fps=25)
    for latency, expected in ((None, 80), (120, 120), (20, 80)):
        mixer.latency_ms = latency
        assert AuxMultiview(None, None, mixer, cfg25, parse_aux_buses([bus_json()], cfg25)[0]).latency_ms() == expected
    mixer.latency_ms = None
    assert view.compositor_params() == {"pgm_delay_frames": 1}
    assert view.inputs()[-1] == view.pgm_edge   # the delay applies to the last input
    assert pages.compositor_params() == {}


class _Built(Exception):
    pass


@pytest.mark.parametrize("kind,limit", [(AuxMultiview, 200), (AuxSourcePages, 240)])
def test_latency_budget_counts_the_pgm_delay(cfg, kind, limit):
    """At 25 fps six ticks are 240 ms, and the multiview's PGM input is held one tick more."""
    cfg25 = replace(cfg, fps=25)
    spec = bus_json() if kind is AuxMultiview else pages_json()
    avp = SimpleNamespace(edges=SimpleNamespace(planCapacity=lambda *_: (_ for _ in ()).throw(_Built())))
    for latency, error in ((limit, _Built), (limit + 1, ConfigError)):
        mixer = SimpleNamespace(add_aux_destination=lambda *args: None, latency_ms=latency)
        with pytest.raises(error):
            kind(avp, None, mixer, cfg25, parse_aux_buses([spec], cfg25)[0]).build(None)


def test_aux_monitors_default_to_one_reference_frame(cfg):
    assert parse_aux_buses([bus_json()], cfg)[0].renditions[0].dpb_size == 1
    custom = bus_json(renditions=[{"id": "monitor", "port": 5010, "dpb_size": 0}])
    assert parse_aux_buses([custom], cfg)[0].renditions[0].dpb_size == 0
