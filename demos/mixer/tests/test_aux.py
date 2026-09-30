from dataclasses import replace
import json
from types import SimpleNamespace
import threading

import pytest

from pyplumber.mixer.aux import (AuxMultiview, AuxSourcePages, aux_fps, base_composition, composition,
                                 multiview_cells, page_composition, page_grid, parse_aux_buses, pvw_layouts,
                                 register_aux_commands, validate_assignments)
from pyplumber.mixer.config import default_latency_ms
from pyplumber.mixer.config import AuxBus, ConfigError, Item, MixerConfig, Rect, Rendition, Scene, Source


def object_set(command):
    """(node, key, value) of a ``node.object.set`` line."""
    node, key, value = command.split(" ", 3)[1:]
    return node, key, json.loads(value)


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
    assert aux_fps(program, full_rate=True) == program   # opt-in: the canvas rate, twice the work at 50/60


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
        node, key, value = object_set(command)
        assert (node, key) == ("aux_multiview_pvw", "base")
        published.append(value)
    avp = SimpleNamespace(executeCommandsFromString=execute,
                          node=lambda _name: SimpleNamespace(getObject=lambda _key: {}))   # follower reachable
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
    mixer = SimpleNamespace(add_aux_destination=lambda *args: None, latency_ms=None)
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
    # The timing a script needs to read the follower's numbers: the bus rate and buffer.
    assert (grid["fps"], grid["latency_ms"], grid["pvw_align"], grid["pgm_delay_frames"]) == (30, 50, "program", 1)
    assert (pages["fps"], pages["latency_ms"]) == (30, 50) and "pvw_align" not in pages


class StatusAvp:
    """Records compositions and follower bases; the test sets what the compositor's status
    reports and the follower's status (``follower``, None while that node is unreachable).
    Like the binding, a command to an unreachable node is logged and dropped, never raised;
    only ``node()`` raises."""

    def __init__(self):
        self.published, self.bases = [], []
        self.status, self.follower = {"suspended": False, "pvw_scene": ""}, None

    def executeCommandsFromString(self, command):
        node, key, value = object_set(command)
        assert key == ("base" if node.endswith("_pvw") else "composition")
        if node.endswith("_pvw"):
            if self.follower is not None:
                self.bases.append(value)
        else:
            self.published.append(value)

    def node(self, name):
        if name.endswith("_pvw") and self.follower is None:
            raise Exception(f"Node {name} doesn't exist.")
        status = self.follower if name.endswith("_pvw") else self.status
        return SimpleNamespace(getObject=lambda _key: dict(status))


def test_encoder_backpressure_suspension_survives_automatic_updates_only(cfg, monkeypatch):
    """Following PVW (here without the follower node, so this thread does it) and rotating pages
    keep a suspended bus suspended; an operator's slot assignment or page turn applies a
    composition without ``enabled``, which resumes it."""
    mixer = SimpleNamespace(add_aux_destination=lambda *args: None)
    avp = StatusAvp()
    grid = AuxMultiview(avp, None, mixer, cfg, parse_aux_buses([bus_json()], cfg)[0])
    avp.status = {"suspended": True, "pvw_scene": "repeat"}
    waits = iter([True])                        # one pass of the PVW poll
    grid.stopped = SimpleNamespace(wait=lambda _s: next(waits))
    grid.run()
    assert avp.published == [{**composition(cfg, ["full"] * 8, "repeat"), "enabled": False}]
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
    # 60 fps program, 30 fps aux: the main mixer's buffer (three 60 fps ticks, 1.5 aux ticks), so
    # the PVW tile can leave with the program; the sources reach both at the same time.
    assert view.latency_ms() == pages.latency_ms() == view.main_latency_ms() == default_latency_ms(60) == 50
    cfg25 = replace(cfg, fps=25)
    for latency, expected in ((None, 80), (120, 120), (20, 20)):
        mixer.latency_ms = latency
        assert AuxMultiview(None, None, mixer, cfg25, parse_aux_buses([bus_json()], cfg25)[0]).latency_ms() == expected
    # A bus's own latency_ms wins over the main mixer's.
    mixer.latency_ms = 50
    assert AuxMultiview(None, None, mixer, cfg, parse_aux_buses([bus_json(latency_ms=70)], cfg)[0]).latency_ms() == 70
    assert view.inputs()[-1] == view.pgm_edge   # the delay applies to the last input


def test_bus_timing_options_parse_and_validate(cfg):
    bus = parse_aux_buses([bus_json()], cfg)[0]
    assert (bus.pvw_align, bus.latency_ms, bus.pgm_delay_frames, bus.full_rate) == ("program", None, 1, False)
    chosen = parse_aux_buses([bus_json(pvw_align="pgm_tile", latency_ms=66.7, pgm_delay_frames=0, full_rate=True)], cfg)[0]
    assert (chosen.pvw_align, chosen.latency_ms, chosen.pgm_delay_frames, chosen.full_rate) == ("pgm_tile", 66.7, 0, True)
    assert chosen.renditions[0].fps == 60   # full rate: the rendition runs at the canvas rate
    assert parse_aux_buses([bus_json(renditions=[{"id": "monitor", "port": 5010, "fps": 60}], full_rate=True)], cfg)
    pages = parse_aux_buses([pages_json(full_rate=True, latency_ms=40)], cfg)[0]
    assert (pages.full_rate, pages.latency_ms, pages.renditions[0].fps) == (True, 40, 60)
    for invalid in (bus_json(pvw_align="nearest"), bus_json(latency_ms=0), bus_json(latency_ms="50"),
                    bus_json(latency_ms=True), bus_json(pgm_delay_frames=7), bus_json(pgm_delay_frames=1.0),
                    bus_json(pgm_delay_frames=True), bus_json(full_rate=1),
                    bus_json(renditions=[{"id": "monitor", "port": 5010, "fps": 60}]),   # full rate not asked for
                    pages_json(pvw_align="program"), pages_json(pgm_delay_frames=0)):
        with pytest.raises(ConfigError):
            parse_aux_buses([invalid], cfg)


class _Built(Exception):
    pass


def test_program_frame_must_reach_the_bus_before_its_deadline(cfg):
    """At the main latency the program frame leaves its compositor when a pgm_delay_frames 0 bus
    would draw it; the PGM pad needs a tick of delay or a longer bus buffer. The checks precede
    the first graph call, which ends the build here."""
    avp = SimpleNamespace(edges=SimpleNamespace(planCapacity=lambda *_: (_ for _ in ()).throw(_Built())))
    mixer = SimpleNamespace(add_aux_destination=lambda *args: None, latency_ms=50, name="mixer")
    def build(**options):
        AuxMultiview(avp, None, mixer, cfg, parse_aux_buses([bus_json(**options)], cfg)[0]).build(None)
    for options in ({"pgm_delay_frames": 0}, {"full_rate": True, "pgm_delay_frames": 0}):
        with pytest.raises(ConfigError, match="pgm_delay_frames"):
            build(**options)
    for options in ({}, {"pgm_delay_frames": 0, "latency_ms": 83.4},   # one aux tick more than the program
                    {"full_rate": True}):                            # 16.7 + 50 > 50: the delay covers it
        with pytest.raises(_Built):
            build(**options)


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


def test_pvw_layouts_and_base_split_the_composition(cfg):
    """The follower node sets pvw[shown] followed by base; composition() is exactly that."""
    layouts = pvw_layouts(cfg)
    reserved = max(len(s.items) for s in cfg.scenes)
    assert set(layouts) == {"full", "repeat", "grid64"}
    assert [l["z"] for l in layouts["repeat"]["layers"]] == [0, 1]
    assert all(l["z"] < reserved for layout in layouts.values() for l in layout["layers"])
    assert layouts["grid64"]["layers"][-1]["z"] == reserved - 1
    pvw_cell = multiview_cells(cfg)[0]
    assert layouts["full"]["layers"][0]["tile"] == {k: pvw_cell[k] for k in ("x", "y", "w", "h")}
    assert layouts["full"]["active_inputs"] == 1
    assignments = ["repeat"] * 8
    base = base_composition(cfg, assignments)
    assert min(l["z"] for l in base["layers"]) == reserved
    assert base["layers"][-1]["input"] == 64 and base["layers"][-1]["z"] == reserved + 16
    merged = composition(cfg, assignments, "full")
    assert merged["layers"] == layouts["full"]["layers"] + base["layers"]
    assert merged["active_inputs"] == "1" + "0" * 63 + "1"
    assert composition(cfg, assignments, "") == base
    with pytest.raises(ConfigError, match="unknown scene"):
        composition(cfg, assignments, "missing")


def test_multiview_builds_a_follower_holding_the_layouts(cfg, monkeypatch):
    import pyplumber.mixer.aux as aux_module
    monkeypatch.setattr(aux_module._AuxOutput, "build", lambda self, options: None)
    nodes = []
    avp = SimpleNamespace(addNode=nodes.append)
    mixer = SimpleNamespace(add_aux_destination=lambda *args: None, latency_ms=None, name="mixer")
    view = AuxMultiview(avp, None, mixer, cfg, parse_aux_buses([bus_json()], cfg)[0])
    view.build(None)
    follower, = nodes
    p = follower.parameters
    assert (p["type"], p["name"]) == ("mixer_pvw_follow", "aux_multiview_pvw")
    assert (p["mixer"], p["compositor"], p["group"]) == ("mixer", "aux_multiview_comp", "aux_multiview")
    assert (p["fps"], p["latency_ms"], p["main_latency_ms"], p["pgm_delay_frames"], p["align"]) == ("30", 50, 50, 1, "program")
    assert (p["auto_restart"], p["on_error"]) == ("off", "off")
    assert p["base"] == {"revision": view.revision, **base_composition(cfg, ["full"] * 8)}
    assert p["pvw"] == pvw_layouts(cfg)
    # A bus with its own timing hands it to the follower and the compositor alike.
    nodes.clear()
    mixer.latency_ms = 60
    bus = parse_aux_buses([bus_json(pvw_align="pgm_tile", latency_ms=80, pgm_delay_frames=2, full_rate=True)], cfg)[0]
    AuxMultiview(avp, None, mixer, cfg, bus).build(None)
    p = nodes[0].parameters
    assert (p["fps"], p["latency_ms"], p["main_latency_ms"], p["pgm_delay_frames"], p["align"]) == ("60", 80, 60, 2, "pgm_tile")


def test_assignment_hands_the_follower_the_base_or_falls_back(cfg):
    mixer = SimpleNamespace(add_aux_destination=lambda *args: None)
    avp = StatusAvp()
    avp.follower = {"base_revision": "stale"}
    grid = AuxMultiview(avp, None, mixer, cfg, parse_aux_buses([bus_json()], cfg)[0])
    scenes = ["repeat"] + [None] * 7
    grid.assign({"expected_revision": grid.revision, "scenes": scenes})
    assert avp.bases == [{"revision": grid.revision, **base_composition(cfg, scenes)}]
    assert avp.published == []
    # Once a second: a follower reporting another base revision (it restarted) gets the current one.
    assert grid._follow() == 1.0
    assert len(avp.bases) == 2 and avp.bases[-1]["revision"] == grid.revision
    avp.follower = {"base_revision": grid.revision, "pvw_scene": "repeat"}
    assert grid._follow() == 1.0 and len(avp.bases) == 2
    assert grid.details()["follower"] == avp.follower
    assert grid.preview == "repeat"   # what the node shows, for a composition set here later
    # Unreachable: this thread follows the preview at 50 ms and sets the composition itself.
    avp.follower = None
    avp.status = {"suspended": False, "pvw_scene": "full"}
    assert grid._follow() == 0.05
    assert avp.published == [{**composition(cfg, scenes, "full"), "enabled": True}]
    assert grid.details()["follower"]["error"]
    # A reassignment then goes to the compositor: the node would drop the base without a word.
    grid.assign({"expected_revision": grid.revision, "scenes": [None] * 8})
    assert avp.published[-1] == composition(cfg, [None] * 8, "full") and len(avp.bases) == 2
