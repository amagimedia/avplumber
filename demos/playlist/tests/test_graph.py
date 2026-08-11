"""Graph-plan regression: exact node chain per worker, the switcher/realtime
output, the never-black invariant, and the cut command ORDER (unpause before
flip).  These lock the AVP command surface the playlist depends on."""
import json

from playlist import (Clip, ElementMode, contains_black_source, cut_commands,
                      pause_node, plan_output_nodes, plan_worker_nodes,
                      worker_out_edge)
from helpers import clips


def _clip():
    return Clip(url="/media/a.mp4", name="a")


def test_worker_chain_is_replay_shaped():
    specs = plan_worker_nodes(0, _clip(), fps=30, width=1920, height=1080)
    types = [s.type for s in specs]
    assert types == ["input_rec", "demux", "dec_video", "speed_video",
                     "rescale_video", "force_fps", "pause"]


def test_worker_chain_edges_are_connected():
    specs = plan_worker_nodes(1, _clip(), 30, 1920, 1080)
    # every src (except the first input) is some upstream node's dst
    dsts = {s.params.get("dst") for s in specs}
    dsts.update(v for s in specs for v in
                (s.params.get("routing") or {}).values())
    for s in specs[1:]:
        src = s.params["src"]
        assert src in dsts, f"{s.name} src {src} dangling"


def test_worker_ends_on_shared_out_edge_paused():
    specs = plan_worker_nodes(1, _clip(), 30, 1920, 1080)
    last = specs[-1]
    assert last.type == "pause"
    assert last.name == pause_node(1)
    assert last.params["dst"] == worker_out_edge(1)
    assert last.params["paused"] is True          # preloaded workers start frozen


def test_force_fps_normalizes_output_cadence():
    specs = plan_worker_nodes(0, _clip(), fps=25, width=1280, height=720)
    ff = next(s for s in specs if s.type == "force_fps")
    assert ff.params["fps"] == "25/1"
    sc = next(s for s in specs if s.type == "rescale_video")
    assert (sc.params["width"], sc.params["height"]) == (1280, 720)


def test_play_from_maps_to_preseek():
    c = Clip(url="/m/a.mp4", play_from_ms=2000)
    specs = plan_worker_nodes(0, c, 30, 1920, 1080)
    inp = specs[0]
    assert inp.params["preseek"] == 2.0


def test_loop_self_sets_input_loop():
    c = Clip(url="/m/a.mp4", element_mode=ElementMode.LOOP_SELF)
    specs = plan_worker_nodes(0, c, 30, 1920, 1080)
    assert specs[0].params["loop"] is True
    c2 = Clip(url="/m/a.mp4", element_mode=ElementMode.PLAY_TO_END)
    specs2 = plan_worker_nodes(0, c2, 30, 1920, 1080)
    assert specs2[0].params["loop"] is False


def test_output_has_switcher_and_realtime_setpts():
    specs = plan_output_nodes(worker_count=2, fps=30, active=0)
    sw = next(s for s in specs if s.type == "source_switcher")
    assert sw.params["src"] == [worker_out_edge(0), worker_out_edge(1)]
    rt = next(s for s in specs if s.type.startswith("realtime"))
    assert rt.params["set_pts"] is True
    assert rt.params["tick_period"] == "1/30"


def test_no_black_source_in_graph():
    """Hard invariant: never a sentinel, never a black fallback."""
    specs = (plan_worker_nodes(0, _clip(), 30, 1920, 1080)
             + plan_worker_nodes(1, _clip(), 30, 1920, 1080)
             + plan_output_nodes(2, 30, 0))
    assert contains_black_source(specs) is False


def test_black_source_detector_trips_on_sentinel():
    from playlist import NodeSpec
    assert contains_black_source([NodeSpec("sentinel_video", "s", "g", {})])


def test_black_source_detector_trips_on_fallback():
    from playlist import NodeSpec
    assert contains_black_source(
        [NodeSpec("source_switcher", "sw", "g", {"fallback_active": 0})])


def test_cut_unpauses_before_flip():
    cmds = cut_commands(1)
    assert len(cmds) == 2
    assert cmds[0] == "resume worker1_pauseteam"
    assert "active" in cmds[1]
    # ordering: the unpause must come strictly before the active flip
    assert cmds.index(cmds[0]) < cmds.index(cmds[1])


def test_cut_scheduled_uses_timeline():
    cmds = cut_commands(1, at_pts_ms=12000)
    assert cmds[1].startswith("timeline.set ")
    payload = json.loads(cmds[1][len("timeline.set "):])
    assert payload["at"] == 12000 and payload["val"] == 1
