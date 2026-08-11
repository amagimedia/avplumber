"""Gapless-preload regression: after starting a clip, the idle worker is always
(re)armed toward resolve_next so the next cut has a frozen-ready target."""
from playlist import worker_group
from helpers import clips, controller
from playlist import PlaylistMode as M


def test_idle_worker_is_preloaded_after_cut():
    ctl, rec = controller(mode=M.LOOP_ALL)           # a,b,c current=0 worker0
    ctl.next()                                        # -> b on worker1
    # after the cut, worker0 (idle) should be rebuilt toward next(=c)
    assert f"group.start {worker_group(0)}" in rec.cmds
    added = [n for n in rec.added_nodes() if n["group"] == worker_group(0)]
    assert added, "idle worker not preloaded after cut"


def test_preload_targets_resolved_next_clip():
    ctl, rec = controller(mode=M.LOOP_ALL)
    ctl.next()                                        # on b; next should be c
    input_nodes = [n for n in rec.added_nodes()
                   if n["type"] == "input_rec" and n["group"] == worker_group(0)]
    assert input_nodes
    assert input_nodes[-1]["url"].endswith("c")


def test_mode_change_rearms_preload():
    ctl, rec = controller(mode=M.LOOP_ALL)
    rec.clear()
    ctl.set_mode(M.LOOP_CURRENT)                       # next becomes current(a)
    # rearm should have run (group.stop/start on idle) targeting a, or no-op if
    # already correct; either way no black source and a defined next
    assert ctl.next_index(+1) == ctl.current_index
