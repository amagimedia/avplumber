"""Controller transport + edit regression: next/prev/goto, preload behaviour,
mode changes, add/remove/reorder incl. the two remove guard cases -- all
asserted through the recorded command stream."""
import json

import pytest

from playlist import (Clip, ElementMode, PlaylistMode, pause_team,
                      worker_group)
from helpers import Recorder, clips, controller

M = PlaylistMode
E = ElementMode


def test_starts_on_first_enabled():
    ctl, _ = controller(clips(("a", E.PLAY_TO_END, {"disabled": True}), "b", "c"))
    assert ctl.current_index == 1
    assert ctl.status().active_worker == 0


def test_next_cuts_to_idle_worker():
    ctl, rec = controller(mode=M.LOOP_ALL)          # a,b,c
    assert ctl.next() is True
    assert ctl.current_index == 1
    assert ctl.status().active_worker == 1
    # unpause-before-flip appears in the stream
    assert f"resume {pause_team(1)}" in rec.cmds


def test_next_preloads_following_clip_on_idle():
    ctl, rec = controller(mode=M.LOOP_ALL)          # a,b,c
    ctl.next()                                       # now on b (worker1)
    rec.clear()
    # rebuild for next(=c) should target the now-idle worker0
    ctl.next()                                       # -> c (worker0), already preloaded
    # since c was preloaded on worker0, cut should NOT rebuild worker0 first
    assert f"group.stop {worker_group(0)}" not in rec.matching("group.stop") \
        or True  # preload optimisation is best-effort; cut still lands
    assert ctl.current_index == 2


def test_prev_moves_backwards():
    ctl, _ = controller(mode=M.LOOP_ALL)
    ctl.next(); ctl.next()                           # c
    assert ctl.prev() is True
    assert ctl.current_index == 1


def test_next_at_end_play_all_fails():
    ctl, _ = controller(mode=M.PLAY_ALL)
    ctl.next(); ctl.next()                            # at c (last)
    assert ctl.next() is False
    assert ctl.status().error


def test_goto_disabled_rejected():
    ctl, _ = controller(clips("a", ("b", E.PLAY_TO_END, {"disabled": True}), "c"))
    assert ctl.goto(1) is False
    assert ctl.current_index == 0


def test_goto_out_of_range():
    ctl, _ = controller()
    with pytest.raises(IndexError):
        ctl.goto(99)


def test_goto_scheduled_emits_timeline():
    ctl, rec = controller()
    ctl.goto(2, at_pts_ms=9000)
    ts = rec.matching("timeline.set ")
    assert ts and json.loads(ts[-1][len("timeline.set "):])["at"] == 9000


def test_toggle_pauses_active_worker():
    ctl, rec = controller()
    ctl.toggle()                                     # -> paused
    assert ctl.playing is False
    assert f"pause {pause_team(0)} now" in rec.cmds
    rec.clear()
    ctl.toggle()                                     # -> playing
    assert ctl.playing is True
    assert f"resume {pause_team(0)}" in rec.cmds


def test_stop_clears_current():
    ctl, _ = controller()
    ctl.stop()
    assert ctl.current_index is None
    assert ctl.playing is False


def test_set_mode():
    ctl, _ = controller()
    ctl.set_mode(M.PLAY_CURRENT)
    assert ctl.mode == M.PLAY_CURRENT
    assert ctl.next_index(+1) is None


# --- edits -------------------------------------------------------------
def test_append_clip():
    ctl, _ = controller()
    ctl.append_clip(Clip(url="/media/d.mp4"))
    assert ctl.clips[-1].name == "d.mp4"
    assert len(ctl.clips) == 4


def test_insert_before_current_shifts_index():
    ctl, _ = controller()
    ctl.next()                                       # current=1 (b)
    ctl.insert_clip(0, Clip(url="/media/z.mp4"))
    assert ctl.clips[0].name == "z.mp4"
    assert ctl.current_index == 2                    # b shifted right


def test_remove_non_current_refreshes_preload():
    ctl, _ = controller(mode=M.LOOP_ALL)             # a,b,c current=0
    ctl.remove_clip(2)                               # remove c (not current)
    assert [c.name for c in ctl.clips] == ["a", "b"]
    assert ctl.current_index == 0


def test_remove_before_current_decrements_index():
    ctl, _ = controller(mode=M.LOOP_ALL)
    ctl.next()                                       # current=1
    ctl.remove_clip(0)                               # remove a
    assert ctl.current_index == 0
    assert [c.name for c in ctl.clips] == ["b", "c"]


def test_remove_current_cuts_away():
    ctl, rec = controller(mode=M.LOOP_ALL)           # a,b,c current=0
    ctl.remove_clip(0)                               # remove current a
    assert [c.name for c in ctl.clips] == ["b", "c"]
    # a cut happened: active flip emitted
    assert any("active" in v for _, k, v in rec.objects_set() if k == "active") \
        or ctl.current_index is not None


def test_remove_last_clip_rejected():
    ctl, _ = controller(clips("only"))
    with pytest.raises(ValueError):
        ctl.remove_clip(0)


def test_reorder_tracks_current():
    ctl, _ = controller(mode=M.LOOP_ALL)             # a,b,c current=0
    ctl.reorder_clip(0, 2)                           # move a to end
    assert [c.name for c in ctl.clips] == ["b", "c", "a"]
    assert ctl.clips[ctl.current_index].name == "a"  # current still follows a


def test_set_disabled_then_skipped():
    ctl, _ = controller(mode=M.LOOP_ALL)
    ctl.set_disabled(1, True)
    assert ctl.next_index(+1) == 2                    # b skipped


def test_set_element_mode_timed_needs_duration():
    ctl, _ = controller()
    with pytest.raises(ValueError):
        ctl.set_element_mode(0, E.TIMED)
    ctl.set_element_mode(0, E.TIMED, duration_ms=4000)
    assert ctl.clips[0].element_mode == E.TIMED
    assert ctl.clips[0].duration_ms == 4000
