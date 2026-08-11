"""Level-1 (playlist) x Level-2 (element) mode truth table for resolve_next.
This is the core regression check for playlist sequencing semantics."""
import pytest

from playlist import (Clip, ElementMode, PlaylistMode, first_enabled,
                      resolve_next)
from helpers import clips

E = ElementMode
M = PlaylistMode


def r(cl, cur, mode, d=+1):
    return resolve_next(cl, cur, mode, d)


def test_play_all_advances_then_stops():
    cl = clips("a", "b", "c")
    assert r(cl, 0, M.PLAY_ALL) == 1
    assert r(cl, 1, M.PLAY_ALL) == 2
    assert r(cl, 2, M.PLAY_ALL) is None          # end -> stop


def test_loop_all_wraps():
    cl = clips("a", "b", "c")
    assert r(cl, 2, M.LOOP_ALL) == 0             # wrap


def test_play_current_stops_after_current():
    cl = clips("a", "b", "c")
    assert r(cl, 1, M.PLAY_CURRENT) is None


def test_loop_current_repeats_current():
    cl = clips("a", "b", "c")
    assert r(cl, 1, M.LOOP_CURRENT) == 1


def test_disabled_clips_skipped():
    cl = clips("a", ("b", E.PLAY_TO_END, {"disabled": True}), "c")
    assert r(cl, 0, M.PLAY_ALL) == 2             # skip b


def test_all_disabled_after_current_stops():
    cl = clips("a", ("b", E.PLAY_TO_END, {"disabled": True}))
    assert r(cl, 0, M.PLAY_ALL) is None


def test_loop_self_holds_in_play_all():
    cl = clips(("a", E.LOOP_SELF), "b")
    # a self-looping clip does not advance on its own under PlayAll/LoopAll
    assert r(cl, 0, M.PLAY_ALL) == 0
    assert r(cl, 0, M.LOOP_ALL) == 0


def test_loop_self_manual_prev_still_moves():
    cl = clips("a", ("b", E.LOOP_SELF), "c")
    # explicit prev overrides loop-self hold
    assert r(cl, 1, M.LOOP_ALL, d=-1) == 0


def test_prev_wraps_only_when_looping():
    cl = clips("a", "b", "c")
    assert r(cl, 0, M.PLAY_ALL, d=-1) is None
    assert r(cl, 0, M.LOOP_ALL, d=-1) == 2


def test_first_enabled():
    cl = clips(("a", E.PLAY_TO_END, {"disabled": True}), "b")
    assert first_enabled(cl) == 1


def test_first_enabled_none():
    cl = clips(("a", E.PLAY_TO_END, {"disabled": True}))
    assert first_enabled(cl) is None


@pytest.mark.parametrize("mode,loops,current", [
    (M.PLAY_ALL, False, False),
    (M.PLAY_CURRENT, False, True),
    (M.LOOP_ALL, True, False),
    (M.LOOP_CURRENT, True, True),
])
def test_mode_flags(mode, loops, current):
    assert mode.loops == loops
    assert mode.current_only == current


def test_timed_requires_duration():
    with pytest.raises(ValueError):
        Clip(url="/m/x", element_mode=E.TIMED)
    Clip(url="/m/x", element_mode=E.TIMED, duration_ms=5000)   # ok
