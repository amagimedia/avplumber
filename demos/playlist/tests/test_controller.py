"""Transport, scheduling and editing through the public controller seam."""

import pytest

from helpers import advance, clips, controller, cues
from playlist import ElementMode as E, PlaylistMode as M, Transition, TransportState as T


def test_starts_stopped_with_first_enabled_selected():
    ctl, backend, _ = controller(clips(("a", E.PLAY_TO_END, {"disabled": True}), "b", "c"))
    s = ctl.status()
    assert (s.transport, s.selected_index, s.active_index) == (T.STOPPED, 1, None)
    assert backend.calls == []


def test_play_takes_first_element_immediately_then_arms_the_next_at_its_end():
    ctl, backend, clock = controller()
    assert ctl.play()
    assert cues(backend)[0][1:4] == (1, "item-a", None)
    advance(ctl, clock, 0)
    s = ctl.status()
    assert (s.transport, s.active_index, s.next_index) == (T.PLAYING, 0, 1)
    # a is 10 s long by default -> b is armed for t=10000 with the global transition
    assert cues(backend)[1] == ("cue", 2, "item-b", 10_000, Transition.CUT, 500)
    assert s.end_ms == 10_000 and s.next_at_ms == 10_000


def test_armed_element_goes_on_air_at_its_time_and_previous_is_parked():
    ctl, backend, clock = controller()
    ctl.play(); advance(ctl, clock, 0)
    advance(ctl, clock, 9_999)
    assert ctl.status().active_index == 0
    advance(ctl, clock, 1)
    s = ctl.status()
    assert (s.active_index, s.next_index, s.end_ms) == (1, 2, 20_000)
    assert ("park", "item-a") in backend.calls
    assert cues(backend)[-1][1:4] == (3, "item-c", 20_000)


def test_position_follows_wallclock_cue_in_and_speed():
    ctl, _, clock = controller(clips(("a", E.PLAY_TO_END, {"play_from_ms": 2000, "speed": 2.0})))
    ctl.play(); advance(ctl, clock, 0)
    advance(ctl, clock, 1000)
    assert ctl.status().position_ms == 4000
    assert ctl.status().end_ms == 4000  # (10000-2000)/2


def test_cue_out_and_timed_define_the_span():
    a = ("a", E.PLAY_TO_END, {"play_from_ms": 1000, "play_to_ms": 4000})
    b = ("b", E.TIMED, {"duration_ms": 2500})
    ctl, backend, clock = controller(clips(a, b))
    ctl.play(); advance(ctl, clock, 0)
    assert ctl.status().end_ms == 3000
    advance(ctl, clock, 3000)
    assert ctl.status().active_index == 1
    assert ctl.status().end_ms == 5500


def test_unknown_length_reports_error_and_arms_nothing():
    ctl, backend, clock = controller()
    backend.lengths["/media/a"] = None
    ctl.play(); advance(ctl, clock, 0)
    s = ctl.status()
    assert "cue-out" in s.error and s.error_index == 0
    assert len(cues(backend)) == 1 and s.next_at_ms is None


def test_pause_disarms_and_resume_shifts_the_schedule():
    ctl, backend, clock = controller()
    ctl.play(); advance(ctl, clock, 0)
    advance(ctl, clock, 4000)
    assert ctl.pause()
    assert backend.calls[-2:] == [("park", "item-b"), ("pause", "item-a")]
    advance(ctl, clock, 3000)
    assert ctl.status().position_ms == 4000
    assert ctl.play()
    assert ("resume", "item-a") in backend.calls
    assert ctl.status().end_ms == 13_000
    assert cues(backend)[-1][1:4] == (3, "item-b", 13_000)


def test_stop_parks_active_and_armed_and_play_restarts_active_from_cue():
    ctl, backend, clock = controller()
    ctl.play(); advance(ctl, clock, 0)
    assert ctl.stop()
    assert ("park", "item-b") in backend.calls and ("park", "item-a") in backend.calls
    assert ctl.status().transport is T.STOPPED and ctl.status().position_ms is None
    backend.clear()
    assert ctl.play()
    assert cues(backend)[0][2:4] == ("item-a", None)


@pytest.mark.parametrize("mode", list(M))
def test_end_of_playlist_behaviour_per_mode(mode):
    ctl, backend, clock = controller(clips("a", "b"), mode=mode)
    ctl.play(); advance(ctl, clock, 0)
    advance(ctl, clock, 10_000)
    s = ctl.status()
    if mode is M.PLAY_CURRENT:
        assert s.transport is T.STOPPED
    elif mode is M.LOOP_CURRENT:
        assert (s.active_index, s.end_ms) == (0, 20_000)   # chain loops natively; schedule restarts
    else:
        assert s.active_index == 1
        advance(ctl, clock, 10_000)
        assert ctl.status().transport is (T.STOPPED if mode is M.PLAY_ALL else T.PLAYING)


def test_loop_self_never_ends_and_timed_restarts_its_schedule():
    ctl, backend, clock = controller(clips(("a", E.LOOP_SELF), ("b", E.TIMED, {"duration_ms": 3000})),
                                     mode=M.LOOP_CURRENT)
    ctl.play(); advance(ctl, clock, 0)
    assert ctl.status().end_ms is None          # LoopSelf never ends by itself
    assert ctl.next() and advance(ctl, clock, 0) is None
    assert ctl.status().active_index == 1
    advance(ctl, clock, 3000)
    assert ctl.status().end_ms == 6000


def test_manual_take_supersedes_armed_next_and_escapes_current_only_modes():
    ctl, backend, clock = controller(mode=M.PLAY_CURRENT)
    ctl.play(); advance(ctl, clock, 0)
    assert ctl.status().next_index is None
    assert ctl.next()
    assert cues(backend)[-1][1:4] == (2, "item-b", None)
    advance(ctl, clock, 0)
    assert ctl.status().active_index == 1
    assert ctl.prev() and advance(ctl, clock, 0) is None
    assert ctl.status().active_index == 0


def test_manual_navigation_does_not_wrap_in_play_modes():
    ctl, backend, clock = controller(clips("a", "b"), mode=M.PLAY_ALL)
    ctl.play(); advance(ctl, clock, 0)
    assert not ctl.prev()
    assert ctl.status().error == ""


def test_selected_element_actions_are_element_addressed():
    ctl, backend, clock = controller()
    ctl.play(); advance(ctl, clock, 0)
    assert ctl.element_pause(2) and backend.calls[-1] == ("pause", "item-c")
    assert ctl.element_stop(2) and backend.calls[-1] == ("park", "item-c")
    assert ctl.status().selected_index == 2 and ctl.status().transport is T.PLAYING
    assert ctl.element_stop(1)                  # b was armed: disarm = park
    assert backend.calls[-1] == ("park", "item-b") and ctl.status().next_at_ms is None
    assert ctl.element_play(2)
    advance(ctl, clock, 0)
    assert ctl.status().active_index == 2


def test_take_on_active_paused_element_resumes_and_on_playing_is_noop():
    ctl, backend, clock = controller()
    ctl.play(); advance(ctl, clock, 0)
    assert not ctl.element_play(0)
    ctl.pause()
    assert ctl.element_play(0) and ctl.status().transport is T.PLAYING


def test_failed_take_keeps_previous_element_and_reports_error():
    ctl, backend, clock = controller(auto_on_air=False)
    ctl.play()
    req = cues(backend)[-1][1]
    ctl.notify_on_air("item-a", req, 0)
    ctl.next()
    req = cues(backend)[-1][1]
    assert ctl.notify_failed("item-b", req, "decoder exploded")
    s = ctl.status()
    assert (s.active_index, s.transport, s.error, s.error_index) == (0, T.PLAYING, "decoder exploded", 1)


def test_stale_on_air_is_ignored():
    ctl, backend, clock = controller(auto_on_air=False)
    ctl.play()
    stale = cues(backend)[-1][1]
    ctl.next()
    assert not ctl.notify_on_air("item-a", stale, 0)
    assert ctl.status().transport is T.LOADING


def test_disabled_take_is_refused_and_disable_skips_in_schedule():
    ctl, backend, clock = controller()
    ctl.play(); advance(ctl, clock, 0)
    ctl.set_disabled(1, True)
    assert ctl.status().next_index == 2
    assert cues(backend)[-1][2] == "item-c"
    assert not ctl.element_play(1) and "disabled" in ctl.status().error


def test_editing_active_element_retakes_it_and_editing_armed_rearms():
    ctl, backend, clock = controller()
    ctl.play(); advance(ctl, clock, 0)
    ctl.update_clip(1, play_to_ms=3000)
    assert cues(backend)[-1][2:4] == ("item-b", 10_000)
    ctl.update_clip(0, speed=2.0)
    assert cues(backend)[-1][2:4] == ("item-a", None)
    advance(ctl, clock, 0)
    assert ctl.status().end_ms == 5000


def test_mode_and_transition_changes_rearm_with_new_settings():
    ctl, backend, clock = controller()
    ctl.play(); advance(ctl, clock, 0)
    ctl.set_transition(Transition.FADE, 800)
    assert cues(backend)[-1][2:] == ("item-b", 10_000, Transition.FADE, 800)
    ctl.set_mode(M.PLAY_CURRENT)
    assert backend.calls[-1] == ("park", "item-b") and ctl.status().next_at_ms is None
    with pytest.raises(ValueError):
        ctl.set_transition(Transition.CUT, 0)


def test_add_remove_reorder_keep_identities_and_capacity():
    ctl, backend, clock = controller()
    ctl.play(); advance(ctl, clock, 0)
    ctl.insert_clip(1, clips("x")[0])
    assert ctl.status().next_index == 1 and cues(backend)[-1][2] == "item-x"
    ctl.reorder_clip(1, 3)
    assert cues(backend)[-1][2] == "item-b"
    removed = ctl.remove_clip(3)
    assert removed.name == "x" and ("remove", "item-x") in backend.calls
    ctl.remove_clip(0)                          # active
    assert ctl.status().transport is T.STOPPED and ctl.status().active_index is None
    with pytest.raises(ValueError):
        ctl2, *_ = controller(clips("only"))
        ctl2.remove_clip(0)
    with pytest.raises(ValueError):
        ctl.insert_clip(0, clips("b")[0])
    ctl3, *_ = controller(clips(*[f"c{i}" for i in range(16)]))
    with pytest.raises(ValueError):
        ctl3.insert_clip(16, clips("overflow")[0])


def test_failed_arm_falls_back_to_an_immediate_cut_at_the_end():
    ctl, backend, clock = controller(auto_on_air=False)
    ctl.play()
    ctl.notify_on_air("item-a", cues(backend)[-1][1], 0)
    armed = cues(backend)[-1]
    assert armed[2:4] == ("item-b", 10_000)
    ctl.notify_failed("item-b", armed[1], "no frame decoded")
    advance(ctl, clock, 10_000)
    assert cues(backend)[-1][2:4] == ("item-b", None) and "no frame decoded" in ctl.status().error
