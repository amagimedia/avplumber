"""Every public playlist and selected-element action at the backend seam."""

import pytest

from helpers import clips, controller, finish_pending
from playlist import Clip, ElementMode as E, PlaylistMode as M, TransportState


def start(ctl, backend, index=0):
    assert ctl.element_play(index)
    finish_pending(ctl, backend)


def test_initial_state_selects_first_enabled_but_starts_stopped():
    ctl, backend = controller(clips(
        ("a", E.PLAY_TO_END, {"disabled": True}), "b", "c"))
    status = ctl.status()
    assert status.selected_index == 1
    assert status.active_index is None
    assert status.pending_index is None
    assert status.transport is TransportState.STOPPED
    assert backend.calls == []


def test_playlist_play_is_two_phase_and_commits_on_matching_ready_event():
    ctl, backend = controller()
    assert ctl.play()
    request = backend.calls[-1][1]
    assert ctl.status().transport is TransportState.LOADING
    assert ctl.status().pending_index == 0
    assert ctl.notify_source_ready(ctl.clips[0].item_id, request + 1) is False
    assert ctl.notify_source_ready(ctl.clips[0].item_id, request) is True
    assert ctl.status().active_index == 0
    assert ctl.status().transport is TransportState.PLAYING


def test_playlist_pause_resume_stop_never_change_output_health():
    ctl, backend = controller()
    start(ctl, backend); backend.clear()
    item_id = ctl.clips[0].item_id

    assert ctl.pause()
    assert backend.calls == [("pause_item", item_id)]
    assert ctl.play()
    assert backend.calls[-1] == ("resume_item", item_id)
    assert ctl.stop()
    assert backend.calls[-2:] == [
        ("cancel_activation",), ("stop_item", item_id)]
    assert ctl.status().transport is TransportState.STOPPED
    assert ctl.status().active_index == 0
    assert ctl.status().output_alive is True


@pytest.mark.parametrize("mode", list(M))
def test_playlist_transport_cycle_in_every_playlist_mode(mode):
    ctl, backend = controller(mode=mode)
    start(ctl, backend)
    item_id = ctl.clips[0].item_id
    backend.clear()

    assert ctl.pause()
    assert ctl.play()
    assert ctl.stop()
    assert ctl.play()

    assert backend.calls[:4] == [
        ("pause_item", item_id),
        ("resume_item", item_id),
        ("cancel_activation",),
        ("stop_item", item_id),
    ]
    assert backend.calls[4][0] == "play_item"
    assert backend.calls[4][2] == item_id


def test_stop_highlight_then_playlist_play_restarts_active_item_at_its_cue():
    ctl, backend = controller(clips(
        ("a", E.PLAY_TO_END, {"play_from_ms": 2500}), "b", "c"))
    start(ctl, backend); ctl.stop(); backend.clear()
    ctl.select(2)

    assert ctl.play()
    operation, _, item_id, clip = backend.calls[-1]
    assert (operation, item_id, clip.play_from_ms) == (
        "play_item", ctl.clips[0].item_id, 2500)
    assert ctl.status().selected_index == 2


def test_pause_highlight_then_playlist_play_resumes_active_item():
    ctl, backend = controller()
    start(ctl, backend); ctl.pause(); backend.clear()
    ctl.select(2)

    assert ctl.play()
    assert backend.calls == [("resume_item", ctl.clips[0].item_id)]
    assert ctl.status().selected_index == 2
    assert ctl.status().active_index == 0


def test_playlist_play_before_first_activation_ignores_highlighted_item():
    ctl, backend = controller()
    ctl.select(2)

    assert ctl.play()
    assert backend.calls[-1][2] == ctl.clips[0].item_id
    assert ctl.status().selected_index == 2


def test_playlist_next_previous_and_all_four_modes():
    ctl, backend = controller(mode=M.PLAY_ALL)
    start(ctl, backend); backend.clear()
    assert ctl.next()
    assert backend.calls[-1][2] == ctl.clips[1].item_id
    finish_pending(ctl, backend)
    assert ctl.prev()
    assert backend.calls[-1][2] == ctl.clips[0].item_id
    for mode in M:
        ctl.set_mode(mode)
        assert ctl.status().mode is mode


def test_selected_item_play_pause_stop_are_item_addressed():
    ctl, backend = controller()
    start(ctl, backend); backend.clear()
    target = ctl.clips[2].item_id
    assert ctl.element_play(2)
    assert backend.calls[-1][2] == target
    finish_pending(ctl, backend)
    backend.clear()
    assert ctl.element_pause(2)
    assert backend.calls == [("pause_item", target)]
    assert ctl.element_stop(2)
    assert backend.calls[-1] == ("stop_item", target)
    assert ctl.status().transport is TransportState.STOPPED


@pytest.mark.parametrize("element_mode", list(E))
def test_selected_item_transport_cycle_in_every_element_mode(element_mode):
    kwargs = {"duration_ms": 1000} if element_mode is E.TIMED else {}
    ctl, backend = controller(clips(("a", element_mode, kwargs), "b"))
    start(ctl, backend)
    item_id = ctl.clips[0].item_id
    backend.clear()

    assert ctl.element_pause(0)
    assert ctl.element_play(0)
    assert ctl.element_stop(0)
    assert ctl.element_play(0)

    assert backend.calls[:3] == [
        ("pause_item", item_id),
        ("resume_item", item_id),
        ("stop_item", item_id),
    ]
    assert backend.calls[3][0] == "play_item"
    assert backend.calls[3][2] == item_id


@pytest.mark.parametrize("mode", list(M))
@pytest.mark.parametrize("direction,start_index", [(+1, 2), (-1, 0)])
def test_manual_navigation_boundary_in_every_playlist_mode(
        mode, direction, start_index):
    ctl, backend = controller(mode=mode)
    start(ctl, backend, start_index)
    backend.clear()

    changed = ctl.next() if direction > 0 else ctl.prev()
    if mode.loops:
        expected = 0 if direction > 0 else 2
        assert changed
        assert backend.calls[-1][2] == ctl.clips[expected].item_id
    else:
        assert changed is False
        assert backend.calls == []


def test_pause_and_stop_on_inactive_item_still_execute_for_that_source_only():
    ctl, backend = controller()
    start(ctl, backend); backend.clear()
    inactive = ctl.clips[2].item_id
    assert ctl.element_pause(2)
    assert ctl.element_stop(2)
    assert backend.calls == [
        ("pause_item", inactive), ("stop_item", inactive)]
    assert ctl.status().active_index == 0
    assert ctl.status().transport is TransportState.PLAYING


def test_superseded_load_ignores_stale_completion():
    ctl, backend = controller()
    ctl.element_play(1)
    first_request = backend.calls[-1][1]
    ctl.element_play(2)
    second_request = backend.calls[-1][1]
    assert second_request > first_request
    assert ctl.notify_source_ready(ctl.clips[1].item_id, first_request) is False
    assert ctl.notify_source_ready(ctl.clips[2].item_id, second_request) is True
    assert ctl.status().active_index == 2


def test_failed_replacement_retains_previous_active_transport_and_frame():
    ctl, backend = controller()
    start(ctl, backend)
    ctl.element_play(2)
    request = backend.calls[-1][1]
    assert ctl.notify_source_failed(
        ctl.clips[2].item_id, request, "cannot open source")
    status = ctl.status()
    assert status.selected_index == 2
    assert status.active_index == 0
    assert status.pending_index is None
    assert status.transport is TransportState.PLAYING
    assert status.error_index == 2
    assert status.error == "cannot open source"


def test_manual_navigation_escapes_loop_self_and_current_only():
    ctl, backend = controller(clips(("a", E.LOOP_SELF), "b"), M.LOOP_CURRENT)
    start(ctl, backend); backend.clear()
    assert ctl.next()
    assert backend.calls[-1][2] == ctl.clips[1].item_id


def test_non_loop_boundary_is_noop_without_error():
    ctl, backend = controller(clips("a", "b"), M.PLAY_CURRENT)
    start(ctl, backend, 1); backend.clear()
    assert ctl.next() is False
    assert backend.calls == []
    assert ctl.status().error == ""


def test_element_mode_and_settings_are_independent_from_playlist_mode():
    ctl, backend = controller()
    ctl.set_mode(M.PLAY_CURRENT)
    ctl.set_element_mode(1, E.TIMED, duration_ms=1500)
    ctl.update_clip(1, play_from_ms=1000, play_to_ms=8000,
                    duration_ms=3000, speed=1.25)
    clip = ctl.clips[1]
    assert ctl.mode is M.PLAY_CURRENT
    assert (clip.element_mode, clip.play_from_ms, clip.play_to_ms,
            clip.duration_ms, clip.speed) == (E.TIMED, 1000, 8000, 3000, 1.25)
    assert backend.calls == []


def test_active_speed_change_rebuilds_source_without_runtime_speed_command():
    ctl, backend = controller()
    start(ctl, backend); backend.clear()
    item_id = ctl.clips[0].item_id
    ctl.update_clip(0, speed=0.5)
    operation, _, called_item_id, clip = backend.calls[-1]
    assert (operation, called_item_id, clip.speed) == (
        "play_item", item_id, 0.5)
    assert not [call for call in backend.calls if call[0] == "set_speed"]


def test_active_cue_or_mode_change_reloads_item():
    ctl, backend = controller()
    start(ctl, backend); backend.clear()
    ctl.update_clip(0, play_from_ms=1000)
    assert backend.calls[-1][0] == "play_item"
    request = backend.calls[-1][1]
    assert ctl.notify_source_ready(ctl.clips[0].item_id, request)
    backend.clear()
    ctl.set_element_mode(0, E.LOOP_SELF)
    assert backend.calls[-1][0] == "play_item"


def test_add_insert_reorder_preserve_active_identity():
    ctl, backend = controller()
    start(ctl, backend, 1)
    active_id = ctl.clips[1].item_id
    ctl.append_clip(Clip(url="/media/d", name="d", item_id="item-d"))
    ctl.insert_clip(0, Clip(url="/media/z", name="z", item_id="item-z"))
    ctl.reorder_clip(2, 4)
    assert ctl.clips[ctl.status().active_index].item_id == active_id


def test_remove_inactive_and_active_release_only_their_item_slots():
    ctl, backend = controller()
    start(ctl, backend); backend.clear()
    inactive_id = ctl.clips[2].item_id
    ctl.remove_clip(2)
    assert backend.calls == [("remove_item", inactive_id)]
    active_id = ctl.clips[0].item_id
    backend.clear()
    ctl.remove_clip(0)
    assert backend.calls == [("remove_item", active_id)]
    assert ctl.status().active_index is None
    assert ctl.status().transport is TransportState.STOPPED


def test_remove_pending_cancels_before_release_and_keeps_previous_item_playing():
    ctl, backend = controller()
    start(ctl, backend); backend.clear()
    active_id = ctl.clips[0].item_id
    pending = ctl.clips[2]
    assert ctl.element_play(2)
    request = backend.calls[-1][1]
    backend.clear()

    ctl.remove_clip(2)

    assert backend.calls == [
        ("cancel_activation",), ("remove_item", pending.item_id)]
    status = ctl.status()
    assert ctl.clips[status.active_index].item_id == active_id
    assert status.pending_index is None
    assert status.transport is TransportState.PLAYING
    assert ctl.notify_source_ready(pending.item_id, request) is False


def test_removing_last_item_and_duplicate_ids_are_rejected():
    ctl, _ = controller(clips("only"))
    with pytest.raises(ValueError):
        ctl.remove_clip(0)
    with pytest.raises(ValueError):
        ctl.append_clip(Clip(url="/other", item_id="item-only"))


def test_disable_enable_stops_only_item_and_selects_an_enabled_successor():
    ctl, backend = controller()
    start(ctl, backend); backend.clear()
    active_id = ctl.clips[0].item_id
    ctl.set_disabled(0, True)
    assert backend.calls == [("stop_item", active_id)]
    assert ctl.status().selected_index == 1
    assert ctl.status().transport is TransportState.STOPPED
    ctl.set_disabled(0, False)
    assert ctl.clips[0].disabled is False


def test_disable_pending_cancels_before_stop_and_returns_to_stopped():
    ctl, backend = controller()
    pending = ctl.clips[2]
    assert ctl.element_play(2)
    request = backend.calls[-1][1]
    backend.clear()

    ctl.set_disabled(2, True)

    assert backend.calls == [
        ("cancel_activation",), ("stop_item", pending.item_id)]
    status = ctl.status()
    assert status.pending_index is None
    assert status.transport is TransportState.STOPPED
    assert status.selected_index == 0
    assert ctl.notify_source_ready(pending.item_id, request) is False


def test_status_uses_cached_output_health():
    ctl, backend = controller()
    assert ctl.status().output_alive is True
    backend.alive = False
    assert ctl.status().output_alive is False
