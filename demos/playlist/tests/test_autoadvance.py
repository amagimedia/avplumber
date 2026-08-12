"""Natural EOF and timed completion through the public controller seam."""

from helpers import clips, controller, finish_pending
from playlist import ElementMode as E, PlaylistMode as M, TransportState


def start(ctl, backend, index=0):
    ctl.select(index)
    assert ctl.play()
    finish_pending(ctl, backend)


def test_play_to_end_requests_next_item_after_eof():
    ctl, backend = controller(clips("a", "b"), M.PLAY_ALL)
    start(ctl, backend); backend.clear()
    backend.eof(ctl.clips[0])
    ctl.poll(100)
    assert backend.calls[-1][0] == "play_item"
    assert backend.calls[-1][2] == ctl.clips[1].item_id
    assert ctl.status().active_index == 0
    assert ctl.status().pending_index == 1


def test_play_to_end_stops_source_not_output_at_non_loop_end():
    ctl, backend = controller(clips("a", "b"), M.PLAY_ALL)
    start(ctl, backend, 1); backend.clear()
    backend.eof(ctl.clips[1])
    ctl.poll(100)
    assert backend.calls[-2:] == [
        ("cancel_activation",), ("stop_item", ctl.clips[1].item_id)]
    assert ctl.status().transport is TransportState.STOPPED
    assert ctl.status().active_index == 1
    assert ctl.status().output_alive is True


def test_loop_all_wraps_and_loop_current_restarts_same_item():
    ctl, backend = controller(clips("a", "b"), M.LOOP_ALL)
    start(ctl, backend, 1); backend.clear()
    backend.eof(ctl.clips[1]); ctl.poll(100)
    assert backend.calls[-1][2] == ctl.clips[0].item_id

    ctl2, backend2 = controller(clips("a", "b"), M.LOOP_CURRENT)
    start(ctl2, backend2, 0); backend2.clear()
    backend2.eof(ctl2.clips[0]); ctl2.poll(100)
    assert backend2.calls[-1][2] == ctl2.clips[0].item_id


def test_loop_self_source_normally_has_no_eof_but_restarts_if_one_arrives():
    ctl, backend = controller(clips(("a", E.LOOP_SELF), "b"), M.PLAY_ALL)
    start(ctl, backend); backend.clear()
    backend.eof(ctl.clips[0]); ctl.poll(100)
    assert backend.calls[-1][2] == ctl.clips[0].item_id


def test_timed_ignores_eof_and_completes_on_unpaused_wall_clock():
    ctl, backend = controller(
        clips(("a", E.TIMED, {"duration_ms": 1000}), "b"), M.PLAY_ALL)
    start(ctl, backend); backend.clear()
    backend.eof(ctl.clips[0])
    ctl.poll(999)
    assert backend.calls == []
    ctl.poll(1000)
    assert backend.calls[-1][2] == ctl.clips[1].item_id


def test_timed_pause_preserves_remaining_duration():
    ctl, backend = controller(
        clips(("a", E.TIMED, {"duration_ms": 1000}), "b"), M.PLAY_ALL)
    start(ctl, backend)
    ctl.poll(400)
    ctl.pause()
    ctl.poll(5000)
    backend.clear()
    ctl.play()
    ctl.poll(5000)
    ctl.poll(5599)
    assert not [call for call in backend.calls if call[0] == "play_item"]
    ctl.poll(5600)
    assert backend.calls[-1][2] == ctl.clips[1].item_id


def test_stale_eof_from_previous_item_is_ignored():
    ctl, backend = controller(clips("a", "b", "c"), M.PLAY_ALL)
    start(ctl, backend)
    ctl.next(); finish_pending(ctl, backend); backend.clear()
    backend.eof(ctl.clips[0]); ctl.poll(100)
    assert backend.calls == []
    assert ctl.status().active_index == 1
