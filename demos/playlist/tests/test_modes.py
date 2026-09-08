"""Public playback-mode contract for the playlist regression harness."""

import pytest

from helpers import clips
from playlist import (Clip, ElementMode, PlaylistMode, first_enabled,
                      resolve_automatic, resolve_manual)

E = ElementMode
M = PlaylistMode


@pytest.mark.parametrize(
    "element_mode,playlist_mode,expected",
    [
        (E.PLAY_TO_END, M.PLAY_ALL, 1),
        (E.PLAY_TO_END, M.PLAY_CURRENT, None),
        (E.PLAY_TO_END, M.LOOP_ALL, 1),
        (E.PLAY_TO_END, M.LOOP_CURRENT, 0),
        (E.TIMED, M.PLAY_ALL, 1),
        (E.TIMED, M.PLAY_CURRENT, None),
        (E.TIMED, M.LOOP_ALL, 1),
        (E.TIMED, M.LOOP_CURRENT, 0),
        (E.LOOP_SELF, M.PLAY_ALL, 0),
        (E.LOOP_SELF, M.PLAY_CURRENT, 0),
        (E.LOOP_SELF, M.LOOP_ALL, 0),
        (E.LOOP_SELF, M.LOOP_CURRENT, 0),
    ],
)
def test_automatic_completion_matrix(element_mode, playlist_mode, expected):
    playlist = clips(("a", element_mode,
                      {"duration_ms": 1000} if element_mode is E.TIMED else {}),
                     "b")
    assert resolve_automatic(playlist, 0, playlist_mode) == expected


def test_play_all_stops_at_end_and_loop_all_wraps():
    playlist = clips("a", "b")
    assert resolve_automatic(playlist, 1, M.PLAY_ALL) is None
    assert resolve_automatic(playlist, 1, M.LOOP_ALL) == 0


def test_automatic_completion_skips_disabled_elements():
    playlist = clips("a", ("b", E.PLAY_TO_END, {"disabled": True}), "c")
    assert resolve_automatic(playlist, 0, M.PLAY_ALL) == 2


@pytest.mark.parametrize("mode", list(M))
def test_manual_next_always_leaves_loop_self(mode):
    playlist = clips(("a", E.LOOP_SELF), "b", "c")
    assert resolve_manual(playlist, 0, mode, +1) == 1


@pytest.mark.parametrize("mode", [M.PLAY_ALL, M.PLAY_CURRENT])
def test_manual_navigation_does_not_wrap_in_non_loop_modes(mode):
    playlist = clips("a", "b", "c")
    assert resolve_manual(playlist, 2, mode, +1) == 2
    assert resolve_manual(playlist, 0, mode, -1) == 0


@pytest.mark.parametrize("mode", [M.LOOP_ALL, M.LOOP_CURRENT])
def test_manual_navigation_wraps_in_loop_modes(mode):
    playlist = clips("a", "b", "c")
    assert resolve_manual(playlist, 2, mode, +1) == 0
    assert resolve_manual(playlist, 0, mode, -1) == 2


def test_manual_navigation_skips_disabled_elements():
    playlist = clips("a", ("b", E.PLAY_TO_END, {"disabled": True}), "c")
    assert resolve_manual(playlist, 0, M.PLAY_ALL, +1) == 2


def test_first_enabled():
    playlist = clips(("a", E.PLAY_TO_END, {"disabled": True}), "b")
    assert first_enabled(playlist) == 1


def test_first_enabled_none():
    assert first_enabled(clips(("a", E.PLAY_TO_END, {"disabled": True}))) is None


def test_timed_requires_duration():
    with pytest.raises(ValueError):
        Clip(url="/m/x", element_mode=E.TIMED)
