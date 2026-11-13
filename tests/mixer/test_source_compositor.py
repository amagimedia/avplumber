"""Source-paced composition: frozen endpoints, media PTS and the shared curves."""

import math
from types import SimpleNamespace

import pytest

from pyplumber.mixer.config import FADE_CURVES
from pyplumber.mixer.source import SourceMove, source_compositor


@pytest.mark.parametrize("curve", FADE_CURVES)
def test_progress_uses_source_timestamps_and_leaves_easing_to_the_native_mixer(curve):
    move = SourceMove(
        [{"input": 0, "dst_x": 0}],
        [{"input": 0, "dst_x": -1080}],
        start=17,
        duration=0.8,
        curve=curve,
    )
    assert move.at(16)["progress"] == 0
    assert move.at(17)["progress"] == 0
    assert move.at(17.4)["progress"] == pytest.approx(0.5)
    assert move.at(18)["progress"] == 1
    assert move.at(17.4) == move.at(17.4)  # repeat/drop/delay does not advance the take
    assert move.at(17.4)["curve"] == curve


def test_detections_and_consumers_cannot_mutate_the_running_take():
    old, new = [{"crop": {"x": 2}}], [{"crop": {"x": 20}}]
    move = SourceMove(old, new, start=0, duration=1)
    old[0]["crop"]["x"] = 100
    new[0]["crop"]["x"] = 200
    packet = move.at(0.5)
    packet["to"][0]["crop"]["x"] = 300
    assert move.at(0.5)["from"][0]["crop"]["x"] == 2
    assert move.at(0.5)["to"][0]["crop"]["x"] == 20


@pytest.mark.parametrize(
    "changes",
    [
        {"start": math.nan},
        {"duration": 0},
        {"duration": -1},
        {"duration": math.inf},
        {"curve": "bounce"},
        {"unmatched": "slide"},
    ],
)
def test_invalid_takes_are_rejected_before_the_graph(changes):
    with pytest.raises(ValueError):
        SourceMove([], [], **{"start": 0, "duration": 1, **changes})


def test_no_timestamp_is_not_a_motion_tick():
    with pytest.raises(ValueError, match="PTS"):
        SourceMove([], [], start=0, duration=1).at(math.nan)


def test_compositor_preserves_the_source_contract():
    params = {
        "src": ["video", "backdrop"],
        "dst": "canvas",
        "layers": [],
        "metadata_key": "layers",
        "move_metadata_key": "move",
    }
    calls = []
    api = SimpleNamespace(CudaRectOverlay=lambda p: calls.append(p) or p)
    assert source_compositor(params, api=api) == params
    assert calls == [params]
    assert "fps" not in calls[0] and "clock_input" not in calls[0]
