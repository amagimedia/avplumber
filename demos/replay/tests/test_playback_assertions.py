"""The paused-picture oracle must detect transient motion between samples."""

from types import SimpleNamespace

import pytest

from test_rust_integration import _assert_stable


@pytest.mark.parametrize("observations,valid", [
    ([], True), ([479], True), ([479, 479], True),
    ([478, 479], False), ([480, 479], False), ([480], False),
])
def test_paused_picture_oracle(monkeypatch, observations, valid):
    frames = []
    sleeps = []

    def advance(_seconds):
        sleeps.append(True)
        if len(sleeps) == 2:
            frames.extend(observations)

    control = SimpleNamespace(
        artifact=SimpleNamespace(fps=60),
        status=lambda: SimpleNamespace(playing=False, frame_number=frames[-1] if frames else 479),
        observation_marker=lambda: len(frames),
        observed_frames_since=lambda marker: frames[marker:],
    )
    monkeypatch.setattr("test_rust_integration.time.sleep", advance)
    if valid:
        _assert_stable(control)
    else:
        with pytest.raises(AssertionError):
            _assert_stable(control)
