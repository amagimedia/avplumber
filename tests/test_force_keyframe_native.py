"""Native node tests; run with the built pyplumber extension on the test host."""

from contextlib import contextmanager
from fractions import Fraction
import subprocess

import pytest

pytest.importorskip("_avplumber")

from pyplumber import AVPlumber
from pyplumber.node import DecVideo, Demux, ForceKeyFrame, Input


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    def generate(fps="60/1", count=125):
        path = tmp_path_factory.mktemp("keyframe-frames") / "frames.nut"
        subprocess.run([
            "ffmpeg", "-v", "error", "-n", "-f", "lavfi", "-i",
            f"testsrc2=size=64x64:rate={fps}", "-frames:v", str(count),
            "-c:v", "rawvideo", "-f", "nut", str(path),
        ], check=True, capture_output=True, timeout=15)
        avp = AVPlumber()
        errors = []
        avp.on_exception = lambda *error: errors.append(error)
        for cls, params in (
            (Input, {"url": str(path), "dst": "packets"}),
            (Demux, {"src": "packets", "routing": {"v:0": "video"}}),
            (DecVideo, {"src": "video", "dst": "frames"}),
        ):
            avp.addNode(cls({"group": "fixture", **params}))
        edge = avp.getEdge("frames", "VideoFrame")
        result = []
        try:
            avp.group("fixture").startNodes()
            for _ in range(count):
                frame = edge.get(5000)
                assert not errors, errors
                result.append(frame)
            return result
        finally:
            avp.shutdown()
    return generate


@contextmanager
def limiter(**params):
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    try:
        avp.addNode(ForceKeyFrame({
            "name": "limit", "src": "input", "dst": "output", **params,
        }), early_create=True)
        source = avp.getEdge("input", "VideoFrame")
        output = avp.getEdge("output", "VideoFrame")
        avp.node("limit").start()

        def step(frame, requests=0):
            for _ in range(requests):
                avp.executeCommandsFromString("node.object.set limit trigger true")
            source.enqueue(frame)
            result = output.get(5000)
            assert not errors, errors
            assert result.pts.timestamp == frame.pts.timestamp
            assert result.data == frame.data
            return result.keyFrame

        yield step, lambda: avp.node("limit").getObject("status")
    finally:
        avp.shutdown()


@pytest.mark.parametrize("fps,minimum,spacing", [
    ("60/1", 100, 6), ("60/1", 150, 9), ("60/1", 200, 12),
    ("60000/1001", 150, 9), ("60000/1001", 200, 12), ("24000/1001", 200, 5),
])
def test_spam_is_coalesced_without_losing_frames(frames, fps, minimum, spacing):
    footage = frames(fps=fps)
    with limiter(min_interval_ms=minimum, interval_sec="1/1") as (step, status):
        keys = [i for i, frame in enumerate(footage) if step(frame, requests=5)]
        assert keys == list(range(0, len(footage), spacing))
        state = status()
        assert state["requested_generation"] == len(footage) * 5
        assert state["triggered_frames"] == len(keys)
        assert state["min_interval_ms"] == minimum
        assert all(Fraction(b - a, 1) / Fraction(fps) >= Fraction(minimum, 1000)
                   for a, b in zip(keys, keys[1:]))


def test_last_request_survives_the_cooldown_and_is_not_repeated(frames):
    with limiter(min_interval_ms=200) as (step, status):
        keys = []
        for i, frame in enumerate(frames(count=30)):
            if step(frame, requests=1 if i in (0, 1) else 0):
                keys.append(i)
            if 1 <= i < 12:
                assert status()["pending"]
                assert status()["forced_generation"] == 1
            if i >= 12:
                assert not status()["pending"]
        assert keys == [0, 12]
        assert status()["forced_generation"] == 2


def test_periodic_keyframe_cannot_bypass_a_recent_trigger(frames):
    with limiter(min_interval_ms=200, interval_sec="1/1") as (step, status):
        keys = [i for i, frame in enumerate(frames()) if step(frame, requests=int(i == 59))]
        assert keys == [0, 59, 71, 120]
        assert status()["periodic_frames"] == 3
        assert status()["triggered_frames"] == 1


@pytest.mark.parametrize("params", [{}, {"min_interval_ms": 0}])
def test_default_preserves_unlimited_triggering(frames, params):
    with limiter(**params) as (step, status):
        assert all(step(frame, requests=1) for frame in frames(count=15))
        assert status()["triggered_frames"] == 15
        assert not status()["pending"]


def test_periodic_forcing_can_be_disabled(frames):
    with limiter(min_interval_ms=200) as (step, status):
        assert not any(step(frame) for frame in frames())
        assert status()["periodic_frames"] == 0


def test_backwards_pts_restarts_the_limit_and_duplicate_pts_do_not(frames):
    footage = frames(count=30)
    with limiter(min_interval_ms=200) as (step, _):
        assert step(footage[24], requests=1)
        assert not step(footage[24], requests=1)
        assert step(footage[0])  # Pending request survives the timeline reset.
        assert not step(footage[0], requests=1)
        assert not step(footage[11])
        assert step(footage[12])


@pytest.mark.parametrize("value", [-1, -0.1, 0.2, True, "200", 2**31])
def test_invalid_interval_is_rejected(value):
    with pytest.raises(RuntimeError, match="min_interval_ms"):
        with limiter(min_interval_ms=value):
            pytest.fail("invalid interval was accepted")
