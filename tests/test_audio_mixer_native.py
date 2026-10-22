"""audio_mixer on the program's video ticks; run with the built pyplumber extension on the test host."""

import ctypes
import json
import math
import subprocess
import time

import pytest

pytest.importorskip("_avplumber")

from pyplumber import AVPlumber
from pyplumber.node import AudioMixer, DecAudio, Demux, Input, Realtime, ResampleAudioForwardChannels

RATE = 48000
BLOCK = 960


def _tone(path, frequency, seconds, channels):
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        f"sine=frequency={frequency}:sample_rate={RATE}:duration={seconds}",
        "-ac", str(channels), "-c:a", "pcm_s16le", str(path),
    ], check=True, capture_output=True, timeout=30)


def _planes(samples):
    n = samples.samplesCount
    return [list((ctypes.c_float * n).from_address(ptr)) for ptr in samples.data_ptr]


def test_a_stopped_source_never_holds_the_mix(tmp_path):
    # Source a plays 10 s of stereo; source b, mono, ends after 1 s and delivers nothing more.
    _tone(tmp_path / "a.wav", 440, 10, 2)
    _tone(tmp_path / "b.wav", 1000, 1, 1)
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    try:
        for tag in ("a", "b"):
            for cls, params in (
                (Input, {"url": str(tmp_path / f"{tag}.wav"), "dst": f"{tag}_packets"}),
                (Demux, {"src": f"{tag}_packets", "routing": {"a:0": f"{tag}_audio"}}),
                (DecAudio, {"src": f"{tag}_audio", "dst": f"{tag}_decoded"}),
                (ResampleAudioForwardChannels, {"src": f"{tag}_decoded", "dst": f"{tag}_resampled",
                                                "dst_sample_rate": RATE, "dst_sample_format": "fltp",
                                                "compensation": 0}),
                (Realtime, {"src": f"{tag}_resampled", "dst": f"{tag}_paced", "set_pts": True,
                            "time_base": "1/120000", "negative_time_tolerance": .25,
                            "discontinuity_threshold": 1.0}),
            ):
                avp.addNode(cls({"name": f"{cls.TYPE}_{tag}", "group": "audio", **params}))
        avp.addNode(AudioMixer({
            "name": "mix", "group": "audio", "src": ["a_paced", "b_paced"], "source_ids": ["a", "b"],
            "outputs": {"program": "program"}, "fps": "25/1", "latency_ms": 80,
        }))
        program = avp.getEdge("program", "AudioSamples")
        started = time.monotonic()
        avp.group("audio").startNodes()

        def event(sources, pts):
            avp.executeCommandsFromString("node.object.set mix transition_ended " + json.dumps(
                {"output": "program", "sources": sources, "pts": pts}))

        # The group creates its nodes as it starts: the first block says the mixer exists.
        blocks = [program.get(2000)]
        event(["a", "b"], 0)
        blocks += [program.get(2000) for _ in range(99)]
        # Mute everything from a sample 0.5 s ahead: the mix goes silent exactly there.
        mute_ns = time.monotonic_ns() + 500_000_000
        event([], mute_ns)
        blocks += [program.get(2000) for _ in range(50)]
        elapsed = time.monotonic() - started
        status = avp.node("mix").getObject("status")
        assert not errors, errors
    finally:
        avp.shutdown()

    assert all(block is not None and block.samplesCount == BLOCK for block in blocks)
    starts = [block.pts.timestamp for block in blocks]
    # Contiguous blocks stamped with their media time on the monotonic clock, not counted from 0.
    assert all(b - a == BLOCK for a, b in zip(starts, starts[1:]))
    assert abs(starts[0] / RATE - started) < 1.0
    # Paced by the deadlines: 150 blocks of 20 ms after the first one's.
    assert 2.7 < elapsed < 3.5

    sources = {s["id"]: s for s in status["sources"]}
    # b stopped after 1 s; the mix went on, with b missing, a continuous and never late.
    assert sources["b"]["missing"] > RATE
    assert sources["a"]["missing"] < RATE // 2   # only before its first frame
    assert sources["a"]["dropped"] == 0 and sources["a"]["jumps"] == 0
    assert status["output_drops"] == 0

    left = [sample for block in blocks for sample in _planes(block)[0]]
    mute = math.ceil(mute_ns * RATE / 1e9) - starts[0]
    assert 2 * RATE < mute < len(left)
    assert max(abs(sample) for sample in left[mute - 2 * BLOCK:mute]) > 0.05   # a alone (its sine peaks at 1/8 -3 dB), long after b ended
    assert all(sample == 0.0 for sample in left[mute:])
