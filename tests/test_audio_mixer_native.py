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


def test_an_interrupted_crossfade_does_not_revert_to_a_stale_progress_event(tmp_path):
    # a plays throughout; b ends after 1 s, like above, so by the time the events below land,
    # a stale gain aimed at b moves nothing on its own - only the gain change on a would show.
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
        avp.group("audio").startNodes()

        def send(key, params):
            avp.executeCommandsFromString("node.object.set mix " + key + " " + json.dumps(params))

        blocks = [program.get(2000)]
        send("transition_ended", {"output": "program", "sources": ["a"], "pts": 0})
        # Run well past b's end: from here on, only a is ever actually audible.
        blocks += [program.get(2000) for _ in range(99)]

        # Queue one mid-crossfade progress tick (as fade.cpp emits once per program frame while a
        # take runs), scheduled for a moment ahead, then interrupt the take at once: the stray
        # tick queued for the abandoned crossfade must not survive to fire later on its own.
        stale_ns = time.monotonic_ns() + 500_000_000
        send("transition_progress", {"output": "program", "kind": "fade",
                                      "from": ["a"], "to": ["b"], "t": 0.5, "pts": stale_ns})
        send("transition_ended", {"output": "program", "sources": ["a"], "pts": 0})

        blocks += [program.get(2000) for _ in range(75)]
        assert not errors, errors
    finally:
        avp.shutdown()

    assert all(block is not None and block.samplesCount == BLOCK for block in blocks)
    starts = [block.pts.timestamp for block in blocks]
    left = [sample for block in blocks for sample in _planes(block)[0]]
    stale_at = math.ceil(stale_ns * RATE / 1e9) - starts[0]
    assert 2 * RATE < stale_at < len(left) - 2 * RATE

    before = max(abs(sample) for sample in left[stale_at - 2 * BLOCK:stale_at])
    after = max(abs(sample) for sample in left[stale_at:stale_at + 2 * BLOCK])
    assert before > 0.05
    # The stale progress event, if not purged by the interrupt, fires here and halves the gain
    # back toward the abandoned crossfade (a: 1.0 -> 0.5): the mix must stay at full gain on a.
    assert after > before * 0.9


def test_an_aux_bus_hears_the_program_as_late_as_its_pgm_tile_shows_it(tmp_path):
    # An AUX bus whose PGM tile is the program frame one aux tick back (40 ms at 25 fps), and
    # whose compositor leaves 40 ms after the program's: its "program" audio is the program mix
    # 1920 samples back, and each of its blocks leaves 40 ms after the program's block.
    delay_ns, extra_ns = 40_000_000, 40_000_000
    _tone(tmp_path / "a.wav", 440, 10, 2)
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    try:
        for cls, params in (
            (Input, {"url": str(tmp_path / "a.wav"), "dst": "a_packets"}),
            (Demux, {"src": "a_packets", "routing": {"a:0": "a_audio"}}),
            (DecAudio, {"src": "a_audio", "dst": "a_decoded"}),
            (Realtime, {"src": "a_decoded", "dst": "a_paced", "set_pts": True, "time_base": "1/120000",
                        "negative_time_tolerance": .25, "discontinuity_threshold": 1.0}),
        ):
            avp.addNode(cls({"name": cls.TYPE, "group": "audio", **params}))
        avp.addNode(AudioMixer({
            "name": "mix", "group": "audio", "src": ["a_paced"], "source_ids": ["a"],
            "outputs": {"program": "program", "aux": "aux"}, "fps": "25/1", "latency_ms": 80,
        }))
        edges = {"program": avp.getEdge("program", "AudioSamples"), "aux": avp.getEdge("aux", "AudioSamples")}
        avp.group("audio").startNodes()

        def event(output, sources, pts, **timing):
            avp.executeCommandsFromString("node.object.set mix transition_ended " + json.dumps(
                {"output": output, "sources": sources, "pts": pts, **timing}))

        first = edges["program"].get(2000)
        event("program", ["a"], 0)
        event("aux", ["program"], 0, latency_ns=80_000_000 + extra_ns, program_delay_ns=delay_ns)
        # Both edges in turn, block by block, with when each block arrived.
        blocks = {"program": [], "aux": []}
        mute_ns = None
        for i in range(120):
            for name, edge in edges.items():
                block = edge.get(2000)
                blocks[name].append((block, time.monotonic_ns()))
            if i == 60:
                mute_ns = time.monotonic_ns() + 500_000_000
                event("program", [], mute_ns)   # the program goes silent: the AUX a frame later
        status = avp.node("mix").getObject("status")
        assert not errors, errors
    finally:
        avp.shutdown()

    assert first is not None
    assert all(block is not None and block.samplesCount == BLOCK for name in blocks for block, _ in blocks[name])
    buses = {b["id"]: b for b in status["buses"]}
    assert buses["aux"]["program_delay_ms"] == 40 and buses["aux"]["extra_latency_ms"] == 40
    assert buses["program"]["program_delay_ms"] == 0 and status["output_drops"] == 0

    def first_silent(name):
        """The first sample of the silence the mute starts: 100 zeros in a row, past the mute."""
        start = blocks[name][0][0].pts.timestamp
        left = [v for block, _ in blocks[name] for v in _planes(block)[0]]
        run = 0
        for i in range(math.ceil(mute_ns * RATE / 1e9) - start, len(left)):
            run = run + 1 if left[i] == 0.0 else 0
            if run == 100:
                return start + i - 99
        raise AssertionError(f"{name} never went silent")

    # The program's mute shows up on the AUX bus exactly 40 ms of samples later.
    assert first_silent("aux") - first_silent("program") == RATE * delay_ns // 1_000_000_000

    # A program block leaves 60-100 ms after its first sample (80 ms after the tick holding its
    # last one); an AUX block 40 ms after that. Each aux get() is the one waiting, so its time is
    # when the block left; the program's blocks are already queued by then.
    held = [at - block.pts.timestamp * 1_000_000_000 // RATE for block, at in blocks["aux"][10:]]
    assert min(held) > 98_000_000, min(held)
    assert sorted(held)[len(held) // 2] < 150_000_000
