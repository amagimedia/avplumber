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
# A gain change glides in over kGainRampSeconds (src/nodes/audio_mixer.cpp): a cut or a
# frame-rate AFV step declicks instead of stepping, so a gain is only fully applied this
# many samples after the event's.
GAIN_RAMP = math.ceil(0.03 * RATE)


def _tone(path, frequency, seconds, channels):
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        f"sine=frequency={frequency}:sample_rate={RATE}:duration={seconds}",
        "-ac", str(channels), "-c:a", "pcm_s16le", str(path),
    ], check=True, capture_output=True, timeout=30)


def _planes(samples):
    n = samples.samplesCount
    return [list((ctypes.c_float * n).from_address(ptr)) for ptr in samples.data_ptr]


MARK_SEC = 1.0


def _marked(path, frequency, channels, rate=RATE, seconds=5):
    """Silence until MARK_SEC, then a steady tone: one sharp edge at a known media time."""
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        f"sine=frequency={frequency}:sample_rate={rate}:duration={seconds},"
        # enable= is evaluated per sample against t; the comma inside lt() must be escaped
        # so the filtergraph parser does not split the chain on it.
        f"volume=volume=0:enable='lt(t\\,{MARK_SEC})'",
        "-ac", str(channels), "-c:a", "pcm_s16le", str(path),
    ], check=True, capture_output=True, timeout=120)


def _onset(samples, frequency, window=BLOCK, hop=120, ratio=0.3):
    """Index of the first window whose energy at `frequency` passes `ratio` of the peak.

    One Goertzel-style correlation per window. The tones are pure and hundreds of Hz apart,
    so each source's edge is located independently of the others in the same mix."""
    w = 2 * math.pi * frequency / RATE
    cos = [math.cos(w * n) for n in range(window)]
    sin = [math.sin(w * n) for n in range(window)]
    mags = []
    for start in range(0, len(samples) - window, hop):
        re = im = 0.0
        for n in range(window):
            v = samples[start + n]
            re += v * cos[n]
            im += v * sin[n]
        mags.append((math.hypot(re, im) / window, start))
    peak = max((m for m, _ in mags), default=0.0)
    at = next((start for m, start in mags if m > ratio * peak), None)
    return at, peak


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
    # The mute lands on `mute` and glides out over GAIN_RAMP; past the ramp the source's peak is
    # below kGainEpsilon and the mixer stops reading it at all, so the mix is exactly zero.
    assert max(abs(sample) for sample in left[mute + GAIN_RAMP // 2:mute + GAIN_RAMP]) < 0.05
    assert all(sample == 0.0 for sample in left[mute + GAIN_RAMP:])


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
    assert 2 * RATE < stale_at < len(left) - GAIN_RAMP - 2 * BLOCK

    before = max(abs(sample) for sample in left[stale_at - 2 * BLOCK:stale_at])
    # Past the ramp: a gain the stale event aimed at (a: 1.0 -> 0.707, equal power at t = 0.5)
    # needs GAIN_RAMP samples to arrive, so a window starting at the event would still read as
    # full gain either way.
    after = max(abs(sample) for sample in left[stale_at + GAIN_RAMP:stale_at + GAIN_RAMP + 2 * BLOCK])
    assert before > 0.05
    # The stale progress event, if not purged by the interrupt, fires here and pulls the gain back
    # toward the abandoned crossfade (a: 1.0 -> 0.707): the mix must stay at full gain on a.
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


def _level(samples, frequency, start, length):
    """The magnitude of `frequency` over samples[start:start + length]."""
    w = 2 * math.pi * frequency / RATE
    re = sum(samples[start + n] * math.cos(w * n) for n in range(length))
    im = sum(samples[start + n] * math.sin(w * n) for n in range(length))
    return math.hypot(re, im) / length


def test_a_ducker_lowers_its_target_where_its_sidechain_is_heard(tmp_path):
    # "mix" plays a steady tone throughout; "key" is silent for MARK_SEC, then a tone. Every bus
    # ducks "mix" by "key". On program both are on air: ducked once key starts. On solo only mix
    # is on air: a post sidechain hears nothing there, so mix plays in full. On solo_pre the
    # sidechain is tapped before the take's gains: ducked although key is off air.
    _tone(tmp_path / "mix.wav", 440, 10, 2)
    _marked(tmp_path / "key.wav", 1000, 2)
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    ducker = {"type": "ducker", "target": ["mix"], "sidechain": ["key"], "threshold_db": -40,
              "depth_db": -20, "attack_ms": 10, "hold_ms": 300, "release_ms": 500}
    buses = ("program", "solo", "solo_pre")
    try:
        for tag in ("mix", "key"):
            for cls, params in (
                (Input, {"url": str(tmp_path / f"{tag}.wav"), "dst": f"{tag}_packets"}),
                (Demux, {"src": f"{tag}_packets", "routing": {"a:0": f"{tag}_audio"}}),
                (DecAudio, {"src": f"{tag}_audio", "dst": f"{tag}_decoded"}),
                (Realtime, {"src": f"{tag}_decoded", "dst": f"{tag}_paced", "set_pts": True,
                            "time_base": "1/120000", "negative_time_tolerance": .25,
                            "discontinuity_threshold": 1.0}),
            ):
                avp.addNode(cls({"name": f"{cls.TYPE}_{tag}", "group": "audio", **params}))
        avp.addNode(AudioMixer({
            "name": "mix", "group": "audio", "src": ["mix_paced", "key_paced"], "source_ids": ["mix", "key"],
            "outputs": {bus: bus for bus in buses}, "fps": "25/1", "latency_ms": 80,
            "processors": {"program": [{"id": "d", **ducker}], "solo": [{"id": "d", **ducker}],
                           "solo_pre": [{"id": "d", **ducker, "sidechain_tap": "pre"}]},
        }))
        edges = {bus: avp.getEdge(bus, "AudioSamples") for bus in buses}
        avp.group("audio").startNodes()

        def event(output, sources):
            avp.executeCommandsFromString("node.object.set mix transition_ended " + json.dumps(
                {"output": output, "sources": sources, "pts": 0}))

        blocks = {bus: [] for bus in buses}
        for i in range(200):
            for bus, edge in edges.items():
                blocks[bus].append(edge.get(2000))
            if i == 0:
                event("program", ["mix", "key"])
                event("solo", ["mix"])
                event("solo_pre", ["mix"])
        status = avp.node("mix").getObject("status")
        avp.executeCommandsFromString("node.object.set mix set_processor " + json.dumps(
            {"output": "program", "id": "d", "depth_db": -6, "enabled": False}))
        edges["program"].get(2000)   # the change applies on the mix thread
        changed = avp.node("mix").getObject("status")
        assert not errors, errors
    finally:
        avp.shutdown()

    for bus in buses:
        assert all(block is not None and block.samplesCount == BLOCK for block in blocks[bus])
    level = {}
    for bus in buses:
        left = [v for block in blocks[bus] for v in _planes(block)[0]]
        # Before the key's edge (0.2-0.6 s in) and long after it (3-3.8 s in).
        level[bus] = (_level(left, 440, 10 * BLOCK, 20 * BLOCK), _level(left, 440, 150 * BLOCK, 40 * BLOCK))
    ratio = {bus: after / before for bus, (before, after) in level.items()}
    assert 0.07 < ratio["program"] < 0.14, ratio     # -20 dB is 0.1
    assert 0.9 < ratio["solo"] < 1.1, ratio
    assert 0.07 < ratio["solo_pre"] < 0.14, ratio

    applied = {b["id"]: b["processors"][0] for b in status["buses"]}
    assert applied["program"]["gain_db"] == -20 and applied["solo_pre"]["gain_db"] == -20
    assert applied["solo"]["gain_db"] == 0
    program = next(b for b in changed["buses"] if b["id"] == "program")["processors"][0]
    assert program["depth_db"] == -6 and program["enabled"] is False


def test_connections_fix_what_an_output_carries_and_others_stay_automatic(tmp_path):
    # Two steady tones, "mix" and "cam". Every bus takes cam, then mix. program carries only mix,
    # not following video: heard through both takes, cam never. manual carries only cam,
    # following video: heard on its take only. auto has no entry: each source on its own take.
    tones = {"mix": 440, "cam": 1000}
    for tag, frequency in tones.items():
        _tone(tmp_path / f"{tag}.wav", frequency, 10, 2)
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    buses = ("program", "manual", "auto")
    try:
        for tag in tones:
            for cls, params in (
                (Input, {"url": str(tmp_path / f"{tag}.wav"), "dst": f"{tag}_packets"}),
                (Demux, {"src": f"{tag}_packets", "routing": {"a:0": f"{tag}_audio"}}),
                (DecAudio, {"src": f"{tag}_audio", "dst": f"{tag}_decoded"}),
                (Realtime, {"src": f"{tag}_decoded", "dst": f"{tag}_paced", "set_pts": True,
                            "time_base": "1/120000", "negative_time_tolerance": .25,
                            "discontinuity_threshold": 1.0}),
            ):
                avp.addNode(cls({"name": f"{cls.TYPE}_{tag}", "group": "audio", **params}))
        avp.addNode(AudioMixer({
            "name": "mix", "group": "audio", "src": ["mix_paced", "cam_paced"], "source_ids": ["mix", "cam"],
            "outputs": {bus: bus for bus in buses}, "fps": "25/1", "latency_ms": 80,
            "connections": {"program": [{"source": "mix", "afv": False}],
                            "manual": [{"source": "cam", "afv": True}]},
        }))
        edges = {bus: avp.getEdge(bus, "AudioSamples") for bus in buses}
        avp.group("audio").startNodes()

        def take(sources):
            for bus in buses:
                avp.executeCommandsFromString("node.object.set mix transition_ended " + json.dumps(
                    {"output": bus, "sources": sources, "pts": 0}))

        blocks = {bus: [] for bus in buses}
        for i in range(200):
            for bus, edge in edges.items():
                blocks[bus].append(edge.get(2000))
            if i == 0:
                take(["cam"])
            if i == 100:
                take(["mix"])
        status = avp.node("mix").getObject("status")
        assert not errors, errors
    finally:
        avp.shutdown()

    for bus in buses:
        assert all(block is not None and block.samplesCount == BLOCK for block in blocks[bus])
    heard = {}
    for bus in buses:
        left = [v for block in blocks[bus] for v in _planes(block)[0]]
        # Well inside the cam take (blocks 20-60) and the mix take (140-180). lavfi's sine is at
        # 1/8 full scale, so a tone heard in full measures 1/16.
        heard[bus] = {(tag, take): _level(left, frequency, start * BLOCK, 40 * BLOCK) > 0.02
                      for tag, frequency in tones.items() for take, start in (("cam", 20), ("mix", 140))}
    on = lambda *pairs: {(tag, take): (tag, take) in pairs for tag in tones for take in ("cam", "mix")}
    assert heard["program"] == on(("mix", "cam"), ("mix", "mix")), heard
    assert heard["manual"] == on(("cam", "cam")), heard
    assert heard["auto"] == on(("cam", "cam"), ("mix", "mix")), heard
    connections = {b["id"]: b["connections"] for b in status["buses"]}
    assert connections == {"program": [{"source": "mix", "afv": False}],
                           "manual": [{"source": "cam", "afv": True}], "auto": "auto"}


# ingest() converts anything that is not already the mixer's own fltp/rate/layout. Each of these
# reaches a different branch, and a general-purpose mixer must place all three on the same
# timeline: the samples belong where their frame's timestamp says, not where the converter
# happened to hand them over. The frequencies rotate between runs so the detector cannot be
# what decides the answer.
@pytest.mark.parametrize("rotate", [0, 1])
def test_a_converted_source_lands_where_its_timestamps_say(tmp_path, rotate):
    # copy: already fltp 48 kHz stereo, so ingest() copies it and is the reference.
    # layout: mono - ResampleAudioForwardChannels passes the layout through (pyplumber/mixer/inputs.py).
    # rate: delivered at 44.1 kHz, so ingest() resamples rather than only re-laying out.
    tones = [440, 1000, 1600]
    paths = {
        "copy": (tones[rotate % 3], 2, RATE),
        "layout": (tones[(rotate + 1) % 3], 1, RATE),
        "rate": (tones[(rotate + 2) % 3], 2, 44100),
    }
    for tag, (frequency, channels, rate) in paths.items():
        _marked(tmp_path / f"{tag}.wav", frequency, channels, rate=rate)
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    try:
        for tag, (_, _, rate) in paths.items():
            for cls, params in (
                (Input, {"url": str(tmp_path / f"{tag}.wav"), "dst": f"{tag}_packets"}),
                (Demux, {"src": f"{tag}_packets", "routing": {"a:0": f"{tag}_audio"}}),
                (DecAudio, {"src": f"{tag}_audio", "dst": f"{tag}_decoded"}),
                # Normalizing to the source's own rate leaves the mixer the rate conversion.
                (ResampleAudioForwardChannels, {"src": f"{tag}_decoded", "dst": f"{tag}_resampled",
                                                "dst_sample_rate": rate, "dst_sample_format": "fltp",
                                                "compensation": 0}),
                (Realtime, {"src": f"{tag}_resampled", "dst": f"{tag}_paced", "set_pts": True,
                            "time_base": "1/120000", "negative_time_tolerance": .25,
                            "discontinuity_threshold": 1.0}),
            ):
                avp.addNode(cls({"name": f"{cls.TYPE}_{tag}", "group": "audio", **params}))
        avp.addNode(AudioMixer({
            "name": "mix", "group": "audio",
            "src": [f"{tag}_paced" for tag in paths], "source_ids": list(paths),
            "outputs": {"program": "program"}, "fps": "25/1", "latency_ms": 80,
        }))
        program = avp.getEdge("program", "AudioSamples")
        avp.group("audio").startNodes()
        blocks = [program.get(3000)]
        avp.executeCommandsFromString("node.object.set mix transition_ended " + json.dumps(
            {"output": "program", "sources": list(paths), "pts": 0}))
        blocks += [program.get(3000) for _ in range(199)]
        status = avp.node("mix").getObject("status")
        assert not errors, errors
    finally:
        avp.shutdown()

    assert all(block is not None and block.samplesCount == BLOCK for block in blocks)
    left = [sample for block in blocks for sample in _planes(block)[0]]
    found = {tag: _onset(left, frequency) for tag, (frequency, _, _) in paths.items()}
    detail = ", ".join(f"{tag} {paths[tag][0]} Hz at {at} (peak {peak:.4f})"
                       for tag, (at, peak) in found.items())
    assert all(at is not None for at, _ in found.values()), f"a tone was never detected: {detail}"
    # Every source carries the same edge at the same media time, so every onset is the same
    # sample of the mix. One detector window of slack covers the inputs' start jitter.
    reference = found["copy"][0]
    for tag, (at, _) in found.items():
        offset = at - reference
        assert abs(offset) <= BLOCK, (
            f"{tag} is {offset} samples ({1000.0 * offset / RATE:.1f} ms) off the copy path: {detail}")
    # Converted audio arrives per frame, not withheld into batches: a source whose samples are
    # held back until a 4096-sample buffer fills reads as missing for that long at the start.
    sources = {s["id"]: s for s in status["sources"]}
    for tag in paths:
        assert sources[tag]["missing"] < RATE // 50, f"{tag}: {sources[tag]}"
        assert sources[tag]["dropped"] == 0 and sources[tag]["jumps"] == 0, f"{tag}: {sources[tag]}"
    assert status["output_drops"] == 0
