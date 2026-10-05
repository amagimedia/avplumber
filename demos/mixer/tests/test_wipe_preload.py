"""A wipe clip is cached whole or startup fails: pyplumber.mixer.clipcache.preload."""
from pathlib import Path

import pytest

from pyplumber.mixer import clipcache
from wipe_loader_sim import WipeLoaderSim

CLIP = "/media/media_wipes/diagonal.mov"
SHORT = {"decoded": 120, "cached": 2}      # the load a stalled upload used to leave behind
LONG = {"decoded": 120, "cached": 129}     # the next clip, with the first one's late frames


def preload(sim, clip=CLIP, timeout_sec=5.0):
    return clipcache.preload(sim, "mixer", clip, timeout_sec=timeout_sec, poll_sec=0.001)


def test_a_whole_clip_is_loaded_once_and_its_loader_stopped():
    sim = WipeLoaderSim()
    assert preload(sim)["frames"] == 120
    assert sim.events[:3] == [f"load {CLIP}", "start", "stop"]
    assert sim.loads == [CLIP]


def test_a_clip_already_cached_is_returned_without_a_load():
    sim = WipeLoaderSim()
    sim.clips[CLIP] = 120
    assert preload(sim)["frames"] == 120
    assert sim.events == []


@pytest.mark.parametrize("bad", (SHORT, LONG))
def test_a_clip_with_the_wrong_frame_count_is_dropped_and_loaded_again(bad):
    sim = WipeLoaderSim()
    sim.script = [bad]
    assert preload(sim)["frames"] == 120
    assert sim.loads == [CLIP, CLIP]
    assert sim.events.index(f"forget {CLIP}") < len(sim.events) - sim.events[::-1].index(f"load {CLIP}") - 1


@pytest.mark.parametrize("bad, cached", ((SHORT, 2), (LONG, 129)))
def test_a_clip_wrong_twice_fails_startup_naming_the_clip_and_both_counts(bad, cached):
    sim = WipeLoaderSim()
    sim.script = [bad, bad]
    with pytest.raises(RuntimeError, match=rf"{CLIP}.* {cached} .* 120 "):
        preload(sim)
    assert sim.loads == [CLIP, CLIP] and CLIP not in sim.clips


def test_silence_is_not_the_end_and_the_timeout_is_a_failure():
    sim = WipeLoaderSim()
    sim.script = [{"decoded": 2, "cached": 2, "ends": False}]   # two frames, then nothing
    with pytest.raises(RuntimeError, match=f"timed out.*{CLIP}"):
        preload(sim, timeout_sec=0.05)
    assert sim.loads == [CLIP]


def test_a_clip_that_ends_uncached_is_loaded_again_like_a_short_one():
    nothing = {"decoded": 120, "cached": 0}   # over the budget, or no picture at all
    sim = WipeLoaderSim()
    sim.script = [nothing]
    assert preload(sim)["frames"] == 120

    sim = WipeLoaderSim()
    sim.script = [nothing, nothing]
    with pytest.raises(RuntimeError, match=f"{CLIP}.*without being cached"):
        preload(sim, timeout_sec=60.0)
    assert sim.loads == [CLIP, CLIP]


def test_the_next_load_waits_until_the_earlier_chain_and_its_frames_are_gone():
    sim = WipeLoaderSim()
    sim.script = [{**SHORT, "stop_polls": 4, "leftover": 3}]
    preload(sim)
    assert sim.dirty_loads == []
    second = len(sim.events) - sim.events[::-1].index(f"load {CLIP}") - 1
    before = sim.events[:second]
    assert before.index("stop") < before.index("clear mixer_wipe_dec_out")
    assert {e for e in before if e.startswith("clear ")} == {"clear mixer_" + e for e in clipcache.LOADER_EDGES}


def test_a_failing_node_aborts_the_wait():
    sim = WipeLoaderSim()
    sim.script = [{"decoded": 2, "cached": 2, "ends": False}]
    def check():
        raise RuntimeError("Mixer startup failed at mixer_wipe_dec")
    with pytest.raises(RuntimeError, match="mixer_wipe_dec"):
        clipcache.preload(sim, "mixer", CLIP, timeout_sec=60.0, poll_sec=0.001, check=check)


def test_compiled_gpu_kernels_are_kept_on_the_media_volume():
    # deploy/l4/compose.yaml is layered over this file, so one entry covers both.
    compose = (Path(__file__).resolve().parents[1] / "compose.yaml").read_text()
    assert "CUDA_CACHE_PATH: /media/" in compose and ":/media:rw" in compose
