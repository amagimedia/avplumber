import pytest

from cut_spam import (is_expected_rejection, latest_id, measured, parse_mix, percentile, pick_scenes,
                      playout_delta, sample as probe_sample, setup_check, summarize, thresholds,
                      transition_payloads, verdict)


def sample(id, state="measured", ms=100.0):
    return {"id": id, "scene": "s", "state": state, "ms": ms if state == "measured" else None}


PROBE = {"endpoint": "encoder_output", "encoder": "janus_encoder",
         "direct": {**sample(7, "superseded"), "recent": [sample(3, ms=90), sample(5, ms=110), sample(6, ms=120)]},
         "previewed": {**sample(4, ms=80), "recent": [sample(2, ms=70), sample(4, ms=80)]}}


def test_percentile_is_nearest_rank():
    assert percentile([], 95) is None
    assert percentile([42.0], 95) == 42.0
    assert percentile(range(1, 21), 95) == 19
    assert percentile(range(1, 21), 50) == 10


def test_summary_of_nothing_has_no_latency():
    assert summarize([]) == {"n": 0, "p50": None, "p95": None, "max": None}


BASE = {"p50": 100, "max": 130}


def test_limits_add_frames_to_the_runs_own_baseline():
    at25, at60 = thresholds(25, BASE), thresholds(60, BASE)
    assert (at25["spam_p95_ms"], at25["spam_max_ms"]) == (190, 270)   # p50 + 2 F + 10, p50 + 4 F + 10
    assert (at25["recovery_p50_ms"], at25["recovery_max_ms"]) == (140, 170)   # p50 + F, max + F
    assert (at60["spam_p95_ms"], at60["spam_max_ms"]) == (143.3, 176.7)
    assert (at60["recovery_p50_ms"], at60["recovery_max_ms"]) == (116.7, 146.7)
    assert (at60["measured_ratio_min"], at60["playout_repeats_max"]) == (0.9, 0)


def test_overrides_win_and_missing_baseline_leaves_no_latency_limit():
    limits = thresholds(60, {"p50": None, "max": None},
                        {"spam_p95_ms": 200, "spam_max_ms": None, "measured_ratio_min": 0.95})
    assert limits["spam_p95_ms"] == 200 and limits["spam_max_ms"] is None
    assert limits["recovery_p50_ms"] is None and limits["measured_ratio_min"] == 0.95


def test_measured_collects_both_categories_and_recent_history_after_an_id():
    assert latest_id(PROBE) == 7
    assert sorted(measured(PROBE, 0)) == [2, 3, 4, 5, 6]
    assert {i: s["ms"] for i, s in measured(PROBE, 4).items()} == {5: 110, 6: 120}
    assert measured(None, 0) == {} and latest_id(None) == 0


def test_a_burst_sample_is_its_last_cut_once_measured():
    assert probe_sample(PROBE, 5, "s") == 110
    assert probe_sample(PROBE, 7, "s") is None   # superseded, never measured
    assert probe_sample(PROBE, 5, "other") is None
    assert probe_sample(None, 1, "s") is None


def test_playout_counters_sum_both_slots_and_must_advance():
    before = {"A": {"frames": 60, "repeats": 1, "missed_deadlines": 0}, "B": {"frames": 0}}
    after = {"A": {"frames": 180, "repeats": 1, "missed_deadlines": 2},
             "B": {"frames": 120, "repeats": 3, "missed_deadlines": 0}}
    assert playout_delta(before, after) == {"frames": 240, "repeats": 3, "missed_deadlines": 2}
    assert playout_delta(after, after) is None   # nothing published during the run
    assert playout_delta(None, after) is None and playout_delta({}, {}) is None


def test_mix_and_scene_selection():
    assert parse_mix("6:2:2") == {"cut": 6, "fade": 2, "wipe": 2}
    for bad in ("6:2", "1:-1:0", "0:0:0", "a:b:c"):
        with pytest.raises(ValueError):
            parse_mix(bad)
    names = ["aux_1", "fullscreen_000", "fullscreen_001", "grid_4"]
    assert pick_scenes(names) == ["fullscreen_000", "fullscreen_001", "grid_4"]
    assert pick_scenes(names, prefix="full") == ["fullscreen_000", "fullscreen_001"]
    with pytest.raises(ValueError, match="two scenes"):
        pick_scenes(names, prefix="grid")
    with pytest.raises(ValueError, match="unknown"):
        pick_scenes(names, ["grid_4", "nope"])


def test_payloads_match_the_web_ui():
    settings = {"fade_seconds": 0.8, "fade_curve": "ease-in", "default_wipe": "b",
                "wipes": [{"id": "a", "path": "/a.mov", "duration_seconds": 1.0},
                          {"id": "b", "path": "/b.mov", "duration_seconds": 0}]}
    payloads = transition_payloads(settings)
    assert payloads["fade"] == {"duration_sec": 0.8, "curve": "ease-in"}
    assert payloads["wipe"] == {"wipe_file": "/b.mov"}   # unknown length: the mixer probes the clip
    assert transition_payloads({})["fade"] == {"duration_sec": 0.5}
    assert transition_payloads({})["wipe"] is None


def test_setup_must_keep_running_the_same_revision():
    assert setup_check(None, None)[0]
    assert setup_check({"phase": "running", "revision": 3}, {"phase": "running", "revision": 3})[0]
    assert not setup_check({"phase": "running", "revision": 3}, {"phase": "running", "revision": 4})[0]
    assert not setup_check({"phase": "running", "revision": 3}, {"phase": "error", "revision": 3})[0]


def test_verdict_fails_a_criterion_without_samples():
    limits = thresholds(60, BASE)
    ok = {"p50": 110, "p95": 140, "max": 170, "measured_ratio": 0.95}
    clean = {"frames": 3600, "repeats": 0, "missed_deadlines": 0}
    criteria = dict((n, c) for n, c, _ in verdict(limits, ok, ok, {"p50": 110, "max": 140}, clean, [],
                                                  (True, ""), (True, "")))
    assert all(criteria.values())
    empty = {**summarize([]), "measured_ratio": None}
    criteria = dict((n, c) for n, c, _ in verdict(limits, empty, empty, empty, None, ["boom"],
                                                  (True, ""), (False, "")))
    assert not any(criteria[n] for n in ("no errors", "spam p95", "spam max", "spam measured", "burst p95",
                                         "burst max", "burst measured", "recovery p50", "recovery max",
                                         "program missed deadlines", "program repeats",
                                         "program on last target"))


def test_verdict_fails_slow_measurement_and_any_program_stall():
    limits = thresholds(60, BASE, {"playout_repeats_max": 2})
    ok = {"p50": 110, "p95": 140, "max": 170, "measured_ratio": 0.95}
    stalled = {"frames": 3600, "repeats": 2, "missed_deadlines": 1}
    criteria = dict((n, c) for n, c, _ in verdict(limits, {**ok, "measured_ratio": 0.8}, ok, ok, stalled,
                                                  [], (True, ""), (True, "")))
    assert not criteria["spam measured"] and criteria["burst measured"]
    assert not criteria["program missed deadlines"] and criteria["program repeats"]


def test_only_known_rejections_are_expected():
    assert is_expected_rejection("400 mixer: transition already in progress")
    assert not is_expected_rejection("mixer: unknown scene: x")
    assert not is_expected_rejection(None)
