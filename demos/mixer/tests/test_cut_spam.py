import pytest

from cut_spam import (is_expected_rejection, latest_id, measured, parse_mix, percentile, pick_scenes,
                      setup_check, summarize, thresholds, transition_payloads, verdict)


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


def test_frame_term_rules_at_25_fps_and_ms_floor_at_60():
    at25, at60 = thresholds(25, 100), thresholds(60, 100)
    assert (at25["spam_p95_ms"], at25["spam_max_ms"], at25["recovery_max_ms"]) == (320, 600, 240)
    assert (at60["spam_p95_ms"], at60["spam_max_ms"], at60["recovery_max_ms"]) == (150, 300, 120)
    assert at25["recovery_p50_ms"] == pytest.approx(165)
    assert at60["recovery_p50_ms"] == 141.7   # 125 + 16.67, rounded to 0.1 ms


def test_overrides_win_and_missing_baseline_leaves_no_recovery_limit():
    limits = thresholds(60, None, {"spam_p95_ms": 200, "spam_max_ms": None})
    assert limits["spam_p95_ms"] == 200 and limits["spam_max_ms"] == 300
    assert limits["recovery_p50_ms"] is None


def test_measured_collects_both_categories_and_recent_history_after_an_id():
    assert latest_id(PROBE) == 7
    assert sorted(measured(PROBE, 0)) == [2, 3, 4, 5, 6]
    assert {i: s["ms"] for i, s in measured(PROBE, 4).items()} == {5: 110, 6: 120}
    assert measured(None, 0) == {} and latest_id(None) == 0


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
    limits = thresholds(60, 100)
    ok = {"p50": 100, "p95": 140, "max": 200}
    criteria = dict((n, c) for n, c, _ in verdict(limits, ok, {"p50": 100, "max": 110}, [],
                                                  (True, ""), (True, "")))
    assert all(criteria.values())
    empty = summarize([])
    criteria = dict((n, c) for n, c, _ in verdict(limits, empty, empty, ["boom"], (True, ""), (False, "")))
    assert not any(criteria[n] for n in ("no errors", "spam p95", "spam max", "recovery p50",
                                         "program on last target"))


def test_only_known_rejections_are_expected():
    assert is_expected_rejection("400 mixer: transition already in progress")
    assert not is_expected_rejection("mixer: unknown scene: x")
    assert not is_expected_rejection(None)
