import pytest

from cut_spam import (PvwSamples, eligible_cuts, is_expected_rejection, last_states, latest_id, measured,
                      parse_mix, percentile, pick_scenes, playout_delta, pvw_line, sample as probe_sample,
                      setup_check, summarize, thresholds, transition_payloads, unmeasured_by_state, verdict)


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
    assert (at60["measured_ratio_min"], at60["playout_repeats_max"]) == (0.9, None)


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


def test_unmeasured_cuts_are_counted_by_the_last_state_the_poller_saw():
    assert last_states(PROBE, 4) == {7: "superseded", 5: "measured", 6: "measured"}
    assert last_states(None, 0) == {}
    states = {11: "interrupted", 12: "measured", 13: "pending", 14: "interrupted"}
    assert unmeasured_by_state({11, 13, 14, 15}, states) == {"interrupted": 2, "pending": 1, "unseen": 1}
    assert unmeasured_by_state(set(), states) == {}


def test_only_cuts_the_next_take_left_time_for_are_eligible():
    takes = [(0.0, "cut", "sent"),     # probe 11: the fade 100 ms later cancels it
             (0.1, "fade", "sent"),
             (0.5, "cut", "sent"),     # probe 12: 300 ms to the next take
             (0.8, "wipe", None),      # rejected, still cancels a pending cut
             (0.85, "cut", "sent"),    # probe 13: 150 ms, short of the gap
             (1.0, "cut", "sent")]     # probe 14: the pause before recovery follows
    assert eligible_cuts(takes, 10, 0.2) == {12, 14}
    assert eligible_cuts([], 10, 0.2) == set()


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
    assert transition_payloads({"fade_color": "#000000"})["fade"] == {"duration_sec": 0.5, "color": "#000000"}
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
    none_eligible = {**ok, "eligible": 0, "measured_ratio": None}
    spam = {n: (c, d) for n, c, d in verdict(limits, none_eligible, ok, ok, stalled, [], (True, ""), (True, ""))}
    assert spam["spam measured"][0] is False and "lower --rate" in spam["spam measured"][1]


def test_only_known_rejections_are_expected():
    assert is_expected_rejection("400 mixer: transition already in progress")
    assert not is_expected_rejection("mixer: unknown scene: x")
    assert not is_expected_rejection(None)


def test_pvw_samples_count_each_follower_change_once_by_kind():
    pvw = PvwSamples()
    assert pvw.summary() is None and pvw_line(None).startswith("pvw       no AUX")
    change = lambda rev, diff, late=0, kind="cut", **extra: {"applied_revision": rev, "align": "program", "kind": kind,
                                                             "pvw_minus_pgm_ms": diff, "pvw_latency_ms": 60 + diff,
                                                             "pgm_latency_ms": 60.0, "last_target_error_ticks": late, **extra}
    pvw.observe({"pvw_latency": {"aux_mv_pvw": change(1, 0.0)}})
    pvw.observe({"pvw_latency": {"aux_mv_pvw": change(1, 0.0)}})            # the same change, polled twice
    pvw.observe({"pvw_latency": {"aux_mv_pvw": change(2, 16.7)}})
    pvw.observe({"pvw_latency": {"aux_mv_pvw": change(3, 33.3, late=1), "aux_mv2_pvw": change(3, 0.0)}})
    pvw.observe({"pvw_latency": {"aux_mv_pvw": {**change(4, 0.0), "pvw_minus_pgm_ms": None}}})   # untimed: a wipe
    # A fade's target tick had passed when the mixer published the swap: a tick late, not a miss.
    pvw.observe({"pvw_latency": {"aux_mv_pvw": change(5, 33.3, late=1, kind="fade", target_unreachable=True,
                                                      pvw_latency_ms=1093.3, pgm_latency_ms=1060.0)}})
    pvw.observe({"pvw_latency": {"aux_mv_pvw": change(6, 33.3, late=1, kind="cut", target_unreachable=True)}})
    pvw.observe({"cut_latency": {}})                                        # a mixer without followers
    pvw.observe(None)
    summary = pvw.summary()
    assert summary["followers"] == ["aux_mv2_pvw", "aux_mv_pvw"] and summary["align"] == ["program"]
    assert sorted(summary["kinds"]) == ["cut", "fade"]
    cuts, fades = summary["kinds"]["cut"], summary["kinds"]["fade"]
    assert cuts["n"] == 5 and cuts["pvw_minus_pgm"] == {"n": 5, "p50": 16.7, "p95": 33.3, "max": 33.3}
    assert cuts["pvw_latency"]["max"] == 93.3 and cuts["pgm_latency"]["p50"] == 60.0   # no fade in the cuts' latencies
    assert (cuts["late"], cuts["unreachable"]) == (1, 1)   # the unreachable cut (a +-1 tick race) is not late
    assert (fades["n"], fades["late"], fades["unreachable"], fades["pgm_latency"]["max"]) == (1, 0, 1, 1060.0)
    line = pvw_line(summary)
    assert "cut n=5" in line and "1 missed their tick, 1 unreachable" in line and "; fade n=1" in line
    assert "aux_mv2_pvw, aux_mv_pvw (program)" in line and "p50=16.7" in line
    only_fades = PvwSamples()
    only_fades.observe({"pvw_latency": {"aux_mv_pvw": change(1, 33.3, late=1, kind="fade", target_unreachable=True)}})
    assert pvw_line(only_fades.summary()).startswith("pvw       aux_mv_pvw (program): fade n=1 ") and "unreachable" in pvw_line(only_fades.summary())
