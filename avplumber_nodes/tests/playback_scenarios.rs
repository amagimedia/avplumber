//! The seek/pause/speed scenario sweeps of the replay demo's old native suite
//! (`test_playback_integration.py`, which drove the C++ core through
//! pyplumber), on the Rust core and against pixel ground truth. Each test
//! keeps its predecessor's shape: the same targets, the same speeds, the same
//! oracle — a paused seek shows exactly its frame and nothing else — but the
//! frame a sink sees is identified by its content, not by a counter.
//!
//! The harness is `common::player`. The tests pace at real time, so the crate
//! runs them single-threaded.

#![cfg(all(feature = "ffmpeg", feature = "async"))]

use std::time::Duration;

mod common;

use common::player::{Config, Player, SeekKind};

/// The silence window after a paused seek in the sweeps: two frame periods
/// at 30 fps. A stray frame from a wrong cutoff shows up at once.
const SETTLE: Duration = Duration::from_millis(60);
const POINT: Duration = Duration::from_millis(150);

/// The three ways of naming a target, cycled through the sweeps.
const KINDS: [SeekKind; 3] = [SeekKind::Media, SeekKind::Relative, SeekKind::Utc];

fn set_speed(p: &Player, rate: f64) {
    p.cmd(&format!("speed.set replay {rate}"));
    // A sign or stride change re-shows the frame on screen; let that settle
    // before the next marker is taken.
    std::thread::sleep(Duration::from_millis(100));
}

/// Old `test_paused_seek_frame_boundaries`, at every frame rate the demo's
/// suite ran (24, 25, 30 and 60 fps stress the millisecond rounding of the
/// seek table differently) and, at 30 fps, at every speed and direction. A
/// paused seek around the boundary between frames `fps-1` and `fps`, both
/// sides of the nearest-frame tie and the old 7 ms discard cutoff included,
/// lands on the nearest frame whatever the rate, and resuming afterwards
/// moves in the configured direction.
#[test]
fn paused_seeks_resolve_frame_boundaries_at_every_fps_and_speed() {
    if common::skip_without_ffmpeg("paused_seeks_resolve_frame_boundaries_at_every_fps_and_speed") {
        return;
    }
    for fps in [24u32, 25, 30, 60] {
        let p = Player::with(&format!("bounds{fps}"), Config::new(fps, 4, true));
        p.start_paused();
        let lo = p.ms(fps as usize - 1);
        let hi = p.ms(fps as usize);
        let mid = (lo + hi) / 2;
        let mut targets = vec![
            lo,
            lo + 1,
            lo + 6,
            lo + 7,
            lo + 8,
            mid - 1,
            mid,
            mid + 1,
            hi - 1,
            hi,
        ];
        targets.sort_unstable();
        targets.dedup();
        let speeds: &[f64] = if fps == 30 {
            &[0.25, 0.5, 1.0, 2.0, -0.25, -0.5, -1.0, -2.0]
        } else {
            &[1.0, -1.0]
        };
        for &speed in speeds {
            set_speed(&p, speed);
            for kind in KINDS {
                for (i, &target) in targets.iter().enumerate() {
                    let origin = if i % 2 == 0 { 0 } else { 2000 };
                    p.seek_paused(SeekKind::Media, origin, SETTLE);
                    let landed = p.seek_paused(kind, target, SETTLE);
                    assert_eq!(
                        landed,
                        p.nearest(target),
                        "{fps} fps, {speed}x, {kind:?} seek to {target} ms"
                    );
                }
            }
            let before = p.frame();
            let m = p.marker();
            p.cmd("resume replay");
            p.wait_frames(m, &format!("resume at {speed}x after paused seeks"), |f| {
                f.last().is_some_and(|&last| {
                    if speed < 0.0 {
                        last < before
                    } else {
                        last > before
                    }
                })
            });
            p.cmd("pause replay now");
            p.assert_stable(&format!("{fps} fps, paused again after {speed}x"));
        }
    }
}

/// Old `test_paused_seek_endpoints_repeated_and_seeded_targets`: both ends,
/// a repeated target, neighbouring targets, forty seeded random ones, each
/// named a different way — then all of them queued without waiting, where
/// the last command decides what is held.
#[test]
fn paused_seek_endpoints_repeated_and_seeded_targets() {
    if common::skip_without_ffmpeg("paused_seek_endpoints_repeated_and_seeded_targets") {
        return;
    }
    let p = Player::new("targets", true);
    p.start_paused();
    let duration = p.duration_ms();
    let mut targets = vec![0, 1, duration - 1, duration, 983, 983, 984, 983, 0];
    let mut seed = 0x35983u64;
    for _ in 0..40 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        targets.push(((seed >> 33) % (duration as u64 + 1)) as i64);
    }
    for (i, &target) in targets.iter().enumerate() {
        p.seek_paused(KINDS[i % 3], target, SETTLE);
    }
    let m = p.marker();
    for &target in &targets {
        p.cmd(&format!("seek replay now {}", p.start_ms + target));
    }
    let last = p.nearest(983);
    p.cmd(&format!("seek replay now {}", p.start_ms + 983));
    p.wait_frames(m, "the last of the queued seeks", |f| {
        f.last() == Some(&last)
    });
    std::thread::sleep(POINT);
    assert_eq!(
        p.last_frame(),
        Some(last),
        "the last command decides the hold"
    );
    assert_eq!(p.frame(), last);
    p.assert_stable("after the queued seeks");
}

/// Old `test_frame_steps_after_unaligned_seek_and_speed_changes`: from a
/// target between two frames, every frame nudge lands exactly `delta` frames
/// away, whatever speed is configured.
#[test]
fn frame_steps_after_unaligned_seek_and_speed_changes() {
    if common::skip_without_ffmpeg("frame_steps_after_unaligned_seek_and_speed_changes") {
        return;
    }
    let p = Player::new("steps", true);
    p.start_paused();
    for speed in [2.0, 0.5, 1.0, -1.0, 1.0] {
        set_speed(&p, speed);
        for delta in [-30i64, -5, -1, 0, 1, 5, 30] {
            let base = p.seek_paused(SeekKind::Media, 1983, SETTLE);
            let m = p.marker();
            p.cmd(&format!("seek replay frame {delta:+}"));
            p.assert_paused_at_within(
                m,
                (base as i64 + delta) as usize,
                &format!("frame step {delta:+} at {speed}x"),
                SETTLE,
            );
        }
    }
    p.assert_stable("after the frame steps");
}

/// Old `test_absolute_frame_seeks_clamp_at_recording_end`.
#[test]
fn absolute_frame_seeks_clamp_at_the_recording_end() {
    if common::skip_without_ffmpeg("absolute_frame_seeks_clamp_at_the_recording_end") {
        return;
    }
    let p = Player::new("clamp_abs", true);
    p.start_paused();
    let count = p.count;
    for target in [0, count - 1, count, count + 1, 0] {
        let m = p.marker();
        p.cmd(&format!("seek replay frame {target}"));
        p.assert_paused_at_within(
            m,
            target.min(count - 1),
            &format!("absolute frame seek {target}"),
            SETTLE,
        );
    }
    p.assert_stable("after the clamped seeks");
}

/// Old `test_relative_seeks_clamp_at_both_recording_edges`: a nudge of one
/// or thirty frames, or one or thirty seconds, past either edge stays on the
/// edge frame.
#[test]
fn relative_seeks_clamp_at_both_recording_edges() {
    if common::skip_without_ffmpeg("relative_seeks_clamp_at_both_recording_edges") {
        return;
    }
    let p = Player::new("clamp_rel", true);
    p.start_paused();
    for (edge, sign) in [(0, -1i64), (p.duration_ms(), 1)] {
        let expected = p.nearest(edge);
        for (unit, amount) in [("frame", 1), ("frame", 30), ("now", 1000), ("now", 30000)] {
            p.seek_paused(SeekKind::Media, edge, SETTLE);
            let m = p.marker();
            let line = format!("seek replay {unit} {:+}", sign * amount);
            p.cmd(&line);
            p.assert_paused_at_within(
                m,
                expected,
                &format!("`{line}` at the edge {edge} ms"),
                SETTLE,
            );
        }
    }
    p.assert_stable("after the edge nudges");
}

/// Old `test_unaligned_seek_with_nvdec_display_buffering`, whose point was
/// that the interactive graph's `low_delay` decoder must not be what makes
/// paused seeks exact: with the decoder at its defaults, frame threads and
/// their delay included, the same unaligned targets still surface exactly
/// their frame.
#[test]
fn paused_seeks_are_exact_with_a_buffering_decoder() {
    if common::skip_without_ffmpeg("paused_seeks_are_exact_with_a_buffering_decoder") {
        return;
    }
    let mut cfg = Config::new(30, 4, true);
    cfg.low_delay = false;
    let p = Player::with("buffered", cfg);
    p.start_paused();
    for target in [0, 974, 0, 983, 2000, 983, 0, 984, 1983] {
        p.seek_paused(SeekKind::Media, target, POINT);
    }
    p.assert_stable("after the buffered-decoder seeks");
}

/// Old `test_playing_seeks_pause_and_speed_transitions`: at each speed, a
/// paused seek, play, a seek *while playing* that lands on its target and
/// carries on from there at that speed's stride, pause, and a paused seek
/// again.
#[test]
fn playing_seeks_pause_and_speed_transitions() {
    if common::skip_without_ffmpeg("playing_seeks_pause_and_speed_transitions") {
        return;
    }
    let p = Player::new("transitions", true);
    p.start_paused();
    for speed in [0.5, 1.0, 2.0, 1.0, 0.5, 2.0] {
        set_speed(&p, speed);
        p.cmd("pause replay now");
        p.seek_paused(SeekKind::Media, 1983, POINT);
        p.cmd("resume replay");
        p.wait_status(&format!("play advances at {speed}x"), |s| {
            s["position_ms"].as_i64().unwrap() > 2200
        });
        let target = p.nearest(983);
        let m = p.marker();
        p.cmd(&format!("seek replay now {}", p.start_ms + 983));
        let frames = p.wait_frames(m, &format!("a playing seek at {speed}x"), |f| f.len() >= 8);
        // A frame in flight at the instant of the command may still go out.
        let at = frames
            .iter()
            .position(|&f| f == target)
            .unwrap_or_else(|| panic!("the target never showed at {speed}x: {frames:?}"));
        assert!(at <= 1, "the target is the first fresh frame: {frames:?}");
        let stride = if speed == 2.0 { 2 } else { 1 };
        assert!(
            Player::steps(&frames[at..at + 6])
                .iter()
                .all(|s| *s == stride),
            "playback continues from the target at stride {stride}: {frames:?}"
        );
        let position = p.position_ms();
        assert!(
            (900..=1600).contains(&position),
            "position follows the playing seek: {position}"
        );
        p.cmd("pause replay now");
        p.assert_stable(&format!("paused after the playing seek at {speed}x"));
        p.seek_paused(SeekKind::Media, 983, POINT);
    }
}

/// Old `test_repeated_active_speed_changes_keep_frame_continuity`: while
/// playing, a dozen speed changes each settle into the right cadence and
/// never step backwards against the direction — the frame on screen may
/// repeat once at a stride change, which is where the run starts.
fn speed_changes_keep_continuity(reverse: bool) {
    let p = Player::with(
        if reverse { "cont_rev" } else { "cont_fwd" },
        Config::new(30, 8, true),
    );
    p.start_paused();
    p.seek_paused(SeekKind::Media, if reverse { 6000 } else { 1000 }, POINT);
    set_speed(&p, if reverse { -1.0 } else { 1.0 });
    p.cmd("resume replay");
    for speed in [2.0, 1.0, 0.25, 1.0, 2.0, 0.5, 0.25, 0.5, 1.0, 0.5, 2.0, 1.0] {
        p.cmd(&format!(
            "speed.set replay {}",
            if reverse { -speed } else { speed }
        ));
        let m = p.marker();
        let frames = p.wait_frames(m, &format!("output at {speed}x"), |f| f.len() >= 11);
        let run = &frames[1..11];
        let sign = if reverse { -1 } else { 1 };
        let deltas: Vec<i64> = Player::steps(run).iter().map(|s| s * sign).collect();
        assert!(
            deltas.iter().all(|d| *d >= 0),
            "never against the direction at {speed}x: {frames:?}"
        );
        if speed == 1.0 {
            assert!(
                deltas.iter().all(|d| *d == 1),
                "consecutive at 1x: {frames:?}"
            );
        } else if speed < 1.0 {
            assert!(
                deltas.iter().all(|d| *d == 0 || *d == 1) && deltas.contains(&1),
                "no skipping below 1x: {frames:?}"
            );
        } else {
            assert!(
                deltas.iter().sum::<i64>() >= deltas.len() as i64 * 3 / 2,
                "skipping at {speed}x: {frames:?}"
            );
        }
        assert_eq!(p.status()["paused"], serde_json::json!(false));
    }
    p.cmd("pause replay now");
    p.assert_stable("paused after the speed changes");
    p.seek_paused(SeekKind::Media, 3983, POINT);
}

#[test]
fn repeated_active_speed_changes_keep_frame_continuity_forward() {
    if common::skip_without_ffmpeg("repeated_active_speed_changes_keep_frame_continuity_forward") {
        return;
    }
    speed_changes_keep_continuity(false);
}

#[test]
fn repeated_active_speed_changes_keep_frame_continuity_reverse() {
    if common::skip_without_ffmpeg("repeated_active_speed_changes_keep_frame_continuity_reverse") {
        return;
    }
    speed_changes_keep_continuity(true);
}

/// Old `test_pause_and_seek_after_recording_end`, looping and not: play
/// through the end, pause, and seeking back plus resuming both work.
#[test]
fn pause_and_seek_after_the_recording_end() {
    if common::skip_without_ffmpeg("pause_and_seek_after_the_recording_end") {
        return;
    }
    for loop_ in [true, false] {
        let p = Player::new(if loop_ { "end_loop" } else { "end_hold" }, loop_);
        p.start_paused();
        let last = p.count - 1;
        p.seek_paused(SeekKind::Media, p.ms(last - 4), POINT);
        let m = p.marker();
        p.cmd("resume replay");
        if loop_ {
            p.wait_frames(m, "the recording loops", |f| f.iter().any(|&x| x < 30));
        } else {
            p.wait_frames(m, "the final frame", |f| f.last() == Some(&last));
            std::thread::sleep(Duration::from_millis(200));
            assert_eq!(p.status()["at_end"], serde_json::json!(true));
        }
        p.cmd("pause replay now");
        p.assert_stable(&format!("paused after the end, loop={loop_}"));
        p.seek_paused(SeekKind::Media, 1000, POINT);
        let from = p.seek_paused(SeekKind::Media, 983, POINT);
        let m = p.marker();
        p.cmd("resume replay");
        let frames = p.wait_frames(m, "resume after the end", |f| f.len() >= 3);
        assert_eq!(&frames[..3], &[from + 1, from + 2, from + 3]);
        assert!(p.position_ms() > 1000);
    }
}

/// Old `test_seek_while_eof_frames_are_draining`: with no loop, pausing and
/// seeking back to the start while the last frames are going out, eight
/// times over, always shows exactly frame 0.
#[test]
fn seeks_while_the_tail_drains() {
    if common::skip_without_ffmpeg("seeks_while_the_tail_drains") {
        return;
    }
    let p = Player::new("drain", false);
    p.start_paused();
    let last = p.count - 1;
    for round in 0..8 {
        p.cmd("pause replay now");
        p.seek_paused(SeekKind::Media, p.ms(last - 3), SETTLE);
        let m = p.marker();
        p.cmd("resume replay");
        p.wait_frames(m, &format!("the tail drains, round {round}"), |f| {
            f.last().is_some_and(|&x| x >= last - 2)
        });
        p.cmd("pause replay now");
        p.seek_paused(SeekKind::Media, 0, SETTLE);
        p.assert_stable(&format!("back at the start, round {round}"));
    }
}

/// The old demux lifecycle suite bounded shutdown with subprocess deadlines;
/// here the graph itself must stop promptly whatever the player is doing:
/// paused, playing, holding the last frame, or in the middle of a burst of
/// seeks.
#[test]
fn stopping_is_prompt_in_every_state() {
    if common::skip_without_ffmpeg("stopping_is_prompt_in_every_state") {
        return;
    }
    let limit = Duration::from_secs(2);

    let mut p = Player::new("stop_paused", true);
    p.start_paused();
    let took = p.stop();
    assert!(took < limit, "stopping while paused took {took:?}");

    let mut p = Player::new("stop_playing", true);
    p.cmd("group.start p");
    p.wait_frames(0, "playback", |f| f.len() >= 5);
    let took = p.stop();
    assert!(took < limit, "stopping while playing took {took:?}");

    let mut p = Player::new("stop_end", false);
    p.start_paused();
    let last = p.count - 1;
    p.seek_paused(SeekKind::Media, p.ms(last - 2), POINT);
    p.cmd("resume replay");
    p.wait_status("the end", |s| s["at_end"] == serde_json::json!(true));
    let took = p.stop();
    assert!(took < limit, "stopping at the end took {took:?}");

    let mut p = Player::new("stop_seeking", true);
    p.start_paused();
    for i in 0..20 {
        p.cmd(&format!(
            "seek replay frame {}",
            if i % 2 == 0 { 20 } else { 90 }
        ));
    }
    let took = p.stop();
    assert!(took < limit, "stopping mid-seeks took {took:?}");
    let outcomes = p.cmd("group.status p");
    assert!(
        !outcomes.contains("failed") && !outcomes.contains("panicked"),
        "clean outcomes after a stop mid-seeks: {outcomes}"
    );
}
