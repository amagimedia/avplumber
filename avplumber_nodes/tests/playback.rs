//! Playback control against ground truth, point checks: seeks land on the
//! exact frame and hold it, play/speed/reverse release the right frames in
//! order, and the tail loops or holds. The harness — the recording, the
//! player graph and the pixel-identified capture — is `common::player`; the
//! scenario sweeps ported from the demo's old suite are in
//! `playback_scenarios.rs`.
//!
//! The CPU backend of the demo's own suite: software H.264 in, libx264 out,
//! one decoder thread so a paused seek can surface its one frame.

#![cfg(all(feature = "ffmpeg", feature = "async"))]

use std::time::{Duration, Instant};

mod common;

use common::player::{Player, WALLCLOCK_ORIGIN_MS, settled_run};

const FPS: u32 = 30;

// ------------------------------------------------------------------ tests

#[test]
fn seeks_land_on_the_exact_frame_and_hold_it_while_paused() {
    if common::skip_without_ffmpeg("seeks_land_on_the_exact_frame_and_hold_it_while_paused") {
        return;
    }
    let p = Player::new("seek", true);
    p.cmd("group.start p");
    p.wait_frames(0, "the first frame", |f| !f.is_empty());
    p.cmd("pause replay now");

    // Absolute frame.
    let m = p.marker();
    p.cmd("seek replay frame 37");
    p.assert_paused_at(m, 37, "seek frame 37");

    // Relative frames, both ways.
    let m = p.marker();
    p.cmd("seek replay frame +5");
    p.assert_paused_at(m, 42, "nudge +5");
    let m = p.marker();
    p.cmd("seek replay frame -30");
    p.assert_paused_at(m, 12, "nudge -30");

    // Relative time: one second is 30 frames.
    let m = p.marker();
    p.cmd("seek replay now +1000");
    p.assert_paused_at(m, 42, "nudge +1 s");
    let m = p.marker();
    p.cmd("seek replay now -0:00.500");
    p.assert_paused_at(m, 27, "nudge -0.5 s");

    // Absolute media time: nearest frame, tie to the later.
    let m = p.marker();
    p.cmd(&format!("seek replay now {}", p.start_ms + 1500));
    p.assert_paused_at(m, 45, "seek to 1.5 s");
    let m = p.marker();
    p.cmd(&format!("seek replay now {}", p.start_ms + 1517));
    p.assert_paused_at(m, 46, "1517 ms is nearer frame 46 (1533) than 45 (1500)");

    // UTC, through the history: media 0 is 12:00:00Z, so 12:00:02Z is frame 60.
    let m = p.marker();
    p.cmd("seek replay now 2026-08-10T12:00:02.000Z");
    p.assert_paused_at(m, 60, "UTC seek");
    assert_eq!(
        p.status()["wallclock_ms"],
        serde_json::json!(WALLCLOCK_ORIGIN_MS + 2000)
    );

    // Clamped at both ends.
    let m = p.marker();
    p.cmd("seek replay frame -999");
    p.assert_paused_at(m, 0, "clamped to the first frame");
    let m = p.marker();
    p.cmd("seek replay end");
    p.assert_paused_at(m, 119, "the last frame");

    // Rapid paused seeks: only the final target must show, and promptly.
    let m = p.marker();
    let started = Instant::now();
    for i in 0..20 {
        p.cmd(&format!(
            "seek replay frame {}",
            if i % 2 == 0 { 40 } else { 70 }
        ));
    }
    let frames = p.wait_frames(m, "the last of 20 rapid seeks", |f| f.last() == Some(&70));
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "rapid seeks settle promptly"
    );
    assert!(
        frames.iter().all(|f| *f == 40 || *f == 70),
        "only targets are ever shown: {frames:?}"
    );
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(p.last_frame(), Some(70));

    let status = p.status();
    assert_eq!(status["fps"], serde_json::json!(FPS));
    assert_eq!(status["frame_count"], serde_json::json!(120));
    assert_eq!(status["paused"], serde_json::json!(true));
    assert_eq!(status["position_ms"], serde_json::json!(70 * 1000 / 30));
}

#[test]
fn play_speed_and_reverse_release_the_right_frames_in_order() {
    if common::skip_without_ffmpeg("play_speed_and_reverse_release_the_right_frames_in_order") {
        return;
    }
    let p = Player::new("play", true);
    p.cmd("group.start p");
    p.wait_frames(0, "the first frame", |f| !f.is_empty());
    p.cmd("pause replay now");
    let m = p.marker();
    p.cmd("seek replay frame 10");
    p.assert_paused_at(m, 10, "seek frame 10");

    // Play: consecutive frames, in order, at roughly real time.
    let m = p.marker();
    let started = Instant::now();
    p.cmd("resume replay");
    let frames = p.wait_frames(m, "10 frames of playback", |f| f.len() >= 10);
    let elapsed = started.elapsed();
    assert_eq!(frames[0], 11, "playback continues from the frame on screen");
    assert!(
        Player::steps(&frames[..10]).iter().all(|s| *s == 1),
        "frames are consecutive: {frames:?}"
    );
    assert!(
        elapsed > Duration::from_millis(200) && elapsed < Duration::from_millis(1500),
        "10 frames at 30 fps take about a third of a second, took {elapsed:?}"
    );
    {
        let seen = p.seen.lock().unwrap();
        let stamps: Vec<i64> = seen.frames.iter().map(|(_, pts)| *pts).collect();
        assert!(
            stamps.windows(2).all(|w| w[1] > w[0]),
            "output PTS is strictly increasing across seeks and play: {stamps:?}"
        );
    }

    // Double speed: every other frame, once the frames already in flight at
    // the change (read at stride one) have gone out.
    p.cmd("speed.set replay 2");
    let m = p.marker();
    let frames = p.wait_frames(m, "a settled run at 2x", |f| settled_run(f, 6, 2));
    assert_eq!(p.status()["rate"], serde_json::json!(2.0));
    assert!(
        frames.windows(2).all(|w| w[1] >= w[0]),
        "never backwards while speeding up (the frame on screen may repeat once): {frames:?}"
    );

    // Half speed: consecutive again, slower.
    p.cmd("speed.set replay 0.5");
    let m = p.marker();
    let started = Instant::now();
    p.wait_frames(m, "a settled run at 0.5x", |f| settled_run(f, 4, 1));
    assert!(
        started.elapsed() > Duration::from_millis(150),
        "frames at half speed take their time"
    );

    // Reverse from a known frame: descending, consecutive.
    p.cmd("pause replay now");
    let m = p.marker();
    p.cmd("seek replay frame 60");
    p.assert_paused_at(m, 60, "seek frame 60");
    let m = p.marker();
    p.cmd("speed.set replay -1");
    p.cmd("resume replay");
    let frames = p.wait_frames(m, "reverse playback", |f| f.len() >= 6);
    assert_eq!(p.status()["direction"], serde_json::json!("reverse"));
    let steps = Player::steps(&frames[..6]);
    assert!(
        steps.iter().all(|s| *s == -1) && frames[0] <= 60,
        "reverse steps back one frame at a time from 60: {frames:?}"
    );

    // Forward again while playing: a direction change flushes and continues.
    p.cmd("speed.set replay 1");
    let m = p.marker() + 1;
    let frames = p.wait_frames(m, "forward again", |f| f.len() >= 4);
    assert!(
        Player::steps(&frames[..4]).iter().all(|s| *s == 1),
        "forward after reverse: {frames:?}"
    );
}

#[test]
fn the_tail_loops_when_asked_and_holds_the_last_frame_otherwise() {
    if common::skip_without_ffmpeg("the_tail_loops_when_asked_and_holds_the_last_frame_otherwise") {
        return;
    }
    // Loop: from near the end, frame 0 follows frame 119.
    let p = Player::new("loop", true);
    p.cmd("group.start p");
    p.wait_frames(0, "the first frame", |f| !f.is_empty());
    p.cmd("pause replay now");
    let m = p.marker();
    p.cmd("seek replay frame 115");
    p.assert_paused_at(m, 115, "seek frame 115");
    let m = p.marker();
    p.cmd("resume replay");
    let frames = p.wait_frames(m, "the loop", |f| f.contains(&0) && f.len() >= 8);
    let wrap = frames.iter().position(|f| *f == 0).unwrap();
    assert_eq!(
        frames[wrap - 1],
        119,
        "frame 0 follows the last frame: {frames:?}"
    );
    assert_eq!(&frames[wrap..wrap + 3], &[0, 1, 2]);
    drop(p);

    // No loop: the last frame stays, at_end is reported, seeking back works.
    let p = Player::new("tail", false);
    p.cmd("group.start p");
    p.wait_frames(0, "the first frame", |f| !f.is_empty());
    p.cmd("pause replay now");
    let m = p.marker();
    p.cmd("seek replay frame 112");
    p.assert_paused_at(m, 112, "seek frame 112");
    p.cmd("resume replay");
    let deadline = Instant::now() + Duration::from_secs(5);
    while p.status()["at_end"] != serde_json::json!(true) {
        assert!(
            Instant::now() < deadline,
            "at_end never reported: {}",
            p.status()
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let frames = p.wait_frames(m, "the last frame", |f| f.last() == Some(&119));
    assert!(Player::steps(&frames).iter().all(|s| *s == 1), "{frames:?}");
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(p.last_frame(), Some(119), "the last frame stays");
    // Seeking back revives playback.
    let m = p.marker();
    p.cmd("pause replay now");
    p.cmd("seek replay frame 10");
    p.assert_paused_at(m, 10, "seek back after the end");
    assert_eq!(p.status()["at_end"], serde_json::json!(false));
    let m = p.marker();
    p.cmd("resume replay");
    let frames = p.wait_frames(m, "playing again", |f| f.len() >= 3);
    assert_eq!(&frames[..3], &[11, 12, 13]);
}
