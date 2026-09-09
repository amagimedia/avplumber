//! The replay player graph under test, with ground truth: a `testsrc2` clip is
//! transcoded into a replay recording, every frame of that recording is decoded
//! by the `ffmpeg` CLI and hashed, and the player graph — `input` in seekable
//! mode → `demux` → `dec_video` → `realtime` → a capture sink — is driven
//! through the control protocol. A released frame is identified by its
//! pixels, so "the frame after `seek frame 37`" means the frame whose hash is
//! the CLI's frame 37, not a timestamp that happens to say so.
//!
//! Shared by `playback.rs` and `playback_scenarios.rs`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::error::NodeError;
use avplumber_f7k::graph::media::Media;
use avplumber_f7k::graph::spec::Spec;
use avplumber_f7k::scaffold::{Blocking, BlockingIo, InputHandler, SingleInput};
use avplumber_f7k::{Instance, control};

/// UTC of media time 0 in the history the player writes: 2026-08-10T12:00:00Z.
pub const WALLCLOCK_ORIGIN_MS: i64 = 1_786_363_200_000;

// ----------------------------------------------------------- capture sink

/// What the sink saw: every released frame's hash and release stamp.
#[derive(Default)]
pub struct Captured {
    pub frames: Vec<(u64, i64)>,
}

fn captures() -> &'static Mutex<HashMap<String, Arc<Mutex<Captured>>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<Mutex<Captured>>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn captured(name: &str) -> Arc<Mutex<Captured>> {
    captures()
        .lock()
        .unwrap()
        .entry(name.to_string())
        .or_default()
        .clone()
}

#[derive(Debug, serde::Deserialize)]
struct CaptureSpec {}

impl NodeSpec for CaptureSpec {
    const TYPE_NAME: &'static str = "test_capture";
    type Node = Blocking<Capture>;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        Ok(Blocking(Capture {
            io: BlockingIo::new(name),
            seen: captured(name),
        }))
    }
}

struct Capture {
    io: BlockingIo,
    seen: Arc<Mutex<Captured>>,
}

impl InputHandler for Capture {
    fn on_spec(&self, _spec: Spec) -> Result<Option<Spec>, NodeError> {
        Ok(None)
    }

    fn on_buffer(&self, buffer: Media) -> Result<Option<Media>, NodeError> {
        if let Media::Video(frame) = &buffer {
            let hash = super::fnv1a(&super::frame_bytes(frame));
            log::debug!(
                "{}: captured hash {hash:016x} at pts {}",
                self.io.name,
                frame.pts
            );
            self.seen.lock().unwrap().frames.push((hash, frame.pts));
        }
        Ok(None)
    }
}

impl SingleInput for Capture {
    fn io(&self) -> &BlockingIo {
        &self.io
    }
}

// ---------------------------------------------------------------- player

/// The recording and the graph a [`Player`] is built with.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub fps: u32,
    pub seconds: u32,
    pub loop_: bool,
    /// One decoder thread with `low_delay`, as the demo runs it, so a paused
    /// seek surfaces its one frame at once. `false` leaves the decoder at its
    /// defaults, frame threads and their delay included.
    pub low_delay: bool,
    /// Decode on a hardware device of this type (`"cuda"`) instead of in
    /// software, exactly as the demo's NVIDIA backend does: `h264_cuvid` into
    /// `cuda` surfaces. The capture sink downloads them to compare pixels; the
    /// graph itself never does.
    pub hwaccel_type: Option<&'static str>,
}

impl Config {
    pub fn new(fps: u32, seconds: u32, loop_: bool) -> Self {
        Self {
            fps,
            seconds,
            loop_,
            low_delay: true,
            hwaccel_type: None,
        }
    }

    /// The same, decoding on a hardware device.
    pub fn on_hardware(mut self, device_type: &'static str) -> Self {
        self.hwaccel_type = Some(device_type);
        self
    }
}

/// How a paused seek names its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeekKind {
    /// Absolute media time.
    Media,
    /// A signed offset from the position on screen.
    Relative,
    /// A UTC timestamp, through the history.
    Utc,
}

pub struct Player {
    inst: Instance,
    by_hash: HashMap<u64, usize>,
    pub seen: Arc<Mutex<Captured>>,
    pub start_ms: i64,
    pub fps: u32,
    /// Frames in the recording.
    pub count: usize,
    /// The seek table: absolute media time of every frame.
    entries_ms: Vec<i64>,
    stopped: bool,
    _scratch: super::Scratch,
}

impl Player {
    /// The demo's configuration at 30 fps on a four-second clip.
    pub fn new(label: &str, loop_: bool) -> Self {
        Self::with(label, Config::new(30, 4, loop_))
    }

    /// Transcodes a fresh clip into a recording, writes its history, decodes
    /// the ground truth, and builds the player graph (not started).
    pub fn with(label: &str, cfg: Config) -> Self {
        super::init_logging();
        let scratch = super::Scratch::new(label);
        let source = scratch.path("source.mp4");
        let recording = scratch.path("replay.ts");
        super::testsrc_mp4(&source, cfg.fps, cfg.seconds);
        {
            let inst = Instance::new();
            avplumber_nodes::register_media_nodes(&inst);
            for line in super::replay_recording_script(&source, &recording, cfg.fps) {
                control::exec_line(&inst, &line).unwrap_or_else(|e| panic!("`{line}`: {e}"));
            }
            super::wait_for_completion(&inst, "g", "out");
            control::exec_line(&inst, "group.stop g").unwrap();
        }
        let table = std::fs::read(scratch.path("replay.ts+seek")).unwrap();
        let entries_ms: Vec<i64> = table
            .chunks_exact(16)
            .map(|record| i64::from_ne_bytes(record[..8].try_into().unwrap()))
            .collect();
        let start_ms = entries_ms[0];
        write_history(&scratch.path("replay.ts+history"), start_ms);

        let truth = super::decoded_frame_hashes(&recording);
        assert_eq!(truth.len(), (cfg.fps * cfg.seconds) as usize);
        assert_eq!(entries_ms.len(), truth.len(), "one seek entry per frame");
        let by_hash: HashMap<u64, usize> = truth.iter().enumerate().map(|(i, h)| (*h, i)).collect();
        assert_eq!(
            by_hash.len(),
            truth.len(),
            "testsrc2 frames are all distinct"
        );

        let name = format!("cap_{label}");
        let inst = Instance::new();
        avplumber_nodes::register_media_nodes(&inst);
        avplumber_f7k::register_spec::<CaptureSpec>(&inst);
        let url = recording.to_str().unwrap();
        let fps = cfg.fps;
        let decoder_options = if cfg.low_delay {
            r#","options":{"threads":"1","flags":"low_delay"}"#
        } else {
            ""
        };
        // The demo's NVIDIA decoder, verbatim: the device, the surface format,
        // and cuvid for h264 only.
        let decoder_hw = match cfg.hwaccel_type {
            Some(_) => concat!(
                r#","pixel_format":"cuda","hwaccel":"gpu""#,
                r#","codec_map":{"h264":"h264_cuvid"},"hwaccel_only_for_codecs":["h264"]"#
            ),
            None => "",
        };
        let mut script = vec!["queue.plan_capacity * 1".to_string()];
        if let Some(device_type) = cfg.hwaccel_type {
            script.push(format!(
                r#"hwaccel.init {{"name":"gpu","type":"{device_type}"}}"#
            ));
        }
        script.extend([
            format!(
                r#"node.add {{"type":"input","name":"in","group":"p","sync_group":"replay","dst":"pkt","url":"{url}","loop":{}}}"#,
                cfg.loop_
            ),
            r#"node.add {"type":"demux","name":"dx","group":"p","src":"pkt","routing":{"v:0":"vpkt"}}"#.into(),
            format!(
                r#"node.add {{"type":"dec_video","name":"dv","group":"p","src":"vpkt","dst":"raw"{decoder_options}{decoder_hw}}}"#
            ),
            format!(
                r#"node.add {{"type":"realtime","name":"rt","group":"p","sync_group":"replay","src":"raw","dst":"paced","tick_period":"1/{fps}"}}"#
            ),
            format!(r#"node.add {{"type":"test_capture","name":"{name}","group":"p","src":"paced"}}"#),
        ]);
        for line in &script {
            control::exec_line(&inst, line).unwrap_or_else(|e| panic!("`{line}`: {e}"));
        }
        Self {
            inst,
            by_hash,
            seen: captured(&name),
            start_ms,
            fps,
            count: truth.len(),
            entries_ms,
            stopped: false,
            _scratch: scratch,
        }
    }

    /// Starts the graph, waits for the first frame, and pauses on it: the
    /// state every paused-seek scenario begins in.
    pub fn start_paused(&self) {
        self.cmd("group.start p");
        self.wait_frames(0, "the first frame", |f| !f.is_empty());
        self.cmd("pause replay now");
    }

    pub fn cmd(&self, line: &str) -> String {
        control::exec_line(&self.inst, line).unwrap_or_else(|e| panic!("`{line}`: {e}"))
    }

    pub fn status(&self) -> serde_json::Value {
        serde_json::from_str(&self.cmd("playback.status replay")).unwrap()
    }

    pub fn position_ms(&self) -> i64 {
        self.status()["position_ms"].as_i64().unwrap()
    }

    pub fn frame(&self) -> usize {
        self.status()["frame"].as_u64().unwrap() as usize
    }

    /// Media time of `frame`, relative to the recording's start.
    pub fn ms(&self, frame: usize) -> i64 {
        self.entries_ms[frame] - self.start_ms
    }

    /// Media time of the last frame, relative to the start.
    pub fn duration_ms(&self) -> i64 {
        self.ms(self.count - 1)
    }

    /// The seek contract: the nearest indexed frame, ties to the later one.
    pub fn nearest(&self, target_ms: i64) -> usize {
        (0..self.count)
            .min_by_key(|&i| ((self.ms(i) - target_ms).abs(), std::cmp::Reverse(i)))
            .unwrap()
    }

    /// The UTC timestamp the history maps `target_ms` to.
    pub fn utc(&self, target_ms: i64) -> String {
        assert!(
            (0..60_000).contains(&target_ms),
            "within the origin's minute"
        );
        format!(
            "2026-08-10T12:00:{:02}.{:03}Z",
            target_ms / 1000,
            target_ms % 1000
        )
    }

    /// The `seek` line naming `target_ms` the `kind` way.
    pub fn seek_line(&self, kind: SeekKind, target_ms: i64) -> String {
        match kind {
            SeekKind::Media => format!("seek replay now {}", self.start_ms + target_ms),
            SeekKind::Relative => format!("seek replay now {:+}", target_ms - self.position_ms()),
            SeekKind::Utc => format!("seek replay now {}", self.utc(target_ms)),
        }
    }

    /// A paused seek to `target_ms`: exactly the nearest frame is released,
    /// nothing else within `settle`, and the status agrees. Returns the frame.
    pub fn seek_paused(&self, kind: SeekKind, target_ms: i64, settle: Duration) -> usize {
        let expected = self.nearest(target_ms);
        let line = self.seek_line(kind, target_ms);
        let m = self.marker();
        self.cmd(&line);
        self.assert_paused_at_within(m, expected, &format!("`{line}`"), settle);
        assert_eq!(
            self.position_ms(),
            self.ms(expected),
            "`{line}`: position is the frame's media time"
        );
        expected
    }

    /// How many frames have been captured so far: the marker of "fresh".
    pub fn marker(&self) -> usize {
        self.seen.lock().unwrap().frames.len()
    }

    /// Frame indices (per the ground truth) captured since `marker`.
    pub fn frames_since(&self, marker: usize) -> Vec<usize> {
        let seen = self.seen.lock().unwrap();
        seen.frames[marker.min(seen.frames.len())..]
            .iter()
            .map(|(hash, _)| {
                *self
                    .by_hash
                    .get(hash)
                    .unwrap_or_else(|| panic!("a released frame matches no ground-truth frame"))
            })
            .collect()
    }

    pub fn last_frame(&self) -> Option<usize> {
        let seen = self.seen.lock().unwrap();
        seen.frames.last().map(|(hash, _)| self.by_hash[hash])
    }

    /// Waits until `pred` holds over the frames released since `marker`.
    pub fn wait_frames(
        &self,
        marker: usize,
        what: &str,
        mut pred: impl FnMut(&[usize]) -> bool,
    ) -> Vec<usize> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let frames = self.frames_since(marker);
            if pred(&frames) {
                return frames;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {what}; released since the marker: {frames:?}; status {}",
                self.status()
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Waits until the status reports `pred`.
    pub fn wait_status(&self, what: &str, mut pred: impl FnMut(&serde_json::Value) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !pred(&self.status()) {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {what}; status {}",
                self.status()
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// The frame a paused seek must land on, and nothing after it: exactly one
    /// fresh frame, the right one, then silence.
    pub fn assert_paused_at(&self, marker: usize, expected: usize, what: &str) {
        self.assert_paused_at_within(marker, expected, what, Duration::from_millis(150));
    }

    /// [`Self::assert_paused_at`] with the silence window chosen: the bulk
    /// scenarios use a shorter one than the point checks.
    pub fn assert_paused_at_within(
        &self,
        marker: usize,
        expected: usize,
        what: &str,
        settle: Duration,
    ) {
        let frames = self.wait_frames(marker, what, |f| !f.is_empty());
        assert_eq!(frames, vec![expected], "{what}: the one frame released");
        std::thread::sleep(settle);
        assert_eq!(
            self.frames_since(marker),
            vec![expected],
            "{what}: stable while paused"
        );
        assert_eq!(
            self.status()["frame"],
            serde_json::json!(expected),
            "{what}: status agrees"
        );
        assert_eq!(self.status()["pending"], serde_json::json!(false));
    }

    /// Nothing is released for a while: the paused picture holds.
    pub fn assert_stable(&self, what: &str) {
        std::thread::sleep(Duration::from_millis(100));
        let m = self.marker();
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(self.frames_since(m), Vec::<usize>::new(), "{what}: stable");
        assert_eq!(self.status()["paused"], serde_json::json!(true), "{what}");
    }

    /// Stops the graph and reports how long that took.
    pub fn stop(&mut self) -> Duration {
        let started = Instant::now();
        self.cmd("group.stop p");
        self.stopped = true;
        started.elapsed()
    }

    /// Steps of the frame index over a run of released frames.
    pub fn steps(frames: &[usize]) -> Vec<i64> {
        frames
            .windows(2)
            .map(|w| w[1] as i64 - w[0] as i64)
            .collect()
    }
}

/// Whether the last `len` frames step by exactly `step`.
pub fn settled_run(frames: &[usize], len: usize, step: i64) -> bool {
    frames.len() >= len
        && frames[frames.len() - len..]
            .windows(2)
            .all(|w| w[1] as i64 - w[0] as i64 == step)
}

pub fn write_history(path: &Path, first_media_ms: i64) {
    let mut bytes = Vec::new();
    for value in [0i64, 0, first_media_ms - WALLCLOCK_ORIGIN_MS, 0] {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

impl Drop for Player {
    fn drop(&mut self) {
        if !self.stopped {
            let _ = control::exec_line(&self.inst, "group.stop p");
        }
    }
}
