//! Helpers for the tests that need real media: generating fixtures with the
//! `ffmpeg` CLI and reading ground truth back with `ffprobe`.
//!
//! Everything here degrades gracefully — [`tools`] returns `None` when the CLI
//! is missing, and the callers skip with a log rather than fail, so `cargo test`
//! stays green on a machine without FFmpeg installed.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

/// Both CLI tools, or `None` when either is missing.
pub fn tools() -> Option<()> {
    for tool in ["ffmpeg", "ffprobe"] {
        let found = Command::new(tool)
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        if !found {
            return None;
        }
    }
    Some(())
}

/// `if skip_without_ffmpeg("my_test") { return }` at the top of a media test.
pub fn skip_without_ffmpeg(test: &str) -> bool {
    if tools().is_some() {
        return false;
    }
    eprintln!("skipping {test}: the ffmpeg/ffprobe CLI is not on PATH");
    true
}

static NEXT: AtomicU32 = AtomicU32::new(0);

/// A scratch directory that removes itself. No `tempfile` dependency: the
/// crate has none, and a pid plus a counter is unique enough for a test run.
pub struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    pub fn new(label: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "avplumber_f7k-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch directory");
        Self { dir }
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Video + two audio tracks, the second tagged `language=eng` so a
/// `streams_filter` (`m:language:eng`) can tell them apart. Stream order is the
/// `-map` order: 0 = video (30 packets), 1 = audio `deu` (88), 2 = audio `eng`
/// (53). The three counts are deliberately distinct — equal ones would let a
/// mis-routed stream pass a count assertion.
pub fn three_stream_mp4(path: &Path) {
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x120:rate=15:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:duration=1.2",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map",
        "2:a",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-g",
        "15",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-metadata:s:a:0",
        "language=deu",
        "-metadata:s:a:1",
        "language=eng",
        path.to_str().expect("utf-8 path"),
    ]);
}

pub fn ffmpeg(args: &[&str]) {
    ffmpeg_log(args);
}

/// Same, returning what `ffmpeg` said — which is where filter results land.
fn ffmpeg_log(args: &[&str]) -> String {
    let output = Command::new("ffmpeg")
        .arg("-nostdin")
        .arg("-y")
        .args(args)
        .output()
        .expect("running ffmpeg");
    let log = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "ffmpeg {args:?} failed:\n{log}");
    log
}

/// Mean video PSNR of `actual` against `expected`, in dB — `f64::INFINITY` when
/// the two decode to identical pixels.
///
/// The counting assertions elsewhere cannot tell a real transcode from a stream
/// of garbage frames of the right size, and this can.
pub fn video_psnr(expected: &Path, actual: &Path) -> f64 {
    let log = ffmpeg_log(&[
        "-i",
        actual.to_str().expect("utf-8 path"),
        "-i",
        expected.to_str().expect("utf-8 path"),
        "-lavfi",
        "[0:v][1:v]psnr",
        "-f",
        "null",
        "-",
    ]);
    // `[Parsed_psnr_0 @ …] PSNR y:38.1 u:… average:37.9 min:… max:…`
    let average = log
        .split_whitespace()
        .find_map(|word| word.strip_prefix("average:"))
        .unwrap_or_else(|| panic!("no psnr average in the ffmpeg log:\n{log}"));
    if average == "inf" {
        return f64::INFINITY;
    }
    average
        .parse()
        .unwrap_or_else(|e| panic!("psnr average `{average}`: {e}"))
}

fn ffprobe(args: &[&str]) -> String {
    let output = Command::new("ffprobe")
        .args(["-v", "error"])
        .args(args)
        .output()
        .expect("running ffprobe");
    assert!(
        output.status.success(),
        "ffprobe {args:?} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// What `ffprobe` says one stream is, as the fields the tests compare against a
/// `Spec::Catalog`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ProbedStream {
    pub index: i32,
    pub codec_type: String,
    pub codec_name: String,
    pub time_base: (i32, i32),
}

pub fn streams(path: &Path) -> Vec<ProbedStream> {
    let raw = ffprobe(&[
        "-show_entries",
        "stream=index,codec_type,codec_name,time_base",
        "-of",
        "csv=p=0",
        path.to_str().expect("utf-8 path"),
    ]);
    let mut streams: Vec<ProbedStream> = Vec::new();
    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        let fields: Vec<&str> = line.trim().split(',').collect();
        assert_eq!(fields.len(), 4, "unexpected ffprobe csv line `{line}`");
        let (num, den) = fields[3].split_once('/').expect("time base num/den");
        let stream = ProbedStream {
            index: fields[0].parse().expect("stream index"),
            codec_name: fields[1].into(),
            codec_type: fields[2].into(),
            time_base: (num.parse().expect("tb num"), den.parse().expect("tb den")),
        };
        // A container with programs — mpegts — lists each stream twice: once in
        // the top-level `stream` section and once inside its program, which
        // `-show_entries stream` matches as well.
        if !streams.iter().any(|seen| seen.index == stream.index) {
            streams.push(stream);
        }
    }
    streams
}

/// `ffprobe -count_packets` for one stream — the ground truth a `null_sink`
/// counter is compared against.
pub fn packet_count(path: &Path, stream: i32) -> u64 {
    let raw = ffprobe(&[
        "-select_streams",
        &stream.to_string(),
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "csv=p=0",
        path.to_str().expect("utf-8 path"),
    ]);
    // First line only, for the same program duplication [`streams`] filters out.
    raw.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default()
        .parse()
        .unwrap_or_else(|e| {
            panic!("nb_read_packets for stream {stream} of {path:?} is `{raw}`: {e}")
        })
}

/// Every packet's `(stream_index, pts, dts)`, in file order.
pub fn packets(path: &Path) -> Vec<(i32, i64, i64)> {
    let raw = ffprobe(&[
        "-show_entries",
        "packet=stream_index,pts,dts",
        "-of",
        "csv=p=0",
        path.to_str().expect("utf-8 path"),
    ]);
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.trim().split(',').collect();
            let parse = |text: &str| text.parse::<i64>().unwrap_or(i64::MIN);
            (
                fields[0].parse().expect("stream index"),
                parse(fields[1]),
                parse(fields[2]),
            )
        })
        .collect()
}

/// Container duration in seconds, `None` when the muxer never wrote one — which
/// is what an unfinalized MP4 looks like.
pub fn duration(path: &Path) -> Option<f64> {
    let raw = ffprobe(&[
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        path.to_str().expect("utf-8 path"),
    ]);
    raw.trim().parse().ok()
}

pub fn wait_until(timeout: std::time::Duration, mut pred: impl FnMut() -> bool, what: &str) {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if pred() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    panic!("timed out waiting for {what}");
}
