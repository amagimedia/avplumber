//! The replay demo's transcode on the CPU backend: `input → demux → dec_video
//! → force_fps → force_keyframe → enc_video(libx264) → mux → output(mpegts +
//! seek table)`, checked against `ffprobe` and against libavformat's own byte
//! seeks.
//!
//! What the seek table promises: one entry per video packet, in order, each
//! naming a byte offset a demuxer can be pointed at to get exactly that frame
//! first. That promise is what makes indexed playback frame exact.

#![cfg(all(feature = "ffmpeg", feature = "async"))]

use std::ffi::CString;
use std::path::Path;

use avplumber_f7k::{Instance, control};
use rusty_ffmpeg::ffi;

mod common;

const FPS: u32 = 30;
const SECONDS: u32 = 3;

struct Recording {
    frames: u64,
    entries: Vec<(i64, u64)>,
}

fn read_seek_table(path: &Path) -> Vec<(i64, u64)> {
    let bytes = std::fs::read(path).expect("the binary seek table");
    assert_eq!(bytes.len() % 16, 0, "whole 16-byte records");
    bytes
        .chunks(16)
        .map(|record| {
            (
                i64::from_ne_bytes(record[..8].try_into().unwrap()),
                u64::from_ne_bytes(record[8..].try_into().unwrap()),
            )
        })
        .collect()
}

fn record(scratch: &common::Scratch) -> Recording {
    common::init_logging();
    let source = scratch.path("source.mp4");
    let target = scratch.path("replay.ts");
    common::testsrc_mp4(&source, FPS, SECONDS);

    let inst = Instance::new();
    avplumber_nodes::register_media_nodes(&inst);
    for line in common::replay_recording_script(&source, &target, FPS) {
        control::exec_line(&inst, &line).unwrap_or_else(|error| panic!("`{line}`: {error}"));
    }
    common::wait_for_completion(&inst, "g", "out");
    control::exec_line(&inst, "group.stop g").expect("group.stop");

    Recording {
        frames: common::packet_count(&target, 0),
        entries: read_seek_table(&scratch.path("replay.ts+seek")),
    }
}

#[test]
fn the_seek_table_indexes_every_frame_of_an_all_intra_recording() {
    if common::skip_without_ffmpeg("the_seek_table_indexes_every_frame_of_an_all_intra_recording") {
        return;
    }
    let scratch = common::Scratch::new("replay-recording");
    let target = scratch.path("replay.ts");
    let recording = record(&scratch);

    assert_eq!(
        recording.frames,
        (FPS * SECONDS) as u64,
        "force_fps conformed the source to {FPS} fps"
    );
    assert_eq!(
        recording.entries.len() as u64,
        recording.frames,
        "one seek entry per video packet"
    );

    // The text table is the same list, human readable.
    let text = std::fs::read_to_string(scratch.path("replay.ts+txt")).expect("the text table");
    let from_text: Vec<(i64, u64)> = text
        .lines()
        .map(|line| {
            let (ts, pos) = line.split_once(' ').expect("`ts pos` lines");
            (ts.parse().unwrap(), pos.parse().unwrap())
        })
        .collect();
    assert_eq!(from_text, recording.entries);

    // Timestamps on the frame grid, offsets increasing.
    let period_ms = 1000.0 / FPS as f64;
    let start = recording.entries[0].0;
    for (index, window) in recording.entries.windows(2).enumerate() {
        let expected = start as f64 + (index + 1) as f64 * period_ms;
        assert!(
            (window[1].0 as f64 - expected).abs() <= 1.0,
            "entry {} is at {} ms, expected {expected:.1}",
            index + 1,
            window[1].0
        );
        assert!(
            window[1].1 > window[0].1,
            "offsets increase: {:?} then {:?}",
            window[0],
            window[1]
        );
    }

    // Every packet is a keyframe: that is what lets a byte seek land on any frame.
    let flags = common::packet_flags(&target, 0);
    assert_eq!(flags.len() as u64, recording.frames);
    assert!(
        flags.iter().all(|f| f.starts_with('K')),
        "all-intra: every packet is a keyframe, got {flags:?}"
    );

    // libavformat itself, pointed at an entry's offset, delivers that frame
    // first — for the first, a middle and the last entry.
    let url = CString::new(target.to_str().unwrap()).unwrap();
    let mut ctx = rsmpeg::avformat::AVFormatContextInput::open(&url).expect("open the recording");
    let video = ctx
        .streams()
        .iter()
        .find(|s| s.codecpar().codec_type == ffi::AVMEDIA_TYPE_VIDEO)
        .map(|s| (s.index, s.time_base))
        .expect("a video stream");
    let last = recording.entries.len() - 1;
    for &probe in &[0usize, last / 3, last / 2, last] {
        let (entry_ms, offset) = recording.entries[probe];
        ctx.seek(-1, offset as i64, ffi::AVSEEK_FLAG_BYTE as i32)
            .unwrap_or_else(|e| panic!("byte seek to entry {probe} at {offset}: {e}"));
        let first = loop {
            match ctx.read_packet().expect("read after seek") {
                Some(packet) if packet.stream_index == video.0 => break packet,
                Some(_) => continue,
                None => panic!("no video packet after seeking to entry {probe}"),
            }
        };
        let pts_ms = first.pts * 1000 * video.1.num as i64 / video.1.den as i64;
        assert!(
            (pts_ms - entry_ms).abs() <= 1,
            "entry {probe} says {entry_ms} ms at byte {offset}, libavformat delivered {pts_ms} ms first"
        );
    }
}
