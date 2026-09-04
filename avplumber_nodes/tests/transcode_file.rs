//! P2: a real transcode, `input → demux → dec_* → enc_* → mux → output`.
//!
//! Driven through [`control::exec_line`] rather than the Rust API, so the script
//! path every real deployment uses is what these tests exercise: node types,
//! parameter names, `src`/`dst`/`group` and the demux `routing` map all have to
//! be spelled the way a `.avplumber` file spells them.
//!
//! Ground truth is `ffprobe` on both files, so this is a parity test against the
//! FFmpeg CLI's own idea of what the container holds. Skipped with a log when the
//! CLI is missing.

#![cfg(all(feature = "ffmpeg", feature = "async"))]

use std::path::Path;
use std::time::Duration;

use avplumber_f7k::{Instance, control};

mod common;

/// Where the transcode ended up, plus what `ffprobe` says about it.
struct Transcoded {
    path: std::path::PathBuf,
    streams: Vec<common::ProbedStream>,
    packets: Vec<(i32, i64, i64)>,
    duration: Option<f64>,
}

/// The script, one `node.add` per node. Stream order in the output is the order
/// of `mux`'s `src`: video first, then audio.
fn script(source: &Path, target: &Path, format: Option<&str>) -> Vec<String> {
    let source = source.to_str().expect("utf-8 path");
    let target = target.to_str().expect("utf-8 path");
    let mut out = vec![
        format!(r#"node.add {{"type":"input","name":"in","group":"g","dst":"e_pkt","url":"{source}"}}"#),
        // `demux` binds `e_v`/`e_a` itself, from this very map.
        r#"node.add {"type":"demux","name":"dx","group":"g","src":"e_pkt","routing":{"v":"e_v","a":"e_a"}}"#.into(),
        r#"node.add {"type":"dec_video","name":"dv","group":"g","src":"e_v","dst":"e_vraw"}"#.into(),
        r#"node.add {"type":"dec_audio","name":"da","group":"g","src":"e_a","dst":"e_araw"}"#.into(),
        // `ultrafast` keeps the test quick; `g` matches the source's GOP so the
        // packet counts are comparable.
        r#"node.add {"type":"enc_video","name":"ev","group":"g","src":"e_vraw","dst":"e_venc","codec":"libx264","options":{"preset":"ultrafast","g":"15"}}"#.into(),
        r#"node.add {"type":"enc_audio","name":"ea","group":"g","src":"e_araw","dst":"e_aenc","codec":"aac"}"#.into(),
        r#"node.add {"type":"mux","name":"mx","group":"g","src":["e_venc","e_aenc"],"dst":"e_mux"}"#.into(),
    ];
    let format = match format {
        Some(name) => format!(r#","format":"{name}""#),
        None => String::new(),
    };
    out.push(format!(
        r#"node.add {{"type":"output","name":"out","group":"g","src":"e_mux","url":"{target}"{format}}}"#
    ));
    out.push("group.start g".into());
    out
}

/// Blocks until `output` reports a clean completion, i.e. until the whole file
/// has been written.
///
/// It reads `outcomes`, the current generation's whole list, not the single
/// `last_outcome` slot: `output` is the last node to *finish*, but not
/// necessarily the last one reported — outcomes reach the supervisor over a
/// channel from two executors, and under load the async `mux`'s lands after
/// `output`'s often enough to matter (measured: 25 of 80 runs). Polling the slot
/// then misses `out`'s completion and waits out the whole timeout.
///
/// The timeout carries the outcomes seen last, because a bare "timed out after
/// 120 s" says nothing about which node stalled.
fn wait_for_output(inst: &Instance) {
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    let mut outcomes = Vec::new();
    while std::time::Instant::now() < deadline {
        let status = control::exec_line(inst, "group.status g").expect("group.status");
        let status: serde_json::Value = serde_json::from_str(&status).expect("group.status json");
        outcomes = status["outcomes"]
            .as_array()
            .expect("group.status outcomes")
            .iter()
            .map(|outcome| outcome.as_str().expect("an outcome string").to_owned())
            .collect();
        for outcome in &outcomes {
            assert!(
                !outcome.starts_with("failed:") && !outcome.starts_with("panicked:"),
                "a node of the transcode graph did not finish cleanly: {outcome}"
            );
        }
        if outcomes
            .iter()
            .any(|outcome| outcome.starts_with("completed:out:"))
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("the output node never finished the file; the group reported {outcomes:?}");
}

/// Runs the whole script, waits for `output` to finish, then stops the group.
fn transcode(source: &Path, target: &Path, format: Option<&str>) -> Transcoded {
    let inst = Instance::new();
    // Media node types are not registered by `Instance::new`, so an embedder that
    // only wants the substrate does not pay for libav.
    avplumber_nodes::register_media_nodes(&inst);

    for line in script(source, target, format) {
        control::exec_line(&inst, &line).unwrap_or_else(|error| panic!("`{line}`: {error}"));
    }

    wait_for_output(&inst);
    // The trailer is written by `output`'s own `stop()`, which the executor has
    // already run by now; stopping the group is what proves it did not need the
    // teardown to do it.
    control::exec_line(&inst, "group.stop g").expect("group.stop");

    Transcoded {
        path: target.to_owned(),
        streams: common::streams(target),
        packets: common::packets(target),
        duration: common::duration(target),
    }
}

/// Every stream's DTS must increase strictly — the one thing
/// `av_interleaved_write_frame` refuses outright, and the reason `output` guards
/// monotonicity a second time after rescaling.
fn assert_monotonic_dts(result: &Transcoded) {
    for stream in &result.streams {
        let mut prev: Option<i64> = None;
        for (_, _, dts) in result
            .packets
            .iter()
            .filter(|(index, _, _)| *index == stream.index)
        {
            if let Some(prev) = prev {
                assert!(
                    *dts > prev,
                    "DTS of stream {} went {prev} -> {dts}",
                    stream.index
                );
            }
            prev = Some(*dts);
        }
        assert!(
            prev.is_some(),
            "stream {} of the output is empty",
            stream.index
        );
    }
}

/// Packets of one stream of the output.
fn count(result: &Transcoded, stream: i32) -> u64 {
    result
        .packets
        .iter()
        .filter(|(index, _, _)| *index == stream)
        .count() as u64
}

/// What the source held, and what the two cases below compare against.
struct Source {
    path: std::path::PathBuf,
    video_packets: u64,
    audio_packets: u64,
    duration: f64,
}

fn probe_source(path: &Path) -> Source {
    Source {
        path: path.to_owned(),
        video_packets: common::packet_count(path, 0),
        audio_packets: common::packet_count(path, 1),
        duration: common::duration(path).expect("the source has a duration"),
    }
}

/// Both cases share every assertion but the container's own quirks.
fn assert_transcode_matches(source: &Source, result: &Transcoded) {
    assert_eq!(result.streams.len(), 2, "video and the routed audio track");
    assert_eq!(result.streams[0].codec_name, "h264");
    assert_eq!(result.streams[0].codec_type, "video");
    assert_eq!(result.streams[1].codec_name, "aac");
    assert_eq!(result.streams[1].codec_type, "audio");

    // One GOP of slack, as the plan allows: the encoder may end a GOP early, and
    // x264's lookahead can hold frames the source did not.
    const GOP: u64 = 15;
    let video = count(result, 0);
    assert!(
        video.abs_diff(source.video_packets) <= GOP,
        "video packets: {video} out, {} in",
        source.video_packets
    );
    // Audio is 1:1 per frame — the decoder's 1024-sample frames go straight back
    // into an AAC encoder — so only the encoder's own priming may differ.
    const AUDIO_SLACK: u64 = 4;
    let audio = count(result, 1);
    assert!(
        audio.abs_diff(source.audio_packets) <= AUDIO_SLACK,
        "audio packets: {audio} out, {} in",
        source.audio_packets
    );

    assert_monotonic_dts(result);

    let duration = result
        .duration
        .expect("the output container must report a duration");
    assert!(
        (duration - source.duration).abs() < 0.05,
        "duration {duration} s against the source's {} s",
        source.duration
    );

    // The counts above hold just as well for a container full of grey frames in
    // the wrong order, so the pictures are compared too. `ultrafast` at the
    // default CRF lands near 40 dB on `testsrc2`; anything below 25 means the
    // frames that came out are not the frames that went in.
    const MIN_PSNR: f64 = 25.0;
    let psnr = common::video_psnr(&source.path, &result.path);
    assert!(
        psnr > MIN_PSNR,
        "the transcoded video is only {psnr} dB from the source"
    );
}

#[test]
fn mp4_to_mp4_matches_the_source() {
    if common::skip_without_ffmpeg("mp4_to_mp4_matches_the_source") {
        return;
    }
    let scratch = common::Scratch::new("transcode-mp4");
    let source = scratch.path("in.mp4");
    common::three_stream_mp4(&source);
    let probed = probe_source(&source);

    // No `format`: the muxer is inferred from the `.mp4` suffix, like C++.
    let result = transcode(&source, &scratch.path("out.mp4"), None);
    assert_transcode_matches(&probed, &result);
    // An MP4 only reports a duration once its `moov` is on disk, so the assertion
    // above is also the test that `write_trailer` ran — which C++ skips whenever
    // the group is stopped instead of reaching EOF.
    assert_eq!(
        result.streams[0].time_base,
        (1, 15360),
        "the mp4 muxer's own time base for a 15 fps stream"
    );
}

#[test]
fn mp4_to_mpegts_uses_the_muxers_own_time_base() {
    if common::skip_without_ffmpeg("mp4_to_mpegts_uses_the_muxers_own_time_base") {
        return;
    }
    let scratch = common::Scratch::new("transcode-ts");
    let source = scratch.path("in.mp4");
    common::three_stream_mp4(&source);
    let probed = probe_source(&source);

    // Named explicitly *and* with a matching suffix, so the `format` parameter is
    // the thing under test rather than the guess.
    let result = transcode(&source, &scratch.path("out.ts"), Some("mpegts"));
    assert_transcode_matches(&probed, &result);
    // The reason `output` remembers the time base libavformat settled on instead
    // of the one the spec asked for: mpegts puts everything in 1/90000, and a
    // packet rescaled into the wrong base lands in the wrong place.
    for stream in &result.streams {
        assert_eq!(
            stream.time_base,
            (1, 90_000),
            "stream {} of an mpegts container",
            stream.index
        );
    }
}
