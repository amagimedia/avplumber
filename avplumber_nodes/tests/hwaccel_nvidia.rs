//! Hardware decoding and encoding on an NVIDIA device, the replay demo's
//! backend: `hwaccel.init` opens one CUDA device, `h264_cuvid` decodes into
//! `cuda` surfaces, and `h264_nvenc` encodes them without a round trip through
//! host memory.
//!
//! The graph is the demo's own. What the tests assert is the thing that is
//! easy to lose: that every frame between the decoder and the encoder is a
//! device surface, that playback control still lands on the exact frame when
//! the decoder is NVDEC, and that the pixels are the ones a software decode of
//! the same recording produces.
//!
//! Everything here is skipped, loudly, when the machine has no usable CUDA
//! device or this FFmpeg has no NVIDIA codecs.

#![cfg(all(feature = "ffmpeg", feature = "async"))]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::error::NodeError;
use avplumber_f7k::graph::media::{Media, PacketExt};
use avplumber_f7k::graph::spec::Spec;
use avplumber_f7k::scaffold::{Blocking, BlockingIo, InputHandler, SingleInput};
use avplumber_f7k::{Instance, control};
use rusty_ffmpeg::ffi;

mod common;

use common::player::{Config, Player, SeekKind};

const FPS: u32 = 30;
const SECONDS: u32 = 4;
/// The demo's NVENC settings, from the C++/pyplumber player.
const NVENC_OPTIONS: &str = concat!(
    r#""b":"4000k","maxrate":"4000k","bufsize":"4000k","g":"30","bf":"0","#,
    r#""preset":"p6","profile":"baseline","tune":"ull","rc":"cbr","#,
    r#""rc-lookahead":"0","zerolatency":"1","delay":"0","forced-idr":"1","#,
    r#""no-scenecut":"1","strict_gop":"1","aud":"1","spatial-aq":"1","temporal-aq":"0""#
);

/// Whether this machine can run the NVIDIA path at all: a CUDA device opens,
/// and both codecs exist. Probed once.
fn nvidia_ready() -> bool {
    static READY: OnceLock<bool> = OnceLock::new();
    *READY.get_or_init(|| {
        if common::tools().is_none() {
            eprintln!("no ffmpeg/ffprobe CLI");
            return false;
        }
        let instance = Instance::new();
        if let Err(message) =
            control::exec_line(&instance, r#"hwaccel.init {"name":"probe","type":"cuda"}"#)
        {
            eprintln!("no CUDA device: {message}");
            return false;
        }
        for (kind, name) in [("decoder", "h264_cuvid"), ("encoder", "h264_nvenc")] {
            let found = match kind {
                "decoder" => {
                    avplumber_f7k::libav::codec::find_decoder(Some(name), ffi::AV_CODEC_ID_H264)
                        .is_ok()
                }
                _ => avplumber_f7k::libav::codec::find_encoder(name).is_ok(),
            };
            if !found {
                eprintln!("this FFmpeg has no {name}");
                return false;
            }
        }
        true
    })
}

fn skip_without_nvidia(test: &str) -> bool {
    if nvidia_ready() {
        return false;
    }
    eprintln!("skipping {test}: no usable NVIDIA device");
    true
}

// ------------------------------------------------------- the surface probe

/// What a frame looked like as it passed: its pixel format, and whether it
/// carried the frame pool that makes it a device surface.
#[derive(Default)]
struct Seen {
    frames: Vec<(i32, bool)>,
}

fn probes() -> &'static Mutex<HashMap<String, Arc<Mutex<Seen>>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<Mutex<Seen>>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn probe(name: &str) -> Arc<Mutex<Seen>> {
    probes()
        .lock()
        .unwrap()
        .entry(name.to_string())
        .or_default()
        .clone()
}

#[derive(Debug, serde::Deserialize)]
struct SurfaceProbeSpec {}

impl NodeSpec for SurfaceProbeSpec {
    const TYPE_NAME: &'static str = "test_surface_probe";
    type Node = Blocking<SurfaceProbe>;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        Ok(Blocking(SurfaceProbe {
            io: BlockingIo::new(name),
            seen: probe(name),
        }))
    }
}

/// Passes frames through untouched and records what they were.
struct SurfaceProbe {
    io: BlockingIo,
    seen: Arc<Mutex<Seen>>,
}

impl InputHandler for SurfaceProbe {
    fn on_spec(&self, spec: Spec) -> Result<Option<Spec>, NodeError> {
        Ok(Some(spec))
    }

    fn on_buffer(&self, buffer: Media) -> Result<Option<Media>, NodeError> {
        if let Media::Video(frame) = &buffer {
            self.seen
                .lock()
                .unwrap()
                .frames
                .push((frame.format, !frame.hw_frames_ctx.is_null()));
        }
        Ok(Some(buffer))
    }
}

impl SingleInput for SurfaceProbe {
    fn io(&self) -> &BlockingIo {
        &self.io
    }
}

// ---------------------------------------------------------- packet capture

#[derive(Default)]
struct Packets {
    seen: Vec<(i64, bool)>,
}

fn packet_sinks() -> &'static Mutex<HashMap<String, Arc<Mutex<Packets>>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<Mutex<Packets>>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn packet_sink(name: &str) -> Arc<Mutex<Packets>> {
    packet_sinks()
        .lock()
        .unwrap()
        .entry(name.to_string())
        .or_default()
        .clone()
}

#[derive(Debug, serde::Deserialize)]
struct PacketSinkSpec {}

impl NodeSpec for PacketSinkSpec {
    const TYPE_NAME: &'static str = "test_packet_sink";
    type Node = Blocking<PacketSink>;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        Ok(Blocking(PacketSink {
            io: BlockingIo::new(name),
            seen: packet_sink(name),
        }))
    }
}

struct PacketSink {
    io: BlockingIo,
    seen: Arc<Mutex<Packets>>,
}

impl InputHandler for PacketSink {
    fn on_spec(&self, _spec: Spec) -> Result<Option<Spec>, NodeError> {
        Ok(None)
    }

    fn on_buffer(&self, buffer: Media) -> Result<Option<Media>, NodeError> {
        if let Media::Packet(packet) = &buffer {
            self.seen
                .lock()
                .unwrap()
                .seen
                .push((packet.pts, packet.is_key()));
        }
        Ok(None)
    }
}

impl SingleInput for PacketSink {
    fn io(&self) -> &BlockingIo {
        &self.io
    }
}

// ------------------------------------------------------------------- tests

/// `hwaccel.init` validates its payload and keeps the device a name already
/// has. None of this needs a GPU.
#[test]
fn hwaccel_init_reports_what_is_wrong_and_keeps_an_existing_device() {
    let instance = Instance::new();
    let bad_json = control::exec_line(&instance, "hwaccel.init not json").unwrap_err();
    assert!(bad_json.contains("JSON object"), "{bad_json}");

    let no_type = control::exec_line(&instance, r#"hwaccel.init {"name":"gpu"}"#).unwrap_err();
    assert!(no_type.contains("\"type\""), "{no_type}");

    let no_name = control::exec_line(&instance, r#"hwaccel.init {"type":"cuda"}"#).unwrap_err();
    assert!(no_name.contains("\"name\""), "{no_name}");

    let unknown =
        control::exec_line(&instance, r#"hwaccel.init {"name":"x","type":"teapot"}"#).unwrap_err();
    assert!(
        unknown.contains("unknown hardware device type"),
        "{unknown}"
    );

    // A node naming a device nothing opened fails to build, rather than
    // quietly running in software.
    avplumber_nodes::register_media_nodes(&instance);
    let message = control::exec_line(
        &instance,
        r#"node.add {"type":"dec_video","name":"dv","group":"g","src":"a","dst":"b","hwaccel":"gpu"}"#,
    )
    .unwrap_err();
    assert!(message.contains("no hardware device `gpu`"), "{message}");

    if skip_without_nvidia("hwaccel_init_reports_what_is_wrong_and_keeps_an_existing_device") {
        return;
    }
    let first = control::exec_line(&instance, r#"hwaccel.init {"name":"gpu","type":"cuda"}"#)
        .expect("a cuda device opens");
    assert!(first.contains("ready"), "{first}");
    let again = control::exec_line(&instance, r#"hwaccel.init {"name":"gpu","type":"cuda"}"#)
        .expect("re-initializing is not an error");
    assert!(again.contains("already exists"), "{again}");
}

/// The demo's transcode leg on the GPU: every frame between `h264_cuvid` and
/// `h264_nvenc` is a CUDA surface, and packets come out the other side.
#[test]
fn frames_stay_on_the_device_from_nvdec_to_nvenc() {
    if skip_without_nvidia("frames_stay_on_the_device_from_nvdec_to_nvenc") {
        return;
    }
    common::init_logging();
    let scratch = common::Scratch::new("nvidia_surfaces");
    let source = scratch.path("source.mp4");
    let recording = scratch.path("replay.ts");
    common::testsrc_mp4(&source, FPS, SECONDS);
    {
        let inst = Instance::new();
        avplumber_nodes::register_media_nodes(&inst);
        for line in common::replay_recording_script(&source, &recording, FPS) {
            control::exec_line(&inst, &line).unwrap_or_else(|e| panic!("`{line}`: {e}"));
        }
        common::wait_for_completion(&inst, "g", "out");
        control::exec_line(&inst, "group.stop g").unwrap();
    }

    let instance = Instance::new();
    avplumber_nodes::register_media_nodes(&instance);
    avplumber_f7k::register_spec::<SurfaceProbeSpec>(&instance);
    avplumber_f7k::register_spec::<PacketSinkSpec>(&instance);
    let seen = probe("surfaces");
    let packets = packet_sink("packets");
    let url = recording.to_str().unwrap();
    let script = [
        r#"hwaccel.init {"name":"gpu","type":"cuda"}"#.to_string(),
        "queue.plan_capacity * 4".to_string(),
        format!(
            r#"node.add {{"type":"input","name":"in","group":"g","dst":"pkt","url":"{url}","eof_mode":"drain"}}"#
        ),
        r#"node.add {"type":"demux","name":"dx","group":"g","src":"pkt","routing":{"v:0":"vpkt"}}"#
            .into(),
        concat!(
            r#"node.add {"type":"dec_video","name":"dv","group":"g","src":"vpkt","dst":"raw","#,
            r#""pixel_format":"cuda","hwaccel":"gpu","codec_map":{"h264":"h264_cuvid"},"#,
            r#""hwaccel_only_for_codecs":["h264"],"options":{"flags":"low_delay"}}"#
        )
        .to_string(),
        r#"node.add {"type":"test_surface_probe","name":"surfaces","group":"g","src":"raw","dst":"probed"}"#
            .into(),
        format!(
            r#"node.add {{"type":"enc_video","name":"ev","group":"g","src":"probed","dst":"enc","codec":"h264_nvenc","hwaccel":"gpu","options":{{{NVENC_OPTIONS}}}}}"#
        ),
        r#"node.add {"type":"test_packet_sink","name":"packets","group":"g","src":"enc"}"#.into(),
        "group.start g".into(),
    ];
    for line in &script {
        control::exec_line(&instance, line).unwrap_or_else(|e| panic!("`{line}`: {e}"));
    }

    let expected = (FPS * SECONDS) as usize;
    let deadline = Instant::now() + Duration::from_secs(30);
    while packets.lock().unwrap().seen.len() < expected {
        assert!(
            Instant::now() < deadline,
            "timed out with {} of {expected} packets; group {}",
            packets.lock().unwrap().seen.len(),
            control::exec_line(&instance, "group.status g").unwrap()
        );
        std::thread::sleep(Duration::from_millis(20));
    }

    let frames = seen.lock().unwrap().frames.clone();
    assert_eq!(frames.len(), expected, "one frame per picture");
    let strays: Vec<_> = frames
        .iter()
        .enumerate()
        .filter(|(_, (format, pooled))| *format != ffi::AV_PIX_FMT_CUDA || !*pooled)
        .collect();
    assert!(
        strays.is_empty(),
        "every frame between the decoder and the encoder must be a pooled CUDA surface; \
         {} of {expected} were not, first {:?}",
        strays.len(),
        strays.first()
    );

    let produced = packets.lock().unwrap().seen.clone();
    assert!(
        produced.iter().any(|(_, key)| *key),
        "NVENC produced a keyframe"
    );
    let pts: Vec<i64> = produced.iter().map(|(pts, _)| *pts).collect();
    assert!(
        pts.windows(2).all(|w| w[1] > w[0]),
        "packet PTS increases: {pts:?}"
    );
    let _ = control::exec_line(&instance, "group.stop g");
}

/// The playback contract, unchanged by the decoder being NVDEC: a paused seek
/// shows exactly the frame asked for, and it is the same picture the software
/// decode of that recording produces — the capture sink downloads the surface
/// and compares pixels against the `ffmpeg` CLI's frames.
///
/// This is what the playback design deferred: whether a byte-exact seek
/// surfaces its target first from `h264_cuvid`.
#[test]
fn seeks_land_on_the_exact_frame_when_decoding_on_the_device() {
    if skip_without_nvidia("seeks_land_on_the_exact_frame_when_decoding_on_the_device") {
        return;
    }
    let p = Player::with(
        "nvidia_seek",
        Config::new(FPS, SECONDS, true).on_hardware("cuda"),
    );
    p.start_paused();

    for target in [0, 983, 1983, 0, 1517, 2000] {
        p.seek_paused(SeekKind::Media, target, Duration::from_millis(150));
    }
    // Frame targets, both ways, and the ends.
    let m = p.marker();
    p.cmd("seek replay frame 37");
    p.assert_paused_at(m, 37, "seek frame 37 on the device");
    let m = p.marker();
    p.cmd("seek replay frame +5");
    p.assert_paused_at(m, 42, "nudge +5 on the device");
    let m = p.marker();
    p.cmd("seek replay frame -30");
    p.assert_paused_at(m, 12, "nudge -30 on the device");
    let m = p.marker();
    p.cmd("seek replay end");
    p.assert_paused_at(m, p.count - 1, "the last frame on the device");

    // UTC, through the history.
    let m = p.marker();
    p.cmd("seek replay now 2026-08-10T12:00:02.000Z");
    p.assert_paused_at(m, 60, "UTC seek on the device");
}

/// Play, speed and reverse on the device, released in the right order.
#[test]
fn play_speed_and_reverse_work_when_decoding_on_the_device() {
    if skip_without_nvidia("play_speed_and_reverse_work_when_decoding_on_the_device") {
        return;
    }
    let p = Player::with(
        "nvidia_play",
        Config::new(FPS, SECONDS, true).on_hardware("cuda"),
    );
    p.start_paused();
    let m = p.marker();
    p.cmd("seek replay frame 10");
    p.assert_paused_at(m, 10, "seek frame 10 on the device");

    let m = p.marker();
    p.cmd("resume replay");
    let frames = p.wait_frames(m, "10 frames of playback on the device", |f| f.len() >= 10);
    assert_eq!(frames[0], 11, "playback continues from the frame on screen");
    assert!(
        Player::steps(&frames[..10]).iter().all(|s| *s == 1),
        "frames are consecutive: {frames:?}"
    );

    p.cmd("pause replay now");
    let m = p.marker();
    p.cmd("seek replay frame 60");
    p.assert_paused_at(m, 60, "seek frame 60 on the device");
    let m = p.marker();
    p.cmd("speed.set replay -1");
    p.cmd("resume replay");
    let frames = p.wait_frames(m, "reverse playback on the device", |f| f.len() >= 6);
    assert!(
        Player::steps(&frames[..6]).iter().all(|s| *s == -1) && frames[0] <= 60,
        "reverse steps back one frame at a time: {frames:?}"
    );
}
