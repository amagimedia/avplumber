//! The live encoder must survive the discontinuities a player sends through
//! it. The replay demo's graph — `input` in seekable mode → `demux` →
//! `dec_video` → `realtime` → `force_keyframe` → `enc_video` (libx264) — is
//! driven through the control protocol and the packets leaving the encoder
//! are counted: a seek and a reversal each send `FlushStart`/`FlushStop`
//! through the encoder, and packets must keep coming after both.
//!
//! Regression: the encoder used to answer `FlushStart` with
//! `avcodec_flush_buffers`, which libx264 implements by draining and which
//! stops its lookahead thread for good, so the browser froze at the first
//! seek while playback itself carried on.

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

mod common;

const FPS: u32 = 30;
const SECONDS: u32 = 4;

/// One packet as the sink saw it: PTS in the encoder's time base and whether
/// it is a keyframe.
#[derive(Default)]
struct Captured {
    packets: Vec<(i64, bool)>,
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
struct PacketCaptureSpec {}

impl NodeSpec for PacketCaptureSpec {
    const TYPE_NAME: &'static str = "test_packet_capture";
    type Node = Blocking<PacketCapture>;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        Ok(Blocking(PacketCapture {
            io: BlockingIo::new(name),
            seen: captured(name),
        }))
    }
}

struct PacketCapture {
    io: BlockingIo,
    seen: Arc<Mutex<Captured>>,
}

impl InputHandler for PacketCapture {
    fn on_spec(&self, _spec: Spec) -> Result<Option<Spec>, NodeError> {
        Ok(None)
    }

    fn on_buffer(&self, buffer: Media) -> Result<Option<Media>, NodeError> {
        if let Media::Packet(packet) = &buffer {
            self.seen
                .lock()
                .unwrap()
                .packets
                .push((packet.pts, packet.is_key()));
        }
        Ok(None)
    }
}

impl SingleInput for PacketCapture {
    fn io(&self) -> &BlockingIo {
        &self.io
    }
}

struct Encoding {
    inst: Instance,
    seen: Arc<Mutex<Captured>>,
    _scratch: common::Scratch,
}

impl Encoding {
    /// A fresh recording and the demo's live graph on it, with `flush` set on
    /// the encoder as given.
    fn new(label: &str, flush: &str) -> Self {
        common::init_logging();
        let scratch = common::Scratch::new(label);
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

        let name = format!("pkts_{label}");
        let inst = Instance::new();
        avplumber_nodes::register_media_nodes(&inst);
        avplumber_f7k::register_spec::<PacketCaptureSpec>(&inst);
        let url = recording.to_str().unwrap();
        // The demo's encoder settings: five threads and zero latency, the
        // configuration that died at the first seek.
        let script = [
            "queue.plan_capacity * 1".to_string(),
            format!(
                r#"node.add {{"type":"input","name":"in","group":"p","sync_group":"replay","dst":"pkt","url":"{url}","loop":true}}"#
            ),
            r#"node.add {"type":"demux","name":"dx","group":"p","src":"pkt","routing":{"v:0":"vpkt"}}"#.into(),
            r#"node.add {"type":"dec_video","name":"dv","group":"p","src":"vpkt","dst":"raw","options":{"threads":"1","flags":"low_delay"}}"#.into(),
            format!(
                r#"node.add {{"type":"realtime","name":"rt","group":"p","sync_group":"replay","src":"raw","dst":"paced","tick_period":"1/{FPS}"}}"#
            ),
            r#"node.add {"type":"force_keyframe","name":"kf","group":"p","src":"paced","dst":"keyed","interval_sec":"1/1"}"#.into(),
            format!(
                r#"node.add {{"type":"enc_video","name":"ev","group":"p","src":"keyed","dst":"enc","codec":"libx264","flush":"{flush}","options":{{"b":"1000000","g":"{FPS}","bf":"0","preset":"ultrafast","profile":"baseline","tune":"zerolatency","threads":"5","x264-params":"aud=1:scenecut=0"}}}}"#
            ),
            format!(r#"node.add {{"type":"test_packet_capture","name":"{name}","group":"p","src":"enc"}}"#),
            "group.start p".into(),
        ];
        for line in &script {
            control::exec_line(&inst, line).unwrap_or_else(|e| panic!("`{line}`: {e}"));
        }
        Self {
            inst,
            seen: captured(&name),
            _scratch: scratch,
        }
    }

    fn cmd(&self, line: &str) -> String {
        control::exec_line(&self.inst, line).unwrap_or_else(|e| panic!("`{line}`: {e}"))
    }

    fn marker(&self) -> usize {
        self.seen.lock().unwrap().packets.len()
    }

    fn packets_since(&self, marker: usize) -> Vec<(i64, bool)> {
        let seen = self.seen.lock().unwrap();
        seen.packets[marker.min(seen.packets.len())..].to_vec()
    }

    /// Waits for `count` packets after `marker`; the failure message carries
    /// the group's outcomes, which is where an encoder error would show.
    fn wait_packets(&self, marker: usize, count: usize, what: &str) -> Vec<(i64, bool)> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let packets = self.packets_since(marker);
            if packets.len() >= count {
                return packets;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {count} packets after {what}; got {packets:?}; group: {}",
                self.cmd("group.status p")
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn all_pts(&self) -> Vec<i64> {
        self.seen
            .lock()
            .unwrap()
            .packets
            .iter()
            .map(|(pts, _)| *pts)
            .collect()
    }
}

impl Drop for Encoding {
    fn drop(&mut self) {
        let _ = control::exec_line(&self.inst, "group.stop p");
    }
}

#[test]
fn the_live_encoder_keeps_encoding_across_seeks_and_reversal() {
    if common::skip_without_ffmpeg("the_live_encoder_keeps_encoding_across_seeks_and_reversal") {
        return;
    }
    let e = Encoding::new("keep", "keep");
    e.wait_packets(0, 5, "start");

    // A paused seek: one frame is released, and it must come out encoded.
    e.cmd("pause replay now");
    let m = e.marker();
    e.cmd("seek replay frame 60");
    e.wait_packets(m, 1, "a paused seek");
    // Nothing else while paused.
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(
        e.packets_since(m).len(),
        1,
        "one packet for one paused frame"
    );

    // Reverse play: a direction change re-seeks (another flush) and then
    // releases a frame per tick, each of which must be encoded.
    e.cmd("speed.set replay -1");
    let m = e.marker();
    e.cmd("resume replay");
    e.wait_packets(m, 8, "reverse play");

    // Forward again, and a burst of rapid seeks while playing.
    e.cmd("speed.set replay 1");
    let m = e.marker();
    e.wait_packets(m, 8, "forward again");
    let m = e.marker();
    for i in 0..10 {
        e.cmd(&format!(
            "seek replay frame {}",
            if i % 2 == 0 { 20 } else { 90 }
        ));
    }
    e.wait_packets(m, 8, "rapid seeks while playing");

    let pts = e.all_pts();
    assert!(
        pts.windows(2).all(|w| w[1] > w[0]),
        "packet PTS is strictly increasing across every discontinuity: {pts:?}"
    );
    assert!(
        e.cmd("group.status p").contains("running"),
        "the group is still running: {}",
        e.cmd("group.status p")
    );
}

#[test]
fn reopen_mode_starts_the_next_packet_with_a_keyframe() {
    if common::skip_without_ffmpeg("reopen_mode_starts_the_next_packet_with_a_keyframe") {
        return;
    }
    let e = Encoding::new("reopen", "reopen");
    // Two seconds of playback so the GOP is well past its keyframe.
    e.wait_packets(0, 10, "start");
    e.cmd("pause replay now");
    let m = e.marker();
    e.cmd("seek replay frame 47");
    let packets = e.wait_packets(m, 1, "a paused seek");
    assert!(
        packets[0].1,
        "a reopened encoder starts with a keyframe: {packets:?}"
    );
    e.cmd("speed.set replay -1");
    let m = e.marker();
    e.cmd("resume replay");
    let packets = e.wait_packets(m, 8, "reverse play after a reopen");
    assert!(
        packets[0].1,
        "the reversal's re-seek reopened too: {packets:?}"
    );
    let pts = e.all_pts();
    assert!(
        pts.windows(2).all(|w| w[1] > w[0]),
        "packet PTS is strictly increasing across reopens: {pts:?}"
    );
}
