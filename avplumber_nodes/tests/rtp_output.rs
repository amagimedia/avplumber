//! The demo's Janus output path on the CPU backend: a recording is decoded,
//! every second a keyframe is forced, libx264 encodes, `bsf` repeats SPS/PPS
//! before each keyframe, and `output` sends RTP to a UDP socket the test owns.
//! The received packets must carry the configured payload type and SSRC, run
//! in sequence, and contain the parameter sets a receiver joining mid-stream
//! needs, which is what `dump_extra` is for.

#![cfg(all(feature = "ffmpeg", feature = "async"))]

use std::collections::HashSet;
use std::net::UdpSocket;
use std::time::Duration;

use avplumber_f7k::{Instance, control};

mod common;

const FPS: u32 = 30;
const SECONDS: u32 = 4;
const PAYLOAD_TYPE: u8 = 96;
const SSRC: u32 = 0x4156_5001;

/// NAL unit types found in one RTP payload (H.264 RFC 6184: single NAL,
/// STAP-A, FU-A start).
fn nal_types(payload: &[u8]) -> Vec<u8> {
    let Some(&first) = payload.first() else {
        return Vec::new();
    };
    match first & 0x1f {
        24 => {
            let mut types = Vec::new();
            let mut rest = &payload[1..];
            while rest.len() >= 3 {
                let size = u16::from_be_bytes([rest[0], rest[1]]) as usize;
                if size == 0 || rest.len() < 2 + size {
                    break;
                }
                types.push(rest[2] & 0x1f);
                rest = &rest[2 + size..];
            }
            types
        }
        28 => {
            if payload.len() >= 2 && payload[1] & 0x80 != 0 {
                vec![payload[1] & 0x1f]
            } else {
                Vec::new()
            }
        }
        kind @ 1..=23 => vec![kind],
        _ => Vec::new(),
    }
}

#[test]
fn rtp_output_carries_the_configured_headers_and_repeats_parameter_sets() {
    if common::skip_without_ffmpeg(
        "rtp_output_carries_the_configured_headers_and_repeats_parameter_sets",
    ) {
        return;
    }
    common::init_logging();
    let scratch = common::Scratch::new("rtp");
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

    let receiver = UdpSocket::bind("127.0.0.1:0").expect("a UDP socket");
    receiver
        .set_read_timeout(Some(Duration::from_millis(500)))
        .unwrap();
    let port = receiver.local_addr().unwrap().port();

    let inst = Instance::new();
    avplumber_nodes::register_media_nodes(&inst);
    let url = recording.to_str().unwrap();
    let script = [
        "queue.plan_capacity * 4".to_string(),
        format!(r#"node.add {{"type":"input","name":"in","group":"o","dst":"pkt","url":"{url}"}}"#),
        r#"node.add {"type":"demux","name":"dx","group":"o","src":"pkt","routing":{"v:0":"vpkt"}}"#.into(),
        r#"node.add {"type":"dec_video","name":"dv","group":"o","src":"vpkt","dst":"raw","options":{"threads":"1"}}"#.into(),
        r#"node.add {"type":"force_keyframe","name":"kf","group":"o","src":"raw","dst":"keyed","interval_sec":"1/1"}"#.into(),
        format!(
            r#"node.add {{"type":"enc_video","name":"ev","group":"o","src":"keyed","dst":"enc","codec":"libx264","options":{{"preset":"ultrafast","tune":"zerolatency","profile":"baseline","g":"{FPS}","bf":"0","b":"400k","maxrate":"400k","bufsize":"400k"}}}}"#
        ),
        r#"node.add {"type":"bsf","name":"headers","group":"o","src":"enc","dst":"annexb","bsf":"dump_extra=freq=keyframe"}"#.into(),
        r#"node.add {"type":"mux","name":"mx","group":"o","src":["annexb"],"dst":"muxed","ts_sort_wait":0}"#.into(),
        format!(
            r#"node.add {{"type":"output","name":"out","group":"o","src":"muxed","url":"rtp://127.0.0.1:{port}?pkt_size=1200","format":"rtp","options":{{"payload_type":{PAYLOAD_TYPE},"ssrc":{SSRC},"rtpflags":"skip_rtcp"}}}}"#
        ),
        "group.start o".into(),
    ];
    for line in &script {
        control::exec_line(&inst, line).unwrap_or_else(|e| panic!("`{line}`: {e}"));
    }
    common::wait_for_completion(&inst, "o", "out");
    control::exec_line(&inst, "group.stop o").unwrap();

    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut buf = [0u8; 2048];
    while let Ok(n) = receiver.recv(&mut buf) {
        packets.push(buf[..n].to_vec());
    }
    assert!(
        packets.len() >= 100,
        "expected the whole clip's worth of RTP packets, got {}",
        packets.len()
    );

    let mut types = HashSet::new();
    let mut sps_at = Vec::new();
    let mut previous_seq: Option<u16> = None;
    for (index, packet) in packets.iter().enumerate() {
        assert!(packet.len() >= 12, "an RTP header");
        assert_eq!(packet[0] >> 6, 2, "RTP version 2");
        assert_eq!(packet[1] & 0x7f, PAYLOAD_TYPE, "payload type");
        assert_eq!(
            u32::from_be_bytes([packet[8], packet[9], packet[10], packet[11]]),
            SSRC,
            "ssrc"
        );
        let seq = u16::from_be_bytes([packet[2], packet[3]]);
        if let Some(previous) = previous_seq {
            assert_eq!(seq, previous.wrapping_add(1), "sequence numbers run");
        }
        previous_seq = Some(seq);
        let csrc_count = (packet[0] & 0x0f) as usize;
        let payload = &packet[12 + 4 * csrc_count..];
        for kind in nal_types(payload) {
            if kind == 7 {
                sps_at.push(index);
            }
            types.insert(kind);
        }
    }
    assert!(types.contains(&5), "IDR slices are sent: {types:?}");
    assert!(
        types.contains(&7) && types.contains(&8),
        "SPS and PPS are in band: {types:?}"
    );
    // libx264 puts the parameter sets in extradata only; `dump_extra` is what
    // makes a keyframe later in the stream carry them again, which is what a
    // receiver joining mid-stream needs.
    assert!(
        sps_at.len() >= 2 && sps_at.iter().any(|&i| i > 10),
        "dump_extra repeats the SPS at a later keyframe, saw it at packets {sps_at:?}"
    );
}
