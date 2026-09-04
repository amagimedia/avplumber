//! P1: real packets from a real container, through `input` and `demux`.
//!
//! Ground truth comes from the `ffprobe` CLI, so these are end-to-end parity
//! tests rather than self-consistency checks. They skip with a log when the CLI
//! is missing.

#![cfg(feature = "ffmpeg")]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use avplumber_f7k::graph::routing::{MEDIA_TYPE_AUDIO, MEDIA_TYPE_VIDEO};
use avplumber_nodes::null_sink;
use avplumber_f7k::{
    Blocked, CatalogStream, Edge, EdgeEvent, EdgeHint, EdgeItem, Instance, Media, Node, NodeBody,
    NodeRequest, PadDirection, Spec, StreamSelection,
};

mod common;

/// Everything one `input` run put on its output edge.
#[derive(Default)]
struct Drained {
    /// `stream_index → packets`.
    counts: BTreeMap<i32, u64>,
    catalogs: Vec<(Option<String>, Vec<CatalogStream>)>,
    saw_eof: bool,
}

/// Runs an `input` node's own body on a thread and drains its output here.
///
/// Driving the body directly — `take_body` returns the same closure the
/// executor would call — is what makes the stream-discarding assertion possible:
/// every packet libavformat produced is visible, with nothing downstream to
/// filter it.
fn drain_input(node: Arc<dyn Node>, edge: Arc<dyn Edge>) -> Drained {
    let worker = std::thread::spawn(move || {
        node.start();
        let mut body = node.clone().take_body();
        let NodeBody::Blocking(step) = &mut body else {
            panic!("input must be a blocking node");
        };
        let result = loop {
            match step() {
                Ok(Blocked::Again) => {}
                Ok(Blocked::Done) => break Ok(()),
                Err(error) => break Err(error),
            }
        };
        drop(body);
        node.stop();
        result
    });

    let mut drained = Drained::default();
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline && !drained.saw_eof {
        match edge.take(200) {
            Some(EdgeItem::Buffer(Media::Packet(packet))) => {
                *drained.counts.entry(packet.stream_index).or_insert(0) += 1;
            }
            Some(EdgeItem::Buffer(other)) => panic!("input produced {:?}", other.media_type()),
            Some(EdgeItem::Event(EdgeEvent::Spec(Spec::Catalog { filter, streams }))) => {
                drained.catalogs.push((filter, streams));
            }
            Some(EdgeItem::Event(EdgeEvent::Spec(other))) => {
                panic!("input published a {:?} spec", other.media())
            }
            Some(EdgeItem::Event(EdgeEvent::Eof)) => drained.saw_eof = true,
            Some(EdgeItem::Event(_)) => {}
            None => {
                if edge.is_closed() {
                    break;
                }
            }
        }
    }
    worker
        .join()
        .expect("input thread")
        .expect("input finished cleanly");
    drained
}

/// An `input` node bound to a fresh edge, with nothing on the consumer side.
fn lone_input(inst: &Instance, url: &Path) -> (Arc<dyn Node>, Arc<dyn Edge>) {
    avplumber_nodes::register_media_nodes(inst);
    inst.create_node(NodeRequest::new(
        "input",
        "in",
        serde_json::json!({ "url": url.to_str().unwrap() }),
    ))
    .expect("input node");
    let edge = inst
        .bind_edge("in", "out", PadDirection::Output, "e")
        .expect("bind input output")
        .edge;
    (inst.node("in").expect("input node").node, edge)
}

#[test]
fn input_publishes_a_catalog_matching_ffprobe() {
    if common::skip_without_ffmpeg("input_publishes_a_catalog_matching_ffprobe") {
        return;
    }
    let scratch = common::Scratch::new("catalog");
    let source = scratch.path("in.mp4");
    common::three_stream_mp4(&source);
    let probed = common::streams(&source);

    let inst = Instance::new();
    let (node, edge) = lone_input(&inst, &source);
    let drained = drain_input(node, edge);

    assert!(drained.saw_eof, "input must announce EOF by default");
    assert_eq!(
        drained.catalogs.len(),
        1,
        "the catalog is published exactly once when nobody posts a hint"
    );
    let (filter, streams) = &drained.catalogs[0];
    assert_eq!(*filter, None, "no consumer asked for a streams_filter");
    assert_eq!(streams.len(), probed.len());

    for (stream, probe) in streams.iter().zip(&probed) {
        assert_eq!(stream.index, probe.index);
        assert!(
            stream.matches_filter,
            "every stream matches when there is no filter"
        );
        let expected_type = match probe.codec_type.as_str() {
            "video" => MEDIA_TYPE_VIDEO,
            "audio" => MEDIA_TYPE_AUDIO,
            other => panic!("unexpected codec type {other}"),
        };
        assert_eq!(stream.codec_type, expected_type, "stream {}", probe.index);
        assert_eq!(
            (stream.spec.time_base.num, stream.spec.time_base.den),
            probe.time_base,
            "time base of stream {}",
            probe.index
        );
        assert!(
            !stream.spec.extra_data.is_empty(),
            "stream {} should carry codec extradata",
            probe.index
        );
    }

    // The two audio tracks are the same codec, the video track is not: enough to
    // show `codec_id` really came from the container.
    assert_eq!(streams[1].spec.codec_id, streams[2].spec.codec_id);
    assert_ne!(streams[0].spec.codec_id, streams[1].spec.codec_id);
    assert_eq!(
        (
            streams[0].spec.frame_rate.num,
            streams[0].spec.frame_rate.den
        ),
        (15, 1),
        "the video stream's nominal rate"
    );

    for probe in &probed {
        assert_eq!(
            drained.counts.get(&probe.index).copied().unwrap_or(0),
            common::packet_count(&source, probe.index),
            "packets of stream {}",
            probe.index
        );
    }
}

#[test]
fn input_discards_the_streams_a_hint_leaves_out() {
    if common::skip_without_ffmpeg("input_discards_the_streams_a_hint_leaves_out") {
        return;
    }
    let scratch = common::Scratch::new("discard");
    let source = scratch.path("in.mp4");
    common::three_stream_mp4(&source);

    let inst = Instance::new();
    let (node, edge) = lone_input(&inst, &source);
    // Posted before the node's first step, so `serve_hints` applies the discard
    // before the first `av_read_frame`.
    edge.post_hint(EdgeHint::Streams(StreamSelection::new(vec![0])));

    let drained = drain_input(node, edge);

    assert!(drained.saw_eof);
    assert_eq!(
        drained.counts.get(&0).copied().unwrap_or(0),
        common::packet_count(&source, 0),
        "the selected stream must arrive in full"
    );
    // `AVDISCARD_ALL` cannot retract what `avformat_find_stream_info` already
    // pulled into libavformat's read-ahead buffer while opening — a packet or two
    // per stream, and C++ avplumber leaks the same ones. Everything after that is
    // never demuxed, which is what the bound checks: 1.2 s of AAC is 53 packets,
    // so a handful proves the discard took hold rather than merely thinned the
    // track.
    const READ_AHEAD_SLACK: u64 = 4;
    for discarded in [1, 2] {
        let leaked = drained.counts.get(&discarded).copied().unwrap_or(0);
        assert!(
            leaked <= READ_AHEAD_SLACK,
            "stream {discarded} was discarded but produced {leaked} packets \
             (of {}); only the open-time read-ahead may leak",
            common::packet_count(&source, discarded)
        );
    }
}

#[cfg(feature = "async")]
mod graph {
    use super::*;

    /// `input → demux → null_sink` per routed stream. Returns the sink names in
    /// the order the routing keys were given.
    fn run_graph(source: &Path, routing: serde_json::Value, filter: Option<&str>) -> Vec<String> {
        let inst = Instance::new();
        avplumber_nodes::register_media_nodes(&inst);

        let mut params = serde_json::json!({ "routing": routing });
        if let Some(filter) = filter {
            params["streams_filter"] = filter.into();
        }

        inst.create_node(NodeRequest::new(
            "input",
            "in",
            serde_json::json!({ "url": source.to_str().unwrap() }),
        ))
        .expect("input node");
        // `demux` creates and binds its own outputs from `routing`, so the script
        // never repeats those edge names.
        inst.create_node(NodeRequest::new("demux", "dx", params))
            .expect("demux node");

        let mut sinks = Vec::new();
        let routes = routing.as_object().expect("routing object");
        for (key, edge) in routes {
            let sink = format!("sink_{}", sanitise(key));
            null_sink::reset(&sink);
            inst.create_node(NodeRequest::new("null_sink", &sink, serde_json::json!({})))
                .expect("null_sink node");
            inst.bind_edge(
                &sink,
                "in",
                PadDirection::Input,
                edge.as_str().expect("edge name"),
            )
            .expect("bind sink");
            sinks.push(sink);
        }

        inst.bind_edge("in", "out", PadDirection::Output, "e_in")
            .expect("bind input output");
        inst.bind_edge("dx", "in", PadDirection::Input, "e_in")
            .expect("bind demux input");

        inst.create_group("g").expect("group");
        for node in ["in", "dx"]
            .iter()
            .map(|s| s.to_string())
            .chain(sinks.clone())
        {
            inst.add_group_member("g", &node).expect("group member");
        }
        inst.start_group("g").expect("start");

        let watched = sinks.clone();
        common::wait_until(
            Duration::from_secs(30),
            || {
                watched
                    .iter()
                    .all(|sink| null_sink::counters(sink).saw_eof())
            },
            "every sink to see EOF",
        );
        inst.stop_group("g").expect("stop");
        sinks
    }

    /// Routing keys make fine pad names but poor node names.
    fn sanitise(key: &str) -> String {
        key.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect()
    }

    #[test]
    fn demux_routes_each_stream_to_its_own_sink() {
        if common::skip_without_ffmpeg("demux_routes_each_stream_to_its_own_sink") {
            return;
        }
        let scratch = common::Scratch::new("demux");
        let source = scratch.path("in.mp4");
        common::three_stream_mp4(&source);

        // Stream 2 (the second audio track) is deliberately left unrouted.
        run_graph(&source, serde_json::json!({ "v": "e_v", "a": "e_a" }), None);

        let video = null_sink::counters("sink_v");
        let audio = null_sink::counters("sink_a");
        assert_eq!(
            video.buffers(),
            common::packet_count(&source, 0),
            "video packets"
        );
        assert_eq!(
            audio.buffers(),
            common::packet_count(&source, 1),
            "first audio track's packets"
        );
        // One `Spec::Packet` each, describing what that output carries.
        assert_eq!(video.specs(), 1, "video spec count");
        assert_eq!(audio.specs(), 1, "audio spec count");
        let spec = video.last_spec.lock().unwrap().clone();
        match spec {
            Some(Spec::Packet(spec)) => {
                assert_eq!((spec.time_base.num, spec.time_base.den), (1, 15360));
                assert!(!spec.extra_data.is_empty());
            }
            other => panic!("demux should forward a packet spec, got {other:?}"),
        }
    }

    #[test]
    fn a_streams_filter_shifts_the_audio_ordinal_upstream() {
        if common::skip_without_ffmpeg("a_streams_filter_shifts_the_audio_ordinal_upstream") {
            return;
        }
        let scratch = common::Scratch::new("filter");
        let source = scratch.path("in.mp4");
        common::three_stream_mp4(&source);

        // `m:language:eng` matches only stream 2, so `a:0` — the first *matching*
        // audio track — is stream 2, while the absolute key `1` still addresses
        // stream 1 and ignores the filter entirely.
        run_graph(
            &source,
            serde_json::json!({ "a:0": "e_eng", "1": "e_abs" }),
            Some("m:language:eng"),
        );

        assert_eq!(
            null_sink::counters("sink_a_0").buffers(),
            common::packet_count(&source, 2),
            "`a:0` under the filter must be the eng track (stream 2)"
        );
        assert_eq!(
            null_sink::counters("sink_1").buffers(),
            common::packet_count(&source, 1),
            "an absolute routing key bypasses the filter"
        );
    }
}
