use std::io::Read;
use std::net::TcpListener;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use serde_json::{Value, json};

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::buffered_edge::BufferedEdge;
use avplumber_f7k::graph::edge::{Edge, EdgeItem, Wakeup};
use avplumber_f7k::graph::grain::{Grain, test_media};
use avplumber_f7k::graph::media::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::node::{Node, Polled};
use avplumber_f7k::graph::poll_ctx::NodePollContext;
use avplumber_f7k::graph::timestamp::{Ts, TsDelta};
use avplumber_f7k::node_api::Polling;
use avplumber_f7k::services::correction::{Action, MemberKind};
use avplumber_f7k::Instance;

use super::{SentinelSpec, SentinelVideo};

const MS: AvpRational = AvpRational { num: 1, den: 1000 };

struct Harness {
    node: Polling<SentinelVideo>,
    input: Arc<dyn Edge>,
    out: Arc<dyn Edge>,
    ctx: NodePollContext,
}

struct Out {
    content: i64,
    input_pts: i64,
    output_pts: i64,
    duration: i64,
}

impl Harness {
    fn new(name: &str, params: Value, sync: Option<&str>) -> Self {
        let spec: SentinelSpec = serde_json::from_value(params.clone()).unwrap();
        let instance = Instance::new();
        let node = spec
            .build(
                name,
                &BuildCtx {
                    instance: &instance,
                    name,
                    params: &params,
                    sync_group: sync,
                },
            )
            .unwrap();
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(32));
        let out: Arc<dyn Edge> = Arc::new(BufferedEdge::new(32));
        node.bind_source("in", input.clone());
        node.bind_sink("out", out.clone());
        node.start();
        Self {
            node,
            input,
            out,
            ctx: NodePollContext::new(Arc::new(AtomicBool::new(false)), Arc::new(Wakeup::new())),
        }
    }

    fn feed(&self, pts: i64, content: i64) {
        let mut grain = test_media(AvpMediaType::VIDEO, pts);
        tag(&mut grain, content as u16);
        assert!(self.input.offer(grain).is_ok());
    }

    fn pump(&mut self) -> Vec<Grain> {
        let mut got = Vec::new();
        for _ in 0..8 {
            match self.node.poll(&mut self.ctx).unwrap() {
                Polled::Again => {}
                Polled::Idle | Polled::Done => break,
            }
            while let Some(EdgeItem::Buffer(grain)) = self.out.try_take() {
                got.push(grain);
            }
        }
        while let Some(EdgeItem::Buffer(grain)) = self.out.try_take() {
            got.push(grain);
        }
        got
    }
}

fn tag(grain: &mut Grain, content: u16) {
    let Grain::Video(frame) = grain else {
        return;
    };
    unsafe {
        let p = frame.data_mut()[0];
        if p.is_null() {
            return;
        }
        *p = (content & 0xff) as u8;
        *p.add(1) = (content >> 8) as u8;
    }
}

fn content_of(grain: &Grain) -> i64 {
    let Grain::Video(frame) = grain else {
        return -1;
    };
    unsafe {
        let p = (*frame.as_ptr()).data[0];
        if p.is_null() {
            return -1;
        }
        (*p as i64) | ((*p.add(1) as i64) << 8)
    }
}

fn ts(val: i64) -> Ts {
    Ts::new(val, MS)
}

fn delta(val: i64) -> TsDelta {
    TsDelta::new(val, MS)
}

fn items(audio: bool) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    let mut t = 0;
    while t < 7000 {
        let pts = if audio && (1000..2500).contains(&t) {
            t + 200
        } else if !audio && (2000..2500).contains(&t) {
            t + 1500
        } else {
            t
        };
        out.push((pts, t));
        t += if audio { 20 } else { 40 };
    }
    out
}

fn play(mode: &str) -> (Harness, Vec<Out>, Vec<Out>) {
    let params = json!({
        "correction_group": "av",
        "convergence": mode,
        "max_slew": 0.1,
        "rebase_threshold": 0.5,
        "max_streams_diff": 0.001,
        "forward_start_shift": true,
        "start_ts": 0,
        "frame_duration": 0.04,
        "timeout": 0.1,
        "freeze": 5
    });
    let mut video = Harness::new("video", params, Some("live"));
    video
        .node
        .0
        .group
        .register_stream("audio", MemberKind::Audio, 0);
    let group = video.node.0.group.clone();
    let v_items = items(false);
    let a_items = items(true);
    let mut vi = 0;
    let mut ai = 0;
    let mut audio_next = 0;
    let mut audio_last = None;
    let mut v_out = Vec::new();
    let mut a_out = Vec::new();
    let mut guard = 0;
    while vi < v_items.len() || ai < a_items.len() {
        guard += 1;
        assert!(guard < 20_000, "fixture did not finish");
        let video_turn = match (v_items.get(vi).map(|i| i.0), a_items.get(ai).map(|i| i.0)) {
            (Some(v), Some(a)) => v <= a,
            (Some(_), None) => true,
            _ => false,
        };
        if video_turn {
            let (pts, content) = v_items[vi];
            video.feed(pts, content);
            for grain in video.pump() {
                let output_pts = grain.ts().ticks();
                let produced = content_of(&grain);
                let input_pts = v_items
                    .iter()
                    .find(|item| item.1 == produced)
                    .map(|item| item.0)
                    .unwrap_or(pts);
                v_out.push(Out {
                    content: produced,
                    input_pts,
                    output_pts,
                    duration: 40,
                });
            }
            vi += 1;
        } else {
            let (pts, content) = a_items[ai];
            let dt = if audio_last.is_some() { 20 } else { 0 };
            let decision = group
                .propose("audio", ts(pts), ts(audio_next), delta(20), delta(dt))
                .unwrap();
            audio_last = Some(pts);
            let (emit, duration) = match decision.action {
                Action::Stretch {
                    emit_pts,
                    sample_delta,
                } => (emit_pts.ticks(), 20 + sample_delta.ticks()),
                Action::Emit { emit_pts } => (emit_pts.ticks(), 20),
                Action::Drop => {
                    ai += 1;
                    audio_next = decision.next_ts.ticks();
                    continue;
                }
                Action::Repeat { .. } => panic!("audio does not repeat"),
            };
            a_out.push(Out {
                content,
                input_pts: pts,
                output_pts: emit,
                duration,
            });
            audio_next = decision.next_ts.ticks();
            ai += 1;
        }
    }
    (video, v_out, a_out)
}

fn assert_grid(outs: &[Out]) {
    for pair in outs.windows(2) {
        assert_eq!(
            pair[1].output_pts - pair[0].output_pts,
            pair[0].duration,
            "grid step {} then {}",
            pair[0].output_pts,
            pair[1].output_pts
        );
    }
}

fn aligned(from: i64, step: i64) -> Vec<i64> {
    let mut t = from;
    if t % step != 0 {
        t += step - (t % step);
    }
    let mut out = Vec::new();
    while t < 7000 {
        out.push(t);
        t += step;
    }
    out
}

fn assert_tail(video: &[Out], audio: &[Out], tail_from: i64) {
    let v: Vec<&Out> = video.iter().filter(|o| o.content >= tail_from).collect();
    let a: Vec<&Out> = audio.iter().filter(|o| o.content >= tail_from).collect();
    assert!(!v.is_empty() && !a.is_empty());
    for pair in v.windows(2) {
        assert_eq!(pair[1].output_pts - pair[0].output_pts, 40);
    }
    for pair in a.windows(2) {
        assert_eq!(pair[1].output_pts - pair[0].output_pts, 20);
    }
    assert_eq!(
        v.iter().map(|o| o.content).collect::<Vec<_>>(),
        aligned(tail_from, 40)
    );
    assert_eq!(
        a.iter().map(|o| o.content).collect::<Vec<_>>(),
        aligned(tail_from, 20)
    );
    let mut shifts = Vec::new();
    for o in v.iter().chain(a.iter()) {
        shifts.push(o.output_pts - o.input_pts);
    }
    let min = *shifts.iter().min().unwrap();
    let max = *shifts.iter().max().unwrap();
    assert!(max - min <= 1, "tail shift spread {min}..{max}");
    for frame in &v {
        let nearest = a
            .iter()
            .min_by_key(|o| (o.output_pts - frame.output_pts).abs())
            .unwrap();
        assert!(
            (nearest.content - frame.content).abs() <= 20,
            "video {} at {} vs audio {}",
            frame.content,
            frame.output_pts,
            nearest.content
        );
    }
}

#[test]
fn broken_then_good_slew_follow_through_the_video_node() {
    let (mut node, video, audio) = play("slew_follow");
    assert_grid(&video);
    assert_grid(&audio);
    let skewed = video.iter().any(|frame| {
        (1000..2500).contains(&frame.output_pts)
            && audio
                .iter()
                .min_by_key(|o| (o.output_pts - frame.output_pts).abs())
                .is_some_and(|audio| (audio.content - frame.content).abs() > 20)
    });
    assert!(skewed, "broken window never separated audio and video content");
    assert_tail(&video, &audio, 5000);
    assert_timeout_backups(&mut node, video.last().unwrap().content);
}

#[test]
fn broken_then_good_snap_through_the_video_node() {
    let (mut node, video, audio) = play("snap");
    assert_grid(&video);
    assert_grid(&audio);
    assert_tail(&video, &audio, 2540);
    assert_timeout_backups(&mut node, video.last().unwrap().content);
}

/// The input has stopped. A poll before the deadline emits nothing. After the
/// deadline, with the live clock ahead of the cursor, five frozen copies of
/// the last frame continue the grid and the shift stays put.
fn assert_timeout_backups(node: &mut Harness, last_content: i64) {
    let shift = node.node.0.group.group_shift().unwrap().ticks();
    let cursor = node
        .node
        .0
        .group
        .member_cursor("video")
        .unwrap()
        .next_ts
        .ticks();
    assert!(matches!(
        node.node.poll(&mut node.ctx).unwrap(),
        Polled::Idle
    ));
    assert!(node.out.try_take().is_none());
    node.node
        .0
        .clock
        .as_ref()
        .unwrap()
        .reset(cursor + 200, MS);
    thread::sleep(Duration::from_millis(150));
    let mut backups = Vec::new();
    for _ in 0..5 {
        match node.node.poll(&mut node.ctx).unwrap() {
            Polled::Again => {}
            other => panic!("expected a frozen frame, got {other:?}"),
        }
        match node.out.try_take() {
            Some(EdgeItem::Buffer(grain)) => backups.push(grain),
            other => panic!("expected a frozen frame, got {other:?}"),
        }
    }
    let pts: Vec<i64> = backups.iter().map(|grain| grain.ts().ticks()).collect();
    assert_eq!(pts, (0..5).map(|i| cursor + i * 40).collect::<Vec<_>>());
    assert!(backups.iter().all(|grain| content_of(grain) == last_content));
    assert_eq!(node.node.0.group.group_shift().unwrap().ticks(), shift);
    let status = node.node.get_object("card_status").unwrap();
    assert_eq!(status["card"], json!(true));
}

#[test]
fn hold_until_discards_frames() {
    let params = json!({
        "start_ts": 0,
        "forward_start_shift": true,
        "frame_duration": 0.04,
        "hold_until_iso": "2099-01-01T00:00:00Z"
    });
    let mut node = Harness::new("v", params, None);
    node.feed(0, 1);
    assert!(node.pump().is_empty());
    assert!(node.input.try_take().is_none());
}

#[test]
fn stall_emits_at_most_five_frozen_frames_and_raises_the_card() {
    let params = json!({
        "correction_group": "live",
        "start_ts": 0,
        "forward_start_shift": true,
        "frame_duration": 0.04,
        "timeout": 0,
        "freeze": 5
    });
    let mut node = Harness::new("v", params, Some("live"));
    node.feed(0, 7);
    let real = node.pump();
    assert_eq!(real.len(), 1);
    assert_eq!(content_of(&real[0]), 7);
    node.node
        .0
        .clock
        .as_ref()
        .unwrap()
        .reset(5_000, MS);
    // The empty poll inside `pump` armed the anti-spin deadline.
    thread::sleep(Duration::from_millis(30));
    let mut backups = Vec::new();
    for _ in 0..5 {
        match node.node.poll(&mut node.ctx).unwrap() {
            Polled::Again => {}
            other => panic!("expected a backup frame, got {other:?}"),
        }
        if let Some(EdgeItem::Buffer(grain)) = node.out.try_take() {
            backups.push(grain.ts().ticks());
        }
    }
    assert_eq!(backups, vec![40, 80, 120, 160, 200]);
    let status = node.node.get_object("card_status").unwrap();
    assert_eq!(status["card"], json!(true));
}

#[test]
fn history_file_and_reporting_url_record_a_shift_change() {
    let dir = std::env::temp_dir();
    let text = dir.join(format!("sentinel-history-{}.txt", std::process::id()));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let mut buf = String::new();
        let _ = sock.read_to_string(&mut buf);
        let _ = tx.send(buf);
    });
    let params = json!({
        "start_ts": 0,
        "forward_start_shift": true,
        "frame_duration": 0.04,
        "convergence": "slew_follow",
        "rebase_threshold": 0.5,
        "history_file_text": text.to_str().unwrap(),
        "reporting_url": format!("http://127.0.0.1:{port}/shift")
    });
    let mut node = Harness::new("v", params, None);
    node.feed(0, 1);
    node.pump();
    node.feed(2_000, 2_000);
    node.pump();
    let body = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(body.contains("\"changed_at\":0"), "{body}");
    let written = std::fs::read_to_string(&text).unwrap();
    let _ = std::fs::remove_file(&text);
    assert!(written.lines().count() >= 2, "{written}");
}

#[test]
fn a_second_member_with_a_different_mode_is_rejected() {
    let instance = Instance::new();
    let first = json!({
        "correction_group": "shared",
        "convergence": "slew",
        "start_ts": 0,
        "forward_start_shift": true,
        "frame_duration": 0.04
    });
    let second = json!({
        "correction_group": "shared",
        "convergence": "snap",
        "start_ts": 0,
        "forward_start_shift": true,
        "frame_duration": 0.04
    });
    let mut a = build_on(&instance, "a", &first);
    let mut b = build_on(&instance, "b", &second);
    a.feed(0, 1);
    a.pump();
    b.feed(0, 1);
    let err = b.node.poll(&mut b.ctx).unwrap_err();
    assert!(err.message.contains("disagrees"), "{}", err.message);
}

fn build_on(instance: &Instance, name: &str, params: &Value) -> Harness {
    let spec: SentinelSpec = serde_json::from_value(params.clone()).unwrap();
    let node = spec
        .build(
            name,
            &BuildCtx {
                instance,
                name,
                params,
                sync_group: None,
            },
        )
        .unwrap();
    let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
    let out: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
    node.bind_source("in", input.clone());
    node.bind_sink("out", out.clone());
    node.start();
    Harness {
        node,
        input,
        out,
        ctx: NodePollContext::new(Arc::new(AtomicBool::new(false)), Arc::new(Wakeup::new())),
    }
}
