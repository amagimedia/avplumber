//! Direct backpressure through `connect_edge` + the poll executor.
//!
//! Requires `--features async` (Poll nodes run on `AsyncExecutor`), and the
//! default build for `Media::Stub`, which exists only when libav is compiled out.

#![cfg(all(feature = "async", not(feature = "ffmpeg")))]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, mpsc};
use std::time::{Duration, Instant};

use avplumber_f7k::graph::AvpMediaType;
use avplumber_f7k::{
    AsyncExecutor, AvpRational, BufferedEdge, Edge, EdgeEvent, EdgeItem, EdgeKind, Executor, Media,
    Node, NodeKind, NodeOutcome, NodePads, NodePollContext, PadDecl, Push, SisoNode,
    SisoPollAdapter, Spec, Tick, register_factory,
};

const TARGET: usize = 5;

struct Ident {
    name: String,
}

impl SisoNode for Ident {
    type Inner = ();
    fn name(&self) -> &str {
        &self.name
    }
    fn on_spec(&self, spec: &Spec) -> Result<((), Spec), String> {
        Ok(((), spec.clone()))
    }
    fn process(&self, _inner: &mut (), buf: Media) -> Result<Option<Media>, String> {
        Ok(Some(buf))
    }
}

struct PollSource {
    name: String,
    sink: OnceLock<Arc<dyn Edge>>,
    produced: Arc<AtomicUsize>,
    spec_sent: AtomicBool,
}

impl Node for PollSource {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn pads(&self) -> NodePads {
        NodePads {
            sources: Vec::new(),
            sinks: vec![PadDecl {
                name: "out".into(),
                media: AvpMediaType::VIDEO,
            }],
        }
    }
    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        let _ = self.sink.set(edge);
    }
    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        let Some(out) = self.sink.get() else {
            return Tick::Done;
        };
        if !self.spec_sent.swap(true, Ordering::AcqRel) {
            out.push_event(EdgeEvent::Spec(video_spec()));
            return Tick::Again;
        }
        let n = self.produced.load(Ordering::Acquire);
        if n >= TARGET {
            out.push_event(EdgeEvent::Eof);
            return Tick::Done;
        }
        match out.offer(make_media(n as i64)) {
            Ok(()) => {
                self.produced.fetch_add(1, Ordering::Release);
                Tick::Again
            }
            Err((Push::Full, _)) => {
                ctx.wait_writable(out.clone());
                Tick::Idle
            }
            Err((Push::Closed | Push::Dropped, _)) => Tick::Done,
            Err((Push::Accepted, _)) => Tick::Again,
        }
    }
}

struct HoldSink {
    name: String,
    source: OnceLock<Arc<dyn Edge>>,
    received: Arc<AtomicUsize>,
    hold: Arc<AtomicBool>,
    release: Arc<dyn Edge>,
}

impl Node for HoldSink {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn pads(&self) -> NodePads {
        NodePads {
            sources: vec![PadDecl {
                name: "in".into(),
                media: AvpMediaType::VIDEO,
            }],
            sinks: Vec::new(),
        }
    }
    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        let _ = self.source.set(edge);
    }
    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        if self.hold.load(Ordering::Acquire) {
            ctx.wait_readable(self.release.clone());
            return Tick::Idle;
        }
        let Some(input) = self.source.get() else {
            return Tick::Done;
        };
        match input.try_take() {
            None if input.is_closed() => Tick::Done,
            None => {
                ctx.wait_readable(input.clone());
                Tick::Idle
            }
            Some(EdgeItem::Buffer(_)) => {
                self.received.fetch_add(1, Ordering::Release);
                Tick::Again
            }
            Some(EdgeItem::Event(EdgeEvent::Eof)) => Tick::Done,
            Some(_) => Tick::Again,
        }
    }
}

fn video_spec() -> Spec {
    Spec::Video {
        width: 8,
        height: 8,
        pix_fmt: 0,
        frame_rate: AvpRational { num: 1, den: 1 },
        sar: AvpRational { num: 1, den: 1 },
        time_base: AvpRational { num: 1, den: 1000 },
    }
}

fn make_media(pts: i64) -> Media {
    #[cfg(feature = "ffmpeg")]
    {
        use rusty_ffmpeg::ffi;
        unsafe {
            let frame = ffi::av_frame_alloc();
            assert!(!frame.is_null());
            (*frame).format = ffi::AV_PIX_FMT_YUV420P as i32;
            (*frame).width = 8;
            (*frame).height = 8;
            assert_eq!(ffi::av_frame_get_buffer(frame, 0), 0);
            (*frame).pts = pts;
            Media::Video(rsmpeg::avutil::AVFrame::from_raw(
                std::ptr::NonNull::new(frame).unwrap(),
            ))
        }
    }
    #[cfg(not(feature = "ffmpeg"))]
    {
        Media::Stub {
            kind: AvpMediaType::VIDEO,
            pts,
        }
    }
}

fn wait_until(timeout: Duration, mut pred: impl FnMut() -> bool, what: &str) {
    let deadline = Instant::now() + timeout;
    while !pred() {
        if Instant::now() > deadline {
            panic!("timed out waiting for {what}");
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

struct DeadlineNode {
    polls: Arc<AtomicUsize>,
}

impl Node for DeadlineNode {
    fn name(&self) -> &str {
        "deadline"
    }

    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }

    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        let poll = self.polls.fetch_add(1, Ordering::SeqCst) + 1;
        if poll == 3 {
            Tick::Done
        } else {
            ctx.wait_deadline(Instant::now() + Duration::from_millis(20));
            Tick::Idle
        }
    }
}

#[test]
fn poll_deadline_wakes_without_an_edge_or_external_tick() {
    let executor = AsyncExecutor::new();
    let polls = Arc::new(AtomicUsize::new(0));
    let (outcome_tx, outcome_rx) = mpsc::channel();
    executor.configure_run(
        9,
        Arc::new(move |outcome| outcome_tx.send(outcome).unwrap()),
    );
    executor.add_node(
        Arc::new(DeadlineNode {
            polls: polls.clone(),
        }),
        Vec::new(),
        Vec::new(),
    );

    let started = Instant::now();
    executor.start().unwrap();
    let outcome = outcome_rx
        .recv_timeout(Duration::from_millis(500))
        .expect("deadline-only Poll node remained parked");
    executor.join();

    assert!(matches!(
        outcome,
        NodeOutcome::Completed {
            name,
            generation: 9
        } if name == "deadline"
    ));
    assert_eq!(polls.load(Ordering::SeqCst), 3);
    assert!(started.elapsed() >= Duration::from_millis(35));
}

struct CancelDeadlineNode {
    run: AtomicUsize,
    first_polls: Arc<AtomicUsize>,
    second_polls: Arc<AtomicUsize>,
}

impl Node for CancelDeadlineNode {
    fn name(&self) -> &str {
        "cancel-deadline"
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn start(&self) {
        self.run.fetch_add(1, Ordering::SeqCst);
    }
    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        let run = self.run.load(Ordering::SeqCst);
        if run == 1 {
            self.first_polls.fetch_add(1, Ordering::SeqCst);
            ctx.wait_deadline(Instant::now() + Duration::from_secs(5));
        } else {
            let poll = self.second_polls.fetch_add(1, Ordering::SeqCst);
            let delay = if poll == 0 {
                Duration::from_millis(100)
            } else {
                Duration::from_secs(5)
            };
            ctx.wait_deadline(Instant::now() + delay);
        }
        Tick::Idle
    }
}

#[test]
fn cancelling_deadline_park_joins_and_cannot_wake_next_run() {
    let executor = AsyncExecutor::new();
    let first_polls = Arc::new(AtomicUsize::new(0));
    let second_polls = Arc::new(AtomicUsize::new(0));
    executor.add_node(
        Arc::new(CancelDeadlineNode {
            run: AtomicUsize::new(0),
            first_polls: first_polls.clone(),
            second_polls: second_polls.clone(),
        }),
        Vec::new(),
        Vec::new(),
    );
    let (first_tx, first_rx) = mpsc::channel();
    executor.configure_run(1, Arc::new(move |outcome| first_tx.send(outcome).unwrap()));
    executor.start().unwrap();
    wait_until(
        Duration::from_secs(1),
        || first_polls.load(Ordering::SeqCst) == 1,
        "first run to park on its deadline",
    );

    let stopping = Instant::now();
    executor.stop();
    executor.join();
    assert!(stopping.elapsed() < Duration::from_millis(200));
    assert!(matches!(
        first_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        NodeOutcome::Cancelled { generation: 1, .. }
    ));

    let (second_tx, second_rx) = mpsc::channel();
    executor.configure_run(2, Arc::new(move |outcome| second_tx.send(outcome).unwrap()));
    executor.start().unwrap();
    wait_until(
        Duration::from_secs(1),
        || second_polls.load(Ordering::SeqCst) == 1,
        "second run to park on its own deadline",
    );
    let second_started = Instant::now();
    wait_until(
        Duration::from_millis(500),
        || second_polls.load(Ordering::SeqCst) == 2,
        "fresh runtime to honor its own deadline",
    );
    assert!(second_started.elapsed() >= Duration::from_millis(80));
    assert!(
        second_started.elapsed() < Duration::from_millis(500),
        "second start inherited the cancelled five-second park"
    );

    let second_stopping = Instant::now();
    executor.stop();
    executor.join();
    assert!(second_stopping.elapsed() < Duration::from_millis(200));
    assert!(matches!(
        second_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        NodeOutcome::Cancelled { generation: 2, .. }
    ));
}

#[test]
fn direct_chain_parks_on_executor_until_tail_drains() {
    let inst = avplumber_f7k::Instance::new();
    let produced = Arc::new(AtomicUsize::new(0));
    let received = Arc::new(AtomicUsize::new(0));
    let hold = Arc::new(AtomicBool::new(true));
    let release: Arc<dyn Edge> = Arc::new(BufferedEdge::new(1));

    register_factory(&inst, "poll_src", {
        let produced = produced.clone();
        move |name, _| {
            Ok(Arc::new(PollSource {
                name: name.to_string(),
                sink: OnceLock::new(),
                produced: produced.clone(),
                spec_sent: AtomicBool::new(false),
            }))
        }
    });
    register_factory(&inst, "ident", |name, _| {
        Ok(Arc::new(SisoPollAdapter::new(Ident {
            name: name.to_string(),
        })))
    });
    register_factory(&inst, "hold_sink", {
        let received = received.clone();
        let hold = hold.clone();
        let release = release.clone();
        move |name, _| {
            Ok(Arc::new(HoldSink {
                name: name.to_string(),
                source: OnceLock::new(),
                received: received.clone(),
                hold: hold.clone(),
                release: release.clone(),
            }))
        }
    });

    inst.create_node(avplumber_f7k::NodeRequest::new(
        "poll_src",
        "src",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_node(avplumber_f7k::NodeRequest::new(
        "ident",
        "fwd",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_node(avplumber_f7k::NodeRequest::new(
        "hold_sink",
        "sink",
        serde_json::json!({}),
    ))
    .unwrap();

    let hop = inst
        .connect_edge("d", "src", "out", "fwd", "in", EdgeKind::Direct)
        .unwrap();
    let tail = inst
        .connect_edge(
            "b",
            "fwd",
            "out",
            "sink",
            "in",
            EdgeKind::Buffered { capacity: 1 },
        )
        .unwrap();

    inst.create_group("g").unwrap();
    inst.add_group_member("g", "src").unwrap();
    inst.add_group_member("g", "fwd").unwrap();
    inst.add_group_member("g", "sink").unwrap();
    inst.start_group("g").unwrap();

    wait_until(
        Duration::from_secs(2),
        || hop.edge.is_full() && tail.edge.is_full(),
        "Direct head to follow a full Buffered tail",
    );
    let parked = produced.load(Ordering::Acquire);
    assert_eq!(
        parked, 1,
        "Direct hops must not hide extra frames behind the cap-1 tail"
    );
    assert_eq!(received.load(Ordering::Acquire), 0);
    std::thread::sleep(Duration::from_millis(40));
    assert_eq!(
        produced.load(Ordering::Acquire),
        parked,
        "source must stay Idle until the tail is writable"
    );

    hold.store(false, Ordering::Release);
    release.push_event(EdgeEvent::FlushStop);

    wait_until(
        Duration::from_secs(2),
        || received.load(Ordering::Acquire) >= TARGET,
        &format!(
            "sink to drain after hold is released (received {}, produced {})",
            received.load(Ordering::Acquire),
            produced.load(Ordering::Acquire)
        ),
    );
    wait_until(
        Duration::from_secs(2),
        || produced.load(Ordering::Acquire) >= TARGET,
        "source to finish after tail drained",
    );
    inst.stop_group("g").unwrap();
}
