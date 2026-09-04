use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use avplumber_f7k::{
    AvpMediaType, AvpRational, Blocked, BlockingExecutor, BufferedEdge, DirectEdge, Edge,
    EdgeEvent, EdgeItem, EdgeKind, EdgeLink, EdgeRestart, EdgeWaker, ExecCtxId, Executor, Graph,
    Group, GroupState, Instance, Media, Node, NodeError, NodeKind, NodeOutcome, NodePads,
    NodePhase, NodePollContext, NodeRequest, PadDecl, Push, RestartPolicy, Spec, Tick, Vertex,
    generation_reader, generation_writer, register_factory,
};

fn media(pts: i64) -> Media {
    Media::Stub {
        kind: AvpMediaType::VIDEO,
        pts,
    }
}

fn spec_with_width(width: i32) -> Spec {
    Spec::Video {
        width,
        height: 8,
        pix_fmt: 0,
        frame_rate: AvpRational { num: 25, den: 1 },
        sar: AvpRational { num: 1, den: 1 },
        time_base: AvpRational { num: 1, den: 1000 },
    }
}

fn spec() -> Spec {
    spec_with_width(8)
}

fn expect_spec_width(edge: &dyn Edge, expected: i32) {
    assert!(matches!(
        edge.try_take(),
        Some(EdgeItem::Event(EdgeEvent::Spec(Spec::Video { width, .. })))
            if width == expected
    ));
}

fn expect_buffer_pts(edge: &dyn Edge, expected: i64) {
    assert!(matches!(
        edge.try_take(),
        Some(EdgeItem::Buffer(buf)) if buf.ts().val == expected
    ));
}

fn wait_for_group_generation(group: &Group, generation: u64) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while group.generation() != generation || group.state() != GroupState::Running {
        assert!(Instant::now() < deadline, "restart timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn wait_until(timeout: Duration, pred: impl Fn() -> bool, what: &str) {
    let deadline = Instant::now() + timeout;
    while !pred() {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn egress_keeps_logical_arc_and_accepted_media_while_fencing_old_writer() {
    let logical: Arc<dyn Edge> = Arc::new(BufferedEdge::new(4));
    let old = generation_writer(logical.clone(), 1);
    assert_eq!(old.push(media(10)), Push::Accepted);
    old.push_event(EdgeEvent::Eof);

    logical.restart(1, 2, EdgeRestart::Egress);
    let new = generation_writer(logical.clone(), 2);

    assert!(Arc::ptr_eq(&logical, &logical));
    assert_eq!(old.push(media(11)), Push::Closed);
    old.push_event(EdgeEvent::Eof);
    assert_eq!(new.push(media(12)), Push::Accepted);
    assert!(!logical.is_closed());
    assert!(matches!(
        logical.try_take(),
        Some(EdgeItem::Buffer(buf)) if buf.ts().val == 10
    ));
    assert!(matches!(
        logical.try_take(),
        Some(EdgeItem::Buffer(buf)) if buf.ts().val == 12
    ));
    assert!(logical.try_take().is_none(), "restart EOF must be removed");
}

#[test]
fn internal_restart_clears_data_and_control_but_rearms_latched_spec() {
    let logical: Arc<dyn Edge> = Arc::new(BufferedEdge::new(2));
    let old = generation_writer(logical.clone(), 1);
    old.push_event(EdgeEvent::Spec(spec()));
    assert!(matches!(
        logical.try_take(),
        Some(EdgeItem::Event(EdgeEvent::Spec(_)))
    ));
    assert_eq!(old.push(media(1)), Push::Accepted);
    old.push_event(EdgeEvent::FlushStop);
    old.push_event(EdgeEvent::Eof);

    logical.restart(1, 2, EdgeRestart::Internal);

    assert!(!logical.is_closed());
    assert_eq!(logical.occupied(), 1);
    assert!(matches!(
        logical.try_take(),
        Some(EdgeItem::Event(EdgeEvent::Spec(_)))
    ));
    assert_eq!(old.push(media(2)), Push::Closed);
    assert_eq!(
        generation_writer(logical.clone(), 2).push(media(3)),
        Push::Accepted
    );
}

#[test]
fn spec_events_keep_queue_order_and_are_delivered_once() {
    let edge: Arc<dyn Edge> = Arc::new(BufferedEdge::new(2));
    edge.push_event(EdgeEvent::Spec(spec_with_width(8)));
    expect_spec_width(&*edge, 8);
    assert_eq!(edge.push(media(1)), Push::Accepted);
    edge.push_event(EdgeEvent::Spec(spec_with_width(16)));

    expect_buffer_pts(&*edge, 1);
    expect_spec_width(&*edge, 16);
    assert!(edge.try_take().is_none());
}

#[test]
fn rearmed_spec_is_readable_and_pop_consumes_only_the_replay() {
    let edge: Arc<dyn Edge> = Arc::new(BufferedEdge::new(2));
    edge.push_event(EdgeEvent::Spec(spec()));
    expect_spec_width(&*edge, 8);
    assert_eq!(edge.push(media(2)), Push::Accepted);

    edge.rearm_spec();

    assert_eq!(edge.occupied(), 2);
    assert!(matches!(
        edge.peek_clone(0),
        Some(EdgeItem::Event(EdgeEvent::Spec(Spec::Video {
            width: 8,
            ..
        })))
    ));
    edge.pop();
    expect_buffer_pts(&*edge, 2);
    assert!(edge.try_take().is_none());
}

#[test]
fn ingress_restart_replays_the_format_active_for_the_first_queued_buffer() {
    let edge: Arc<dyn Edge> = Arc::new(BufferedEdge::new(2));
    edge.push_event(EdgeEvent::Spec(spec_with_width(8)));
    expect_spec_width(&*edge, 8);
    assert_eq!(edge.push(media(3)), Push::Accepted);
    edge.push_event(EdgeEvent::Spec(spec_with_width(16)));
    assert_eq!(edge.push(media(4)), Push::Accepted);
    edge.push_event(EdgeEvent::Eof);

    edge.reset_for_restart(EdgeRestart::Ingress);

    expect_spec_width(&*edge, 8);
    expect_buffer_pts(&*edge, 3);
    expect_spec_width(&*edge, 16);
    expect_buffer_pts(&*edge, 4);
    assert!(edge.try_take().is_none());
    assert!(!edge.is_closed());
}

#[test]
fn egress_restart_preserves_queued_format_transitions_without_replay() {
    let edge: Arc<dyn Edge> = Arc::new(BufferedEdge::new(2));
    edge.push_event(EdgeEvent::Spec(spec_with_width(8)));
    expect_spec_width(&*edge, 8);
    assert_eq!(edge.push(media(5)), Push::Accepted);
    edge.push_event(EdgeEvent::Spec(spec_with_width(16)));
    assert_eq!(edge.push(media(6)), Push::Accepted);
    edge.push_event(EdgeEvent::Eof);

    edge.reset_for_restart(EdgeRestart::Egress);

    expect_buffer_pts(&*edge, 5);
    expect_spec_width(&*edge, 16);
    expect_buffer_pts(&*edge, 6);
    assert!(edge.try_take().is_none());
    assert!(!edge.is_closed());
}

#[test]
fn ingress_preserves_bounded_media_and_removes_stale_terminal_state() {
    let logical: Arc<dyn Edge> = Arc::new(BufferedEdge::new(1));
    assert_eq!(logical.push(media(4)), Push::Accepted);
    logical.push_event(EdgeEvent::Eof);

    logical.restart(0, 0, EdgeRestart::Ingress);

    assert!(!logical.is_closed());
    assert!(logical.is_full());
    assert!(matches!(
        logical.try_take(),
        Some(EdgeItem::Buffer(buf)) if buf.ts().val == 4
    ));
    assert!(logical.try_take().is_none());
}

struct FlagWaker(Arc<AtomicBool>);

impl EdgeWaker for FlagWaker {
    fn wake(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[test]
fn interrupt_releases_blocking_peek_and_both_waiter_directions() {
    for edge in [
        Arc::new(BufferedEdge::new(1)) as Arc<dyn Edge>,
        Arc::new(DirectEdge::new()) as Arc<dyn Edge>,
    ] {
        let readable = Arc::new(AtomicBool::new(false));
        let writable = Arc::new(AtomicBool::new(false));
        edge.notify_readable(Box::new(FlagWaker(readable.clone())));
        edge.notify_writable(Box::new(FlagWaker(writable.clone())));
        let (tx, rx) = mpsc::channel();
        let waiter = {
            let edge = edge.clone();
            std::thread::spawn(move || tx.send(edge.peek_clone(-1)).unwrap())
        };
        std::thread::sleep(Duration::from_millis(10));

        edge.interrupt();
        let interrupted = rx.recv_timeout(Duration::from_millis(100));
        if interrupted.is_err() {
            edge.push_event(EdgeEvent::FlushStop);
        }
        waiter.join().unwrap();

        assert!(matches!(interrupted, Ok(None)));
        assert!(readable.load(Ordering::SeqCst));
        assert!(writable.load(Ordering::SeqCst));
    }
}

struct BlockingPeekNode {
    input: Arc<dyn Edge>,
    entered: Mutex<Option<mpsc::Sender<()>>>,
}

impl Node for BlockingPeekNode {
    fn name(&self) -> &str {
        "blocking-peek"
    }
    fn process(&self) -> Blocked {
        if let Some(entered) = self.entered.lock().unwrap().take() {
            entered.send(()).unwrap();
        }
        let _ = self.input.peek_clone(-1);
        Blocked::Done
    }
}

#[test]
fn executor_stop_interrupts_blocking_peek_and_join_completes() {
    let edge: Arc<dyn Edge> = Arc::new(BufferedEdge::new(1));
    let (entered_tx, entered_rx) = mpsc::channel();
    let node: Arc<dyn Node> = Arc::new(BlockingPeekNode {
        input: edge.clone(),
        entered: Mutex::new(Some(entered_tx)),
    });
    let exec = Arc::new(BlockingExecutor::new());
    exec.add_node(node, vec![edge], Vec::new());
    exec.configure_run(1, Arc::new(|_| {}));
    exec.start().unwrap();
    entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    exec.stop();
    let (joined_tx, joined_rx) = mpsc::channel();
    let joining = {
        let exec = exec.clone();
        std::thread::spawn(move || {
            exec.join();
            joined_tx.send(()).unwrap();
        })
    };
    joined_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("blocking peek must be interrupted by executor stop");
    joining.join().unwrap();
}

struct NaturallyDone;

impl Node for NaturallyDone {
    fn name(&self) -> &str {
        "naturally-done"
    }
    fn process(&self) -> Blocked {
        Blocked::Done
    }
}

#[test]
fn executor_reports_completion_without_emitting_eof_before_supervision() {
    let output: Arc<dyn Edge> = Arc::new(BufferedEdge::new(1));
    let exec = BlockingExecutor::new();
    let (outcome_tx, outcome_rx) = mpsc::channel();
    exec.add_node(Arc::new(NaturallyDone), Vec::new(), vec![output.clone()]);
    exec.configure_run(
        1,
        Arc::new(move |outcome| outcome_tx.send(outcome).unwrap()),
    );
    exec.start().unwrap();
    assert!(matches!(
        outcome_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        NodeOutcome::Completed { .. }
    ));
    exec.join();
    assert!(output.try_take().is_none());
    assert!(!output.is_closed());
}

struct GatedDirectConsumer {
    input: OnceLock<Arc<dyn Edge>>,
    entered: Mutex<Option<mpsc::Sender<()>>>,
    release: Arc<Barrier>,
    received: Arc<AtomicUsize>,
}

impl Node for GatedDirectConsumer {
    fn name(&self) -> &str {
        "gated"
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        let _ = self.input.set(edge);
    }
    fn poll(&self, _ctx: &mut NodePollContext) -> Tick {
        if let Some(entered) = self.entered.lock().unwrap().take() {
            entered.send(()).unwrap();
            self.release.wait();
        }
        if self.input.get().and_then(|edge| edge.try_take()).is_some() {
            self.received.fetch_add(1, Ordering::SeqCst);
        }
        Tick::Idle
    }
}

#[test]
fn concurrent_direct_fence_does_not_wait_inside_consumer_poll() {
    let direct = Arc::new(DirectEdge::new());
    let logical: Arc<dyn Edge> = direct.clone();
    let old = generation_writer(logical.clone(), 1);
    let (entered_tx, entered_rx) = mpsc::channel();
    let release = Arc::new(Barrier::new(2));
    let stale_received = Arc::new(AtomicUsize::new(0));
    let consumer = Arc::new(GatedDirectConsumer {
        input: OnceLock::new(),
        entered: Mutex::new(Some(entered_tx)),
        release: release.clone(),
        received: stale_received.clone(),
    });
    consumer.bind_source("in", generation_reader(logical.clone(), 1));
    direct.set_consumer(consumer.clone());
    let offering = {
        let old = old.clone();
        std::thread::spawn(move || old.push(media(1)))
    };
    entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let (fenced_tx, fenced_rx) = mpsc::channel();
    let fencing = {
        let logical = logical.clone();
        std::thread::spawn(move || {
            logical.restart(1, 2, EdgeRestart::Internal);
            fenced_tx.send(()).unwrap();
        })
    };

    let fenced_while_polling = fenced_rx.recv_timeout(Duration::from_millis(100)).is_ok();
    let (fresh_entered_tx, fresh_entered_rx) = mpsc::channel();
    let fresh_release = Arc::new(Barrier::new(2));
    let fresh_received = Arc::new(AtomicUsize::new(0));
    let replacement = Arc::new(GatedDirectConsumer {
        input: OnceLock::new(),
        entered: Mutex::new(Some(fresh_entered_tx)),
        release: fresh_release.clone(),
        received: fresh_received.clone(),
    });
    replacement.bind_source("in", generation_reader(logical.clone(), 2));
    direct.set_consumer(replacement.clone());
    let fresh = generation_writer(logical.clone(), 2);
    let fresh_events = fresh.clone();
    let fresh_offering = std::thread::spawn(move || fresh.push(media(2)));
    fresh_entered_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap();

    // The old generation finishes while generation 2 has an active offer.
    release.wait();
    let in_flight = offering.join().unwrap();
    fencing.join().unwrap();
    fresh_release.wait();
    let fresh_result = fresh_offering.join().unwrap();
    assert_eq!(
        stale_received.load(Ordering::SeqCst),
        0,
        "the stale consumer must not consume generation 2 media"
    );
    assert_eq!(fresh_result, Push::Accepted);
    assert_eq!(fresh_received.load(Ordering::SeqCst), 1);
    fresh_events.push_event(EdgeEvent::FlushStop);
    assert!(consumer.input.get().unwrap().try_take().is_none());
    assert!(matches!(
        replacement.input.get().unwrap().try_take(),
        Some(EdgeItem::Event(EdgeEvent::FlushStop))
    ));

    assert!(
        fenced_while_polling,
        "fencing must not hold a generation lock across consumer.poll"
    );
    assert_eq!(
        in_flight,
        Push::Closed,
        "an offer discarded by the restart must not report success"
    );
    assert_eq!(old.push(media(2)), Push::Closed);
}

#[test]
fn direct_internal_restart_clears_events_and_inflight_but_rearms_spec() {
    let direct = Arc::new(DirectEdge::new());
    let logical: Arc<dyn Edge> = direct.clone();
    let old = generation_writer(logical.clone(), 1);
    old.push_event(EdgeEvent::Spec(spec()));
    assert!(matches!(
        logical.try_take(),
        Some(EdgeItem::Event(EdgeEvent::Spec(_)))
    ));
    old.push_event(EdgeEvent::FlushStop);

    let (entered_tx, entered_rx) = mpsc::channel();
    let release = Arc::new(Barrier::new(2));
    let consumer = Arc::new(GatedDirectConsumer {
        input: OnceLock::new(),
        entered: Mutex::new(Some(entered_tx)),
        release: release.clone(),
        received: Arc::new(AtomicUsize::new(0)),
    });
    consumer.bind_source("in", logical.clone());
    direct.set_consumer(consumer.clone());
    let offering = {
        let old = old.clone();
        std::thread::spawn(move || old.push(media(1)))
    };
    entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(
        logical.occupied() >= 2,
        "FlushStop and inflight must both occupy the Direct hop"
    );

    logical.restart(1, 2, EdgeRestart::Internal);

    assert!(!logical.is_closed());
    assert_eq!(logical.occupied(), 1);
    assert!(matches!(
        logical.try_take(),
        Some(EdgeItem::Event(EdgeEvent::Spec(_)))
    ));
    assert_eq!(old.push(media(2)), Push::Closed);

    release.wait();
    assert_eq!(offering.join().unwrap(), Push::Closed);

    let received = Arc::new(AtomicUsize::new(0));
    let replacement = Arc::new(PollNode {
        name: "after-restart".into(),
        input: OnceLock::new(),
        received: Some(received.clone()),
    });
    replacement.bind_source("in", logical.clone());
    direct.set_consumer(replacement.clone());
    assert_eq!(
        generation_writer(logical.clone(), 2).push(media(3)),
        Push::Accepted
    );
    assert_eq!(received.load(Ordering::SeqCst), 1);
}

struct BlockingWaker {
    entered: Mutex<Option<mpsc::Sender<()>>>,
    release: Arc<Barrier>,
}

impl EdgeWaker for BlockingWaker {
    fn wake(&self) {
        if let Some(entered) = self.entered.lock().unwrap().take() {
            entered.send(()).unwrap();
            self.release.wait();
        }
    }
}

#[test]
fn buffered_offer_discarded_by_internal_restart_returns_closed() {
    let logical: Arc<dyn Edge> = Arc::new(BufferedEdge::new(2));
    let old = generation_writer(logical.clone(), 1);
    let (entered_tx, entered_rx) = mpsc::channel();
    let release = Arc::new(Barrier::new(2));
    logical.notify_readable(Box::new(BlockingWaker {
        entered: Mutex::new(Some(entered_tx)),
        release: release.clone(),
    }));
    let offering = std::thread::spawn(move || old.push(media(7)));
    entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();

    logical.restart(1, 2, EdgeRestart::Internal);
    assert_eq!(logical.occupied(), 0);
    release.wait();

    assert_eq!(
        offering.join().unwrap(),
        Push::Closed,
        "cleared media must not have been reported as accepted"
    );
}

struct PollNode {
    name: String,
    input: OnceLock<Arc<dyn Edge>>,
    received: Option<Arc<AtomicUsize>>,
}

struct FalliblePollNode {
    name: String,
}

impl Node for FalliblePollNode {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
}

impl Node for PollNode {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn direct_poll_is_infallible(&self) -> bool {
        true
    }
    fn pads(&self) -> NodePads {
        if self.received.is_some() {
            NodePads {
                sources: vec![PadDecl {
                    name: "in".into(),
                    media: AvpMediaType::VIDEO,
                }],
                sinks: Vec::new(),
            }
        } else {
            NodePads {
                sources: Vec::new(),
                sinks: vec![PadDecl {
                    name: "out".into(),
                    media: AvpMediaType::VIDEO,
                }],
            }
        }
    }
    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        let _ = self.input.set(edge);
    }
    fn poll(&self, _ctx: &mut NodePollContext) -> Tick {
        if self.input.get().and_then(|edge| edge.try_take()).is_some() {
            self.received
                .as_ref()
                .unwrap()
                .fetch_add(1, Ordering::SeqCst);
        }
        Tick::Idle
    }
}

fn vertex(name: &str, node: Arc<dyn Node>) -> Vertex {
    Vertex {
        name: name.into(),
        node,
        sources: Default::default(),
        sinks: Default::default(),
        source_media: Default::default(),
        sink_media: Default::default(),
    }
}

#[test]
fn direct_constraints_are_rechecked_before_start() {
    let mut graph = Graph::new();
    let producer: Arc<dyn Node> = Arc::new(PollNode {
        name: "producer".into(),
        input: OnceLock::new(),
        received: None,
    });
    let consumer: Arc<dyn Node> = Arc::new(PollNode {
        name: "consumer".into(),
        input: OnceLock::new(),
        received: Some(Arc::new(AtomicUsize::new(0))),
    });
    let edge: Arc<dyn Edge> = Arc::new(DirectEdge::new());
    graph.add_vertex(vertex("producer", producer)).unwrap();
    graph.add_vertex(vertex("consumer", consumer)).unwrap();
    graph
        .add_edge(EdgeLink {
            name: "direct".into(),
            producer: "producer".into(),
            producer_pad: "out".into(),
            consumer: "consumer".into(),
            consumer_pad: "in".into(),
            edge,
        })
        .unwrap();
    let graph = Arc::new(Mutex::new(graph));

    let boundary = Group::new("boundary".into(), graph.clone());
    boundary.add(
        "producer",
        ExecCtxId::EventLoop {
            name: "loop".into(),
        },
    );
    assert!(boundary.start().unwrap_err().contains("group boundary"));

    let contexts = Group::new("contexts".into(), graph.clone());
    contexts.add("producer", ExecCtxId::EventLoop { name: "a".into() });
    contexts.add("consumer", ExecCtxId::EventLoop { name: "b".into() });
    assert!(contexts.start().unwrap_err().contains("same ExecCtxId"));

    graph.lock().unwrap().vertex_mut("consumer").unwrap().node = Arc::new(FalliblePollNode {
        name: "consumer".into(),
    });
    let fallible = Group::new("fallible".into(), graph);
    fallible.add(
        "producer",
        ExecCtxId::EventLoop {
            name: "loop".into(),
        },
    );
    fallible.add(
        "consumer",
        ExecCtxId::EventLoop {
            name: "loop".into(),
        },
    );
    assert!(
        fallible
            .start()
            .unwrap_err()
            .contains("no longer guarantees infallible polling")
    );
}

#[test]
fn direct_consumer_is_rewired_to_reconstructed_node() {
    let inst = Instance::new();
    let received = Arc::new(AtomicUsize::new(0));
    register_factory(&inst, "producer", |name, _| {
        Ok(Arc::new(PollNode {
            name: name.into(),
            input: OnceLock::new(),
            received: None,
        }))
    });
    register_factory(&inst, "consumer", {
        let received = received.clone();
        move |name, _| {
            Ok(Arc::new(PollNode {
                name: name.into(),
                input: OnceLock::new(),
                received: Some(received.clone()),
            }))
        }
    });
    inst.create_node(NodeRequest::new(
        "producer",
        "producer",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_node(NodeRequest::new(
        "consumer",
        "consumer",
        serde_json::json!({}),
    ))
    .unwrap();
    let edge = inst
        .connect_edge(
            "direct",
            "producer",
            "out",
            "consumer",
            "in",
            EdgeKind::Direct,
        )
        .unwrap()
        .edge;
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "producer").unwrap();
    inst.add_group_member("g", "consumer").unwrap();

    inst.reconstruct_group("g").unwrap();
    assert_eq!(edge.push(media(9)), Push::Accepted);
    assert_eq!(received.load(Ordering::SeqCst), 1);
}

#[test]
fn generation_writer_forwards_edge_introspection_and_lifecycle() {
    let logical: Arc<dyn Edge> = Arc::new(DirectEdge::new());
    let writer = generation_writer(logical.clone(), 1);
    assert!(writer.is_direct());
    assert_eq!(writer.writer_generation(), 1);
    assert!(writer.fence_generation(1, 2));
    assert_eq!(logical.writer_generation(), 2);
    writer.reset_for_restart(EdgeRestart::Internal);
    assert_eq!(writer.push(media(1)), Push::Closed);
}

#[cfg(feature = "async")]
struct RestartDirectConsumer {
    name: String,
    consume: bool,
    input: OnceLock<Arc<dyn Edge>>,
    received: Arc<AtomicUsize>,
}

#[cfg(feature = "async")]
struct RestartDirectSource {
    name: String,
    writers: Arc<Mutex<Vec<Arc<dyn Edge>>>>,
}

#[cfg(feature = "async")]
impl Node for RestartDirectSource {
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
        self.writers.lock().unwrap().push(edge);
    }
    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        ctx.wait_tick();
        Tick::Idle
    }
}

#[cfg(feature = "async")]
impl Node for RestartDirectConsumer {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn direct_poll_is_infallible(&self) -> bool {
        true
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
        let _ = self.input.set(edge);
    }
    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        let input = self.input.get().unwrap();
        if !self.consume {
            ctx.wait_readable(input.clone());
            return Tick::Idle;
        }
        match input.try_take() {
            Some(EdgeItem::Buffer(_)) => {
                self.received.fetch_add(1, Ordering::SeqCst);
                Tick::Again
            }
            Some(_) => Tick::Again,
            None => {
                ctx.wait_readable(input.clone());
                Tick::Idle
            }
        }
    }
}

#[cfg(feature = "async")]
struct RestartDirectForward {
    name: String,
    input: OnceLock<Arc<dyn Edge>>,
    output: OnceLock<Arc<dyn Edge>>,
}

#[cfg(feature = "async")]
impl Node for RestartDirectForward {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn direct_poll_is_infallible(&self) -> bool {
        true
    }
    fn pads(&self) -> NodePads {
        NodePads {
            sources: vec![PadDecl {
                name: "in".into(),
                media: AvpMediaType::VIDEO,
            }],
            sinks: vec![PadDecl {
                name: "out".into(),
                media: AvpMediaType::VIDEO,
            }],
        }
    }
    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        let _ = self.input.set(edge);
    }
    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        let _ = self.output.set(edge);
    }
    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        let Some(src) = self.input.get() else {
            return Tick::Done;
        };
        let Some(sink) = self.output.get() else {
            return Tick::Done;
        };
        if sink.is_full() {
            ctx.wait_writable(sink.clone());
            return Tick::Idle;
        }
        match src.peek_clone(0) {
            Some(EdgeItem::Event(_)) => {
                ctx.wait_tick();
                Tick::Idle
            }
            None => {
                ctx.wait_readable(src.clone());
                Tick::Idle
            }
            Some(EdgeItem::Buffer(_)) => match src.try_take() {
                Some(EdgeItem::Buffer(buf)) => match sink.offer(buf) {
                    Ok(()) => Tick::Again,
                    Err((Push::Full, _)) => Tick::Idle,
                    Err((Push::Closed | Push::Dropped, _)) => Tick::Done,
                    Err((Push::Accepted, _)) => Tick::Again,
                },
                _ => Tick::Again,
            },
        }
    }
}

#[cfg(feature = "async")]
struct RestartHoldSink {
    name: String,
    input: OnceLock<Arc<dyn Edge>>,
    received: Arc<AtomicUsize>,
    hold: Arc<AtomicBool>,
    release: Arc<dyn Edge>,
}

#[cfg(feature = "async")]
impl Node for RestartHoldSink {
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
        let _ = self.input.set(edge);
    }
    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        if self.hold.load(Ordering::Acquire) {
            ctx.wait_readable(self.release.clone());
            return Tick::Idle;
        }
        let Some(input) = self.input.get() else {
            return Tick::Done;
        };
        match input.try_take() {
            Some(EdgeItem::Buffer(_)) => {
                self.received.fetch_add(1, Ordering::SeqCst);
                Tick::Again
            }
            Some(_) => Tick::Again,
            None => {
                ctx.wait_readable(input.clone());
                Tick::Idle
            }
        }
    }
}

#[cfg(feature = "async")]
#[test]
fn live_direct_restart_clears_old_state_and_fuses_into_replacement() {
    let inst = Instance::new();
    let writers = Arc::new(Mutex::new(Vec::new()));
    let received = Arc::new(AtomicUsize::new(0));
    let consumer_builds = Arc::new(AtomicUsize::new(0));
    register_factory(&inst, "direct_source", {
        let writers = writers.clone();
        move |name, _| {
            Ok(Arc::new(RestartDirectSource {
                name: name.into(),
                writers: writers.clone(),
            }))
        }
    });
    register_factory(&inst, "direct_consumer", {
        let builds = consumer_builds.clone();
        let received = received.clone();
        move |name, _| {
            let generation = builds.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(Arc::new(RestartDirectConsumer {
                name: name.into(),
                consume: generation > 1,
                input: OnceLock::new(),
                received: received.clone(),
            }))
        }
    });
    let mut source = NodeRequest::new("direct_source", "source", serde_json::json!({}));
    source.restart = Some(RestartPolicy::RestartGroup);
    inst.create_node(source).unwrap();
    inst.create_node(NodeRequest::new(
        "direct_consumer",
        "consumer",
        serde_json::json!({}),
    ))
    .unwrap();
    let logical = inst
        .connect_edge(
            "direct",
            "source",
            "out",
            "consumer",
            "in",
            EdgeKind::Direct,
        )
        .unwrap()
        .edge;
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "source").unwrap();
    inst.add_group_member("g", "consumer").unwrap();
    inst.start_group("g").unwrap();
    let group = inst.group("g").unwrap();
    let stale = writers.lock().unwrap()[0].clone();
    stale.push_event(EdgeEvent::FlushStop);
    assert_eq!(logical.occupied(), 1);

    group.report_outcome(NodeOutcome::Failed {
        name: "source".into(),
        generation: 1,
        err: NodeError::new("source", NodePhase::Process, "restart direct"),
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    while group.generation() != 2 || group.state() != GroupState::Running {
        assert!(Instant::now() < deadline, "Direct restart timed out");
        std::thread::sleep(Duration::from_millis(1));
    }

    assert_eq!(logical.occupied(), 0);
    assert_eq!(stale.push(media(1)), Push::Closed);
    let fresh = writers.lock().unwrap()[1].clone();
    assert_eq!(fresh.push(media(2)), Push::Accepted);
    assert_eq!(received.load(Ordering::SeqCst), 1);
    inst.stop_group("g").unwrap();
}

#[cfg(feature = "async")]
#[test]
fn direct_stop_start_and_auto_restart_keep_nodes_edges_and_media_on_one_generation() {
    let inst = Instance::new();
    let writers = Arc::new(Mutex::new(Vec::new()));
    let received = Arc::new(AtomicUsize::new(0));
    register_factory(&inst, "continuous_direct_source", {
        let writers = writers.clone();
        move |name, _| {
            Ok(Arc::new(RestartDirectSource {
                name: name.into(),
                writers: writers.clone(),
            }))
        }
    });
    register_factory(&inst, "continuous_direct_sink", {
        let received = received.clone();
        move |name, _| {
            Ok(Arc::new(RestartDirectConsumer {
                name: name.into(),
                consume: true,
                input: OnceLock::new(),
                received: received.clone(),
            }))
        }
    });
    let mut source = NodeRequest::new("continuous_direct_source", "source", serde_json::json!({}));
    source.restart = Some(RestartPolicy::RestartGroup);
    inst.create_node(source).unwrap();
    inst.create_node(NodeRequest::new(
        "continuous_direct_sink",
        "sink",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.connect_edge(
        "continuous-direct",
        "source",
        "out",
        "sink",
        "in",
        EdgeKind::Direct,
    )
    .unwrap();
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "source").unwrap();
    inst.add_group_member("g", "sink").unwrap();

    inst.start_group("g").unwrap();
    let generation_1 = writers.lock().unwrap()[0].clone();
    assert_eq!(generation_1.push(media(1)), Push::Accepted);
    assert_eq!(received.load(Ordering::SeqCst), 1);
    inst.stop_group("g").unwrap();

    inst.start_group("g").unwrap();
    let generation_2 = writers.lock().unwrap()[1].clone();
    assert_eq!(generation_1.push(media(11)), Push::Closed);
    assert_eq!(generation_2.push(media(2)), Push::Accepted);
    assert_eq!(received.load(Ordering::SeqCst), 2);

    let group = inst.group("g").unwrap();
    group.report_outcome(NodeOutcome::Failed {
        name: "source".into(),
        generation: 2,
        err: NodeError::new("source", NodePhase::Process, "automatic generation 3"),
    });
    wait_for_group_generation(&group, 3);
    let generation_3 = writers.lock().unwrap()[2].clone();
    assert_eq!(generation_1.push(media(12)), Push::Closed);
    assert_eq!(generation_2.push(media(22)), Push::Closed);
    assert_eq!(generation_3.push(media(3)), Push::Accepted);
    assert_eq!(received.load(Ordering::SeqCst), 3);
    inst.stop_group("g").unwrap();
}

#[cfg(feature = "async")]
fn register_direct_forward(inst: &Instance, ty: &str) {
    register_factory(inst, ty, |name, _| {
        Ok(Arc::new(RestartDirectForward {
            name: name.into(),
            input: OnceLock::new(),
            output: OnceLock::new(),
        }))
    });
}

#[cfg(feature = "async")]
#[test]
fn live_direct_only_chain_restart_stores_nothing_and_rewires_both_hops() {
    let inst = Instance::new();
    let writers = Arc::new(Mutex::new(Vec::new()));
    let received = Arc::new(AtomicUsize::new(0));
    register_factory(&inst, "chain_source", {
        let writers = writers.clone();
        move |name, _| {
            Ok(Arc::new(RestartDirectSource {
                name: name.into(),
                writers: writers.clone(),
            }))
        }
    });
    register_direct_forward(&inst, "chain_ident");
    register_factory(&inst, "chain_sink", {
        let received = received.clone();
        move |name, _| {
            Ok(Arc::new(RestartDirectConsumer {
                name: name.into(),
                consume: true,
                input: OnceLock::new(),
                received: received.clone(),
            }))
        }
    });
    let mut source = NodeRequest::new("chain_source", "source", serde_json::json!({}));
    source.restart = Some(RestartPolicy::RestartGroup);
    inst.create_node(source).unwrap();
    inst.create_node(NodeRequest::new(
        "chain_ident",
        "ident",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_node(NodeRequest::new(
        "chain_sink",
        "sink",
        serde_json::json!({}),
    ))
    .unwrap();
    let hop1 = inst
        .connect_edge("d1", "source", "out", "ident", "in", EdgeKind::Direct)
        .unwrap()
        .edge;
    let hop2 = inst
        .connect_edge("d2", "ident", "out", "sink", "in", EdgeKind::Direct)
        .unwrap()
        .edge;
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "source").unwrap();
    inst.add_group_member("g", "ident").unwrap();
    inst.add_group_member("g", "sink").unwrap();
    inst.start_group("g").unwrap();

    let stale = writers.lock().unwrap()[0].clone();
    assert_eq!(stale.push(media(1)), Push::Accepted);
    assert_eq!(hop1.occupied(), 0);
    assert_eq!(hop2.occupied(), 0);
    assert_eq!(received.load(Ordering::SeqCst), 1);

    stale.push_event(EdgeEvent::FlushStop);
    assert!(hop1.occupied() >= 1);

    let group = inst.group("g").unwrap();
    group.report_outcome(NodeOutcome::Failed {
        name: "source".into(),
        generation: 1,
        err: NodeError::new("source", NodePhase::Process, "restart direct chain"),
    });
    wait_for_group_generation(&group, 2);

    assert_eq!(hop1.occupied(), 0);
    assert_eq!(hop2.occupied(), 0);
    assert_eq!(stale.push(media(11)), Push::Closed);
    let fresh = writers.lock().unwrap()[1].clone();
    assert_eq!(fresh.push(media(2)), Push::Accepted);
    assert_eq!(hop1.occupied(), 0);
    assert_eq!(hop2.occupied(), 0);
    assert_eq!(received.load(Ordering::SeqCst), 2);
    inst.stop_group("g").unwrap();
}

#[cfg(feature = "async")]
#[test]
fn live_direct_chain_restart_rebinds_tails_to_buffered() {
    let inst = Instance::new();
    let writers = Arc::new(Mutex::new(Vec::new()));
    let received = Arc::new(AtomicUsize::new(0));
    let hold = Arc::new(AtomicBool::new(true));
    let release: Arc<dyn Edge> = Arc::new(BufferedEdge::new(1));
    register_factory(&inst, "tail_source", {
        let writers = writers.clone();
        move |name, _| {
            Ok(Arc::new(RestartDirectSource {
                name: name.into(),
                writers: writers.clone(),
            }))
        }
    });
    register_direct_forward(&inst, "tail_ident");
    register_factory(&inst, "tail_hold_sink", {
        let received = received.clone();
        let hold = hold.clone();
        let release = release.clone();
        move |name, _| {
            Ok(Arc::new(RestartHoldSink {
                name: name.into(),
                input: OnceLock::new(),
                received: received.clone(),
                hold: hold.clone(),
                release: release.clone(),
            }))
        }
    });
    let mut source = NodeRequest::new("tail_source", "source", serde_json::json!({}));
    source.restart = Some(RestartPolicy::RestartGroup);
    inst.create_node(source).unwrap();
    inst.create_node(NodeRequest::new(
        "tail_ident",
        "ident",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_node(NodeRequest::new(
        "tail_ident",
        "ident2",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_node(NodeRequest::new(
        "tail_hold_sink",
        "sink",
        serde_json::json!({}),
    ))
    .unwrap();
    let hop1 = inst
        .connect_edge("d1", "source", "out", "ident", "in", EdgeKind::Direct)
        .unwrap()
        .edge;
    let hop2 = inst
        .connect_edge("d2", "ident", "out", "ident2", "in", EdgeKind::Direct)
        .unwrap()
        .edge;
    let tail = inst
        .connect_edge(
            "b",
            "ident2",
            "out",
            "sink",
            "in",
            EdgeKind::Buffered { capacity: 1 },
        )
        .unwrap()
        .edge;
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "source").unwrap();
    inst.add_group_member("g", "ident").unwrap();
    inst.add_group_member("g", "ident2").unwrap();
    inst.add_group_member("g", "sink").unwrap();
    inst.start_group("g").unwrap();

    let generation_1 = writers.lock().unwrap()[0].clone();
    assert_eq!(generation_1.push(media(1)), Push::Accepted);
    wait_until(
        Duration::from_secs(2),
        || hop1.is_full() && hop2.is_full() && tail.is_full(),
        "Direct hops must follow the full Buffered tail",
    );
    assert_eq!(hop1.occupied(), 0);
    assert_eq!(hop2.occupied(), 0);

    let group = inst.group("g").unwrap();
    group.report_outcome(NodeOutcome::Failed {
        name: "source".into(),
        generation: 1,
        err: NodeError::new("source", NodePhase::Process, "restart direct tails"),
    });
    wait_for_group_generation(&group, 2);

    assert_eq!(generation_1.push(media(11)), Push::Closed);
    let generation_2 = writers.lock().unwrap()[1].clone();
    assert_eq!(generation_2.push(media(2)), Push::Accepted);
    wait_until(
        Duration::from_secs(2),
        || hop1.is_full() && hop2.is_full() && tail.is_full(),
        "restart must rebind Direct tails to the same Buffered hop",
    );
    assert_eq!(hop1.occupied(), 0);
    assert_eq!(hop2.occupied(), 0);

    hold.store(false, Ordering::Release);
    release.push_event(EdgeEvent::FlushStop);
    wait_until(
        Duration::from_secs(2),
        || received.load(Ordering::SeqCst) >= 1,
        "hold sink must drain after release",
    );
    inst.stop_group("g").unwrap();
}

struct EgressSource {
    name: String,
    writers: Arc<Mutex<Vec<Arc<dyn Edge>>>>,
}

impl Node for EgressSource {
    fn name(&self) -> &str {
        &self.name
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
        self.writers.lock().unwrap().push(edge);
    }
    fn process(&self) -> Blocked {
        std::thread::sleep(Duration::from_millis(1));
        Blocked::Again
    }
}

#[test]
fn supervisor_restart_preserves_egress_arc_media_and_suppresses_eof() {
    let inst = Instance::new();
    let writers = Arc::new(Mutex::new(Vec::new()));
    register_factory(&inst, "egress_source", {
        let writers = writers.clone();
        move |name, _| {
            Ok(Arc::new(EgressSource {
                name: name.into(),
                writers: writers.clone(),
            }))
        }
    });
    register_factory(&inst, "outside_sink", |name, _| {
        Ok(Arc::new(PollNode {
            name: name.into(),
            input: OnceLock::new(),
            received: Some(Arc::new(AtomicUsize::new(0))),
        }))
    });
    let mut source = NodeRequest::new("egress_source", "source", serde_json::json!({}));
    source.restart = Some(RestartPolicy::RestartGroup);
    inst.create_node(source).unwrap();
    inst.create_node(NodeRequest::new(
        "outside_sink",
        "outside",
        serde_json::json!({}),
    ))
    .unwrap();
    let external = inst
        .connect_edge(
            "egress",
            "source",
            "out",
            "outside",
            "in",
            EdgeKind::Buffered { capacity: 4 },
        )
        .unwrap()
        .edge;
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "source").unwrap();
    inst.start_group("g").unwrap();
    let group = inst.group("g").unwrap();
    let stale = writers.lock().unwrap()[0].clone();
    assert_eq!(stale.push(media(21)), Push::Accepted);
    stale.push_event(EdgeEvent::Eof);

    group.report_outcome(NodeOutcome::Failed {
        name: "source".into(),
        generation: 1,
        err: NodeError::new("source", NodePhase::Process, "restart"),
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    while group.generation() != 2 || group.state() != GroupState::Running {
        assert!(Instant::now() < deadline, "restart timed out");
        std::thread::sleep(Duration::from_millis(1));
    }

    assert!(Arc::ptr_eq(
        &external,
        &inst.edge_link("egress").unwrap().edge
    ));
    assert_eq!(stale.push(media(22)), Push::Closed);
    let fresh = writers.lock().unwrap()[1].clone();
    assert_eq!(fresh.push(media(23)), Push::Accepted);
    assert!(matches!(
        external.try_take(),
        Some(EdgeItem::Buffer(buf)) if buf.ts().val == 21
    ));
    assert!(matches!(
        external.try_take(),
        Some(EdgeItem::Buffer(buf)) if buf.ts().val == 23
    ));
    assert!(external.try_take().is_none());
    inst.stop_group("g").unwrap();
}

#[test]
fn buffered_stop_start_and_manual_restart_keep_nodes_edges_and_media_on_one_generation() {
    let inst = Instance::new();
    let writers = Arc::new(Mutex::new(Vec::new()));
    register_factory(&inst, "continuous_buffered_source", {
        let writers = writers.clone();
        move |name, _| {
            Ok(Arc::new(EgressSource {
                name: name.into(),
                writers: writers.clone(),
            }))
        }
    });
    register_factory(&inst, "continuous_buffered_sink", |name, _| {
        Ok(Arc::new(PollNode {
            name: name.into(),
            input: OnceLock::new(),
            received: Some(Arc::new(AtomicUsize::new(0))),
        }))
    });
    inst.create_node(NodeRequest::new(
        "continuous_buffered_source",
        "source",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_node(NodeRequest::new(
        "continuous_buffered_sink",
        "outside",
        serde_json::json!({}),
    ))
    .unwrap();
    let logical = inst
        .connect_edge(
            "continuous-buffered",
            "source",
            "out",
            "outside",
            "in",
            EdgeKind::Buffered { capacity: 4 },
        )
        .unwrap()
        .edge;
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "source").unwrap();

    inst.start_group("g").unwrap();
    let generation_1 = writers.lock().unwrap()[0].clone();
    assert_eq!(generation_1.push(media(1)), Push::Accepted);
    assert!(matches!(
        logical.try_take(),
        Some(EdgeItem::Buffer(buffer)) if buffer.ts().val == 1
    ));
    inst.stop_group("g").unwrap();

    inst.start_group("g").unwrap();
    let generation_2 = writers.lock().unwrap()[1].clone();
    assert_eq!(generation_1.push(media(11)), Push::Closed);
    assert_eq!(generation_2.push(media(2)), Push::Accepted);
    assert!(matches!(
        logical.try_take(),
        Some(EdgeItem::Buffer(buffer)) if buffer.ts().val == 2
    ));

    inst.restart_group("g").unwrap();
    let group = inst.group("g").unwrap();
    wait_for_group_generation(&group, 3);
    let generation_3 = writers.lock().unwrap()[2].clone();
    assert_eq!(generation_1.push(media(12)), Push::Closed);
    assert_eq!(generation_2.push(media(22)), Push::Closed);
    assert_eq!(generation_3.push(media(3)), Push::Accepted);
    assert!(matches!(
        logical.try_take(),
        Some(EdgeItem::Buffer(buffer)) if buffer.ts().val == 3
    ));
    inst.stop_group("g").unwrap();
}
