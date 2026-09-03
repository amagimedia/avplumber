//! Integration smoke: stub_source → stub_sink, plus native-control and ABI
//! construction paths.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use std::{ffi::c_void, os::raw::c_char};

use avplumber_f7k::abi::{
    AvpCore, AvpNode, avp_core_create, avp_core_destroy, avp_create_edge, avp_create_group,
    avp_create_node, avp_destroy_edge, avp_destroy_group, avp_destroy_node, avp_group_add,
    avp_lookup_node, avp_node_bind_sink, avp_node_bind_source, avp_node_impl, avp_node_set_impl,
    avp_register_node_factory, avp_start_group, avp_stop_group,
};
use avplumber_f7k::graph::AvpMediaType;
use avplumber_f7k::{
    AvpNodeVtable, Blocked, Edge, EdgeEvent, EdgeItem, Media, Node, NodeKind, Push,
    register_factory,
};

const NFRAMES: usize = 3;
static C_FACTORY_CALLS: AtomicUsize = AtomicUsize::new(0);
static C_FACTORY_DESTROYS: AtomicUsize = AtomicUsize::new(0);

extern "C" fn c_factory_process(handle: *mut c_void) -> i32 {
    let counter = avp_node_impl(handle.cast::<AvpNode>()).cast::<AtomicUsize>();
    unsafe { &*counter }.fetch_add(1, Ordering::Release);
    3
}

extern "C" fn c_factory_destroy(_handle: *mut c_void) {
    C_FACTORY_DESTROYS.fetch_add(1, Ordering::Release);
}

static C_FACTORY_VTABLE: AvpNodeVtable = AvpNodeVtable {
    start: None,
    stop: None,
    destroy: Some(c_factory_destroy),
    process: Some(c_factory_process),
    poll: None,
    query_interface: None,
};

extern "C" fn c_factory(
    _core: *mut AvpCore,
    node: *mut AvpNode,
    _params: *const c_char,
) -> *mut AvpNode {
    avp_node_set_impl(
        node,
        (&C_FACTORY_CALLS as *const AtomicUsize).cast_mut().cast(),
        &C_FACTORY_VTABLE,
    );
    node
}

struct StubSource {
    name: String,
    sink: OnceLock<Arc<dyn Edge>>,
    produced: AtomicUsize,
}

impl StubSource {
    fn new(name: String) -> Self {
        Self {
            name,
            sink: OnceLock::new(),
            produced: AtomicUsize::new(0),
        }
    }
}

impl Node for StubSource {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Blocking
    }
    fn pads(&self) -> avplumber_f7k::NodePads {
        avplumber_f7k::NodePads {
            sources: Vec::new(),
            sinks: vec![avplumber_f7k::PadDecl {
                name: "out".into(),
                media: AvpMediaType::VIDEO,
            }],
        }
    }
    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        let _ = self.sink.set(edge);
    }
    fn process(&self) -> Blocked {
        let sink = self.sink.get().expect("sink bound before process");
        let n = self.produced.load(Ordering::Acquire);
        if n >= NFRAMES {
            sink.push_event(EdgeEvent::Eof);
            return Blocked::Done;
        }
        let buf = make_media(n as i64);
        match sink.push(buf) {
            Push::Accepted => {
                self.produced.store(n + 1, Ordering::Release);
                Blocked::Again
            }
            Push::Full => Blocked::Again,
            Push::Closed | Push::Dropped => Blocked::Done,
        }
    }
}

struct StubSink {
    name: String,
    source: OnceLock<Arc<dyn Edge>>,
    received: Arc<AtomicUsize>,
}

impl StubSink {
    fn new(name: String, received: Arc<AtomicUsize>) -> Self {
        Self {
            name,
            source: OnceLock::new(),
            received,
        }
    }
}

impl Node for StubSink {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Blocking
    }
    fn pads(&self) -> avplumber_f7k::NodePads {
        avplumber_f7k::NodePads {
            sources: vec![avplumber_f7k::PadDecl {
                name: "in".into(),
                media: AvpMediaType::VIDEO,
            }],
            sinks: Vec::new(),
        }
    }
    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        let _ = self.source.set(edge);
    }
    fn process(&self) -> Blocked {
        let source = self.source.get().expect("source bound before process");
        match source.take(-1) {
            Some(EdgeItem::Buffer(_)) => {
                self.received.fetch_add(1, Ordering::Release);
                Blocked::Again
            }
            Some(EdgeItem::Event(EdgeEvent::Eof)) => Blocked::Done,
            Some(_) => Blocked::Again,
            None => Blocked::Done,
        }
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
            (*frame).width = 64;
            (*frame).height = 64;
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

fn register_stubs(inst: &avplumber_f7k::Instance, received: Arc<AtomicUsize>) {
    register_factory(inst, "stub_source", |name, _params| {
        Ok(Arc::new(StubSource::new(name.to_string())))
    });
    register_factory(inst, "stub_sink", move |name, _params| {
        Ok(Arc::new(StubSink::new(name.to_string(), received.clone())))
    });
}

#[test]
fn smoke_2node() {
    let core = avp_core_create();
    let inst = unsafe { &mut *core };

    let sink_received = Arc::new(AtomicUsize::new(0));
    register_stubs(inst, sink_received.clone());

    let src = avp_create_node(
        core,
        c"stub_source".as_ptr(),
        c"src".as_ptr(),
        c"{}".as_ptr(),
        std::ptr::null_mut(),
    );
    let sink = avp_create_node(
        core,
        c"stub_sink".as_ptr(),
        c"sink".as_ptr(),
        c"{}".as_ptr(),
        std::ptr::null_mut(),
    );
    assert!(
        !src.is_null() && !sink.is_null(),
        "create_node returned null"
    );

    let edge = avp_create_edge(
        core,
        c"e".as_ptr(),
        src,
        c"out".as_ptr(),
        sink,
        c"in".as_ptr(),
        std::ptr::null(),
    );
    assert!(!edge.is_null(), "create_edge returned null");

    let group = avp_create_group(core, c"g".as_ptr());
    avp_group_add(group, src);
    avp_group_add(group, sink);

    assert_eq!(
        avp_start_group(group, std::ptr::null_mut()),
        0,
        "start_group failed"
    );

    let deadline = Instant::now() + Duration::from_secs(10);
    while sink_received.load(Ordering::Acquire) < NFRAMES {
        if Instant::now() > deadline {
            panic!(
                "timed out: received {}/{} frames",
                sink_received.load(Ordering::Acquire),
                NFRAMES
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(sink_received.load(Ordering::Acquire), NFRAMES);

    avp_stop_group(group, std::ptr::null_mut());

    avp_destroy_edge(core, edge);
    avp_destroy_node(core, src);
    avp_destroy_node(core, sink);
    avp_destroy_group(core, group);
    avp_core_destroy(core);
}

#[test]
fn abi_named_binds_share_the_native_edge() {
    let core = avp_core_create();
    let inst = unsafe { &*core };
    register_stubs(inst, Arc::new(AtomicUsize::new(0)));
    let src = avp_create_node(
        core,
        c"stub_source".as_ptr(),
        c"src".as_ptr(),
        c"{}".as_ptr(),
        std::ptr::null_mut(),
    );
    let sink = avp_create_node(
        core,
        c"stub_sink".as_ptr(),
        c"sink".as_ptr(),
        c"{}".as_ptr(),
        std::ptr::null_mut(),
    );

    let producer_edge = avp_node_bind_sink(src, c"media".as_ptr(), AvpMediaType::VIDEO, 4);
    let consumer_edge = avp_node_bind_source(sink, c"media".as_ptr(), AvpMediaType::VIDEO, 4);

    assert!(!producer_edge.is_null());
    assert_eq!(producer_edge, consumer_edge);
    let link = inst.edge_link("media").expect("completed ABI edge");
    assert_eq!(
        (link.producer.as_str(), link.consumer.as_str()),
        ("src", "sink")
    );
    assert_eq!(
        (link.producer_pad.as_str(), link.consumer_pad.as_str()),
        ("out", "in")
    );

    avp_destroy_edge(core, producer_edge);
    avp_destroy_node(core, src);
    avp_destroy_node(core, sink);
    avp_core_destroy(core);
}

#[test]
fn c_factory_callback_keeps_its_original_handle() {
    C_FACTORY_CALLS.store(0, Ordering::Release);
    C_FACTORY_DESTROYS.store(0, Ordering::Release);
    let core = avp_core_create();
    avp_register_node_factory(core, c"c_stub".as_ptr(), c_factory);
    let node = avp_create_node(
        core,
        c"c_stub".as_ptr(),
        c"c_node".as_ptr(),
        c"{}".as_ptr(),
        std::ptr::null_mut(),
    );
    assert!(!node.is_null());
    let group = avp_create_group(core, c"g".as_ptr());
    avp_group_add(group, node);
    assert_eq!(avp_start_group(group, std::ptr::null_mut()), 0);

    let deadline = Instant::now() + Duration::from_secs(1);
    while C_FACTORY_CALLS.load(Ordering::Acquire) == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(C_FACTORY_CALLS.load(Ordering::Acquire), 1);

    avp_stop_group(group, std::ptr::null_mut());
    assert_eq!(avp_start_group(group, std::ptr::null_mut()), 0);
    let deadline = Instant::now() + Duration::from_secs(1);
    while C_FACTORY_CALLS.load(Ordering::Acquire) < 2 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    std::thread::sleep(Duration::from_millis(10));
    avp_stop_group(group, std::ptr::null_mut());
    assert_eq!(
        C_FACTORY_CALLS.load(Ordering::Acquire),
        2,
        "restart must schedule each member exactly once"
    );
    avp_destroy_node(core, node);
    assert_eq!(
        C_FACTORY_DESTROYS.load(Ordering::Acquire),
        2,
        "both fresh C generations must be destroyed exactly once"
    );
    avp_destroy_group(core, group);
    avp_core_destroy(core);
}

#[test]
fn destroying_a_half_bound_node_releases_its_edge_endpoint() {
    let inst = avplumber_f7k::Instance::new();
    register_stubs(&inst, Arc::new(AtomicUsize::new(0)));
    inst.create_node(avplumber_f7k::NodeRequest::new(
        "stub_source",
        "src",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.bind_edge("src", "out", avplumber_f7k::PadDirection::Output, "media")
        .unwrap();
    inst.destroy_node("src").unwrap();

    inst.create_node(avplumber_f7k::NodeRequest::new(
        "stub_sink",
        "sink",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.bind_edge("sink", "in", avplumber_f7k::PadDirection::Input, "media")
        .unwrap();
    inst.create_node(avplumber_f7k::NodeRequest::new(
        "stub_source",
        "src",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.bind_edge("src", "out", avplumber_f7k::PadDirection::Output, "media")
        .unwrap();

    let link = inst
        .edge_link("media")
        .expect("replacement endpoint linked");
    assert_eq!(
        (link.producer.as_str(), link.consumer.as_str()),
        ("src", "sink")
    );
}

#[test]
fn synthetic_clock_maps_pts() {
    use avplumber_f7k::graph::AvpRational;
    use avplumber_f7k::services::clock::{SyncGroup, SyntheticClock};
    let c = SyntheticClock::new();
    c.reset(0, AvpRational { num: 1, den: 1000 });
    assert_eq!(c.map_to_wall(100, AvpRational { num: 1, den: 1000 }), 100);
    c.set_paused(true);
    assert_eq!(
        c.map_to_wall(100, AvpRational { num: 1, den: 1000 }),
        avplumber_f7k::AVP_NOPTS
    );
}

#[test]
fn timeline_latest_at_or_before() {
    use avplumber_f7k::graph::AvpRational;
    use avplumber_f7k::services::timeline::{InMemoryTimeline, SharedTimeline};
    let t = InMemoryTimeline::new();
    t.set("ch", "k", 10, "\"a\"");
    t.set("ch", "k", 20, "\"b\"");
    let tb = AvpRational { num: 1, den: 1000 };
    assert_eq!(t.get("ch", "k", 15, tb), "\"a\"");
    assert_eq!(t.get("ch", "k", 20, tb), "\"b\"");
}

#[test]
fn control_unknown_command() {
    let core = avp_core_create();
    let inst = unsafe { &mut *core };
    let err = avplumber_f7k::abi::control::exec_line(inst, "nope").unwrap_err();
    assert!(err.contains("unknown"));
    avp_core_destroy(core);
}

#[test]
fn control_constructs_nodes_without_creating_c_handles() {
    let mut inst = avplumber_f7k::Instance::new();
    let received = Arc::new(AtomicUsize::new(0));
    register_stubs(&inst, received.clone());

    avplumber_f7k::control::exec_script(
        &mut inst,
        r#"
node.add {"type":"stub_source","name":"src","group":"g","dst":"media"}
node.add {"type":"stub_sink","name":"sink","group":"g","src":"media"}
group.start g
"#,
    )
    .unwrap();

    let core = &mut inst as *mut avplumber_f7k::Instance;
    assert!(
        avp_lookup_node(core, c"sink".as_ptr()).is_null(),
        "native control must not manufacture an ABI handle"
    );
    let link = inst.edge_link("media").expect("implicit edge is topology");
    assert_eq!(link.producer, "src");
    assert_eq!(link.consumer, "sink");
    assert_eq!(
        (link.producer_pad.as_str(), link.consumer_pad.as_str()),
        ("out", "in")
    );
    let deadline = Instant::now() + Duration::from_secs(1);
    while received.load(Ordering::Acquire) != NFRAMES && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(received.load(Ordering::Acquire), NFRAMES);
    avplumber_f7k::control::exec_line(&mut inst, "group.stop g").unwrap();
}

#[test]
fn spec_rearm_delivers_latched_spec() {
    use avplumber_f7k::AvpMediaType;
    use avplumber_f7k::AvpRational;
    use avplumber_f7k::graph::{BufferedEdge, Edge, EdgeEvent, EdgeItem, Spec};
    let e = BufferedEdge::new(8);
    e.push_event(EdgeEvent::Spec(Spec::Video {
        width: 1920,
        height: 1080,
        pix_fmt: 0,
        frame_rate: AvpRational { num: 25, den: 1 },
        sar: AvpRational { num: 1, den: 1 },
        time_base: AvpRational { num: 1, den: 25 },
    }));
    let _ = e.take(0);
    e.rearm_spec();
    match e.take(0) {
        Some(EdgeItem::Event(EdgeEvent::Spec(Spec::Video { width, .. }))) => {
            assert_eq!(width, 1920)
        }
        other => panic!("expected spec, got {other:?}"),
    }
    let _ = AvpMediaType::VIDEO;
}

struct PollNop {
    name: String,
}

impl Node for PollNop {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
}

#[test]
fn direct_edge_rejects_blocking_endpoints() {
    let inst = avplumber_f7k::Instance::new();
    register_factory(&inst, "blk", |name, _| {
        Ok(Arc::new(StubSource::new(name.to_string())))
    });
    register_factory(&inst, "polln", |name, _| {
        Ok(Arc::new(PollNop {
            name: name.to_string(),
        }))
    });
    inst.create_node(avplumber_f7k::NodeRequest::new(
        "blk",
        "src",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_node(avplumber_f7k::NodeRequest::new(
        "polln",
        "dst",
        serde_json::json!({}),
    ))
    .unwrap();
    let result = inst.connect_edge(
        "e",
        "src",
        "out",
        "dst",
        "in",
        avplumber_f7k::EdgeKind::Direct,
    );
    match result {
        Err(avplumber_f7k::CoreError::Invalid(msg)) => {
            assert!(msg.contains("Poll"), "{msg}");
        }
        Err(other) => panic!("expected Invalid, got {other}"),
        Ok(_) => panic!("DirectEdge must not connect a blocking node"),
    }
}

#[test]
fn edge_topology_mutation_is_rejected_while_an_endpoint_group_runs() {
    let inst = avplumber_f7k::Instance::new();
    register_stubs(&inst, Arc::new(AtomicUsize::new(0)));
    inst.create_node(avplumber_f7k::NodeRequest::new(
        "stub_source",
        "src",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_node(avplumber_f7k::NodeRequest::new(
        "stub_sink",
        "sink",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "src").unwrap();
    inst.start_group("g").unwrap();

    let error = inst
        .connect_edge(
            "late",
            "src",
            "out",
            "sink",
            "in",
            avplumber_f7k::EdgeKind::Buffered { capacity: 2 },
        )
        .err()
        .expect("connect must reject live topology mutation");
    assert!(error.to_string().contains("topology"), "{error}");
    let error = inst
        .bind_edge(
            "src",
            "out",
            avplumber_f7k::PadDirection::Output,
            "half-late",
        )
        .err()
        .expect("bind must reject live topology mutation");
    assert!(error.to_string().contains("topology"), "{error}");
    inst.stop_group("g").unwrap();
}
