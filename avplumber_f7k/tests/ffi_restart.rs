use std::ffi::c_void;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use avplumber_f7k::abi::{
    AvpCore, AvpEdge, AvpNode, avp_core_create, avp_core_destroy, avp_create_edge,
    avp_create_group, avp_create_node, avp_destroy_group, avp_destroy_node, avp_edge_push,
    avp_group_add, avp_group_add_checked, avp_group_status, avp_lookup_node, avp_node_bind_sink,
    avp_node_bind_source, avp_node_impl, avp_node_query_interface, avp_node_set_impl,
    avp_register_node_factory, avp_restart_group, avp_start_group, avp_stop_group, avp_string_free,
};
use avplumber_f7k::{
    AvpBuffer, AvpInterfaceId, AvpMediaType, AvpNodeVtable, Blocked, EdgeRestart, GroupState, Node,
    NodeError, NodeOutcome, NodePhase, NodeRequest, RestartPolicy, register_factory,
};

/// One ABI buffer, freshly owned. [`avp_edge_push`] adopts the pointer whether
/// the edge accepts it or not, so every push needs its own — and with libav
/// compiled in the pointer really is an `AVFrame` that the push will free, which
/// is why this is cfg-paired rather than a dangling address in both builds.
fn abi_buffer() -> AvpBuffer {
    #[cfg(feature = "ffmpeg")]
    {
        let mut frame = rsmpeg::avutil::AVFrame::new();
        frame.set_width(2);
        frame.set_height(2);
        frame.set_format(rusty_ffmpeg::ffi::AV_PIX_FMT_GRAY8);
        frame.alloc_buffer().expect("2x2 gray8 abi frame");
        AvpBuffer {
            media: AvpMediaType::VIDEO,
            ptr: frame.into_raw().as_ptr() as *mut c_void,
        }
    }
    #[cfg(not(feature = "ffmpeg"))]
    {
        // `Media::Stub` carries the address as a timestamp and never reads it.
        AvpBuffer {
            media: AvpMediaType::VIDEO,
            ptr: std::ptr::dangling_mut(),
        }
    }
}

static TEST_LOCK: Mutex<()> = Mutex::new(());
static FACTORY_CALLS: AtomicUsize = AtomicUsize::new(0);
static DESTROYS: AtomicUsize = AtomicUsize::new(0);
static FIRST_HANDLE: AtomicUsize = AtomicUsize::new(0);
static STARTED_BEFORE_OLD_DESTROY: AtomicUsize = AtomicUsize::new(0);
static PUSH_EDGE: AtomicUsize = AtomicUsize::new(0);
static PUSH_FLOW: AtomicUsize = AtomicUsize::new(usize::MAX);
static HELPER_LEASE: AtomicUsize = AtomicUsize::new(0);

#[test]
fn group_status_and_restart_errors_are_available_through_c_abi() {
    let core = avp_core_create();
    let group = avp_create_group(core, c"status_group".as_ptr());
    let mut status = std::ptr::null_mut();
    let mut err = std::ptr::null();

    assert_eq!(avp_group_status(group, &mut status, &mut err), 0);
    assert!(err.is_null());
    let json = unsafe { std::ffi::CStr::from_ptr(status) }
        .to_str()
        .unwrap();
    assert!(json.contains(r#""state":"idle""#));
    avp_string_free(status);

    assert_eq!(avp_restart_group(group, &mut err), -1);
    assert!(!err.is_null());
    let message = unsafe { std::ffi::CStr::from_ptr(err) }.to_str().unwrap();
    assert!(message.contains("cannot restart from Idle"));
    avp_string_free(err.cast_mut());

    avp_destroy_group(core, group);
    avp_core_destroy(core);
}

#[test]
fn checked_c_group_add_reports_duplicate_policy_membership() {
    let _guard = TEST_LOCK.lock().unwrap();
    FACTORY_CALLS.store(0, Ordering::SeqCst);
    FIRST_HANDLE.store(0, Ordering::SeqCst);
    let core = avp_core_create();
    avp_register_node_factory(core, c"c_restart".as_ptr(), factory);
    let node = avp_create_node(
        core,
        c"c_restart".as_ptr(),
        c"member".as_ptr(),
        c"{\"auto_restart\":\"group\",\"token\":17}".as_ptr(),
        std::ptr::null_mut(),
    );
    let first = avp_create_group(core, c"first".as_ptr());
    let second = avp_create_group(core, c"second".as_ptr());
    let mut err = std::ptr::null();

    assert_eq!(avp_group_add_checked(first, node, &mut err), 0);
    assert!(err.is_null());
    assert_eq!(avp_group_add_checked(second, node, &mut err), -1);
    assert!(!err.is_null());
    assert!(
        unsafe { std::ffi::CStr::from_ptr(err) }
            .to_str()
            .unwrap()
            .contains("exactly one group")
    );
    avp_string_free(err.cast_mut());

    avp_destroy_group(core, second);
    avp_destroy_group(core, first);
    avp_destroy_node(core, node);
    avp_core_destroy(core);
}

struct CState {
    generation: usize,
}

extern "C" fn process(handle: *mut c_void) -> i32 {
    let state = avp_node_impl(handle.cast::<AvpNode>()).cast::<CState>();
    assert!(!state.is_null());
    let _ = unsafe { (*state).generation };
    std::thread::sleep(Duration::from_millis(1));
    0
}

extern "C" fn start(handle: *mut c_void) {
    let state = avp_node_impl(handle.cast::<AvpNode>()).cast::<CState>();
    let generation = unsafe { (*state).generation };
    if DESTROYS.load(Ordering::SeqCst) < generation.saturating_sub(1) {
        STARTED_BEFORE_OLD_DESTROY.fetch_add(1, Ordering::SeqCst);
    }
}

extern "C" fn destroy(handle: *mut c_void) {
    let state = avp_node_impl(handle.cast::<AvpNode>()).cast::<CState>();
    assert!(!state.is_null());
    unsafe {
        drop(Box::from_raw(state));
    }
    DESTROYS.fetch_add(1, Ordering::SeqCst);
}

static VTABLE: AvpNodeVtable = AvpNodeVtable {
    start: Some(start),
    stop: None,
    destroy: Some(destroy),
    process: Some(process),
    poll: None,
    query_interface: None,
};

extern "C" fn push_process(_handle: *mut c_void) -> i32 {
    let edge = PUSH_EDGE.load(Ordering::SeqCst) as *mut AvpEdge;
    let buffer = abi_buffer();
    PUSH_FLOW.store(avp_edge_push(edge, &buffer) as usize, Ordering::SeqCst);
    0
}

static PUSH_VTABLE: AvpNodeVtable = AvpNodeVtable {
    start: None,
    stop: None,
    destroy: Some(destroy),
    process: Some(push_process),
    poll: None,
    query_interface: None,
};

extern "C" fn helper_process(_handle: *mut c_void) -> i32 {
    std::thread::sleep(Duration::from_millis(1));
    0
}

static HELPER_VTABLE: AvpNodeVtable = AvpNodeVtable {
    start: None,
    stop: None,
    destroy: Some(destroy),
    process: Some(helper_process),
    poll: None,
    query_interface: None,
};

extern "C" fn push_factory(
    _core: *mut AvpCore,
    node: *mut AvpNode,
    _params: *const c_char,
) -> *mut AvpNode {
    let state = Box::into_raw(Box::new(CState { generation: 1 }));
    avp_node_set_impl(node, state.cast(), &PUSH_VTABLE);
    node
}

extern "C" fn helper_factory(
    _core: *mut AvpCore,
    node: *mut AvpNode,
    _params: *const c_char,
) -> *mut AvpNode {
    let lease = avp_node_bind_sink(node, c"helper-edge".as_ptr(), AvpMediaType::VIDEO, 4);
    if !lease.is_null() {
        HELPER_LEASE.store(lease as usize, Ordering::SeqCst);
    }
    let state = Box::into_raw(Box::new(CState { generation: 1 }));
    avp_node_set_impl(node, state.cast(), &HELPER_VTABLE);
    node
}

extern "C" fn factory(
    _core: *mut AvpCore,
    node: *mut AvpNode,
    params: *const c_char,
) -> *mut AvpNode {
    assert!(
        avp_node_impl(node).is_null(),
        "factory entry must not expose stale pending C state"
    );
    let first = FIRST_HANDLE.load(Ordering::SeqCst);
    if first == 0 {
        FIRST_HANDLE.store(node as usize, Ordering::SeqCst);
    } else {
        assert_eq!(
            node as usize, first,
            "every C generation must be built through the stable AvpNode handle"
        );
    }
    let params = unsafe { std::ffi::CStr::from_ptr(params) }
        .to_str()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(params).unwrap(),
        serde_json::json!({"token": 17})
    );
    let generation = FACTORY_CALLS.fetch_add(1, Ordering::SeqCst) + 1;
    let state = Box::into_raw(Box::new(CState { generation }));
    avp_node_set_impl(node, state.cast(), &VTABLE);
    node
}

static GATE_ENTERED: AtomicBool = AtomicBool::new(false);
static GATE_RESUME: AtomicBool = AtomicBool::new(false);

/// Same as `factory`, except the replacement generation parks inside the C
/// factory until the test releases it.
extern "C" fn gated_factory(
    _core: *mut AvpCore,
    node: *mut AvpNode,
    _params: *const c_char,
) -> *mut AvpNode {
    assert!(
        avp_node_impl(node).is_null(),
        "factory entry must not expose stale pending C state"
    );
    let generation = FACTORY_CALLS.fetch_add(1, Ordering::SeqCst) + 1;
    if generation == 2 {
        GATE_ENTERED.store(true, Ordering::SeqCst);
        while !GATE_RESUME.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    let state = Box::into_raw(Box::new(CState { generation }));
    avp_node_set_impl(node, state.cast(), &VTABLE);
    node
}

fn wait_for(what: &str, mut pred: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while !pred() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn wait_for_generation(core: *mut AvpCore, generation: u64) {
    let group = unsafe { &*core }.group("g").unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while group.generation() != generation || group.state() != GroupState::Running {
        assert!(Instant::now() < deadline, "restart timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn trigger_restart(core: *mut AvpCore, generation: u64) {
    let group = unsafe { &*core }.group("g").unwrap();
    group.report_outcome(NodeOutcome::Failed {
        name: "worker".into(),
        generation,
        err: NodeError::new("worker", NodePhase::Process, "planned restart"),
    });
    wait_for_generation(core, generation + 1);
}

#[test]
fn c_handle_stays_stable_across_repeated_factory_generations() {
    let _guard = TEST_LOCK.lock().unwrap();
    FACTORY_CALLS.store(0, Ordering::SeqCst);
    DESTROYS.store(0, Ordering::SeqCst);
    FIRST_HANDLE.store(0, Ordering::SeqCst);
    STARTED_BEFORE_OLD_DESTROY.store(0, Ordering::SeqCst);

    let core = avp_core_create();
    avp_register_node_factory(core, c"c_restart".as_ptr(), factory);
    let node = avp_create_node(
        core,
        c"c_restart".as_ptr(),
        c"worker".as_ptr(),
        c"{\"auto_restart\":\"group\",\"token\":17}".as_ptr(),
        std::ptr::null_mut(),
    );
    assert!(!node.is_null());
    let group = avp_create_group(core, c"g".as_ptr());
    avp_group_add(group, node);
    assert_eq!(avp_start_group(group, std::ptr::null_mut()), 0);
    wait_for_generation(core, 1);

    trigger_restart(core, 1);
    assert_eq!(avp_lookup_node(core, c"worker".as_ptr()), node);
    trigger_restart(core, 2);
    assert_eq!(avp_lookup_node(core, c"worker".as_ptr()), node);
    assert_eq!(FACTORY_CALLS.load(Ordering::SeqCst), 3);
    assert_eq!(
        STARTED_BEFORE_OLD_DESTROY.load(Ordering::SeqCst),
        0,
        "the old C generation must be destroyed before replacement start"
    );

    avp_stop_group(group, std::ptr::null_mut());
    avp_destroy_node(core, node);
    assert_eq!(
        DESTROYS.load(Ordering::SeqCst),
        3,
        "each successful C generation must be destroyed exactly once"
    );
    avp_destroy_group(core, group);
    avp_core_destroy(core);
}

#[test]
fn stale_c_callback_cannot_push_through_stable_edge_handle() {
    let _guard = TEST_LOCK.lock().unwrap();
    DESTROYS.store(0, Ordering::SeqCst);
    PUSH_FLOW.store(usize::MAX, Ordering::SeqCst);
    let core = avp_core_create();
    avp_register_node_factory(core, c"push_source".as_ptr(), push_factory);
    register_factory(unsafe { &*core }, "native_sink", |name, _| {
        Ok(Arc::new(NativeFactoryNode { name: name.into() }))
    });
    let producer = avp_create_node(
        core,
        c"push_source".as_ptr(),
        c"producer".as_ptr(),
        c"{}".as_ptr(),
        std::ptr::null_mut(),
    );
    let consumer = avp_create_node(
        core,
        c"native_sink".as_ptr(),
        c"consumer".as_ptr(),
        c"{}".as_ptr(),
        std::ptr::null_mut(),
    );
    let edge = avp_create_edge(
        core,
        c"egress".as_ptr(),
        producer,
        c"out".as_ptr(),
        consumer,
        c"in".as_ptr(),
        std::ptr::null(),
    );
    assert!(!edge.is_null());
    PUSH_EDGE.store(edge as usize, Ordering::SeqCst);

    let stale = unsafe { &*core }.node("producer").unwrap().node;
    stale.set_generation(1);
    let logical = unsafe { &*core }.edge_link("egress").unwrap().edge;
    logical.restart(1, 2, EdgeRestart::Egress);
    stale.process();

    assert_eq!(PUSH_FLOW.load(Ordering::SeqCst), 3);
    drop(stale);
    avp_destroy_node(core, producer);
    avp_destroy_node(core, consumer);
    avp_core_destroy(core);
}

#[test]
fn create_edge_handle_rejects_writes_after_producer_generation_changes() {
    let _guard = TEST_LOCK.lock().unwrap();
    let core = avp_core_create();
    avp_register_node_factory(core, c"push_source".as_ptr(), push_factory);
    register_factory(unsafe { &*core }, "native_sink", |name, _| {
        Ok(Arc::new(NativeFactoryNode { name: name.into() }))
    });
    let producer = avp_create_node(
        core,
        c"push_source".as_ptr(),
        c"producer".as_ptr(),
        c"{}".as_ptr(),
        std::ptr::null_mut(),
    );
    let consumer = avp_create_node(
        core,
        c"native_sink".as_ptr(),
        c"consumer".as_ptr(),
        c"{}".as_ptr(),
        std::ptr::null_mut(),
    );
    let edge = avp_create_edge(
        core,
        c"created".as_ptr(),
        producer,
        c"out".as_ptr(),
        consumer,
        c"in".as_ptr(),
        std::ptr::null(),
    );
    let logical = unsafe { &*core }.edge_link("created").unwrap().edge;
    let accepted = abi_buffer();
    assert_eq!(avp_edge_push(edge, &accepted) as usize, 0);

    logical.restart(1, 2, EdgeRestart::Egress);

    // A second buffer, not the same one again: the push above adopted the first.
    let fenced = abi_buffer();
    assert_eq!(
        avp_edge_push(edge, &fenced) as usize,
        3,
        "the returned create-edge writer lease must be generation-fenced"
    );
    avp_destroy_node(core, producer);
    avp_destroy_node(core, consumer);
    avp_core_destroy(core);
}

#[test]
fn stale_c_helper_thread_is_fenced_by_generation_lease() {
    let _guard = TEST_LOCK.lock().unwrap();
    HELPER_LEASE.store(0, Ordering::SeqCst);
    let core = avp_core_create();
    avp_register_node_factory(core, c"helper_source".as_ptr(), helper_factory);
    register_factory(unsafe { &*core }, "native_sink", |name, _| {
        Ok(Arc::new(NativeFactoryNode { name: name.into() }))
    });
    let producer = avp_create_node(
        core,
        c"helper_source".as_ptr(),
        c"producer".as_ptr(),
        c"{\"auto_restart\":\"group\"}".as_ptr(),
        std::ptr::null_mut(),
    );
    let consumer = avp_create_node(
        core,
        c"native_sink".as_ptr(),
        c"consumer".as_ptr(),
        c"{}".as_ptr(),
        std::ptr::null_mut(),
    );
    let lease = avp_node_bind_sink(producer, c"helper-edge".as_ptr(), AvpMediaType::VIDEO, 4);
    assert!(!lease.is_null());
    HELPER_LEASE.store(lease as usize, Ordering::SeqCst);
    assert!(
        !avp_node_bind_source(consumer, c"helper-edge".as_ptr(), AvpMediaType::VIDEO, 4).is_null()
    );
    let old_lease = HELPER_LEASE.load(Ordering::SeqCst);
    let logical = unsafe { &*core }.edge_link("helper-edge").unwrap().edge;
    logical.restart(1, 2, EdgeRestart::Egress);
    let new_lease =
        avp_node_bind_sink(producer, c"helper-edge".as_ptr(), AvpMediaType::VIDEO, 4) as usize;
    assert_ne!(old_lease, new_lease);
    let stale_flow = std::thread::spawn(move || {
        let buffer = abi_buffer();
        avp_edge_push(old_lease as *mut AvpEdge, &buffer) as usize
    })
    .join()
    .unwrap();
    let fresh_flow = std::thread::spawn(move || {
        let buffer = abi_buffer();
        avp_edge_push(new_lease as *mut AvpEdge, &buffer) as usize
    })
    .join()
    .unwrap();

    assert_eq!(
        stale_flow, 3,
        "stale helper-thread lease must report EOF/closed"
    );
    assert_eq!(fresh_flow, 0);
    avp_destroy_node(core, producer);
    avp_destroy_node(core, consumer);
    avp_core_destroy(core);
}

#[test]
fn stop_keeps_a_late_c_generation_off_the_stable_handle() {
    let _guard = TEST_LOCK.lock().unwrap();
    FACTORY_CALLS.store(0, Ordering::SeqCst);
    DESTROYS.store(0, Ordering::SeqCst);
    FIRST_HANDLE.store(0, Ordering::SeqCst);
    STARTED_BEFORE_OLD_DESTROY.store(0, Ordering::SeqCst);
    GATE_ENTERED.store(false, Ordering::SeqCst);
    GATE_RESUME.store(false, Ordering::SeqCst);

    let core = avp_core_create();
    avp_register_node_factory(core, c"c_gated".as_ptr(), gated_factory);
    let node = avp_create_node(
        core,
        c"c_gated".as_ptr(),
        c"worker".as_ptr(),
        c"{\"auto_restart\":\"group\"}".as_ptr(),
        std::ptr::null_mut(),
    );
    assert!(!node.is_null());
    let group = avp_create_group(core, c"g".as_ptr());
    avp_group_add(group, node);
    assert_eq!(avp_start_group(group, std::ptr::null_mut()), 0);
    wait_for_generation(core, 1);

    let native_group = unsafe { &*core }.group("g").unwrap();
    native_group.report_outcome(NodeOutcome::Failed {
        name: "worker".into(),
        generation: 1,
        err: NodeError::new("worker", NodePhase::Process, "planned restart"),
    });
    wait_for("the C replacement factory to be entered", || {
        GATE_ENTERED.load(Ordering::SeqCst)
    });
    let externally_visible = avp_node_impl(node).cast::<CState>();
    assert!(
        !externally_visible.is_null(),
        "helper threads must keep seeing the published generation during build"
    );
    assert_eq!(
        unsafe { (*externally_visible).generation },
        1,
        "pending factory state must be scoped to the factory thread"
    );

    avp_stop_group(group, std::ptr::null_mut());
    GATE_RESUME.store(true, Ordering::SeqCst);
    wait_for("the cancelled C generation to be destroyed", || {
        DESTROYS.load(Ordering::SeqCst) == 1
    });

    assert_eq!(
        unsafe { (*(avp_node_impl(node).cast::<CState>())).generation },
        1,
        "a cancelled rebuild must leave the stable handle on the live generation"
    );
    assert_eq!(avp_lookup_node(core, c"worker".as_ptr()), node);
    assert_eq!(native_group.generation(), 1);
    assert_eq!(native_group.state(), GroupState::Idle);

    avp_destroy_node(core, node);
    assert_eq!(
        DESTROYS.load(Ordering::SeqCst),
        2,
        "the live generation must still be destroyed exactly once"
    );
    avp_destroy_group(core, group);
    avp_core_destroy(core);
}

struct NativeInterfaceNode {
    name: String,
    generation: usize,
}

impl Node for NativeInterfaceNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&self) -> Blocked {
        std::thread::sleep(Duration::from_millis(1));
        Blocked::Again
    }

    fn query_interface(&self, _iface: AvpInterfaceId) -> Option<*const c_void> {
        Some(self.generation as *const c_void)
    }
}

#[test]
fn stable_c_handle_tracks_native_replacement_node() {
    let _guard = TEST_LOCK.lock().unwrap();
    let core = avp_core_create();
    let calls = Arc::new(AtomicUsize::new(0));
    register_factory(unsafe { &*core }, "native_restart", {
        let calls = calls.clone();
        move |name, _| {
            let generation = calls.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(Arc::new(NativeInterfaceNode {
                name: name.into(),
                generation,
            }))
        }
    });
    let node = avp_create_node(
        core,
        c"native_restart".as_ptr(),
        c"native".as_ptr(),
        c"{\"auto_restart\":\"group\"}".as_ptr(),
        std::ptr::null_mut(),
    );
    let group = avp_create_group(core, c"g".as_ptr());
    avp_group_add(group, node);
    assert_eq!(avp_start_group(group, std::ptr::null_mut()), 0);
    assert_eq!(
        avp_node_query_interface(node, AvpInterfaceId::Decoder as u32) as usize,
        1
    );

    let native_group = unsafe { &*core }.group("g").unwrap();
    native_group.report_outcome(NodeOutcome::Failed {
        name: "native".into(),
        generation: 1,
        err: NodeError::new("native", NodePhase::Process, "replace native node"),
    });
    wait_for_generation(core, 2);

    assert_eq!(
        avp_node_query_interface(node, AvpInterfaceId::Decoder as u32) as usize,
        2
    );
    avp_stop_group(group, std::ptr::null_mut());
    avp_destroy_node(core, node);
    avp_destroy_group(core, group);
    avp_core_destroy(core);
}

struct NativeFactoryNode {
    name: String,
}

impl Node for NativeFactoryNode {
    fn name(&self) -> &str {
        &self.name
    }
}

#[test]
fn failed_multi_member_rebuild_destroys_and_clears_pending_c_generation() {
    let _guard = TEST_LOCK.lock().unwrap();
    FACTORY_CALLS.store(0, Ordering::SeqCst);
    DESTROYS.store(0, Ordering::SeqCst);
    FIRST_HANDLE.store(0, Ordering::SeqCst);
    STARTED_BEFORE_OLD_DESTROY.store(0, Ordering::SeqCst);
    let core = avp_core_create();
    avp_register_node_factory(core, c"c_restart".as_ptr(), factory);
    let native_calls = Arc::new(AtomicUsize::new(0));
    register_factory(unsafe { &*core }, "fails_once", {
        let calls = native_calls.clone();
        move |name, _| {
            let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
            if call == 2 {
                return Err("fail after C replacement was built".into());
            }
            Ok(Arc::new(NativeFactoryNode { name: name.into() }))
        }
    });
    let c_node = avp_create_node(
        core,
        c"c_restart".as_ptr(),
        c"c_worker".as_ptr(),
        c"{\"auto_restart\":\"group\",\"token\":17}".as_ptr(),
        std::ptr::null_mut(),
    );
    let mut native_request = NodeRequest::new("fails_once", "native", serde_json::json!({}));
    native_request.restart = Some(RestartPolicy::Off);
    unsafe { &*core }.create_node(native_request).unwrap();
    let group = avp_create_group(core, c"g".as_ptr());
    avp_group_add(group, c_node);
    unsafe { &*core }.add_group_member("g", "native").unwrap();
    assert_eq!(avp_start_group(group, std::ptr::null_mut()), 0);

    let native_group = unsafe { &*core }.group("g").unwrap();
    native_group.report_outcome(NodeOutcome::Failed {
        name: "c_worker".into(),
        generation: 1,
        err: NodeError::new("c_worker", NodePhase::Process, "rebuild both"),
    });
    let deadline = Instant::now() + Duration::from_secs(3);
    while native_group.status().last_error.as_deref() != Some("fail after C replacement was built")
    {
        assert!(Instant::now() < deadline, "failed rebuild timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        DESTROYS.load(Ordering::SeqCst),
        1,
        "the unpublished C replacement must be destroyed"
    );
    assert_eq!(
        unsafe { (*(avp_node_impl(c_node).cast::<CState>())).generation },
        1,
        "the stable handle must still expose the old live generation"
    );

    while native_group.state() != GroupState::Running || native_group.generation() != 2 {
        assert!(Instant::now() < deadline, "automatic retry timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        unsafe { (*(avp_node_impl(c_node).cast::<CState>())).generation },
        3
    );
    avp_stop_group(group, std::ptr::null_mut());
    unsafe { &*core }.destroy_node("native").unwrap();
    avp_destroy_node(core, c_node);
    assert_eq!(DESTROYS.load(Ordering::SeqCst), 3);
    avp_destroy_group(core, group);
    avp_core_destroy(core);
}
