use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

#[cfg(feature = "async")]
use avplumber_f7k::graph::NodeFuture;
#[cfg(feature = "async")]
use avplumber_f7k::{AsyncExecutor, ExecutorState, NodeKind};
use avplumber_f7k::{
    AvpMediaType, Blocked, BlockingExecutor, BuildCtx, Edge, EdgeKind, ExecCtxId, Executor, Graph,
    Group, GroupState, Instance, Node, NodeBody, NodeError, NodeOutcome, NodePads, NodePhase,
    NodeRequest, NodeSpec, PadDecl, RestartPolicy, Vertex, register_factory, register_spec,
};

/// Any buffer at all. The one push in this suite goes to a closed edge and is
/// rejected before anything looks inside it, so an empty packet does; the pair is
/// cfg'd because `Media::Stub` exists only when libav is compiled out.
#[cfg(feature = "ffmpeg")]
fn any_buffer() -> avplumber_f7k::Media {
    avplumber_f7k::Media::Packet(rsmpeg::avcodec::AVPacket::new())
}

#[cfg(not(feature = "ffmpeg"))]
fn any_buffer() -> avplumber_f7k::Media {
    avplumber_f7k::Media::Stub {
        kind: AvpMediaType::VIDEO,
        pts: 1,
    }
}

struct FailingNode {
    name: String,
}

impl Node for FailingNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        let name = self.name.clone();
        NodeBody::Blocking(Box::new(move || {
            Err(NodeError::new(
                name.clone(),
                NodePhase::Process,
                "planned failure",
            ))
        }))
    }
}

struct RunningNode {
    name: String,
}

impl Node for RunningNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&self) -> Blocked {
        std::thread::sleep(Duration::from_millis(1));
        Blocked::Again
    }
}

fn named_vertex(name: &str) -> Vertex {
    Vertex {
        name: name.into(),
        node: Arc::new(RunningNode { name: name.into() }),
        sources: Default::default(),
        sinks: Default::default(),
        source_media: Default::default(),
        sink_media: Default::default(),
    }
}

fn worker_vertex() -> Vertex {
    named_vertex("worker")
}

fn worker_graph() -> Arc<Mutex<Graph>> {
    let mut graph = Graph::new();
    graph.add_vertex(worker_vertex()).unwrap();
    Arc::new(Mutex::new(graph))
}

fn running_group_on(graph: Arc<Mutex<Graph>>) -> Group {
    let group = Group::new("test".into(), graph);
    group.add_with_policy("worker", ExecCtxId::Blocking, RestartPolicy::RestartGroup);
    group.start().unwrap();
    group
}

fn running_group() -> Group {
    running_group_on(worker_graph())
}

fn worker_fault(generation: u64) -> NodeOutcome {
    NodeOutcome::Failed {
        name: "worker".into(),
        generation,
        err: NodeError::new("worker", NodePhase::Process, "boom"),
    }
}

/// Commands and outcomes share one FIFO queue served by the manager thread, so
/// a completed round-trip proves every earlier report was already processed.
fn drain_manager_queue(group: &Group) {
    group
        .start()
        .expect("barrier round-trip on a running group must succeed");
}

fn wait_until(group: &Group, what: &str, mut pred: impl FnMut(&Group) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(4);
    while !pred(group) {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {what}; group is {:?} at generation {}",
            group.state(),
            group.generation()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn wait_for_state(group: &Group, wanted: GroupState) {
    wait_until(group, &format!("state {wanted:?}"), |g| g.state() == wanted);
}

fn wait_for(what: &str, mut pred: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(4);
    while !pred() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn restart_policy_strings_are_validated_during_construction() {
    let invalid =
        NodeRequest::from_json("unused", "worker", r#"{"auto_restart":"sometimes"}"#).unwrap_err();
    assert!(invalid.to_string().contains("invalid auto_restart"));

    let unsupported =
        NodeRequest::from_json("unused", "worker", r#"{"auto_restart":true}"#).unwrap_err();
    assert!(
        unsupported
            .to_string()
            .contains("restart_node is unsupported")
    );

    let request = NodeRequest::from_json(
        "unused",
        "worker",
        r#"{"auto_restart":"panic","on_error":"exit"}"#,
    )
    .unwrap();
    assert_eq!(request.restart, Some(RestartPolicy::Panic));
    assert_eq!(request.on_error, Some(RestartPolicy::Exit));

    for unsupported in [
        r#"{"auto_restart":"on"}"#,
        r#"{"auto_restart":"restart_node"}"#,
        r#"{"auto_restart":true}"#,
    ] {
        assert!(
            NodeRequest::from_json("unused", "worker", unsupported)
                .unwrap_err()
                .to_string()
                .contains("restart_node is unsupported")
        );
    }
    for alias in ["group", "restart_group"] {
        let json = format!(r#"{{"auto_restart":"{alias}"}}"#);
        assert_eq!(
            NodeRequest::from_json("unused", "worker", &json)
                .unwrap()
                .restart,
            Some(RestartPolicy::RestartGroup)
        );
    }
    assert!(
        NodeRequest::from_json("unused", "worker", r#"{"on_error":"maybe"}"#)
            .unwrap_err()
            .to_string()
            .contains("invalid on_error")
    );
}

#[test]
fn clean_completion_uses_auto_restart_but_error_can_be_disabled() {
    let graph = worker_graph();
    let group = Group::new("test".into(), graph.clone());
    group.add_full(
        "worker",
        ExecCtxId::Blocking,
        RestartPolicy::RestartGroup,
        Some(RestartPolicy::Off),
        None,
    );
    group.set_restart_hook(Arc::new(move |request| {
        let mut graph = graph.lock().unwrap();
        graph.remove_vertex("worker");
        graph.add_vertex(worker_vertex())?;
        Ok(request.reconstructed())
    }));
    group.start().unwrap();
    let generation = group.generation();

    group.report_outcome(NodeOutcome::Completed {
        name: "worker".into(),
        generation,
    });
    wait_until(&group, "clean completion restart", |g| {
        g.generation() == generation + 1 && g.state() == GroupState::Running
    });

    group.report_outcome(worker_fault(group.generation()));
    wait_for_state(&group, GroupState::Failed);
    assert_eq!(group.status().restart_count, 1);
    assert_eq!(group.status().last_error.as_deref(), Some("boom"));
    group.stop();
}

#[test]
fn cancellation_never_triggers_restart_policy() {
    let group = running_group();
    let generation = group.generation();
    group.report_outcome(NodeOutcome::Cancelled {
        name: "worker".into(),
        generation,
    });
    drain_manager_queue(&group);
    assert_eq!(group.state(), GroupState::Running);
    assert_eq!(group.status().restart_count, 0);
    assert_eq!(group.status().last_outcome, None);
    group.stop();
}

#[test]
fn clean_completion_with_off_propagates_one_natural_eof() {
    let edge: Arc<dyn Edge> = Arc::new(avplumber_f7k::BufferedEdge::new(4));
    let mut vertex = worker_vertex();
    vertex.sinks.insert("out".into(), edge.clone());
    let mut graph = Graph::new();
    graph.add_vertex(vertex).unwrap();
    let group = Group::new("test".into(), Arc::new(Mutex::new(graph)));
    group.add_with_policy("worker", ExecCtxId::Blocking, RestartPolicy::Off);
    group.start().unwrap();
    let generation = group.generation();

    let completed = NodeOutcome::Completed {
        name: "worker".into(),
        generation,
    };
    group.report_outcome(completed.clone());
    group.report_outcome(completed);
    drain_manager_queue(&group);

    assert!(matches!(
        edge.take(0),
        Some(avplumber_f7k::EdgeItem::Event(
            avplumber_f7k::EdgeEvent::Eof
        ))
    ));
    assert!(
        edge.take(0).is_none(),
        "EOF must be propagated exactly once"
    );
    assert_eq!(group.state(), GroupState::Running);
    group.stop();
}

/// `last_outcome` holds one summary, so the node that finished before the last
/// one is unobservable through it — a client polling for "did node X finish?"
/// can miss X entirely between two polls. `outcomes` is the generation's whole
/// list, in arrival order, and is what such a client must read.
#[test]
fn every_outcome_of_a_generation_is_recorded_not_only_the_last() {
    let mut graph = Graph::new();
    graph.add_vertex(worker_vertex()).unwrap();
    graph.add_vertex(named_vertex("second")).unwrap();
    let group = Group::new("test".into(), Arc::new(Mutex::new(graph)));
    group.add_with_policy("worker", ExecCtxId::Blocking, RestartPolicy::Off);
    group.add_with_policy("second", ExecCtxId::Blocking, RestartPolicy::Off);
    group.start().unwrap();
    let generation = group.generation();

    for name in ["worker", "second"] {
        group.report_outcome(NodeOutcome::Completed {
            name: name.into(),
            generation,
        });
    }
    drain_manager_queue(&group);

    let status = group.status();
    assert_eq!(
        status.outcomes,
        vec![
            format!("completed:worker:generation={generation}"),
            format!("completed:second:generation={generation}"),
        ]
    );
    assert_eq!(
        status.last_outcome,
        Some(format!("completed:second:generation={generation}")),
        "the single slot keeps only the newest, which is the whole point of the list"
    );
    group.stop();
}

/// The list describes one generation, so a restarted group does not report the
/// previous generation's completions as if its own nodes had finished.
#[test]
fn a_restarted_generation_starts_with_an_empty_outcome_list() {
    let graph = worker_graph();
    let group = running_group_on(graph.clone());
    group.set_restart_hook(Arc::new(move |request| {
        let mut graph = graph.lock().unwrap();
        graph.remove_vertex("worker");
        graph.add_vertex(worker_vertex())?;
        Ok(request.reconstructed())
    }));
    let generation = group.generation();

    group.report_outcome(NodeOutcome::Completed {
        name: "worker".into(),
        generation,
    });
    wait_until(&group, "the restarted generation", |g| {
        g.generation() == generation + 1 && g.state() == GroupState::Running
    });

    let status = group.status();
    assert!(
        status.outcomes.is_empty(),
        "generation {} reported nothing yet, but its status lists {:?}",
        status.generation,
        status.outcomes
    );
    assert_eq!(
        status.last_outcome,
        Some(format!("completed:worker:generation={generation}")),
        "the slot is not generation-scoped, which is why the list has to be"
    );
    group.stop();
}

#[test]
fn failure_with_off_marks_failed_without_egress_eof() {
    let edge: Arc<dyn Edge> = Arc::new(avplumber_f7k::BufferedEdge::new(4));
    let mut vertex = worker_vertex();
    vertex.sinks.insert("out".into(), edge.clone());
    let mut graph = Graph::new();
    graph.add_vertex(vertex).unwrap();
    let group = Group::new("test".into(), Arc::new(Mutex::new(graph)));
    group.add_with_policy("worker", ExecCtxId::Blocking, RestartPolicy::Off);
    group.start().unwrap();

    group.report_outcome(worker_fault(group.generation()));
    wait_for_state(&group, GroupState::Failed);

    assert!(edge.take(0).is_none());
    assert!(
        !edge.is_closed(),
        "failure must not masquerade as natural EOF"
    );
    group.stop();
}

#[test]
fn policy_node_must_belong_to_exactly_one_native_group() {
    let inst = Instance::new();
    register_factory(&inst, "membership_node", |name, _| {
        Ok(Arc::new(RunningNode {
            name: name.to_string(),
        }))
    });
    let mut request = NodeRequest::new("membership_node", "worker", serde_json::json!({}));
    request.restart = Some(RestartPolicy::RestartGroup);
    inst.create_node(request).unwrap();
    inst.create_group("one").unwrap();
    inst.create_group("two").unwrap();

    inst.add_group_member("one", "worker").unwrap();
    let error = inst.add_group_member("two", "worker").unwrap_err();
    assert!(error.to_string().contains("exactly one group"));

    let node = inst.node("worker").unwrap();
    let second = inst.group("two").unwrap();
    second.add_full(
        "worker",
        node.exec_ctx,
        node.restart,
        node.on_error,
        node.service_hint,
    );
    let error = inst.start_group("one").unwrap_err();
    assert!(error.to_string().contains("found 2"));
    second.remove("worker");

    inst.start_group("one").unwrap();
    assert!(inst.restart_group("one").is_ok());
    inst.stop_group("one").unwrap();
}

#[test]
fn manual_restart_rebuilds_the_running_group() {
    let graph = worker_graph();
    let group = running_group_on(graph.clone());
    group.set_restart_hook(Arc::new(move |request| {
        assert!(request.manual);
        let mut graph = graph.lock().unwrap();
        graph.remove_vertex("worker");
        graph.add_vertex(worker_vertex())?;
        Ok(request.reconstructed())
    }));
    let generation = group.generation();

    group.restart().unwrap();
    wait_until(&group, "manual restart", |g| {
        g.generation() == generation + 1 && g.state() == GroupState::Running
    });
    assert_eq!(group.status().restart_count, 1);
    group.stop();
}

#[test]
fn control_exposes_group_status_and_restart_errors() {
    let inst = Instance::new();
    inst.create_group("g").unwrap();

    let status = avplumber_f7k::control::exec_line(&inst, "group.status g").unwrap();
    let status: serde_json::Value = serde_json::from_str(&status).unwrap();
    assert_eq!(status["state"], "idle");
    assert_eq!(status["generation"], 0);
    assert_eq!(status["restart_count"], 0);

    let error = avplumber_f7k::control::exec_line(&inst, "group.restart g").unwrap_err();
    assert!(error.contains("cannot restart from Idle"));
    assert!(
        avplumber_f7k::control::exec_line(&inst, "group.status missing")
            .unwrap_err()
            .contains("unknown group")
    );
    let unsupported = avplumber_f7k::control::exec_line(
        &inst,
        r#"node.add {"type":"unused","name":"orphan","auto_restart":"group"}"#,
    )
    .unwrap_err();
    assert!(unsupported.contains("requires group membership"));
}

#[test]
fn panic_and_exit_route_to_distinct_instance_hooks() {
    let inst = Instance::new();
    register_factory(&inst, "running_hook_node", |name, _| {
        Ok(Arc::new(RunningNode {
            name: name.to_string(),
        }))
    });
    let (tx, rx) = mpsc::channel();
    inst.set_supervisor_action_hook(Arc::new(move |action| {
        tx.send(action).unwrap();
    }));

    for (name, policy) in [
        ("panic_worker", RestartPolicy::Panic),
        ("exit_worker", RestartPolicy::Exit),
    ] {
        let mut request = NodeRequest::new("running_hook_node", name, serde_json::json!({}));
        request.on_error = Some(policy);
        inst.create_node(request).unwrap();
        inst.create_group(name).unwrap();
        inst.add_group_member(name, name).unwrap();
        inst.start_group(name).unwrap();
        let group = inst.group(name).unwrap();
        group.report_outcome(NodeOutcome::Failed {
            name: name.into(),
            generation: group.generation(),
            err: NodeError::new(name, NodePhase::Process, "hook failure"),
        });
        wait_for_state(&group, GroupState::Failed);
    }

    assert!(matches!(
        rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        avplumber_f7k::SupervisorAction::Panic { .. }
    ));
    assert!(matches!(
        rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        avplumber_f7k::SupervisorAction::Exit { .. }
    ));
}

#[test]
fn supervisor_action_hook_can_reenter_stop_without_manager_deadlock() {
    let inst = Arc::new(Instance::new());
    register_factory(&inst, "reentrant_hook_node", |name, _| {
        Ok(Arc::new(RunningNode {
            name: name.to_string(),
        }))
    });
    let mut request = NodeRequest::new("reentrant_hook_node", "worker", serde_json::json!({}));
    request.on_error = Some(RestartPolicy::Exit);
    inst.create_node(request).unwrap();
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "worker").unwrap();
    inst.start_group("g").unwrap();

    let weak = Arc::downgrade(&inst);
    let (done_tx, done_rx) = mpsc::channel();
    inst.set_supervisor_action_hook(Arc::new(move |_| {
        weak.upgrade().unwrap().stop_group("g").unwrap();
        done_tx.send(()).unwrap();
    }));
    let group = inst.group("g").unwrap();
    group.report_outcome(worker_fault(group.generation()));

    done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("reentrant hook deadlocked on the group manager");
    assert_eq!(group.state(), GroupState::Idle);
}

#[test]
fn default_exit_action_stops_all_instance_groups_orderly() {
    let inst = Instance::new();
    register_factory(&inst, "exit_node", |name, _| {
        Ok(Arc::new(RunningNode {
            name: name.to_string(),
        }))
    });
    for (node_name, group_name) in [("exit_worker", "exit_group"), ("peer", "peer_group")] {
        let mut request = NodeRequest::new("exit_node", node_name, serde_json::json!({}));
        if node_name == "exit_worker" {
            request.on_error = Some(RestartPolicy::Exit);
        }
        inst.create_node(request).unwrap();
        inst.create_group(group_name).unwrap();
        inst.add_group_member(group_name, node_name).unwrap();
        inst.start_group(group_name).unwrap();
    }
    let exit_group = inst.group("exit_group").unwrap();
    let peer_group = inst.group("peer_group").unwrap();

    exit_group.report_outcome(NodeOutcome::Failed {
        name: "exit_worker".into(),
        generation: exit_group.generation(),
        err: NodeError::new("exit_worker", NodePhase::Process, "orderly exit"),
    });

    wait_for("orderly instance shutdown", || {
        exit_group.state() == GroupState::Idle && peer_group.state() == GroupState::Idle
    });
}

#[test]
fn default_panic_action_shuts_down_before_escalating() {
    let inst = Instance::new();
    register_factory(&inst, "panic_node", |name, _| {
        Ok(Arc::new(RunningNode {
            name: name.to_string(),
        }))
    });
    for (node_name, group_name) in [("panic_worker", "panic_group"), ("peer", "peer_group")] {
        let mut request = NodeRequest::new("panic_node", node_name, serde_json::json!({}));
        if node_name == "panic_worker" {
            request.on_error = Some(RestartPolicy::Panic);
        }
        inst.create_node(request).unwrap();
        inst.create_group(group_name).unwrap();
        inst.add_group_member(group_name, node_name).unwrap();
        inst.start_group(group_name).unwrap();
    }
    let panic_group = inst.group("panic_group").unwrap();
    let peer_group = inst.group("peer_group").unwrap();

    panic_group.report_outcome(NodeOutcome::Failed {
        name: "panic_worker".into(),
        generation: panic_group.generation(),
        err: NodeError::new("panic_worker", NodePhase::Process, "panic escalation"),
    });

    wait_for("panic shutdown before escalation", || {
        panic_group.state() == GroupState::Idle && peer_group.state() == GroupState::Idle
    });
}

#[test]
fn blocking_failure_is_delivered_before_join() {
    let exec = BlockingExecutor::new();
    let (tx, rx) = mpsc::channel();
    exec.configure_run(
        7,
        Arc::new(move |outcome| {
            tx.send(outcome).unwrap();
        }),
    );
    exec.add_node(
        Arc::new(FailingNode {
            name: "broken".into(),
        }),
        Vec::new(),
        Vec::new(),
    );

    exec.start().unwrap();
    let outcome = rx
        .recv_timeout(Duration::from_secs(1))
        .expect("failure must be reported without waiting for join");

    assert!(matches!(
        outcome,
        NodeOutcome::Failed {
            name,
            generation: 7,
            err,
        } if name == "broken" && err.message == "planned failure"
    ));
    exec.join();
}

#[cfg(feature = "async")]
struct AsyncTestNode {
    name: String,
    hangs: bool,
}

#[cfg(feature = "async")]
impl Node for AsyncTestNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> NodeKind {
        NodeKind::Async
    }

    fn run_async(self: Arc<Self>) -> NodeFuture {
        Box::pin(async move {
            if self.hangs {
                std::future::pending::<()>().await;
                Ok(())
            } else {
                Err(NodeError::new(
                    self.name.clone(),
                    NodePhase::Async,
                    "planned async failure",
                ))
            }
        })
    }
}

#[cfg(feature = "async")]
#[test]
fn async_failure_is_not_hidden_behind_an_earlier_handle() {
    let exec = AsyncExecutor::new();
    let (tx, rx) = mpsc::channel();
    exec.configure_run(
        11,
        Arc::new(move |outcome| {
            tx.send(outcome).unwrap();
        }),
    );
    exec.add_node(
        Arc::new(AsyncTestNode {
            name: "hanging".into(),
            hangs: true,
        }),
        Vec::new(),
        Vec::new(),
    );
    exec.add_node(
        Arc::new(AsyncTestNode {
            name: "broken".into(),
            hangs: false,
        }),
        Vec::new(),
        Vec::new(),
    );

    exec.start().unwrap();
    let outcome = rx
        .recv_timeout(Duration::from_secs(1))
        .expect("second task failure must bypass the first task's handle");

    assert!(matches!(
        outcome,
        NodeOutcome::Failed {
            name,
            generation: 11,
            err,
        } if name == "broken" && err.message == "planned async failure"
    ));
    exec.stop();
    exec.join();
}

#[cfg(feature = "async")]
#[test]
fn async_executor_reports_stopping_before_it_is_joined() {
    let exec = AsyncExecutor::new();
    exec.add_node(
        Arc::new(AsyncTestNode {
            name: "hanging".into(),
            hangs: true,
        }),
        Vec::new(),
        Vec::new(),
    );

    exec.start().unwrap();
    exec.stop();
    assert_eq!(exec.state(), ExecutorState::Stopping);

    exec.join();
    assert_eq!(exec.state(), ExecutorState::Stopped);
}

#[test]
fn stale_generation_outcome_is_ignored() {
    let group = running_group();
    assert_eq!(group.generation(), 1);

    group.report_outcome(NodeOutcome::Failed {
        name: "worker".into(),
        generation: 0,
        err: NodeError::new("worker", NodePhase::Process, "stale"),
    });
    drain_manager_queue(&group);

    assert_eq!(group.state(), GroupState::Running);
    assert_eq!(group.generation(), 1);
    group.stop();
}

#[test]
fn completion_does_not_suppress_a_later_fault_from_the_same_node() {
    let group = running_group();
    let generation = group.generation();

    group.report_outcome(NodeOutcome::Completed {
        name: "worker".into(),
        generation,
    });
    group.report_outcome(worker_fault(generation));
    wait_for_state(&group, GroupState::Failed);

    group.stop();
}

#[test]
fn start_failure_leaves_the_group_failed() {
    let group = Group::new("test".into(), Arc::new(Mutex::new(Graph::new())));
    group.add_with_policy("ghost", ExecCtxId::Blocking, RestartPolicy::RestartGroup);

    let err = group.start().unwrap_err();

    assert!(err.contains("ghost"), "unexpected start error: {err}");
    assert_eq!(group.state(), GroupState::Failed);
}

#[test]
fn duplicate_faults_are_coalesced_and_failed_reconstruction_retries() {
    let graph = worker_graph();
    let group = running_group_on(graph.clone());
    let calls = Arc::new(AtomicUsize::new(0));
    group.set_restart_hook({
        let calls = calls.clone();
        Arc::new(move |request| {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err("planned rebuild failure".into());
            }
            let mut graph = graph.lock().unwrap();
            graph.remove_vertex("worker");
            graph.add_vertex(worker_vertex())?;
            Ok(request.reconstructed())
        })
    });
    let generation = group.generation();
    let started = Instant::now();

    group.report_outcome(worker_fault(generation));
    group.report_outcome(worker_fault(generation));
    wait_until(&group, "retry to start rebuilt generation", |g| {
        g.generation() == generation + 1 && g.state() == GroupState::Running
    });

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(started.elapsed() >= Duration::from_millis(1900));
    assert_eq!(group.status().restart_count, 1);
    assert_eq!(group.status().last_error, None);
    group.stop();
}

#[test]
fn restart_is_refused_when_the_hook_leaves_the_old_nodes_in_place() {
    let group = running_group();
    let calls = Arc::new(AtomicUsize::new(0));
    group.set_restart_hook({
        let calls = calls.clone();
        Arc::new(move |request| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(request.reconstructed())
        })
    });
    let generation = group.generation();

    group.report_outcome(worker_fault(generation));
    wait_for("reconstruction identity rejection", || {
        calls.load(Ordering::SeqCst) == 1
            && group.state() == GroupState::Backoff
            && group.status().last_error.is_some()
    });

    assert_eq!(
        group.generation(),
        generation,
        "a claimed reconstruction that reuses the old node must not start a generation"
    );
    group.stop();
}

#[test]
fn restart_starts_a_new_generation_once_the_hook_rebuilds_the_nodes() {
    let graph = worker_graph();
    let group = running_group_on(graph.clone());
    group.set_restart_hook(Arc::new(move |request| {
        let mut graph = graph.lock().unwrap();
        graph.remove_vertex("worker");
        graph.add_vertex(worker_vertex())?;
        Ok(request.reconstructed())
    }));
    let generation = group.generation();

    group.report_outcome(worker_fault(generation));
    wait_until(&group, "the rebuilt generation to run", |g| {
        g.generation() == generation + 1 && g.state() == GroupState::Running
    });

    group.stop();
}

#[test]
fn manual_stop_cancels_pending_restart() {
    let group = running_group();
    let calls = Arc::new(AtomicUsize::new(0));
    group.set_restart_hook({
        let calls = calls.clone();
        Arc::new(move |request| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(request.reconstructed())
        })
    });
    let generation = group.generation();

    group.report_outcome(NodeOutcome::Panicked {
        name: "worker".into(),
        generation,
        message: "boom".into(),
    });
    wait_for_state(&group, GroupState::Backoff);
    group.stop();
    std::thread::sleep(Duration::from_millis(150));

    assert_eq!(group.state(), GroupState::Idle);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn manual_restart_forces_and_coalesces_pending_backoff() {
    let graph = worker_graph();
    let group = running_group_on(graph.clone());
    group.set_restart_hook(Arc::new(move |request| {
        assert!(request.manual);
        let mut graph = graph.lock().unwrap();
        graph.remove_vertex("worker");
        graph.add_vertex(worker_vertex())?;
        Ok(request.reconstructed())
    }));
    let generation = group.generation();
    group.report_outcome(worker_fault(generation));
    wait_for_state(&group, GroupState::Backoff);

    let forced = Instant::now();
    group.restart().unwrap();
    group.restart().unwrap();
    wait_until(&group, "forced pending restart", |candidate| {
        candidate.generation() == generation + 1 && candidate.state() == GroupState::Running
    });

    assert!(forced.elapsed() < Duration::from_millis(500));
    assert_eq!(group.status().restart_count, 1);
    group.stop();
}

#[test]
fn stop_remains_responsive_while_reconstruction_factory_is_blocked() {
    let inst = Instance::new();
    let (started_tx, started_rx) = mpsc::channel();
    let calls = Arc::new(AtomicUsize::new(0));
    register_factory(&inst, "slow_rebuild", {
        let calls = calls.clone();
        move |name, _| {
            if calls.fetch_add(1, Ordering::SeqCst) == 1 {
                started_tx.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(400));
            }
            Ok(Arc::new(RunningNode {
                name: name.to_string(),
            }))
        }
    });
    let mut request = NodeRequest::new("slow_rebuild", "worker", serde_json::json!({}));
    request.restart = Some(RestartPolicy::RestartGroup);
    inst.create_node(request).unwrap();
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "worker").unwrap();
    inst.start_group("g").unwrap();
    let group = inst.group("g").unwrap();
    let generation = group.generation();
    group.report_outcome(worker_fault(generation));
    started_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("restart hook did not begin");

    let before = Instant::now();
    inst.stop_group("g").unwrap();

    assert!(
        before.elapsed() < Duration::from_millis(100),
        "stop waited for the reconstruction factory"
    );
    assert_eq!(group.state(), GroupState::Idle);
    assert!(
        inst.start_group("g").is_err(),
        "a cancelled reconstruction must finish before another generation starts"
    );
}

struct RebuiltNode {
    name: String,
    fail: bool,
}

impl Node for RebuiltNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        if self.fail {
            let name = self.name.clone();
            NodeBody::Blocking(Box::new(move || {
                Err(NodeError::new(
                    name.clone(),
                    NodePhase::Process,
                    "first generation fails",
                ))
            }))
        } else {
            NodeBody::Blocking(Box::new(|| {
                std::thread::sleep(Duration::from_millis(1));
                Ok(Blocked::Again)
            }))
        }
    }
}

#[test]
fn instance_restart_calls_factory_again_for_fresh_node_state() {
    let inst = Instance::new();
    let calls = Arc::new(AtomicUsize::new(0));
    register_factory(&inst, "rebuilt", {
        let calls = calls.clone();
        move |name, params| {
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(params).unwrap(),
                serde_json::json!({"initial": 17})
            );
            let generation = calls.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(Arc::new(RebuiltNode {
                name: name.to_string(),
                fail: generation == 1,
            }))
        }
    });
    let mut request = NodeRequest::new("rebuilt", "worker", serde_json::json!({"initial": 17}));
    request.restart = Some(RestartPolicy::RestartGroup);
    let first = inst.create_node(request).unwrap().node;
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "worker").unwrap();

    inst.start_group("g").unwrap();
    let group = inst.group("g").expect("created group");
    wait_until(&group, "the factory-built replacement generation", |g| {
        g.generation() == 2 && g.state() == GroupState::Running
    });

    let second = inst.node("worker").unwrap().node;
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        !Arc::ptr_eq(&first, &second),
        "restart must install a fresh factory result"
    );
    inst.stop_group("g").unwrap();
}

#[test]
fn start_after_stop_reconstructs_every_member_before_advancing_generation() {
    let inst = Instance::new();
    let calls = Arc::new(AtomicUsize::new(0));
    register_factory(&inst, "fresh_on_start", {
        let calls = calls.clone();
        move |name, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(RunningNode {
                name: name.to_string(),
            }))
        }
    });
    inst.create_node(NodeRequest::new(
        "fresh_on_start",
        "worker",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "worker").unwrap();

    inst.start_group("g").unwrap();
    let first = inst.node("worker").unwrap().node;
    assert_eq!(inst.group("g").unwrap().generation(), 1);
    inst.stop_group("g").unwrap();

    inst.start_group("g").unwrap();
    let second = inst.node("worker").unwrap().node;
    assert_eq!(inst.group("g").unwrap().generation(), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        !Arc::ptr_eq(&first, &second),
        "a consumed node body must never be retaken after stop"
    );
    inst.stop_group("g").unwrap();
}

static SPEC_BUILDS: AtomicUsize = AtomicUsize::new(0);

#[derive(serde::Deserialize)]
struct RestartSpec {
    token: usize,
}

impl NodeSpec for RestartSpec {
    const TYPE_NAME: &'static str = "restart_spec";
    type Node = RebuiltNode;

    fn build(self, name: &str, ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        assert_eq!(self.token, 41);
        assert_eq!(ctx.params["token"], 41);
        let _timeline = ctx.timeline("spec-rebuild");
        let generation = SPEC_BUILDS.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(RebuiltNode {
            name: name.into(),
            fail: generation == 1,
        })
    }
}

#[test]
fn register_spec_rebuilds_from_canonical_params_and_build_context() {
    SPEC_BUILDS.store(0, Ordering::SeqCst);
    let inst = Instance::new();
    register_spec::<RestartSpec>(&inst);
    let mut request = NodeRequest::new(
        RestartSpec::TYPE_NAME,
        "worker",
        serde_json::json!({"token": 41}),
    );
    request.restart = Some(RestartPolicy::RestartGroup);
    inst.create_node(request).unwrap();
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "worker").unwrap();

    inst.start_group("g").unwrap();
    let group = inst.group("g").unwrap();
    wait_until(&group, "register_spec replacement generation", |group| {
        group.generation() == 2 && group.state() == GroupState::Running
    });

    assert_eq!(SPEC_BUILDS.load(Ordering::SeqCst), 2);
    inst.stop_group("g").unwrap();
}

#[test]
fn failed_multi_node_rebuild_is_atomic_and_can_be_retried() {
    let inst = Instance::new();
    let a_calls = Arc::new(AtomicUsize::new(0));
    let b_calls = Arc::new(AtomicUsize::new(0));
    register_factory(&inst, "a", {
        let calls = a_calls.clone();
        move |name, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(RunningNode {
                name: name.to_string(),
            }))
        }
    });
    register_factory(&inst, "b", {
        let calls = b_calls.clone();
        move |name, _| {
            let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
            if call == 2 {
                return Err("planned second-generation construction failure".into());
            }
            Ok(Arc::new(RunningNode {
                name: name.to_string(),
            }))
        }
    });
    let mut a_request = NodeRequest::new("a", "a", serde_json::json!({}));
    a_request.restart = Some(RestartPolicy::RestartGroup);
    inst.create_node(a_request).unwrap();
    inst.create_node(NodeRequest::new("b", "b", serde_json::json!({})))
        .unwrap();
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "a").unwrap();
    inst.add_group_member("g", "b").unwrap();
    let old_a = inst.node("a").unwrap().node;
    let old_b = inst.node("b").unwrap().node;

    inst.start_group("g").unwrap();
    let group = inst.group("g").unwrap();
    let generation = group.generation();
    group.report_outcome(NodeOutcome::Failed {
        name: "a".into(),
        generation,
        err: NodeError::new("a", NodePhase::Process, "trigger rebuild"),
    });
    wait_for("the atomic failure to be recorded", || {
        group.status().last_error.as_deref()
            == Some("planned second-generation construction failure")
    });
    wait_until(&group, "automatic retry", |g| {
        g.generation() == generation + 1 && g.state() == GroupState::Running
    });

    assert!(!Arc::ptr_eq(&old_a, &inst.node("a").unwrap().node));
    assert!(!Arc::ptr_eq(&old_b, &inst.node("b").unwrap().node));
    assert_eq!(a_calls.load(Ordering::SeqCst), 3);
    assert_eq!(b_calls.load(Ordering::SeqCst), 3);
    inst.stop_group("g").unwrap();
}

struct BoundNode {
    name: String,
    source: bool,
    fail: bool,
    binds: Arc<AtomicUsize>,
    observed_edges: Arc<Mutex<Vec<Arc<dyn Edge>>>>,
    edge: OnceLock<Arc<dyn Edge>>,
}

impl Node for BoundNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn pads(&self) -> NodePads {
        let pad = PadDecl {
            name: if self.source { "out" } else { "in" }.into(),
            media: AvpMediaType::VIDEO,
        };
        if self.source {
            NodePads {
                sources: Vec::new(),
                sinks: vec![pad],
            }
        } else {
            NodePads {
                sources: vec![pad],
                sinks: Vec::new(),
            }
        }
    }

    fn bind_source(&self, _name: &str, edge: Arc<dyn Edge>) {
        self.binds.fetch_add(1, Ordering::SeqCst);
        self.observed_edges.lock().unwrap().push(edge.clone());
        assert!(self.edge.set(edge).is_ok());
    }

    fn bind_sink(&self, _name: &str, edge: Arc<dyn Edge>) {
        self.binds.fetch_add(1, Ordering::SeqCst);
        self.observed_edges.lock().unwrap().push(edge.clone());
        assert!(self.edge.set(edge).is_ok());
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        if self.fail {
            let name = self.name.clone();
            NodeBody::Blocking(Box::new(move || {
                Err(NodeError::new(
                    name.clone(),
                    NodePhase::Process,
                    "restart bound graph",
                ))
            }))
        } else {
            NodeBody::Blocking(Box::new(|| {
                std::thread::sleep(Duration::from_millis(1));
                Ok(Blocked::Again)
            }))
        }
    }
}

#[test]
fn reconstruction_rebinds_fresh_nodes_to_existing_buffered_edges() {
    let inst = Instance::new();
    let source_calls = Arc::new(AtomicUsize::new(0));
    let source_binds = Arc::new(AtomicUsize::new(0));
    let sink_binds = Arc::new(AtomicUsize::new(0));
    let source_edges = Arc::new(Mutex::new(Vec::new()));
    let sink_edges = Arc::new(Mutex::new(Vec::new()));
    register_factory(&inst, "bound_source", {
        let calls = source_calls.clone();
        let binds = source_binds.clone();
        let observed_edges = source_edges.clone();
        move |name, _| {
            let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(Arc::new(BoundNode {
                name: name.into(),
                source: true,
                fail: call == 1,
                binds: binds.clone(),
                observed_edges: observed_edges.clone(),
                edge: OnceLock::new(),
            }))
        }
    });
    register_factory(&inst, "bound_sink", {
        let binds = sink_binds.clone();
        let observed_edges = sink_edges.clone();
        move |name, _| {
            Ok(Arc::new(BoundNode {
                name: name.into(),
                source: false,
                fail: false,
                binds: binds.clone(),
                observed_edges: observed_edges.clone(),
                edge: OnceLock::new(),
            }))
        }
    });
    let mut source = NodeRequest::new("bound_source", "source", serde_json::json!({}));
    source.restart = Some(RestartPolicy::RestartGroup);
    inst.create_node(source).unwrap();
    inst.create_node(NodeRequest::new(
        "bound_sink",
        "sink",
        serde_json::json!({}),
    ))
    .unwrap();
    let edge = inst
        .connect_edge(
            "media",
            "source",
            "out",
            "sink",
            "in",
            EdgeKind::Buffered { capacity: 2 },
        )
        .unwrap()
        .edge;
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "source").unwrap();
    inst.add_group_member("g", "sink").unwrap();

    inst.start_group("g").unwrap();
    let group = inst.group("g").unwrap();
    wait_until(&group, "bound replacement generation", |group| {
        group.generation() == 2 && group.state() == GroupState::Running
    });

    assert_eq!(source_binds.load(Ordering::SeqCst), 2);
    assert_eq!(sink_binds.load(Ordering::SeqCst), 2);
    assert!(
        source_edges
            .lock()
            .unwrap()
            .iter()
            .all(|bound| !Arc::ptr_eq(bound, &edge)),
        "producer bindings must be generation-fenced writer views"
    );
    assert!(
        sink_edges
            .lock()
            .unwrap()
            .iter()
            .all(|bound| Arc::ptr_eq(bound, &edge))
    );
    assert!(Arc::ptr_eq(&edge, &inst.edge_link("media").unwrap().edge));
    inst.stop_group("g").unwrap();
}

#[test]
fn idle_binding_after_stop_is_rebound_to_the_next_fresh_generation() {
    let inst = Instance::new();
    let source_edges = Arc::new(Mutex::new(Vec::new()));
    let sink_edges = Arc::new(Mutex::new(Vec::new()));
    register_factory(&inst, "idle_bound_source", {
        let observed_edges = source_edges.clone();
        move |name, _| {
            Ok(Arc::new(BoundNode {
                name: name.into(),
                source: true,
                fail: false,
                binds: Arc::new(AtomicUsize::new(0)),
                observed_edges: observed_edges.clone(),
                edge: OnceLock::new(),
            }))
        }
    });
    register_factory(&inst, "idle_bound_sink", {
        let observed_edges = sink_edges.clone();
        move |name, _| {
            Ok(Arc::new(BoundNode {
                name: name.into(),
                source: false,
                fail: false,
                binds: Arc::new(AtomicUsize::new(0)),
                observed_edges: observed_edges.clone(),
                edge: OnceLock::new(),
            }))
        }
    });
    inst.create_node(NodeRequest::new(
        "idle_bound_source",
        "source",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_node(NodeRequest::new(
        "idle_bound_sink",
        "sink",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "source").unwrap();
    inst.add_group_member("g", "sink").unwrap();
    inst.start_group("g").unwrap();
    inst.stop_group("g").unwrap();

    inst.bind_edge("source", "out", avplumber_f7k::PadDirection::Output, "late")
        .unwrap();
    inst.bind_edge("sink", "in", avplumber_f7k::PadDirection::Input, "late")
        .unwrap();
    let generation_1 = source_edges.lock().unwrap()[0].clone();

    inst.start_group("g").unwrap();
    let generation_2 = source_edges.lock().unwrap()[1].clone();
    assert_eq!(generation_1.push(any_buffer()), avplumber_f7k::Push::Closed);
    assert_eq!(generation_2.writer_generation(), 2);
    assert_eq!(sink_edges.lock().unwrap().len(), 2);
    inst.stop_group("g").unwrap();
}

struct PausingSource {
    name: String,
    publish_barrier: Option<Arc<Barrier>>,
}

impl Node for PausingSource {
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

    fn bind_sink(&self, _name: &str, _edge: Arc<dyn Edge>) {
        if let Some(barrier) = &self.publish_barrier {
            barrier.wait();
            barrier.wait();
        }
    }
}

struct RebindableSink {
    name: String,
}

impl Node for RebindableSink {
    fn name(&self) -> &str {
        &self.name
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
}

#[test]
fn reconstruction_aborts_if_edge_binding_changes_before_publish() {
    let inst = Arc::new(Instance::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(2));
    register_factory(&inst, "pausing_source", {
        let calls = calls.clone();
        let barrier = barrier.clone();
        move |name, _| {
            let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(Arc::new(PausingSource {
                name: name.into(),
                publish_barrier: (call == 2).then(|| barrier.clone()),
            }))
        }
    });
    register_factory(&inst, "rebindable_sink", |name, _| {
        Ok(Arc::new(RebindableSink { name: name.into() }))
    });
    let old = inst
        .create_node(NodeRequest::new(
            "pausing_source",
            "source",
            serde_json::json!({}),
        ))
        .unwrap()
        .node;
    inst.create_node(NodeRequest::new(
        "rebindable_sink",
        "sink",
        serde_json::json!({}),
    ))
    .unwrap();
    inst.connect_edge(
        "media",
        "source",
        "out",
        "sink",
        "in",
        EdgeKind::Buffered { capacity: 2 },
    )
    .unwrap();
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "source").unwrap();

    let rebuilding = {
        let inst = inst.clone();
        std::thread::spawn(move || inst.reconstruct_group("g"))
    };
    barrier.wait();
    inst.destroy_edge("media").unwrap();
    inst.connect_edge(
        "media",
        "source",
        "out",
        "sink",
        "in",
        EdgeKind::Buffered { capacity: 2 },
    )
    .unwrap();
    barrier.wait();

    let error = rebuilding.join().unwrap().unwrap_err();
    assert!(
        error.to_string().contains("edge bindings changed"),
        "{error}"
    );
    assert!(Arc::ptr_eq(&old, &inst.node("source").unwrap().node));
}

struct GatedNode {
    name: String,
    fail: bool,
    drops: Option<Arc<AtomicUsize>>,
}

impl Node for GatedNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        if self.fail {
            let name = self.name.clone();
            NodeBody::Blocking(Box::new(move || {
                Err(NodeError::new(
                    name.clone(),
                    NodePhase::Process,
                    "first generation fails",
                ))
            }))
        } else {
            NodeBody::Blocking(Box::new(|| {
                std::thread::sleep(Duration::from_millis(1));
                Ok(Blocked::Again)
            }))
        }
    }
}

impl Drop for GatedNode {
    fn drop(&mut self) {
        if let Some(drops) = &self.drops {
            drops.fetch_add(1, Ordering::SeqCst);
        }
    }
}

struct GatedRebuild {
    /// Set once the reconstruction worker is inside the replacement factory,
    /// i.e. after the transaction started but before it can publish.
    entered: Arc<AtomicBool>,
    resume: Arc<AtomicBool>,
    /// Counts destroyed replacements; the first generation never counts.
    drops: Arc<AtomicUsize>,
}

/// A one-member group whose first generation faults and whose replacement
/// factory parks until the test releases it.
fn gated_rebuild() -> (Arc<Instance>, Arc<dyn Node>, GatedRebuild) {
    let inst = Arc::new(Instance::new());
    let gate = GatedRebuild {
        entered: Arc::new(AtomicBool::new(false)),
        resume: Arc::new(AtomicBool::new(false)),
        drops: Arc::new(AtomicUsize::new(0)),
    };
    let calls = Arc::new(AtomicUsize::new(0));
    register_factory(&inst, "gated", {
        let entered = gate.entered.clone();
        let resume = gate.resume.clone();
        let drops = gate.drops.clone();
        move |name, _| {
            let generation = calls.fetch_add(1, Ordering::SeqCst) + 1;
            if generation == 2 {
                entered.store(true, Ordering::SeqCst);
                while !resume.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            Ok(Arc::new(GatedNode {
                name: name.into(),
                fail: generation == 1,
                drops: (generation > 1).then(|| drops.clone()),
            }))
        }
    });
    let mut request = NodeRequest::new("gated", "worker", serde_json::json!({}));
    request.restart = Some(RestartPolicy::RestartGroup);
    let first = inst.create_node(request).unwrap().node;
    inst.create_group("g").unwrap();
    inst.add_group_member("g", "worker").unwrap();
    inst.start_group("g").unwrap();
    wait_for("the reconstruction worker to reach the factory", || {
        gate.entered.load(Ordering::SeqCst)
    });
    (inst, first, gate)
}

#[test]
fn stop_cancels_an_in_flight_reconstruction_before_it_publishes() {
    let (inst, first, gate) = gated_rebuild();
    inst.stop_group("g").unwrap();
    gate.resume.store(true, Ordering::SeqCst);

    wait_for("the cancelled replacement to be destroyed", || {
        gate.drops.load(Ordering::SeqCst) == 1
    });
    assert!(
        Arc::ptr_eq(&first, &inst.node("worker").unwrap().node),
        "a cancelled reconstruction must not publish its replacement"
    );
    let group = inst.group("g").unwrap();
    assert_eq!(group.generation(), 1);
    assert_eq!(group.state(), GroupState::Idle);
}

#[test]
fn destroying_a_node_after_stop_waits_out_a_cancelled_reconstruction() {
    let (inst, _first, gate) = gated_rebuild();
    inst.stop_group("g").unwrap();

    let destroying = {
        let inst = inst.clone();
        std::thread::spawn(move || inst.destroy_node("worker"))
    };
    std::thread::sleep(Duration::from_millis(50));
    assert!(
        !destroying.is_finished(),
        "destroy must not free node state a cancelled rebuild can still touch"
    );

    gate.resume.store(true, Ordering::SeqCst);
    destroying.join().unwrap().unwrap();
    assert!(inst.node("worker").is_none());
    assert_eq!(gate.drops.load(Ordering::SeqCst), 1);
}
