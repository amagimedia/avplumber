//! Supervisor substrate. A `Group` is a supervisor unit: ordered start/stop.

pub mod topo;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::exec::{
    AsyncExecutor, BlockingExecutor, ExecCtxId, Executor, Generation, NodeOutcome, OutcomeReporter,
};
use crate::factory::RestartPolicy;
use crate::graph::{EdgeEvent, EdgeRestart, Graph, Node, generation_writer};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupState {
    Idle,
    Starting,
    Running,
    Quiescing,
    Backoff,
    Stopping,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct GroupStatus {
    pub state: GroupState,
    pub generation: Generation,
    pub restart_count: u64,
    pub last_error: Option<String>,
    pub last_outcome: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SupervisorAction {
    Panic { group: String, message: String },
    Exit { group: String, message: String },
}

pub type SupervisorActionHook = Arc<dyn Fn(SupervisorAction) + Send + Sync>;

#[derive(Clone, Debug)]
pub struct RestartRequest {
    pub failed_generation: Generation,
    pub next_generation: Generation,
    pub trigger: Option<NodeOutcome>,
    pub manual: bool,
}

impl RestartRequest {
    /// Mints the proof that the members were rebuilt for `next_generation`.
    /// The manager still checks the graph, so minting it without replacing the
    /// nodes does not buy a restart.
    pub fn reconstructed(&self) -> Reconstruction {
        Reconstruction {
            generation: self.next_generation,
        }
    }
}

/// Evidence that a [`RestartHook`] rebuilt the group's members. Only
/// [`RestartRequest::reconstructed`] mints one, which ties it to the generation
/// the manager is about to start.
#[derive(Clone, Debug)]
pub struct Reconstruction {
    generation: Generation,
}

/// Rebuilds a group's members after a fault, between the failed generation's
/// teardown and the next generation's start.
///
/// The hook must install fresh `Node` instances in the graph for every member:
/// a node body is taken once per run, so restarting the surviving instances
/// would re-run consumed bodies. The manager verifies this against the node
/// identities it started with and refuses the restart if any member survived,
/// so a hook that cannot rebuild should return `Err`; the manager remains in
/// `Backoff` and retries the complete reconstruction transaction.
///
/// The hook runs on a dedicated reconstruction worker thread so the manager
/// stays responsive while a factory is slow. A concurrent `stop` therefore does
/// not wait for the hook: it bumps the cancellation epoch
/// ([`Group::reconstruction_epoch`]), and the hook must check that the epoch it
/// started with is still current before it makes any replacement visible.
/// The hook must not join the group's workers (`Group::drop`, or a `destroy`
/// that waits them out), which would deadlock on itself.
pub type RestartHook = Arc<dyn Fn(RestartRequest) -> Result<Reconstruction, String> + Send + Sync>;

pub struct Group {
    name: String,
    config: Arc<GroupConfig>,
    state: Arc<Mutex<GroupState>>,
    generation: Arc<AtomicU64>,
    restart_hook: Arc<Mutex<Option<RestartHook>>>,
    manager_tx: mpsc::Sender<ManagerCommand>,
    manager_thread: Mutex<Option<JoinHandle<()>>>,
    reconstruction_workers: Arc<Mutex<Vec<JoinHandle<()>>>>,
    active_reconstructions: Arc<AtomicUsize>,
    metrics: Arc<Mutex<GroupMetrics>>,
    action_hook: Arc<Mutex<Option<SupervisorActionHook>>>,
    /// Bumped by every teardown. A reconstruction that started under an older
    /// epoch has been cancelled and must not publish; see [`RestartHook`].
    cancel_epoch: Arc<AtomicU64>,
}

struct GroupConfig {
    members: Mutex<Vec<String>>,
    restart: Mutex<HashMap<String, NodeRestartPolicy>>,
    graph: Arc<Mutex<Graph>>,
    assignment: Mutex<HashMap<String, ExecCtxId>>,
    service_group: Mutex<HashMap<String, String>>,
}

#[derive(Clone, Copy)]
struct NodeRestartPolicy {
    auto_restart: RestartPolicy,
    on_error: Option<RestartPolicy>,
}

#[derive(Default)]
struct GroupMetrics {
    restart_count: u64,
    last_error: Option<String>,
    last_outcome: Option<String>,
}

enum ManagerCommand {
    Start(mpsc::Sender<Result<(), String>>),
    Stop(mpsc::Sender<()>),
    Restart(mpsc::Sender<Result<(), String>>),
    Outcome(NodeOutcome),
    ReconstructionFinished {
        token: u64,
        next_generation: Generation,
        result: Result<Reconstruction, String>,
    },
    Shutdown(mpsc::Sender<()>),
}

struct ManagerRuntime {
    executors: HashMap<ExecCtxId, Arc<dyn Executor>>,
    executor_order: Vec<ExecCtxId>,
    generation: Generation,
    seen_outcomes: HashSet<String>,
    /// Nodes the current generation took bodies from. Kept past teardown so a
    /// restart can tell a rebuilt member from a surviving one.
    node_identities: Vec<(String, Arc<dyn Node>)>,
}

impl ManagerRuntime {
    fn new() -> Self {
        Self {
            executors: HashMap::new(),
            executor_order: Vec::new(),
            generation: 0,
            seen_outcomes: HashSet::new(),
            node_identities: Vec::new(),
        }
    }
}

struct PendingRestart {
    deadline: Instant,
    request: RestartRequest,
}

impl Group {
    pub fn new(name: String, graph: Arc<Mutex<Graph>>) -> Self {
        let config = Arc::new(GroupConfig {
            members: Mutex::new(Vec::new()),
            restart: Mutex::new(HashMap::new()),
            graph,
            assignment: Mutex::new(HashMap::new()),
            service_group: Mutex::new(HashMap::new()),
        });
        let state = Arc::new(Mutex::new(GroupState::Idle));
        let generation = Arc::new(AtomicU64::new(0));
        let restart_hook = Arc::new(Mutex::new(None));
        let reconstruction_workers = Arc::new(Mutex::new(Vec::new()));
        let active_reconstructions = Arc::new(AtomicUsize::new(0));
        let metrics = Arc::new(Mutex::new(GroupMetrics::default()));
        let action_hook = Arc::new(Mutex::new(None));
        let (manager_tx, manager_rx) = mpsc::channel();
        let manager_thread = std::thread::Builder::new()
            .name(format!("avp-group-{name}"))
            .spawn({
                let name = name.clone();
                let config = config.clone();
                let state = state.clone();
                let generation = generation.clone();
                let restart_hook = restart_hook.clone();
                let reconstruction_workers = reconstruction_workers.clone();
                let active_reconstructions = active_reconstructions.clone();
                let metrics = metrics.clone();
                let action_hook = action_hook.clone();
                let outcome_tx = manager_tx.clone();
                move || {
                    manager_loop(
                        &name,
                        config,
                        state,
                        generation,
                        restart_hook,
                        reconstruction_workers,
                        active_reconstructions,
                        metrics,
                        action_hook,
                        outcome_tx,
                        manager_rx,
                    )
                }
            })
            .expect("group manager thread");
        Self {
            name,
            config,
            state,
            generation,
            restart_hook,
            manager_tx,
            manager_thread: Mutex::new(Some(manager_thread)),
            reconstruction_workers,
            active_reconstructions,
            metrics,
            action_hook,
            cancel_epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn add(&self, node_name: &str, ctx: ExecCtxId) {
        self.add_with_policy(node_name, ctx, RestartPolicy::Off);
    }

    pub fn add_with_policy(&self, node_name: &str, ctx: ExecCtxId, auto_restart: RestartPolicy) {
        self.add_full(node_name, ctx, auto_restart, None, None);
    }

    pub fn add_full(
        &self,
        node_name: &str,
        ctx: ExecCtxId,
        auto_restart: RestartPolicy,
        on_error: Option<RestartPolicy>,
        service_hint: Option<String>,
    ) {
        if self.state() != GroupState::Idle || self.has_active_reconstruction() {
            log::warn!("group {}: add {} ignored (not idle)", self.name, node_name);
            return;
        }
        let mut members = self.config.members.lock().unwrap();
        if !members.iter().any(|n| n == node_name) {
            members.push(node_name.to_string());
        }
        drop(members);
        self.config
            .assignment
            .lock()
            .unwrap()
            .insert(node_name.to_string(), ctx);
        self.config.restart.lock().unwrap().insert(
            node_name.to_string(),
            NodeRestartPolicy {
                auto_restart,
                on_error,
            },
        );
        if let Some(h) = service_hint {
            self.config
                .service_group
                .lock()
                .unwrap()
                .insert(node_name.to_string(), h);
        }
    }

    pub fn restart_policy(&self, node_name: &str) -> RestartPolicy {
        self.config
            .restart
            .lock()
            .unwrap()
            .get(node_name)
            .map(|policy| policy.auto_restart)
            .unwrap_or(RestartPolicy::Off)
    }

    pub fn wants_auto_restart(&self, node_name: &str) -> bool {
        !matches!(self.restart_policy(node_name), RestartPolicy::Off)
    }

    pub fn remove(&self, node_name: &str) {
        if self.state() != GroupState::Idle || self.has_active_reconstruction() {
            log::warn!(
                "group {}: remove {} ignored (not idle)",
                self.name,
                node_name
            );
            return;
        }
        self.config
            .members
            .lock()
            .unwrap()
            .retain(|n| n != node_name);
        self.config.assignment.lock().unwrap().remove(node_name);
        self.config.restart.lock().unwrap().remove(node_name);
        self.config.service_group.lock().unwrap().remove(node_name);
    }

    pub fn members(&self) -> Vec<String> {
        self.config.members.lock().unwrap().clone()
    }

    pub fn state(&self) -> GroupState {
        *self.state.lock().unwrap()
    }

    pub fn generation(&self) -> Generation {
        self.generation.load(Ordering::Acquire)
    }

    pub fn status(&self) -> GroupStatus {
        let metrics = self.metrics.lock().unwrap();
        GroupStatus {
            state: self.state(),
            generation: self.generation(),
            restart_count: metrics.restart_count,
            last_error: metrics.last_error.clone(),
            last_outcome: metrics.last_outcome.clone(),
        }
    }

    pub fn set_restart_hook(&self, hook: RestartHook) {
        *self.restart_hook.lock().unwrap() = Some(hook);
    }

    pub fn set_action_hook(&self, hook: SupervisorActionHook) {
        *self.action_hook.lock().unwrap() = Some(hook);
    }

    pub(crate) fn has_active_reconstruction(&self) -> bool {
        self.active_reconstructions.load(Ordering::Acquire) != 0
    }

    /// Token a reconstruction transaction must recheck right before it
    /// publishes: a different value means a teardown cancelled it meanwhile.
    pub(crate) fn reconstruction_epoch(&self) -> u64 {
        self.cancel_epoch.load(Ordering::Acquire)
    }

    /// Waits for cancelled workers to unwind, so their abandoned replacements
    /// (and any C destroy callback) are gone before the caller frees anything
    /// they still reference. Called from a worker it would join, it is a no-op.
    pub(crate) fn join_reconstruction_workers(&self) {
        let current = std::thread::current().id();
        // Joining under the worker list lock would block the manager thread
        // when it queues the next reconstruction.
        let workers = self
            .reconstruction_workers
            .lock()
            .unwrap()
            .drain(..)
            .collect::<Vec<_>>();
        for worker in workers {
            if worker.thread().id() != current {
                let _ = worker.join();
            }
        }
    }

    pub fn report_outcome(&self, outcome: NodeOutcome) {
        let _ = self.manager_tx.send(ManagerCommand::Outcome(outcome));
    }

    /// Starts the next generation, or returns Ok if the group already runs.
    /// A failed start leaves the group `Failed`; `stop` clears it back to
    /// `Idle` before another attempt.
    pub fn start(&self) -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        self.manager_tx
            .send(ManagerCommand::Start(tx))
            .map_err(|_| format!("group {} manager stopped", self.name))?;
        rx.recv()
            .unwrap_or_else(|_| Err(format!("group {} manager stopped", self.name)))
    }

    /// Tears the group down sink-first and returns it to `Idle`, cancelling a
    /// pending restart. Takes precedence over any fault already queued.
    ///
    /// Does not wait for a reconstruction already running on a worker thread;
    /// bumping the epoch first guarantees that such a worker can no longer
    /// publish, whether or not the manager has processed the stop yet.
    pub fn stop(&self) {
        self.cancel_epoch.fetch_add(1, Ordering::AcqRel);
        let (tx, rx) = mpsc::channel();
        if self.manager_tx.send(ManagerCommand::Stop(tx)).is_ok() {
            let _ = rx.recv();
        }
    }

    pub fn restart(&self) -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        self.manager_tx
            .send(ManagerCommand::Restart(tx))
            .map_err(|_| format!("group {} manager stopped", self.name))?;
        rx.recv()
            .unwrap_or_else(|_| Err(format!("group {} manager stopped", self.name)))
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        self.cancel_epoch.fetch_add(1, Ordering::AcqRel);
        let (tx, rx) = mpsc::channel();
        if self.manager_tx.send(ManagerCommand::Shutdown(tx)).is_ok() {
            let _ = rx.recv();
        }
        if let Some(thread) = self.manager_thread.lock().unwrap().take() {
            let _ = thread.join();
        }
        self.join_reconstruction_workers();
    }
}

fn set_state(state: &Mutex<GroupState>, value: GroupState) {
    *state.lock().unwrap() = value;
}

fn manager_loop(
    group_name: &str,
    config: Arc<GroupConfig>,
    state: Arc<Mutex<GroupState>>,
    generation: Arc<AtomicU64>,
    restart_hook: Arc<Mutex<Option<RestartHook>>>,
    reconstruction_workers: Arc<Mutex<Vec<JoinHandle<()>>>>,
    active_reconstructions: Arc<AtomicUsize>,
    metrics: Arc<Mutex<GroupMetrics>>,
    action_hook: Arc<Mutex<Option<SupervisorActionHook>>>,
    outcome_tx: mpsc::Sender<ManagerCommand>,
    rx: mpsc::Receiver<ManagerCommand>,
) {
    const RESTART_BACKOFF: Duration = Duration::from_secs(1);

    let mut runtime = ManagerRuntime::new();
    let mut pending: Option<PendingRestart> = None;
    let mut reconstruction_token = 0_u64;
    let mut active_reconstruction: Option<(u64, RestartRequest)> = None;
    loop {
        let command = if let Some(restart) = &pending {
            match rx.recv_timeout(restart.deadline.saturating_duration_since(Instant::now())) {
                Ok(command) => command,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let restart = pending.take().unwrap();
                    let next_generation = restart.request.next_generation;
                    let hook = restart_hook.lock().unwrap().clone();
                    let Some(hook) = hook else {
                        log::error!(
                            "group {group_name}: restart failed: no restart transaction hook installed"
                        );
                        set_state(&state, GroupState::Failed);
                        continue;
                    };
                    reconstruction_token = reconstruction_token.wrapping_add(1);
                    let token = reconstruction_token;
                    active_reconstruction = Some((token, restart.request.clone()));
                    let tx = outcome_tx.clone();
                    active_reconstructions.fetch_add(1, Ordering::AcqRel);
                    let active = active_reconstructions.clone();
                    let worker = std::thread::Builder::new()
                        .name(format!("avp-reconstruct-{group_name}-{next_generation}"))
                        .spawn(move || {
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    hook(restart.request)
                                }))
                                .unwrap_or_else(|_| {
                                    Err("restart transaction hook panicked".to_string())
                                });
                            let _ = tx.send(ManagerCommand::ReconstructionFinished {
                                token,
                                next_generation,
                                result,
                            });
                            active.fetch_sub(1, Ordering::AcqRel);
                        })
                        .expect("reconstruction worker thread");
                    reconstruction_workers.lock().unwrap().push(worker);
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match rx.recv() {
                Ok(command) => command,
                Err(_) => break,
            }
        };

        match command {
            ManagerCommand::Start(reply) => {
                let current = *state.lock().unwrap();
                let result = if current == GroupState::Running {
                    Ok(())
                } else if active_reconstructions.load(Ordering::Acquire) != 0 {
                    Err(format!(
                        "group {group_name} reconstruction is still in progress"
                    ))
                } else if current != GroupState::Idle {
                    Err(format!("group {group_name} cannot start from {current:?}"))
                } else {
                    start_generation(
                        group_name,
                        &config,
                        &state,
                        &generation,
                        &outcome_tx,
                        &mut runtime,
                    )
                };
                let _ = reply.send(result);
            }
            ManagerCommand::Stop(reply) => {
                pending = None;
                active_reconstruction = None;
                set_state(&state, GroupState::Stopping);
                stop_runtime(&mut runtime);
                runtime.node_identities.clear();
                set_state(&state, GroupState::Idle);
                let _ = reply.send(());
            }
            ManagerCommand::Restart(reply) => {
                let current = *state.lock().unwrap();
                let result = if let Some(restart) = pending.as_mut() {
                    restart.deadline = Instant::now();
                    restart.request.manual = true;
                    Ok(())
                } else if active_reconstruction.is_some() {
                    Ok(())
                } else if !matches!(current, GroupState::Running | GroupState::Failed) {
                    Err(format!(
                        "group {group_name} cannot restart from {current:?}"
                    ))
                } else if restart_hook.lock().unwrap().is_none() {
                    Err(format!(
                        "group {group_name} has no restart transaction hook"
                    ))
                } else {
                    let next_generation = runtime.generation.saturating_add(1);
                    set_state(&state, GroupState::Quiescing);
                    prepare_restart_edges(&config, runtime.generation, next_generation);
                    stop_runtime(&mut runtime);
                    pending = Some(PendingRestart {
                        deadline: Instant::now(),
                        request: RestartRequest {
                            failed_generation: runtime.generation,
                            next_generation,
                            trigger: None,
                            manual: true,
                        },
                    });
                    set_state(&state, GroupState::Backoff);
                    Ok(())
                };
                let _ = reply.send(result);
            }
            ManagerCommand::Outcome(outcome) => {
                if pending.is_some()
                    || active_reconstruction.is_some()
                    || *state.lock().unwrap() != GroupState::Running
                    || outcome.generation() != runtime.generation
                {
                    continue;
                }
                if matches!(outcome, NodeOutcome::Cancelled { .. }) {
                    continue;
                }
                let outcome_kind = if outcome.is_fault() { "error" } else { "clean" };
                if !runtime
                    .seen_outcomes
                    .insert(format!("{outcome_kind}:{}", outcome.name()))
                {
                    continue;
                }
                {
                    let mut metrics = metrics.lock().unwrap();
                    metrics.last_outcome = Some(outcome_summary(&outcome));
                    metrics.last_error = outcome_error(&outcome);
                }
                let policy = config
                    .restart
                    .lock()
                    .unwrap()
                    .get(outcome.name())
                    .copied()
                    .unwrap_or(NodeRestartPolicy {
                        auto_restart: RestartPolicy::Off,
                        on_error: None,
                    });
                let action = if outcome.is_fault() {
                    policy.on_error.unwrap_or(policy.auto_restart)
                } else {
                    policy.auto_restart
                };
                if action == RestartPolicy::Off && !outcome.is_fault() {
                    propagate_natural_eof(&config, outcome.name());
                    continue;
                }
                set_state(&state, GroupState::Quiescing);
                let next_generation = runtime.generation.saturating_add(1);
                match action {
                    RestartPolicy::RestartGroup if restart_hook.lock().unwrap().is_some() => {
                        prepare_restart_edges(&config, runtime.generation, next_generation);
                        stop_runtime(&mut runtime);
                        pending = Some(PendingRestart {
                            deadline: Instant::now() + RESTART_BACKOFF,
                            request: RestartRequest {
                                failed_generation: outcome.generation(),
                                next_generation,
                                trigger: Some(outcome),
                                manual: false,
                            },
                        });
                        set_state(&state, GroupState::Backoff);
                    }
                    RestartPolicy::Panic | RestartPolicy::Exit => {
                        stop_runtime(&mut runtime);
                        let message = metrics
                            .lock()
                            .unwrap()
                            .last_outcome
                            .clone()
                            .unwrap_or_else(|| "unknown outcome".into());
                        let hook = action_hook.lock().unwrap().clone();
                        if let Some(hook) = hook {
                            let supervisor_action = if action == RestartPolicy::Panic {
                                SupervisorAction::Panic {
                                    group: group_name.to_string(),
                                    message,
                                }
                            } else {
                                SupervisorAction::Exit {
                                    group: group_name.to_string(),
                                    message,
                                }
                            };
                            set_state(&state, GroupState::Failed);
                            let thread_name = format!("avp-action-{group_name}");
                            if let Err(error) = std::thread::Builder::new()
                                .name(thread_name)
                                .spawn(move || hook(supervisor_action))
                            {
                                log::error!(
                                    "group {group_name}: failed to dispatch supervisor action: {error}"
                                );
                            }
                        } else {
                            set_state(&state, GroupState::Failed);
                        }
                    }
                    RestartPolicy::Off | RestartPolicy::RestartGroup => {
                        stop_runtime(&mut runtime);
                        set_state(&state, GroupState::Failed);
                    }
                }
            }
            ManagerCommand::ReconstructionFinished {
                token,
                next_generation,
                result,
            } => {
                let Some((active_token, retry_request)) = active_reconstruction.as_ref() else {
                    continue;
                };
                if *active_token != token || retry_request.next_generation != next_generation {
                    continue;
                }
                let retry_request = retry_request.clone();
                active_reconstruction = None;
                let result = result
                    .and_then(|proof| {
                        check_reconstruction(&config, &runtime, &proof, next_generation)
                    })
                    .and_then(|()| {
                        start_generation(
                            group_name,
                            &config,
                            &state,
                            &generation,
                            &outcome_tx,
                            &mut runtime,
                        )
                    });
                if let Err(err) = result {
                    log::error!("group {group_name}: restart failed: {err}");
                    metrics.lock().unwrap().last_error = Some(err);
                    pending = Some(PendingRestart {
                        deadline: Instant::now() + RESTART_BACKOFF,
                        request: retry_request,
                    });
                    set_state(&state, GroupState::Backoff);
                } else {
                    let mut metrics = metrics.lock().unwrap();
                    metrics.restart_count += 1;
                    metrics.last_error = None;
                }
            }
            ManagerCommand::Shutdown(reply) => {
                set_state(&state, GroupState::Stopping);
                stop_runtime(&mut runtime);
                runtime.node_identities.clear();
                set_state(&state, GroupState::Idle);
                let _ = reply.send(());
                break;
            }
        }
    }
}

fn outcome_summary(outcome: &NodeOutcome) -> String {
    match outcome {
        NodeOutcome::Completed { name, generation } => {
            format!("completed:{name}:generation={generation}")
        }
        NodeOutcome::Failed {
            name,
            generation,
            err,
        } => format!("failed:{name}:generation={generation}:{}", err.message),
        NodeOutcome::Panicked {
            name,
            generation,
            message,
        } => format!("panicked:{name}:generation={generation}:{message}"),
        NodeOutcome::Cancelled { name, generation } => {
            format!("cancelled:{name}:generation={generation}")
        }
    }
}

fn outcome_error(outcome: &NodeOutcome) -> Option<String> {
    match outcome {
        NodeOutcome::Failed { err, .. } => Some(err.message.clone()),
        NodeOutcome::Panicked { message, .. } => Some(message.clone()),
        NodeOutcome::Completed { .. } | NodeOutcome::Cancelled { .. } => None,
    }
}

fn propagate_natural_eof(config: &GroupConfig, node_name: &str) {
    let graph = config.graph.lock().unwrap();
    if let Some(vertex) = graph.vertex(node_name) {
        for edge in vertex.sinks.values() {
            edge.push_event(EdgeEvent::Eof);
        }
    }
}

/// Verifies that the hook's claim matches the graph: every member the failed
/// generation ran must now be a different `Node` instance.
fn check_reconstruction(
    config: &GroupConfig,
    runtime: &ManagerRuntime,
    token: &Reconstruction,
    next_generation: Generation,
) -> Result<(), String> {
    if token.generation != next_generation {
        return Err(format!(
            "restart hook proved reconstruction for generation {} while generation {next_generation} is starting",
            token.generation
        ));
    }
    let graph = config.graph.lock().unwrap();
    for (name, previous) in &runtime.node_identities {
        if let Some(vertex) = graph.vertex(name)
            && Arc::ptr_eq(&vertex.node, previous)
        {
            return Err(format!(
                "restart hook claimed reconstruction but node `{name}` is still the instance whose body was already taken"
            ));
        }
    }
    Ok(())
}

fn start_generation(
    group_name: &str,
    config: &GroupConfig,
    state: &Mutex<GroupState>,
    generation_counter: &AtomicU64,
    outcome_tx: &mpsc::Sender<ManagerCommand>,
    runtime: &mut ManagerRuntime,
) -> Result<(), String> {
    set_state(state, GroupState::Starting);
    match build_generation(config, outcome_tx, runtime) {
        Ok(generation) => {
            generation_counter.store(generation, Ordering::Release);
            set_state(state, GroupState::Running);
            log::debug!("group {group_name}: started generation {generation}");
            Ok(())
        }
        Err(err) => {
            stop_runtime(runtime);
            set_state(state, GroupState::Failed);
            Err(err)
        }
    }
}

/// Builds and starts the next generation. On error the caller tears down
/// whatever was already started; nothing here leaves the group runnable.
fn build_generation(
    config: &GroupConfig,
    outcome_tx: &mpsc::Sender<ManagerCommand>,
    runtime: &mut ManagerRuntime,
) -> Result<Generation, String> {
    let next_generation = runtime.generation.saturating_add(1);
    let reporter: OutcomeReporter = Arc::new({
        let outcome_tx = outcome_tx.clone();
        move |outcome| {
            let _ = outcome_tx.send(ManagerCommand::Outcome(outcome));
        }
    });
    let members = config.members.lock().unwrap().clone();
    let assignment = config.assignment.lock().unwrap();
    let hints = config.service_group.lock().unwrap();
    let graph = config.graph.lock().unwrap();
    validate_direct_edges(&members, &assignment, graph.links())?;
    let order = topo::topo_sort(&members, graph.links())?;
    let mut executors: HashMap<ExecCtxId, Arc<dyn Executor>> = HashMap::new();
    let mut executor_order = Vec::new();
    let mut node_identities = Vec::new();
    for name in &order {
        let ctx = assignment
            .get(name)
            .ok_or_else(|| format!("node {name} has no exec ctx assigned"))?;
        if let Some(hint) = hints.get(name)
            && let ExecCtxId::EventLoop { name: event_loop } = ctx
            && event_loop != hint
        {
            log::info!(
                "node {name}: event_loop `{event_loop}` differs from service group `{hint}` (allowed)"
            );
        }
        let exec = executors.entry(ctx.clone()).or_insert_with(|| {
            executor_order.push(ctx.clone());
            match ctx {
                ExecCtxId::Blocking => Arc::new(BlockingExecutor::new()),
                ExecCtxId::EventLoop { .. } | ExecCtxId::TickSource { .. } => {
                    Arc::new(AsyncExecutor::new())
                }
            }
        });
        let vertex = graph
            .vertex(name)
            .ok_or_else(|| format!("node {name} not in graph"))?;
        vertex.node.set_generation(next_generation);
        exec.add_node(
            vertex.node.clone(),
            vertex.sources.values().cloned().collect(),
            vertex
                .sinks
                .values()
                .cloned()
                .map(|edge| generation_writer(edge, next_generation))
                .collect(),
        );
        node_identities.push((name.clone(), vertex.node.clone()));
    }
    drop(graph);
    drop(hints);
    drop(assignment);

    for ctx in &executor_order {
        executors
            .get(ctx)
            .unwrap()
            .configure_run(next_generation, reporter.clone());
    }
    // Keep every body that may be consumed by a failed start as reconstruction
    // proof. A successful generation replaces this history with its live set.
    remember_attempt_identities(runtime, &node_identities);
    for (started, ctx) in executor_order.iter().enumerate() {
        if let Err(err) = executors.get(ctx).unwrap().start() {
            // Hand the started prefix over so the caller tears it down
            // sink-first along with the rest of the failed attempt.
            runtime.executor_order = executor_order[..started].to_vec();
            runtime.executors = executors;
            return Err(err);
        }
    }
    runtime.executors = executors;
    runtime.executor_order = executor_order;
    runtime.generation = next_generation;
    runtime.seen_outcomes.clear();
    runtime.node_identities = node_identities;
    Ok(next_generation)
}

fn remember_attempt_identities(
    runtime: &mut ManagerRuntime,
    attempted: &[(String, Arc<dyn Node>)],
) {
    for (name, node) in attempted {
        if !runtime
            .node_identities
            .iter()
            .any(|(old_name, old)| old_name == name && Arc::ptr_eq(old, node))
        {
            runtime.node_identities.push((name.clone(), node.clone()));
        }
    }
}

/// Group membership and executor assignment are mutable until start, so Direct
/// co-location is enforced here rather than when the edge is connected.
fn validate_direct_edges(
    members: &[String],
    assignment: &HashMap<String, ExecCtxId>,
    links: &[crate::graph::EdgeLink],
) -> Result<(), String> {
    let members = members.iter().map(String::as_str).collect::<HashSet<_>>();
    for link in links.iter().filter(|link| link.edge.is_direct()) {
        let producer_inside = members.contains(link.producer.as_str());
        let consumer_inside = members.contains(link.consumer.as_str());
        if producer_inside != consumer_inside {
            return Err(format!(
                "Direct edge `{}` cannot cross a group boundary",
                link.name
            ));
        }
        if producer_inside && assignment.get(&link.producer) != assignment.get(&link.consumer) {
            return Err(format!(
                "Direct edge `{}` requires producer and consumer in the same ExecCtxId",
                link.name
            ));
        }
    }
    Ok(())
}

fn prepare_restart_edges(config: &GroupConfig, old: Generation, new: Generation) {
    let members = config.members.lock().unwrap().clone();
    let members = members.iter().map(String::as_str).collect::<HashSet<_>>();
    let graph = config.graph.lock().unwrap();
    for link in graph.links() {
        let producer_inside = members.contains(link.producer.as_str());
        let consumer_inside = members.contains(link.consumer.as_str());
        let kind = match (producer_inside, consumer_inside) {
            (true, true) => Some(EdgeRestart::Internal),
            (true, false) => Some(EdgeRestart::Egress),
            (false, true) => Some(EdgeRestart::Ingress),
            (false, false) => None,
        };
        if let Some(kind) = kind {
            link.edge.restart(old, new, kind);
        }
    }
}

fn stop_runtime(runtime: &mut ManagerRuntime) {
    for ctx in runtime.executor_order.iter().rev() {
        if let Some(exec) = runtime.executors.get(ctx) {
            exec.stop();
        }
    }
    for ctx in runtime.executor_order.iter().rev() {
        if let Some(exec) = runtime.executors.get(ctx) {
            exec.join();
        }
    }
    runtime.executors.clear();
    runtime.executor_order.clear();
}

#[cfg(test)]
mod identity_tests {
    use super::*;
    use crate::graph::Vertex;

    struct TestNode(&'static str);

    impl Node for TestNode {
        fn name(&self) -> &str {
            self.0
        }
    }

    #[test]
    fn failed_start_identity_rejects_noop_retry_reconstruction() {
        let previous: Arc<dyn Node> = Arc::new(TestNode("worker"));
        let attempted: Arc<dyn Node> = Arc::new(TestNode("worker"));
        let mut runtime = ManagerRuntime::new();
        runtime.node_identities.push(("worker".into(), previous));
        remember_attempt_identities(&mut runtime, &[("worker".into(), attempted.clone())]);

        let mut graph = Graph::new();
        graph
            .add_vertex(Vertex {
                name: "worker".into(),
                node: attempted,
                sources: HashMap::new(),
                sinks: HashMap::new(),
                source_media: HashMap::new(),
                sink_media: HashMap::new(),
            })
            .unwrap();
        let config = GroupConfig {
            members: Mutex::new(vec!["worker".into()]),
            restart: Mutex::new(HashMap::new()),
            graph: Arc::new(Mutex::new(graph)),
            assignment: Mutex::new(HashMap::new()),
            service_group: Mutex::new(HashMap::new()),
        };

        let error = check_reconstruction(&config, &runtime, &Reconstruction { generation: 2 }, 2)
            .unwrap_err();
        assert!(error.contains("body was already taken"));
    }
}
