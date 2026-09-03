//! AsyncExecutor — one current-thread tokio runtime per event-loop / tick-source.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

#[cfg(feature = "async")]
use std::future::Future;
#[cfg(feature = "async")]
use std::task::{Context, Poll as TaskPoll};
#[cfg(feature = "async")]
use std::time::Instant;

use crate::exec::{Executor, ExecutorState, Generation, OutcomeReporter};
use crate::graph::{Edge, Node, Wakeup};

#[cfg(feature = "async")]
use crate::exec::NodeOutcome;
#[cfg(feature = "async")]
use crate::graph::{NodeBody, NodePollContext, Tick};

#[cfg(feature = "async")]
const AGAIN_BUDGET: u32 = 32;

pub struct AsyncExecutor {
    inner: Mutex<Inner>,
    stop: Arc<AtomicBool>,
    tick: Arc<Wakeup>,
}

struct Slot {
    node: Arc<dyn Node>,
    sources: Vec<Arc<dyn Edge>>,
    #[allow(dead_code)]
    sinks: Vec<Arc<dyn Edge>>,
}

struct Inner {
    nodes: Vec<Slot>,
    started: bool,
    state: ExecutorState,
    thread: Option<JoinHandle<()>>,
    generation: Generation,
    reporter: OutcomeReporter,
}

impl AsyncExecutor {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                nodes: Vec::new(),
                started: false,
                state: ExecutorState::Created,
                thread: None,
                generation: 0,
                reporter: Arc::new(|outcome| log::debug!("unobserved node outcome: {outcome:?}")),
            }),
            stop: Arc::new(AtomicBool::new(false)),
            tick: Arc::new(Wakeup::new()),
        }
    }
}

impl Default for AsyncExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor for AsyncExecutor {
    fn add_node(
        &self,
        node: Arc<dyn Node>,
        sources: Vec<Arc<dyn Edge>>,
        sinks: Vec<Arc<dyn Edge>>,
    ) {
        let mut g = self.inner.lock().unwrap();
        if g.started {
            log::warn!("AsyncExecutor::add_node: live add not supported");
            return;
        }
        g.nodes.push(Slot {
            node,
            sources,
            sinks,
        });
    }

    fn remove_node(&self, name: &str) {
        self.inner
            .lock()
            .unwrap()
            .nodes
            .retain(|s| s.node.name() != name);
    }

    fn configure_run(&self, generation: Generation, reporter: OutcomeReporter) {
        let mut g = self.inner.lock().unwrap();
        if g.started {
            log::warn!("AsyncExecutor::configure_run: executor already started");
            return;
        }
        g.generation = generation;
        g.reporter = reporter;
    }

    fn start(&self) -> Result<(), String> {
        #[cfg(feature = "async")]
        {
            return start_tokio(self);
        }
        #[cfg(not(feature = "async"))]
        {
            let _ = self;
            Err("AsyncExecutor requires the `async` feature (tokio current-thread runtime)".into())
        }
    }

    fn stop(&self) {
        self.stop.store(true, Ordering::Release);
        self.tick.notify();
        let mut g = self.inner.lock().unwrap();
        g.state = ExecutorState::Stopping;
        for slot in &g.nodes {
            for src in &slot.sources {
                src.interrupt();
            }
        }
    }

    fn join(&self) {
        let handle = self.inner.lock().unwrap().thread.take();
        if let Some(h) = handle {
            let _ = h.join();
        }
        let mut g = self.inner.lock().unwrap();
        g.started = false;
        g.state = ExecutorState::Stopped;
    }

    fn tick(&self) {
        self.tick.notify();
    }

    fn state(&self) -> ExecutorState {
        self.inner.lock().unwrap().state
    }
}

#[cfg(feature = "async")]
fn start_tokio(exec: &AsyncExecutor) -> Result<(), String> {
    let mut g = exec.inner.lock().unwrap();
    if g.started {
        return Ok(());
    }
    g.started = true;
    g.state = ExecutorState::Running;
    exec.stop.store(false, Ordering::Release);
    let nodes: Vec<Slot> = g
        .nodes
        .iter()
        .map(|s| Slot {
            node: s.node.clone(),
            sources: s.sources.clone(),
            sinks: s.sinks.clone(),
        })
        .collect();
    let stop = exec.stop.clone();
    let tick = exec.tick.clone();
    let generation = g.generation;
    let reporter = g.reporter.clone();
    drop(g);

    let handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .expect("tokio current-thread runtime");
        let local = tokio::task::LocalSet::new();
        rt.block_on(local.run_until(async move {
            let mut handles = Vec::new();
            for slot in nodes {
                handles.push(tokio::task::spawn_local(run_slot_reported(
                    slot,
                    stop.clone(),
                    tick.clone(),
                    generation,
                    reporter.clone(),
                )));
            }
            for h in handles {
                if let Err(e) = h.await {
                    log::error!("async node worker panicked outside outcome capture: {e}");
                }
            }
        }));
    });
    exec.inner.lock().unwrap().thread = Some(handle);
    Ok(())
}

#[cfg(feature = "async")]
async fn run_slot_reported(
    slot: Slot,
    stop: Arc<AtomicBool>,
    tick: Arc<Wakeup>,
    generation: Generation,
    reporter: OutcomeReporter,
) {
    let name = slot.node.name().to_string();
    let node = slot.node.clone();
    let outcome = match CatchUnwind::new(run_slot(slot, stop, tick, generation)).await {
        Ok(outcome) => outcome,
        Err(payload) => {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| node.stop()));
            NodeOutcome::Panicked {
                name,
                generation,
                message: panic_message(payload),
            }
        }
    };
    reporter(outcome);
}

#[cfg(feature = "async")]
async fn run_slot(
    slot: Slot,
    stop: Arc<AtomicBool>,
    tick: Arc<Wakeup>,
    generation: Generation,
) -> NodeOutcome {
    slot.node.start();
    let name = slot.node.name().to_string();
    let body = slot.node.clone().take_body();
    let outcome = match body {
        NodeBody::Async(fut) => {
            tokio::select! {
                r = fut => match r {
                    Ok(()) => {
                        NodeOutcome::Completed { name: name.clone(), generation }
                    }
                    Err(err) => NodeOutcome::Failed {
                        name: name.clone(),
                        generation,
                        err,
                    },
                },
                _ = WaitWake { w: tick.clone() }.wait_until_stop(stop.clone()) => {
                    NodeOutcome::Cancelled { name: name.clone(), generation }
                }
            }
        }
        NodeBody::Poll(step) => run_poll(step, stop, tick, name.clone(), generation).await,
        NodeBody::Blocking(_) => NodeOutcome::Failed {
            name: name.clone(),
            generation,
            err: crate::graph::NodeError::new(
                name.clone(),
                crate::graph::NodePhase::Start,
                "async executor received a blocking body",
            ),
        },
    };
    slot.node.stop();
    outcome
}

#[cfg(feature = "async")]
async fn run_poll(
    mut step: Box<dyn FnMut(&mut NodePollContext) -> Result<Tick, crate::graph::NodeError> + Send>,
    stop: Arc<AtomicBool>,
    tick: Arc<Wakeup>,
    name: String,
    generation: Generation,
) -> NodeOutcome {
    let mut ctx = NodePollContext::new(stop.clone(), tick);
    let mut again = 0u32;
    loop {
        if stop.load(Ordering::Acquire) {
            return NodeOutcome::Cancelled { name, generation };
        }
        match step(&mut ctx) {
            Ok(Tick::Done) => {
                return NodeOutcome::Completed { name, generation };
            }
            Err(err) => {
                return NodeOutcome::Failed {
                    name,
                    generation,
                    err,
                };
            }
            Ok(Tick::Again) => {
                ctx.clear_park();
                again += 1;
                if again >= AGAIN_BUDGET {
                    again = 0;
                    tokio::task::yield_now().await;
                }
            }
            Ok(Tick::Idle) => {
                again = 0;
                IdlePark {
                    ctx: &mut ctx,
                    timer: None,
                }
                .await;
                ctx.clear_park();
            }
        }
    }
}

#[cfg(feature = "async")]
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(feature = "async")]
struct CatchUnwind<F> {
    future: std::pin::Pin<Box<F>>,
}

#[cfg(feature = "async")]
impl<F> CatchUnwind<F> {
    fn new(future: F) -> Self {
        Self {
            future: Box::pin(future),
        }
    }
}

#[cfg(feature = "async")]
impl<F: Future> Future for CatchUnwind<F> {
    type Output = Result<F::Output, Box<dyn std::any::Any + Send>>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> TaskPoll<Self::Output> {
        let this = self.get_mut();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            this.future.as_mut().poll(cx)
        })) {
            Ok(TaskPoll::Ready(value)) => TaskPoll::Ready(Ok(value)),
            Ok(TaskPoll::Pending) => TaskPoll::Pending,
            Err(payload) => TaskPoll::Ready(Err(payload)),
        }
    }
}

#[cfg(feature = "async")]
struct WaitWake {
    w: Arc<Wakeup>,
}

#[cfg(feature = "async")]
impl WaitWake {
    #[cfg(feature = "async")]
    async fn wait_until_stop(self, stop: Arc<AtomicBool>) {
        loop {
            if stop.load(Ordering::Acquire) {
                return;
            }
            self.await_once().await;
        }
    }

    async fn await_once(&self) {
        WaitOnce { w: self.w.clone() }.await
    }
}

#[cfg(feature = "async")]
struct WaitOnce {
    w: Arc<Wakeup>,
}

#[cfg(feature = "async")]
impl Future for WaitOnce {
    type Output = ();
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> TaskPoll<()> {
        if self.w.poll_notified(cx.waker()) {
            TaskPoll::Ready(())
        } else {
            TaskPoll::Pending
        }
    }
}

#[cfg(feature = "async")]
struct IdlePark<'a> {
    ctx: &'a mut NodePollContext,
    timer: Option<std::pin::Pin<Box<tokio::time::Sleep>>>,
}

#[cfg(feature = "async")]
impl Future for IdlePark<'_> {
    type Output = ();
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> TaskPoll<()> {
        let this = self.get_mut();
        if this.ctx.ready_now() {
            return TaskPoll::Ready(());
        }
        this.ctx.register_idle_waker(cx.waker().clone());
        if this.ctx.ready_now() {
            return TaskPoll::Ready(());
        }
        if this.timer.is_none()
            && let Some(deadline) = this.ctx.take_deadline()
        {
            if Instant::now() >= deadline {
                return TaskPoll::Ready(());
            }
            this.timer = Some(Box::pin(tokio::time::sleep_until(
                tokio::time::Instant::from_std(deadline),
            )));
        }
        if let Some(timer) = &mut this.timer
            && timer.as_mut().poll(cx).is_ready()
        {
            return TaskPoll::Ready(());
        }
        TaskPoll::Pending
    }
}
