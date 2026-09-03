//! BlockingExecutor. One OS thread per blocking body.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::exec::{Executor, ExecutorState, Generation, NodeOutcome, OutcomeReporter};
use crate::graph::{Blocked, Edge, Node, NodeBody};

pub struct BlockingExecutor {
    inner: Mutex<Inner>,
    stop: Arc<AtomicBool>,
}

#[derive(Clone)]
struct Slot {
    node: Arc<dyn Node>,
    sources: Vec<Arc<dyn Edge>>,
    sinks: Vec<Arc<dyn Edge>>,
}

struct Inner {
    nodes: Vec<Slot>,
    running: Vec<Slot>,
    handles: Vec<JoinHandle<()>>,
    started: bool,
    state: ExecutorState,
    generation: Generation,
    reporter: OutcomeReporter,
}

impl BlockingExecutor {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                nodes: Vec::new(),
                running: Vec::new(),
                handles: Vec::new(),
                started: false,
                state: ExecutorState::Created,
                generation: 0,
                reporter: Arc::new(|outcome| log::debug!("unobserved node outcome: {outcome:?}")),
            }),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Default for BlockingExecutor {
    fn default() -> Self {
        Self::new()
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

impl Executor for BlockingExecutor {
    fn add_node(
        &self,
        node: Arc<dyn Node>,
        sources: Vec<Arc<dyn Edge>>,
        sinks: Vec<Arc<dyn Edge>>,
    ) {
        let mut g = self.inner.lock().unwrap();
        if g.started {
            log::warn!("BlockingExecutor::add_node: live add not supported");
            return;
        }
        g.nodes.push(Slot {
            node,
            sources,
            sinks,
        });
    }

    fn remove_node(&self, name: &str) {
        let mut g = self.inner.lock().unwrap();
        g.nodes.retain(|s| s.node.name() != name);
    }

    fn configure_run(&self, generation: Generation, reporter: OutcomeReporter) {
        let mut g = self.inner.lock().unwrap();
        if g.started {
            log::warn!("BlockingExecutor::configure_run: executor already started");
            return;
        }
        g.generation = generation;
        g.reporter = reporter;
    }

    fn start(&self) -> Result<(), String> {
        let mut g = self.inner.lock().unwrap();
        if g.started {
            return Ok(());
        }
        g.started = true;
        g.state = ExecutorState::Running;
        self.stop.store(false, Ordering::Release);
        let nodes = g
            .nodes
            .iter()
            .map(|s| Slot {
                node: s.node.clone(),
                sources: s.sources.clone(),
                sinks: s.sinks.clone(),
            })
            .collect::<Vec<_>>();
        let generation = g.generation;
        let reporter = g.reporter.clone();
        g.running = nodes.clone();
        drop(g);

        for slot in nodes {
            let stop = self.stop.clone();
            let reporter = reporter.clone();
            let handle = thread::spawn(move || {
                let name = slot.node.name().to_string();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    slot.node.start();
                    let mut body = slot.node.clone().take_body();
                    let outcome = match &mut body {
                        NodeBody::Blocking(step) => loop {
                            if stop.load(Ordering::Acquire) {
                                break NodeOutcome::Cancelled {
                                    name: name.clone(),
                                    generation,
                                };
                            }
                            match step() {
                                Ok(Blocked::Done) => {
                                    break NodeOutcome::Completed {
                                        name: name.clone(),
                                        generation,
                                    };
                                }
                                Ok(Blocked::Again) => {}
                                Err(err) => {
                                    break NodeOutcome::Failed {
                                        name: name.clone(),
                                        generation,
                                        err,
                                    };
                                }
                            }
                        },
                        _ => NodeOutcome::Failed {
                            name: name.clone(),
                            generation,
                            err: crate::graph::NodeError::new(
                                name.clone(),
                                crate::graph::NodePhase::Start,
                                "blocking executor received a non-blocking body",
                            ),
                        },
                    };
                    slot.node.stop();
                    outcome
                }));
                let outcome = match result {
                    Ok(outcome) => outcome,
                    Err(payload) => {
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            slot.node.stop()
                        }));
                        NodeOutcome::Panicked {
                            name,
                            generation,
                            message: panic_message(payload),
                        }
                    }
                };
                reporter(outcome);
            });
            self.inner.lock().unwrap().handles.push(handle);
        }
        Ok(())
    }

    fn stop(&self) {
        self.stop.store(true, Ordering::Release);
        let mut g = self.inner.lock().unwrap();
        g.state = ExecutorState::Stopping;
        for slot in g.running.iter().chain(g.nodes.iter()) {
            for src in &slot.sources {
                src.interrupt();
            }
        }
    }

    fn join(&self) {
        let handles: Vec<JoinHandle<()>> = std::mem::take(&mut self.inner.lock().unwrap().handles);
        for h in handles {
            if h.join().is_err() {
                log::error!("blocking node worker panicked outside outcome capture");
            }
        }
        let mut g = self.inner.lock().unwrap();
        g.started = false;
        g.running.clear();
        g.state = ExecutorState::Stopped;
    }

    fn state(&self) -> ExecutorState {
        self.inner.lock().unwrap().state
    }
}
