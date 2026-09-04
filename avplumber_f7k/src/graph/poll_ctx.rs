//! Readiness context for `NodeBody::Poll`. Register waiters, then return `Tick::Idle`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Waker;
use std::time::Instant;

use crate::graph::edge::{Edge, Wakeup};

pub struct NodePollContext {
    cancel: Arc<AtomicBool>,
    tick: Arc<Wakeup>,
    self_wake: Arc<Wakeup>,
    deadline: Option<Instant>,
    readable: Vec<Arc<dyn Edge>>,
    writable: Vec<Arc<dyn Edge>>,
    waiting: bool,
}

impl NodePollContext {
    pub fn new(cancel: Arc<AtomicBool>, tick: Arc<Wakeup>) -> Self {
        Self {
            cancel,
            tick,
            self_wake: Arc::new(Wakeup::new()),
            deadline: None,
            readable: Vec::new(),
            writable: Vec::new(),
            waiting: false,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    pub fn wait_readable(&mut self, edge: Arc<dyn Edge>) {
        edge.arm_readable();
        self.readable.push(edge);
        self.waiting = true;
    }

    pub fn wait_writable(&mut self, edge: Arc<dyn Edge>) {
        edge.arm_writable();
        self.writable.push(edge);
        self.waiting = true;
    }

    pub fn wait_deadline(&mut self, when: Instant) {
        self.deadline = Some(when);
        self.waiting = true;
    }

    pub fn wait_tick(&mut self) {
        self.tick.arm();
        self.waiting = true;
    }

    pub fn wake(&mut self) {
        self.self_wake.notify();
        self.waiting = false;
    }

    #[allow(dead_code)]
    pub(crate) fn take_deadline(&mut self) -> Option<Instant> {
        self.deadline.take()
    }

    #[allow(dead_code)]
    pub(crate) fn needs_park(&self) -> bool {
        self.waiting
    }

    #[allow(dead_code)]
    pub(crate) fn ready_now(&self) -> bool {
        self.is_cancelled()
            || self
                .readable
                .iter()
                .any(|e| e.occupied() > 0 || e.is_closed())
            || self
                .writable
                .iter()
                .any(|e| !e.is_full() || e.is_closed() || e.has_hints())
    }

    #[allow(dead_code)]
    pub(crate) fn register_idle_waker(&self, waker: Waker) {
        self.tick.register_waker(waker.clone());
        self.self_wake.register_waker(waker.clone());
        for e in &self.readable {
            e.register_readable_waker(waker.clone());
        }
        for e in &self.writable {
            e.register_writable_waker(waker.clone());
        }
    }

    /// Drops waiters registered during the last poll. Executors call this
    /// between steps; a test that drives a poll body by hand does the same,
    /// otherwise a park from one step leaks into the next.
    pub fn clear_park(&mut self) {
        self.waiting = false;
        self.deadline = None;
        self.readable.clear();
        self.writable.clear();
    }

    #[allow(dead_code)]
    pub(crate) fn tick_wakeup(&self) -> Arc<Wakeup> {
        self.tick.clone()
    }
}
