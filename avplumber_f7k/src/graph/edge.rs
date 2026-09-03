//! Edge dataflow: owned `Media`, control events, and `Push`.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll as TaskPoll, Waker};
use std::time::Duration;

use crate::graph::media::Media;
use crate::graph::spec::Spec;

#[derive(Clone, Debug)]
pub enum EdgeEvent {
    Eof,
    FlushStart,
    FlushStop,
    Spec(Spec),
}

#[derive(Clone)]
pub enum EdgeItem {
    Buffer(Media),
    Event(EdgeEvent),
}

impl std::fmt::Debug for EdgeItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeItem::Buffer(m) => write!(f, "Buffer({:?})", m.media_type()),
            EdgeItem::Event(e) => write!(f, "Event({e:?})"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Push {
    Accepted,
    Dropped,
    Full,
    Closed,
}

/// How an edge stores **buffers**. Control events never consume capacity.
///
/// Nodes see only [`Edge`]. Construction:
/// - [`Buffered`](EdgeKind::Buffered) — default queue. `capacity == 0` → 64
///   (translated in [`BufferedEdge::new`](crate::graph::BufferedEdge::new)).
///   Cross-thread / burst. `src`/`dst` / `bind_edge` always this kind.
/// - [`Direct`](EdgeKind::Direct) — capacity 0. `offer` runs the consumer
///   (and the rest of a Direct-only chain) on the same executor;
///   `Push::Full` is congestion at the first Buffered edge or sink after the
///   chain. Never fuse across a blocking or async node. Explicit
///   `connect_edge` only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    Buffered { capacity: usize },
    Direct,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeRestart {
    Internal,
    Egress,
    Ingress,
}

impl Default for EdgeKind {
    fn default() -> Self {
        Self::Buffered { capacity: 0 }
    }
}

pub struct Wakeup {
    inner: Mutex<WakeupInner>,
    cv: Condvar,
}

struct WakeupInner {
    pending: u64,
    async_wakers: Vec<Waker>,
}

impl Wakeup {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(WakeupInner {
                pending: 0,
                async_wakers: Vec::new(),
            }),
            cv: Condvar::new(),
        }
    }

    pub fn wait(&self, timeout_ms: i32) -> bool {
        let mut g = self.inner.lock().unwrap();
        let start = std::time::Instant::now();
        loop {
            if g.pending > 0 {
                g.pending -= 1;
                return true;
            }
            match timeout_ms.cmp(&0) {
                std::cmp::Ordering::Less => {
                    g = self.cv.wait(g).unwrap();
                }
                std::cmp::Ordering::Equal => return false,
                std::cmp::Ordering::Greater => {
                    let remain =
                        Duration::from_millis(timeout_ms as u64).saturating_sub(start.elapsed());
                    let (g2, to) = self.cv.wait_timeout(g, remain).unwrap();
                    g = g2;
                    if to.timed_out() {
                        return false;
                    }
                }
            }
        }
    }

    pub fn register_waker(&self, waker: Waker) {
        let mut g = self.inner.lock().unwrap();
        Self::push_waker(&mut g.async_wakers, waker);
    }

    fn push_waker(wakers: &mut Vec<Waker>, waker: Waker) {
        if wakers.iter().any(|w| w.will_wake(&waker)) {
            return;
        }
        wakers.push(waker);
    }

    pub fn arm(&self) {
        // Presence of a waiter is expressed by a registered waker / cond wait.
    }

    pub fn notify(&self) {
        let mut g = self.inner.lock().unwrap();
        g.pending = g.pending.saturating_add(1);
        let wakers = std::mem::take(&mut g.async_wakers);
        drop(g);
        for waker in wakers {
            waker.wake();
        }
        self.cv.notify_one();
    }

    pub fn poll_notified(&self, waker: &Waker) -> bool {
        let mut g = self.inner.lock().unwrap();
        if g.pending > 0 {
            g.pending -= 1;
            return true;
        }
        Self::push_waker(&mut g.async_wakers, waker.clone());
        if g.pending > 0 {
            g.pending -= 1;
            true
        } else {
            false
        }
    }
}

impl Default for Wakeup {
    fn default() -> Self {
        Self::new()
    }
}

pub trait EdgeWaker: Send + Sync {
    fn wake(&self);
}

pub(crate) struct EdgeQueue {
    queue: VecDeque<QueueEntry>,
    buffer_capacity: usize,
    latched_spec: Option<Spec>,
    spec_delivered: bool,
    closed: bool,
    interrupted: bool,
    readable_waker: Option<Box<dyn EdgeWaker>>,
    writable_waker: Option<Box<dyn EdgeWaker>>,
}

struct QueueEntry {
    item: EdgeItem,
    discarded: Option<Arc<AtomicBool>>,
}

impl EdgeQueue {
    pub fn new(buffer_capacity: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(buffer_capacity),
            buffer_capacity,
            latched_spec: None,
            spec_delivered: true,
            closed: false,
            interrupted: false,
            readable_waker: None,
            writable_waker: None,
        }
    }

    fn buffer_count(&self) -> usize {
        self.queue
            .iter()
            .filter(|entry| matches!(entry.item, EdgeItem::Buffer(_)))
            .count()
    }

    pub fn try_push(&mut self, buf: Media) -> Result<(), (Push, Media)> {
        self.try_push_entry(buf, None)
    }

    pub fn try_push_tracked(
        &mut self,
        buf: Media,
        discarded: Arc<AtomicBool>,
    ) -> Result<(), (Push, Media)> {
        self.try_push_entry(buf, Some(discarded))
    }

    fn try_push_entry(
        &mut self,
        buf: Media,
        discarded: Option<Arc<AtomicBool>>,
    ) -> Result<(), (Push, Media)> {
        if self.closed {
            return Err((Push::Closed, buf));
        }
        if self.buffer_count() >= self.buffer_capacity {
            return Err((Push::Full, buf));
        }
        self.queue.push_back(QueueEntry {
            item: EdgeItem::Buffer(buf),
            discarded,
        });
        Ok(())
    }

    pub fn push_event(&mut self, ev: EdgeEvent) {
        match &ev {
            EdgeEvent::FlushStart => {
                self.queue.retain(|entry| {
                    let keep = !matches!(entry.item, EdgeItem::Buffer(_));
                    if !keep && let Some(discarded) = &entry.discarded {
                        discarded.store(true, Ordering::Release);
                    }
                    keep
                });
            }
            EdgeEvent::Spec(spec) => {
                self.latched_spec = Some(spec.clone());
                self.spec_delivered = false;
            }
            EdgeEvent::Eof => self.closed = true,
            _ => {}
        }
        self.queue.push_back(QueueEntry {
            item: EdgeItem::Event(ev),
            discarded: None,
        });
    }

    pub fn try_peek(&self) -> Option<EdgeItem> {
        if !self.spec_delivered {
            if let Some(spec) = self.latched_spec.clone() {
                return Some(EdgeItem::Event(EdgeEvent::Spec(spec)));
            }
        }
        self.queue.front().map(|entry| entry.item.clone())
    }

    pub fn try_take(&mut self) -> Option<EdgeItem> {
        if !self.spec_delivered {
            if let Some(spec) = self.latched_spec.clone() {
                self.spec_delivered = true;
                return Some(EdgeItem::Event(EdgeEvent::Spec(spec)));
            }
        }
        let entry = self.queue.pop_front()?;
        let item = entry.item;
        if matches!(item, EdgeItem::Event(EdgeEvent::Spec(_))) {
            self.spec_delivered = true;
        }
        Some(item)
    }

    pub fn pop(&mut self) {
        if let Some(entry) = self.queue.pop_front() {
            let item = entry.item;
            if matches!(item, EdgeItem::Event(EdgeEvent::Spec(_))) {
                self.spec_delivered = true;
            }
        } else if !self.spec_delivered {
            self.spec_delivered = true;
        }
    }

    pub fn occupied(&self) -> usize {
        self.queue.len()
    }
    pub fn is_closed(&self) -> bool {
        self.closed
    }
    pub fn is_full(&self) -> bool {
        self.buffer_count() >= self.buffer_capacity
    }
    pub fn current_spec(&self) -> Option<Spec> {
        self.latched_spec.clone()
    }
    pub fn rearm_spec(&mut self) {
        if self.latched_spec.is_some() {
            self.spec_delivered = false;
        }
    }

    pub fn reset_for_restart(&mut self, kind: EdgeRestart) {
        match kind {
            EdgeRestart::Internal => {
                for entry in self.queue.drain(..) {
                    if let Some(discarded) = entry.discarded {
                        discarded.store(true, Ordering::Release);
                    }
                }
            }
            EdgeRestart::Egress | EdgeRestart::Ingress => {
                self.queue
                    .retain(|entry| matches!(entry.item, EdgeItem::Buffer(_)));
            }
        }
        self.closed = false;
        self.interrupted = false;
        self.rearm_spec();
    }

    pub fn interrupt(&mut self) {
        self.interrupted = true;
    }

    pub fn take_interrupt(&mut self) -> bool {
        std::mem::take(&mut self.interrupted)
    }

    pub fn take_readable_cb(&mut self) -> Option<Box<dyn EdgeWaker>> {
        self.readable_waker.take()
    }
    pub fn take_writable_cb(&mut self) -> Option<Box<dyn EdgeWaker>> {
        self.writable_waker.take()
    }
    pub fn set_readable_cb(&mut self, w: Box<dyn EdgeWaker>) {
        self.readable_waker = Some(w);
    }
    pub fn set_writable_cb(&mut self, w: Box<dyn EdgeWaker>) {
        self.writable_waker = Some(w);
    }
}

pub struct EdgeReady<'a> {
    edge: &'a dyn Edge,
    writable: bool,
}

impl Future for EdgeReady<'_> {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> TaskPoll<()> {
        if self.writable {
            if !self.edge.is_full() || self.edge.is_closed() {
                return TaskPoll::Ready(());
            }
            self.edge.register_writable_waker(cx.waker().clone());
            if !self.edge.is_full() || self.edge.is_closed() {
                return TaskPoll::Ready(());
            }
        } else {
            if self.edge.occupied() > 0 || self.edge.is_closed() {
                return TaskPoll::Ready(());
            }
            self.edge.register_readable_waker(cx.waker().clone());
            if self.edge.occupied() > 0 || self.edge.is_closed() {
                return TaskPoll::Ready(());
            }
        }
        TaskPoll::Pending
    }
}

pub trait Edge: Send + Sync {
    /// Enqueue `buf`, or give it back. `push` drops the buffer on failure
    /// (C ABI). Cooperative nodes must use `offer`.
    fn offer(&self, buf: Media) -> Result<(), (Push, Media)>;
    fn push(&self, buf: Media) -> Push {
        match self.offer(buf) {
            Ok(()) => Push::Accepted,
            Err((status, _)) => status,
        }
    }
    fn push_event(&self, ev: EdgeEvent);
    fn take(&self, timeout_ms: i32) -> Option<EdgeItem>;
    fn peek_clone(&self, timeout_ms: i32) -> Option<EdgeItem>;
    fn pop(&self);
    fn occupied(&self) -> usize;
    fn current_spec(&self) -> Option<Spec>;
    fn rearm_spec(&self);
    fn notify_readable(&self, node: Box<dyn EdgeWaker>);
    fn notify_writable(&self, node: Box<dyn EdgeWaker>);
    fn is_closed(&self) -> bool;
    fn is_full(&self) -> bool;
    fn arm_readable(&self);
    fn arm_writable(&self);
    fn register_readable_waker(&self, waker: Waker);
    fn register_writable_waker(&self, waker: Waker);

    fn writer_generation(&self) -> u64 {
        0
    }
    fn offer_generation(&self, generation: u64, buf: Media) -> Result<(), (Push, Media)> {
        let active = self.writer_generation();
        if active != 0 && generation != active {
            Err((Push::Closed, buf))
        } else {
            self.offer(buf)
        }
    }
    fn push_event_generation(&self, generation: u64, ev: EdgeEvent) {
        let active = self.writer_generation();
        if active == 0 || generation == active {
            self.push_event(ev);
        }
    }
    fn fence_generation(&self, _old: u64, _new: u64) -> bool {
        true
    }
    fn reset_for_restart(&self, _kind: EdgeRestart) {}
    fn restart(&self, old: u64, new: u64, kind: EdgeRestart) {
        if kind == EdgeRestart::Ingress || self.fence_generation(old, new) {
            self.reset_for_restart(kind);
        }
    }
    fn interrupt(&self) {}

    fn try_take(&self) -> Option<EdgeItem> {
        self.take(0)
    }
    fn take_generation(&self, _generation: u64, timeout_ms: i32) -> Option<EdgeItem> {
        self.take(timeout_ms)
    }
    fn peek_clone_generation(&self, _generation: u64, timeout_ms: i32) -> Option<EdgeItem> {
        self.peek_clone(timeout_ms)
    }
    fn pop_generation(&self, _generation: u64) {
        self.pop();
    }
    fn try_take_generation(&self, _generation: u64) -> Option<EdgeItem> {
        self.try_take()
    }

    /// Direct hops follow this edge for `is_full` / writable wait. Buffered
    /// ignores it.
    fn attach_direct_tail(&self, _tail: Option<Arc<dyn Edge>>) {}
    fn attach_direct_consumer(&self, _consumer: Arc<dyn crate::graph::Node>) {}
    fn is_direct(&self) -> bool {
        false
    }
}

struct GenerationReader {
    edge: Arc<dyn Edge>,
    generation: u64,
}

pub fn generation_reader(edge: Arc<dyn Edge>, generation: u64) -> Arc<dyn Edge> {
    if edge.is_direct() {
        Arc::new(GenerationReader { edge, generation })
    } else {
        edge
    }
}

impl Edge for GenerationReader {
    fn offer(&self, buf: Media) -> Result<(), (Push, Media)> {
        self.edge.offer(buf)
    }
    fn push_event(&self, ev: EdgeEvent) {
        self.edge.push_event(ev);
    }
    fn take(&self, timeout_ms: i32) -> Option<EdgeItem> {
        self.edge.take_generation(self.generation, timeout_ms)
    }
    fn peek_clone(&self, timeout_ms: i32) -> Option<EdgeItem> {
        self.edge.peek_clone_generation(self.generation, timeout_ms)
    }
    fn pop(&self) {
        self.edge.pop_generation(self.generation);
    }
    fn occupied(&self) -> usize {
        if self.edge.writer_generation() == self.generation {
            self.edge.occupied()
        } else {
            0
        }
    }
    fn current_spec(&self) -> Option<Spec> {
        if self.edge.writer_generation() == self.generation {
            self.edge.current_spec()
        } else {
            None
        }
    }
    fn rearm_spec(&self) {
        self.edge.rearm_spec();
    }
    fn notify_readable(&self, node: Box<dyn EdgeWaker>) {
        self.edge.notify_readable(node);
    }
    fn notify_writable(&self, node: Box<dyn EdgeWaker>) {
        self.edge.notify_writable(node);
    }
    fn is_closed(&self) -> bool {
        self.edge.writer_generation() != self.generation || self.edge.is_closed()
    }
    fn is_full(&self) -> bool {
        self.edge.is_full()
    }
    fn arm_readable(&self) {
        self.edge.arm_readable();
    }
    fn arm_writable(&self) {
        self.edge.arm_writable();
    }
    fn register_readable_waker(&self, waker: Waker) {
        self.edge.register_readable_waker(waker);
    }
    fn register_writable_waker(&self, waker: Waker) {
        self.edge.register_writable_waker(waker);
    }
    fn writer_generation(&self) -> u64 {
        self.edge.writer_generation()
    }
    fn take_generation(&self, generation: u64, timeout_ms: i32) -> Option<EdgeItem> {
        (generation == self.generation)
            .then(|| self.edge.take_generation(self.generation, timeout_ms))
            .flatten()
    }
    fn peek_clone_generation(&self, generation: u64, timeout_ms: i32) -> Option<EdgeItem> {
        (generation == self.generation)
            .then(|| self.edge.peek_clone_generation(self.generation, timeout_ms))
            .flatten()
    }
    fn pop_generation(&self, generation: u64) {
        if generation == self.generation {
            self.edge.pop_generation(self.generation);
        }
    }
    fn try_take(&self) -> Option<EdgeItem> {
        self.edge.try_take_generation(self.generation)
    }
    fn try_take_generation(&self, generation: u64) -> Option<EdgeItem> {
        (generation == self.generation)
            .then(|| self.edge.try_take_generation(self.generation))
            .flatten()
    }
    fn interrupt(&self) {
        self.edge.interrupt();
    }
    fn attach_direct_tail(&self, tail: Option<Arc<dyn Edge>>) {
        self.edge.attach_direct_tail(tail);
    }
    fn attach_direct_consumer(&self, consumer: Arc<dyn crate::graph::Node>) {
        self.edge.attach_direct_consumer(consumer);
    }
    fn is_direct(&self) -> bool {
        true
    }
}

struct GenerationWriter {
    edge: Arc<dyn Edge>,
    generation: u64,
}

pub fn generation_writer(edge: Arc<dyn Edge>, generation: u64) -> Arc<dyn Edge> {
    Arc::new(GenerationWriter { edge, generation })
}

impl Edge for GenerationWriter {
    fn offer(&self, buf: Media) -> Result<(), (Push, Media)> {
        self.edge.offer_generation(self.generation, buf)
    }
    fn push_event(&self, ev: EdgeEvent) {
        self.edge.push_event_generation(self.generation, ev);
    }
    fn take(&self, timeout_ms: i32) -> Option<EdgeItem> {
        self.edge.take(timeout_ms)
    }
    fn peek_clone(&self, timeout_ms: i32) -> Option<EdgeItem> {
        self.edge.peek_clone(timeout_ms)
    }
    fn pop(&self) {
        self.edge.pop();
    }
    fn occupied(&self) -> usize {
        self.edge.occupied()
    }
    fn current_spec(&self) -> Option<Spec> {
        self.edge.current_spec()
    }
    fn rearm_spec(&self) {
        self.edge.rearm_spec();
    }
    fn notify_readable(&self, node: Box<dyn EdgeWaker>) {
        self.edge.notify_readable(node);
    }
    fn notify_writable(&self, node: Box<dyn EdgeWaker>) {
        self.edge.notify_writable(node);
    }
    fn is_closed(&self) -> bool {
        self.edge.is_closed() || self.edge.writer_generation() != self.generation
    }
    fn is_full(&self) -> bool {
        self.edge.is_full()
    }
    fn arm_readable(&self) {
        self.edge.arm_readable();
    }
    fn arm_writable(&self) {
        self.edge.arm_writable();
    }
    fn register_readable_waker(&self, waker: Waker) {
        self.edge.register_readable_waker(waker);
    }
    fn register_writable_waker(&self, waker: Waker) {
        self.edge.register_writable_waker(waker);
    }
    fn writer_generation(&self) -> u64 {
        self.edge.writer_generation()
    }
    fn offer_generation(&self, generation: u64, buf: Media) -> Result<(), (Push, Media)> {
        if generation != self.generation {
            return Err((Push::Closed, buf));
        }
        self.edge.offer_generation(self.generation, buf)
    }
    fn push_event_generation(&self, generation: u64, ev: EdgeEvent) {
        if generation == self.generation {
            self.edge.push_event_generation(self.generation, ev);
        }
    }
    fn fence_generation(&self, old: u64, new: u64) -> bool {
        self.edge.fence_generation(old, new)
    }
    fn reset_for_restart(&self, kind: EdgeRestart) {
        self.edge.reset_for_restart(kind);
    }
    fn restart(&self, old: u64, new: u64, kind: EdgeRestart) {
        self.edge.restart(old, new, kind);
    }
    fn interrupt(&self) {
        self.edge.interrupt();
    }
    fn attach_direct_tail(&self, tail: Option<Arc<dyn Edge>>) {
        self.edge.attach_direct_tail(tail);
    }
    fn attach_direct_consumer(&self, consumer: Arc<dyn crate::graph::Node>) {
        self.edge.attach_direct_consumer(consumer);
    }
    fn is_direct(&self) -> bool {
        self.edge.is_direct()
    }
}

impl dyn Edge {
    pub fn wait_readable(&self) -> EdgeReady<'_> {
        EdgeReady {
            edge: self,
            writable: false,
        }
    }
    pub fn wait_writable(&self) -> EdgeReady<'_> {
        EdgeReady {
            edge: self,
            writable: true,
        }
    }
    pub fn readable(&self) -> EdgeReady<'_> {
        self.wait_readable()
    }
    pub fn writable(&self) -> EdgeReady<'_> {
        self.wait_writable()
    }
}
