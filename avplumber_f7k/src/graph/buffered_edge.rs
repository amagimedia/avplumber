//! Buffered edge: a bounded queue of owned `Media` plus control events.
//!
//! Default kind. Producer and consumer may run on different executors; the
//! queue absorbs bursts. `capacity == 0` is 64. Capacity counts **buffers
//! only** — Spec/Eof/Flush items still enqueue when the media side is full.
//! `FlushStart` drops queued buffers, keeps events, then appends itself.
//!
//! Blocking `take(-1)` parks on a condvar. Poll/async register a waker.
//! One producer, one consumer: one waker per direction.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Waker;

use crate::graph::edge::{
    Edge, EdgeEvent, EdgeHint, EdgeHintCell, EdgeItem, EdgeQueue, EdgeRestart, EdgeWaker, Push,
    Wakeup,
};
use crate::graph::media::Media;
use crate::graph::spec::Spec;

pub struct BufferedEdge {
    inner: Mutex<EdgeQueue>,
    readable: Wakeup,
    writable: Wakeup,
    /// Outside `inner`: hints are not part of the ordered path, and staying out
    /// of the queue is also what makes them survive `reset_for_restart`.
    hints: EdgeHintCell,
    writer_generation: AtomicU64,
}

impl BufferedEdge {
    pub fn new(capacity: usize) -> Self {
        let cap = if capacity == 0 { 64 } else { capacity };
        Self {
            inner: Mutex::new(EdgeQueue::new(cap)),
            readable: Wakeup::new(),
            writable: Wakeup::new(),
            hints: EdgeHintCell::default(),
            writer_generation: AtomicU64::new(1),
        }
    }

    fn fire(cb: Option<Box<dyn EdgeWaker>>) {
        if let Some(cb) = cb {
            cb.wake();
        }
    }

    fn wait_readable(&self, timeout_ms: i32) -> bool {
        if timeout_ms == 0 {
            return false;
        }
        self.readable.wait(timeout_ms)
    }
}

impl Edge for BufferedEdge {
    fn offer(&self, buf: Media) -> Result<(), (Push, Media)> {
        let mut g = self.inner.lock().unwrap();
        match g.try_push(buf) {
            Ok(()) => {
                let cb = g.take_readable_cb();
                drop(g);
                self.readable.notify();
                Self::fire(cb);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn push_event(&self, ev: EdgeEvent) {
        let mut g = self.inner.lock().unwrap();
        g.push_event(ev);
        let cb = g.take_readable_cb();
        drop(g);
        self.readable.notify();
        Self::fire(cb);
    }

    fn post_hint(&self, hint: EdgeHint) {
        self.hints.post(hint);
        // Same wake pattern as `push_event`, mirrored to the producer side.
        let cb = self.inner.lock().unwrap().take_writable_cb();
        self.writable.notify();
        Self::fire(cb);
    }

    fn take_hints(&self) -> Vec<EdgeHint> {
        self.hints.take()
    }

    fn has_hints(&self) -> bool {
        self.hints.pending()
    }

    fn take(&self, timeout_ms: i32) -> Option<EdgeItem> {
        loop {
            if let Some(item) = self.try_take() {
                return Some(item);
            }
            if self.inner.lock().unwrap().take_interrupt() {
                return None;
            }
            if timeout_ms == 0 || !self.wait_readable(timeout_ms) {
                return self.try_take();
            }
        }
    }

    fn peek_clone(&self, timeout_ms: i32) -> Option<EdgeItem> {
        loop {
            {
                let g = self.inner.lock().unwrap();
                if let Some(item) = g.try_peek() {
                    return Some(item);
                }
            }
            if self.inner.lock().unwrap().take_interrupt() {
                return None;
            }
            if timeout_ms == 0 || !self.wait_readable(timeout_ms) {
                return self.inner.lock().unwrap().try_peek();
            }
        }
    }

    fn pop(&self) {
        let mut g = self.inner.lock().unwrap();
        g.pop();
        let cb = g.take_writable_cb();
        drop(g);
        self.writable.notify();
        Self::fire(cb);
    }

    fn occupied(&self) -> usize {
        self.inner.lock().unwrap().occupied()
    }
    fn current_spec(&self) -> Option<Spec> {
        self.inner.lock().unwrap().current_spec()
    }
    fn rearm_spec(&self) {
        self.inner.lock().unwrap().rearm_spec();
    }
    fn notify_readable(&self, node: Box<dyn EdgeWaker>) {
        self.inner.lock().unwrap().set_readable_cb(node);
    }
    fn notify_writable(&self, node: Box<dyn EdgeWaker>) {
        self.inner.lock().unwrap().set_writable_cb(node);
    }
    fn is_closed(&self) -> bool {
        self.inner.lock().unwrap().is_closed()
    }
    fn is_full(&self) -> bool {
        self.inner.lock().unwrap().is_full()
    }
    fn arm_readable(&self) {
        self.readable.arm();
    }
    fn arm_writable(&self) {
        self.writable.arm();
    }
    fn register_readable_waker(&self, waker: Waker) {
        self.readable.register_waker(waker);
    }
    fn register_writable_waker(&self, waker: Waker) {
        self.writable.register_waker(waker);
    }

    fn writer_generation(&self) -> u64 {
        self.writer_generation.load(Ordering::Acquire)
    }

    fn offer_generation(&self, generation: u64, buf: Media) -> Result<(), (Push, Media)> {
        let recovery = buf.clone();
        let discarded = Arc::new(AtomicBool::new(false));
        let result = {
            let mut queue = self.inner.lock().unwrap();
            if self.writer_generation() != generation {
                return Err((Push::Closed, buf));
            }
            match queue.try_push_tracked(buf, discarded.clone()) {
                Ok(()) => Ok(queue.take_readable_cb()),
                Err(error) => Err(error),
            }
        };
        match result {
            Ok(callback) => {
                self.readable.notify();
                Self::fire(callback);
                if discarded.load(Ordering::Acquire) && self.writer_generation() != generation {
                    Err((Push::Closed, recovery))
                } else {
                    Ok(())
                }
            }
            Err(error) => Err(error),
        }
    }

    fn push_event_generation(&self, generation: u64, ev: EdgeEvent) {
        let callback = {
            let mut queue = self.inner.lock().unwrap();
            if self.writer_generation() != generation {
                return;
            }
            queue.push_event(ev);
            queue.take_readable_cb()
        };
        self.readable.notify();
        Self::fire(callback);
    }

    fn fence_generation(&self, old: u64, new: u64) -> bool {
        let _queue = self.inner.lock().unwrap();
        self.writer_generation
            .compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    fn reset_for_restart(&self, kind: EdgeRestart) {
        let (readable, writable) = {
            let mut g = self.inner.lock().unwrap();
            g.reset_for_restart(kind);
            (g.take_readable_cb(), g.take_writable_cb())
        };
        // Hints outlive the queue reset; re-arm them because the restart may
        // have replaced the producer that already drained them.
        self.hints.rearm();
        self.readable.notify();
        self.writable.notify();
        Self::fire(readable);
        Self::fire(writable);
    }

    fn restart(&self, old: u64, new: u64, kind: EdgeRestart) {
        let callbacks = {
            let mut queue = self.inner.lock().unwrap();
            if kind != EdgeRestart::Ingress
                && self
                    .writer_generation
                    .compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
            {
                return;
            }
            queue.reset_for_restart(kind);
            (queue.take_readable_cb(), queue.take_writable_cb())
        };
        self.hints.rearm();
        self.readable.notify();
        self.writable.notify();
        Self::fire(callbacks.0);
        Self::fire(callbacks.1);
    }

    fn interrupt(&self) {
        let (readable, writable) = {
            let mut queue = self.inner.lock().unwrap();
            queue.interrupt();
            (queue.take_readable_cb(), queue.take_writable_cb())
        };
        self.readable.notify();
        self.writable.notify();
        Self::fire(readable);
        Self::fire(writable);
    }

    fn try_take(&self) -> Option<EdgeItem> {
        let mut g = self.inner.lock().unwrap();
        let item = g.try_take()?;
        let cb = g.take_writable_cb();
        drop(g);
        self.writable.notify();
        Self::fire(cb);
        Some(item)
    }
}
