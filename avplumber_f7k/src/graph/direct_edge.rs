//! Direct edge: capacity 0. `offer` fuses into the consumer on this executor.
//!
//! No media slot is stored on the hop after `offer` returns. `Push::Full` is
//! congestion at the first [`BufferedEdge`](crate::graph::BufferedEdge) (or
//! sink) after a Direct-only chain — `is_full` and writable wait follow that
//! tail. Control events still queue (they never count as buffer capacity).
//!
//! Never fuse across a blocking or async node; `connect_edge` rejects those.
//! Selection is explicit `EdgeKind::Direct`, not inferred from co-location.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::task::Waker;

use crate::graph::edge::{
    Edge, EdgeEvent, EdgeHint, EdgeHintCell, EdgeItem, EdgeQueue, EdgeRestart, EdgeWaker, Push,
    Wakeup,
};
use crate::graph::media::Media;
use crate::graph::node::{Node, Tick};
use crate::graph::poll_ctx::NodePollContext;
use crate::graph::spec::Spec;

const FUSE_AGAIN_BUDGET: u32 = 32;
const OFFER_PENDING: u8 = 0;
const OFFER_CONSUMED: u8 = 1;
const OFFER_DISCARDED: u8 = 2;

pub struct DirectEdge {
    state: Mutex<DirectState>,
    consumer: Mutex<Option<Weak<dyn Node>>>,
    tail: Mutex<Option<Arc<dyn Edge>>>,
    readable: Wakeup,
    writable: Wakeup,
    /// Outside `state`: hints are not part of the ordered path, and staying out
    /// of the event queue is also what makes them survive a restart.
    hints: EdgeHintCell,
    writer_generation: AtomicU64,
}

struct DirectState {
    events: EdgeQueue,
    inflight: Option<Media>,
    offering: bool,
    offer_serial: u64,
    active_offer: Option<u64>,
    active_status: Option<Arc<AtomicU8>>,
    active_cancel: Option<Arc<AtomicBool>>,
}

impl DirectState {
    fn invalidate_offer(&mut self) {
        if self.inflight.take().is_some()
            && let Some(status) = &self.active_status
        {
            status.store(OFFER_DISCARDED, Ordering::Release);
        }
        if let Some(cancel) = &self.active_cancel {
            cancel.store(true, Ordering::Release);
        }
        self.offering = false;
        self.active_offer = None;
        self.active_status = None;
        self.active_cancel = None;
    }

    fn take_inflight(&mut self) -> Option<Media> {
        let media = self.inflight.take()?;
        if let Some(status) = &self.active_status {
            status.store(OFFER_CONSUMED, Ordering::Release);
        }
        Some(media)
    }
}

impl DirectEdge {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(DirectState {
                events: EdgeQueue::new(0),
                inflight: None,
                offering: false,
                offer_serial: 0,
                active_offer: None,
                active_status: None,
                active_cancel: None,
            }),
            consumer: Mutex::new(None),
            tail: Mutex::new(None),
            readable: Wakeup::new(),
            writable: Wakeup::new(),
            hints: EdgeHintCell::default(),
            writer_generation: AtomicU64::new(1),
        }
    }

    pub fn set_consumer(&self, node: Arc<dyn Node>) {
        *self.consumer.lock().unwrap() = Some(Arc::downgrade(&node));
    }

    fn fire(cb: Option<Box<dyn EdgeWaker>>) {
        if let Some(cb) = cb {
            cb.wake();
        }
    }

    fn tail(&self) -> Option<Arc<dyn Edge>> {
        self.tail.lock().unwrap().clone()
    }

    fn offer_checked(&self, generation: Option<u64>, buf: Media) -> Result<(), (Push, Media)> {
        let recovery = buf.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let status = Arc::new(AtomicU8::new(OFFER_PENDING));
        let offer_id = {
            let mut state = self.state.lock().unwrap();
            if generation.is_some_and(|value| self.writer_generation() != value)
                || state.events.is_closed()
            {
                return Err((Push::Closed, buf));
            }
            if state.offering {
                return Err((Push::Full, buf));
            }
            state.offer_serial = state.offer_serial.wrapping_add(1);
            let offer_id = state.offer_serial;
            state.offering = true;
            state.inflight = Some(buf);
            state.active_offer = Some(offer_id);
            state.active_status = Some(status.clone());
            state.active_cancel = Some(cancel.clone());
            offer_id
        };
        self.readable.notify();

        let consumer = self
            .consumer
            .lock()
            .unwrap()
            .as_ref()
            .and_then(Weak::upgrade);
        if let Some(consumer) = consumer {
            let mut ctx = NodePollContext::new(cancel.clone(), Arc::new(Wakeup::new()));
            for _ in 0..FUSE_AGAIN_BUDGET {
                let is_active = {
                    let state = self.state.lock().unwrap();
                    state.active_offer == Some(offer_id) && state.inflight.is_some()
                };
                if !is_active || ctx.is_cancelled() {
                    break;
                }
                match consumer.poll(&mut ctx) {
                    Tick::Again => ctx.clear_park(),
                    Tick::Idle | Tick::Done => break,
                }
            }
        }

        let (leftover, fenced) = {
            let mut state = self.state.lock().unwrap();
            let leftover = if state.active_offer == Some(offer_id) {
                state.offering = false;
                state.active_offer = None;
                state.active_status = None;
                state.active_cancel = None;
                state.inflight.take()
            } else {
                None
            };
            (
                leftover,
                generation.is_some_and(|value| self.writer_generation() != value)
                    || cancel.load(Ordering::Acquire),
            )
        };
        if status.load(Ordering::Acquire) == OFFER_DISCARDED {
            return Err((Push::Closed, recovery));
        }
        match leftover {
            Some(buf) if fenced => Err((Push::Closed, buf)),
            Some(buf) => Err((Push::Full, buf)),
            None => {
                self.writable.notify();
                Ok(())
            }
        }
    }

    fn try_take_checked(&self, generation: Option<u64>) -> Option<EdgeItem> {
        let (item, cb) = {
            let mut state = self.state.lock().unwrap();
            if generation.is_some_and(|value| self.writer_generation() != value) {
                return None;
            }
            let item = state
                .events
                .try_take()
                .or_else(|| state.take_inflight().map(EdgeItem::Buffer))?;
            (item, state.events.take_writable_cb())
        };
        self.writable.notify();
        Self::fire(cb);
        Some(item)
    }

    fn try_peek_checked(&self, generation: Option<u64>) -> Option<EdgeItem> {
        let state = self.state.lock().unwrap();
        if generation.is_some_and(|value| self.writer_generation() != value) {
            return None;
        }
        state
            .events
            .try_peek()
            .or_else(|| state.inflight.clone().map(EdgeItem::Buffer))
    }

    /// Removes exactly what [`Self::try_peek_checked`] would return, which the
    /// events-before-`inflight` order makes non-obvious: popping the `inflight`
    /// buffer while an event is still queued would drop that buffer *and* tell
    /// the producer it was consumed. Delegating keeps one precedence rule.
    fn pop_checked(&self, generation: Option<u64>) {
        let _ = self.try_take_checked(generation);
    }
}

impl Default for DirectEdge {
    fn default() -> Self {
        Self::new()
    }
}

impl Edge for DirectEdge {
    fn offer(&self, buf: Media) -> Result<(), (Push, Media)> {
        self.offer_checked(None, buf)
    }

    fn push_event(&self, ev: EdgeEvent) {
        let cb = {
            let mut state = self.state.lock().unwrap();
            state.events.push_event(ev);
            state.events.take_readable_cb()
        };
        self.readable.notify();
        Self::fire(cb);
    }

    fn post_hint(&self, hint: EdgeHint) {
        self.hints.post(hint);
        // Same wake pattern as `push_event`, mirrored to the producer side.
        let cb = self.state.lock().unwrap().events.take_writable_cb();
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
            if let Some(item) = self.try_take_checked(None) {
                return Some(item);
            }
            if self.state.lock().unwrap().events.take_interrupt() {
                return None;
            }
            if timeout_ms == 0 || !self.readable.wait(timeout_ms) {
                return self.try_take();
            }
        }
    }

    fn peek_clone(&self, timeout_ms: i32) -> Option<EdgeItem> {
        loop {
            if let Some(item) = self.try_peek_checked(None) {
                return Some(item);
            }
            if self.state.lock().unwrap().events.take_interrupt() {
                return None;
            }
            if timeout_ms == 0 || !self.readable.wait(timeout_ms) {
                return self.try_peek();
            }
        }
    }

    fn pop(&self) {
        self.pop_checked(None);
    }

    fn occupied(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.events.occupied() + usize::from(state.inflight.is_some())
    }

    fn current_spec(&self) -> Option<Spec> {
        self.state.lock().unwrap().events.current_spec()
    }
    fn rearm_spec(&self) {
        self.state.lock().unwrap().events.rearm_spec();
    }
    fn notify_readable(&self, node: Box<dyn EdgeWaker>) {
        self.state.lock().unwrap().events.set_readable_cb(node);
    }
    fn notify_writable(&self, node: Box<dyn EdgeWaker>) {
        if let Some(tail) = self.tail() {
            tail.notify_writable(node);
        } else {
            self.state.lock().unwrap().events.set_writable_cb(node);
        }
    }
    fn is_closed(&self) -> bool {
        self.state.lock().unwrap().events.is_closed()
    }
    fn is_full(&self) -> bool {
        if self.is_closed() {
            return true;
        }
        self.tail().map(|t| t.is_full()).unwrap_or(false)
    }
    fn arm_readable(&self) {
        self.readable.arm();
    }
    fn arm_writable(&self) {
        self.writable.arm();
        if let Some(tail) = self.tail() {
            tail.arm_writable();
        }
    }
    fn register_readable_waker(&self, waker: Waker) {
        self.readable.register_waker(waker);
    }
    fn register_writable_waker(&self, waker: Waker) {
        self.writable.register_waker(waker.clone());
        if let Some(tail) = self.tail() {
            tail.register_writable_waker(waker);
        }
    }

    fn writer_generation(&self) -> u64 {
        self.writer_generation.load(Ordering::Acquire)
    }

    fn offer_generation(&self, generation: u64, buf: Media) -> Result<(), (Push, Media)> {
        self.offer_checked(Some(generation), buf)
    }

    fn push_event_generation(&self, generation: u64, ev: EdgeEvent) {
        let cb = {
            let mut state = self.state.lock().unwrap();
            if self.writer_generation() != generation {
                return;
            }
            state.events.push_event(ev);
            state.events.take_readable_cb()
        };
        self.readable.notify();
        Self::fire(cb);
    }

    fn fence_generation(&self, old: u64, new: u64) -> bool {
        let _state = self.state.lock().unwrap();
        self.writer_generation
            .compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    fn reset_for_restart(&self, kind: EdgeRestart) {
        let (readable, writable) = {
            let mut state = self.state.lock().unwrap();
            state.invalidate_offer();
            state.events.reset_for_restart(kind);
            (
                state.events.take_readable_cb(),
                state.events.take_writable_cb(),
            )
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
            let mut state = self.state.lock().unwrap();
            if kind != EdgeRestart::Ingress
                && self
                    .writer_generation
                    .compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
            {
                return;
            }
            state.invalidate_offer();
            state.events.reset_for_restart(kind);
            (
                state.events.take_readable_cb(),
                state.events.take_writable_cb(),
            )
        };
        self.hints.rearm();
        self.readable.notify();
        self.writable.notify();
        Self::fire(callbacks.0);
        Self::fire(callbacks.1);
    }

    fn interrupt(&self) {
        let (readable, writable) = {
            let mut state = self.state.lock().unwrap();
            state.events.interrupt();
            (
                state.events.take_readable_cb(),
                state.events.take_writable_cb(),
            )
        };
        self.readable.notify();
        self.writable.notify();
        Self::fire(readable);
        Self::fire(writable);
    }

    fn try_take(&self) -> Option<EdgeItem> {
        self.try_take_checked(None)
    }

    fn take_generation(&self, generation: u64, timeout_ms: i32) -> Option<EdgeItem> {
        loop {
            if let Some(item) = self.try_take_checked(Some(generation)) {
                return Some(item);
            }
            if self.writer_generation() != generation
                || self.state.lock().unwrap().events.take_interrupt()
            {
                return None;
            }
            if timeout_ms == 0 || !self.readable.wait(timeout_ms) {
                return self.try_take_checked(Some(generation));
            }
        }
    }

    fn peek_clone_generation(&self, generation: u64, timeout_ms: i32) -> Option<EdgeItem> {
        loop {
            if let Some(item) = self.try_peek_checked(Some(generation)) {
                return Some(item);
            }
            if self.writer_generation() != generation
                || self.state.lock().unwrap().events.take_interrupt()
            {
                return None;
            }
            if timeout_ms == 0 || !self.readable.wait(timeout_ms) {
                return self.try_peek_checked(Some(generation));
            }
        }
    }

    fn pop_generation(&self, generation: u64) {
        self.pop_checked(Some(generation));
    }

    fn try_take_generation(&self, generation: u64) -> Option<EdgeItem> {
        self.try_take_checked(Some(generation))
    }

    fn attach_direct_tail(&self, tail: Option<Arc<dyn Edge>>) {
        *self.tail.lock().unwrap() = tail;
    }

    fn attach_direct_consumer(&self, consumer: Arc<dyn Node>) {
        self.set_consumer(consumer);
    }

    fn is_direct(&self) -> bool {
        true
    }
}

impl DirectEdge {
    fn try_peek(&self) -> Option<EdgeItem> {
        self.try_peek_checked(None)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, OnceLock};
    use std::task::{Context, Poll as TaskPoll, RawWaker, RawWakerVTable, Waker};

    use super::*;
    use crate::graph::buffer::{AvpMediaType, AvpRational};
    use crate::graph::buffered_edge::BufferedEdge;
    use crate::graph::node::NodeKind;
    use crate::graph::spec::Spec;

    struct Fwd {
        name: &'static str,
        src: OnceLock<Arc<dyn Edge>>,
        sink: OnceLock<Arc<dyn Edge>>,
    }

    impl Fwd {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                src: OnceLock::new(),
                sink: OnceLock::new(),
            }
        }
    }

    impl Node for Fwd {
        fn name(&self) -> &str {
            self.name
        }
        fn kind(&self) -> NodeKind {
            NodeKind::Poll
        }
        fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
            let _ = self.src.set(edge);
        }
        fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
            let _ = self.sink.set(edge);
        }
        fn poll(&self, ctx: &mut NodePollContext) -> Tick {
            let Some(src) = self.src.get() else {
                return Tick::Done;
            };
            if let Some(sink) = self.sink.get() {
                if sink.is_full() {
                    ctx.wait_writable(sink.clone());
                    return Tick::Idle;
                }
            }
            match src.try_take() {
                None => {
                    ctx.wait_readable(src.clone());
                    Tick::Idle
                }
                Some(EdgeItem::Buffer(buf)) => {
                    let Some(sink) = self.sink.get() else {
                        return Tick::Again;
                    };
                    match sink.offer(buf) {
                        Ok(()) => Tick::Again,
                        Err((Push::Full, _)) => Tick::Idle,
                        Err((Push::Closed | Push::Dropped, _)) => Tick::Done,
                        Err((Push::Accepted, _)) => Tick::Again,
                    }
                }
                Some(EdgeItem::Event(_)) => Tick::Again,
            }
        }
    }

    struct IdleNode {
        name: &'static str,
        src: OnceLock<Arc<dyn Edge>>,
    }

    impl Node for IdleNode {
        fn name(&self) -> &str {
            self.name
        }
        fn kind(&self) -> NodeKind {
            NodeKind::Poll
        }
        fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
            let _ = self.src.set(edge);
        }
        fn poll(&self, ctx: &mut NodePollContext) -> Tick {
            if let Some(src) = self.src.get() {
                ctx.wait_readable(src.clone());
            }
            Tick::Idle
        }
    }

    fn stub(pts: i64) -> Media {
        crate::graph::media::test_media(AvpMediaType::VIDEO, pts)
    }

    fn video_spec() -> Spec {
        Spec::Video {
            width: 8,
            height: 8,
            pix_fmt: 0,
            frame_rate: AvpRational { num: 1, den: 1 },
            sar: AvpRational { num: 1, den: 1 },
            time_base: AvpRational { num: 1, den: 1000 },
        }
    }

    fn buffer_pts(edge: &dyn Edge) -> Vec<i64> {
        let mut pts = Vec::new();
        while let Some(item) = edge.try_take() {
            if let EdgeItem::Buffer(buf) = item {
                pts.push(buf.ts().val);
            }
        }
        pts
    }

    fn recording_waker(flag: Arc<AtomicBool>) -> Waker {
        const VTABLE: RawWakerVTable =
            RawWakerVTable::new(clone_waker, wake, wake_by_ref, drop_waker);
        unsafe fn clone_waker(ptr: *const ()) -> RawWaker {
            unsafe { Arc::increment_strong_count(ptr as *const AtomicBool) };
            RawWaker::new(ptr, &VTABLE)
        }
        unsafe fn wake(ptr: *const ()) {
            unsafe {
                wake_by_ref(ptr);
                drop_waker(ptr);
            }
        }
        unsafe fn wake_by_ref(ptr: *const ()) {
            unsafe { (*ptr.cast::<AtomicBool>()).store(true, Ordering::SeqCst) };
        }
        unsafe fn drop_waker(ptr: *const ()) {
            drop(unsafe { Arc::from_raw(ptr as *const AtomicBool) });
        }
        let ptr = Arc::into_raw(flag);
        unsafe { Waker::from_raw(RawWaker::new(ptr.cast(), &VTABLE)) }
    }

    /// Direct → poll node → Direct → poll node → Buffered(1).
    /// Returned nodes must stay alive: Direct holds only `Weak` consumers.
    fn two_hop_direct_chain() -> (
        Arc<DirectEdge>,
        Arc<DirectEdge>,
        Arc<dyn Edge>,
        Arc<Fwd>,
        Arc<Fwd>,
    ) {
        let d1 = Arc::new(DirectEdge::new());
        let d2 = Arc::new(DirectEdge::new());
        let buf: Arc<dyn Edge> = Arc::new(BufferedEdge::new(1));
        let a = Arc::new(Fwd::new("a"));
        let b = Arc::new(Fwd::new("b"));
        a.bind_source("in", d1.clone());
        a.bind_sink("out", d2.clone());
        b.bind_source("in", d2.clone());
        b.bind_sink("out", buf.clone());
        d1.set_consumer(a.clone());
        d2.set_consumer(b.clone());
        d1.attach_direct_tail(Some(d2.clone()));
        d2.attach_direct_tail(Some(buf.clone()));
        (d1, d2, buf, a, b)
    }

    #[test]
    fn offer_fuses_into_consumer_and_stores_nothing() {
        let hop = Arc::new(DirectEdge::new());
        let out: Arc<dyn Edge> = Arc::new(BufferedEdge::new(4));
        let node = Arc::new(Fwd::new("fwd"));
        node.bind_source("in", hop.clone());
        node.bind_sink("out", out.clone());
        hop.set_consumer(node.clone());
        hop.attach_direct_tail(Some(out.clone()));

        hop.push_event(EdgeEvent::Spec(video_spec()));
        assert!(hop.offer(stub(3)).is_ok());
        assert!(buffer_pts(&*hop).is_empty());
        assert_eq!(buffer_pts(&*out), vec![3]);
        assert!(!hop.is_full());
    }

    #[test]
    fn offer_returns_full_when_consumer_does_not_take() {
        let hop = Arc::new(DirectEdge::new());
        let node = Arc::new(IdleNode {
            name: "idle",
            src: OnceLock::new(),
        });
        node.bind_source("in", hop.clone());
        hop.set_consumer(node.clone());

        match hop.offer(stub(1)) {
            Err((Push::Full, buf)) => assert_eq!(buf.ts().val, 1),
            Ok(()) => panic!("expected Full, offer succeeded"),
            Err((status, _)) => panic!("expected Full, got {status:?}"),
        }
        assert!(hop.try_take().is_none());
    }

    #[test]
    fn chain_is_full_when_buffered_tail_is_full() {
        let (d1, d2, buf, _a, _b) = two_hop_direct_chain();

        assert_eq!(buf.push(stub(0)), Push::Accepted);
        assert!(d1.is_full());
        assert!(d2.is_full());
        match d1.offer(stub(9)) {
            Err((Push::Full, buf)) => assert_eq!(buf.ts().val, 9),
            Ok(()) => panic!("expected Full, offer succeeded"),
            Err((status, _)) => panic!("expected Full, got {status:?}"),
        }
        assert_eq!(buffer_pts(&*d1), Vec::<i64>::new());
        assert_eq!(buffer_pts(&*d2), Vec::<i64>::new());
        assert_eq!(buffer_pts(&*buf), vec![0]);
    }

    #[test]
    fn writable_wait_on_head_wakes_when_tail_drains() {
        let (d1, d2, buf, _a, _b) = two_hop_direct_chain();
        assert_eq!(buf.push(stub(0)), Push::Accepted);
        assert!(d1.is_full());
        assert!(d2.is_full());

        let head: &dyn Edge = d1.as_ref();
        let mut ready = std::pin::pin!(head.wait_writable());
        let woken = Arc::new(AtomicBool::new(false));
        let waker = recording_waker(woken.clone());
        let mut cx = Context::from_waker(&waker);
        assert_eq!(ready.as_mut().poll(&mut cx), TaskPoll::Pending);
        assert!(
            !woken.load(Ordering::SeqCst),
            "writable wait must stay pending while the Buffered tail is full"
        );

        assert!(buf.try_take().is_some());
        assert!(
            woken.load(Ordering::SeqCst),
            "draining the tail must wake a waiter registered on the head Direct"
        );
        assert!(!d1.is_full());
        assert!(!d2.is_full());
        assert_eq!(ready.as_mut().poll(&mut cx), TaskPoll::Ready(()));
    }

    #[test]
    fn closed_offer_is_closed() {
        let hop = DirectEdge::new();
        hop.push_event(EdgeEvent::Eof);
        match hop.offer(stub(1)) {
            Err((Push::Closed, _)) => {}
            Ok(()) => panic!("expected Closed, offer succeeded"),
            Err((status, _)) => panic!("expected Closed, got {status:?}"),
        }
    }
}
