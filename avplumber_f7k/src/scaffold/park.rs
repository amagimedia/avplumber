//! The wait a blocking node does when its output is full, and the push loop
//! built on it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::graph::edge::{Edge, EdgeWaker, Push};
use crate::graph::error::NodeError;
use crate::graph::media::Media;

/// A blocking node's own condvar: what it waits on when it cannot proceed, and
/// what an edge wake or [`Node::interrupt`](crate::graph::node::Node::interrupt)
/// signals.
///
/// A Poll node that runs out of room calls `ctx.wait_writable(edge)` and
/// *returns* — the executor does the sleeping. A blocking body is a plain
/// closure on its own thread ([`BlockingExecutor`](crate::exec::BlockingExecutor)),
/// so it has to block itself, and it has two places to do that:
///
/// - **waiting for input** — `edge.take(-1)` blocks on the edge's own condvar,
///   and the executor can break it: `stop` interrupts each of the node's
///   *source* edges;
/// - **waiting for room on its output** — nothing in the framework blocks here.
///   [`Edge::offer`] is non-blocking and answers [`Push::Full`], and the
///   executor never interrupts a node's *sink* edges, so the output edge cannot
///   carry the stop signal either.
///
/// `Park` covers the second case: wait until my output changed, or until I am
/// told to stop. [`push_blocking`] is the retry loop built on it; `input` also
/// uses it as an interruptible sleep for its `stop_delay` countdown, so a stop
/// lands in milliseconds instead of after the whole delay.
///
/// Two fields, two kinds of signal. `woken` is edge-triggered and consumed by
/// [`Self::wait`]: it is the condvar predicate, and it is what closes the window
/// between "`offer` returned `Full`" and "`wait`" — a wake landing in between
/// sets it, so the waiter returns at once instead of sleeping through a wakeup
/// that already happened. `interrupted` is sticky level state, readable without
/// waiting via [`Self::is_interrupted`] and cleared only by [`Self::reset`], so a
/// node interrupted while parked finishes instead of retrying.
///
/// Held as an `Arc` **beside** the node's `Mutex<State>`, never inside it, for
/// two independent reasons: the waker handed to [`Edge::notify_writable`] is
/// owned by the *edge* and fires from whichever thread reaches the far end, and
/// [`Node::interrupt`](crate::graph::node::Node::interrupt) runs on the control
/// thread while the node's own thread sits parked *holding* its state lock — a
/// park reachable only through that lock would deadlock the very stop it exists
/// to deliver.
#[derive(Default)]
pub struct Park {
    woken: Mutex<bool>,
    cv: Condvar,
    interrupted: AtomicBool,
}

/// What lets an edge wake a [`Park`]: edges wake through [`EdgeWaker`], and this
/// is the newtype that makes a park one.
struct ParkWaker(Arc<Park>);

impl EdgeWaker for ParkWaker {
    fn wake(&self) {
        self.0.wake();
    }
}

impl Park {
    /// Installs this park as `edge`'s writable wake. Call before the attempt
    /// that may fail, so a wake landing between the attempt and [`Self::wait`]
    /// is not lost — the callback is one-shot, hence re-armed every round.
    ///
    /// One-shot because the edge *takes* the stored waker when it fires
    /// (`take_writable_cb`). Hence the `Arc`: the box the edge keeps has to own a
    /// handle on this park, which outlives the call.
    pub fn arm_writable(self: &Arc<Self>, edge: &Arc<dyn Edge>) {
        edge.notify_writable(Box::new(ParkWaker(self.clone())));
    }

    pub fn wake(&self) {
        *self.woken.lock().unwrap() = true;
        self.cv.notify_all();
    }

    /// Waits for a wake, at most `timeout_ms`. The timeout is a backstop: it
    /// bounds how long a missed wake can stall a node, so `stop` still lands.
    pub fn wait(&self, timeout_ms: u64) {
        let mut woken = self.woken.lock().unwrap();
        if !*woken {
            let (guard, _) = self
                .cv
                .wait_timeout(woken, Duration::from_millis(timeout_ms))
                .unwrap();
            woken = guard;
        }
        *woken = false;
    }

    /// From [`Node::interrupt`](crate::graph::node::Node::interrupt): any thread,
    /// any time, and deliberately without touching the node's state lock — see
    /// the type's docs.
    pub fn interrupt(&self) {
        self.interrupted.store(true, Ordering::Release);
        self.wake();
    }

    pub fn is_interrupted(&self) -> bool {
        self.interrupted.load(Ordering::Acquire)
    }

    /// From `Node::start`, since a node instance may be started again.
    pub fn reset(&self) {
        self.interrupted.store(false, Ordering::Release);
        *self.woken.lock().unwrap() = false;
    }
}

/// How a [`push_blocking`] ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pushed {
    /// The edge took it (or deliberately discarded it).
    Ok,
    /// The node was interrupted while parked; the buffer is dropped.
    Interrupted,
    /// The edge is closed — downstream is gone, so the node is done.
    Closed,
}

/// Push from a blocking body, parking for room instead of dropping.
///
/// `while_parked` runs on every wake, before retrying: that is where a producer
/// answers [hints](crate::graph::EdgeHint) posted by its consumer. Without it, a
/// full edge plus a consumer waiting for an answer only the producer can give
/// would deadlock.
pub fn push_blocking(
    park: &Arc<Park>,
    edge: &Arc<dyn Edge>,
    buffer: Media,
    mut while_parked: impl FnMut() -> Result<(), NodeError>,
) -> Result<Pushed, NodeError> {
    let mut buffer = buffer;
    loop {
        if park.is_interrupted() {
            return Ok(Pushed::Interrupted);
        }
        park.arm_writable(edge);
        match edge.offer(buffer) {
            Ok(()) => return Ok(Pushed::Ok),
            Err((Push::Full, returned)) => buffer = returned,
            Err((Push::Closed, _)) => return Ok(Pushed::Closed),
            // `Dropped` means the edge took the buffer and discarded it, so
            // there is nothing left to retry with; `Accepted` never comes back
            // as an error.
            Err((Push::Dropped | Push::Accepted, _)) => return Ok(Pushed::Ok),
        }
        while_parked()?;
        park.wait(PARK_TIMEOUT_MS);
    }
}

/// Long enough to be a backstop rather than a poll loop; short enough that a
/// missed wake costs a step, not a stall.
pub const PARK_TIMEOUT_MS: u64 = 50;
