//! Optional helpers for writing nodes: convenience, never contracts.
//!
//! [`Node`](crate::graph::node::Node) is the whole runtime contract, and
//! nothing here appears in any of its signatures — a node that implements the
//! trait by hand loses nothing. What lives here is the machinery every node
//! would otherwise reimplement:
//!
//! - [`siso`] — the linear single-input / single-output case (C++ `NodeSISO`):
//!   [`SisoNode`] plus one wrapper per executor kind;
//! - [`park`] — [`Park`], the condvar a *blocking* body waits on when its output
//!   has no room, and [`push_blocking`], the retry loop built on it;
//! - [`body`] — [`BlockingStep`]/[`PollStep`], the fallible step a node writes
//!   instead of an infallible [`NodeBody`](crate::graph::node::NodeBody);
//! - [`edge_slot`] — [`EdgeSlot`], one rebindable edge.
//!
//! All of it is `pub`: the media nodes live in their own crate
//! (`avplumber_nodes`), and so does anyone else's node set.
//!
//! Contrast [`NodePollContext`](crate::graph::poll_ctx::NodePollContext), which
//! lives in [`graph`](crate::graph) precisely because it *is* part of a
//! contract: it appears in `NodeBody::Poll`'s signature.

pub mod body;
pub mod edge_slot;
pub mod park;
pub mod siso;

pub use body::{BlockingStep, PollStep, blocking_body, poll_body};
pub use edge_slot::EdgeSlot;
pub use park::{PARK_TIMEOUT_MS, Park, Pushed, push_blocking};
pub use siso::{SisoAdapter, SisoAsyncAdapter, SisoNode, SisoPollAdapter};
