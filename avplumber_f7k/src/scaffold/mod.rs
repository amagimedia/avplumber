//! Optional helpers for writing nodes: convenience, never contracts.
//!
//! [`Node`](crate::graph::node::Node) is the whole runtime contract, and
//! nothing here appears in any of its signatures — a node that implements the
//! trait by hand loses nothing. What lives here is the machinery every node
//! would otherwise reimplement:
//!
//! - [`blocking`] — [`BlockingNode`], the fallible `step` of a node with its own
//!   thread, and [`Blocking`], the `Node` wrapper that supplies name, kind,
//!   pads, binding, park reset and interrupt from the node's [`BlockingIo`];
//! - [`single_input`] — [`SingleInput`], the blocking node with one input
//!   written as [`InputHandler`] reactions (`on_spec`, `on_buffer`, `on_flush`,
//!   `on_eof`) instead of a `take` loop, and [`react`], the classification
//!   every consumer of an edge would otherwise repeat;
//! - [`poll`] — the same pair for a cooperative node, [`PollNode`] and
//!   [`Polling`], with the opt-in to the infallible Direct path;
//! - [`io`] — [`Io`] and [`BlockingIo`], the name and the two edge slots (plus
//!   the park) a node keeps, with the `require`/`error`/`push` helpers on them;
//! - [`siso`] — the linear single-input / single-output transform (C++
//!   `NodeSISO`): [`SisoNode`] plus one wrapper per executor kind, built on the
//!   above;
//! - [`park`] — [`Park`], the condvar a *blocking* body waits on when its output
//!   has no room, and [`push_blocking`], the retry loop built on it;
//! - [`edge_slot`] — [`EdgeSlot`], one rebindable edge.
//!
//! Choosing between the node traits: a body that calls libav, or blocks on
//! `take(-1)`, is blocking — a [`SingleInput`] if it has one input, a
//! [`BlockingNode`] if it has none or several. A body that only shuffles
//! buffers and can always say what it is waiting for is a [`PollNode`], which
//! is also the only kind that can promise infallibility and fuse across a
//! Direct edge.
//!
//! All of it is `pub`: the media nodes live in their own crate
//! (`avplumber_nodes`), and so does anyone else's node set.
//!
//! Contrast [`NodePollContext`](crate::graph::poll_ctx::NodePollContext), which
//! lives in [`graph`](crate::graph) precisely because it *is* part of a
//! contract: it appears in `NodeBody::Poll`'s signature.

pub mod blocking;
pub mod edge_slot;
pub mod io;
pub mod park;
pub mod poll;
pub mod single_input;
pub mod siso;

pub use blocking::{Blocking, BlockingNode};
pub use edge_slot::EdgeSlot;
pub use io::{BlockingIo, Io, flush_at_head};
pub use park::{PARK_TIMEOUT_MS, Park, Parked, Pushed, push_blocking};
pub use poll::{PollNode, Polling};
pub use single_input::{InputHandler, Reaction, SingleInput, react};
pub use siso::{SisoAdapter, SisoAsyncAdapter, SisoNode, SisoPollAdapter};
