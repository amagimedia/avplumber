//! Optional helpers for writing nodes: convenience, never contracts.
//!
//! [`Node`](crate::graph::node::Node) is the whole runtime contract, and
//! nothing here appears in any of its signatures — a node that implements the
//! trait by hand loses nothing. What lives here is the machinery every node
//! would otherwise reimplement:
//!
//! - [`blocking`] — [`BlockingNode`], the fallible `step` of a node with its own
//!   thread, [`Blocking`], the `Node` wrapper that supplies name, kind, pads,
//!   binding, park reset and interrupt from the node's [`BlockingIo`], and
//!   [`Park`], the condvar that body waits on when its output has no room;
//! - [`single_input`] — [`SingleInput`], the blocking node with one input
//!   written as [`InputHandler`] reactions (`on_spec`, `on_buffer`, `on_flush`,
//!   `on_eof`) instead of a `take` loop, and [`react`], the classification
//!   every consumer of an edge would otherwise repeat;
//! - [`poll`] — the same pair for a cooperative node, [`PollNode`] and
//!   [`Polling`], with the opt-in to the infallible Direct path;
//! - [`poll_input`] — [`PollInput`], the Poll sibling of [`SingleInput`]:
//!   [`InputHandler`] reactions plus an output stash on [`PollIo`];
//! - [`io`] — [`Io`] and [`EdgeSlot`], the name and the two edge slots a node
//!   keeps;
//! - [`siso`] — the linear single-input / single-output transform (C++
//!   `NodeSISO`): [`SisoNode`] plus one wrapper per executor kind, built on the
//!   above. `process` is 1:N; `on_eof` drains; an unchanged spec is skipped.
//! - [`objects`] — [`NodeObjects`], named knobs for the control thread. Not a
//!   grain adapter: a node that has keys implements it, whatever its body.
//!
//! Choosing between the node traits: a body that calls libav, or blocks on
//! `take(-1)`, is blocking — a [`SingleInput`] if it has one input, a
//! [`BlockingNode`] if it has none or several. A body that only shuffles
//! buffers and can always say what it is waiting for is a [`PollNode`]. With
//! one input transform, write [`SisoNode`] and pick an adapter; [`PollInput`]
//! is the cooperative loop when the node is not a transform. Direct
//! infallibility is a poll-body promise. Knobs are [`NodeObjects`], not those
//! traits.
//!
//! All of it is `pub`: the media nodes live in their own crate
//! (`avplumber_nodes`), and so does anyone else's node set.
//!
//! Contrast [`NodePollContext`](crate::graph::poll_ctx::NodePollContext), which
//! lives in [`graph`](crate::graph) precisely because it *is* part of a
//! contract: it appears in `NodeBody::Poll`'s signature.

pub mod blocking;
pub mod io;
pub mod objects;
pub mod poll;
pub mod poll_input;
pub mod single_input;
pub mod siso;

pub use blocking::{
    Blocking, BlockingIo, BlockingNode, PARK_TIMEOUT_MS, Park, Parked, Pushed, push_blocking,
};
pub use io::{EdgeSlot, Io, flush_at_head};
pub use objects::NodeObjects;
pub use poll::{PollNode, Polling};
pub use poll_input::{DrainPending, PollInput, PollIo};
pub use single_input::{EofAction, InputHandler, Reaction, SingleInput, react};
pub use siso::{SisoAdapter, SisoAsyncAdapter, SisoNode, SisoPollAdapter};
