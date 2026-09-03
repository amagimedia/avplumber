# avplumber Rust-native framework core

C ABI in `include/`. Read [`ARCHITECTURE.md`](ARCHITECTURE.md) before changing
the core.

## Improvements over C++ avplumber

* edge != queue. Other types of edges supported for direct data transfer without buffering.
* named in/out pads (`PadDecl` / `NodePads`) are ports on the node; topology
  is pad → edge → pad. C++ mostly used the named queue (`src`/`dst` →
  `EdgeManager`) as the port. `In<T>`/`Out<T>` exist as typed edge wrappers
  but are unused; live binding is still `bind_source`/`bind_sink` of an `Edge`.
* written in Rust so hopefully no more random memory corruption crashes (?)

## Current state

This crate currently exposes the node-independent substrate for review:

- owned `Ts` / `Spec` / `Media`
- `NodeBody::{Blocking, Poll, Async}` taken once at start
- buffered/direct edges with readiness wakeups
- blocking executor (one OS thread per node) and cooperative
  `AsyncExecutor` (one current-thread Tokio runtime per event-loop /
  tick-source; enable `--features async`)
- `SyncGroup` atomic clock snapshots, `CorrectionGroup` state (shared
  expected timeline + per-member cursors, no Sentinel policy)
- named `BuildCtx` services and script-compatible `node.add` / `src` /
  `dst` / `queue.plan_capacity` (TCP deferred)
- supervisor groups with `restart_group`, generation-fenced edges, and
  Direct hops that store no media after `offer` returns

## Parked / adapters

Parked until after this review: seek helper, pyplumber notes, C++
`node_compat` shim, and the old Tier-R node sketches (`parked/`).
`SisoNode` is in `src/scaffold.rs` as a one-in/one-out transform, with
blocking, poll, and async wrappers.
