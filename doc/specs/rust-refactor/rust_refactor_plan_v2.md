# Rust Core Refactor Plan for avplumber — v2

> Supersedes `rust_refactor_plan.old.md`, which is kept only for reference.
>
> Written after reading: `graph_core.hpp`, `graph_base.hpp`,
> `graph_interfaces.hpp`, `graph_mgmt.{hpp,cpp}`, `EventLoop.hpp`, `TickSource.hpp`,
> `edge_types.hpp`, the four Team headers (`RealTimeTeam`, `PauseControlTeam`,
> `SpeedControlTeam`, `InputSeekTeam`), and the scheduler-adjacent nodes
> (`input_rec.cpp`, `realtime.cpp`, `speed.cpp`, `pause.cpp`, `rescale_video.cpp`).
>
> **Companion documents.** `rust_refactor_native_core.md` specifies the core's
> internal data model, the node contract, and the native embedder surface — this
> document specifies the C ABI that is a compat layer over it, plus the edge/control
> model, scheduling, services and phasing. `rust_refactor_impl_breakdown.md` is the
> build sheet (node tiers, milestones, effort); `rust_refactor_headers.md` holds the
> draft headers. Blockquoted notes below point at the native-side detail for
> sections where both documents describe the same mechanism from two sides.

---

## 0. The load-bearing decisions

Each is justified in the section noted:

1. **Capability discovery splits into three registers, and the upstream walk is
   deleted.** The graph is threaded with ~30 optional capabilities discovered today
   by `dynamic_cast` + `findNodeUp<T>()` (a per-call upstream walk). Reading the call
   sites, they are three different jobs: *Facts* (format) become `Spec` latched
   in-band on the edge; *Controls* (direction/reset/seek) are addressed to core
   services; only *live queries* on an already-known node keep an explicit
   `query_interface(node, id)`. No graph walk survives — which also kills the
   `input_hold_` lifetime hack. (§4.4, §4.4.1, §6)

2. **Causal dataflow: source-time buffers + a shared master clock read at the
   output.** Edges are pipes carrying buffers (PTS never rewritten in transit) plus
   a minimal set of causal in-band control tokens (EOF, flush/seek, format). Rate,
   offset and pause live on the master clock (`SyncGroup`) and are applied only at
   the output — not threaded through the graph. This deletes `IFlushAndSeek`, the
   4-phase seek handshake, the per-edge `flushing_`/`flushed_` flags, the
   `speedChanged` reach-back, and the `AVFrame::metadata` timeline smuggling. Format
   is recovered from the edge's latched `Spec` (§4.4.1), not a pull walk, which is
   what makes node restart during live op safe. (§3, §3.6, §4.4, §6)

3. **The four Teams and the scheduler-adjacent "nodes" become core Rust
   services**, not ported/shimmed C++ nodes. `RealTimeTeam` *is* the sync + seek
   coordinator; `realtime`/`speed`/`pause` are thin members of it. (§6)

4. **Scheduling is "hybrid OS threads + a tokio current-thread runtime per clock
   domain, tick-as-event."** Blocking codec/IO/CUDA calls need real threads; the
   clock/tick path needs jitter-free timing. The tick is delivered as a
   `tokio::sync::Notify` event (not `tokio::time`), so tokio's timer wheel never
   sits on the frame-perfect path; `tokio::time` is opt-in with a jitter caveat.
   Async/await is what makes *stateful non-blocking* nodes tractable — the big
   boilerplate + safety win — and it brings the full tokio I/O + sync ecosystem
   with it. It is **not** the default: strategy follows what a body does, and
   synchronous FFmpeg/CUDA work is `Blocking` whichever language wrote it. (§5)

5. **Run the *existing C++ nodes* on the Rust core to parity first**, before
   porting any node to Rust. The core swap is the one monolithic risky step; node
   porting is naturally incremental (strangler-fig). (§10)

6. **Raw `AVFrame*`/`AVPacket*` cross the FFI, and avcpp is reconstituted inside
   the C++ shim** — not avcpp objects across the boundary. Validated by `.raw()`
   being used ~390× in nodes: they already operate at the raw-struct level. (§2, §8)

7. **pyplumber re-bridging is an explicit workstream**, not an afterthought. (§9)

8. **The C ABI is a compat layer over a native Rust core, not the core's internal
   model.** Internally the edge carries owned rsmpeg `AVFrame`/`AVPacket`
   (refcounting by `Drop`), `Spec` is a rich enum, and timestamps are a
   `Ts{val, tb}` type; the flat `#[repr(C)]` `AvpBuffer`/`AvpSpec` exist only in
   `abi/`, and conversion happens in one `FfiNode` adapter via
   `into_raw`/`from_raw` (a pointer move — identical runtime representation, no
   copy, no refcount op). The ABI co-evolves with the core: in-tree C++ nodes and
   pyplumber are recompiled together, with **no cross-version ABI freeze and no
   third-party `dlopen` plugin contract**. Native Rust embedders also get a typed
   surface (graph builder, serde-deserialized params, `Arc<dyn SyncGroup>`).
   Governing constraint: **one substrate contract, two adapters** — never two
   parallel node contracts. (`rust_refactor_native_core.md`)

9. **The C ABI is a first-class *embedder* interface, not just a node-call
   interface.** Graph management (create node, connect, start/stop group,
   destroy) is exposed as ABI calls (§4.7); the TCP control server is a thin
   JSON wrapper over them. This is required by avplumber's own pyplumber (§9)
   and OBS-embed (§12 item 4) plans — an embedder drives the core via the ABI without
   speaking the line protocol. Domain services (SyncGroup, Teams, outputs,
   hwaccel) are **not** named functions in the core header — they're behind two
   discovery mechanisms (`avp_node_query_interface` for node capabilities,
   `avp_core_query_service` for core services), keeping `avplumber_core.h`
   framework-only with domain vtables in separate headers (§4.8).
   (§4.7, §4.8, §10 Phase 0, §12 item 6)

10. **`SyncGroup` is a trait, not a struct.** The core owns the playback→wall
    mapping contract; the implementation is pluggable (`WallClock` for live,
    `SourceTimeClock` for deterministic offline render, `SyntheticClock` for
    tests). Justified by avplumber's own offline-render and testability needs,
    not by a hypothetical embedder. (§3.3, §6)

11. **The Rust code structure separates substrate, executor, and supervisor.**
    The substrate (`graph/`, = `graph_core.hpp`) is engine-agnostic and *must not
    depend on `exec/`* — fixing the C++ leak where `NonBlockingNodeBase` references
    `EventLoop`. The executor (`exec/`, = `graph_mgmt`'s `NodeWrapper`+`EventLoop`+
    `TickSource`) is a **set-level**, pluggable abstraction (not per-node): a runtime
    that owns a set of co-located nodes and schedules them. The supervisor
    (`supervisor/`, = `NodeGroup`) is **fixed** lifecycle (state machine, topo sort,
    auto_restart) and drives nodes via the `Executor` trait. Inter-executor routing
    goes through the substrate-level edge `Wakeup`, not executor-to-executor. This is
    the seam a future `FixpointExecutor` (§5.5, for a non-causal host) would use
    without rewriting `graph/` or `supervisor/`. (§5.4, §5.5)

---

## 1. Feasibility verdict

**Feasible. The hard parts are interface discovery and the Teams/seek machinery,
not the FFI or avcpp.**

- Rust↔C++ has no stable ABI; both speak C cleanly. Everything below lives on a C
  ABI (`avplumber_core.h`).
- avcpp stays entirely C++-side. Rust never interprets frame semantics; it moves
  opaque `AVFrame*`/`AVPacket*` and reads a few scalar fields via FFmpeg C
  functions (`av_frame_get_*`, direct struct reads through `rusty_ffmpeg`).
- Per-frame FFI overhead is negligible (~100 calls/s per node for a stereo A/V
  stream) — this is exactly the article's "templates were premature optimization"
  point; runtime dispatch is free at this cadence.

**Effort (honest):** MVP core + shim + a handful of nodes ≈ 2–3 months for 2–3
devs. *Full parity* (98 node files, 4 Teams, `input_rec`'s seek tables, pyplumber,
CUDA/GL/TensorRT, OBS embed) is materially longer. Treat it as MVP-then-strangler,
never a big-bang parity target.

**Top risks, ranked:**
1. Capability discovery across the boundary (mitigated by §4.4's three-register
   split: `Spec` in-band for Facts, services for Controls, direct `query_interface`
   for live queries — no upstream walk).
2. The Teams + 4-phase seek being replaced without behavior regressions on live
   streams (mitigated by §3/§6's causal dataflow model + the §10 parity-first ordering).
3. `.raw()` refcount correctness through the shim (mitigated by §2/§8's ownership
   contract).

---

## 2. Data model: raw frames across the FFI

> **Scope.** This section describes the **ABI** payload — the flat struct that
> crosses the boundary. It is not the core's *internal* payload: the edge queue
> holds an owned `Media` enum (rsmpeg `AVFrame`/`AVPacket`, plus an RAII `Opaque`
> variant for `EglImageFrame`/`MetadataFrame`), and `AvpBuffer` appears only in
> `abi/`. See `rust_refactor_native_core.md` §2.

### 2.1 The edge payload

C++'s `Edge<T>` template specialization over 5 types (`av::Packet`,
`av::VideoFrame`, `av::AudioSamples`, `EglImageFrame`, `MetadataFrame`) collapses
to a **runtime tagged union**:

```c
typedef enum {
    AVP_MEDIA_PACKET      = 1,  // AVPacket*
    AVP_MEDIA_VIDEO       = 2,  // AVFrame*
    AVP_MEDIA_AUDIO       = 3,  // AVFrame*
    AVP_MEDIA_EGL         = 4,  // opaque EglImageFrame* (C++-owned)
    AVP_MEDIA_METADATA    = 5,  // opaque MetadataFrame* (C++-owned)
} AvpMediaType;

typedef struct {
    AvpMediaType type;
    void*        ptr;      // AVFrame*/AVPacket* for 2/3/1, opaque for 4/5
} AvpBuffer;             // PTS is on the AVFrame/AVPacket, in source time (§3.3).
                          // No epoch: seek clears queues rather than tagging them (§3.4).
```

- For `PACKET/VIDEO/AUDIO`, `ptr` is a genuine FFmpeg C object. Rust owns a
  reference via `av_frame_ref`/`av_packet_ref` on enqueue and
  `av_frame_free`/`av_packet_free` on drop. Rust reads `pts/dts/duration/time_base`
  through `rusty_ffmpeg` (either direct struct access or the `av_*` accessors).
- For `EGL/METADATA`, `ptr` is a C++-owned opaque object. Rust treats it as a
  `void*` with a type-erased retain/release pair supplied by the C++ side (two
  function pointers registered once per media type — see `AvpMediaVtable` in §4).
  This keeps GL/DMA-BUF and custom metadata frames working without Rust
  understanding them.

**Why raw pointers, not avcpp objects:** `AVFrame`/`AVPacket` have a stable,
FFmpeg-guaranteed C layout; `av::VideoFrame` has a C++-ABI-dependent layout. And
nodes already reach through `.raw()` constantly — **391 occurrences over 344 lines
in 63 files** under `src/nodes/`, 416 across all of `src/`
(`grep -roE '(\.|->)raw\(\)' src/nodes/ | wc -l`) — so `.raw()` returning the exact
pointer Rust passed around is *alignment*, not friction.

### 2.2 avcpp lives in the shim (see §8)

The C++ node shim wraps an incoming `AvpBuffer.ptr` into the appropriate avcpp
object such that `.raw()` returns that same pointer and refcount stays balanced.
On output it extracts `.raw()` and hands the pointer back. Existing node bodies —
including `av::VideoRescaler`, side-data walks, `av_dict_copy` — are unchanged.

**Three fields do not live in the `AVFrame`/`AVPacket` struct and must be
bridged explicitly by the adapter.** avcpp keeps `m_timeBase`, `m_streamIndex` and
`m_isComplete` in the C++ wrapper (`deps/avcpp/src/frame.h:163-172`) and — verified
by grep — **never reads or writes the native `time_base` field at all**. So:

1. **Timebase.** A native node trusting `frame.time_base` on a C++-produced frame
   reads `{0,0}`, not the real timebase. The adapter bridges `m_timeBase` ↔ native
   `time_base` in both directions. The native field is otherwise unclaimed, so the
   core adopts it as the authoritative carrier (§3.3).
2. **`isComplete`.** Defaults to `false` (`frame.h:354`), and `avutils.hpp:172`
   gates on `isComplete() && pts().isValid()`. An adapter that forgets
   `setComplete(true)` produces frames C++ nodes **silently discard** — no error,
   no output. The adapter sets it on every real frame outbound — but **not
   unconditionally**: incompleteness is how EOF/flush is encoded today
   (`createEofMarker<T>()` is a default-constructed `T`, `avutils.hpp:155-157`, read
   as "flush frame" at `encoders.cpp:146` and `decoders.cpp:289`). On `Eof` the
   adapter must synthesize that marker instead. See `rust_refactor_native_core.md`
   §3.4 for the exact shape (including the 1-byte NoPts packet case) — missing it
   hangs the pipeline at end of stream.
3. **Frame `streamIndex` is dead** — its only two reads (`sentinel.cpp:611`, `:707`)
   are commented out, so it need not cross. Packet `stream_index` is native
   (`packet.cpp:267`) and round-trips for free.

Also note avcpp's `setTimeBase` is a **rescale, not a setter** (`frame.h:228`), and
is asymmetric: `Frame::setTimeBase` rescales `pts`/`best_effort_timestamp` but not
`duration`, while `Packet::setTimeBase` calls `av_packet_rescale_ts`, which does
(`packet.cpp:334`). Per-node check on port; do not silently "fix" it. Detail in
`rust_refactor_native_core.md` §3.

---

## 3. Edge model: buffers + causal control (the keystone)

This is the central design change. It replaces: EOF markers, `IFlushAndSeek`, the
4-phase seek, per-edge flush flags, in-flight PTS rescaling, the `speedChanged`
upstream walk, and the `AVFrame::metadata` timeline smuggling in `input_rec`.

### 3.1 The model: dataflow, not a graph engine

The mental model is **unix pipes / a modular synthesizer, not GStreamer.** An edge
is a pipe carrying an ordered stream of buffers with occasional in-band control
tokens. **Control is causal:** a control token takes effect *where and when it
arrives*, exactly like bytes in a pipe or a control-voltage change in a synth.
**Nothing is ever applied retroactively to buffers already downstream.** The entire
apparatus in the C++ core that exists to *fake* instantaneous, retroactive control —
the RTTI graph walks, the Team singletons, the `speedChanged` reach-back that
re-rescales queued frames — is deleted, not ported.

The design has exactly three moving parts, and no more:

1. **Data plane** — buffers carrying **source-time PTS, never rewritten in
   transit.** A frame's PTS means the same thing at every hop from demux to sink.
2. **In-band control** — a small set of causal tokens on the pipe: end-of-stream,
   the seek/flush discontinuity, and format changes. That's it.
3. **One shared primitive: the master clock** (`SyncGroup`, §6). It owns the
   playback→wall mapping — offset, rate, pause state — and is read **at the output**
   for A/V sync and reset on seek. This is the synth's master clock: one shared time
   source that parallel pipes agree on. It is the *only* thing media forces on us
   that plain pipes lack, and it exists solely for cross-stream sync and coherent
   seek.

The codebase already proves the shape works: EOF is *already* an in-band token
(`createEofMarker`/`isEofMarker`). We generalize that one working mechanism and
throw the out-of-band machinery away.

### 3.2 What travels on an edge

```rust
enum EdgeItem {
    Buffer(AvpBuffer),   // source-time PTS, never rewritten in transit
    Event(EdgeEvent),
}

enum EdgeEvent {
    /// End of stream. A node flushes internal state, forwards Eof, finishes.
    Eof,

    /// Seek / discontinuity. Preempts the pipe: clears queued Buffers and flushes
    /// internal codec/filter state downstream, immediately. See §3.4.
    FlushStart,
    FlushStop,

    /// Stream format description (video: w/h/pixfmt; audio: rate/fmt/layout;
    /// frame rate; time base). Sent before the first Buffer and whenever the
    /// format changes. Latched on the edge: a (re)connecting consumer is handed
    /// the current Spec as the head item before any Buffer — no pull walk. See
    /// §4.4.1 and §3.6.
    Spec(StreamSpec),
}
```

There is **no `Segment` event and no per-buffer epoch.** Rate and offset are not in
the stream at all — they live on the master clock (§3.3). Format is causal and
in-band but not cached/replayed (§3.6). This is the whole event set.

### 3.3 Timestamps and the master clock (what replaces `Segment`)

> **Timestamps are a type, not a bare `i64`, and the buffer carries its
> own authoritative timebase.** rsmpeg supplies **no** timestamp abstraction —
> `avutil/timestamp.rs` is two formatting functions over `(i64, AVRational)`, and
> field access is a bare `i64` via `Deref`. Since avcpp *does* have
> `av::Timestamp{value, timebase}`, porting C++ → Rust would otherwise regress to
> untyped integers. The core therefore defines `Ts { val: i64, tb: AVRational }`
> (12 bytes, `Copy`), read/written through `FrameExt::ts()`/`set_ts()` which touch
> `pts` and `time_base` together so they cannot desync. `Ts::rescale()` is a
> separate, explicitly-named operation — never implicit. `Ts` does **not** implement
> `Add`/`Sub` across differing timebases, making a scale mix-up a compile error
> rather than a plausible wrong number.
>
> Two specifics: `AVRational` derives only `Debug, Copy, Clone` — **no
> `PartialEq`** — so `Ts` equality must compare as rationals (`av_cmp_q`), or
> `{1,25}` and `{2,50}` would compare unequal. And the **buffer** carries the
> authoritative timebase in the native `AVFrame`/`AVPacket` `time_base` field, while
> `Spec` *declares* the expected one (a contract to check against, not the carrier)
> — because 37 avcpp `setTimeBase` sites change timebase per buffer, and routing
> that through `Spec` alone would require one `Spec` event per frame plus
> queue-order reconstruction by every consumer. `Media::Opaque` has no `AVFrame`, so
> that variant carries `time_base` explicitly. Detail:
> `rust_refactor_native_core.md` §2.3, §3.

**PTS on edges is always source time, and no node rewrites it.** Decode, filter,
scale, split, mux — every node passes PTS through unchanged. The playback→wall
mapping happens in exactly one place: **the master clock, read at the output** when
a frame is released for presentation. Playback *rate is a property of that clock*,
not a value threaded through the graph. So:

- **Speed change** = change the clock's rate. O(1), at the clock. Every buffer
  already in flight keeps its source-time PTS and is simply mapped through the new
  rate when the output releases it. No in-band token, no queue drain, no reach-back.
  This is the "fuse speed into the output clock" resolution: in a synth, playback
  rate *is* the master clock — it belongs at the output stage, not mid-graph.
- **Pause** = freeze the clock. The output stops releasing; back-pressure propagates
  upstream through the pipes naturally. No pause handshake.

Contrast the C++, where `speed.cpp` bakes rate into the PTS mid-graph
(`rescaleFrameTS`→`setPts`) and therefore must **reach back and un-bake queued
frames** on every rate change — the ~120-line `speedChanged` upstream walk
(`lockProcessing()` on every node, re-rescaling in-flight PTSes, recovering the
original PTS via the `AVFrame::metadata` `frame_ts` dict and a side `scaled_pts_`
list). All of that existed only because the frame was the wrong place to store a
mutable timeline. Keep source time on the frame and the entire mechanism vanishes.

**Topology, not machinery, gives responsive control.** The one real cost of causal
control is that a change is delayed by whatever buffer sits between the control
point and the point of observation. That is a *topology* property. Two rules keep it
small, and they are design guidance, not framework features:

- **Keep heavy buffers upstream of the control point.** Put the jitter/source buffer
  *before* speed/pause; keep the path from there to the output shallow. Then even a
  purely in-band control change would be cheap — and the clock-based approach makes
  it free.
- **Fuse speed/pause into the output clock stage.** With no buffer between the knob
  and the clock, the change is O(1) by construction. This is why `speed`/`pause`/
  `realtime` become one output-clock **core service** (§6), not mid-graph nodes.

`sync_group` (which master clock a stream reads) and `ts_source` (which source
timeline the input stamps) are therefore **not** properties of a stream event. The
first is a node/edge configuration selecting a clock domain; the second is an
input-subsystem concern handled where timestamps originate (§6.3). Neither pollutes
the data plane.

**`SyncGroup` is a trait, not a struct.** The core owns the *contract* — the
playback→wall mapping (offset, rate, pause) read at the output — but the
implementation is pluggable, because avplumber itself runs under several clock
regimes that want different backends:

- `WallClock` — live streaming / real-time playback. The wall mapping is
  derived from a monotonic clock; this is the default and what `realtime`
  uses today.
- `SourceTimeClock` — offline render. The clock advances purely from
  source-time PTS, with no wall mapping. This is what makes a `script.avplumber
  → output file` render deterministic and reproducible (no wallclock, no
  real-time IO jitter), a real avplumber feature, not a future-embedder concern.
- `SyntheticClock` — test fixtures. A controllable clock for reproducible
  test runs without depending on wall time.

An embedder linking the core as a library (§4.7) may supply its own
implementation of the trait, but that is a side effect: the justification is
avplumber's own offline-render and testability needs. The §6 mapping is to the
*trait*, with `WallClock` as the default implementation that reproduces today's
behavior.

### 3.4 Seek = a flushing discontinuity

Seek is fundamentally different from speed: a speed change *preserves* buffers (that
was the trap — trying to rewrite them), whereas a seek *discards* them (jump to a
new position, throw away what was queued). Discarding is trivially safe — there is
no per-frame state to recover — so seek is allowed to **preempt the pipe.**

Old (imperative, RTTI, ordered 4-phase across `NodeSingleInput` + `RealTimeTeam`,
`graph_base.hpp:84-170`):

```
_start:    pauseProcessing() all upstream; startFlushing()+finishConsumer()+
           finishProducer() every upstream edge
flushAndSeek: lockProcessing()/unlockProcessing() barrier on every upstream node
_finish:   edge.clear() + stopFlushing() every upstream edge
_complete: dynamic_cast upstream to IDecoder::discardUntil, IPlaybackControl::
           seekAndPause, IInputReset::resetInput; resume all
```

New (a flush that clears queues + a clock reset, no pause handshake, no RTTI):

```
1. Source (input_rec's replacement) issues FlushStart downstream. Unlike a
   Buffer, a flush PREEMPTS: each edge it crosses clears its queued Buffers
   immediately (drop — no rewrite), and each node flushes its own internal
   state (decoder DPB, rescaler, FRUC buffers) where that state lives.
   This is why seek is instant and not delayed by queue depth: the discontinuity
   discards the pipe rather than waiting to drain it.
2. Source repositions (seek table lookup, byte/frame/ts resolution — the
   input_rec logic, now a core InputReader method, §6.3).
3. The master clock is reset to the new position (one shared reset for all
   streams in the sync_group — this is what keeps A/V coherent across a seek).
4. Source emits FlushStop, then fresh Spec, then Buffers at the new position.
   The output resumes mapping source-time PTS through the reset clock.
```

Because a flush clears queues on the way down and the clock reset is a single shared
operation, there is **no need for epoch tagging**: there are no stale buffers left in
flight to distinguish (they were dropped), and the output only maps buffers it
receives after `FlushStop` through the reset clock. Ordering between flush and the
buffers that follow it is intrinsic to the pipe. Epoch tagging is what you need when
stale buffers are allowed to linger; clearing on flush removes the problem instead of
tracking it.

### 3.5 Direct edges (zero-queue) fit naturally

An edge is an interface, not necessarily a queue:

- `BufferedEdge`: lock-free ring (today's `Edge<T>`, but carrying `EdgeItem`).
  Required across a thread boundary or where bursts must be absorbed.
- `DirectEdge`: no queue. Producing an item directly invokes the consumer's
  handler; backpressure is the return value (`Flow::Backpressure`). Used when
  producer and consumer share an executor.

This is the push/`chain()` model and it implements the article's wish to forward
backpressure across *non-adjacent* non-blocking nodes: a fused chain propagates it
by construction. **Hard invariant:** never fuse across a blocking boundary — a slow
consumer would stall the producer's OS thread. The scheduler chooses edge type
from whether the two nodes share an execution context (§5.3).

### 3.6 Node restart during live operation

The dataflow model keeps the C++'s accidental crash-tolerance instead of breaking
it. Today every frame re-carries its own truth: `rescale_video` re-probes format
per frame (`sourceChanged()`); `realtime` re-derives its offset from the shared
clock. So when `restart_node`/`restart_group` fires on error during live op (wired
as `onFinished` callbacks in `NodeManager::createNode`), the recreated node
reconnects to its named edges and picks state up — nothing was stored only-once
upstream, so nothing is missed.

We preserve this property by **not introducing sticky per-stream state on the
edge.** Concretely:

- **Timeline survives restart because it isn't on the edge.** Rate/offset/pause live
  on the master clock (§3.3), which is a core-owned service, not node state. A
  restarted node rejoins and reads the current clock exactly as `realtime` does
  today (`ready_=false` → converge) — no timeline to replay, nothing to miss.
- **Format is recovered from the edge latch, not a pull walk and not replay.** The
  edge latches the current `Spec` (§4.4.1); a (re)connecting consumer, on its first
  pop, is handed that latched `Spec` as the head item before any buffer, re-inits in
  `on_spec`, and proceeds. This is *local* current state of one pipe, not replayed
  global policy (contrast the deleted `Segment`/epoch). So there is no upstream
  interface walk to recover format (`findNodeUp` is deleted, §4.4) and no edge-side
  sticky cache/replay machinery.
- **Producer restart reuses the seek discontinuity.** A restarted producer self-
  issues `FlushStart` → reposition → `FlushStop` → fresh `Spec` (§3.4), clearing
  stale downstream buffers via the exact same flush mechanism. A consumer restart
  reads the latched `Spec` + current clock as above. A whole-group restart is both.

The result: no sticky-event cache, no replay-on-reconnect, no epoch bookkeeping —
the recovery story is "read the shared clock, pull current format," which is what
the working C++ already does per frame, just done once at (re)start instead of
repeatedly.

---

## 4. The C ABI: `avplumber_core.h`

Design rules: opaque handles; all ownership transfers stated explicitly; UTF-8
`const char*` for names; JSON params as strings (parsed with `serde_json` in Rust,
`nlohmann::json` in C++); no C++ types, no `std::` anything crosses the line.

> **What this ABI is for.** It serves **in-tree C++ nodes and pyplumber,
> recompiled together with the core** — it is a compat layer that co-evolves, not a
> frozen contract. There is deliberately **no cross-version ABI stability guarantee
> and no third-party `dlopen` plugin surface**; freezing it would pin the C++
> out-of-band seek and Team machinery into a permanent interface, which is precisely
> what this refactor deletes. Consequences: the native path is **lossless**, while
> the C path may be lossy where no C++ node reads the field (e.g. the
> `AvpSpec` channel-layout degradation in §4.4.1). Rust embedders use the native
> typed surface (`rust_refactor_native_core.md` §5) rather than this header; the C
> entry points are reimplemented on top of it, so existing calls keep working.

### 4.1 Handles and media vtables

```c
typedef struct AvpCore      AvpCore;       // the Rust core / one instance
typedef struct AvpNode      AvpNode;       // a node instance (Rust- or C++-backed)
typedef struct AvpEdge      AvpEdge;       // an edge endpoint
typedef struct AvpExecutor  AvpExecutor;   // a clock domain / executor (was EventLoop)
typedef struct AvpPeek      AvpPeek;       // a live borrow of an edge's head item
                                           // (§4.3); must be released or consumed
                                           // before the edge advances

// Retain/release for C++-owned opaque media (EGL, Metadata frames).
// Registered once per media type so Rust can move them without understanding them.
typedef struct {
    void (*retain)(void* obj);
    void (*release)(void* obj);
    int64_t (*get_pts)(void* obj);        // in get_time_base units, or AV_NOPTS
    void (*get_time_base)(void* obj, int* num, int* den);
} AvpMediaVtable;

void avp_register_media_type(AvpCore*, AvpMediaType, const AvpMediaVtable*);
```

`AVFrame*`/`AVPacket*` need no vtable — Rust calls FFmpeg C functions directly.

### 4.2 Node vtable

> **Scope.** This vtable is the **ABI**, consumed by the `FfiNode` adapter.
> The *internal* node contract is `trait Node` + `NodeBody`, where each node
> declares exactly one run strategy — `Blocking` / `Poll` / `Async` — as an enum
> variant carrying its body, instead of the current "leave `process` or `poll` NULL"
> convention (which lets a node implementing neither silently report EOF). Strategy
> follows **what the body does, not which language wrote it**: synchronous FFmpeg
> and CUDA calls are `Blocking`, so `Blocking` is the *default* for native Rust
> nodes too, and `Async` is for bodies awaiting several inputs or an input plus a
> clock. `FfiNode`'s required surface is small: only 6 C++ nodes are non-blocking
> (`force_fps`, `obs_video_sink`, `pause`, `realtime`, `smooth_timestamps`,
> `speed`), and 3 of those are Tier C (absorbed into core services), leaving
> exactly **3** `Poll` nodes. Internally `AvpFlow` splits into `Push` (edge-op
> result), `Tick`/`Blocked` (node-invocation result), and `Result<_, NodeError>` for
> errors — so `Backpressure` is unrepresentable as a node result and errors carry a
> message. `AvpFlow` itself stays one flat enum in the header. See
> `rust_refactor_native_core.md` §4.1–4.3.

The **mandatory** surface is small; the richness is in `query_interface` (§4.4).

```c
typedef enum { AVP_FLOW_PUSHED, AVP_FLOW_DROP, AVP_FLOW_BACKPRESSURE,
               AVP_FLOW_EOF, AVP_FLOW_ERROR } AvpFlow;

typedef struct {
    // Lifecycle
    void      (*start)(AvpNode*);
    void      (*stop)(AvpNode*);
    void      (*destroy)(AvpNode*);       // release the C++/Rust node object

    // Blocking nodes: called on a dedicated OS thread until it returns EOF/ERROR.
    // Non-blocking nodes leave this NULL and implement poll() instead.
    AvpFlow   (*process)(AvpNode*);

    // Non-blocking nodes: cooperative poll. `item` is the next input (may be an
    // Event). Returns BACKPRESSURE to be re-polled when the sink drains, DROP to
    // discard, PUSHED on success. NULL for blocking nodes.
    AvpFlow   (*poll)(AvpNode*, const AvpBuffer* item);

    // Capability discovery. Returns a const vtable pointer for `iface`, or NULL.
    const void* (*query_interface)(AvpNode*, AvpInterfaceId iface);
} AvpNodeVtable;
```

### 4.3 Edge ops (called by nodes, implemented by Rust)

```c
// Producer side
AvpFlow avp_edge_push(AvpEdge*, const AvpBuffer* buf);      // BACKPRESSURE if full
void    avp_edge_push_event(AvpEdge*, const AvpEdgeEvent*); // never drops

// Consumer side (non-blocking: timeout_ms==0 returns immediately).
// One dequeue pair serves both buffers and events: `AvpItem` is
// `{ is_event; buffer; event }`, because buffers and control tokens share one
// ordered queue (§3.2) — a flush must never be observed out of order with the
// buffers around it. Full declarations: `rust_refactor_headers.md`.
//
// peek is an acquire/release/consume triple over an opaque AvpPeek handle, because
// the borrow it hands out needs an enforced lifetime — the queue slot is destroyed
// in place by pop, so a bare `AvpBuffer` borrow dangles. The shim wraps this in a
// move-only C++ RAII guard; nodes never see the handle. Rationale, including the
// use-after-free this fixes in the existing C++ tree:
// rust_refactor_native_core.md §4.3.1.
AvpPeek* avp_edge_peek(AvpEdge*, int timeout_ms, AvpItem* out); // NULL = none;
                                                 // *out BORROWED until release/consume
void     avp_edge_peek_release(AvpPeek*);        // leave the item queued
// Pops the head. `out` is NULLABLE and that choice is the ownership decision:
//   out == NULL -> the core RELEASES its ref (plain pop). Use this when the caller
//                  already made its own ref from the borrow — which the avcpp shim
//                  does, since constructing av::VideoFrame from an AVFrame* refs it.
//   out != NULL -> the core MOVES its ref to *out; the caller must release it.
//                  Use this from plain C that never materialized a copy.
// Getting this backwards leaks (shim + move) or double-frees (no ref + release).
int      avp_edge_peek_consume(AvpPeek*, AvpBuffer* out /*nullable*/);
void    avp_edge_pop(AvpEdge*);
int     avp_edge_occupied(AvpEdge*);

// Consumer side, ownership-transferring: removes the head item and MOVES the core's
// reference to the caller, which must release it (or forward it via avp_edge_push,
// which moves it on again). 1 = got, 0 = none.
// Distinct from peek+pop, which BORROWS: a peeked pointer is invalid after pop
// unless the caller ref'd it. Exists because the shim's Source<T>::get() returns an
// OWNED avcpp object — with only peek/pop it would need an av_frame_ref per frame
// to construct one. take() hands over the existing ref instead, so the common path
// performs zero refcount operations, and "who owns Rust's ref" is an ownership
// transfer stated in the signature rather than a documented contract (§8.2).
int     avp_edge_take(AvpEdge*, int timeout_ms, AvpItem* out);

// Wakeup registration for non-blocking nodes (replaces processWhenSignalled).
// The core re-polls the node when the edge becomes readable / writable.
void    avp_edge_notify_readable(AvpEdge*, AvpNode*);
void    avp_edge_notify_writable(AvpEdge*, AvpNode*);
```

### 4.4 Capability discovery — three registers, not one graph walk

The ~30 interfaces in `graph_interfaces.hpp` are discovered today by
`dynamic_cast` + `findNodeUp<T>()` — a per-call upstream walk. **That walk is not
reproduced across the FFI.** It is fragile (a downstream node holding an upstream
interface pointer while the producer stops/restarts — e.g. `decoders.cpp`'s
`input_hold_` workaround, which the source comments admit is a hack), and it is a
false economy: one "walk-and-cast" mechanism serving three unrelated jobs. Reading
the actual call sites shows the interfaces split cleanly into **three registers**,
each with its own, simpler mechanism. No upstream graph walk survives.

**Register 1 — Facts (structural properties of the stream) → `Spec`.**
`IVideoFormatSource`, `IAudioMetadataSource`, `IFrameRateSource`, `ITimeBaseSource`
are read-only descriptions of what flows through an edge: width/height/pixfmt,
sample rate/format/layout, frame rate, time base. They change only at a
discontinuity. These become **`Spec`** — a value latched on the edge and carried
in-band (§4.4.1). A consumer reads the current `Spec` off its *own input edge*; it
never walks. `findNodeUp<IVideoFormatSource>()` is deleted.

**Register 2 — Controls (mutating commands into a running node) → core services.**
`IPlaybackControl` (direction), `IInputReset`, and the seek/speed/pause verbs are
mutations a downstream node used to reach *up* the graph to apply. In the
master-clock model (§6) they are addressed to the `SyncGroup`/seek service instead:
direction is the sign of the clock `rate`, reset rides the `FlushStart`
discontinuity. No node reaches into another node to mutate it, so there is nothing
to dangle on restart. `avp_find_interface_up` is deleted for these too.

**Register 3 — Live queries (ask a *specific, named* node something it computes).**
`IDecoder::discard_until`, `ISentinel` stats, `IReturnsObjects`/`IInputsObjects`
(`node.param.get/set`), `IStreamsInput` demux enumeration. These are real runtime
calls, but every caller already knows *which* node it means (the control protocol
addresses a node by name; an adjacent node talks to its immediate neighbour). So
they keep a **direct** query — `avp_node_query_interface(node, id)` — on an
explicitly-referenced node. There is no traversal: you query the node you already
hold, not "walk up until something answers."

`IJackSink` looks like a Register-3 query and is **not** one: the direction is
inverted. `JackClient` holds `std::vector<std::weak_ptr<IJackSink>>` and calls
*into* the sink's `jack_process(nframes)` from JACK's real-time thread
(`src/JackClient.hpp:16,52,57`; `src/graph_interfaces.hpp:455`; sole implementer
`src/nodes/jack/jack_sink.cpp:17`). Nobody asks the sink a question — the sink
registers itself with a service that then drives it. So it becomes a dedicated
`avp_jack_client_add_sink` on the JACK service (§4.8), not a capability id, and both
`JackClient` and `jack_sink` stay C++ (Tier S).

`IEncoder` and `IMuxer` look like Register-3 queries but are **not**. Their C++
handshake (`initFromFormatContext*` / `setOutput`
/ `openEncoder` / `setOutputPostOpen`) existed only because `findNodeUp` could not
fork across the muxer's N inputs (so the muxer re-exposed the encoder handshake as
`IMuxer` and fanned the query out per input edge), and because queries do not flow
through transforms (so `bsf`/`packet_relay` re-implemented `IEncoder` "not really"
to relay the query upstream). Under Spec-on-edge both dissolve: the muxer
aggregates per-stream Specs like it aggregates packets; relays forward/transform
Spec like they forward packets; the encoder self-opens and emits
codec/codecpar/timebase as Spec. The one backward dependency — the container's
`AVFMT_GLOBALHEADER` flag driving the encoder's `AV_CODEC_FLAG_GLOBAL_HEADER` —
collapses to a fixed encoder default (always set; muxers/bsfs inline extradata
per container — see §4.4.2). No bidirectional negotiation survives.

```c
// Stable, APPEND-ONLY enum, defined in avplumber_interfaces.h (§4.8) — not here:
// a node capability is domain vocabulary, and the framework header carries none.
typedef uint32_t AvpInterfaceId;

// Direct capability query on a node the caller already holds (control protocol,
// adjacent node, or self). This is the ONLY interface-discovery primitive.
// There is NO avp_find_interface_up: Facts travel as Spec on the edge (§4.4.1),
// Controls are addressed to core services (§6), so no upstream walk exists.
const void* avp_node_query_interface(AvpNode*, uint32_t iface_id);
```

Reclassification of the C++ interfaces into registers:

| Interface (C++)            | Register | New mechanism |
|----------------------------|----------|---------------|
| `IVideoFormatSource`       | Fact     | `Spec` (video) latched on edge |
| `IAudioMetadataSource`     | Fact     | `Spec` (audio) latched on edge |
| `IFrameRateSource`         | Fact     | `Spec` field |
| `ITimeBaseSource`          | Fact     | `Spec` field |
| `IPlaybackControl`         | Control  | `SyncGroup` rate sign (§6) |
| `IInputReset`              | Control  | rides `FlushStart` (§3.4) |
| `IDecoder`                 | Query    | direct `query_interface` (named node) |
| `IEncoder` / `IMuxer`      | ~~Query~~ | **removed** — stream config is `Spec` on the edge (Register-1); see §4.4.2 |
| `ISentinel`                | Query    | direct (stats) |
| `IReturnsObjects`/`IInputsObjects` | Query | direct (`node.param` bridge) |
| `IStreamsInput`            | Query    | direct (demux enumeration) |
| `IJackSink`                | ~~Query~~ | **removed** — direct registration on the JACK service; see below |
| `INeedsOutputFrameSize`    | reverse  | see §12 (downstream→upstream hint) |
| `IPreferredFormatReceiver` | reverse  | see §12 (format negotiation direction) |
| `IFrameNumber` / `IFrameTimestamp` | — | were read by the deleted `speedChanged` sync walk; now per-frame buffer metadata, not a capability. Candidate for removal (§12) |

The surviving `AvpInterfaceId` enum therefore lists only the **Register-3** ids, and
lives in `avplumber_interfaces.h` rather than the framework header (§4.8, headers
§1.1). Facts moved to `Spec`; Controls moved to services. The enum is append-only, so
the ids vacated by `IEncoder`/`IMuxer` (2 and 3) stay vacant.

#### 4.4.1 `Spec` — Facts as latched edge state (replaces the upstream walk)

`Spec` is an `EdgeItem` variant (§3.2) — the resolved description of the frames
flowing through an edge (video: w/h/pixfmt; audio: rate/fmt/layout; plus frame
rate, time base). Not GStreamer "caps": there is no capability set and no
negotiation — it is a single concrete value asserting "this is what flows here
now," forward-only, updated when it changes.

> **`Spec` is an enum internally, not a flat struct.** Three variants —
> `Video{…}`, `Audio{…, layout: AVChannelLayout}`, `Packet{codecpar, time_base}` —
> so `width` cannot be meaninglessly present on an audio spec, the audio channel
> layout is carried **whole and losslessly** (the flat `AvpSpec`'s documented
> `CUSTOM`/`AMBISONIC` → `UNSPEC` degradation applies only to the C projection, and
> only to C++ nodes that read the field), and the `Packet` variant can carry real
> `codecpar`, which §4.4.2 requires and a flat struct has no room for. `Spec`
> is `Clone`, not `Copy`: the edge latch holds one and clones on delivery, so
> `current_spec()` returns a clone. The latching mechanism described below applies
> unchanged. The lossy projection to flat `AvpSpec` happens only in the adapter, so
> the degradations are documented in exactly one place. See
> `rust_refactor_native_core.md` §2.2.
>
> Note also that the **buffer**, not `Spec`, carries the authoritative timebase
> (§3.3); `Spec`'s `time_base` is the *declared* one, checked against the buffer at
> the adapter seam in debug builds.

**The edge latches it.** Every edge holds one `Spec` slot. Whenever a `Spec` token
passes, the slot is overwritten. When a node attaches to an edge and first pulls,
the edge re-presents the latched `Spec` as the head item *before* any buffer. This
single mechanism — in the core edge, written once — is what removes the special
case: there is no "pull current truth at start/restart/late-join." **Every reader,
first-start or restarted or joined mid-stream, follows one contract:**

> Pop your input edge. The first item is always the current `Spec`. Buffers follow.
> A `Spec` change arrives in-band as a new `Spec` token *before* the frames it
> describes.

Because the latch decouples producer-start from consumer-start, no synchronized
negotiation phase and no particular start order is required (this is also what
makes restart safe — §3.6).

**The node-side contract on `NodeSISO` is one optional hook, `on_spec`.** The shim
base classifies each `EdgeItem` before the node sees it: a `Spec` item goes to
`on_spec(Spec) -> Spec` (whose return is forwarded downstream and latched on the
output edge); a frame goes to the node's existing `process()`. Three roles, three
bodies:

- **Pass-through** (firewall, split, null_sink): does *not* override `on_spec`.
  Base default is identity → the `Spec` is forwarded unchanged. **~80 Tier-S nodes
  get correct `Spec` forwarding for free, including for media dimensions they know
  nothing about, with zero source edits.**
- **Query-only** (encoder, sentinel): overrides `on_spec` to *read* the format and
  (re)configure itself, then returns it (or emits its output-domain `Spec`).
  Replaces the old `findNodeUp<IVideoFormatSource>()`-at-init. Strictly better: it
  also fires on mid-stream format change, which init-time pull got wrong.
- **Transform** (rescale_video, filters, resample_audio — the nodes that today
  re-implement `IVideoFormatSource` with their *output* values): overrides
  `on_spec` to compute and return a *new* `Spec`. The base forwards the new value,
  not the incoming one — so a consumer N hops down reads the transformed `Spec` off
  its adjacent edge, exactly what `findNodeUp` returned by stopping the walk at the
  transform.

**Initialization timing follows the origin, not a global policy.** The base
guarantees `on_spec` fires before the first `process()`, so a node inits in
`on_spec` and `process()` may assume it is configured — with no
"construct vs. first-frame" branch in node code:

- **Static-format origins** (demux from `codecpar`, `assume_metadata`) emit `Spec`
  as their first output on `start()`; the whole init cascade (decode→scale→encode→
  mux header) settles during startup, before frame 1 — **no added jitter, identical
  to today's construct-time init.**
- **Dynamic-format origins** (a decoder that only learns real pixfmt after frame 1)
  emit `Spec` at frame 1; downstream inits then. That latency is inherent to the
  stream — the encoder *cannot* open before the format is known — not introduced by
  the model. Today's `findNodeUp`-at-init either got lucky via `codecpar` or was
  simply broken for such a source; `on_spec` handles it correctly.

This is what makes **restart during live operation** safe (§3.6) without edge-side
*sticky replay*: `Spec` is not replayed global state (contrast the deleted
`Segment`/epoch — that was global playback *policy*); it is the local, in-place
current value of one pipe, exactly the modular-synth "the cable carries the current
signal spec" model. A restarted node attaches, pops its latched `Spec`, re-inits,
and proceeds. It also kills `input_hold_`: once the core supervisor owns node
lifetime (§6), no downstream node pins an upstream one to keep an interface alive.

#### 4.4.2 Stream config as `Spec` — `IEncoder`/`IMuxer` removed

The encoder↔muxer↔output handshake in C++ (`IEncoder`'s `setOutput`/
`openEncoder`/`setOutputPostOpen` + `IMuxer`'s `initFromFormatContext*`) was a
bidirectional query chain. It is gone; stream config (codec, codecpar, timebase)
is `Spec` on the edge (Register-1), forward-only. The handshake only existed
because of two `findNodeUp` limitations, not because the data required it:

- **Fan-out at joins.** `findNodeUp` returns one node and cannot fork across the
  muxer's N inputs. So the muxer was made to implement `IMuxer` purely to fan the
  query out per input edge (`mux.cpp`'s `s.edge->findNodeUp<IEncoder>()` per
  stream). Under `Spec`, the muxer aggregates N input Specs into one output Spec
  — data fan-in, identical to how it aggregates packets. `process()` is untouched.
- **Queries don't flow through transforms.** `bsf`/`packet_relay` re-implement
  `IEncoder` "not really" (`bsf.cpp`, `packet_relay.cpp`) to relay the query
  upstream. Under `Spec`, relays forward/transform `Spec` like they forward
  packets — no `// not really` boilerplate.

The encoder self-opens (its codecpar/timebase are its own; the muxer rescales
PTS as it does today via `calculateGlobalShift`) and emits `codec`/`codecpar`/
timebase as `Spec`. The output reads the aggregated stream-config `Spec` from its
input edge and does `addStream`/`writeHeader`/`writePackets`/`writeTrailer`
itself. No upstream queries.

**The one backward dependency — `AVFMT_GLOBALHEADER` → `AV_CODEC_FLAG_GLOBAL_HEADER`
— collapses to a fixed encoder default.** The encoder always sets
`AV_CODEC_FLAG_GLOBAL_HEADER` (headers in `extradata`, not inline). Verified
against FFmpeg sources: every container the `output` node targets handles the
extradata without a back-channel —

- *globalheader containers* (MP4/MOV `movenc.c`, MKV/WebM `matroskaenc.c`, FLV,
  NUT, DASH, HLS, segment, smooth, RTSP, ASF, …) write `extradata` into the
  header (`movenc.c:3113` writes `par->extradata` into the stsd/avcC). Not
  setting the flag → empty avcC → broken playback. This is the real failure mode.
- *MPEG-TS* (`mpegtsenc.c`, non-globalheader) prepends `extradata` at keyframes
  itself: `int extradd = (pkt->flags & AV_PKT_FLAG_KEY) ? extradata_size : 0`
  (`mpegtsenc.c:1914`), with the comment *"SPS and PPS are assumed to be available
  in 'extradata' if not found in-band."* Harmless with the flag set.
- *raw H264/HEVC bitstream* (`ff_h264_muxer`/`ff_hevc_muxer`, non-globalheader;
  `ff_raw_write_packet` writes only `pkt->data` and discards `extradata`) is
  saved by the auto-applied `h264_mp4toannexb`/`hevc_mp4toannexb` BSF
  (`rawenc.c:386-392`): `h264_mp4toannexb_init` parses AVCC `extradata` into
  `s->sps`/`s->pps` (`h264_mp4toannexb.c:260-275`) and `h264_mp4toannexb_filter`
  prepends them at IDR (`h264_mp4toannexb.c:384-397`). The raw file is decodable.

**M2 verification note (the one edge):** this relies on the encoder emitting
*AVCC* (length-prefixed) `extradata` under `GLOBAL_HEADER` — the standard for
libx264/nvenc/videotoolbox/vulkan/qsv/libaom/libsvtav1/librav1e/libvvenc/etc.
If an encoder ever emitted *Annex-B* `extradata` with the flag set, the BSF would
no-op (`h264_mp4toannexb.c:267-270`, "looks like Annex B already") and the raw
muxer would discard it → broken raw file. No current encoder does this; record it
as an M2 check when porting the encoder/output/mux.

### 4.5 Factory registration (replaces `DECLNODE` + `generate_node_list`)

```c
typedef AvpNode* (*AvpNodeFactoryFn)(AvpCore*, const char* json_params,
                                     const AvpNodeVtable** out_vtable);

void avp_register_node_factory(AvpCore*, const char* type_name, AvpNodeFactoryFn);
```

> **Params are deserialized into a typed struct, once, by the registry.**
> The native contract is a `NodeSpec` trait (`TYPE_NAME`, associated `Node` type,
> `build(self, &BuildCtx)`) whose params type is `DeserializeOwned`, replacing the
> current `Fn(&str, &str)` where each node parses its own JSON blob. With
> `#[serde(deny_unknown_fields)]`, a typo'd parameter becomes a startup error naming
> the node and listing expected fields — today it is silently ignored and the node
> runs with a default (`realtime` alone has **15** `params.count(...)` guards).
>
> **The dynamic `node.add` path keeps working**, because registration *erases* the
> type: `from_value::<S>` is baked inside the registered closure and monomorphized
> once per node type, so the registry stores a uniform
> `Fn(&str, Value, &BuildCtx) -> Result<Arc<dyn Node>, BuildError>`. One registry
> serves the native builder, `.avplumber` scripts, TCP, and pyplumber; the stringly
> paths gain better errors but not compile-time checking.
>
> Envelope and node params are split by **extract-then-remainder**, not
> `#[serde(flatten)]` — serde does not support `deny_unknown_fields` together with
> `flatten`, and `node.add`'s single flat object mixes framework keys (`group`,
> `auto_restart`, `on_error`) with node keys. avplumber's non-scalar param forms get
> one `Deserialize` newtype each (`Ratio` for `"30/1"`, `Seconds`, `EdgeRef`),
> written once instead of per node. See `rust_refactor_native_core.md` §5.1–5.2.

- C++ side: a `__attribute__((constructor))` (or an explicit registration TU
  emitted by a build step) calls `avp_register_node_factory` per node type. The
  shell-script index disappears.
- Rust side: a proc macro + `linkme`/`inventory` distributed slice registers at
  startup — no codegen script at all (§7).

### 4.6 Instance-shared objects, teams, control protocol

```c
// Instance-shared registry (opaque, string-keyed). Teams live here today; in the
// new core the four Teams are native Rust services, but the generic registry
// stays for node-defined shared objects (sentinel, etc.).
void* avp_instance_shared_get(AvpCore*, const char* type_key, const char* name);
void  avp_instance_shared_put(AvpCore*, const char* type_key, const char* name,
                              void* obj, const AvpMediaVtable* /*retain/release*/);

// Control protocol: Rust owns the TCP server and JSON parsing. Commands that
// target node internals route through query_interface (e.g. node.param.set object
// -> IInputsObjects). The TCP server is a thin JSON wrapper over the
// graph-management ABI (§4.7); embedders (pyplumber §9, OBS §12 item 4, any host
// linking the core as a library) call that ABI directly without speaking TCP.
// No C ABI needed by nodes for the protocol itself.
```

### 4.7 Graph-management ABI (embedder entry point)

§4.1–§4.3 is the surface *nodes* call. An embedder (the TCP control server,
pyplumber, an OBS plugin, or any host linking the core as a library) needs a
separate surface to *drive* the core: build a graph, connect edges, start/stop
groups. These ops are first-class ABI calls — the TCP server is implemented on
top of them, not the only entry point. This is what makes the core embeddable
without forcing an embedder to speak the line protocol, and it is required by
avplumber's own pyplumber (§9) and OBS-embed (§12 item 4) plans — not a hypothetical
future use case.

```c
typedef struct AvpGroup AvpGroup;   // a supervisor unit (§5.3): ordered start/stop

// Graph construction. Params are UTF-8 JSON. Returns NULL on error; *err holds
// a caller-freed UTF-8 message.
AvpNode* avp_create_node(AvpCore*, const char* type_name, const char* instance_name,
                         const char* json_params, const char** err);
AvpEdge* avp_create_edge(AvpCore*, const char* name,
                         AvpNode* producer, const char* out_pad,
                         AvpNode* consumer, const char* in_pad,
                         const AvpEdgeCoupling* coupling /* NULL = default */);

// Groups (supervisor units — §5.3). Ordered start/stop, auto_restart policy.
AvpGroup* avp_create_group (AvpCore*, const char* name);
void      avp_group_add    (AvpGroup*, AvpNode*);
void      avp_group_remove (AvpGroup*, AvpNode*);

// Lifecycle. start() is idempotent on a started group; stop() drains edges per
// §3 before joining threads. destroy() returns ownership to the core.
int   avp_start_group   (AvpGroup*, const char** err);   // 0 ok, -1 err
int   avp_stop_group    (AvpGroup*, const char** err);
void  avp_destroy_node  (AvpCore*, AvpNode*);
void  avp_destroy_edge  (AvpCore*, AvpEdge*);
void  avp_destroy_group (AvpCore*, AvpGroup*);

// Introspection (for embedders that didn't build the graph themselves, e.g.
// a control protocol server attached to an already-running instance).
AvpNode*  avp_lookup_node (AvpCore*, const char* name);
AvpGroup* avp_lookup_group(AvpCore*, const char* name);
```

Design rules carry over from §4: opaque handles, explicit ownership (destroy
returns the handle to the core, which frees it), no `std::`/C++ types cross.
`AvpGroup` is new here but was already implied by §5.3's supervisor; making it
an explicit handle lets an embedder drive group lifecycle without the TCP
protocol, and lets the supervisor's topo-sort + auto_restart be reused by any
embedder.

### 4.8 Service discovery — keeping `avplumber_core.h` framework-only

> **Natively, services are `Arc<dyn Trait>` keyed by `TypeId`; the C
> vtable is generated per service on top.** A registry storing
> `HashMap<AvpServiceId, *const c_void>` (as the skeleton does) hands a Rust caller a
> raw pointer to a C vtable describing a Rust object it already owns — and **nothing
> keeps the pointed-to object alive**. Native storage is therefore
> `HashMap<TypeId, Arc<dyn Any + Send + Sync>>` with typed
> `Instance::service::<T>() -> Option<Arc<T::Iface>>`, so `SyncGroup` and friends are
> reached as `Arc<dyn SyncGroup>` with no cast and a correct lifetime.
> `avp_core_query_service` keeps the signature below: it looks up the `Arc` and
> returns a pointer to a static vtable whose functions downcast and call the trait —
> so vtable construction is one generated site per service rather than the primary
> representation. Likewise the native graph builder uses typed `NodeRef<T>` handles
> with media checked at `connect()`, not stringly `bind_source`/`bind_sink` (a
> string-keyed bind silently skips the insert on a name mismatch while still
> returning a valid-looking handle). See `rust_refactor_native_core.md` §5.3–5.4.

§4.7 is the *framework* surface (graph, nodes, edges, groups, lifecycle). Domain
services — the §6 core services (SyncGroup, SharedTimeline, seek) and node
capabilities (IDecoder, ISentinel, IStreamsInput, hwaccel specifics) — are **not**
named functions in `avplumber_core.h`. Naming `avp_syncgroup_set_rate`,
`avp_team_seek`, `avp_muxer_write_header`, etc. in the core header would bake
A/V-playback vocabulary into the framework header and force every embedder to
carry it. Two discovery mechanisms (one already in §4.4) keep the core header
clean:

**Node capabilities (Register 3 of §4.4): `avp_node_query_interface(node, id)`.**
A node that implements a capability (`IDecoder` for a decoder, `ISentinel` for a
capture card, a hypothetical `ICudaSurface` for a hwaccel node) returns a const
vtable pointer for that interface ID. The *mechanism* is in
`avplumber_core.h`; the interface IDs and their vtables live in a separate
`avplumber_interfaces.h`. **Decoders, hwaccel, sentinels — all node-level
capabilities — go here, not in the core header.** (Encoder/muxer stream config
is `Spec` on the edge, §4.4.2 — not a capability.)

**Core services (Register 2 of §4.4): `avp_core_query_service(core, id)`.** The
§6 services (SyncGroup the master clock, SharedTimeline, the seek service) are
core-owned but domain-specific. An embedder reaches them by service ID, not by a
named per-verb function:

```c
typedef uint32_t AvpServiceId;   // stable, append-only — see avplumber_services_*.h

// Returns a const service vtable pointer, or NULL if the service isn't
// registered in this build. The mechanism is in avplumber_core.h; each
// service's ID + vtable lives in its own header.
const void* avp_core_query_service(AvpCore*, AvpServiceId);
```

Each service's ID and vtable live in its own header
(`avplumber_services_clock.h` for SyncGroup, `avplumber_services_timeline.h`
for SharedTimeline, etc.). An embedder who doesn't care about A/V playback (a
generic graph host, a test harness) includes only `avplumber_core.h` and links a
smaller surface; an embedder that wants playback control includes the service
headers it needs.

**Net header layout:**

- `avplumber_core.h` — framework only: handles, node vtable, edge ops,
  factory, instance-shared registry, graph management (§4.7), and the two
  discovery mechanisms (`avp_node_query_interface`, `avp_core_query_service`).
  No domain vocabulary.
- `avplumber_interfaces.h` — Register-3 node-capability IDs + vtables
  (IDecoder, ISentinel, IReturnsObjects, IInputsObjects, IStreamsInput).
  (IEncoder/IMuxer removed — stream config is `Spec`, §4.4.2; IJackSink
  removed — wired via direct registration on the JackClient service, §4.4.)
- `avplumber_services_*.h` — one per core service: SyncGroup (clock),
  SharedTimeline, seek, etc. The §6 Teams-as-services map to service IDs +
  vtables in these headers, reached via `avp_core_query_service`.

This is the same separation §4.4 already made for capabilities; §4.8 just
names the core-service side of it and fixes the header boundary. A future
host reusing the substrate (per §4.7) includes `avplumber_core.h` and defines
its own service headers — it never has to carry avplumber's playback vocabulary.

---

## 5. Scheduling

### 5.1 Two execution models, kept — but reframed

Today (confirmed in `graph_mgmt.cpp`): blocking nodes get a `std::thread` running
`NodeWrapper::threadFunction`; non-blocking nodes (`NonBlockingNodeBase`) run on a
shared `EventLoop` (`poll()` reactor over eventfd) or are driven by a `TickSource`.
`NodeGroup` runs a management-thread state machine (`EMPTY/STOPPED/STARTED/MIXED/
RESTART/FINISH_THREAD`) and topologically sorts nodes for ordered start/stop.

Both models stay; what changes is *how* the non-blocking side is written and timed.

> **The strategy is per-node and explicit, and `Blocking` is the default
> for native Rust nodes too.** Each node returns one `NodeBody` variant
> (`Blocking` / `Poll` / `Async`), chosen by **what the body does, not which language
> wrote it**. Synchronous FFmpeg and CUDA calls (`av_read_frame`,
> `avcodec_send_packet`, `av_interleaved_write_frame`, `sws_scale`, `swr_convert`)
> have no async form, and running them on a tokio worker would stall every node in
> the clock domain — so a *native Rust* demuxer or encoder is `Blocking`, exactly like
> its C++ counterpart, and one-input leaf nodes (`split`, `firewall`,
> `packet_relay`, `null_sink`) are `Blocking` too because `edge.take_blocking()`
> reads more plainly than a future. `Async` earns its place where a body awaits
> several inputs or an input plus a clock (`select!`), which is perhaps 2–3 of the
> Tier-R set. Mixing costs nothing: rsmpeg's media types are `Send`, so a frame
> produced on an OS thread and consumed by a tokio task needs no wrapper. The case
> for async below is scoped to those nodes, not to native code in general. See
> `rust_refactor_native_core.md` §4.1.

### 5.2 Hybrid, with tokio — tick-as-event, not timer-as-tick

- **Blocking nodes → real OS threads.** You cannot `.await` inside
  `avcodec_send_packet`, a CUDA kernel launch, or a blocking `read()`. Each blocking
  node owns a thread; the body is a straight loop calling `process()`. If a blocking
  node genuinely wants async networking it may build its own tokio runtime on its
  thread, but that is the node's concern, not the framework's.
- **Non-blocking nodes → a tokio current-thread runtime per clock domain.**
  Async/await is adopted here because it is precisely where the current design is
  weakest — the docs warn that stateful non-blocking nodes are hard
  (`processWhenSignalled` + `weak_ptr` capture dance, the "hidden queue of size 1",
  UAF footguns). An `async fn` holds state across `.await` points for free; the
  compiler builds the state machine. This is the single biggest boilerplate + safety
  win and directly serves the project's "avoid boilerplate" value. Adopting tokio
  here (rather than a bespoke executor) additionally brings `tokio::net`,
  `tokio::io`, `tokio::sync`, and `tokio_util` to every non-blocking node with no
  custom reactor to maintain.
- **Why tokio, and why this still preserves frame-perfect output.** The demanding
  case is frame-perfect output driven by the mixer's tick, and the standing objection
  to tokio is that "its timer wheel adds jitter." That objection applies to
  `tokio::time` — not to the runtime as a whole. The
  frame-perfect path is *tick-driven, not timer-driven*: the mixer delivers the tick
  as a `tokio::sync::Notify` (or `mpsc`/`watch`) event, which the node `select!`s on.
  No `tokio::time` await sits on that path, so no timer-wheel jitter reaches it. The
  `new_current_thread` scheduler is cooperative and single-threaded — it drains
  ready tasks before parking — which is precisely the existing
  `EventLoop::fastExecute` inline-drain trick (run callbacks synchronously when you
  can grab the lock). No work-stealing, no cross-thread preemption: the "opaque
  scheduling policy" concern does not apply to the current-thread runtime.
- **Runtime construction.** One runtime per clock domain, I/O driver on, timer
  driver off by default:

  ```rust
  let rt = tokio::runtime::Builder::new_current_thread()
      .enable_io()           // tokio::net, tokio::io — available to every node
      // .enable_time()      // deliberately OFF by default; opt-in per domain
      .build()?;
  ```

  The runtime runs on a dedicated thread via `rt.block_on(supervisor_loop)`; each
  node's `run()` is `rt.spawn()`-ed as a task on its clock domain's runtime. This is
  single-threaded per domain by construction — one executor per clock domain, which
  is what the timing model requires.
- **The tick as event.** The mixer (a core service, or a blocking node on its own
  thread) calls `tick_notify.notify_one()`. `Notify::notify_one()` is cross-thread
  and wakes via the runtime's I/O-driver eventfd — sub-microsecond on Linux,
  comparable to the bespoke eventfd wakeup it replaces. The node loop is plain
  `tokio::select!`:

  ```rust
  async fn run(mut self, ctx: NodeCtx) -> Result<()> {
      loop {
          tokio::select! {
              _   = ctx.tick.notified()  => self.on_tick(&ctx).await?,  // external, not tokio::time
              buf = ctx.input.recv()     => match buf {
                  Item::Buffer(b) => self.on_buffer(b, &ctx).await?,
                  Item::Event(e)  => self.on_event(e, &ctx).await?,    // Flush/Spec/Eof
              }
          }
      }
  }
  ```

- **`tokio::net` / `tokio::io` work directly.** A node doing RTMP/TCP/IPC just
  `.await`s `TcpStream::connect`, `AsyncReadExt::read`, etc. — no sidecar runtime,
  no channel bridge. `tokio::sync::*` (Mutex, RwLock, mpsc, oneshot, broadcast,
  Notify, Semaphore) is available everywhere; it is executor-agnostic and would
  have worked even on a bespoke executor, but adopting tokio removes the need to
  maintain that executor at all.
- **`tokio::time` is opt-in, with a documented jitter caveat.** Nodes that need a
  watchdog/timeout enable `.enable_time()` on their domain's runtime (or the
  decision is made domain-wide) and accept that *those specific awaits* carry
  timer-wheel jitter. Frame-perfect paths stay on the tick event. Rule, enforced in
  review and lint: **do not use `tokio::time` for frame pacing; use the tick
  event.** With `.enable_time()` off by default, a stray `tokio::time::sleep`
  panics ("time driver not enabled") rather than silently introducing jitter — the
  failure mode is loud.

### 5.3 Group management and edge-type selection

- `NodeGroup`'s state machine, topological sort, and `auto_restart`/`on_error`
  (`restart_node`/`restart_group`/`panic`/`exit`) move into the core as a Rust
  supervisor. Ordered start/stop is preserved.
- During graph construction the core assigns each node to an execution context
  (a blocking thread, or a named tokio current-thread runtime / clock domain). For
  an edge, if both endpoints share the same tokio `Runtime` (clock domain) and
  neither is blocking → `DirectEdge`; otherwise `BufferedEdge`. The point to hold is
  that this is *derived* from execution context, not guessed.

### 5.4 Rust code structure — substrate vs executor vs supervisor

The C++ already separates the substrate (`graph_core.hpp`: `Node`, `Edge`,
`Source`/`Sink`, `NonBlockingNodeBase`) from the execution engines (`graph_mgmt.{hpp,cpp}`:
`NodeWrapper` branches on `isNonBlocking()` → `std::thread`+`threadFunction` or
`EventLoop`+`TickSource`; `NodeGroup` is the supervisor above both). The Rust
preserves and sharpens this split. Three rules govern it.

**Rule 1 — the substrate (`graph/`) depends on nothing in `exec/`.** This is the
one real improvement over C++: `NonBlockingNodeBase` in `graph_core.hpp` directly
references `EventLoop` (`event_loop_`, `processInEventLoop`,
`wrappedProcessNonBlocking(EventLoop&, ...)`), so the C++ substrate *knows about*
the non-blocking engine. The Rust substrate must not reproduce this leak. The node
contract is just `trait Node` returning one `NodeBody` variant (§4.2); the executor
reads that variant to decide how to drive the node, and the node names no executor
type. Across the ABI the same split is two function pointers, `process()` and
`poll()`. The reactor (tokio) lives entirely in `exec::async`. This is the seam a
future `FixpointExecutor` (§5.5) would use.

**Rule 2 — the executor is a *set-level* abstraction, not per-node.** A non-blocking
executor does not run one node; it runs a *set* of co-located nodes cooperatively —
when a packet lands on an edge, the executor wakes the *consumer* node and polls it.
This is multi-node scheduling (the C++ `EventLoop`), not "start this one thread." The
blocking side is one thread per node, but that is still a *strategy for running a set*,
not a per-node trait. The unit is "a runtime that owns a set of nodes and schedules
them":

```rust
// exec/mod.rs — an Executor runs a SET of co-located nodes
trait Executor {
    fn add_node   (&self, node: NodeEntry) -> Result<()>;
    fn remove_node(&self, name: &str)     -> Result<()>;
    fn start     (&self) -> Result<()>;   // begin running all added nodes
    fn stop      (&self) -> Result<()>;
    fn interrupt (&self);
    fn join      (&self);                 // wait for all node tasks/threads to finish
}
```

- `BlockingExecutor` — `add_node` spawns an OS thread per node running `process()`
  in a loop (≈ `threadFunction`). Owns the `JoinHandle`s.
- `AsyncExecutor` — wraps a tokio current-thread runtime (§5.2); `add_node` spawns
  the node's `async fn run()` as a task; edge readiness wakes tasks via
  `tokio::select!`. One executor per clock domain.

**Rule 3 — inter-executor routing goes through the edge, not executor-to-executor.**
When an async producer pushes to an edge whose consumer is a blocking node on a
different executor, the producer's executor calls `edge.push(item)` (signaling the
edge's wakeup); the blocking consumer's thread is parked in `edge.peek()` and wakes
on that same wakeup. The edge is the bridge. No executor knows about another
executor — they all interact with the same substrate-level edge, which carries a
`Wakeup` primitive that *any* executor can signal and *any* executor can wait on.
This is the C++ `Event` on `EdgeBase::produced_`/`consumed_` (a pthread condvar,
cross-thread); the Rust equivalent is a substrate-level `Wakeup` with both a
sync-wait path (blocking thread) and an async-waker registration path (tokio task).
One small adapter, but it lives in `graph/edge.rs`, not `exec/`.

**Fixed vs pluggable — where the management logic is pluggable:**

| Layer | Role | Pluggable? |
|---|---|---|
| `graph/` (substrate) | data plane, node contract, edge `Wakeup` | fixed (shared) |
| `supervisor/` (was `NodeGroup`) | lifecycle: state machine, topo sort for start/stop order, `auto_restart` | **fixed** |
| `exec/` (was `NodeWrapper`+`EventLoop`+`TickSource`) | scheduling a running set: who gets polled when an edge is readable, how backpressure yields | **pluggable** |
| `services/` (§6) | clock, timeline, seek — behind `avp_core_query_service` (§4.8) | fixed (with trait impls, §3.3) |
| `abi/` (§4.7) | embedder entry; `control/` is built on top | fixed |

The supervisor tells the executor "start these N nodes" (in topo order); the
executor decides how to run them (threads vs tasks vs fixpoint iteration). The
supervisor never calls `thread::spawn` directly — it calls `executor.add_node` /
`executor.start`. That is what makes the executor pluggable and the supervisor
engine-agnostic.

**Module layout:**

```
avplumber_f7k/
  include/                  — the C headers (headers §1): avplumber_core.h (framework
                              only), avplumber_interfaces.h, avplumber_services_*.h
  src/
    lib.rs                  — crate root, Instance (was NodeManager)
    scaffold.rs             — Transform/Sink helper traits over NodeBody (native-core §4.2)
    graph/                  — SUBSTRATE (engine-agnostic), = graph_core.hpp
      mod.rs                 — Graph, Node, Pad (structure only)
      edge.rs                — Edge, EdgeItem, EdgeEvent, Spec, DirectEdge, Wakeup
      buffered_edge.rs       — BufferedEdge: the queue impl + flush drain
      buffer.rs              — Media, AvpBuffer, AvpMediaType, AvpMediaVtable
      capability.rs          — AvpInterfaceId, AvpServiceId, query contracts
      node_vtable.rs         — NodeBody (Blocking/Poll/Async) + the C AvpNodeVtable
    exec/                   — EXECUTORS (set-level), = graph_mgmt NodeWrapper branch
      mod.rs                 — Executor trait
      blocking.rs            — BlockingExecutor: OS thread per node
      async_rt.rs            — AsyncExecutor: tokio current-thread per clock domain
                               (not `async.rs`: `async` is a reserved keyword)
    supervisor/             — = graph_mgmt NodeGroup
      mod.rs                 — Group: state machine, auto_restart (fixed, engine-agnostic)
      topo.rs                — DFS topological sort (from NodeGroupUtils)
    factory/                — node factory registry (§4.5)
    services/               — core services (§6), behind avp_core_query_service (§4.8)
      mod.rs                 — service registry
      clock.rs               — SyncGroup trait + WallClock/SourceTimeClock/SyntheticClock (§3.3)
      timeline.rs            — SharedTimeline
      seek.rs                — SeekIndex
    abi/                    — C ABI surface (§4.7 embedder entry, §4.8 service discovery)
      mod.rs, node.rs, edge_ops.rs, graph_mgmt.rs, registry.rs, control.rs
  tests/                    — integration tests (smoke_2node.rs is the M0 gate, §8)
```

**Dependency direction (enforced):** `graph/` → (nothing in `exec`/`supervisor`/
`services`). `exec/` → `graph/`. `supervisor/` → `graph/` + `exec/` (via `Executor`
trait). `services/` → `graph/`. `abi/` → `graph/` + `factory/` + `supervisor/` +
`services/`. `control/` → `abi/`. A future host reusing the substrate links
`graph/` + `abi/` and supplies its own `Executor` impl + service headers — it never
links `exec/` or `control/` if it doesn't want avplumber's schedulers or TCP
protocol.

### 5.5 Future: a `FixpointExecutor` (not for avplumber)

The `Executor` trait is the seam a future host (per §4.7) would use to add a
non-causal engine. A `FixpointExecutor` runs its set sample-by-sample, iterating
any cycle to convergence before ticking — for audio zero-delay feedback loops, or
the video analog (§3.1 keeps causality as an *executor* property, not a substrate
invariant, so this is possible without rewriting `graph/` or `supervisor/`). The
supervisor's lifecycle (start/stop/restart, auto_restart) is unchanged — it still
calls `executor.start()`. Only the scheduling inside the executor differs. This is
why the executor is at the set level and pluggable: it's the right shape for a
fixpoint engine to be *added* later, not retrofitted. avplumber does not build this;
it's noted here so the §5.4 seams are designed to admit it.

---

## 6. Playback control redesign — Teams as core services

### 6.1 Why the Teams must be core

Reading confirmed the Teams are not helpers hanging off nodes — they *are* the
distributed scheduler state:

- `RealTimeTeam` synchronizes `offset_` to the smallest offset across the team (so
  streams stay jointly buffered) **and** is the `IFlushAndSeek` coordinator that
  drives the 4-phase seek across all `seek_targets_`. `realtime` nodes are thin
  members registering a `weak_ptr`.
- `SpeedControlTeam` maintains `last_pts_/last_sync_/shift_` and computes a
  piecewise-linear PTS remap (`scalePTS`) — the playback→wall mapping that in the
  new model lives on the master clock and is applied at the output (§3.3).
- `PauseControlTeam` holds `paused_` + a condvar, `pause_at_`, and lists of
  `IInputReset`/`IFlushAndSeek`/`IPlaybackControl` weak refs.
- `InputSeekTeam` fans `ISeekAt` add/clear to members.
- `SharedTimeline` (a fifth `InstanceShared` singleton, structurally a Team) is a
  named, PTS-keyed key/value store: the control protocol/mixer schedule values at a
  source-time PTS and `TimelineReader` nodes read "the value in effect at this
  frame's PTS". Not on the data plane (no frames), but it crosses the boundary in
  both directions (Rust control writes, C++ Tier-S nodes read), so it becomes a core
  service with its own header and vtable (`avplumber_services_timeline.h`,
  headers §1.2), reached via `avp_core_query_service` — not named `avp_timeline_*`
  functions in the framework header (§4.8).

Porting these as shimmed C++ nodes would drag the whole out-of-band, RTTI-driven,
Team-singleton machinery into the new core. Instead:

### 6.2 The mapping

| Old Team mechanism | New core mechanism |
|---|---|
| `RealTimeTeam::offset_` sync to smallest | `SyncGroup` **master clock**: one shared playback→wall mapping (offset) the group's output stages read at release time (§3.3) |
| `RealTimeTeam::flushAndSeek*` 4-phase | core issues `FlushStart`/`FlushStop` (queue-clearing discontinuity) + one master-clock reset for the group (§3.4) |
| `SpeedControlTeam::scalePTS` remap | `rate` on the master clock; applied at the output when a frame is released. O(1) speed change, no in-flight rewrite (§3.3) |
| direction sign-flip broadcast | master-clock `rate < 0` |
| `PauseControlTeam` condvar | freeze the master clock: the output stops releasing, back-pressure propagates upstream through the pipes (§3.3) |
| `pause_at_` | a scheduled core action that freezes the clock at a target playback time |
| `InputSeekTeam` fan-out | core `seek_at` table keyed by `SyncGroup` |
| `SharedTimeline` `set/get/gc` keyed by PTS | core `SharedTimeline` service (`AVP_SERVICE_TIMELINE`); control writes, C++ `TimelineReader` nodes read the value in effect at a frame's PTS (§4.8, headers §1.2) |
| `RealTimeSpeed` reading `AVFrame::metadata` `frame_ts/frame_no/wallclock` | source-time PTS on the frame + the master clock, not dict round-trips |

These core mechanisms are **core services, not named ABI functions** — an embedder
reaches them via `avp_core_query_service(core, service_id)` (§4.8), with each
service's vtable in its own header (`avplumber_services_clock.h` for SyncGroup,
`avplumber_services_timeline.h` for `AvpTimeline`, etc.). This keeps
`avplumber_core.h` free of A/V-playback vocabulary; the §6 services live behind
the discovery mechanism, not as `avp_syncgroup_*` / `avp_timeline_*` calls in the
core header.

### 6.3 `input_rec` becomes a core `InputReader`

`input_rec.cpp` mixes three concerns: (a) demux/read loop, (b) a **seek-table**
subsystem (`seek_table_`, background `seekThreadFun`, byte/frame/ts resolution),
and (c) a **custom-timeline** subsystem (`ETimestampSource`, `ts_offsets_`,
`setFrameMetadataTimestamps` writing `av_dict_set`). In the new core:

- (a) stays a node (it's genuine IO), backed by a real thread.
- (b) the seek table + `resolveSeekTarget` become a core `SeekIndex` service the
  node owns; `StreamTarget` resolution (frame/live/end/bytes/wallclock/sync) is
  core code.
- (c) `TimestampSource` and the offset table become an **input-side stamping
  policy**: the `InputReader` computes each buffer's source-time PTS according to
  the selected `ts_source` (Input/Wallclock/Sync) *at read time*, and stamps the
  frame's PTS directly — no `av_dict_set`, no per-frame metadata smuggling. From
  then on the PTS is plain source time and every downstream node treats it
  uniformly (§3.3). `ts_source` is thus an input configuration, not a value that
  travels the graph; the output stage maps that source-time PTS through the master
  clock like any other. The `realtime` replacement does no `av_dict_get` + `atoll`
  per frame, unlike today's.

### 6.4 What gets deleted

`IFlushAndSeek` (interface + the `NodeSingleInput` 4-phase impl,
`graph_base.hpp:84-170`); `EdgeBase::flushing_/flushed_` and
`startFlushing/stopFlushing/maybeFlush`; the EOF-marker special path (folded into
`EdgeEvent::Eof`); the `pauseProcessing()`/`lockProcessing()` cross-node barrier
used only by seek. `Node::pauseProcessing` stays for the executor's group-suspend.
Also deleted outright: `speed.cpp::speedChanged` (the ~120-line upstream reach-back
that re-rescaled in-flight PTSes), `SpeedControlTeam::scalePTS` as a *node-level*
remap, and the `AVFrame::metadata` `frame_ts`/`scaled_pts_` round-trip — all
obviated by source-time PTS + a master clock applied at the output (§3.3).

---

## 7. Node authoring — traits + proc macros (the boilerplate win)

> **Types used below.** Scaffolds are built on the native types: `Media` (owned rsmpeg
> values), `Ts` for timestamps, and typed `In<T>`/`Out<T>` pad handles with the media
> type checked at bind and reported as a connect-time error naming both pads. `Flow`
> splits into `Push` (edge-op result) and `Tick`/`Blocked` (node-invocation result),
> so the `Transform`/`FlowNode` sketch below returns the narrower type. Params come
> from a `NodeSpec` struct (§4.5), so the macro emits no hand-written param
> extraction.
>
> **Typed-pad code size was measured, not assumed.** In a standalone crate with five
> media types, one non-generic `BufferedEdge`, and a generic `Firewall<T>`, each
> `Firewall<T>::run` instantiation was **67–71 bytes** (~350 B for all five), and
> `In<T>`/`Out<T>` emitted **no symbols at all** — fully inlined, since the unwrap is
> one discriminant compare and `PhantomData` is zero-sized. The C++ template blowup
> this resembles came from templating the *whole edge stack* (`Edge<T>`,
> `EdgeWrapper<T>`, `EdgeSource<T>`, `EdgeSink<T>` — ring buffer, condvars, flush
> logic, stats — × 5 types, plus 11 `DECLNODE_ATD` sites); here the edge stack is
> non-generic and only the thin pad handle is typed. Honest limits: the ×5
> instantiation cost exists in any scheme supporting generic nodes,
> media-*specific* nodes instantiate once at zero cost and are the majority, and
> **`async fn` in a generic trait duplicates the whole state machine — so async node
> bodies stay non-generic where the media type is fixed.** See
> `rust_refactor_native_core.md` §4.4.

### 7.1 Registration

`DECLNODE` + `generate_node_list` (shell + `find` in the Makefile) → a proc macro
with compile-time registration:

```rust
#[avp_node("rescale_video")]           // registers factory via linkme slice
struct RescaleVideo { dst: VideoParams }
```

No index file, no codegen script, no X-macro. The macro emits the factory fn, the
`AvpNodeVtable`, the `query_interface` glue for whatever capability traits the type
implements, and serde param parsing.

### 7.2 Scaffolds

The 30-line non-blocking template in `developing_nodes.md` becomes a one-method
trait with framework-provided polling/backpressure/tick/flush handling:

```rust
// Stateless-ish transform: framework owns the edge dance.
trait Transform {
    fn transform(&mut self, buf: Buffer) -> Flow;   // push/drop/backpressure
    fn on_event(&mut self, _e: &EdgeEvent) {}        // default: forward
}

// Stateful flow control when needed: explicit poll, state held across awaits.
trait FlowNode {
    async fn run(self, ctx: NodeCtx) -> Result<()>;
}
```

`firewall` in full:

```rust
#[avp_node("firewall")]
struct Firewall;
impl Transform for Firewall {
    fn transform(&mut self, b: Buffer) -> Flow {
        if b.pts().is_valid() { Flow::push(b) } else { Flow::drop() }
    }
}
```

The `Source`/`Sink` wrapper indirection that `developing_nodes.md` flags as
"Refactor needed / never used" simply does not exist in the new API.

---

## 8. C++ backward-compat shim

Goal: existing *leaf/processing* C++ nodes (codecs, filters, muxers, IO, metadata)
recompile against a new header with minimal edits and run under the Rust core. The
scheduler-adjacent nodes (`realtime`/`speed`/`pause`/`input_rec`) are **not**
shimmed — they become core services (§6).

### 8.1 What the shim provides

A header `avplumber_node_compat.hpp` that re-implements the API surface nodes use —
`Node`, `NodeSISO<In,Out>`, `NodeSingleInput/Output`, `NodeMultiInput/Output`,
`Source<T>`/`Sink<T>`, `createCommon`, `NodeCreationInfo`, and the `DECLNODE*`
macros — backed by the C ABI instead of the old core. `findNodeUp<T>()` is **not**
re-implemented as a walk (§4.4 deletes it); the Fact interfaces it used to reach for
are served by `Spec` (below).

- `Source<T>::get/peek/pop` → wrap the returned `AvpBuffer.ptr` into
  `av::VideoFrame`/`av::Packet`/`av::AudioSamples` **adopting** the pointer so
  `.raw()` returns it and refcount is balanced (retain on wrap, release on wrapper
  drop; the buffer Rust handed us already carries a ref).
  `get()` maps to `avp_edge_take` (§4.3), not peek+pop — it returns an owned object,
  so it receives an owned ref and the common path costs no refcount operation.
- **`T* peek()` is retired on both sides**, replaced by a move-only RAII **borrow
  guard** (`Source<T>::peek()` → `Peeked`, with `.get()`/`.mut()`/`.consume()`;
  drop = leave it queued) mirroring the `avp_edge_peek`/`_release`/`_consume` triple
  in §4.3. The bare pointer cannot survive the seam: it points into a **live avcpp
  object** whose C++ members the caller dereferences (`frmin->isComplete()`,
  `frmin->timeBase()`), so the shim would have to hand out a pointer into the core's
  queue. It is also *already unsound in the current C++ core* —
  `ReaderWriterQueue::pop()` destroys the element in place, and
  `cuda_rect_overlay.cpp` holds a peeked pointer across a pop. The guard needs no
  invalidation protocol and lets a multi-input node hold two borrows as two values.
  Full spec, the bug, and the call-site census:
  `rust_refactor_native_core.md` §4.3.1. Convert the C++ tree to the guard **before**
  writing the shim.
- `Sink<T>::put` → extract `.raw()`, build an `AvpBuffer`, `avp_edge_push`;
  translate `BACKPRESSURE` to the old `put(..., drop_if_full)` bool contract.
- `DECLNODE(nodetype, Class)` → emits `avp_register_node_factory("nodetype",
  &Class::__avp_factory)` + a generated `query_interface` that maps the surviving
  (Register-3, §4.4) interface-ID table to `dynamic_cast<IFoo*>(this)`. So a node
  that today does `class Foo : public NodeSISO<...>, public IDecoder` keeps working:
  the shim's generated `query_interface(ID_IDecoder)` returns a C vtable trampolining
  to the node's virtual methods. (Register-1 Fact interfaces like
  `IVideoFormatSource` are *not* in this table — see `Spec` mapping next.)
- **`Spec` ⇄ Fact-interface bridge.** The shim base's `on_spec` hook connects the C
  ABI `Spec` token (§4.4.1) to the node's existing C++ code with no source edits in
  the common case: for a node that implements a Fact interface with its *output*
  values (`rescale_video`, `filters` — Register-1 Transform), the shim reads those
  virtual getters (`width()`/`height()`/…) after the node processes the incoming
  `Spec` and emits the resulting `Spec` downstream; for a node that *read* an
  upstream Fact at init (encoder's `findNodeUp<IVideoFormatSource>()`), the shim
  delivers the latched input `Spec` to a generated `on_spec` that feeds the same
  fields the old init pulled. Nodes that neither produce nor consume format inherit
  identity forwarding for free.
- `DECLNODE_ATD*` (type auto-detect) → the shim registers one factory per
  concrete media type, since the runtime tag replaces template specialization.
- EOF: the shim delivers `EdgeEvent::Eof` to nodes as the old EOF marker via
  `peek`, so `isEofMarker`/`consumeEofIfPresent` code paths still fire during the
  transition.

### 8.2 The `.raw()` contract (the refcount risk)

The one thing that must be exactly right: when the shim wraps `AvpBuffer.ptr`
(an `AVFrame*` Rust owns a ref to) into an `av::VideoFrame`, and the node calls
`.raw()` and mutates metadata/side-data, that must not corrupt Rust's ref. Rule:
Rust transfers *one* ref into the node with each delivered buffer; the shim wrapper
owns that ref and frees it on destruction unless the node forwards it via `put`
(which moves the ref into the outgoing `AvpBuffer`). This mirrors avcpp's existing
move/copy semantics, so idiomatic node code stays correct.

**Scope of this rule.** The one-ref-per-delivery contract above is a *lifetime*
contract for the **linear one-in / one-out** case. It guarantees the buffer is not
freed while the node holds it, and is freed exactly once afterwards. It does **not**
by itself make fan-out or in-place mutation safe — see §8.2.1.

### 8.2.1 Fan-out and shared-buffer mutation invariants

> **For native Rust nodes, three of the five invariants below are unrepresentable
> rather than contractual. For C++ nodes all five are contractual, exactly as
> written.**
>
> | Invariant | Native (owned `Media`) | C++ (via shim) |
> |---|---|---|
> | distinct struct per edge | `Media` is **moved** into one edge; a second edge requires `.clone()` (= `av_frame_clone`, a new struct). Enqueueing the same value twice **does not compile.** | contractual |
> | never touch one struct from two threads | rsmpeg is `Send` but **deliberately not `Sync`** (verified). Sharing one `AVFrame` across threads is a **type error.** | contractual |
> | lifetime / no double-free | `Drop` | contractual (§8.2) |
> | CoW before in-place write | `make_writable()` exists and returns `Result`, but **calling it stays the author's responsibility** — review-enforced | contractual |
> | no in-place `side_data` mutation | **review-enforced** | contractual |
>
> So the fan-out hazard in native `split`/`one_to_many` is caught by the type system
> rather than by review. That is a **narrowing** of the risk surface, not its
> elimination: the two mutation rules survive for everyone.
>
> One gap to fill: rsmpeg's `AVPacket` has **no `Clone`** (verified —
> `avcodec/packet.rs` is 61 lines with no `Clone` impl), so a packet-edge fan-out
> needs a `PacketExt::clone_ref()` wrapping `av_packet_clone`. The test requirement
> at the end of this section also covers the `into_raw`→C→`from_raw` round-trip
> (asserting the refcount is unchanged, proving the hand-off is a move) and the
> adapter's timebase/`isComplete` bridges (§2.2), which types cannot catch. See
> `rust_refactor_native_core.md` §6.

`av_frame_ref`/`av_packet_ref` share the underlying `AVBuffer` (whose `refcount`
is `atomic_uint` — concurrent holders across threads are safe for the *buffer*),
but they give each holder its **own struct**: `metadata` is deep-copied, `side_data`
entries are shared via `av_buffer_ref`, and scalar fields (`pts`/`format`/`width`/…)
are plain-copied. **The `AVFrame`/`AVPacket` struct container has no locking.** The
resulting invariant, which the linear §8.2 contract does not capture, is:

> Two threads may hold refs to the same buffer, but must never touch the same
> `AVFrame*`/`AVPacket*` **struct** concurrently.

Three rules follow. They are currently satisfied only by luck of the present node
set; the native-Rust `split`/`one_to_many` rewrites (Phase 2 proof set, §10) are
precisely where a naive implementation introduces a double-free / use-after-free
race, so they are stated here as hard requirements:

1. **Fan-out ref's a distinct struct per edge.** `split` and `one_to_many`
   must `av_frame_ref`/`av_frame_clone` (rsmpeg `AVFrame::clone`
   → `av_frame_clone`) a **new struct** into *each* output edge. **Never enqueue the
   same `AVFrame*`/`AVPacket*` into more than one edge** — the shared atomic buffer
   refcount does not save you, because the race/double-free is on the struct
   container, not the buffer. This is exactly what the C++ `split.cpp`/`one_to_many.cpp`
   do today: each `EdgeSink<T>::put(*data)` copies the avcpp object (= one
   `av_frame_ref` into a fresh struct per edge).

2. **In-place writes require copy-on-write.** A shared buffer (`refcount > 1`) must
   not be written through `data[]`/sample pointers without first calling
   `av_frame_make_writable` (rsmpeg `AVFrame::make_writable`), which copies when the
   ref is shared. Today this is latently safe because the FFmpeg-heavy mutators
   (rescale, filter, encode) **produce new frames** rather than mutate inputs in
   place — an accident of the current node set, not an enforced rule. Native Rust
   nodes that mutate pixels/samples in place must honour CoW explicitly.

3. **`side_data` content is shared; do not mutate it in place across a fan-out.**
   `metadata` (the dict) is deep-copied by `av_frame_ref` and is safe to mutate per
   holder, but `side_data` *content* is shared-refcounted — mutating it in place
   races with other consumers. §8.2's mention of "mutates metadata/side-data"
   addresses only **lifetime** (don't unref Rust's ref); it does **not** license
   in-place side-data content edits on a shared frame.

**C++ bumping the refcount is the safe case.** If a shim node calls `.raw()` and
does its own `av_frame_ref` to retain a copy, that is a new atomic ref balanced by
its own unref on a *distinct* struct — it cannot corrupt Rust's ref or free Rust's
struct. The only C++-side hazards are the same two above (freeing a ref it does not
own; in-place-mutating a shared buffer/side-data), which the avcpp move/copy
semantics the shim mirrors already handle for idiomatic node code.

**Test requirement (extends the refcount unit tests in `rust_refactor_headers.md`).**
The Phase-1 refcount tests must explicitly cover fan-out: push one frame to N edges,
drop all N wrappers, and assert via `av_buffer_get_ref_count` that the buffer
refcount returns to its exact pre-push value with no double-free (run under
ASan/`valgrind`). Add a matching wrap→mutate-in-place→verify-CoW case.

### 8.3 What still needs edits

Nodes that reach for Team singletons (`InstanceSharedObjects<RealTimeTeam>::get`)
or the seek interfaces must switch to the core equivalents — but those are exactly
the scheduler-adjacent nodes we're rewriting anyway. Pure processing nodes should
need only an include swap + recompile.

---

## 9. pyplumber re-bridging (explicit workstream)

`pyplumber` currently builds `_avplumber.cpython-*.so` via pybind11 against the C++
core (`avplumber_pybind.cpp`), and there are `IPythonNode` nodes that call into
Python (`src/nodes/python/*`). Two options:

1. **Keep pybind11, retarget it at the C ABI.** Least churn; the Python-facing API
   is unchanged. The `IPythonNode` nodes become C++ shim nodes (§8) that hold a
   `py::object` and are registered through the factory ABI.
2. **PyO3 wrapping the Rust core directly.** Cleaner Rust→Python story, but rewrites
   the binding layer and the Python node bridge.

Recommendation: **(1) for parity, consider (2) later.** Either way this is real
work (GIL handling around node start/stop already needs care — see the Python
lock-drop in `NodeWrapper::stop`), so it is scheduled as its own phase, not assumed
free.

---

## 10. Phasing (parity first)

The core swap is the one monolithic risky step; node porting is incremental. So we
prove the ABI/shim against the *hardest real thing* before writing any Rust node.

> **The native data model belongs in Phase 0/1, before any node is ported**, because
> it changes the edge's payload *type* — building the edge on raw pointers first means
> writing the refcount code twice and deleting it once.
>
> - **Phase 0:** add `rsmpeg` (0.18.0, feature `ffmpeg8` — requires `rusty_ffmpeg`
>   `0.16.7`, which resolves to the version already pinned, so one copy of the FFI
>   types resolves); define `Media`, the `Spec` enum, `Ts`, `FrameExt`/`PacketExt`;
>   split `Flow` into `Push`/`Tick`/`Blocked`; fix the two skeleton defects below. The
>   2-node prototype runs on `Media` internally with the ABI as its outer skin.
> - **Phase 1:** the `FfiNode` adapter and its three bridges (timebase, `isComplete`,
>   `Spec` projection), plus `avp_edge_take` (§4.3), land alongside the shim. This is
>   where the timebase/`isComplete` correctness work concentrates, so the **parity
>   harness must be running here** — this is the phase whose whole purpose is diffable
>   behaviour. The harness needs a real `SyncGroup` implementation for determinism
>   (the skeleton's `SyntheticClock` is a stub returning `AVP_NOPTS`); that is
>   Phase-1 work, not free.
> - **Phase 3** is correspondingly cheaper: typed pads and `NodeSpec` land with the
>   builder, and `split`/`one_to_many` are move-checked by the compiler (§8.2.1).
>
> **Two defects in the current `avplumber_f7k` skeleton are Phase-0 prerequisites,
> not side quests:** (1) it does not compile — `exec/blocking.rs:42` calls
> `log::warn!` with no `log` dependency; (2) `graph/buffer.rs`'s `retain` calls
> **both** `av_frame_ref` (`:117`) and `av_packet_ref` (`:120`) with one argument
> when the signatures are `(dst, src)`, so the `ffmpeg` feature has evidently never
> been compiled. (2) becomes moot once `Drop` replaces those functions.

**Phase 0 — ABI + dataflow model spec.** Finalize `avplumber_core.h` (§4),
*including the graph-management ABI surface (§4.7)* — not just the node/edge
ops, since pyplumber (§9) and OBS embed (§12 item 4) drive the core via that surface.
Finalize the `EdgeItem` model + master clock (§3) and the interface-ID table.
Prototype a trivial 2-node graph (demux → null_sink) end-to-end through the C
ABI, driven via the §4.7 ABI directly (no TCP server yet).

**Phase 1 — Rust core running existing C++ nodes via the shim, to parity.** Build:
runtime graph + edges (`BufferedEdge` first), TCP control protocol (`serde_json`),
factory registry, blocking-thread scheduler, one tokio current-thread runtime, `NodeGroup`
supervisor with topo-sort + auto_restart. Ship the compat header (§8). **Target
pipeline:** seekable `input_rec` → decode → `realtime` → encode/mux, run to parity
against the C++ core. During this window the *old* 4-phase seek still works through
the shim (Teams still exist as instance-shared C++ objects reached via the generic
registry). This validates raw-frame FFI, `query_interface`, and `.raw()` refcounts
on genuinely avcpp-heavy nodes while you can still diff behavior.

**Phase 2 — Causal control playback + Teams as core services.** Implement
`FlushStart/FlushStop/Spec` (Spec latched on the edge, §4.4.1), the `SyncGroup` master clock (rate/offset/pause read
at the output) + the `AvpTimeline` shared store, and reimplement the four Teams +
`SharedTimeline` + `input_rec`'s SeekIndex/timeline natively (§6). Cut the target pipeline over from the old 4-phase seek to the
flush+clock-reset discontinuity; delete `IFlushAndSeek`, the flush flags, and
`speedChanged` once no shimmed node depends on them. (Brief window where old and new
playback coexist.)

**Phase 3 — Native Rust node authoring.** `Transform`/`FlowNode` traits, the
`#[avp_node]` proc macro, `linkme` registration (§7). Port simple leaf nodes
(`firewall`, `null_sink`, `split`, `one_to_many`) as proof.

**Phase 4 — Async executor hardening + DirectEdge fusion.** Move non-blocking
nodes to the async model (§5.2); enable `DirectEdge` for co-located chains (§3.5);
benchmark latency vs the C++ EventLoop, especially on the mixer-tick path.

**Phase 5 — pyplumber (§9).**

**Phase 6 — Incremental leaf-node porting.** Strangler-fig. CUDA/GL/TensorRT nodes
stay C++ indefinitely behind the shim — no reason to move them.

**Phase 7 — Retire C++ core management.** Once nothing depends on it. `.avplumber`
scripts and the control protocol remain compatible throughout.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Interface discovery across `.so` (RTTI doesn't cross) | `query_interface` + closed interface-ID table; shim autogenerates it from `dynamic_cast` (§4.4, §8) |
| Replacing Teams/4-phase seek without live-stream regressions | Causal dataflow preserves the existing math (`scalePTS`/`offset_` → master clock applied at the output); parity-first phasing keeps a diffable reference (§3, §6, §10) |
| `.raw()` refcount corruption through the shim | One-ref-per-delivery ownership contract mirroring avcpp move/copy semantics (§8.2) |
| Fan-out double-free / struct-level data race in native `split`/`one_to_many` | Distinct-struct-per-edge + CoW + no in-place side-data mutation invariants; fan-out refcount test under ASan (§8.2.1) |
| Clock jitter on the mixer-tick path | Mixer tick delivered as a `tokio::sync::Notify` event on a current-thread runtime — `tokio::time` is off by default, so the timer wheel never touches the frame-perfect path (§5.2) |
| Stateful node reset (decoder DPB, rescaler, FRUC) on seek | `FlushStart` preempts the pipe: clears queues on the way down and is handled *inside* each stateful node, where the state lives (§3.4) |
| Responsive live control (speed/pause) despite queue latency | Rate/offset/pause live on the master clock, applied at the output — O(1), no in-flight rewrite; topology (heavy buffers upstream of the control point) keeps any residual latency small (§3.3) |
| Node/group restart during live op | No sticky per-stream state to miss: timeline is on the core-owned master clock; current format is read from the edge's latched `Spec` (§4.4.1, no pull walk); producer restart reuses the flush discontinuity (§3.6) |
| avcpp depth (`.raw()` ×391 in `src/nodes/`) | Nodes already operate at raw-struct level; pass `AVFrame*`/`AVPacket*` and keep avcpp C++-side (§2) |
| `DECLNODE_ATD` template dispatch | Runtime media tag + one factory registration per concrete type (§8.1) |
| CUDA/GL/TensorRT unportable to Rust | They stay C++ behind the shim, permanently (§10 Phase 6) |
| pyplumber binding churn | Retarget pybind11 at the C ABI first; PyO3 optional later (§9) |
| Per-frame FFI overhead | ~100 calls/s/node; measured early; batch only if ever >1% (§1) |

---

## 12. Open questions

1. **Reverse (against-the-flow) format hints.** `Spec` (§4.4.1) and the whole §4.4
   model flow *downstream*. Two interfaces go the other way: `IPreferredFormatReceiver`
   lets a downstream node push a preferred pixfmt/resolution *upstream* (see
   `rescale_video` ctor), and `INeedsOutputFrameSize` pushes an audio frame size
   upstream from the encoder. These are the only capabilities `Spec` does *not*
   absorb. Options: (a) a direct `query_interface` on the *immediate downstream*
   node the producer already holds (no walk — a producer knows its consumer edge);
   (b) an upstream "hint" token type mirroring `Spec` backwards. Leaning **(a)**:
   it's a single named-neighbour query, needs no against-the-flow token, and stays
   consistent with "no graph walk" — the producer asks its one downstream, not "walk
   down until someone answers." Confirm no case needs the hint to travel more than
   one hop.
2. **DirectEdge + control tokens.** When a chain is fused, do tokens still
   materialize as `EdgeItem`s or become direct method calls (`on_event`)? Probably
   direct calls, with the same handler the buffered path uses. Note flush must still
   preempt (clear) any queue on a `BufferedEdge` leg of the same chain.
3. **MetadataFrame across FFI.** `MetadataFrame` is a custom C++ type; the
   `AvpMediaVtable` retain/release handles ownership, but nodes that *read* its
   fields need either a C accessor surface or to stay C++-only. Enumerate which
   nodes touch metadata payload vs. pass it through.
   The opacity here is irreducible: `MetadataFrame` holds an
   avcpp `av::Timestamp` plus a `Parameters` JSON-ish object
   (`src/metadata_frame.hpp:8`), and `EglImageFrame` holds GL/DMA-BUF state
   (`src/hwaccel/EglImageFrame.hpp:18`) — neither is expressible as an owned Rust
   type. The native `Media::Opaque` variant gives both a **lifetime-safe** home
   (RAII over the retain/release vtable, so `Drop`/`Clone` are correct inside the
   Rust core) but **not field access**. A Rust node still cannot read these fields
   without a new C accessor surface. Since `Opaque` carries no `AVFrame`, that
   variant also carries `time_base` explicitly (matching `EglImageFrame`'s own
   `tb_`, `EglImageFrame.hpp:45`). The enumeration is still owed.
4. **OBS embed (`EMBED_IN=obs`).** Where does the embed boundary sit relative to the
   C ABI? Likely the OBS plugin links the Rust core as a static lib and registers
   OBS-specific nodes via the same factory ABI. Confirm no reliance on the C++ core
   object model.
5. **Reverse playback.** A master-clock `rate < 0` handles presentation ordering,
   but reverse decode requires GOP-granular buffering the current code handles
   specially (`discardUntil(ts=0)`); confirm the InputReader owns this, not
   individual nodes.
6. **Minimal embedder ABI surface.** §4.7 lists the graph-management calls and
   §4.8 the service-discovery mechanism an embedder uses to drive the core
   without the TCP protocol (required by pyplumber §9 and OBS §12 item 4). What is
   the *smallest* such surface that covers all three known embedders (TCP
   server, pyplumber, OBS) without leaking node-internal concerns upward? In
   particular: does an embedder need to inspect/select `AvpEdgeCoupling` (§3.5)
   at create-edge time, or is the core's auto-assignment (§5.3) always
   sufficient? Does group lifecycle need a "drain without stop" op for graceful
   reconfiguration, or is stop→reconfigure→start enough? And which core services
   (§6) get a service ID vs. stay internal-only — e.g., is `SeekIndex` a service
   an embedder reaches, or an internal helper of `InputReader`? Resolve in
   Phase 0 alongside the rest of `avplumber_core.h` and the service-header split.
