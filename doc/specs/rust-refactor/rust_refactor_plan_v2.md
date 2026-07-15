# Rust Core Refactor Plan for avplumber — v2

> Supersedes `rust_refactor_plan.md`. v1 remains a valid high-level sketch; this
> document keeps its migration spirit but rewrites the parts that primary-source
> reading of the code showed to be wrong, under-specified, or improvable.
>
> This version was written after reading: `graph_core.hpp`, `graph_base.hpp`,
> `graph_interfaces.hpp`, `graph_mgmt.{hpp,cpp}`, `EventLoop.hpp`, `TickSource.hpp`,
> `edge_types.hpp`, the four Team headers (`RealTimeTeam`, `PauseControlTeam`,
> `SpeedControlTeam`, `InputSeekTeam`), and the scheduler-adjacent nodes
> (`input_rec.cpp`, `realtime.cpp`, `speed.cpp`, `pause.cpp`, `rescale_video.cpp`).

---

## 0. What changed from v1

Seven concrete changes, each justified in the section noted:

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

4. **Scheduling is "hybrid OS threads + a custom single-threaded executor per
   clock domain," not "tokio."** Blocking codec/IO/CUDA calls need real threads;
   the clock/tick path needs jitter-free timing tokio's timer wheel doesn't give.
   Async/await is still adopted for non-blocking nodes — that is the big
   boilerplate + safety win. (§5)

5. **Phases are reordered: run the *existing C++ nodes* on the Rust core to parity
   first**, before porting any node to Rust. The core swap is the one monolithic
   risky step; node porting is naturally incremental (strangler-fig). (§10)

6. **Raw `AVFrame*`/`AVPacket*` cross the FFI, and avcpp is reconstituted inside
   the C++ shim** — not avcpp objects across the boundary. Validated by `.raw()`
   being used 393× in nodes: they already operate at the raw-struct level. (§2, §8)

7. **pyplumber re-bridging is an explicit workstream**, not an afterthought. (§9)

---

## 1. Feasibility verdict (revised)

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
nodes already reach through `.raw()` **393×** — they operate at the raw-struct
level constantly, so `.raw()` returning the exact pointer Rust passed around is
*alignment*, not friction.

### 2.2 avcpp lives in the shim (see §8)

The C++ node shim wraps an incoming `AvpBuffer.ptr` into the appropriate avcpp
object such that `.raw()` returns that same pointer and refcount stays balanced.
On output it extracts `.raw()` and hands the pointer back. Existing node bodies —
including `av::VideoRescaler`, side-data walks, `av_dict_copy` — are unchanged.

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
buffers that follow it is intrinsic to the pipe. (v1's epoch scheme was compensating
for a design that let stale buffers linger; clearing on flush removes the problem
rather than tracking it.)

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

### 4.1 Handles and media vtables

```c
typedef struct AvpCore      AvpCore;       // the Rust core / one instance
typedef struct AvpNode      AvpNode;       // a node instance (Rust- or C++-backed)
typedef struct AvpEdge      AvpEdge;       // an edge endpoint
typedef struct AvpExecutor  AvpExecutor;   // a clock domain / executor (was EventLoop)

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

// Consumer side (non-blocking: timeout_ms==0 returns immediately)
int     avp_edge_peek(AvpEdge*, int timeout_ms, AvpBuffer* out); // 1 got,0 none
int     avp_edge_peek_event(AvpEdge*, AvpEdgeEvent* out);        // 1 got,0 none
void    avp_edge_pop(AvpEdge*);
int     avp_edge_occupied(AvpEdge*);

// Wakeup registration for non-blocking nodes (replaces processWhenSignalled).
// The core re-polls the node when the edge becomes readable / writable.
void    avp_edge_notify_readable(AvpEdge*, AvpNode*);
void    avp_edge_notify_writable(AvpEdge*, AvpNode*);
```

### 4.4 Capability discovery — three registers, not one graph walk

The ~30 interfaces in `graph_interfaces.hpp` are discovered today by
`dynamic_cast` + `findNodeUp<T>()` — a per-call upstream walk. v1 modelled this
1:1 as `avp_find_interface_up` (a walk across the FFI). That is **wrong**: it's
fragile (a downstream node holding an upstream interface pointer while the producer
stops/restarts — e.g. `decoders.cpp`'s `input_hold_` workaround, which the source
comments admit is a hack), and it's a false economy (one "walk-and-cast" mechanism
serving three unrelated jobs). Reading the actual call sites shows the interfaces
split cleanly into **three registers**, each with its own, simpler mechanism. No
upstream graph walk survives.

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
`IDecoder::discard_until`, `IEncoder`, `IMuxer`, `ISentinel` stats,
`IReturnsObjects`/`IInputsObjects` (`node.param.get/set`), `IStreamsInput` demux
enumeration, `IJackSink` pull. These are real runtime calls, but every caller
already knows *which* node it means (the control protocol addresses a node by name;
an adjacent node talks to its immediate neighbour). So they keep a **direct**
query — `avp_node_query_interface(node, id)` — on an explicitly-referenced node.
There is no traversal: you query the node you already hold, not "walk up until
something answers."

```c
typedef uint32_t AvpInterfaceId;   // stable, closed enum — see table below

// Direct capability query on a node the caller already holds (control protocol,
// adjacent node, or self). This is the ONLY interface-discovery primitive.
// There is NO avp_find_interface_up: Facts travel as Spec on the edge (§4.4.1),
// Controls are addressed to core services (§6), so no upstream walk exists.
const void* avp_node_query_interface(AvpNode*, AvpInterfaceId);
```

Reclassification of the v1 table (18 ids → registers):

| v1 Interface (C++)         | Register | New mechanism |
|----------------------------|----------|---------------|
| `IVideoFormatSource`       | Fact     | `Spec` (video) latched on edge |
| `IAudioMetadataSource`     | Fact     | `Spec` (audio) latched on edge |
| `IFrameRateSource`         | Fact     | `Spec` field |
| `ITimeBaseSource`          | Fact     | `Spec` field |
| `IPlaybackControl`         | Control  | `SyncGroup` rate sign (§6) |
| `IInputReset`              | Control  | rides `FlushStart` (§3.4) |
| `IDecoder`                 | Query    | direct `query_interface` (named node) |
| `IEncoder`                 | Query    | direct `query_interface` |
| `IMuxer`                   | Query    | direct `query_interface` |
| `ISentinel`                | Query    | direct (stats) |
| `IReturnsObjects`/`IInputsObjects` | Query | direct (`node.param` bridge) |
| `IStreamsInput`            | Query    | direct (demux enumeration) |
| `IJackSink`                | Query    | direct (adjacent pull) |
| `INeedsOutputFrameSize`    | reverse  | see §12 (downstream→upstream hint) |
| `IPreferredFormatReceiver` | reverse  | see §12 (format negotiation direction) |
| `IFrameNumber` / `IFrameTimestamp` | — | were read by the deleted `speedChanged` sync walk; now per-frame buffer metadata, not a capability. Candidate for removal (§12) |

The surviving `AvpInterfaceId` enum therefore lists only the **Register-3** ids
(headers §interface discovery). Facts moved to `Spec`; Controls moved to services.

#### 4.4.1 `Spec` — Facts as latched edge state (replaces the upstream walk)

`Spec` is an `EdgeItem` variant (§3.2) — the resolved description of the frames
flowing through an edge (video: w/h/pixfmt; audio: rate/fmt/layout; plus frame
rate, time base). Not GStreamer "caps": there is no capability set and no
negotiation — it is a single concrete value asserting "this is what flows here
now," forward-only, updated when it changes.

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

### 4.5 Factory registration (replaces `DECLNODE` + `generate_node_list`)

```c
typedef AvpNode* (*AvpNodeFactoryFn)(AvpCore*, const char* json_params,
                                     const AvpNodeVtable** out_vtable);

void avp_register_node_factory(AvpCore*, const char* type_name, AvpNodeFactoryFn);
```

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
// -> IInputsObjects). No C ABI needed by nodes for the protocol itself.
```

---

## 5. Scheduling

### 5.1 Two execution models, kept — but reframed

Today (confirmed in `graph_mgmt.cpp`): blocking nodes get a `std::thread` running
`NodeWrapper::threadFunction`; non-blocking nodes (`NonBlockingNodeBase`) run on a
shared `EventLoop` (`poll()` reactor over eventfd) or are driven by a `TickSource`.
`NodeGroup` runs a management-thread state machine (`EMPTY/STOPPED/STARTED/MIXED/
RESTART/FINISH_THREAD`) and topologically sorts nodes for ordered start/stop.

We keep the two models but change *how* the non-blocking side is written and timed.

### 5.2 Hybrid, not tokio

- **Blocking nodes → real OS threads.** You cannot `.await` inside
  `avcodec_send_packet`, a CUDA kernel launch, or a blocking `read()`. Each blocking
  node owns a thread; the body is a straight loop calling `process()`.
- **Non-blocking nodes → a custom single-threaded executor per clock domain.**
  Async/await is adopted here because it is precisely where the current design is
  weakest — the docs warn that stateful non-blocking nodes are hard
  (`processWhenSignalled` + `weak_ptr` capture dance, the "hidden queue of size 1",
  UAF footguns). An `async fn` holds state across `.await` points for free; the
  compiler builds the state machine. This is the single biggest boilerplate + safety
  win and directly serves the project's "avoid boilerplate" value.
- **Why not default tokio:** the original motivating bug was frame-perfect output
  driven by the mixer's tick. tokio's timer wheel adds jitter. We run a bespoke
  single-threaded executor whose timer is driven either by a `timerfd`/wallclock or
  by an **external tick callback from the mixer**. The tick becomes a `Stream` a
  node `select!`s on:

  ```rust
  async fn run(mut self, ctx: NodeCtx) -> Result<()> {
      loop {
          select! {
              _   = ctx.tick.next()      => self.on_tick(&ctx).await?,
              buf = ctx.input.next()     => match buf {
                  Item::Buffer(b) => self.on_buffer(b, &ctx).await?,
                  Item::Event(e)  => self.on_event(e, &ctx).await?,  // Flush/Spec/Eof
              }
          }
      }
  }
  ```

  This keeps async ergonomics without inheriting an opaque scheduling policy. (The
  existing `EventLoop::fastExecute` inline-drain trick — run callbacks synchronously
  when you can grab the lock — maps to the executor polling ready tasks before
  parking.)

### 5.3 Group management and edge-type selection

- `NodeGroup`'s state machine, topological sort, and `auto_restart`/`on_error`
  (`restart_node`/`restart_group`/`panic`/`exit`) move into the core as a Rust
  supervisor. Ordered start/stop is preserved.
- During graph construction the core assigns each node to an execution context
  (a blocking thread, or a named executor/clock domain). For an edge, if both
  endpoints share an executor and neither is blocking → `DirectEdge`; otherwise
  `BufferedEdge`. This is the co-location decision v1 described; the point to hold
  is that it is derived from execution context, not guessed.

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
  service with its own ABI (`avp_timeline_*`, headers §shared timeline).

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
| `SharedTimeline` `set/get/gc` keyed by PTS | core `AvpTimeline` service; control writes, C++ `TimelineReader` nodes read the value in effect at a frame's PTS (§shared-timeline ABI) |
| `RealTimeSpeed` reading `AVFrame::metadata` `frame_ts/frame_no/wallclock` | source-time PTS on the frame + the master clock, not dict round-trips |

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
  clock like any other. The `realtime` replacement no longer does `av_dict_get` +
  `atoll` per frame.

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

- `Source<T>::get/peek/pop` → `avp_edge_peek/pop` + wrap the returned
  `AvpBuffer.ptr` into `av::VideoFrame`/`av::Packet`/`av::AudioSamples` **adopting**
  the pointer so `.raw()` returns it and refcount is balanced (retain on wrap,
  release on wrapper drop; the buffer Rust handed us already carries a ref).
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

## 10. Phasing (reordered to de-risk)

The core swap is the one monolithic risky step; node porting is incremental. So we
prove the ABI/shim against the *hardest real thing* before writing any Rust node.

**Phase 0 — ABI + dataflow model spec.** Finalize `avplumber_core.h` (§4), the
`EdgeItem` model + master clock (§3), and the interface-ID table. Prototype a
trivial 2-node graph (demux → null_sink) end-to-end through the C ABI.

**Phase 1 — Rust core running existing C++ nodes via the shim, to parity.** Build:
runtime graph + edges (`BufferedEdge` first), TCP control protocol (`serde_json`),
factory registry, blocking-thread scheduler, one custom executor, `NodeGroup`
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

## 11. Risks & mitigations (revised)

| Risk | Mitigation |
|---|---|
| Interface discovery across `.so` (RTTI doesn't cross) | `query_interface` + closed interface-ID table; shim autogenerates it from `dynamic_cast` (§4.4, §8) |
| Replacing Teams/4-phase seek without live-stream regressions | Causal dataflow preserves the existing math (`scalePTS`/`offset_` → master clock applied at the output); parity-first phasing keeps a diffable reference (§3, §6, §10) |
| `.raw()` refcount corruption through the shim | One-ref-per-delivery ownership contract mirroring avcpp move/copy semantics (§8.2) |
| Fan-out double-free / struct-level data race in native `split`/`one_to_many` | Distinct-struct-per-edge + CoW + no in-place side-data mutation invariants; fan-out refcount test under ASan (§8.2.1) |
| Clock jitter on the mixer-tick path | Bespoke single-threaded executor with external-tick-driven timer, not tokio's timer wheel (§5.2) |
| Stateful node reset (decoder DPB, rescaler, FRUC) on seek | `FlushStart` preempts the pipe: clears queues on the way down and is handled *inside* each stateful node, where the state lives (§3.4) |
| Responsive live control (speed/pause) despite queue latency | Rate/offset/pause live on the master clock, applied at the output — O(1), no in-flight rewrite; topology (heavy buffers upstream of the control point) keeps any residual latency small (§3.3) |
| Node/group restart during live op | No sticky per-stream state to miss: timeline is on the core-owned master clock; current format is read from the edge's latched `Spec` (§4.4.1, no pull walk); producer restart reuses the flush discontinuity (§3.6) |
| avcpp depth (`.raw()` ×393) | Nodes already operate at raw-struct level; pass `AVFrame*`/`AVPacket*` and keep avcpp C++-side (§2) |
| `DECLNODE_ATD` template dispatch | Runtime media tag + one factory registration per concrete type (§8.1) |
| CUDA/GL/TensorRT unportable to Rust | They stay C++ behind the shim, permanently (§10 Phase 6) |
| pyplumber binding churn | Retarget pybind11 at the C ABI first; PyO3 optional later (§9) |
| Per-frame FFI overhead | ~100 calls/s/node; measured early; batch only if ever >1% (§1) |

---

## 12. Open questions (carried + new)

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
4. **OBS embed (`EMBED_IN=obs`).** Where does the embed boundary sit relative to the
   C ABI? Likely the OBS plugin links the Rust core as a static lib and registers
   OBS-specific nodes via the same factory ABI. Confirm no reliance on the C++ core
   object model.
5. **Reverse playback.** A master-clock `rate < 0` handles presentation ordering,
   but reverse decode requires GOP-granular buffering the current code handles
   specially (`discardUntil(ts=0)`); confirm the InputReader owns this, not
   individual nodes.
