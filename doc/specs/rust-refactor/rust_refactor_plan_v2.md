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

1. **Interface discovery is a first-class subsystem, not a table row.** The graph
   is threaded with ~30 optional capabilities discovered by `dynamic_cast` and
   `findNodeUp<T>()`. RTTI does not cross a C-ABI/`.so` boundary. We add an
   explicit `query_interface(node, iface_id)` mechanism. (§4, §5)

2. **In-band edge events/segments are the primary playback mechanism**, with
   epochs as a secondary cheap-staleness check. This deletes `IFlushAndSeek`, the
   4-phase seek handshake, and the per-edge `flushing_`/`flushed_` flags. Events do
   **not** replace the capability interfaces — the data-plane push (events) and the
   pull-plane query (interfaces) are complementary and both are kept, which is what
   makes node restart during live op safe. (§3, §3.6, §4.4, §6)

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
1. Interface discovery across the boundary (mitigated by §4's `query_interface`).
2. The Teams + 4-phase seek being replaced without behavior regressions on live
   streams (mitigated by §3/§6's event model + the §10 parity-first ordering).
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
    uint64_t     epoch;    // staleness/seek generation (see §3.4)
} AvpBuffer;
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

## 3. Edge model: `Buffer | Event` streams (the keystone)

This is the central design change. It replaces: EOF markers, `IFlushAndSeek`, the
4-phase seek, per-edge flush flags, in-flight PTS rescaling, and the
`AVFrame::metadata` timeline smuggling in `input_rec`.

### 3.1 The insight

The codebase **already** carries one in-band control signal on edges: EOF, via
`createEofMarker`/`isEofMarker`/`consumeEofIfPresent`. Everything else
(seek/flush/speed/pause/custom-timeline) is done *out of band* — by walking the
graph with RTTI, flipping atomics, and broadcasting through Team singletons. We
generalize the one thing that already works: **an edge carries an ordered stream
of `Buffer` or `Event`.** This is the GStreamer buffers+events+segments model, and
adopting its shape is what makes eventual GStreamer interop tractable.

### 3.2 The event set

```rust
enum EdgeItem {
    Buffer(AvpBuffer),
    Event(EdgeEvent),
}

enum EdgeEvent {
    /// End of stream. Replaces the EOF marker. Flows downstream; a node flushes
    /// its internal state, forwards Eof, and finishes.
    Eof,

    /// Begin discarding. Downstream drops Buffers (and flushes internal codec/
    /// filter state) until FlushStop. Replaces startFlushing() + the pause
    /// handshake.
    FlushStart { epoch: u64 },

    /// Resume normal flow. Carries the epoch that subsequent Buffers will bear.
    FlushStop { epoch: u64 },

    /// STICKY. Timeline definition. Everything after this event is interpreted
    /// relative to it. Replaces: SpeedControlTeam::scalePTS remap, RealTimeTeam
    /// offset broadcast, and input_rec's ts_offsets_ / setFrameMetadataTimestamps.
    /// Cached on the edge and replayed to a (re)connecting consumer (§3.6).
    Segment(Segment),

    /// STICKY. Stream format (caps). Sent before the first Buffer and whenever
    /// format changes. Complements (does not replace) the IVideoFormatSource /
    /// IAudioMetadataSource pull queries. Cached on the edge and replayed to a
    /// (re)connecting consumer, so a restarted node never misses it (§3.6).
    Caps(StreamCaps),
}
```

`Segment` is the workhorse. It is exactly the piecewise-linear remap
`SpeedControlTeam::scalePTS` already computes, promoted to a first-class value:

```rust
struct Segment {
    time_base:   Rational,    // units of the ts fields below
    base:        i64,         // output PTS assigned to `start`
    start:       i64,         // first valid source PTS in this segment
    rate:        f64,         // playback rate; negative = reverse
    // clock domain this segment is synchronized against (RealTimeTeam offset_).
    // Multiple streams sharing a sync_group stay buffered together.
    sync_group:  SyncGroupId,
    // which source timeline these timestamps express (input_rec ETimestampSource)
    ts_source:   TimestampSource, // Input | Wallclock | SyncTime
}
```

A node maps a buffer PTS to output PTS with pure local arithmetic:
`out = base + ((pts - start) as f64 / rate) rescaled to output tb`. No graph walk,
no Team broadcast, no `av_dict_set`/`av_dict_get` round-trip per frame.

### 3.3 Seek, redrawn

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

New (declarative, in-band, no pause handshake):

```
1. Core bumps the sync_group epoch: E -> E+1.
2. Source node (input_rec's replacement) emits, in-band on its output edge:
      Event(FlushStart { epoch: E+1 })
   Every downstream node, on receiving FlushStart:
      - discards queued Buffers with epoch < E+1
      - flushes its own internal state (decoder DPB, rescaler, FRUC buffers)
      - forwards FlushStart, returns to idle for that stream
   Stateful reset thus happens *where the state lives*, triggered by one event.
3. Source repositions (seek table lookup, byte/frame/ts resolution — the
   input_rec logic, now a core InputReader method, §6.3).
4. Source emits:
      Event(Segment { base, start, rate, sync_group, ts_source })
      Event(FlushStop { epoch: E+1 })
      Buffer(..., epoch: E+1), Buffer(..., epoch: E+1), ...
```

No `pauseProcessing()`, no cross-node mutex barrier, no ordering contract between
four separately-invoked phases. Ordering is intrinsic: events cannot overtake the
buffers they follow because they share the queue.

### 3.4 Epochs as the cheap check

Every `AvpBuffer` and the Flush events carry a `u64 epoch`. A node's fast path is:
`if buf.epoch < self.current_epoch { drop }`. This handles the window between "seek
issued" and "FlushStart observed" for buffers already in flight past a node, and
lets a node cheaply reject stragglers without inspecting the event stream mid-batch.
Epoch is the optimization; the event stream is the mechanism. (v1 had this
inverted — epoch-only cannot flush a decoder's internal DPB.)

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

### 3.6 Sticky events and node restart during live operation

This is the one correctness hazard the event model introduces, and it must be
designed for — not left implicit.

**The problem.** Today the design is crash-tolerant *by accident*: every frame
re-carries its own truth. `rescale_video` re-probes format per frame
(`sourceChanged()`); `realtime` re-derives its offset and reads
`frame_ts`/`frame_no`/`wallclock` from `AVFrame::metadata` each frame. So when
`restart_node`/`restart_group` fires on error during live op (wired as `onFinished`
callbacks in `NodeManager::createNode`), the recreated node reconnects to its named
edges and picks state up from the next frame. Nothing is sticky, so nothing is
missed.

The event model breaks this: `Caps` and `Segment` are **sticky** — sent once,
upstream, before the first buffer. A node that restarts mid-stream has already
missed them. Naively it would then **stall** (waiting for a `Segment` that is never
re-sent) or **mis-time / mass-drop** (no timeline; wrong epoch → treats live
buffers as stale). That is a regression vs. today. Three mechanisms remove it, and
they compose:

**(1) Sticky-event caching + replay on (re)connection (GStreamer's model — hard
requirement of every `Edge` impl).** Each edge caches its current sticky set
(`Caps`, `Segment`, current epoch), updated **at dequeue time** so the cache always
reflects the last sticky event the consumer actually passed. When a restarted node
registers as the edge's consumer, the edge replays the cache **first**, in canonical
order (`Caps → Segment`), then drains the intact queue. Because events are in-band
and the queue preserves ordering, this is correct even across a mid-segment
boundary:

```
Before crash, consumer had processed:  Caps, SegA, buf1      cache = {Caps, SegA}
Queue still holds:                     [SegB, buf2(segB)]
Replay to the new consumer:            Caps, SegA, then SegB, buf2
  ⇒ buf2 correctly under SegB; buf1 lost (expected — it was in-flight/in the DPB)
```

For a `DirectEdge` (no queue) the cache alone is replayed, then direct calls resume.

**(2) Restart *is* a discontinuity — it reuses the seek path, no separate protocol.**
If the **producer** side restarts, on `start()` it self-issues
`FlushStart{epoch+1}` → re-probe → fresh `Caps`/`Segment` → `FlushStop`. That clears
stale downstream buffers and re-establishes format+timeline via the exact §3.3
mechanism. A **consumer** restart is covered by replay (1) from its still-live
upstream edge. A whole-**group** restart is just both at once: cross-boundary edges
replay from the still-running neighbour; intra-group edges get a fresh flush+segment
from their restarted producer.

**(3) Epoch counter and `SyncGroup` clock are core-owned, so they survive restart.**
The epoch monotonic counter lives in the core/`SyncGroup`, not the node — so a
post-restart epoch is strictly greater than any in-flight buffer's epoch (stale
drops are correct), and it does not reset to 0. Likewise the `SyncGroup` shared
clock/offset persists, so a restarted `realtime` node rejoins and re-syncs against
it (`ready_=false` → converge) exactly as today, but without re-deriving the clock
from scratch.

**(4) Interfaces are the second recovery path** (why §4.4 keeps them). Independently
of sticky replay, a restarted node can pull the current truth via
`findNodeUp<IVideoFormatSource>` / `<IPlaybackControl>` — belt-and-suspenders, and
the same query nodes already use to self-initialize. This is the concrete reason the
event model does **not** replace those interfaces.

**Requirement stated plainly:** sticky-event replay is mandatory in the `Edge`
implementation. Without it the event model is strictly *worse* than the current
design on crash tolerance; with it (plus 2–4) restart is safe and, in the
producer/group case, semantically identical to a localized seek.

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

### 4.4 Interface discovery — the subsystem v1 missed

The ~30 interfaces in `graph_interfaces.hpp` are load-bearing and discovered by
`dynamic_cast`/`findNodeUp<T>()`. RTTI does not cross `.so`. So:

```c
typedef uint32_t AvpInterfaceId;   // stable, closed enum — see table below

// Direct query on a node.
const void* avp_node_query_interface(AvpNode*, AvpInterfaceId);

// Graph walk (replaces EdgeBase::findNodeUp<T>). Rust owns traversal; at each hop
// it calls the node's query_interface across the FFI (node may be C++ or Rust).
const void* avp_find_interface_up(AvpEdge* from, AvpInterfaceId);
```

Each `AvpInterfaceId` names a plain-C vtable struct. Stable ID assignment for the
current interface set (closed enum; new interfaces append):

| Id | Interface (C++)            | Purpose |
|----|----------------------------|---------|
| 1  | `IDecoder`                 | codec name, `discard_until(pts)` |
| 2  | `IEncoder`                 | codec params, `set_output` |
| 3  | `IMuxer`                   | init from format context |
| 4  | `IVideoFormatSource`       | width/height/pixfmt (pull query; complements `Caps`) |
| 5  | `IAudioMetadataSource`     | sample rate/format/layout (pull query; complements `Caps`) |
| 6  | `IFrameRateSource`         | frame rate |
| 7  | `ITimeBaseSource`          | time base |
| 8  | `IPlaybackControl`         | direction, target conversion, playback-direction query |
| 9  | `IInputReset`              | reset internal state (pull path; complements `FlushStart`) |
| 10 | `IFrameNumber`             | current frame number |
| 11 | `IFrameTimestamp`          | current ts / wallclock / eof |
| 12 | `ISentinel`                | card/signal-present status for stats |
| 13 | `IPreferredFormatReceiver` | downstream format hints upstream |
| 14 | `INeedsOutputFrameSize`    | audio frame sizing |
| 15 | `IReturnsObjects`/`IInputsObjects` | node.param.get/set object bridge |
| 16 | `IStreamsInput`            | demux stream enumeration |
| 17 | `IJackSink`                | JACK pull callback |
| …  | (append-only)              | |

**Events and interfaces are complementary — both are kept.** Earlier drafts
proposed dissolving items 4/5/8/9 (`IVideoFormatSource`, `IAudioMetadataSource`,
`IPlaybackControl`, `IInputReset`) into the event model. That is reversed: they
remain first-class interfaces alongside the events. The two mechanisms serve
different planes:

- **Events (`Caps`/`Segment`/`FlushStart`) are the data-plane push** — ordered,
  per-stream, efficient, and how a running node normally learns format/timeline
  and flushes internal state.
- **Interfaces are the pull-plane query** — how a node *(re)initializes* by asking
  upstream for the current truth (`findNodeUp<IVideoFormatSource>`,
  `<IPlaybackControl>`), and how the control protocol and stats reach into a node.

This complementarity is what makes **restart during live operation** safe (§3.6): a
node that restarts mid-stream and missed the sticky `Caps`/`Segment` has a second
recovery path — pull current state via the interface — in addition to the edge's
sticky-event replay. It also matches how nodes already self-initialize today
(`rescale_video`'s ctor queries `IPreferredFormatReceiver`). So the full interface
table is retained long-term, not treated as a temporary compat shim.

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
                  Item::Event(e)  => self.on_event(e, &ctx).await?,  // Flush/Segment/Eof
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
  piecewise-linear PTS remap (`scalePTS`) — i.e. it already computes a `Segment`.
- `PauseControlTeam` holds `paused_` + a condvar, `pause_at_`, and lists of
  `IInputReset`/`IFlushAndSeek`/`IPlaybackControl` weak refs.
- `InputSeekTeam` fans `ISeekAt` add/clear to members.

Porting these as shimmed C++ nodes would drag the whole out-of-band, RTTI-driven,
Team-singleton machinery into the new core. Instead:

### 6.2 The mapping

| Old Team mechanism | New core mechanism |
|---|---|
| `RealTimeTeam::offset_` sync to smallest | `SyncGroup` clock domain in the executor; a shared clock the group's `Segment.base` is measured against |
| `RealTimeTeam::flushAndSeek*` 4-phase | core issues `FlushStart`/`Segment`/`FlushStop` on the source edge (§3.3) |
| `SpeedControlTeam::scalePTS` remap | `Segment { rate, base, start }` (§3.2); nodes apply local arithmetic |
| direction sign-flip broadcast | `Segment { rate < 0 }` |
| `PauseControlTeam` condvar | executor suspends the group's tasks; a paused group simply isn't polled |
| `pause_at_` | a scheduled core action that emits the pause at a target PTS |
| `InputSeekTeam` fan-out | core `seek_at` table keyed by `SyncGroup` |
| `RealTimeSpeed` reading `AVFrame::metadata` `frame_ts/frame_no/wallclock` | fields on `Buffer`/`Segment`, not dict round-trips |

### 6.3 `input_rec` becomes a core `InputReader`

`input_rec.cpp` mixes three concerns: (a) demux/read loop, (b) a **seek-table**
subsystem (`seek_table_`, background `seekThreadFun`, byte/frame/ts resolution),
and (c) a **custom-timeline** subsystem (`ETimestampSource`, `ts_offsets_`,
`setFrameMetadataTimestamps` writing `av_dict_set`). In the new core:

- (a) stays a node (it's genuine IO), backed by a real thread.
- (b) the seek table + `resolveSeekTarget` become a core `SeekIndex` service the
  node owns; `StreamTarget` resolution (frame/live/end/bytes/wallclock/sync) is
  core code.
- (c) `TimestampSource` and the offset table become part of `Segment` emission.
  The node emits a `Segment { ts_source, base, start, ... }` instead of stamping
  every frame's dict. Downstream nodes that need the custom timeline read it from
  the current segment; the `realtime` replacement no longer does `av_dict_get` +
  `atoll` per frame.

### 6.4 What gets deleted

`IFlushAndSeek` (interface + the `NodeSingleInput` 4-phase impl,
`graph_base.hpp:84-170`); `EdgeBase::flushing_/flushed_` and
`startFlushing/stopFlushing/maybeFlush`; the EOF-marker special path (folded into
`EdgeEvent::Eof`); the `pauseProcessing()`/`lockProcessing()` cross-node barrier
used only by seek. `Node::pauseProcessing` stays for the executor's group-suspend.

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
`Source<T>`/`Sink<T>`, `createCommon`, `NodeCreationInfo`, the `DECLNODE*` macros,
and `findNodeUp<T>()` — but backed by the C ABI instead of the old core.

- `Source<T>::get/peek/pop` → `avp_edge_peek/pop` + wrap the returned
  `AvpBuffer.ptr` into `av::VideoFrame`/`av::Packet`/`av::AudioSamples` **adopting**
  the pointer so `.raw()` returns it and refcount is balanced (retain on wrap,
  release on wrapper drop; the buffer Rust handed us already carries a ref).
- `Sink<T>::put` → extract `.raw()`, build an `AvpBuffer`, `avp_edge_push`;
  translate `BACKPRESSURE` to the old `put(..., drop_if_full)` bool contract.
- `DECLNODE(nodetype, Class)` → emits `avp_register_node_factory("nodetype",
  &Class::__avp_factory)` + a generated `query_interface` that maps the closed
  interface-ID table to `dynamic_cast<IFoo*>(this)`. So a node that today does
  `class Foo : public NodeSISO<...>, public IDecoder` keeps working: the shim's
  generated `query_interface(ID_IDecoder)` returns a C vtable trampolining to the
  node's virtual methods.
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

**Phase 0 — ABI + event model spec.** Finalize `avplumber_core.h` (§4), the
`EdgeItem`/`Segment` model (§3), and the interface-ID table. Prototype a trivial
2-node graph (demux → null_sink) end-to-end through the C ABI.

**Phase 1 — Rust core running existing C++ nodes via the shim, to parity.** Build:
runtime graph + edges (`BufferedEdge` first), TCP control protocol (`serde_json`),
factory registry, blocking-thread scheduler, one custom executor, `NodeGroup`
supervisor with topo-sort + auto_restart. Ship the compat header (§8). **Target
pipeline:** seekable `input_rec` → decode → `realtime` → encode/mux, run to parity
against the C++ core. During this window the *old* 4-phase seek still works through
the shim (Teams still exist as instance-shared C++ objects reached via the generic
registry). This validates raw-frame FFI, `query_interface`, and `.raw()` refcounts
on genuinely avcpp-heavy nodes while you can still diff behavior.

**Phase 2 — Event/segment playback + Teams as core services.** Implement
`FlushStart/FlushStop/Segment/Caps`, `SyncGroup` clock domains, and reimplement the
four Teams + `input_rec`'s SeekIndex/timeline natively (§6). Cut the target
pipeline over from the old 4-phase seek to the event model; delete `IFlushAndSeek`
and the flush flags once no shimmed node depends on them. (Brief window where old
and new playback coexist.)

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
| Replacing Teams/4-phase seek without live-stream regressions | Event/segment model preserves the existing math (`scalePTS`→`Segment`, `offset_`→`SyncGroup`); parity-first phasing keeps a diffable reference (§3, §6, §10) |
| `.raw()` refcount corruption through the shim | One-ref-per-delivery ownership contract mirroring avcpp move/copy semantics (§8.2) |
| Clock jitter on the mixer-tick path | Bespoke single-threaded executor with external-tick-driven timer, not tokio's timer wheel (§5.2) |
| Stateful node reset (decoder DPB, rescaler, FRUC) on seek | `FlushStart` is handled *inside* each stateful node, where the state lives (§3.3) |
| **Node/group restart during live op misses sticky `Caps`/`Segment`** (regression vs. today's per-frame state) | **Mandatory** sticky-event replay on the edge + restart-as-discontinuity (producer self-flushes) + core-owned epoch/`SyncGroup` + interface pull as second recovery path (§3.6) |
| avcpp depth (`.raw()` ×393) | Nodes already operate at raw-struct level; pass `AVFrame*`/`AVPacket*` and keep avcpp C++-side (§2) |
| `DECLNODE_ATD` template dispatch | Runtime media tag + one factory registration per concrete type (§8.1) |
| CUDA/GL/TensorRT unportable to Rust | They stay C++ behind the shim, permanently (§10 Phase 6) |
| pyplumber binding churn | Retarget pybind11 at the C ABI first; PyO3 optional later (§9) |
| Per-frame FFI overhead | ~100 calls/s/node; measured early; batch only if ever >1% (§1) |

---

## 12. Open questions (carried + new)

1. **Caps negotiation direction.** `IPreferredFormatReceiver` lets a downstream node
   push a preferred pixfmt/resolution *upstream* (see `rescale_video` ctor calling
   `findNodeUp<IPreferredFormatReceiver>()`). Does this stay a query (upstream
   `query_interface`) or become an upstream **event** (a `Reconfigure` traveling
   against the data flow)? GStreamer uses upstream events (`RECONFIGURE`). Leaning
   event, for symmetry with §3.
2. **DirectEdge + events.** When a chain is fused, do events still materialize as
   `EdgeItem`s or become direct method calls (`on_event`)? Probably direct calls,
   with the same handler the buffered path uses.
3. **MetadataFrame across FFI.** `MetadataFrame` is a custom C++ type; the
   `AvpMediaVtable` retain/release handles ownership, but nodes that *read* its
   fields need either a C accessor surface or to stay C++-only. Enumerate which
   nodes touch metadata payload vs. pass it through.
4. **OBS embed (`EMBED_IN=obs`).** Where does the embed boundary sit relative to the
   C ABI? Likely the OBS plugin links the Rust core as a static lib and registers
   OBS-specific nodes via the same factory ABI. Confirm no reliance on the C++ core
   object model.
5. **Reverse playback.** `Segment.rate < 0` is defined, but reverse decode requires
   GOP-granular buffering the current code handles specially
   (`discardUntil(ts=0)`); confirm the InputReader owns this, not individual nodes.
