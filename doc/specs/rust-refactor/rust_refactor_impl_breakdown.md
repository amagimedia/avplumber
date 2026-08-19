# Rust Refactor — Implementation Breakdown (language split & node plan)

> Companion to `rust_refactor_plan_v2.md` (the design). This file is the actionable
> build sheet: what is written in Rust, what stays C++, how many nodes fall in each
> bucket, and the exact recipe for adapting an existing C++ node.
>
> Inventory basis: `src/nodes/` = **38 top-level `.cpp`** + subdirs
> (`neural_net` 37, `python` 9, `hwaccel` 6, `debug` 2, `obs` 2, `jack` 1,
> `_unfinished` 3-excluded) ≈ **91 node types**.
>
> The core's internal data model is owned Rust values, with the C ABI as a compat
> layer over it (`rust_refactor_native_core.md`); this file says what the Tier-R nodes
> are written against (§2), what the shim owes them (§3), the milestone content (§4),
> and where the effort sits (§5).
>
> **Section-reference convention** (four documents, so bare `§` is ambiguous): a plain
> `§N` means *this file*. `plan-v2 §N` means `rust_refactor_plan_v2.md`.
> `native-core §N` means `rust_refactor_native_core.md`.

---

## 0. The one-paragraph answer

Rewrite the **core** in Rust (scheduler, edges, causal-dataflow control + master
clock + shared timeline, group supervisor, control protocol, the 4 Teams, and the
seek/timeline logic currently buried in `input_rec`). Reimplement **~5 "nodes"/singletons
that are really core concerns** (`realtime`, `speed`, `pause`, `input_rec`'s
seek/timeline half, plus the `SharedTimeline` store) in Rust as core services. Write
**~8 native Rust nodes** — the trivial + pure-plumbing set — as the proof/dogfood set.
Leave the **other ~80 node types in C++**, recompiled against a new `node_common.hpp`
shim, **unchanged at the source level in the common case**, forever. CUDA/TensorRT/GL/JACK/OBS/Python
nodes are never ported.

So: **Rust = core + ~5 services + ~8 nodes. C++ = ~80 nodes, permanently.**

---

## 1. Language split at a glance

| Layer | Language | What | Size |
|---|---|---|---|
| Scheduler, edges (`Buffer\|Event`, flush-preempts-queue), executors, group supervisor, control protocol, factory registry, `query_interface`, `SyncGroup` master clock | **Rust (new)** | The core | the bulk of new code |
| Teams: `RealTimeTeam`, `SpeedControlTeam`, `PauseControlTeam`, `InputSeekTeam` | **Rust (rewrite as core services)** | were `InstanceShared` C++ singletons | 4 headers → core modules |
| `realtime`, `speed`, `pause` + `input_rec`'s seek-table/timeline half | **Rust (fold into core)** | scheduler-adjacent "nodes" | ~2.1k C++ LOC absorbed |
| Trivial + pure-plumbing nodes (proof set) | **Rust (new nodes)** | firewall, null_sink, split, … | ~8 nodes |
| `input_rec`'s demux/read half, decoders, encoders, mux, demux, filters, rescale/resample, bsf, sentinel, IPC, metadata nodes | **C++ (shim, recompile)** | avcpp/FFmpeg-heavy | ~40 nodes |
| neural_net (yolo/rtdetr/draw/preprocess/sport_specific/ocr/nvof), hwaccel (EGL/DMA-BUF/CUDA), jack, obs, python | **C++ (shim, never ported)** | CUDA/TensorRT/GL/embed | ~55 nodes |
| avcpp | **C++ (unchanged dependency)** | reconstituted inside the shim | — |

---

## 2. Node inventory, classified into three tiers

### Tier C — becomes CORE Rust (stops being a node)

These are scheduler concerns wearing a node costume. Do **not** shim them; rewrite
their logic into the core.

| File | LOC | Absorbed into |
|---|---|---|
| `realtime.cpp` | 568 | executor clock domain + `SyncGroup` master clock; maps source PTS→wall at the output |
| `speed.cpp` | 278 | `SpeedControlTeam` service; `scalePTS` → master-clock `rate` applied at the output |
| `pause.cpp` | 97 | `PauseControlTeam` service; freeze the master clock + executor group-suspend |
| `input_rec.cpp` (seek/timeline half) | ~600 of 1202 | `SeekIndex` + input-side PTS stamping policy (plan-v2 §6.3) |
| `RealTimeTeam.hpp` / `SpeedControlTeam.hpp` / `PauseControlTeam.hpp` / `InputSeekTeam.hpp` | — | core Team services |
| `SharedTimeline.hpp` | ~190 | fifth core service: PTS-keyed k/v store; control writes, `TimelineReader` C++ nodes read (`AVP_SERVICE_TIMELINE` vtable, plan-v2 §4.8) |

`input_rec.cpp`'s **demux/read half stays C++** (Tier S) — it's genuine IO on avcpp
`FormatContext`. Only its seek-table + `ETimestampSource`/`ts_offsets_` +
`setFrameMetadataTimestamps` logic moves to Rust.

### Tier R — native RUST nodes (the proof / dogfood set, ~8)

Chosen because they are trivial or pure edge-plumbing, exercise the new node API and
the `DirectEdge`/routing paths, and touch no avcpp internals beyond `.pts()`.

| File | LOC | Why Rust |
|---|---|---|
| `firewall.cpp` | 18 | trivial filter; canonical `Transform` example |
| `null_sink.cpp` | 24 | trivial sink |
| `split.cpp` | 60 | pure fan-out plumbing (exercises multi-output) |
| `one_to_many.cpp` | 70 | pure routing (exercises `DirectEdge` fusion) |
| `packet_relay.cpp` | 91 | pass-through / relay |
| `force_keyframe.cpp` | 158 | packet-flag inspection (light `.raw()` read) |
| `debug/*` (2) | small | corruption/test nodes; safe to dogfood |
| `join_metadata.cpp` | 155 | multi-input / PTS-sync exemplar — see below |

**Why `join_metadata` is in Tier R.** It is the one node that proves the parts of the
contract the rest of Tier R never touches: two inputs consumed at different rates,
PTS-matched merge, and two-sided EOF. Verified against
`src/nodes/join_metadata.cpp`:

- Declared `DECLNODE_ATD_RAW(join_metadata, JoinMetadata)` (`:155`), so it only ever
  instantiates for `VideoFrame` + `AudioSamples`
  (`EDGE_RAW_DATA_TYPES`, `src/edge_types.hpp`) — no `Packet`, no `MetadataFrame`,
  no `EglImageFrame`. That halves the port surface.
- Its whole body is timestamp comparison plus `av_dict_copy` of
  `AVFrame.metadata` (`:109`) and a side-data walk over
  `av_frame_get_side_data` / `av_buffer_ref` / `av_frame_new_side_data_from_buf`
  (`:111-125`). No avfilter, no codec context, no hardware.
- Its `isUsable()` (`(!data.isNull()) && data.isComplete() && data.pts().isValid()`)
  is exactly the three-part predicate the native model is supposed to make
  unnecessary — `Media` is non-null by construction, EOF is an `EdgeEvent` rather
  than an incomplete frame (native-core §3.4), and `Ts` carries its own validity. So
  the port is also a readable demonstration that the model removes a class of guard.

Three caveats, none blocking:

1. **Two small rsmpeg gaps**, listed in native-core §2.5 alongside the others:
   `AVFrame.metadata` has no accessor and `av_frame_new_side_data_from_buf` is
   unwrapped in rsmpeg 0.18.0 *(verified in `src/avutil/frame.rs`; `AVDictionary`
   including `copy` and `AVBufferRef` **are** wrapped, and `get_side_data` exists at
   `frame.rs:287-293`)*. Two short `unsafe` helpers in our own crate, both generally
   useful beyond this node. Upstreaming them is optional and off the critical path.
2. **It is not trivial plumbing like the rest of Tier R** — it has real sync
   semantics and a two-sided end-of-stream. Sequence it **last in M4**, after the
   trivial nodes have shaken out the API.
3. **It does not remove the multi-input peek work.** `mux`, `filters`, and
   `source_switcher` still peek from C++, so the §3.2 borrow guard is required
   regardless. Porting `join_metadata` removes 4 of 8 Tier-S peek sites, not the
   feature.

`picture_buffer_sink.cpp` (43 LOC) was a candidate here but stays **Tier S**: it
shares an `InstanceShared<PictureBuffer>` slot with `sentinel` (Tier S, C++), which
reads the frame back via `getFrame()`. Keeping it C++ avoids a Rust-writes /
C++-reads `AVFrame*` crossing through a shared singleton for a trivial node — not
worth an ABI. (The alternative — an `avp_picture_buffer_{set,get}` surface — is
recorded in §5 as rejected.)

**What Tier-R nodes are written against.** Owned `Media` values (`rsmpeg`
`AVFrame`/`AVPacket`), not `AvpBuffer` raw pointers. Four of them are materially safer
for it:

- `split` / `one_to_many` fan out one input to N outputs — the classic
  retain-N-times-or-double-free site. With `Media` the fan-out is
  `for out in outs { out.push(m.clone())? }` (`av_frame_ref` under the hood).
  Under-cloning is a move error the compiler rejects; over-cloning is not a leak
  because the extra value drops. Neither failure mode survives to runtime.
  `AVPacket` has **no `Clone` in rsmpeg 0.18** *(verified: `src/avcodec/packet.rs`
  is 61 lines and implements only `new`/`rescale_ts`/`Debug`/`Default`/`Drop`)*,
  so packet fan-out needs an explicit `av_packet_ref`-based helper — a named M4
  task, not a surprise.
- `firewall` reads the timestamp, so it is the smallest node that exercises `Ts`
  and `FrameExt::ts()` (native-core §2.3–2.4) end to end.
- `force_keyframe` inspects `AVPacket.flags`, which rsmpeg exposes natively —
  no `.raw()` escape hatch needed.

Everything else that *could* be ported later is deferred — there is no reason to
rewrite a working avcpp node.

### Tier S — stays C++, recompiled against the shim (~80)

**Never ported (hard-C++ deps):**
- `neural_net/**` (37): yolo, rtdetr, draw ×7, preprocess, sport_specific ×23,
  ocr, nvof, utils ×2 — CUDA/TensorRT.
- `hwaccel/**` (6): EGL/DMA-BUF/CUDA interop.
- `jack/*` (1), `obs/*` (2), `python/*` (9), `crop_metadata_cuda.cpp` (530).

**Recompile-and-keep (avcpp/FFmpeg-heavy, no reason to port):**
- Container/codec: `demux` (195), `input` (264), `output` (160), `raw_output`
  (175), `mux` (360), `bsf` (167), `encoders` (263), `decoders` (469),
  `input_rec` demux half.
- DSP/format: `filters` (877, avfilter), `rescale_video` (146), `resample_audio`
  (398), `reinterpret_planes` (462), `force_fps` (229), `smooth_timestamps` (292),
  `extract_timestamps` (246).
- Metadata/routing/other: `source_switcher` (216), `preheat_video_router` (505),
  `scte35_parse` (67), `store_metadata` (305),
  `assume_metadata` (82), `cc_data_extractor` (60), `write_audio_envelope` (287),
  `sentinel` (1129), `ipc_audio_source` (123), `ipc_socket_audio_source` (240).
  (`join_metadata`, 155, is Tier R — see above.)

---

## 3. How to adapt an existing C++ node

**Design goal: a Tier-S node recompiles with zero source edits in the common case.**
The trick is that `node_common.hpp` *becomes* the compat shim, so nodes keep
including the same header, keep using `NodeSISO`/`Source`/`Sink`, and keep their
`DECLNODE` line.

### 3.1 What the shim re-implements (backed by the C ABI, not the old core)

`avplumber_node_compat.hpp` (included by `node_common.hpp`) provides, with the same
names/signatures as today:

- `Node`, `NodeSingleInput/Output`, `NodeMultiInput/Output`, `NodeSISO<In,Out>`,
  `NonBlockingNode<>` — but their bodies call `avp_edge_*` and hold an `AvpNode*`.
- `Source<T>`/`Sink<T>` + `createCommon` — marshal avcpp ↔ `AVFrame*`/`AVPacket*`
  across `avp_edge_take/peek/pop/push` (the plan-v2 §§2, 8 raw-pointer contract).
  `Source<T>::peek()` returns a borrow guard rather than a `T*` (§3.2,
  native-core §4.3.1).
- `DECLNODE*` macros — expand to `avp_register_node_factory(...)` + an auto-generated
  `query_interface` that maps the surviving (Register-3, plan-v2 §4.4) interface-ID
  table to `dynamic_cast<IFoo*>`.
- The live-query interfaces (`IDecoder`, `ISentinel`, `IStreamsInput`,
  `IReturnsObjects`/`IInputsObjects`) — header-identical; a node still writes
  `class Foo : public NodeSISO<…>, public IDecoder`. The **Fact** interfaces
  (`IVideoFormatSource`, `IAudioMetadataSource`, `IFrameRateSource`,
  `ITimeBaseSource`) are served via the `on_spec` hook + latched `AvpSpec` on the
  edge, not the interface table (plan-v2 §4.4.1) — a node that reimplemented them
  keeps its getters; the shim reads them from `on_spec`. `IEncoder`/`IMuxer` are
  **not** in the table either (plan-v2 §4.4.2): their handshake becomes `Spec`
  aggregation, which is the one place a shimmed node's *source* does change — **23
  `initFromFormatContext*`/`setOutput*`/`openEncoder` sites across 4 files**
  (`mux.cpp`, `encoders.cpp`, `packet_relay.cpp`, `bsf.cpp`) are rewritten as
  `on_spec`. `IJackSink` (plan-v2 §4.4) likewise switches to direct registration on
  the JACK service. These are the only shimmed nodes not recompiling untouched, and
  they are named in the M2 audits (§3.5).

**Three fields the shim must bridge explicitly.** What crosses is the plain
`AVFrame`/`AVPacket`, but avcpp's `Frame`/`Packet` carry state *outside* the struct —
so "hand over the pointer" is not by itself a faithful conversion. Verified against
`deps/avcpp/src/frame.h` and `deps/avcpp/src/packet.cpp`:

| avcpp member | Where it lives | Shim obligation |
|---|---|---|
| `m_timeBase` | C++ member; avcpp **never writes the native `AVFrame.time_base` field** | on delivery, `setTimeBase()` from the buffer's `Ts.tb`; on `put()`, copy the wrapper's timebase back into the native field |
| `m_isComplete` | C++ member; gates `operator bool()` (`frame.h:278`) | set from `Media` presence — a buffer that reached the edge *is* complete; on `put()`, drop an incomplete frame rather than pushing it |
| `m_streamIndex` (frames) | C++ member, **dead** | nothing. Its only two reads are inside comments (`sentinel.cpp:611`, `:707`). Packet `stream_index` is native (`packet.cpp:267`) and needs no bridge |

The timebase bridge has one asymmetry that must be preserved: avcpp's
`Frame::setTimeBase` **rescales** `pts` and `best_effort_timestamp`
(`frame.h:228-254`) and `Packet::setTimeBase` calls `av_packet_rescale_ts`
(`packet.cpp:334-345`, so it also rescales `duration`). The shim therefore sets
the timebase **before** the node sees the timestamp, and must not call
`setTimeBase` again on the way out unless a rescale is actually intended.

### 3.2 The refcount rule (the one thing that must be exactly right)

Rust hands the node **one reference** per delivered buffer. The shim's avcpp wrapper
owns it and frees on destruction, **unless** the node forwards it via `put()` (which
moves the ref into the outgoing `AvpBuffer`). This mirrors avcpp's existing
move/copy semantics, so `av::VideoFrame in = source_->get(); … sink_->put(out);`
stays correct, and `.raw()` returns the exact `AVFrame*` Rust passed.

**`Source<T>::get()` uses `avp_edge_take`, not `peek`+`pop`.** `get()`
returns an *owned* avcpp object, so building one from a peeked (borrowed) pointer
would cost an `av_frame_ref` per frame — and would make "who owns Rust's ref" a
documented contract rather than a checked one. `avp_edge_take` removes the head
and **moves** the core's reference to the caller (native-core §4.3): the common path
performs zero refcount operations, and the ownership transfer is stated in the
signature.

Inside the core the rule is not a rule at all: the reference is owned by an
`AVFrame`/`AVPacket` value and released by `Drop`. The three hand-rolled
`release()` sites in the skeleton's `graph/buffered_edge.rs` (FlushStart drain, `pop`,
`Drop`) disappear — see §4 M0.

**`peek()` cannot return a `T*` — that shape is already unsound in the C++ core.**
`Edge<T>::peek` is declared `virtual T* peek(int timeout_ms)`
(`src/graph_core.hpp:161`, forwarding to `moodycamel::ReaderWriterQueue::peek`) and
callers dereference it as a full avcpp object — `frmin->isComplete()`,
`frmin->timeBase()`, `pkt->pts()`. Those are C++ members (§3.1), so the returned
pointer must address an object whose `m_timeBase` / `m_isComplete` are already
correct, and the core's queue holds `Media`, which has no such object. Worse,
`ReaderWriterQueue::pop()` destroys the element in place
(`deps/readerwriterqueue/readerwriterqueue.h:442`, `element->~T();`), so **any peeked
pointer held across a `pop()` on the same edge dangles** — and one node in the tree
does exactly that (see the pre-M2 fix below).

**So `peek()` returns an RAII borrow guard**: a move-only `Peeked` object that owns
the avcpp wrapper it hands out, with `.get()` / `.mut()` / `.consume()` and "drop =
leave it queued". Full spec, including the C ABI mirror (`avp_edge_peek` /
`_release` / `_consume` over an opaque `AvpPeek`) and the call-site impact table, is
**native-core §4.3.1**. For this build sheet the consequences are:

- **Little shim machinery.** No slot table, no invalidation protocol, no "which port
  owns which wrapper" bookkeeping. The guard is a value; a multi-input node that peeks
  two edges simultaneously (`join_metadata.cpp:51-52`) just holds two guards.
- **Peek is not free, unlike `take`:** constructing the wrapper from an `AVFrame*`
  costs one `av_frame_ref` (`deps/avcpp/src/frame.h:129-135`). Since the guard already
  holds its own ref, its `.consume()` calls `avp_edge_peek_consume(p, NULL)` — the
  form in which the core *releases* its ref rather than moving it out
  (native-core §4.3.1) — and the guard's already-built object becomes the return
  value. Net: **one extra `av_frame_ref`** versus `take` alone. `get()` on a
  never-peeked port still uses `take` and pays nothing. The penalty lands where a node
  peeks and *declines* repeatedly — the backpressure/waiting-for-other-input case
  (`mux`, `source_switcher`) — worth measuring in M2 rather than assuming.
- **Peek cannot be dropped from the ABI.** 11 of the surviving sites are in
  never-ported nodes (`cuda_rect_overlay` 8, `cuda_to_egl_image`, `nvof_fruc`,
  `obs_video_sink`) — exactly the nodes that will never be rewritten to avoid it.

Census (`grep -rn -- "->peek(" src/`, excluding `_unfinished`): **49 sites in 24
files**, of which **33 in 15 files survive into the shim**. By tier:

- **Tier S — the shim's permanent burden, 18 sites in 10 files:** `filters` (4),
  `source_switcher` (4), `mux` (3), `bsf`, `preheat_video_router`, `raw_output`,
  `extract_timestamps`, `sentinel`, `smooth_timestamps`, `force_fps`. Three of them
  (`filters`, `source_switcher`, `mux`) peek two edges at once, which is why the guard
  has to be a value a node can hold twice.
- **Never ported, 11 sites:** `cuda_rect_overlay` (8), `cuda_to_egl_image`,
  `nvof_fruc`, `obs_video_sink`.
- **`graph_base.hpp`, 4 sites** — the base class's own `consumeEofIfPresent` and
  least-PTS helpers, inherited by every multi-input node. These convert once and cover
  many nodes.
- **Tier C, 6 sites** — `realtime` (2), `speed` (2), `pause` (2); absorbed into core
  services, so their peeks disappear.
- **Tier R, 10 sites** — `join_metadata` (4), `split` (2), `one_to_many`,
  `force_keyframe`, `debug/*` (2); ported to Rust, so they peek the native edge and
  never touch the shim.

**Pre-M2 fix required in the existing C++ tree.**
`src/nodes/hwaccel/cuda_rect_overlay.cpp` stores a peeked pointer
(`meta_src = p;`, `:766`), pops those edges (`:781-790`), then dereferences it
(`:828` → `mergeLayersForTick` `:499` → `clearCanvas` `:501` /
`av_frame_copy_props` `:524-525`). This is a use-after-free today. It has gone
unnoticed because `~Frame()` nulls `m_raw`, so the `metadata_src->raw()` guard
usually short-circuits and metadata is *silently dropped*; a producer refill in the
freed slot turns it into reading a different frame's props. **Convert `T* peek()` to
the guard in the C++ tree and fix this site before the shim is written** — 24 files of
mostly mechanical edits (25 of 49 sites compile unchanged on `operator bool`, 21 need
one-token `nullptr`-comparison edits, 1 mutating site in `bsf.cpp:118-121` needs
`.mut()`), verifiable against the current core with the current tests, and it means
the shim is specified against a sound API rather than one we intend to delete.

### 3.3 Worked example — `firewall` (unchanged C++, then optional Rust rewrite)

C++ today (works as-is under the shim, no edits):
```cpp
#include "node_common.hpp"                 // now pulls the compat shim
template<typename T> class Firewall : public NodeSISO<T,T> {
  void process() override {
    T d = this->source_->get();
    if (!d.pts().isValid()) return;
    this->sink_->put(d);
  }
  static std::shared_ptr<Firewall> create(NodeCreationInfo& nci){
    return NodeSISO<T,T>::template createCommon<Firewall>(nci.edges, nci.params);
  }
};
DECLNODE_ATD(firewall, Firewall)          // shim registers one factory per media tag
```
Optional native Rust (Tier R) — owned `Media` in, `Ts` for the timestamp, and a
transform returning `Result<Option<Media>, NodeError>` (`None` = drop); the *edge* op
returns `Push` while the *node invocation* returns `Tick`/`Blocked`
(native-core §4.2):
```rust
#[avp_node("firewall")]
struct Firewall;
impl Transform for Firewall {
  fn transform(&mut self, m: Media) -> Result<Option<Media>, NodeError> {
    Ok(if m.ts().is_valid() { Some(m) } else { None })   // None = drop, no leak
  }
}
```
Note what is *not* written: no retain, no release, no "did I forward it" audit.
Dropping the frame is `None`; the `Media` destructor frees it.

### 3.4 Worked example — `rescale_video` (Tier S, stays C++ forever)

Body is unchanged: `av::VideoRescaler`, `av_dict_copy`, the `nb_side_data` walk. It
is a Register-1 **Transform** (plan-v2 §§4.4, 4.4.1): today it *implements*
`IVideoFormatSource` with its output size and *reads* the input format. Under the
shim, the `on_spec` bridge does both automatically — the base feeds the incoming
`AvpSpec` to the node and forwards the node's output-format `Spec` downstream — so
its existing `width()`/`height()` getters are reused with **no source edit**. The
one wrinkle is `findNodeUp<IPreferredFormatReceiver>()` (a *reverse*, downstream→
upstream hint, not a Fact): that is plan-v2 §12 item 1 — resolved as a direct
`query_interface` on the immediate downstream node, a one-line shim change, not a
walk. The shim delivers an `AVFrame*` wrapped as `av::VideoFrame`; that pointer comes
out of a core-owned `Media::Video(AVFrame)` via `into_raw()`, which is a pointer move
with no copy and no refcount operation *(verified: rsmpeg's `wrap_pure!` `into_raw` is
`mem::forget` + pointer return, `src/macros.rs:44`)*, plus the §3.1
timebase/`isComplete` bridge. This node is the proof that avcpp-heavy nodes need no
rewrite.

### 3.5 Nodes that DO need source edits

Only nodes that reach into the machinery being replaced. "All Tier-S recompiles
unchanged" is the common case, not a universal claim — this section is the exception
list, and the two audits at the end of it are what make the list trustworthy.

- **Team users** — `realtime`/`speed`/`pause`/`input_rec` call
  `InstanceSharedObjects<RealTimeTeam>::get(...)`. These are Tier C (rewritten in
  Rust), so no surviving C++ node touches a Team.
- **`IFlushAndSeek` overriders** — the old 4-phase base methods are gone from the
  shim base; nodes now receive a `FlushStart` control token instead (timeline is on
  the master clock, not delivered as an event). Grep shows the
  only real overriders are the Tier-C nodes + the `NodeSingleInput` base itself.
  A Tier-S node that had custom flush logic implements `on_event(FlushStart)` — a few
  lines, only where it actually kept internal state (decoders, filters).
  - **M3 pre-cutover audit (do before deleting `IFlushAndSeek`):** the claim "only
    Tier-C nodes + the base override it" is the load-bearing correctness assumption
    of the new seek (plan-v2 §3.4 step 1 — each node flushes its *own* state on
    `FlushStart`). The risk is a Tier-S node that keeps seek-relevant internal state
    (decoder DPB, avfilter graph, FRUC/smoothing/reorder buffers) but relied on the
    base's edge-clear + `pauseProcessing` barrier to reset it, **without** overriding
    `IFlushAndSeek`. Such a node would carry stale state across the discontinuity.
    Audit every Tier-S node under `src/nodes/**` for internal buffers that survive an
    edge clear (grep for members holding `av::*`/`AVFrame*`/`std::deque`/reorder
    state across `process()` calls, and for `discardUntil`/`resetInput`/DPB-flush
    calls); each hit that isn't already an `IFlushAndSeek` overrider needs an explicit
    `on_event(FlushStart)` in its shim. Output: a list of the (expected few) nodes
    that need the hook, verified before `IFlushAndSeek` is removed in M3.
- **`IEncoder`/`IMuxer` implementers** — the largest source edit in the shim,
  because the interface itself is deleted rather than re-hosted (plan-v2 §4.4.2).
  **23 `initFromFormatContext*` / `setOutput*` / `openEncoder` sites across 4 files**
  (`mux.cpp`, `encoders.cpp`, `packet_relay.cpp`, `bsf.cpp`) become `on_spec`: the
  muxer aggregates one input `Spec` per stream instead of fanning a query across its
  N inputs; `packet_relay`/`bsf` forward or transform `Spec` instead of relaying a
  query upstream ("not really" `IEncoder`); the encoder self-opens from its input
  `Spec` and emits its output `Spec`, always setting `AV_CODEC_FLAG_GLOBAL_HEADER`
  instead of asking the container. Do this **with** the M2 shim, not before — it is
  the change the parity run is meant to catch.
- **`IJackSink`** — `jack_sink` registers with the JACK service
  (`avp_jack_client_add_sink`) instead of being discovered by query (plan-v2 §4.4).
  One node, one call site (`src/nodes/jack/jack_sink.cpp:17`).
- **`MetadataFrame` field readers** — fine C++↔C++ (opaque pointer moved by Rust as
  `Media::Opaque`, which gives it a lifetime the core can enforce but no field
  access). A problem only if a *Rust* node needs to read metadata fields (none in
  Tier R), which would need C accessors — deferred, and irreducible (plan-v2 §12
  item 3).
- **Nodes whose behaviour depends on a bridged field** — the §3.1 bridge
  is generic, but two cases need a per-node look during M2 because their *semantics*
  live in the bridged member. Counts below are measured, not estimated:
  - **`isComplete` gating — 19 read sites across 26 files that touch
    `isComplete`/`setComplete`** (`grep -rn "isComplete" src/nodes/`). Most are the
    benign shape `if (!x.isComplete()) return;` — an early-out that the shim's
    "a buffer on the edge is complete" invariant simply makes unreachable, which is
    correct and needs no edit. The ones to look at are where incompleteness is
    *meaningful state*, not a guard:
    - `encoders.cpp:146` / `decoders.cpp:289` — these read incompleteness as
      **"this is a flush marker"** (the comment at `encoders.cpp:147` says so). EOF
      markers are default-constructed `T` (`avutils.hpp:155-157`), so
      `m_isComplete == false` is load-bearing signal, not absence of data. The
      adapter must synthesize the marker on `Eof` rather than setting complete
      unconditionally — see native-core §3.4. **A miss here hangs the pipeline at end
      of stream instead of crashing**, so it is the one item in this list that must be
      tested, not just read.
    - `reinterpret_planes.cpp:439` — `out.setComplete(in.isComplete())` propagates
      the flag through, so its output completeness is only as good as the bridge's.
    - `sentinel.cpp:567` — `last_frame_.isComplete()` is *cached across calls*, so
      it reads the flag on a frame the shim no longer owns.
    - `filters.cpp:585,614,662` — combines `isComplete()` with a timebase
      validity check on a **peeked** pointer, so it depends on two bridges plus the
      peek guard at once. Good first integration test for §3.2.
  - **Per-buffer timebase changes — 37 `setTimeBase` sites, 32 of them in
    `src/nodes/`.** Those that change the timebase *mid-stream*
    (`smooth_timestamps`, `extract_timestamps`, `force_fps`, the rescale/resample
    pair) are the reason the buffer carries the authoritative timebase rather than
    relying on `Spec` alone (native-core §3.3). For each, confirm the shim writes the
    changed timebase back into the native field on `put()`, or the change is
    silently lost downstream. Note that avcpp's default `Rational` is `{0,0}`
    (`deps/avcpp/src/rational.cpp:10-13`), and `filters.cpp` explicitly tests for
    a zero num/den — so a missed bridge fails *loudly* there and silently
    elsewhere.

  These are *audit* tasks, not rewrites, but "expected small hit count" would be
  wrong: budget for reading ~26 files and editing a handful. They are the M2
  analogue of the M3 `FlushStart`-state audit above, and like it, they gate the
  parity claim rather than following it.

---

## 4. What to build, in order (milestones × language)

Maps onto `rust_refactor_plan_v2.md` §10. Each milestone names the language and the
node count it unlocks.

| M | Deliverable | Lang | Nodes runnable after |
|---|---|---|---|
| **M-pre** | In the **existing C++ tree**, independent of any Rust: replace `Edge<T>::peek`'s `T*` return with the move-only borrow guard (native-core §4.3.1) and fix the `cuda_rect_overlay` use-after-free it exposes (§3.2). 24 files, mostly one-token edits, one real bug. Verifiable against the current core with the current tests — do it here so M2's shim is specified against a sound API | C++ | unchanged (behaviour-preserving except the fixed bug) |
| **M0** | Two layers, built together: (a) native model — add `rsmpeg 0.18`, define `Media`/`Spec`/`Ts`/`FrameExt`/`PacketExt`, split `Flow` into `Push`/`Tick`/`Blocked`; (b) `avplumber_core.h` ABI structs + master clock. Base `BufferedEdge` on `Media` (so no manual `retain`/`release` is written at all). **Fix the 2 skeleton defects first** (missing `log` dep in `exec/blocking.rs`; `av_frame_ref`/`av_packet_ref` arity in `graph/buffer.rs:117,120` — the latter proves the `ffmpeg` feature has never compiled). 2-node smoke graph (demux→null_sink) over the ABI | Rust + C header | 0 |
| **M1** | Core MVP: graph, `BufferedEdge` (with latched `Spec`, plan-v2 §4.4.1), blocking-thread sched, one executor, control protocol, factory registry, group supervisor, direct `query_interface` (no upstream walk). `trait Node` + `NodeBody` is the internal contract from here on (`Blocking` default, `Poll`, `Async`); registry is `NodeSpec` + serde with type erasure so `node.add` stays dynamic (native-core §5.1–5.2); typed `NodeRef<T>`/pad handles checked at `connect` | Rust | — |
| **M2** | `node_common.hpp` compat shim + `DECLNODE*` re-impl + avcpp marshalling + refcount contract. Concretely: the `FfiNode` adapter (one `Node` impl wrapping an `AvpNode*`, blocking path + exactly **3** `Poll` nodes), `avp_edge_take`, the three §3.1 bridges (timebase, `isComplete`, no-op streamIndex), and the **peek borrow guard** over `AvpPeek` (§3.2, native-core §4.3.1 — assumes M-pre has landed). Includes the two §3.5 audits and the `IEncoder`/`IMuxer`→`on_spec` rewrite (23 sites, 4 files) | C++ | **all Tier-S recompile** except the §3.5 exception list; run `input_rec→decode→realtime→encode→mux` to **parity** (old 4-phase seek still via shim) |
| **M3** | Causal-control playback: `FlushStart/Stop` (flush preempts queues) + `Spec` latched on edges (plan-v2 §4.4.1), `SyncGroup` master clock (rate/offset/pause at the output) + `AvpTimeline` shared store; rewrite the 4 Teams + `SharedTimeline` + `input_rec` seek/timeline in Rust; cut pipeline over; delete `IFlushAndSeek` + `speedChanged` (**gated on the §3.5 FlushStart-state audit**) | Rust | Tier C live |
| **M4** | Tier-R port + ergonomics on top of M1's contract: `Transform` convenience trait, `#[avp_node]` proc macro, the `av_packet_ref` fan-out helper (`AVPacket` has no `Clone`) | Rust | +8 Rust nodes |
| **M5** | Async executor + `DirectEdge` fusion; benchmark mixer-tick latency | Rust | — |
| **M6** | pyplumber: retarget pybind11 at the C ABI (or PyO3) | C++/Rust | python nodes |
| **M7** | Strangler: opportunistic Tier-S → Rust only where it pays; retire C++ core mgmt | mixed | — |

**Parity gate is M2**, on purpose: it proves the ABI + shim + refcounts on the
*hardest real pipeline* before a single node is rewritten in Rust, while you can
still diff against the C++ core. It also proves the *adapter* — that
`Media` → `AVFrame*` → `av::VideoFrame` and back is lossless for every field a C++
node actually reads. That is the whole risk of the native-core shape, and M2 is where
it is measured, before ~80 nodes depend on it.

**Why the native model sits in M0 rather than later.** It does not front-load work:
`Media` *replaces* code M0 would otherwise write (a hand-rolled `retain`/`release`
pair and its call sites), and M1's edge has to store *something* either way. The
sequencing constraint is only that the model precedes the edge that carries it and the
adapter that converts it — M0 → M1 → M2. The native node API is likewise not gated
behind M4: M1's `trait Node` *is* the contract, and M4 adds only the ergonomics on top.

---

## 5. Effort tally (rough)

- **Native data model (M0)** — small and mostly declarative: one dependency, three
  enums/newtypes, two extension traits, one `Flow` split. Its net effect on total
  effort is roughly **zero or negative**: it adds type definitions and removes
  hand-rolled refcounting plus the `AvpSpec` lossiness workarounds. The cost that *is*
  real is the M2 adapter (below).
- **New Rust core (M1)** — the real work: scheduler, edges, executors,
  supervisor, protocol, registry, interface bridge.
- **Rust services (M3)** — ~2.1k C++ LOC of Team + realtime/speed/input_rec logic
  plus the ~190-LOC `SharedTimeline` store, *re-expressed* (not line-ported) onto the
  master clock + flush discontinuity. Conceptually the hardest, because it's the
  seek/sync correctness surface.
- **C++ peek reshape (M-pre)** — 24 files, 49 call sites, one real bug fixed. Small,
  and it does not depend on any Rust work, so it can run in parallel with M0/M1.
  Counted separately because it is spent in the *old* tree and pays off in the new one.
- **C++ shim (M2)** — where the native-core shape concentrates its cost, and
  **larger than "one header"**: the `FfiNode` adapter, the three field bridges, the
  peek guard (§3.2), and the two §3.5 audits (~26 files to read). Bounded in *scope*
  but not small; high-leverage, since it unlocks ~80 nodes at once. It buys the
  decoupling: the ABI can lag the core instead of dictating it.
- **Rust nodes (M4)** — ~8 small nodes, mostly to validate the API and macro. The
  contract lands in M1, so M4 is the port plus a proc macro, not a new API design.
  `join_metadata` is the exception — real sync semantics, so budget it separately and
  sequence it **last** (§2).
- **Never touched** — ~55 CUDA/TensorRT/GL/JACK/OBS/Python nodes; they only need to
  recompile against the shim.

**Bottom line on "how many Rust nodes":** ~5 dissolve into the core (4 Teams'-worth +
`SharedTimeline`), ~8 become native Rust, ~80 stay C++ behind the shim (of which ~55
permanently). The Rust node count is deliberately small — the win is the core +
causal-dataflow model + boilerplate reduction, not rewriting working media nodes.

**Singletons deliberately NOT given an ABI** (verified against `InstanceShared<>`
usage): `EventLoop`/`TickSource` dissolve into the Rust executor (no surviving
singleton); `HWAccelDevice`, `JackClient`, `TimestampExtractorTeam` are touched only
by never-ported C++ nodes (pure C++-internal); `NamedEvent` is touched only by the
control layer (a Rust core primitive, no node crossing); `PTSCorrectorCommon` lives
inside `sentinel` (C++) and is only read for stats via the generic `getObject` query.
`PictureBuffer` **rejected** an `avp_picture_buffer_{set,get}` ABI in favor of
keeping `picture_buffer_sink` in Tier S (see §2, Tier R note). `MixerTransitionScheduler`
+ `MixerState` stay C++ orchestration reachable only from `avplumber.cpp` control;
once `AvpTimeline` exists the mixer can be driven from Rust through it, but that
decision is deferred.
