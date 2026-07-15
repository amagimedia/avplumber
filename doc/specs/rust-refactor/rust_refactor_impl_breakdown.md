# Rust Refactor — Implementation Breakdown (language split & node plan)

> Companion to `rust_refactor_plan_v2.md` (the design). This file is the actionable
> build sheet: what is written in Rust, what stays C++, how many nodes fall in each
> bucket, and the exact recipe for adapting an existing C++ node.
>
> Inventory basis: `src/nodes/` = **38 top-level `.cpp`** + subdirs
> (`neural_net` 37, `python` 9, `hwaccel` 6, `debug` 2, `obs` 2, `jack` 1,
> `_unfinished` 3-excluded) ≈ **91 node types**.

---

## 0. The one-paragraph answer

Rewrite the **core** in Rust (scheduler, edges, causal-dataflow control + master
clock + shared timeline, group supervisor, control protocol, the 4 Teams, and the
seek/timeline logic currently buried in `input_rec`). Reimplement **~5 "nodes"/singletons
that are really core concerns** (`realtime`, `speed`, `pause`, `input_rec`'s
seek/timeline half, plus the `SharedTimeline` store) in Rust as core services. Write
**~7 native Rust nodes** — the trivial + pure-plumbing set — as the proof/dogfood set.
Leave the **other ~80 node types in C++**, recompiled against a new `node_common.hpp`
shim, **unchanged at the source level in the common case**, forever. CUDA/TensorRT/GL/JACK/OBS/Python
nodes are never ported.

So: **Rust = core + ~5 services + ~7 nodes. C++ = ~80 nodes, permanently.**

---

## 1. Language split at a glance

| Layer | Language | What | Size |
|---|---|---|---|
| Scheduler, edges (`Buffer\|Event`, flush-preempts-queue), executors, group supervisor, control protocol, factory registry, `query_interface`, `SyncGroup` master clock | **Rust (new)** | The core | the bulk of new code |
| Teams: `RealTimeTeam`, `SpeedControlTeam`, `PauseControlTeam`, `InputSeekTeam` | **Rust (rewrite as core services)** | were `InstanceShared` C++ singletons | 4 headers → core modules |
| `realtime`, `speed`, `pause` + `input_rec`'s seek-table/timeline half | **Rust (fold into core)** | scheduler-adjacent "nodes" | ~2.1k C++ LOC absorbed |
| Trivial + pure-plumbing nodes (proof set) | **Rust (new nodes)** | firewall, null_sink, split, … | ~7 nodes |
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
| `input_rec.cpp` (seek/timeline half) | ~600 of 1202 | `SeekIndex` + input-side PTS stamping policy (§6.3 of design) |
| `RealTimeTeam.hpp` / `SpeedControlTeam.hpp` / `PauseControlTeam.hpp` / `InputSeekTeam.hpp` | — | core Team services |
| `SharedTimeline.hpp` | ~190 | fifth core service: PTS-keyed k/v store; control writes, `TimelineReader` C++ nodes read (ABI `avp_timeline_*`) |

`input_rec.cpp`'s **demux/read half stays C++** (Tier S) — it's genuine IO on avcpp
`FormatContext`. Only its seek-table + `ETimestampSource`/`ts_offsets_` +
`setFrameMetadataTimestamps` logic moves to Rust.

### Tier R — native RUST nodes (the proof / dogfood set, ~7)

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

`picture_buffer_sink.cpp` (43 LOC) was a candidate here but stays **Tier S**: it
shares an `InstanceShared<PictureBuffer>` slot with `sentinel` (Tier S, C++), which
reads the frame back via `getFrame()`. Keeping it C++ avoids a Rust-writes /
C++-reads `AVFrame*` crossing through a shared singleton for a trivial node — not
worth an ABI. (The alternative — an `avp_picture_buffer_{set,get}` surface — is
recorded in §5 as rejected.)

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
  `scte35_parse` (67), `store_metadata` (305), `join_metadata` (155),
  `assume_metadata` (82), `cc_data_extractor` (60), `write_audio_envelope` (287),
  `sentinel` (1129), `ipc_audio_source` (123), `ipc_socket_audio_source` (240).

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
  across `avp_edge_peek/pop/push` (the §2/§8 raw-pointer contract).
- `DECLNODE*` macros — expand to `avp_register_node_factory(...)` + an auto-generated
  `query_interface` that maps the surviving (Register-3, design §4.4) interface-ID
  table to `dynamic_cast<IFoo*>`.
- The live-query interfaces (`IDecoder`, `IEncoder`, `IMuxer`, `IStreamsInput`, …) —
  header-identical; a node still writes `class Foo : public NodeSISO<…>, public
  IDecoder`. The **Fact** interfaces (`IVideoFormatSource`, `IAudioMetadataSource`,
  `IFrameRateSource`, `ITimeBaseSource`) are served via the `on_spec` hook + latched
  `AvpSpec` on the edge, not the interface table (design §4.4.1) — a node that
  reimplemented them keeps its getters; the shim reads them from `on_spec`.

### 3.2 The refcount rule (the one thing that must be exactly right)

Rust hands the node **one reference** per delivered buffer. The shim's avcpp wrapper
owns it and frees on destruction, **unless** the node forwards it via `put()` (which
moves the ref into the outgoing `AvpBuffer`). This mirrors avcpp's existing
move/copy semantics, so `av::VideoFrame in = source_->get(); … sink_->put(out);`
stays correct, and `.raw()` returns the exact `AVFrame*` Rust passed.

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
Optional native Rust (Tier R):
```rust
#[avp_node("firewall")]
struct Firewall;
impl Transform for Firewall {
  fn transform(&mut self, b: Buffer) -> Flow {
    if b.pts().is_valid() { Flow::push(b) } else { Flow::drop() }
  }
}
```

### 3.4 Worked example — `rescale_video` (Tier S, stays C++ forever)

Body is unchanged: `av::VideoRescaler`, `av_dict_copy`, the `nb_side_data` walk. It
is a Register-1 **Transform** (design §4.4/§4.4.1): today it *implements*
`IVideoFormatSource` with its output size and *reads* the input format. Under the
shim, the `on_spec` bridge does both automatically — the base feeds the incoming
`AvpSpec` to the node and forwards the node's output-format `Spec` downstream — so
its existing `width()`/`height()` getters are reused with **no source edit**. The
one wrinkle is `findNodeUp<IPreferredFormatReceiver>()` (a *reverse*, downstream→
upstream hint, not a Fact): that is the §12.1 open question — resolved as a direct
`query_interface` on the immediate downstream node, a one-line shim change, not a
walk. The shim delivers an `AVFrame*` wrapped as `av::VideoFrame`. This remains the
proof that avcpp-heavy nodes need no rewrite.

### 3.5 Nodes that DO need source edits

Only nodes that reach into the machinery being replaced:

- **Team users** — `realtime`/`speed`/`pause`/`input_rec` call
  `InstanceSharedObjects<RealTimeTeam>::get(...)`. These are Tier C (rewritten in
  Rust), so no surviving C++ node touches a Team.
- **`IFlushAndSeek` overriders** — the old 4-phase base methods are gone from the
  shim base; nodes now receive a `FlushStart` control token instead (timeline is on
  the master clock, not delivered as an event). Grep shows the
  only real overriders are the Tier-C nodes + the `NodeSingleInput` base itself.
  A Tier-S node that had custom flush logic implements `on_event(FlushStart)` — a few
  lines, only where it actually kept internal state (decoders, filters).
- **`MetadataFrame` field readers** — fine C++↔C++ (opaque pointer moved by Rust). A
  problem only if a *Rust* node needs to read metadata fields (none in Tier R), which
  would need C accessors — deferred (open question §12.3 in the design).

---

## 4. What to build, in order (milestones × language)

Maps onto `rust_refactor_plan_v2.md` §10. Each milestone names the language and the
node count it unlocks.

| M | Deliverable | Lang | Nodes runnable after |
|---|---|---|---|
| **M0** | `avplumber_core.h` + `EdgeItem`/`AvpSpec`/master-clock structs; 2-node smoke graph (demux→null_sink) over the ABI | Rust + C header | 0 |
| **M1** | Core MVP: graph, `BufferedEdge` (with latched `Spec`, §4.4.1), blocking-thread sched, one executor, control protocol, factory registry, group supervisor, direct `query_interface` (no upstream walk) | Rust | — |
| **M2** | `node_common.hpp` compat shim + `DECLNODE*` re-impl + avcpp marshalling + refcount contract | C++ | **all Tier-S recompile**; run `input_rec→decode→realtime→encode→mux` to **parity** (old 4-phase seek still via shim) | 
| **M3** | Causal-control playback: `FlushStart/Stop` (flush preempts queues) + `Spec` latched on edges (§4.4.1), `SyncGroup` master clock (rate/offset/pause at the output) + `AvpTimeline` shared store; rewrite the 4 Teams + `SharedTimeline` + `input_rec` seek/timeline in Rust; cut pipeline over; delete `IFlushAndSeek` + `speedChanged` | Rust | Tier C live |
| **M4** | Native node API: `Transform`/`FlowNode` traits, `#[avp_node]` proc macro, `linkme` registry; port Tier-R proof set | Rust | +7 Rust nodes |
| **M5** | Async executor + `DirectEdge` fusion; benchmark mixer-tick latency | Rust | — |
| **M6** | pyplumber: retarget pybind11 at the C ABI (or PyO3) | C++/Rust | python nodes |
| **M7** | Strangler: opportunistic Tier-S → Rust only where it pays; retire C++ core mgmt | mixed | — |

**Parity gate is M2**, on purpose: it proves the ABI + shim + refcounts on the
*hardest real pipeline* before a single node is rewritten in Rust, while you can
still diff against the C++ core.

---

## 5. Effort tally (rough)

- **New Rust core (M1)** — the real work: scheduler, edges, executors,
  supervisor, protocol, registry, interface bridge.
- **Rust services (M3)** — ~2.1k C++ LOC of Team + realtime/speed/input_rec logic
  plus the ~190-LOC `SharedTimeline` store, *re-expressed* (not line-ported) onto the
  master clock + flush discontinuity. Conceptually the hardest, because it's the
  seek/sync correctness surface.
- **C++ shim (M2)** — one header + marshalling; bounded, high-leverage (unlocks ~80
  nodes at once).
- **Rust nodes (M4)** — ~7 small nodes, mostly to validate the API and macro.
- **Never touched** — ~55 CUDA/TensorRT/GL/JACK/OBS/Python nodes; they only need to
  recompile against the shim.

**Bottom line on "how many Rust nodes":** ~5 dissolve into the core (4 Teams'-worth +
`SharedTimeline`), ~7 become native Rust, ~80 stay C++ behind the shim (of which ~55
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
