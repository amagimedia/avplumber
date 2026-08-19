# Native Rust core with an FFI adapter

> **Status:** approved design, 2026-08-19. This document specifies the core's
> internal data model, the node contract, and the native embedder surface.
> `rust_refactor_plan_v2.md` specifies the C ABI, the edge/control model,
> scheduling, services and phasing around it; `rust_refactor_impl_breakdown.md` is
> the build sheet.
>
> Written after reading rsmpeg 0.18.0 sources, `rusty_ffmpeg` 0.16.7 bindings,
> FFmpeg 8 `frame.h`/`packet.h`, avcpp `frame.h`/`packet.{h,cpp}`/`ffmpeg.h`, and
> the current `avplumber_f7k` skeleton. Every claim marked *(verified)* was checked
> against those sources rather than recalled.

---

## 1. The shape in one paragraph

**The C ABI is a compat layer, not the core's internal data model.** The core's
internal model is owned Rust values (rsmpeg `AVFrame`/`AVPacket`), refcounting is
`Drop`, and the raw `*mut c_void` appears only inside one adapter file that C++
nodes are called through. The flat `#[repr(C)]` `AvpBuffer`/`AvpSpec` live in
`abi/` and nowhere else. Native Rust nodes never see a raw pointer.

**The governing constraint, which shapes everything below: one substrate
contract, two adapters — not two parallel node contracts.** There is exactly one
`trait Node`, one edge implementation, one flush semantic, one `Spec`-forwarding
rule. C++ nodes reach them through `FfiNode`; Rust nodes implement them directly.
Two independent node contracts would let flush semantics, `Spec` forwarding and
lifecycle drift apart and be fixed in only one place — that is the failure mode
this design exists to avoid, and it is the reason a "pluggable whole engine"
option was rejected (it would freeze the C++ out-of-band seek and Team machinery
into a permanent interface, which is precisely what the refactor deletes).

### 1.1 Why this is nearly free

`rsmpeg::avutil::AVFrame` is a `NonNull<ffi::AVFrame>` in a newtype *(verified —
the `wrap_pure!` macro in rsmpeg `src/macros.rs`)*. `into_raw()` destructures it
and calls `mem::forget`; `from_raw()` reconstructs it. So native↔C conversion is
a **pointer move with no refcount operation and no copy** — identical runtime
representation on both sides. The native model costs nothing at the boundary, and
the boundary is where the only cost could be.

### 1.2 Why the ABI is not the internal model

Three concrete costs of using the flat ABI structs as the core's own payload:

1. **Hand-rolled refcounting in three places.** The skeleton's `BufferedEdge`
   calls `release()` on the `FlushStart` arm (`buffered_edge.rs:71`), in `pop`
   (`:123`), and in `Drop` (`:169`). Each is a hand-audited site; all three
   collapse into `Drop` plus `queue.clear()`.
2. **A lossy `Spec` by construction.** `AvpSpec` is a flat `#[repr(C)]` struct
   whose own comment records that `CUSTOM`/`AMBISONIC` channel layouts "degrade
   to `UNSPEC` — the `map` is not carried across this FFI" (`edge.rs:23-33`).
   As the internal model that loss would be unconditional. As a compat layer, it
   applies only to C++ nodes, and only to those that read the field.
3. **Coupling.** If the internal model *is* the ABI, every internal refactor
   breaks ~80 C++ nodes plus pyplumber simultaneously. Separated, the ABI stays
   put while the core evolves.

---

## 2. Native data model

### 2.1 `Media`

```rust
pub enum Media {
    Packet(AVPacket),        // rsmpeg owned; Drop = av_packet_free
    Video(AVFrame),
    Audio(AVFrame),
    Opaque(OpaqueFrame),     // C++-owned: EglImageFrame, MetadataFrame
}
```

`OpaqueFrame` is an RAII wrapper over `(NonNull<c_void>, &'static AvpMediaVtable)`:
`Drop` calls `release`, `Clone` calls `retain`. So C++-owned media is lifetime-safe
*inside* the Rust core even though the core cannot interpret it. This is required:
`MetadataFrame` holds an avcpp `av::Timestamp` and a `Parameters` JSON-ish object
(`src/metadata_frame.hpp:8`), and `EglImageFrame` holds GL/DMA-BUF state
(`src/hwaccel/EglImageFrame.hpp:18`) — neither is expressible as an owned Rust
type, and a Rust node cannot read their fields without a new C accessor surface
(plan-v2 §12 item 3, still open).

Because `Media` is moved into an edge, enqueueing the same value into two edges
does not compile; a second edge requires `.clone()`. See §6.1 for why that
matters.

### 2.2 `Spec` becomes an enum

```rust
pub enum Spec {
    Video { width: i32, height: i32, pix_fmt: i32,
            frame_rate: AVRational, sar: AVRational, time_base: AVRational },
    Audio { sample_rate: i32, sample_fmt: i32,
            layout: AVChannelLayout, time_base: AVRational },   // owned → lossless
    Packet { codecpar: AVCodecParameters, time_base: AVRational },
}
```

Three gains over the flat struct: `width` cannot be meaninglessly present on an
audio spec; the channel-layout degradation disappears from the native path
because `AVChannelLayout` is carried whole; and the `Packet` variant can hold real
`codecpar`, which plan-v2 §4.4.2 (stream config as `Spec`, replacing the
`IEncoder`/`IMuxer` handshake) requires and the flat struct has no room for.

`Spec` becomes `Clone` rather than `Copy`. The edge latch holds one and clones on
delivery. `current_spec()` therefore returns `Option<Spec>` by clone, not by copy
— a signature change from `edge.rs`.

Projection to the flat C `AvpSpec` happens **only** in the adapter, so the
documented degradations live in exactly one file.

### 2.3 Timestamps: `Ts`, and why rsmpeg forces us to define it

rsmpeg has **no timestamp abstraction at all** *(verified — `avutil/timestamp.rs`
is two formatting functions taking `(ts: i64, tb: AVRational)` as separate
arguments; there is no `Timestamp` type in the crate)*. Field access is a bare
`i64` through `Deref` to the raw struct. rsmpeg's design intent is
`unsafe`-elimination, not domain modelling.

This is a regression risk worth naming plainly: avcpp *has* `av::Timestamp{value,
timebase}`, so porting C++ → Rust moves from a typed timestamp to a bare integer
unless we add the type back. We add it back:

```rust
#[derive(Clone, Copy, Debug)]
pub struct Ts { pub val: i64, pub tb: AVRational }

impl Ts {
    pub fn rescale(self, to: AVRational) -> Ts;   // av_rescale_q; explicit
    pub fn is_valid(self) -> bool;                // val != AV_NOPTS_VALUE
}
```

`AVRational` is `#[repr(C)] Copy`, 8 bytes, deriving only `Debug, Copy, Clone`
*(verified — `rusty_ffmpeg` `binding.rs:8586`)*. Two consequences:

- `Ts` is 12 bytes of `Copy` data composed directly from it, so accessors inline
  to the same field loads one would write by hand.
- **`AVRational` has no `PartialEq`.** `Ts` must define equality as rational
  comparison (`av_cmp_q`), not field-wise: field-wise would call `{1,25}` and
  `{2,50}` different. `Ts` deliberately does **not** derive `PartialEq`.

`Ts` does not implement `Add`/`Sub` across differing timebases. Mixing scales
becomes a compile error instead of a plausible wrong number — the single most
valuable property of the type, and the root cause of the bug class in §3.

`AvpRational` (`graph/buffer.rs:9`) is already field-identical to `AVRational`,
so the C boundary needs a cast, not a conversion.

### 2.4 `FrameExt` / `PacketExt`

rsmpeg is thin, so the core adds one extension trait per media type. This is the
same trait that fills the gaps in §2.5:

```rust
pub trait FrameExt {
    fn ts(&self) -> Ts;                 // reads .pts and .time_base together
    fn set_ts(&mut self, ts: Ts);       // writes both — cannot desync
}

impl FrameExt for AVFrame {
    fn ts(&self) -> Ts { Ts { val: self.pts, tb: self.time_base } }
    fn set_ts(&mut self, ts: Ts) { self.set_pts(ts.val); self.set_time_base(ts.tb); }
}
```

`set_ts` writes both fields, so it is structurally impossible to update the
timestamp and forget the timebase. It deliberately does **not** rescale —
`Ts::rescale` is the separate, explicitly-named operation. Conflating the two is
exactly avcpp's trap (§3.2).

### 2.5 Gaps in rsmpeg the core must fill

All *(verified)* against rsmpeg 0.18.0:

| Gap | Detail | Fix |
|---|---|---|
| `AVPacket` has no `Clone` | `avcodec/packet.rs` is 61 lines: `new`, `rescale_ts`, `Debug`, `Default`, `Drop`, setters. No `Clone`. | `PacketExt::clone_ref()` wrapping `av_packet_clone`. Needed by `split`/`one_to_many` on packet edges. |
| No timestamp type | §2.3 | `Ts` + `FrameExt`/`PacketExt` |
| Partial setters | `settable!` covers frame `width/height/pts/time_base/pict_type/nb_samples/format/ch_layout/sample_rate`; packet `pts/dts/stream_index/flags/duration/pos`. Not frame `sample_aspect_ratio`, not frame `duration`. | Same extension traits, via rsmpeg's `UnsafeDerefMut` |
| No `AVFrame.metadata` accessor | `AVDictionary` **is** wrapped, including `copy` (`avutil/dict.rs:10,105`), but the frame's `metadata` pointer is not exposed | `FrameExt::metadata()/metadata_mut()` borrowing the field as an `AVDictionary` view. Needed by `join_metadata` (Tier R) and any future metadata node |
| `av_frame_new_side_data_from_buf` unwrapped | `get_side_data` exists (`avutil/frame.rs:287-293`) and `AVBufferRef` is wrapped (`avutil/buffer.rs:7`), but there is no way to *attach* an existing buffer as side data | `FrameExt::add_side_data_from_buf()`. Same consumer: `join_metadata`'s side-data walk |
| No `Sync` | `wrap_pure!` emits `unsafe impl Send` only | **Not a gap — a feature.** See §6.1. |

Upstreamable, and worth doing, but the core carries them locally either way.

### 2.6 Dependency

rsmpeg 0.18.0 requires `rusty_ffmpeg` `0.16.7` and resolves to
`0.16.7+ffmpeg.8` *(verified in a throwaway probe crate's lockfile)* — the version
the existing `Cargo.toml` already pins. One copy of the FFI types resolves; no
duplicate-type errors. Feature `ffmpeg8`.

---

## 3. Timebase: the one place "just forward it in Spec" corrupts data

This section exists because the obvious two answers are both wrong, and the
failure is silent.

### 3.1 avcpp keeps the timebase outside the struct

avcpp stores `m_timeBase`, `m_streamIndex`, `m_isComplete` in the C++ wrapper
(`deps/avcpp/src/frame.h:163-172`), not in `AVFrame`. Decisively:

> `grep time_base` across avcpp's `frame.h`, `frame.cpp`, `packet.h`, `packet.cpp`
> returns **nothing** *(verified)*. avcpp never reads or writes the native
> `AVFrame::time_base` / `AVPacket::time_base` field.

Two consequences, opposite in sign:

- A native Rust node that trusts `frame.time_base` on a frame produced by a C++
  node reads **`{0,0}`** — not the real timebase. Silent garbage timestamps.
- The native field is therefore **unclaimed**. Nothing in the C++ tree reads it,
  so the native core can adopt it with zero conflict.

### 3.2 avcpp's `setTimeBase` is a rescale, not a setter

`Frame::setTimeBase` rewrites `m_raw->pts` when the timebase changes
(`frame.h:228`), and `setTimeBase(av::Rational())` — i.e. `{0,0}`, avcpp's default
*(verified, `rational.cpp:10`)* — is the idiom for *"forget my timebase without
touching my pts"*, used before `setPts()` to adopt a new one wholesale. There are
**37** `setTimeBase` sites; the forget-idiom appears in `speed.cpp:35`,
`realtime.cpp:271`, `smooth_timestamps.cpp:218` and others.

Several of those change timebase **per buffer**. Under a Spec-only model each
would have to emit a `Spec` event per frame, and every consumer would have to
reconstruct which `Spec` was in effect for which buffer by tracking queue
ordering. That is a bug class we delete for the price of 8 bytes in a field that
already exists.

**Asymmetry to carry into every port:** `Frame::setTimeBase` rescales `pts` and
`best_effort_timestamp` but **not** `duration`; `Packet::setTimeBase` calls
`av_packet_rescale_ts`, which **does** rescale duration *(verified,
`packet.cpp:334`)*. A correct native implementation therefore changes
frame-duration behaviour on port. Flag it per node; do not silently "fix" it.

### 3.3 Decision: the buffer carries the authoritative timebase

FFmpeg 8 has `time_base` on both `AVFrame` (`frame.h:467`) and `AVPacket`
(`packet.h:535`) *(verified against system headers)*. The timebase already has a
home inside the struct being passed, so a wrapper type pairing an rsmpeg value
with a separate timebase field would duplicate storage the struct provides.

- The native core reads and writes the **struct field**, via `FrameExt::ts` /
  `set_ts`. rsmpeg has `set_time_base` for `AVFrame`; `AVPacket` gets it from
  `PacketExt`.
- The adapter bridges `m_timeBase` ↔ `time_base` explicitly in both directions
  (§4.2).
- `Media::Opaque` has no `AVFrame`, so that variant carries an explicit
  `time_base` — matching `EglImageFrame`, which already has its own `tb_`
  (`src/hwaccel/EglImageFrame.hpp:45`).
- **`Spec` still declares the expected timebase**, and remains useful as a
  contract to check against (§6.2). It is not the carrier.

**Caveat to record.** FFmpeg's own comment on `AVPacket.time_base` says its value
"will be by default ignored on input to decoders or muxers." We use it as *our*
carrier, not as a request to libavcodec — fine. But it also means demuxer output
packets will not have it set, so the demux node must populate it from the stream
timebase rather than assuming FFmpeg did.

### 3.4 Fields that are *not* the timebase

- **`m_isComplete` defaults to `false`** (`frame.h:354`). `avutils.hpp:172` gates
  on `isComplete() && pts().isValid()`, and `operator bool()` is `isValid() &&
  isComplete()` (`frame.h:278`). An adapter that forgets `setComplete(true)`
  produces frames that C++ nodes **silently discard** — no error, no output. So the
  adapter sets it on every real frame.

  **But not unconditionally** — that would break EOF. Incompleteness *does* reach
  edges today, because it is how the flush/EOF signal is encoded:
  `createEofMarker<T>()` is a **default-constructed `T`** (`avutils.hpp:155-157`),
  i.e. `pts` = NoPts *and* `m_isComplete == false`, and nodes read that state as
  meaning:

  | Site | Reads incompleteness as |
  |---|---|
  | `encoders.cpp:146` | `if (frame.isComplete())` — comment says *"not a flush frame"* |
  | `decoders.cpp:289` | `pkt.isComplete() && !isEofMarker(pkt)` — same, for packets |
  | `split.cpp:36` | warns if it is about to forward an incomplete non-EOF frame (diagnostic only) |

  Under the new model EOF is an `EdgeEvent::Eof`, not a buffer, so the core never
  carries an incomplete buffer. The obligation lands on the adapter instead: when it
  delivers `Eof` to a C++ node it must **synthesize the marker the node expects** — a
  default-constructed `T` for frames, and for packets specifically a 1-byte packet
  with NoPts, since `isEofMarker(const av::Packet&)` tests
  `p.pts().isNoPts() && p.size() == 1` (`avutils.cpp:127-130`, and `createEofPacket`
  builds `av::Packet({0xFF})`). Get this wrong and encoders/decoders never flush —
  a hang at end of stream, not a crash.
- **Frame `m_streamIndex` is dead.** It is side-band with no native home, but the
  only two reads in the tree (`sentinel.cpp:611`, `:707`) are inside comments
  *(verified)*. It does not need to cross the boundary.
- **Packet `stream_index` is native** — `Packet::streamIndex()` returns
  `raw()->stream_index` *(verified, `packet.cpp:267`)*. Round-trips for free.

---

## 4. The node contract and the adapter

### 4.1 `NodeBody`: one explicit run strategy per node

The skeleton's trait has both `process()` and `poll()` defaulting to `Flow::Eof`
(`node_vtable.rs:38-41`), so a node implementing neither silently reports EOF, and
the executor picks a path by which function pointer is non-NULL. Make it
structural:

```rust
pub trait Node: Send + Sync {
    fn name(&self) -> &str;
    fn body(&self) -> NodeBody;                       // asked once, at start
    fn on_spec(&self, spec: &Spec) -> Result<Spec>;    // default: identity forward
    fn query_interface(&self, id: InterfaceId) -> Option<&dyn Any>;
}

pub enum NodeBody {
    Blocking(Box<dyn FnMut() -> Result<Blocked, NodeError> + Send>),
    Poll    (Box<dyn FnMut() -> Result<Tick,    NodeError> + Send>),
    Async   (BoxFuture<'static, Result<(), NodeError>>),
}
```

`query_interface` returns `&dyn Any` natively — a Rust↔Rust capability query needs
no `const void*`. The C `avp_node_query_interface` keeps its vtable-pointer
signature (§4.3).

**Strategy is chosen by what the body does, not by which language wrote it.**

| Body does | Strategy |
|---|---|
| `av_read_frame`, `avcodec_send_packet`, `av_interleaved_write_frame`, `sws_scale`, `swr_convert`, CUDA launches | **Blocking** — FFmpeg's API is synchronous; no async form exists. Blocking a tokio worker would stall every node in the clock domain. |
| awaits several inputs, or an input plus a clock | **Async** — `select!` over two edges is where straight-line code gets genuinely awkward. |
| must run co-scheduled on a frame-perfect tick | **Async** or **Poll** — shares the clock-domain thread, no thread-scheduling jitter. |

So a native Rust demuxer or encoder is `Blocking`, exactly like its C++
counterpart, and `split`/`firewall`/`packet_relay`/`null_sink` are `Blocking` too
— one input, and `edge.take_blocking()` reads more plainly than a future.
**`Blocking` is the default recommendation for native nodes**, not an FFI-only
fallback; `Async` earns its place in perhaps 2–3 of the Tier-R set.

Mixing is free: rsmpeg's types are `Send` *(verified)*, so a frame produced on an
OS thread and consumed by a tokio task needs no wrapper.

### 4.2 `Flow` splits in two

`Flow { Pushed, Drop, Backpressure, Eof, Error }` (`edge.rs`) serves as both the
result of one edge operation and the result of one node invocation.
`Backpressure` is meaningless as a node result; `Pushed` is meaningless for a node
that pushed to three sinks; `Eof` means "edge closed" in one job and "I am
finished" in the other. Split:

```rust
pub enum Push  { Accepted, Dropped, Full, Closed }   // one edge push attempt
pub enum Tick  { Again, Idle, Done }                 // one Poll invocation
pub enum Blocked { Again, Done }                     // one Blocking invocation
```

Each body returns only the outcomes possible for it. `Idle` is type-excluded from
`Blocking`: a blocking body with nothing to do blocks on its input and does not
return, and if it *could* return `Idle` the executor would spin. `Async` needs no
variants — `Idle` is `.await`, `Done` is the future completing.

Errors move from a payload-free `Flow::Error` into `Result<_, NodeError>`, so the
supervisor gets a message and a source instead of "node 4 said Error".

Executor behaviour:

| Body returns | Blocking (OS thread) | Poll (tokio current_thread) |
|---|---|---|
| `Again` | call again immediately | re-poll up to a fairness budget, then yield |
| `Idle` | *unrepresentable* | `await` the edge `Notify`, then re-poll |
| `Done` | emit `Eof` on all sinks, `stop()`, thread exits | same, then deregister the task |
| `Err(e)` | report to supervisor, exit | report to supervisor, exit |

**Emitting `Eof` on `Done` is the framework's job**, not the node's — otherwise
every one of ~80 ports must remember it. The edge sets its `closed` flag on `Eof`,
so a node that pushes `Eof` itself is idempotent with the automatic emission.

### 4.3 `FfiNode`: the whole FFI surface, in one file

`FfiNode` implements `Node` by holding `(self_ptr, &'static AvpNodeVtable)` and
returns `NodeBody::Blocking` or `NodeBody::Poll`. Its required surface is small:
only **6** C++ nodes are non-blocking (`force_fps`, `obs/obs_video_sink`, `pause`,
`realtime`, `smooth_timestamps`, `speed`), and three of those (`pause`,
`realtime`, `speed`) are Tier C — absorbed into core services. So the adapter needs
the blocking `process()` path plus exactly **three** `Poll` nodes: `force_fps`,
`obs_video_sink`, `smooth_timestamps`.

Per hand-off:

| C ABI call | Native side | Cost |
|---|---|---|
| `avp_edge_take` | `edge.take()` → `Media::Video(f)` → `f.into_raw()` | pointer move; ownership transferred |
| `avp_edge_peek` + `_release`/`_consume` | `edge.peek()` → `Peeked<'_, T>` guard | borrow; one `av_frame_ref` in the shim (§4.3.1) |
| `avp_edge_push` | `AVFrame::from_raw(ptr)` adopt → `edge.push(..)` | pointer move |

**`avp_edge_take` exists so `get()` costs nothing.** The shim's `Source<T>::get()`
returns an *owned* avcpp object; with only peek/pop it would need an `av_frame_ref`
per frame to build one. `take()` transfers the core's ref directly — zero refcount
operations in the common path — and makes "who owns Rust's ref" an ownership
transfer stated in the signature rather than a documented contract (plan-v2 §8.2).

**Peek is the asymmetric case, and `T* peek()` does not survive.** `Edge<T>::peek`
is declared `virtual T* peek(int timeout_ms)` (`src/graph_core.hpp:161`) — it
returns a pointer to a live **avcpp object**, and callers dereference it as one:
`frmin->isComplete()`, `frmin->timeBase()`, `pkt->pts()`. Those are C++ members
(§3), so the pointer must address an object whose `m_timeBase`/`m_isComplete` are
already bridged. The core's queue holds `Media` and contains no such object, so the
shim cannot forward a pointer into it. **§4.3.1 replaces the signature rather than
emulating it** — emulation (a per-port cache of materialized wrappers) would
preserve an API that is already unsound in the current C++ core, and put
cache-invalidation logic in the one component that must be simple.

#### 4.3.1 `peek()` returns a borrow guard, not a pointer

**The current signature is already unsound, independent of this refactor.** `pop()`
destroys the queue slot in place — `element->~T()`, `readerwriterqueue.h:442` — so
every peeked `T*` dangles after the matching pop. One node does this today:

`cuda_rect_overlay.cpp` stores a peeked pointer in `meta_src` (`:766`), then the
next loop **pops those same edges** (`:787`, popping every input whose
`pts() == min_ts` — which is exactly the frame `meta_src` points at), then
dereferences it two calls deeper: `processComposite(..., meta_src)` (`:828`) →
`metadata_src->raw()` (`:501`, `:524`).

It has not been caught because the failure is quiet: `~Frame()` runs
`av_frame_free(&m_raw)`, which nulls `m_raw`, so `if (metadata_src &&
metadata_src->raw())` usually short-circuits and the composite silently loses its
metadata. If the producer thread refills the slot first, it reads a *different*
frame's props instead. Note the same function handles the other pointers correctly —
`src_for_layer[i]` is repointed at `held_[i]` after copying (`:786-789`). `meta_src`
just never got the same treatment. **This is a pre-existing bug in the C++ tree,
not something the refactor introduces** — but it settles the API question.

So the substrate offers no raw-pointer peek. Instead:

```rust
// Native side. The guard borrows the edge; the borrow checker enforces the rest.
pub struct Peeked<'e, T> { edge: &'e dyn Edge, _t: PhantomData<T> }
impl<'e, T> Peeked<'e, T> {
    pub fn get(&self) -> &Media;          // immutable view of the head
    pub fn get_mut(&mut self) -> &mut Media;
    pub fn consume(self) -> Media;        // pop + hand over ownership
}                                          // drop = leave it queued
impl<T> In<T> {
    pub fn peek(&self, timeout: Timeout) -> Option<Peeked<'_, T>>;
}
```

`cuda_rect_overlay`'s bug is a compile error in this form: `consume()` takes `self`,
so the guard is gone before the pop can be observed, and holding a `&Media` across
it fails the borrow check.

The C ABI mirrors it as an opaque handle, because the never-ported nodes need it:

| C ABI call | Meaning |
|---|---|
| `avp_edge_peek(edge, timeout_ms, AvpBuffer* out)` → `AvpPeek*` | acquire a borrow; `NULL` = nothing there. `*out` is valid only until release/consume |
| `avp_edge_peek_release(AvpPeek*)` | leave the item queued (guard dropped) |
| `avp_edge_peek_consume(AvpPeek*, AvpBuffer* out /*nullable*/)` | pop. `out == NULL`: the core releases its ref (the caller already made its own). `out != NULL`: the core moves its ref to `*out` |

The nullable `out` on `_consume` is the one place this ABI still asks the caller to
get an ownership decision right, so it is stated at the declaration (plan-v2 §4.3):
**the avcpp shim always passes `NULL`**, because constructing `av::VideoFrame` from
the borrowed `AVFrame*` already took a ref. Plain C that never materialized a copy
passes a buffer and releases it. Passing `out` from the shim leaks; passing `NULL`
without having ref'd double-frees. This asymmetry is why the shim wraps the triple in
a guard once and no node calls it directly.

The shim's `Source<T>::peek()` returns a **move-only RAII guard** wrapping
`AvpPeek`, exposing `operator bool` / `operator->` / `operator*`, a named `.mut()`
for the one mutating site, and `.consume()`. It constructs the avcpp object once, in
its own storage, with the §3 bridge applied — so peek costs one `av_frame_ref`
(`frame.h:129-135`) and `.consume()` then costs nothing, since the guard's object
*is* the result. Same total as `take()` for peek-then-consume; the extra ref is paid
only by a node that peeks and declines, which is the wait-for-other-input case
(`mux`, `source_switcher`).

No cache, no invalidation protocol, no per-port slot table: **the guard's scope is
the lifetime.** Multi-input nodes hold two guards
(`join_metadata.cpp:51-52`) — naturally, because they are values.

Two different counts matter, and they are easy to conflate. **The M-pre conversion of
the C++ tree touches every peek: 49 sites in 24 files** (`grep -rn -- "->peek("
src/`, excluding `_unfinished`), 4 of them in `graph_base.hpp` — the base class's own
`consumeEofIfPresent`/least-PTS helpers, which every multi-input node inherits.
**The shim's permanent burden is the 33 sites in 15 files** that are neither Tier C
(absorbed: `realtime` 2, `speed` 2, `pause` 2) nor Tier R (ported: `split` 2,
`join_metadata` 4, `one_to_many`, `force_keyframe`, `debug/*` 2) — i.e. 18 in Tier-S
recompiles, 11 in never-ported nodes (`cuda_rect_overlay` 8, `cuda_to_egl_image`,
`nvof_fruc`, `obs_video_sink`), and the 4 base-class sites.

Two populations size M-pre, and they must not be added together: the **peek-call
sites** (where the guard is acquired — these are what stop compiling) and the
**pointer-usage sites** (where the result is tested — these are what get edited).

Peek-call sites, partitioning all 49:

| Call-site form | Count | Effect |
|---|---|---|
| `T* p = edge->peek();` — a declaration | 43 | **stops compiling**; `auto p = …` |
| `p = edge->peek();` — assign to an existing pointer (`source_switcher.cpp:83`, `filters.cpp:584`) | 2 | **stops compiling**; restructure to a scoped guard |
| `bool have_input = src->peek(0);` (`debug/delaygen.cpp:15`) | 1 | `.has_value()` / `operator bool` |
| `while (src->peek(0))` (`debug/jittergen.cpp:15`) | 1 | compiles unchanged (`operator bool`) |
| inline `… ->peek() != nullptr` (`cuda_rect_overlay.cpp:656`) | 1 | one-token edit |
| bare call for its side effect (`pause.cpp:15`) | 1 | Tier C — disappears |

Of the 45 sites that bind a pointer, the *first subsequent test* is a
`nullptr` comparison at **21** and a boolean test at **23** (1 untested); the
comparisons are a one-token edit each, the boolean tests compile unchanged through
`operator bool`. Two sites additionally do more than test: `bsf.cpp:121` mutates
through the pointer (`pktp->setTimeBase(...)` → `.mut()->setTimeBase(...)`), and
`cuda_rect_overlay` stores it beyond the pop — **the bug**, which must become
`.consume()`.

That the declaration breaks in 45 of 49 places is the point: 24 files get looked at
once, each mechanically, and the one real defect cannot hide among them. It is a
`sed`-scale change plus one genuine fix — and it is done in the C++ tree
**before** M2, so the shim is written against an API that is already sound.

Honest limit: in C++ this makes the dangerous form unnatural, not impossible — a
caller can still `&*guard` and outlive it. Only the native path gets a real
guarantee. The lifetime bound to a *value* is still what makes the shim's job
tractable, and it removes the only known instance.

The adapter also owns the bridges from §3:

1. `m_timeBase` ↔ native `time_base`, both directions.
2. `setComplete(true)` on every outgoing **real** frame — plus, on `Eof`,
   synthesizing the incomplete marker C++ nodes flush on (§3.4: default-constructed
   `T` for frames; 1-byte NoPts packet for `av::Packet`).
3. `Spec` → flat `AvpSpec` projection, with the channel-layout degradation
   documented here and nowhere else.
4. The `Source<T>::peek()` borrow guard (§4.3.1) — no cache, but it owns the avcpp
   object it hands out.

`AvpFlow` in the header stays one flat enum — it is the compat layer, and the shim
already returns it:

| C returns | Blocking path | Poll path |
|---|---|---|
| `PUSHED` / `DROP` | `Ok(Again)` | `Ok(Again)` |
| `BACKPRESSURE` | adapter waits on the sink's writable `Wakeup`, then `Ok(Again)` | `Ok(Idle)` |
| `EOF` | `Ok(Done)` | `Ok(Done)` |
| `ERROR` | `Err(NodeError::Node { name })` | same |

The `BACKPRESSURE`-on-blocking row would otherwise busy-spin: a blocking C++ node
should not return it, but if it does, the adapter absorbs it by performing the wait
the node skipped.

**Accepted loss:** `AVP_FLOW_ERROR` carries no string, so a C++ node's error
reaches the supervisor with only its node name. If that proves insufficient, one
`avp_node_last_error(node) -> const char*` fixes it. Left out until a port needs
it.

### 4.4 Typed pads

`In<T>` / `Out<T>` are newtypes over `Arc<dyn Edge>` plus `PhantomData<T>`, with
the media type checked at **bind** time and reported as a connect-time error
naming both pads. This addresses the README's "in/out pads as a separate data
structure, not merely a queue" point.

Monomorphization cost was measured, not assumed: in a standalone crate with five
media types, one non-generic `BufferedEdge`, and a generic `Firewall<T>`, each
`Firewall<T>::run` instantiation was **67–71 bytes** (~350 B for all five), and
**`In<T>`/`Out<T>` emitted no symbols at all** — fully inlined, since `unwrap` is
one discriminant compare and `PhantomData` is zero-sized.

The C++ template blowup this resembles came from templating the *entire edge
stack* — `Edge<T>`, `EdgeWrapper<T>`, `EdgeSource<T>`, `EdgeSink<T>` (ring buffer,
condvars, flush logic, stats) × 5 types, plus 11 `DECLNODE_ATD` sites
(`src/graph_core.hpp:156-227, 369, 673`). Here the edge stack is non-generic and
only the thin pad handle is typed.

Honest limits: the ×5 instantiation cost exists in *any* scheme that supports
generic nodes (that is what `DECLNODE_ATD` means); media-specific nodes
instantiate once at zero cost and are the majority; and the real risk is `async fn`
in generic traits duplicating whole state machines — so **async node bodies stay
non-generic where the media type is fixed.**

---

## 5. Native embedder and service API

Three places where the skeleton forces a Rust caller through a C-shaped keyhole.
Each gets a native form; the C entry point is reimplemented on top, so
`register_factory`, the M0 smoke test, and the plan-v2 §4.7 graph-management ABI keep
working.

### 5.1 Params: `serde` instead of `&str`

A factory today is `Fn(&str, &str) -> Result<Arc<dyn Node>, String>`
(`factory/mod.rs:14`) — the second `&str` is a JSON blob each node parses itself.
Natively:

```rust
pub trait NodeSpec: DeserializeOwned {
    const TYPE_NAME: &'static str;
    type Node: Node;
    fn build(self, ctx: &BuildCtx) -> Result<Self::Node, BuildError>;
}
```

The registry parses once and hands over a typed struct, so a missing field or bad
enum value produces one `serde` error with a field path, generated where the type
is known — instead of ~80 hand-written `params["fps"].as_f64().ok_or(...)` chains
each with its own wording. `#[serde(deny_unknown_fields)]` turns a typo'd
parameter into an error rather than a silently-ignored default, which the current
`Value`-poking cannot do. Concretely: `realtime` has **15** `params.count(...)`
guards (`src/nodes/realtime.cpp`), every one of which silently accepts a
misspelled key today.

`BuildCtx` is where a node reaches its clock and the shared registry — typed, so
`ctx.sync_group("main")` returns `Arc<dyn SyncGroup>`.

### 5.2 Dynamic `node.add` keeps working — via type erasure at registration

`node.add` does `json::parse(arg)` then `createNode(params)`
(`src/avplumber.cpp:382`), and the same flat object carries **both** framework keys
(`group`, `auto_restart`, `on_error` — `graph_mgmt.cpp:392-435`) and node keys
(`tick_period`, `speed`, `team`). So the dynamic path needs a two-stage parse.

The registry cannot store `Fn(S) -> _` per distinct `S`, so registration erases it
— the `from_value::<S>` call is baked *inside* the closure and monomorphized once
per node type at register time:

```rust
type ErasedFactory = Arc<dyn Fn(&str, Value, &BuildCtx)
                          -> Result<Arc<dyn Node>, BuildError> + Send + Sync>;

pub fn register<S: NodeSpec>(&mut self) {
    self.factories.insert(S::TYPE_NAME, Arc::new(|name, v, ctx| {
        let spec: S = serde_json::from_value(v).map_err(|e| BuildError::params(name, e))?;
        Ok(Arc::new(spec.build(ctx)?))
    }));
}
```

Envelope and node params split by **extract-then-remainder**, not
`#[serde(flatten)]`:

```rust
let mut obj: Map<String, Value> = serde_json::from_str(arg)?;
let env  = NodeEnvelope::extract(&mut obj)?;   // removes type/name/group/auto_restart/…
let node = registry.create(&env.type_name, &env.name, Value::Object(obj), &ctx)?;
```

serde does not support `deny_unknown_fields` together with `flatten`, and
`deny_unknown_fields` is the feature earning its keep — so envelope keys are
removed first, keeping it usable on every node `Spec`.

One registry serves the native builder, `.avplumber` scripts, TCP, and pyplumber.
**The stringly paths gain better errors but not compile-time checking** — only
in-process Rust embedders get that. A bad *value* in a script still fails at
startup; wrapping the error with node name and type is the whole gain there.

avplumber's params are not plain scalars: `tick_period` is `"30/1"` parsed by
`parseRatio` (`realtime.cpp:488`), and several params are float seconds converted
to timebase units. These become newtypes with one `Deserialize` impl each —
`Ratio`, `Seconds` — written once and reused across ~80 nodes. Edge-name params
(`input_ts_queue`) become `EdgeRef(String)`, resolved by `BuildCtx` at build.

### 5.3 Services: `Arc<dyn T>` instead of a vtable pointer

`ServiceRegistry` stores `HashMap<AvpServiceId, *const c_void>`
(`services/mod.rs`), so a Rust caller gets a raw pointer to a C vtable describing
a Rust object it already owns — and **nothing keeps the pointed-to object alive**.
Replace the storage with `HashMap<TypeId, Arc<dyn Any + Send + Sync>>`:

```rust
impl Instance {
    pub fn service<T: Service>(&self) -> Option<Arc<T::Iface>>;
    pub fn register_service<T: Service>(&self, impl_: Arc<T::Iface>);
}
```

`avp_core_query_service` becomes: look up the `Arc`, return a pointer to a static
vtable whose functions downcast and call the trait. Vtable construction moves to
one generated site per service, and the `Arc` fixes the lifetime hazard.

### 5.4 Graph building: typed handles instead of stringly bind

```rust
let mut g = GraphBuilder::new(&instance);
let dec = g.add::<Decoder>(DecoderSpec { .. })?;      // NodeRef<Decoder>
let rt  = g.add::<Realtime>(RealtimeSpec { .. })?;
g.connect(dec.out(), rt.input(), Capacity(16))?;      // media-checked here (§4.4)
g.build()?;
```

`bind_endpoint` (`abi/node.rs:90`) resolves the vertex by name and — if the name
does not match — **silently skips the insert** (`if let Some(v) =
g.vertex_mut(...)`) while still recording the link and returning a valid-looking
handle. That failure mode disappears when the handle *is* the reference.

---

## 6. What the compiler now enforces, and what it does not

### 6.1 Fan-out and mutation invariants

Five invariants govern sharing an `AVFrame`/`AVPacket` (plan-v2 §8.2.1 states them
in full, along with why the present node set satisfies them only by luck). For
**native** nodes, three stop being contractual:

| Invariant | Native enforcement |
|---|---|
| distinct struct per edge | `Media` is *moved* into one edge; a second edge requires `.clone()` (= `av_frame_clone`, new struct). Enqueueing the same value twice does not compile. |
| never touch one struct from two threads | rsmpeg is `Send` but **deliberately not `Sync`** *(verified)*. Sharing one `AVFrame` across threads is a **type error**. |
| lifetime / no double-free | `Drop` |
| CoW before in-place write | `make_writable()` exists and returns `Result`, but calling it remains the node author's responsibility — **review-enforced, not compiler-enforced.** |
| no in-place `side_data` mutation | **review-enforced.** Unchanged from v2. |

So three of five become unrepresentable natively and two remain contractual. For
**C++** nodes all five remain contractual. This is a real narrowing of the risk
surface, not its elimination.

### 6.2 Seam checks for what types cannot catch

The adapter is the one place every buffer crosses, so in debug builds it asserts
per hand-off that `tb.num != 0 && tb.den != 0`, and that the buffer's timebase
equals the edge's latched `Spec` timebase unless a `Spec` event declared the
change. A node that rewrites a timebase without announcing it fails a test rather
than desyncing A/V in production.

### 6.3 Testing

Detailed in a follow-up section of the implementation plan; the shape is:

- **Refcount, under ASan, `--features ffmpeg`** (plan-v2 §8.2.1's requirement):
  fan-out balance across N edges; flush balance via `FlushStart` mid-queue; CoW
  split; `into_raw`→C→`from_raw` round-trip asserting the refcount is unchanged.
- **Adapter**, targeting the silent-corruption modes above: timebase round-trip
  native→avcpp→native asserting `Ts` equality; `isComplete` set on every outgoing
  real frame; `Ts` equality is rational (`{1,25} == {2,50}`); `PacketExt::clone_ref`
  refcount balance; demux populates packet `time_base`. Two more from §3.4/§4.3,
  both **failure modes that produce a hang or a stale read rather than a wrong
  value**, so unit tests are the only place they surface cheaply:
  - **EOF marker synthesis** — feed `EdgeEvent::Eof` to an `FfiNode` wrapping a
    stub C++ node and assert `isEofMarker(received)` is true, for `av::Packet`
    (1-byte, NoPts) and for a frame (default-constructed). Then assert a real
    encoder actually flushes: `Eof` in → trailing packets out.
  - **Peek guard** (§4.3.1) — dropping a guard leaves the item queued and the next
    peek sees the same head; `.consume()` pops exactly once and the refcount is
    balanced; two guards on two ports of one node are simultaneously valid
    (`join_metadata`'s pattern); `peek(timeout)` still blocks for the timeout. Plus a
    regression test for the `cuda_rect_overlay` defect: a node that peeks, consumes,
    and then reads the consumed value must see intact props — the bug's signature is
    silently-empty metadata, so assert on the props, not on absence of a crash.
- **Graph**, extending `tests/smoke_2node.rs` rather than replacing it (it drives
  the plan-v2 §4.7 ABI and remains the proof the C path works): media mismatch at
  `connect()`; `deny_unknown_fields` error naming the node; `Poll` returning `Idle`
  does not spin; `Done` emits `Eof` on all sinks exactly once.
- **Parity harness** — the one that actually de-risks the refactor: run the M2
  pipeline (`input_rec` → decode → `realtime` → encode/mux) on both cores and diff
  output PTS sequences frame-by-frame. Timebase and `isComplete` bugs are exactly
  the class that passes unit tests and shows up as drift over minutes. This
  requires `SyntheticClock` (`services/clock.rs:53`) to be real rather than a stub
  returning `AVP_NOPTS` — an M1 item, not free.

---

## 7. Known defects in the current skeleton

Found while verifying this design; each is a prerequisite, not a side quest.

1. **`avplumber_f7k` does not compile.** `exec/blocking.rs:42` calls `log::warn!`
   with no `log` dependency in `Cargo.toml` (`error[E0433]: cannot find module or
   crate 'log'`).
2. **`graph/buffer.rs` `retain` passes the wrong arity, on both media kinds** —
   `av_frame_ref(buf.ptr as *mut AVFrame)` at `:117` and
   `av_packet_ref(buf.ptr as *mut AVPacket)` at `:120`, each with one argument; the
   real signatures are `av_frame_ref(dst, src)` / `av_packet_ref(dst, src)`. Would
   fail to compile under the `ffmpeg` feature. Moot once §2 replaces these functions
   with `Drop`, but two independent instances of the same error is the evidence: the
   `ffmpeg` feature has never been compiled.

---

## 8. Where this document sits

- **`rust_refactor_plan_v2.md`** — the C ABI (`avplumber_core.h`), the edge/control
  model (buffers + causal tokens + flush-preempts-queue), capability discovery,
  scheduling, core services, the C++ shim, phasing, and the open questions. This
  document specifies the core's *internal* model that the ABI is a compat layer
  over; where the two describe the same thing, this one describes the native side
  and v2 the C side.
- **`rust_refactor_impl_breakdown.md`** — the build sheet: node tiers, the recipe
  for adapting a C++ node, milestones, effort.
- **`rust_refactor_headers.md`** — draft C/C++ meant to be dropped into `src/`.

**Milestone placement.** The native data model changes the edge's payload *type*,
so it lands in M0/M1, before any node is ported — building the edge on raw pointers
first means writing the refcount code twice and deleting it once. The `FfiNode`
adapter and its bridges land with the shim in M2, which is where the
timebase/`isComplete` correctness work concentrates and where the parity harness
must already be running. The C++-tree peek conversion (§4.3.1) is M-pre: no Rust
dependency, so it runs in parallel with M0/M1. Full table:
`rust_refactor_impl_breakdown.md` §4.

### 8.1 Decisions locked by this design

- Native path is **lossless**; the C path may be lossy where no C++ node reads
  the field.
- The C ABI **co-evolves** with the core — in-tree C++ nodes and pyplumber are
  recompiled together. No frozen/versioned ABI for third-party plugins.
- Conversion happens **at the node boundary** (`FfiNode`), not at the edge and not
  via two edge implementations. Two edge impls would have to agree on
  flush-preempts-queue, `Spec` latching, and backpressure — for the saving of a
  pointer move.
- Edge typing is **typed pad handles, checked at bind**.
