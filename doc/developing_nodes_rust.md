# Writing nodes in Rust

This is the Rust counterpart of [developing_nodes.md](developing_nodes.md). Read
the [README](../README.md) first: it explains what a node, an edge, a group and
a script are, and none of that is repeated here. The design behind the Rust core
is in [specs/rust-refactor/](specs/rust-refactor/); you do not need it to write
a node, but `rust_refactor_native_core.md` is where the *why* of `Spec`, `Media`
and `NodeBody` lives.

The guide is organised the way a node file is: where it lives, how it is
registered, which kind of body it has, and then each part of the file in order.
The last sections are about keeping nodes small and alike: what to reuse, what
to extract, and what to leave alone.

## Two crates, one direction of dependency

- `avplumber_f7k` is the framework: graph, edges, executors, supervisor, the
  control protocol, the libav helpers and the node-authoring scaffold.
- `avplumber_nodes` holds the media nodes, one node (or one family of nodes that
  share parameters, like `dec_video`/`dec_audio`) per file.

The framework knows nothing about the nodes crate. Anything a node needs from the
framework is `pub`; anything two nodes need is a framework helper, not a copy.

A node is registered in `avplumber_nodes/src/lib.rs`, in `register_media_nodes`:

```rust
avplumber_f7k::register_spec::<null_sink::NullSinkSpec>(inst);
```

Nodes that touch libav are declared under `#[cfg(feature = "ffmpeg")]`, both
the `pub mod` line and the registration, so the framework and the pure-Rust
nodes still build without FFmpeg. `mux` is the example of a node that stays in
the default build: it orders timestamps and describes a container, but never
calls libav.

## The contract, and why you do not implement it

The runtime contract is one trait, `Node` (`avplumber_f7k/src/graph/node.rs`).
It has a dozen methods and every one of them has a default. The executors drive
a node through `NodeBody`, which `Node::take_body` produces once at start, and
which is one of three things:

| `NodeKind` | body | runs on | one step returns |
|---|---|---|---|
| `Blocking` | `FnMut() -> Result<Blocked, NodeError>` | its own OS thread | `Blocked::{Again, Done}` |
| `Poll` | `FnMut(&mut NodePollContext) -> Result<Tick, NodeError>` | a shared event loop | `Tick::{Again, Idle, Done}` |
| `Async` | a future | the same event loop | when it completes |

A node implements neither `Node` nor a body. It implements one of three smaller
traits from `avplumber_f7k::scaffold`, and is registered behind the matching
newtype:

| you write | you register | the wrapper supplies |
|---|---|---|
| `impl InputHandler` + `impl SingleInput` for `X` | `type Node = Blocking<X>` | everything the row below supplies, plus the whole read loop: which hook each item on the input calls, forwarding of the control events, pushing what a hook produced |
| `impl BlockingNode for X` | `type Node = Blocking<X>` | name, kind, pads, edge binding, park reset on start, park wake on interrupt, `process` |
| `impl PollNode for X` | `type Node = Polling<X>` | name, kind, pads, edge binding, the Direct-edge opt-in, `poll` |

`SingleInput` is the one to reach for. A blocking node with one input never
writes a loop, only reactions — `on_spec`, `on_buffer`, `on_flush`,
`on_flush_stop`, `on_eof` —
and, when it has output that is not a reaction to an input, what happens before
each read (`before_take`). It is a `BlockingNode` through a blanket impl, so it
is wrapped and registered like one. `BlockingNode` itself is for the nodes with
no input (`input`) or several; `PollNode` for the cooperative ones, whatever
their pad count (`mux`, `demux`).

Every trait asks for `io()` — the node's name and its two edge slots, kept as a
field of the node — and one body method. Everything else — `pads`,
`start`, `stop`, `interrupt`, `bind_source`, `bind_sink`, `before_take` — is a
hook with a default, overridden only by the nodes that need it. The C++
`NodeSISO`/`NodeSingleInput` base classes did the same job with inheritance;
here it is composition, because a newtype can be given a `Node` impl without
the node traits colliding.

If you find yourself implementing `Node` directly for a media node, stop and
check what the scaffold is missing. Implementing it by hand is legitimate for
test fixtures and for adapters (`FfiNode` wraps a C vtable that way), not for
a node that reads or writes media.

## Blocking or poll?

Decide this first; it determines the trait, the shape of `step`, and which
thread your code runs on.

**Blocking** is for a body that has to wait somewhere the executor cannot see:
inside libav (`av_read_frame`, `avcodec_send_packet`,
`av_interleaved_write_frame`, `sws_scale`), inside a device driver, or simply on
its input with `take(-1)`. It costs a thread and a context switch per hop, and
buys straight-line code: `input`, `dec_*`, `enc_*` and `output` are blocking, and
so is anything that calls a synchronous library. This is the default choice, not
a fallback.

**Poll** is for a body that never waits: it takes what is there, pushes what it
can, and when it cannot proceed it tells the `NodePollContext` what it is waiting
for and returns `Tick::Idle`. Several such nodes share one event loop thread, so
a chain of them passes buffers without a context switch. `mux` and `demux` are
poll nodes: they only move packets between edges and never touch libav.

A poll node is also the only kind that can promise to be **infallible**, which
is what a `DirectEdge` needs: a Direct edge runs its consumer's `step` from
inside the producer's `offer`, on the producer's stack, where there is no
executor to report an error to. `PollNode::direct_consumer_is_infallible`
returning `true` is that promise, and connecting a Direct edge to a node that
does not make it is rejected. Make the promise only when `step` cannot return
`Err` by construction: parameters already validated in `build`, no libav, no
"unexpected item" arms that fail. A fallible poll node works everywhere else
and is what most poll nodes are.

The `Async` kind has an executor but no scaffold yet. `SisoAsyncAdapter` shows
what implementing `Node` for it looks like; use it only when `select!` over two
inputs or an input plus a clock is genuinely what the node does.

## Anatomy of a node file

The files in `avplumber_nodes/src/` follow one order, and a reader relies on it.
`null_sink.rs` is the smallest complete example; `decode.rs` is the reference
for a blocking node with codec state; `demux.rs` for a poll node with several
output pads; `mux.rs` for several input pads and a hand-stepped unit test.

1. **Module doc.** What the node does, which C++ node it ports, and what it
   deliberately does differently or has not implemented yet. The "deferred"
   list is part of the contract: a parameter that C++ accepted and this node
   does not must be *rejected* in `build`, not silently ignored, and the module
   doc says so.
2. **Imports.** Framework paths spelled out (`avplumber_f7k::graph::edge::...`)
   rather than the crate-root re-exports, so a reader can find the definition.
3. **Constants** with the C++ name they correspond to in the doc comment.
4. **The spec struct** — the node's parameters, `#[derive(Deserialize)]`.
5. **`impl NodeSpec`** — the type name, the wrapper type, and `build`.
6. **`State`** — everything that changes while running, behind one `Mutex`.
7. **The node struct** — `io`, the spec, build-time derived fields, `state`.
8. **`impl InputHandler` + `impl SingleInput`** (or `impl BlockingNode`, or
   `impl PollNode`) — the reactions, then `io`, `state`, `pads` and the hooks.
9. **Private helpers** in an `impl Node {}` block, in the order `step` calls them.
10. **Free functions** that need no `self`, and `#[cfg(test)] mod tests`.

The smallest node, trimmed of the counters its tests read:

```rust
#[derive(Debug, serde::Deserialize)]
pub struct NullSinkSpec {}

impl NodeSpec for NullSinkSpec {
    const TYPE_NAME: &'static str = "null_sink";
    type Node = Blocking<NullSink>;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        Ok(Blocking(NullSink {
            io: BlockingIo::new(name),
            counters: counters(name),
        }))
    }
}

pub struct NullSink {
    io: BlockingIo,
    counters: Arc<Counters>,
}

impl InputHandler for NullSink {
    fn on_spec(&self, _spec: Spec) -> Result<Option<Spec>, NodeError> {
        self.counters.specs.fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }

    fn on_buffer(&self, _buffer: Media) -> Result<Option<Media>, NodeError> {
        self.counters.buffers.fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }
}

impl SingleInput for NullSink {
    fn io(&self) -> &BlockingIo {
        &self.io
    }
}
```

`on_flush` and `on_eof` keep their defaults: nothing to discard, and `Eof`
finishes the node. There is no loop, no `match` on the input, no lock (the
counters are atomics), and no pad declaration, the last on purpose (see
*Edges, pads and `Io`*).

## Parameters: the spec struct

The script's JSON object, minus the envelope keys the control layer strips
(`type`, `name`, `group`, `src`, `dst`, `auto_restart`, ...), is deserialized
straight into the spec struct. Rules that keep this honest:

- **Declare every parameter once.** The node keeps the whole spec struct as a
  field (`params: DecoderParams`) and reads `self.params.codec` at runtime. Do
  not copy fields out into the node struct one by one; that is how C++ ended up
  with parameters declared three times.
- **Derive what needs deriving, in `build`.** A field that is only useful once
  parsed (`pixel_format` into an `AVPixelFormat`, `eof_mode` into a `bool`,
  `routing` into `Vec<Route>`) becomes a separate field on the node, and the
  spec field's doc comment says "build-time only" so nobody reads the raw
  string later. `input.rs` and `decode.rs` both do this.
- **`#[serde(default)]` on every optional field**, `Option<T>` when absence
  means something different from a default value.
- **Validate in `build`, return `Err(String)`.** A bad parameter fails
  `node.add` with the message; that is the only place a user sees it. Runtime
  code assumes parameters are valid.
- **Reject what you have not implemented.** `hwaccel: Option<Value>` on the
  decoder exists only so `build` can say "not implemented in the Rust core yet"
  instead of quietly decoding on the CPU.
- **Two type names, one parameter set:** a `#[serde(transparent)]` newtype per
  type name around the shared struct, each with its own `TYPE_NAME` and a
  one-line `build`. See `VideoDecoderSpec`/`AudioDecoderSpec`.
- **Parameters that name edges** (demux's `routing` values) are returned from
  `NodeSpec::bindings`, so a script does not have to repeat them in `dst`.
- `BuildCtx` is how a node reaches instance services by name (`ctx.clock(..)`,
  `ctx.correction(..)`, `ctx.timeline(..)`). It is the only way; a node never
  holds an `Instance`.

## State

A node is shared: the executor holds it in an `Arc`, `interrupt` arrives from
another thread, and a poll node's `step` may be entered from a producer's
`offer`. So every method takes `&self`, and mutable state lives behind
interior mutability, typically one `Mutex<State>` locked where it is used:

```rust
let state = &mut *self.state.lock().unwrap();
```

That line is the node's, not the scaffold's: every hook takes `&self`, and
what a node does inside is its own business. One `Mutex<State>` locked per hook
is the common shape (the codec nodes), a handful of atomics is another
(`null_sink`), several fields with their own locks a third. The framework
asks only for `Send + Sync`. Nothing is held across the wait for input, so a
reader on another thread — a status query, one day — is never made to wait
for the next packet.

What goes where:

- **`State`**: codec contexts, counters, "have I published the spec yet",
  pending buffers, anything reset by `start`.
- **The node struct**: `io`, the spec, build-time derived values, and anything a
  libav callback points at through `opaque` (`decode.rs` keeps its
  `PixelFormatRequest` in an `Arc` on the node for exactly that reason: the
  pointer must outlive every context it is handed to, and `State` is replaced
  on a restart).
- **Not edges.** Edges live in `Io`'s slots, or in a node-owned map for the
  multi-pad case, never inside `State`: they are bound before `start`, rebound
  on a reconstruction, and read on every step.

`start` runs before the first step of every run. Whether the instance is new
depends on how the run began: after a fault the supervisor rebuilds every
member from its spec and refuses to restart a surviving instance (the
`RestartHook` docs in `supervisor/mod.rs` explain why), while a group that is
stopped and started by hand keeps its instances. Write `start` for both cases:
reset what a run must not inherit (counters, pending buffers, the codec's
internal buffers), keep what is expensive and still valid, and say which is
which in a comment. The decoder keeps its open codec and `input_spec`, because
the edge re-delivers its latched `Spec` at the start of a run and a spec that
compares equal must not reopen the codec.

`stop` runs on the body's thread after the last step, whatever ended the body:
`Done`, `Err`, or a group stop. It is where `output` writes the trailer and
`input` closes the container. Do not do that from `Drop`; `stop` has a thread
and a defined moment, `Drop` has neither.

## Edges, pads and `Io`

`Io` holds the name and two `EdgeSlot`s. `BlockingIo` adds the park a blocking
body waits on. The helpers on them are the whole reason they exist, so use them
rather than reaching into the slots:

| call | what it does |
|---|---|
| `self.io.input()?` / `self.io.output()?` | the bound edge, or a `NodeError` — an unbound pad is a script mistake and fails the group |
| `self.io.error(phase, msg)` | a `NodeError` carrying this node's name |
| `self.io.push(&out, buffer)?` | blocking only: push, parking for room; `Blocked::Done` when interrupted or the edge closed |
| `self.io.push_with(&out, buffer, \|\| ...)` | the same, running the closure on every wake while parked (see *hints* below) |
| `self.io.is_interrupted()` | blocking only: a stop landed; return `Done` rather than block again |
| `self.io.wait(ms)` | blocking only: an interruptible sleep, for "libav wants time, not data" |

`pads` declares what the node connects to. Use the constructors:
`NodePads::input(media)` for a sink, `NodePads::output(media)` for a source,
`NodePads::siso(in, out)` for a transform. A node that declares no pads skips
the media-type check at connect; `null_sink` does that on purpose because it
accepts everything, and its comment says so. Everything else declares pads.

The names `in`/`out` are conventions, not requirements: a script's `src`/`dst`
binds to the single declared pad on that side whatever it is called. Names only
matter for nodes with several pads on one side, and there the pad names are
the script's own strings: `mux` names its input pads after the `src` entries,
`demux` names its output pads after the `routing` keys. Those nodes keep the
extra edges in their own map and override `bind_source`/`bind_sink`; the `Io`
slot on that side stays empty.

## The blocking body: reactions, not a loop

A `SingleInput` never reads its edge. The scaffold's loop does, once per step:

1. return `Done` if a stop has landed;
2. call `before_take`, and make its answer the step's result if it has one;
3. block on the input with `take(-1)`; an empty result means the edge is
   closed or the stop interrupted the wait, so `on_closed` runs and the node
   is done;
4. classify the item and call its hook (`react`): `on_spec`, `on_buffer`,
   `on_flush`, `on_flush_stop` or `on_eof`;
5. forward what the hook did not: the spec `on_spec` returned, `FlushStart`
   after `on_flush`, `FlushStop` with its `resume_at` after `on_flush_stop`,
   and `Eof` when `on_eof` said `Done`; push what `on_buffer` produced,
   parking for room.

`on_flush_stop` is where seek precision lives. A source that can only
reposition to a keyframe says in the `FlushStop` which position it aimed for,
and a node that turns buffered input into timestamped output — the decoder —
drops what lies below it until the first frame at or past it. Nothing reaches
into the decoder from outside the graph for this; the C++ `discardUntil` and
`flush_magic` have no counterpart.

What the encoder writes, with the bodies elided:

```rust
impl InputHandler for Encoder {
    fn on_spec(&self, spec: Spec) -> Result<Option<Spec>, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        // Open, or reopen if the format changed; answer with the codec parameters.
    }

    fn on_buffer(&self, buffer: Media) -> Result<Option<Media>, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        self.load(state, buffer);
        Ok(None)
    }

    fn on_flush(&self) {
        // flush_buffers, reset the pump, forget the timestamps.
    }

    fn on_eof(&self) -> Result<Blocked, NodeError> {
        // Start the drain; `before_take` finishes it.
        Ok(Blocked::Again)
    }

    fn on_closed(&self) {
        self.log_drops(&self.state.lock().unwrap());
    }
}

impl SingleInput for Encoder {
    fn io(&self) -> &BlockingIo { &self.io }
    fn pads(&self) -> NodePads { NodePads::siso(self.media, AvpMediaType::PACKET) }
    fn start(&self) { /* reset counters; the open codec survives */ }

    fn before_take(&self) -> Result<Option<Blocked>, NodeError> {
        // Emit what the pump holds; after the drain forward `Eof` and finish;
        // drive a codec that refused its last input, waiting if it stalled.
    }
}
```

The rules that follow from the loop:

- **A hook returns what the loop forwards; it does not push.** `on_spec`
  returns the spec to publish (the encoder's codec parameters) or `None` when
  it is not known yet (the decoder reads it off the first frame). `on_buffer`
  returns the buffer to push, or `None` for a consumer, or for a codec whose
  output appears later. The one push a node writes itself is in `before_take`,
  through `self.io.push`, because that is where a codec's output shows up.
- **`before_take` is for output that is not a reaction to an input**: a codec's
  pending frames, the drain after `Eof`, a retry of an input the codec refused.
  `Some` makes it the step's result and skips the read.
- **`on_eof` chooses between finishing and draining.** `Done`, the default,
  forwards `Eof` and finishes. A codec returns `Again`; its `before_take`
  forwards `Eof` itself once the pump is empty, then returns `Some(Done)`.
- **Codec `EAGAIN` is not an error.** `libav::pump::Pump` implements the
  send/receive/EAGAIN protocol for both directions, stashes the refused input,
  queues the outputs, and reports `Progress::Stalled` when nothing moved, which
  is when `self.io.wait(PARK_TIMEOUT_MS)` is the answer rather than a spin.
  Both codec nodes use it; a third should too.
- **Lock inside the hook, not around the step.** Each hook takes what it
  needs and releases it before the loop blocks for the next input. `interrupt`
  runs on another thread and must never take a lock the body may hold.
- **`interrupt` reaches the park through the wrapper.** Override it only when
  the body can also block *inside* a library; `input` does, to fire libav's
  `AVIOInterruptCB`. Keep it non-blocking and idempotent, it runs on the
  control thread.

A `BlockingNode` without an input, like `input`, writes its own `step`, and
`input.rs` is the model for that: `self.io.push_with` for the push, so hints
are answered while parked; `self.io.is_interrupted()` before blocking inside
libav; `self.io.wait(ms)` for an interruptible sleep.

For a node that is a pure per-buffer transform with state rebuilt from `Spec`
(a filter, a rescaler, a bitstream filter), do not write even the hooks:
implement `SisoNode` (`on_spec`, `process`, `on_flush`) and register
`SisoAdapter<Yours>`, which is a `SingleInput` over it.

## The poll body

```rust
fn step(&self, ctx: &mut NodePollContext) -> Result<Tick, NodeError> {
    let input = self.io.input()?;
    let out = self.io.output()?;
    let state = &mut *self.state.lock().unwrap();

    // 1. A buffer held back by a full output goes first.
    if let Some(buffer) = state.pending.take() {
        match out.offer(buffer) {
            Ok(()) | Err((Push::Dropped | Push::Accepted, _)) => {}
            Err((Push::Full, buffer)) => {
                state.pending = Some(buffer);
                ctx.wait_writable(out);
                return Ok(Tick::Idle);
            }
            Err((Push::Closed, _)) => return Ok(Tick::Done),
        }
    }
    // 2. Then one item from the input, without waiting.
    let Some(item) = input.try_take() else {
        if input.is_closed() {
            return Ok(Tick::Done);
        }
        ctx.wait_readable(input);
        return Ok(Tick::Idle);
    };
    // 3. Classify and emit, stashing on `Full` exactly as above.
    Ok(Tick::Again)
}
```

The rules:

- **Never block.** No `take(-1)`, no sleep, no libav call that can wait on I/O.
  The event loop is shared; one blocking node stalls every node on it.
- **`Idle` only after registering a waiter.** `wait_readable`, `wait_writable`,
  `wait_deadline(Instant)` or `wait_tick()`. An `Idle` with nothing registered
  is woken only by an explicit tick of the executor, which on an event loop
  without a `tick_source` means never.
- **`Again` means "call me right back".** The executor honours it up to a
  fairness budget and then yields to the other nodes, so a loop of `Again`s is
  fine; a step that does a lot of work before returning `Again` is not.
- **Stash on `Full`, do not drop.** A buffer already taken from the input has
  left the queue; if the output has no room it goes into `state.pending` and is
  offered first next time. `demux` and `mux` both do this, and it is the poll
  counterpart of `self.io.push` parking.
- **Take the lock inside `step` only.** A Direct edge may call `step` from the
  producer's thread while your own scheduled poll is idle; the mutex is what
  makes that safe, and holding it across a wait would deadlock the producer.

## `Spec` and hints: how nodes learn about each other

C++ nodes walked *up* the graph (`findNodeUp<IStreamsInput>()`) to ask a
producer about its format. Rust nodes never see other nodes. What they see is
the edge, and the edge carries a latched `Spec` (`graph/spec.rs`): the last
`EdgeEvent::Spec` pushed on it, re-delivered to a consumer that binds or
restarts. So:

- **A producer publishes what it produces**, once, before the first buffer,
  and again only when it changes: the decoder publishes `Spec::Video`/`Audio`
  read off the first decoded frame; the encoder publishes `Spec::Packet` with
  the opened context's codec parameters; `input` publishes `Spec::Catalog`;
  `mux` publishes `Spec::Mux`, the whole container description.
- **A consumer configures itself from the spec it receives.** The decoder
  opens on `Spec::Packet`; `output` creates the container from `Spec::Mux`.
- **A re-delivered spec is compared, not obeyed.** `libav::codec::same_spec`
  says whether the new one differs. Equal: nothing to do, and say so at debug
  level. Different: reopen, and say so at info level. The decoder is the model.
- **A buffer that arrives before its spec is dropped and counted**, never
  processed with guessed parameters. The count is logged once, at finish
  (`log_drops`).

Some questions only the producer can answer, and the consumer holds the
question: `demux`'s `streams_filter` is an ffmpeg stream specifier that only the
`AVFormatContext` owner can evaluate. That travels *upstream* as an `EdgeHint`
posted on the edge; the producer drains hints with `take_hints()` and answers
with a new spec. Two rules make this deadlock-free, and `input.rs` shows both:
the producer checks hints on every step before it reads, and it also checks
them while parked on a full output, which is what the closure argument of
`push_with` is for.

## Errors and logging

`NodeError` is a name, a `NodePhase` and a message. `step` returning `Err`
fails the node; the supervisor then applies the group's restart policy. So
`Err` is for situations the node cannot recover from on its own, and that a
restart can plausibly fix or at least should surface:

- an unbound or closed-forever edge, a missing container description, a codec
  that will not open, a timeout on a network source;
- twenty consecutive write failures, not one (`output.rs`); two hundred decode
  errors, not one (`Pump`). Tolerate what libav tolerates, count it, fail past
  a threshold.

Everything else is **drop, count, log once**:

- a packet without timestamps, a frame before the spec, a packet for a stream
  the container does not have: increment a counter in `State`, return `Again`;
- log the counters at finish or in `stop`, with the reason, at info level;
- a condition that would repeat every buffer gets a `warned: bool` next to it
  and one warning (`passthrough_warned` in `encode.rs`).

Every log line starts with the node's name: `log::info!("{}: opened ...",
self.io.name)`. Levels: `error` for what will fail the node, `warn` for what
lost media or changed behaviour, `info` for one-time facts a user of the
script wants (what was opened, at which format, why it finished), `debug` for
per-decision detail. Nothing per buffer above `trace`.

## Reuse: what goes where

The rule in `AGENTS.md` is *no copy-paste between nodes*. In practice the
question is always "where does the shared piece live", and there are four
answers:

1. **The scaffold** (`avplumber_f7k/src/scaffold/`): anything about being a
   node — edges, parking, pushing, name, hooks. If two nodes have the same
   `impl` block modulo their own state, the block belongs here. This is how
   `Blocking`/`Polling` came to exist (five nodes had the same seven methods),
   and then `SingleInput` (four of them had the same five-arm `match` on their
   input).
2. **The libav helpers** (`avplumber_f7k/src/libav/`): anything about talking
   to FFmpeg that is not specific to one node. Before writing one, look at what
   is there:
   - `codec`: `find_decoder`, `find_encoder`, `open_codec` (with option
     leftovers logged), `apply_packet_spec`, `apply_media_spec`,
     `video_spec_of`, `audio_spec_of`, `packet_spec_of`, `same_spec`,
     `parse_pix_fmt`/`parse_sample_fmt` (the `?`-prefixed "preferred, not
     required" syntax), `codec_name`, `pix_fmt_name`.
   - `dict::Options`: a JSON object to an `AVDictionary`, and back, with
     `warn_leftovers` for what libav did not consume. Never build a dictionary
     by hand in a node.
   - `pump::Pump`: send/receive/EAGAIN for encode and decode.
   - `error`: `av_error`, `is_eagain`, `is_eof`, `code_of`.
3. **A private helper on the node**, when only this node needs it. Extract as
   soon as `step` stops fitting on a screen; name helpers after what they do
   in the protocol (`on_spec`, `emit`, `stamp`, `publish_spec`, `serve_hints`,
   `finalize`), and order them as `step` calls them.
4. **A new node**, when the shared behaviour is a *stage*, not a helper. A
   timestamp rewrite that two nodes would both like to do is one node placed
   between them, not a helper both call.

Two things are *not* reuse and are refused in review:

- **Modifying the framework as a workaround** for something a node, an
  interface or a shared object could do properly. Framework changes are made
  deliberately, called out in the change description, and justified by more
  than one caller.
- **A second copy with one line changed.** If the existing helper almost fits,
  change the helper and its callers. `push_with` is the result of exactly
  that: `input` needed the same push loop as the codecs plus a hook.

Shapes that repeat across the current nodes, and which you should recognise
before writing a fourth copy:

| shape | seen in | where it lives now |
|---|---|---|
| which hook an input item calls, forwarding of control events | every single-input node | `react`, through `SingleInput` and the Siso adapters |
| "output first, then codec, then input" step order | decode, encode | `before_take` of both; a shared codec-node layer over `Pump` is the next extraction if a third codec node appears |
| pending buffer stashed on `Full` | demux, mux | `SisoPollAdapter` for the single-output case |
| "input spec re-delivered unchanged" | decode, encode, output | `same_spec` plus the same three-line `if` |
| drop counters logged at finish | every node | `log_drops` per node |
| two type names over one parameter struct | decode, encode | `#[serde(transparent)]` newtypes |

## Cleanliness, in one list

- One node (or one family) per file; the file order in *Anatomy* above.
- Every `pub` item in the nodes crate is `pub` because `lib.rs` or a test needs
  it. The node struct is `pub` for tests; its fields are not.
- Doc comments say what is non-obvious: the C++ name, the reason for a
  deviation, which invariant a field relies on. They do not restate the code.
- No `unwrap` on anything that can fail at runtime. `lock().unwrap()` is fine:
  a poisoned mutex means a panic already happened on this node.
- No `unsafe` in a node unless a libav callback or a raw context field forces
  it, and then a `// Safety:` comment names the invariant (see `choose_pix_fmt`
  and `open_input`).
- Timestamps go through `Ts` and the `FrameExt`/`PacketExt` setters, which
  keep `time_base` in sync. Never write `pts` and forget the base; the whole
  design doc §3 is about why.
- No node-local ratio or option parsing. Parameters are typed by serde; ratios
  and dictionaries have helpers.
- Formatting: `rustfmt` on the files you touch. Do not reformat the tree.

## Testing

Three levels, each with an example to copy:

- **Hand-stepped unit tests** (`mux.rs`, `#[cfg(test)]`): build the node
  through `NodeSpec::build` with an `Instance` and a `BuildCtx`, bind
  `BufferedEdge`s by hand, call `start`, then call the node's `step` directly
  and `ctx.clear_park()` between steps, exactly as an executor would. Every
  decision is observable one step at a time, and no thread or runtime is
  involved. This is the right level for ordering, grace periods and drop
  policies.
- **Body-driven tests** (`tests/demux_packets.rs`): take the node's body with
  `take_body` and loop it on a thread while draining its output here. Use this
  when the node talks to libav and the assertion is about what libav produced.
- **Script-driven integration tests** (`tests/transcode_file.rs`): feed the
  same `node.add` lines a user would write through `control::exec_line`,
  start the group, wait for the outcome. This is the only test that proves the
  parameter names, the envelope keys and the pad binding all agree.

The framework's `testing` feature exposes `graph::media::test_media`, a stub
buffer carrying only a media type and a timestamp, so a unit test of a node
that does not inspect pixels needs no FFmpeg at all. The nodes crate enables it
as a dev-dependency.

Run the tests in every feature configuration the change touches: the default
build (no libav, `Media::Stub`), `--features async` (the poll executor and
Direct-edge tests only exist there) and an `ffmpegN` selector matching the
FFmpeg you link against. Several substrate suites are `cfg(not(feature =
"ffmpeg"))` and only run in the default build.
