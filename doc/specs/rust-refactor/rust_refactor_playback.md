# Playback control in the Rust core: service, seekable source, pacing

Status: agreed design; all four steps of §12 have landed. Written against the tree as of
September 2026 (`avplumber_f7k` and `avplumber_nodes` on `rust-dev`).

This document narrows plan-v2 §3.3, §3.4 and §6 to what the replay demo
(`demos/replay/`) needs, and records the decisions taken while comparing the
C++ demo with the Rust crates. It replaces the C++ `input_rec`, `realtime`,
`speed`, `pause`, the four Teams and `SharedTimeline`'s role in playback with
one service, a seekable mode of the existing `input` node and one pacing node.
Nothing here is a shim over C++; every node named below is native Rust.

## 1. Decisions

| Decision | Consequence |
| --- | --- |
| No C++ compatibility layer. Plan-v2's "Tier S stays C++ behind a shim" does not apply; the tree already went native for input, demux, codecs, mux and output. | `force_fps`, `force_keyframe` and `bsf` are Rust ports; `input_rec` is not ported at all, its container half becomes a mode of `input` and the rest becomes the service. |
| Hardware acceleration is out of scope for now. | Software H.264 decode and `libx264`, which is the CPU backend the demo's tests already support. The `hwaccel` parameters stay rejected at build. |
| No Python inside the Rust process. | The demo's Python position probe becomes a status query. The TUI is either a Python client over the TCP control protocol or a Rust TUI. |
| The pacing node stamps output PTS from release time. | This is the one node that rewrites PTS, the continuity exception plan-v2 reserves for Sentinel. Everything upstream of it carries media time. |
| A seekable source never pushes `Eof`. | A pushed `Eof` closes a Rust edge for buffers until a restart reset. At the tail the source idles or loops, and the service reports `at_end`. The last frame stays on screen and seeking back works. Finite transcode keeps `Eof` unchanged. |
| `FlushStop` carries an optional resume timestamp. | Decoders drop frames before it. Indexed seeks pass `None`; a plain time seek to a keyframe gets frame-exact results without any reach into the decoder. |
| Direct edges fuse the poll chain, buffered edges stay at blocking boundaries. | In-flight media is at most one item per buffered hop plus codec state. That is why one in-band flush suffices and no out-of-band barrier exists. |

What is deliberately **not** ported from C++: the four-phase `flushAndSeek`
barrier, `pauseProcessing`/`lockProcessing` on upstream nodes, `IFlushAndSeek`,
`IPlaybackControl`, `ISpeed::speedChanged` and its PTS un-baking walk,
`discardUntil`, `flush_magic`, `waiting_for_frame`, `hold_at_eof`,
`stop_on_eof`, `send_eof`, `stop_delay`, `timestamp_source` and the
`frame_no`/`frame_ts`/`wallclock` frame metadata, `pass_on_seek`, the
transition-gate pause team, `sync_node`, `team.link`.

## 2. Ownership

Three owners, one name. Nodes bind to a playback group by a single `sync_group`
parameter; the control protocol addresses the same name.

| Owner | Owns | Does not own |
| --- | --- | --- |
| **Playback service** (`services/playback.rs`, one per group name) | the clock (rate, paused, anchor), loading and querying the seek index and history, target resolution, read planning (seek requests, reverse stepping, skip stride, loop), the last released position, scheduled pause/seek, `at_end` | any media, any executor, any node |
| **Source** (`input` in seekable mode) | the container: open, read one packet, seek to a byte offset or timestamp when the service says so, emit the flush pair around a discontinuity | the index, the current position, the meaning of a target, direction, pause |
| **Pacing node** (`realtime`) | when a frame is released, the output PTS, reporting what was released | rate, position arithmetic, seek targets |

Why the input does not own seek: relative targets need the position the output
is showing, not the position the reader has reached; UTC and frame resolution
need the index and history, which are per recording, while the policy is per
group; and the demo README already anticipates several slots composing into one
application, where one seek fans out to every bound source with one shared
clock reset. Once the index lives in the service, reverse stepping, the skip
stride and loop are index lookups too, so they move with it. What is left for
the node is only what needs the `AVFormatContext`, and that is not enough to
justify a second node type beside `input`.

The service is shared `Arc` state under `services/`, like `ClockService`,
`CorrectionService` and `TimelineService`. Service membership is not execution
placement (see `ARCHITECTURE.md`, "three coordinate systems").

## 3. The in-band discontinuity

The only seek mechanism is a pair of edge events travelling with the data:

```
FlushStart                 // edge drops queued Buffers on push; nodes drop what they hold
  source repositions
FlushStop { resume_at }    // nodes resume; decoders drop frames with pts < resume_at
Spec (if changed), Buffers at the new position
```

Already in the tree: `EdgeEvent::FlushStart` drops queued buffers in
`push_event`; `SingleInput` calls `on_flush` and forwards both events;
`dec_*`/`enc_*` flush their codec in `on_flush`; `demux` and `mux` forward.

Landed (step 1 of §12):

- `EdgeEvent::FlushStop` is `FlushStop { resume_at: Option<Ts> }`. The C ABI
  projection carries it as `resume_at` plus `resume_at_tb`, `AVP_NOPTS` for
  none.
- `InputHandler` and `SisoNode` have an `on_flush_stop(resume_at)` hook; the
  scaffold calls it and then forwards the event, payload intact. `dec_*`
  stores the cutoff and drops decoded frames below it; the first frame at or
  past it clears the cutoff, a frame without a timestamp passes, and a new
  `FlushStart` disarms it.

Also landed: every node between source and pacing forwards both events;
`force_keyframe` resets its periodic phase on `FlushStart` and `force_fps` its
grid. Two rules the implementation needed that the design had not spelled out:
a node holding a buffer for a full output waits for a *flush at the head* of
its input (`NodePollContext::wait_flush`, `BlockingIo::push_from`), not for the
input being readable, so a flush passes a pipeline backed up behind a paused
output without any node spinning; and the pacing node releases nothing while
the service reports a seek in flight (`Playback::seek_in_flight`), since a
pause or resume can land between the seek and its flush.

Ordering is intrinsic to the pipe, so there is no epoch tagging.

## 4. The playback service

```rust
pub struct Playback {
    name: String,
    clock: Arc<dyn SyncGroup>,               // ClockService entry of the same name
    sources: Mutex<Vec<BoundSource>>,        // wake handle + mailbox per bound input
    index: Mutex<Option<Arc<SeekIndex>>>,    // loaded from the path the source passes at build
    loop_: AtomicBool,                       // from the source's `loop` parameter
    released: AtomicI64,                     // media time of the last released frame, ms
    serial: AtomicU64,                       // bumped on every release
    pending: Mutex<Option<ResolvedTarget>>,  // a seek requested but not yet observed
    at_end: AtomicBool,
    scheduled: Mutex<Scheduled>,             // pause-at and seek-at lists
}

struct BoundSource {
    wake: Arc<Park>,                          // the node's own park, so a command reaches a parked reader
    mailbox: Mutex<Option<ResolvedTarget>>,
}

/// What the source does before its next read. Asked on the source's thread.
pub enum ReadPlan {
    Continue,                                         // forward, rate <= 1: just read
    Reposition { to: SeekTo, discontinuity: bool },   // discontinuity → wrap in FlushStart/FlushStop
}

pub enum SeekTo {
    Bytes(u64),                               // indexed
    Time { media: Ts, resume_at: Option<Ts> }, // non-indexed
}
```

`BuildCtx` gains `playback(name) -> Arc<Playback>` beside `clock`,
`correction` and `timeline`. The service creates or reuses the `ClockService`
entry of the same name, so a node that only wants the clock still sees the
same one. `Playback::bind_source(park, index_path, history_path, loop)` is
called by `input` at build; it loads the index (§5) and registers the wake.

Operations, each one command:

| Operation | What the service does |
| --- | --- |
| `seek(target)` | resolve to an index entry (§5); `clock.reset(media_ts)`; store `pending`; put it in every bound mailbox and wake |
| `set_rate(r)` | `clock.set_rate(r)`; if the sign changed, `seek(Relative::Frames(0))`; `r == 0` is `pause()` |
| `pause()` / `resume()` | `clock.set_paused(true/false)` |
| `pause_at(target)` / `seek_at(when, target)` / `clear` | stored; checked against every reported release |
| `report_release(media_ms)` | store `released`, bump `serial`, clear `pending` if reached, fire scheduled actions |
| `status()` | JSON, see §8 |

Asked by the source, on its own thread:

| Call | What the service answers |
| --- | --- |
| `plan_read(source, current_bytes)` | a mailbox entry as a discontinuity; else, from the clock snapshot: `rate < 0` gives the previous index entry as a plain byte seek, `rate > 1` on an all-intra index gives the entry `round(rate)` ahead, otherwise `Continue` |
| `plan_tail(source, current_bytes)` | with `loop`, a discontinuity to the first entry (or the last when reversing); otherwise sets `at_end` and answers `Idle` |

Relative targets resolve against `pending` if a seek is outstanding, otherwise
against `released`. That is what keeps the demo's rapid paused seeks correct.

Multiple sources per group are supported by construction: one clock reset, one
fan-out. The demo binds one.

## 5. SeekIndex

Loaded by the service, in `services/playback.rs`, from the paths the source
passes at build (`<url>+seek` and `<url>+history` unless overridden). The node
never parses either file. Formats are unchanged:

- seek table: native-endian `(i64 timestamp_ms, u64 byte_offset)` records,
  timestamps and offsets non-decreasing;
- history: native-endian `(i64 changed_at, i64 input_offset, i64
  wallclock_offset, i64 output_offset)` records, first `changed_at == 0`.

API:

```rust
impl SeekIndex {
    fn len(&self) -> usize;
    fn entry(&self, frame: usize) -> (i64 /*ms*/, u64 /*bytes*/);
    fn nearest(&self, media_ms: i64) -> usize;          // ties select the later frame
    fn frame_of(&self, media_ms: i64) -> usize;         // last entry with ts <= media_ms
    fn prev(&self, bytes: u64) -> Option<usize>;        // for reverse stepping
    fn media_to_wallclock(&self, ms: i64) -> i64;       // history lookup by changed_at
    fn wallclock_to_media(&self, ms: i64) -> i64;
    fn fps(&self) -> Option<u32>;                       // the demo's inference, for status
    fn reload(&self) -> io::Result<usize>;              // growing recordings, later
}
```

Target grammar is the C++ one so the demo's controller keeps working:
`12000`, `+500`, `-500` (ms, signed is relative), `01:02:03`, `01:00.150`,
`+0:10`, `2026-08-10T12:00:00.000` (wallclock), `frame 120`, `frame +5`,
`frame -30`. A wallclock target is converted to media time through the history
before lookup. `end` and `live` resolve to the last entry, `live` minus a
configurable delay.

Edge PTS stays media time from the container. The three C++ timelines
(media, sync, output) collapse to one on the edge plus a history lookup in the
service for UTC in and out.

## 6. The source: `input` in seekable mode

There is no `input_rec` in the Rust tree. `input` gains an optional mode,
switched on by `sync_group`; without it the node behaves exactly as today and
the transcode graph keeps using it unchanged. Live inputs (UDP, RTP, SRT,
devices) and plain VOD reads never set `sync_group`: they keep the interrupt
timeout, the EAGAIN-is-not-an-error rule, `Eof` plus `stop_delay` at stream
end and the group restart policy for reconnects. Such an input may still feed
a `realtime` node, because the pacing node binds to the clock by group name
and the service serves that clock whether or not a source is bound; pause and
rate work, only seek commands on such a group fail, with "no seekable source".

The three tiers of `input`, each a superset of the previous:

| Tier | Parameters | Gives |
| --- | --- | --- |
| finite | none of the below | read to the end, `Eof`; transcode, plain VOD, live |
| seekable, no index | `sync_group`, `seek_table: ""` | timestamp seeks on any container through the demuxer's time seek plus `resume_at`, pause, rate; no reverse, no frame targets |
| indexed | `sync_group` and a seek table | byte-exact seeks, frame targets, reverse, loop, UTC through the history |

New parameters, all only meaningful with `sync_group`: `seek_table` and
`history` (default `<url>+seek` and `<url>+history`; an empty string means no
index, so only timestamp seeks work), `loop` (bool, default false),
`live_delay` (seconds, for `live` targets). `eof_mode` and `stop_delay` are
rejected at build together with `sync_group`, because a seekable source never
announces an end.

What changes in the body, one blocking step:

1. Before the read, `playback.plan_read(self, avio position)`. On
   `Reposition { discontinuity: true }`: push `FlushStart`, seek, push
   `FlushStop { resume_at }`. On `Reposition { discontinuity: false }`: just
   seek (a reverse step or a skip stride). Byte targets use `AVSEEK_FLAG_BYTE`.
2. Read one packet, push it. Backpressure parks the thread on the node's
   `Park`, which is also the wake handle the service holds; a command arriving
   while the reader is parked wakes it, and flush events never take a buffer
   slot, so `FlushStart` can be pushed into a full edge and the drop makes room.
3. At the container tail, `playback.plan_tail(self, avio position)`. A
   reposition is handled as in step 1; `Idle` waits on the park with a short
   timeout and returns `Again`. `Eof` is never pushed in this mode.

That is about fifty lines of glue in `nodes/input.rs`; every decision behind
them is a service call. The catalog, `EdgeHint` handling, timeout and interrupt
paths are shared with the finite mode because they are the same code.

Removed relative to C++ `input_rec`: `preseek`, `ts_offsets`, `team`,
`pause_team`, `speed_team`, `timestamp_source`, `stop_on_eof`, `send_eof`,
`start_ts`/`stop_ts`, `stream-limits` objects, the seek-at table, the
background index reload thread (a `reload` method on the service remains for a
later live mode), and all the seek resolution and direction logic, which is
now §4 and §5. The C++ file is 1,214 lines.

## 7. The pacing node: `realtime`

Type name kept; file `nodes/realtime.rs`; a `PollNode` that is infallible, so
it may sit at the end of a Direct chain and its consumer may be Direct too.

Parameters: `sync_group` (required), `tick_period` (rational, optional: output
PTS snaps to this grid and the node also wakes on it), `timebase` (of the
output PTS, default `1/1000` or the tick period), and the three live-stream
thresholds carried over from C++ `realtime`, all in seconds:
`negative_time_discard` (a frame mapping behind now by less than this is
released late, beyond it dropped; default two frame periods),
`negative_time_tolerance` (behind now by more than this means the anchor is
wrong, re-anchor instead of dropping; default 0.25), and
`discontinuity_threshold` (ahead of now by more than this is a PTS jump, not a
frame to wait for; re-anchor; default 1). The replay graph sets the first to
one frame period, as the demo does today, and never trips the other two
because its discontinuities are explicit flushes.

Step:

1. Peek the input. `FlushStart`: drop the held frame, set `release_one`,
   forward. `FlushStop`: forward. `Spec`: forward.
2. For a buffer with media PTS `p`: `wall = clock.map_to_wall(p)`.
   - paused and not `release_one`: `Idle`, wake on readable and on the clock's
     epoch change;
   - `wall - now > discontinuity_threshold`: re-anchor (step 3) and release;
   - `wall > now`: `Idle` with deadline `wall` (the Poll executor installs a
     timer);
   - `now - wall > negative_time_tolerance`: re-anchor and release;
   - `now - wall > negative_time_discard`: pop and drop;
   - otherwise release: stamp PTS with the release time in `timebase` (snapped
     to the tick grid when set), push, pop, clear `release_one`,
     `playback.report_release(media_ms(p))`.
3. Anchoring: on the first buffer after start, and on either re-anchor above,
   `clock.join_offset` with this frame's offset, which elects the smallest
   offset across the group's members so every stream stays jointly buffered,
   exactly the rule `RealTimeTeam::updateOffset` implemented. After a seek the
   service has already reset the clock, so the first frame maps to now and no
   re-anchor fires. Re-anchoring is the pacing node's job in both modes; the
   service's reset and the node's join use the same clock API.

The clock's epoch counter makes pause, resume, rate and reset observable
without a callback: the node compares the snapshot epoch on each wake and a
wake is registered on the service for epoch changes.

`speed`, `pause` and `realtime` from C++ are this one node plus the service.
`force_fps` is unnecessary in the player, because release times already sit on
the tick grid; it stays a node for the transcode graph.

## 8. Control surface

Commands, in `control/mod.rs`, all addressed by group name:

| Command | Effect |
| --- | --- |
| `seek <group> now <target>` | absolute or relative time, ms or clock or ISO wallclock |
| `seek <group> frame <N\|+N\|-N>` | absolute or relative frame |
| `seek <group> at <when> <target>`, `seek <group> clear` | scheduled seeks |
| `seek <group> live`, `seek <group> end` | index-relative targets |
| `pause <group> now`, `pause <group> at <target>`, `resume <group>` | clock freeze |
| `speed.set <group> <rate>`, `speed.get <group>` | clock rate, negative is reverse, zero is pause |
| `playback.status <group>` | JSON: `position_ms`, `frame`, `wallclock_ms`, `duration_ms`, `frame_count`, `fps`, `rate`, `paused`, `direction`, `at_end`, `serial`, `pending` |
| `node.object.set <node> <key> <json>`, `node.object.get <node> <key>` | typed hook on `Node` with a default that errors; `force_keyframe` implements `trigger` and `status` |
| `queue.plan_capacity * <n>` | wildcard, which the demo uses for every edge |

The demo's controller sends its three team names to `pause`, `speed.set` and
`seek`; with one service those three constants become one name and the
transition team disappears. `observation_marker` and `observed_frames_since`
map to `serial` plus polling; a push subscription can be added later on the
pattern of `stats.subscribe`.

Transport: `avp_core_serve_tcp` currently returns -1. It becomes a
newline-delimited loop over `exec_line` on a Tokio listener, one reply line per
command, the same wire format as the C++ server. A `bin` target `avplumber`
loads a script file and serves a port. That is the whole embedding story for
this demo; there is no in-process Python.

## 9. Other nodes to port

| Node | Kind | Notes |
| --- | --- | --- |
| `force_fps` | Poll | C++ 229 lines. Transcode graph only. Duplicate or drop to a fixed grid, reset on `FlushStart`. |
| `force_keyframe` | Poll, infallible | C++ 158 lines. `interval_sec` periodic phase plus a coalescing trigger through `node.object.set`. Sets `pict_type` I on the frame. |
| `bsf` | Blocking `SingleInput` | C++ 167 lines. `AVBSFContext` through rsmpeg's `bitstream` module. Consumes the packet `Spec`, publishes `par_out` as a new packet `Spec`, rescales through `time_base_out`. |
| `output` seek table | extend | `seek_table` and `seek_table_text` parameters. Binary and text records for stream 0, `avio_seek(pb, 0, SEEK_CUR)` for the offset, flushed per record. One growing file each, not C++'s four rotating backing files behind a symlink: the Python client's publish step changes accordingly. |
| `dec_*` | extend | `on_flush_stop(resume_at)` cutoff. Nothing else changes; `flush_magic` and friends stay undeclared. |
| `demux` | unchanged | It already forwards flush events; with no `Eof` from the source it never finishes early. |

## 10. Sequences

Seek while paused: command; service resolves to entry `k`, resets the clock to
`ts_k`, sets `pending`, requests. Source: `FlushStart`, byte-seek to `k`,
`FlushStop`, frame `k`, frame `k+1` until the capacity-one edges fill. Pacing:
drops its held frame, releases frame `k` once because of `release_one`, holds
`k+1`. Status shows frame `k`, serial bumped once, stable thereafter.

Speed 1 to 2: `clock.set_rate(2)`, then, because the read stride changed, the
same internal seek to the frame on screen a direction change uses: the frames
in flight were read at stride one and would show a burst of the old motion. One
repeated frame, then `plan_read` answers with the index entry two ahead as a
plain byte seek. A change that keeps the stride (1 to 0.5) touches nothing in
flight. No gate, no drain, no forced keyframe.

Speed 1 to -1: `set_rate(-1)` then an internal seek to the current frame, so the
pipe flushes once; from then on `plan_read` answers with the previous entry
before every read.

Loop: the source hits the tail and asks `plan_tail`, which only notes it and
answers `Idle`: the viewer still has the pipeline's worth of frames to see, and
looping on the reader's say-so would flush them (which is how the first
implementation showed frame 0 after `seek end`). When the pacing node reports
the last frame released, the service resets the clock and puts a discontinuity
to the first entry in the mailbox; a viewer paused on the last frame loops on
`resume`. One clock reset, one flush, playback continues from frame 0 with a
monotonic output PTS because the pacing node stamps release time.

End without loop: `at_end` is set when the last frame is released with the
reader idle at the tail; the source waits on its park. Nothing new reaches the
pacing node, so the last frame stays on the output. A later seek clears
`at_end`, wakes the park, and everything runs as above.

## 11. Testing

- Rust unit tests: `SeekIndex` nearest and tie rules, history conversions,
  fps inference; `realtime` driven with `SyntheticClock` (release, hold,
  late drop, `release_one`, output PTS grid); `Playback` resolution of every
  target form, pending versus released, sign-flip seek.
- Rust integration tests in `avplumber_nodes/tests/`: `input` in seekable mode
  on a fixture produced by the transcode graph, exercising absolute, relative,
  wallclock and frame seeks, reverse, loop and end, through
  `control::exec_line` and `playback.status`. The finite mode keeps its
  existing tests untouched, which is the check that the mode switch changed
  nothing there.
- The demo's old native suites, ported: the seek/pause/speed scenario sweeps
  of `test_playback_integration.py` are `avplumber_nodes/tests/playback_scenarios.rs`
  (24/25/30/60 fps boundaries at every speed, endpoints and seeded targets,
  frame steps, clamping at both edges, a buffering decoder, playing seeks,
  speed-change continuity forward and reverse, pause and seek after the end,
  seeks while the tail drains, prompt stopping); the transcode packet count
  and the RTP header checks of `test_transcode_integration.py`, and the
  shutdown deadlines of `test_demux_shutdown_integration.py`, are in the
  demo's `tests/test_rust_integration.py` on the Python client, which
  replaces `pyplumber` with a socket, `PositionProbe` with `playback.status`
  polling, and `isWorking` with `group.status`. The paused-picture oracle's
  self-test stayed as `tests/test_playback_assertions.py`.
- Hardware decode and encode: `avplumber_nodes/tests/hwaccel_nvidia.rs` runs
  the demo's NVIDIA graph and checks the surfaces between the codecs, then
  repeats the seek, play, speed and reverse assertions with `h264_cuvid` as
  the decoder, comparing pixels against the software decode of the same
  recording. Skipped, loudly, without a usable device. The demo's Python
  suite runs its end-to-end tests on both backends.
- The live encoder across discontinuities: `avplumber_nodes/tests/encode_after_seek.rs`
  (packets keep coming after a paused seek, reversal and rapid seeks, in both
  `flush` modes) and the RTP-across-seeks test of the Python suite. Regression
  for the encoder that answered `FlushStart` with `avcodec_flush_buffers`.

## 12. Order of work

1. `FlushStop { resume_at }`, `on_flush_stop`, decoder cutoff. **Landed.**
2. `force_fps`, `force_keyframe` with `node.object.*`, seek-table writer on
   `output`, wildcard `plan_capacity`, TCP server and the `avplumber` binary.
   Gate: transcode on CPU. **Landed**: `avplumber_nodes/tests/replay_recording.rs`,
   and the binary runs the same script as a batch job.
3. `SeekIndex`, `Playback` service, the seekable mode of `input`, `realtime`,
   playback commands, `playback.status`. Gate: the playback suite on CPU.
   **Landed**: `avplumber_nodes/tests/playback.rs` drives the player graph
   through the control protocol and identifies every released frame by its
   pixels against the frames the `ffmpeg` CLI decodes from the same recording.
4. `bsf`, RTP output check, the Python client, the TUI decision. **Landed**:
   `avplumber_nodes/tests/rtp_output.rs` sends the Janus leg to a UDP socket
   and checks headers, sequence and the repeated parameter sets; the demo's
   Python (`demos/replay/`) keeps its Textual TUI and controller and became a
   client of the Rust executable over the control protocol
   (`control_client.py`), with `test_rust_integration.py` running the same
   RUN V2 exercise end to end. The C++ position probe is a `playback.status`
   poll, and the RTCP listener stayed in Python.

## 13. Deferred

- ~~Hardware acceleration~~ **landed**: `services/hwaccel.rs` holds the named
  devices, `hwaccel.init` opens them, and `dec_video`/`enc_video` take
  `hwaccel` (plus `hwaccel_only_for_codecs` on the decoder). The demo's NVIDIA
  backend is the C++ one: `h264_cuvid` into `cuda` surfaces, `h264_nvenc` out
  of them, nothing in between that could copy a frame to host memory.
  `avplumber_nodes/tests/hwaccel_nvidia.rs` asserts that every frame between
  the two is a pooled CUDA surface.

  The check this section asked for found a real gap. A byte-exact seek *is*
  frame exact on `h264_cuvid` — `resume_at` was never needed for precision —
  but NVDEC holds its last frame until another packet arrives, so a paused
  seek to the final frame of a recording surfaced nothing: there is no next
  packet. That is the other half of what `flush_magic` did, and it is now
  [`EdgeEvent::Drain`](../../../avplumber_f7k/src/graph/edge.rs): the seekable
  source sends it once each time it idles with nothing left to read, the
  decoder gives up what libavcodec holds and carries on. Every target except
  the very last frame worked without it, which is exactly why it needed a test.
- Live, still-growing recordings: `SeekIndex::reload` on a timer, `live` with
  a delay.
- Audio and several sources per group: the service already fans out; the clock
  already elects a shared offset.
- A push subscription for releases instead of polling `playback.status`.
