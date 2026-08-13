# Replay VOD Transcoder and Single-Slot Player

## Goal

Add two PyPlumber applications under `demos/replay/`:

1. transcode one VOD input into the seekable replay MPEG-TS format used by the
   replay recorder; and
2. play one such recording through one controllable replay slot and publish
   video directly to a Janus streaming mountpoint without OBS or Stream Studio
   Gateway at runtime.

The player must expose every current Stream Studio Gateway v2 single-source
playback control through an icon-first Textual interface and a headless
regression mode. Its internal module must leave a clean composition seam for a
future N-input, A/B replay switcher without implementing A/B behavior now.

## Scope

The first version is deliberately limited to:

- one VOD input to the transcoder;
- one video stream in the replay MPEG-TS output;
- one replay recording loaded by the player;
- one playback slot;
- one H.264 RTP video output to Janus;
- NVIDIA hardware decoding and `h264_nvenc` encoding without CPU frame
  transfers; and
- Stream Studio Gateway v2 playback controls only.

The first version does not include audio, live recording, multiple sources,
A/B sections, source selection, clips, bins, playlists, transitions, or OBS.
Those are separate product behaviors rather than prerequisites for proving the
recording format and playback controls.

## Production Contracts Being Reproduced

The replay recorder in `tra-swarm` writes MPEG-TS with video as mux stream zero
and configures H.264 as all-intra, baseline profile, and without B-frames. The
AVPlumber `output` node records a packed 16-byte seek-table entry before each
stream-zero packet:

```text
int64 timestamp_ms
uint64 byte_offset
```

The recorder's `sentinel_video` also tracks the relationship between corrected
media timestamps and the system UTC clock. It writes packed 32-byte history
records to `output.ts+history`:

```text
int64 changed_at
int64 input_timestamp_offset
int64 wallclock_offset
int64 output_timestamp_offset
```

`input_rec` uses this history independently of the seek table. The history
converts a shared wallclock or synchronization target into a recording-local
media timestamp; the seek table then converts that timestamp or frame into an
MPEG-TS byte offset. Both sidecars are therefore part of a complete replay
artifact.

The current Stream Studio Gateway v2 replay player reduces its playback
operations to these AVPlumber primitives:

- `pause` and `resume` on a pause team;
- absolute and relative `seek ... now` on a realtime team;
- absolute and relative `seek ... frame` on a realtime team; and
- signed `speed.set` on a speed team.

Reverse play, scrubbing, go-to-live, and play toggling are stateful compositions
of those primitives. The standalone player will preserve those semantics
without depending on Gateway state or the OBS MSE source wrapper.

## Directory Shape

```text
demos/replay/
|-- README.md
|-- requirements.txt
|-- replay.py
|-- transcode.py
|-- player.py
`-- tests/
    |-- test_controller.py
    |-- test_graph.py
    |-- test_seek_table.py
    `-- test_tui.py
```

`transcode.py` and `player.py` are the two executable PyPlumber applications.
`replay.py` is an internal module, not a third application. It owns graph
construction, control translation, playback state, probing, and lifecycle
logic shared by the executable applications and tests.

## Module Interface

The replay module is a deep module: callers provide configuration and playback
intent, while node topology, edge names, team names, command strings, state
restoration, and shutdown ordering remain implementation details.

The playback control interface is intentionally small:

```python
controller.execute(operation, value=None) -> PlaybackStatus
controller.status() -> PlaybackStatus
```

`PlaybackStatus` reports at least:

- playing and paused state;
- forward or reverse direction;
- configured playback speed percent;
- active scrubbing speed percent;
- current media timestamp;
- mapped UTC timestamp;
- current frame number;
- media duration;
- source readiness; and
- the latest command or graph error.

The player graph builder returns a slot object that exposes only its output
video edge and controller. A future orchestrator can create several slot
instances, route their output edges into A/B selectors, and dispatch one
operation to one or both section controllers. Version one creates exactly one
slot and connects its edge directly to the Janus output graph.

## Transcode Graph

The transcode application builds this graph:

```text
input
  -> demux first video stream
  -> dec_video(CUDA)
  -> force_fps
  -> force_keyframe
  -> enc_video(h264_nvenc)
  -> mux(video is stream zero)
  -> output(mpegts plus binary and text seek tables)
```

The encoder contract is:

- codec `h264_nvenc` with CUDA frames supplied directly;
- baseline profile;
- GOP size one;
- no B-frames; and
- NVENC VBR constant-quality mode with `cq=17` and no bitrate cap.

The current recorder expresses its quality intent as `crf=17` for both
`libx264` and `h264_nvenc`. Standard FFmpeg NVENC exposes `cq`, not `crf`, and
AVPlumber only logs encoder options left unconsumed by FFmpeg. The standalone
transcoder therefore uses the supported NVENC equivalent and requires a
remote encoder smoke test to prove that no quality option is silently ignored.

The force-keyframe node reinforces the all-intra contract. The requested frame
rate is a required transcoder argument; there is no guessed or default rate.
The application runs without realtime pacing so offline conversion can be
faster than playback speed. Source dimensions are preserved. CUDA format
normalization may convert to NV12 when required by baseline NVENC, but it does
not resize or download frames.

Both applications initialize a CUDA hardware context and fail at startup when
the required NVIDIA decoder, CUDA-frame path, or NVENC encoder is unavailable.
There is no software fallback because the deployment and acceptance hosts are
NVIDIA Fedora systems.

The output basename produces:

```text
output.ts
output.ts+seek
output.ts+txt
output.ts+history
```

The transcoder synthesizes one constant history mapping for its continuous,
normalized output. `--wallclock-start` accepts either a timezone-qualified
ISO-8601 value or `now`, and defaults to `now`. A timezone-less explicit value
is rejected. `now` is captured once in UTC and assigned to replay frame zero;
it does not advance according to offline transcoding time. Future batch/N-input
conversion must capture one `now` value for the whole batch rather than once
per input.

Let `P0` be the first timestamp in the completed seek table and `W0` the
selected wallclock origin, both in milliseconds. The single packed history
record is `(changed_at=0, input_offset=0, wallclock_offset=P0-W0,
output_offset=0)`. This makes `input_rec` map any stored timestamp `P` to
`W0 + (P-P0)` without assuming that an MPEG-TS mux starts at zero.

The output node retains its rotating seek-table backing files because that is
the live-recorder contract. The canonical `+seek` and `+txt` paths must point
to complete data after finite output closes.

The application refuses to overwrite any member of an output family unless
`--force` is passed. It writes to a sibling staging location, waits for the
output node to finish, shuts down the graph, and validates the MPEG-TS and all
sidecars. Publication uses same-filesystem atomic replacements, with the
MPEG-TS path published last as the completed-artifact marker. A failed run
must not expose a new MPEG-TS path that points at missing or invalid sidecars.

## Seek-Table Finalization

The current `output` node publishes a canonical seek table only after each
batch of 60 stream-zero packets. On finite output, the final 1 through 59
entries remain unpublished; a clip shorter than 60 frames may have no
canonical seek table at all.

Fix `src/nodes/output.cpp` so graceful flush:

1. writes the pending entries to the current binary and text backing files;
2. flushes the selected backing files;
3. publishes the canonical binary and text paths; and
4. is safe when there are no pending entries or flush is requested more than
   once.

The rotation and finalization paths should share a helper so the publication
rules are defined once. This is an intentional output-node behavior correction
for every finite recording, not a replay-demo workaround. It must not change
the packed seek-table format or the live rotation interval.

## Player Graph

The player application builds this graph:

```text
input_rec(output.ts with output.ts+seek and output.ts+history)
  -> demux video
  -> dec_video(CUDA)
  -> speed_video
  -> pause(speed-transition gate)
  -> force_fps
  -> pause(playback)
  -> realtime<VideoFrame>
  -> position probe
  -> force_keyframe
  -> enc_video(h264_nvenc, 4 Mbit/s CBR, ultra-low latency)
  -> bsf(dump_extra=freq=keyframe)
  -> mux
  -> output(rtp)
```

`input_rec` auto-loads both adjacent sidecars, uses zero preseek and wallclock
as its synchronization timestamp source, and enables looping by default.
`--no-loop` disables looping, and the TUI makes the active loop mode visible.
The player infers nominal FPS from the replay artifact, cross-checks it against
the seek-table cadence, and fails when it is absent, invalid, or inconsistent.
The playback graph uses separate playback-pause and speed-transition gates,
plus speed and realtime teams. Their names are namespaced internally so future
slot and section composition does not require changing the controller
interface.

The video is decoded and re-encoded for Janus. Re-encoding is required because
pause, reverse, seek, and speed changes create a new continuous output
timeline. Passing the recorded H.264 packets through directly would not
produce correct realtime RTP timing after those operations.

For control responsiveness, every player and Janus-output edge has capacity
one. Testing a seven-packet decoder-input exception did not improve the 2x to
1x transition, so the final graph keeps the smaller capacity throughout. The
player decodes every source frame; `speed_video` rescales timestamps and
`force_fps` performs fast-playback selection. Active decreases from above 1x
to 1x or below use the pre-`force_fps` gate defined in the speed-decrease
transition design. Direction changes retain the existing flush-and-seek
behavior.

At 25 fps, a same-direction speed change must reach the requested cadence at
the post-realtime position probe within two output ticks after the command is
acknowledged. This measurement excludes NVENC, Janus, WebRTC, and display
latency. A graph-construction test pins the capacity-one default so later
changes cannot silently restore deep buffering.

The pass-through position probe records the seek-table frame metadata and
mapped wallclock metadata that actually reach the output branch. It forwards
CUDA frames without reading or copying their pixels. The TUI and regression
runner use this observed position rather than estimating it from elapsed
wallclock time. User-facing elapsed time is zero-based at the first seek-table
entry; raw container timestamps remain internal.

The default Janus contract matches the existing demos:

- RTP port 5004 and RTCP port 5005;
- dynamic payload type 96;
- configurable host and SSRC;
- `h264_nvenc` H.264 baseline output at fixed 4 Mbit/s CBR;
- a one-second GOP, no B-frames, and ultra-low-latency settings matching the
  mixer demo;
- SPS/PPS repeated at keyframes; and
- RTCP PLI/FIR requests trigger a forced keyframe.

Janus is an already configured, video-only Streaming plugin mountpoint. The
player sends RTP/RTCP and reuses `pyplumber.rtcp_feedback`; it does not create,
delete, or administer the mountpoint. The 4 Mbit/s Janus rate is fixed in
version one and is not a TUI control.

## Playback Operations

The controller implements all current v2 single-source playback operations:

| Operation | Semantics |
| --- | --- |
| Play | Resume in the retained direction when configured speed is nonzero |
| Pause | Pause immediately while retaining configured speed |
| Reverse play | Resume with the negative configured speed magnitude |
| Media timestamp set | Absolute millisecond seek |
| Seek frames | Relative signed frame seek |
| Seek seconds | Relative signed millisecond seek |
| Playback speed get/set | Read or set 0 through 400 percent |
| Scrubbing get/set | Read or apply temporary -500 through 500 percent speed |
| Stop scrubbing | Restore the pre-scrub play/pause, direction, and speed state |
| Go to live | Seek to `max(start, duration - 3000 ms)` |
| Current play toggle | Select play or pause from current state |

A speed of zero leaves playback paused and reports that a nonzero speed must be
selected; Play does not claim a playing state while speed remains zero. This
corrects an inconsistent Gateway state transition while retaining its useful
behavior. Setting a nonzero speed respects the current forward/reverse mode.
Scrubbing retains the Gateway's 20 percent dead zone and 100 ms command
throttle, does not overwrite configured playback speed, and restores the
saved direction instead of always returning to forward playback. Frame and
second nudges preserve the current play/pause state.

The fixed frame and time nudge controls use these signed values:

```text
-30, -5, -1, +1, +5, +30 frames
-30, -5, -1, +1, +5, +30 seconds
```

For finite VOD, the Gateway operation named "go to live" is presented as
`TAIL -3s`; calling it "live" would misrepresent the source.

## Textual Interface

`player.py` runs the PyPlumber graph and Textual interface in one process. It
also supports `--no-tui --exercise-v2` for deterministic regression runs.

The TUI is an icon-first transport console rather than a command list. It
contains:

- recording and Janus status;
- an observed timestamp/frame timeline;
- the mapped UTC timestamp and a `GO TO UTC` input;
- play, pause, reverse, scrub, and tail transport controls;
- symmetric frame and second nudge rows;
- speed presets and current speed;
- absolute timestamp entry;
- a v2 regression trigger and result summary; and
- a footer with keyboard shortcuts.

Icons retain compact text labels and tooltips because icon-only controls are
ambiguous across terminal fonts. State colors are green for forward play,
amber for pause, violet for reverse, blue while seeking, and red for failures.

Suggested layout:

```text
+- AVPLUMBER REPLAY --------------------------------------------------+
| JANUS LIVE    recording.ts                    current / duration   |
| timestamp     ----------*-----------------    frame / speed / dir  |
+- TRANSPORT ---------------------------------------------------------+
| SCRUB BACK   REVERSE   PAUSE   PLAY   SCRUB FWD   TAIL -3s         |
+- FRAME NUDGE -------------------------------------------------------+
| -30f   -5f   -1f                         +1f   +5f   +30f           |
+- TIME NUDGE --------------------------------------------------------+
| -30s   -5s   -1s                         +1s   +5s   +30s           |
+- SPEED / ABSOLUTE SEEK --------------------------------------------+
| 0.25x  0.5x  1x  2x           elapsed input         GO            |
| UTC timestamp input                                  GO TO UTC     |
+- REGRESSION --------------------------------------------------------+
| Exercise v2 controls                         pass/fail summary      |
+--------------------------------------------------------------------+
```

## Regression Strategy

Pure Python tests exercise the replay module's interface rather than its
internal node names:

- graph construction and production-format parameters;
- every playback operation and value conversion;
- playback state transitions and scrub restoration;
- speed and scrubbing range validation;
- TUI buttons and keyboard bindings; and
- binary seek-table and timestamp-history parsing and validation.

Finite-output integration tests cover both fewer than 60 frames and a length
that ends within a later 60-entry batch. They require:

- a canonical binary and text sidecar;
- one complete binary record per video packet;
- monotonic timestamps and byte offsets; and
- all encoded video frames reported as keyframes.

The headless playback exercise observes graph output and checks:

1. play advances;
2. pause remains stationary;
3. absolute seeking reaches the target;
4. all configured frame nudges move in the requested direction and amount;
5. all configured second nudges work when the recording duration permits;
6. slow and fast playback approximate the requested rate;
7. reverse playback decreases timestamps;
8. forward and reverse scrubbing work and restore prior state;
9. tail seek reaches approximately duration minus three seconds; and
10. an absolute UTC seek reaches the frame predicted by `+history`; and
11. rapid seeks while paused followed by shutdown finish before a timeout.

The final check directly targets regressions caused by decoder input blocking
or failure to wake during seek and shutdown.

Manual acceptance runs the same exercise while viewing the Janus mountpoint.
It checks for freezes, stale frames, excessive seek delay, missing keyframes,
and discontinuous output.

## Error Handling and Shutdown

Before playback, the application validates that:

- the MPEG-TS and binary seek table exist and are readable;
- the binary file size is a multiple of 16 bytes;
- the seek table is nonempty with monotonic timestamps and offsets; and
- the 32-byte history file contains a valid frame-zero wallclock mapping;
- the recording exposes a video stream.

Graph exceptions are captured by the PyPlumber exception callback. The TUI
shows the latest failure in a red status area; headless mode exits nonzero.
Commands validate their arguments before mutating state.

Shutdown stops the RTCP feedback listener, stops the AVPlumber graph, waits for
node threads, and then exits Textual. Timeout failures retain enough operation
and observed-position context to diagnose where playback stopped progressing.

## Future N-Input A/B Extension

The future switcher should compose the existing slot interface rather than
copy or specialize its graph:

```text
N replay slots -> A selector -> A output edge
               -> B selector -> B output edge
```

Section controllers can dispatch to one selected slot or linked teams. The
future work adds source lifecycle, slot selection, A/B visibility and linking,
but does not change the single-slot playback operation semantics, position
probe, or Janus output module defined here.

All future inputs in one synchronized section must use the same nominal frame
rate. The slot builder may accept an optional target canvas internally for
that future composition, but version one preserves the source dimensions and
does not expose dormant A/B format logic.

## Acceptance Criteria

The feature is complete when:

- one command transcodes a VOD through CUDA/NVENC to a video-only all-intra
  replay MPEG-TS;
- the canonical binary seek table includes the final encoded frame;
- the transcoder publishes a valid wallclock history sidecar and never exposes
  a newly incomplete output family;
- the player opens the generated recording without OBS or Gateway;
- Janus displays the video-only, 4 Mbit/s NVENC RTP output;
- the TUI exposes every agreed v2 playback control;
- headless regression mode verifies the controls from observed positions;
- paused rapid seek and shutdown do not hang; and
- tracked files contain no private endpoints or machine-specific paths.
