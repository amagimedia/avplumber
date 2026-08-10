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
- software decoding and `libx264` encoding by default; and
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
  -> dec_video
  -> force_fps
  -> force_keyframe
  -> enc_video(libx264)
  -> mux(video is stream zero)
  -> output(mpegts plus binary and text seek tables)
```

The default encoder contract is:

- codec `libx264`;
- baseline profile;
- GOP size one;
- no B-frames; and
- CRF 17.

The force-keyframe node reinforces the all-intra contract. The requested frame
rate is explicit and defaults to the existing replay example's 60 fps. The
application runs without realtime pacing so offline conversion can be faster
than playback speed.

The output basename produces:

```text
output.ts
output.ts+seek
output.ts+txt
```

The output node retains its rotating seek-table backing files because that is
the live-recorder contract. The canonical `+seek` and `+txt` paths must point
to complete data after finite output closes.

The application refuses to overwrite any output or sidecar unless `--force`
is passed. It waits for the output node to finish, shuts down the graph, and
then validates the artifacts before returning success.

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
input_rec(output.ts with output.ts+seek)
  -> demux video
  -> dec_video
  -> speed_video
  -> force_fps
  -> pause
  -> realtime<VideoFrame>
  -> position probe
  -> force_keyframe
  -> enc_video(libx264, low latency)
  -> bsf(dump_extra=freq=keyframe)
  -> mux
  -> output(rtp)
```

`input_rec` uses the binary seek table, zero preseek, input timestamps, and
looping enabled by default. The playback graph uses one set of pause, speed,
and realtime teams. Their names are namespaced internally so future slot and
section composition does not require changing the controller interface.

The video is decoded and re-encoded for Janus. Re-encoding is required because
pause, reverse, seek, and speed changes create a new continuous output
timeline. Passing the recorded H.264 packets through directly would not
produce correct realtime RTP timing after those operations.

The pass-through position probe records the seek-table frame metadata that
actually reaches the output branch. The TUI and regression runner use this
observed position rather than estimating it from elapsed wallclock time.

The default Janus contract matches the existing demos:

- RTP port 5004 and RTCP port 5005;
- dynamic payload type 96;
- configurable host and SSRC;
- H.264 baseline low-latency output;
- SPS/PPS repeated at keyframes; and
- RTCP PLI/FIR requests trigger a forced keyframe.

The defaults are software-only so graph and playback testing do not require a
GPU. A future CUDA/NVENC implementation must be a separate zero-copy graph
configuration; it must not insert CPU download/upload stages into a CUDA path.

## Playback Operations

The controller implements all current v2 single-source playback operations:

| Operation | Semantics |
| --- | --- |
| Play | Restore configured forward speed and resume |
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

A speed of zero pauses playback, matching Gateway behavior. Setting a nonzero
speed respects the current forward/reverse mode. Scrubbing does not overwrite
the configured playback speed.

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
| 0.25x  0.5x  1x  2x  4x       timestamp input       GO            |
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
- binary seek-table parsing and validation.

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
10. rapid seeks while paused followed by shutdown finish before a timeout.

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

## Acceptance Criteria

The feature is complete when:

- one command transcodes a VOD to a video-only all-intra replay MPEG-TS;
- the canonical binary seek table includes the final encoded frame;
- the player opens the generated recording without OBS or Gateway;
- Janus displays the video-only RTP output;
- the TUI exposes every agreed v2 playback control;
- headless regression mode verifies the controls from observed positions;
- paused rapid seek and shutdown do not hang; and
- tracked files contain no private endpoints or machine-specific paths.
