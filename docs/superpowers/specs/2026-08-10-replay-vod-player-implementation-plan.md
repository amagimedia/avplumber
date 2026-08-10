# Replay VOD Transcoder and Single-Slot Player Implementation Plan

Date: 2026-08-10

Design: `docs/superpowers/specs/2026-08-10-replay-vod-player-design.md`

## Outcome

Deliver two executable PyPlumber applications:

- `demos/replay/transcode.py` converts one VOD through CUDA/NVENC into a
  video-only, all-intra replay MPEG-TS with complete seek and wallclock-history
  sidecars.
- `demos/replay/player.py` plays one replay recording through one slot,
  publishes fixed 4 Mbit/s H.264 NVENC RTP directly to Janus, and exposes all
  agreed Stream Studio Gateway v2 single-source playback controls through
  Textual and headless regression modes.

The implementation must not depend on OBS or Stream Studio Gateway at runtime.

## Confirmed Test Seams

Tests will observe behavior only through these agreed seams:

1. **Replay module interface** — configuration in, playback operations and
   `PlaybackStatus` out.
2. **Executable interface** — CLI arguments, exit status, and documented
   artifacts.
3. **Recording artifact interface** — MPEG-TS plus the documented packed seek
   table and timestamp-history sidecars.
4. **Playback output interface** — observed frame/timestamp progress from the
   pass-through probe and RTP packets emitted by the graph.
5. **Textual interface** — visible state and user actions through Textual's
   test pilot.

Graph-construction tests may inject the same lightweight AVPlumber adapter
pattern used by `demos/mixer/tests/test_graph.py`. They assert externally
meaningful graph contracts such as video-only routing, codec options, team
configuration, and RTP settings. They must not assert incidental node ordering
or private helper calls.

## Constraints

- Keep version one video-only, one input, and one output slot.
- Require the NVIDIA stack. Keep decoded frames in CUDA through processing and
  encode both outputs with `h264_nvenc`; do not add a software fallback or any
  `hwdownload`/`hwupload` workaround.
- Do not change graph management, control protocol, main, or sentinel code.
- The only C++ behavior change is seek-table finalization in the existing
  output node.
- Do not copy the mixer TUI or control code. Reuse its Textual patterns while
  implementing replay-specific widgets and behavior.
- Keep this a small demo. Production Python remains limited to `replay.py`,
  `transcode.py`, and `player.py`; prefer plain functions, frozen dataclasses,
  and one controller over framework layers or generalized orchestration.
- Keep tests compact and demonstrative. Parameterize repeated control values,
  share one lightweight graph adapter, and test public behavior instead of
  creating mocks or fixtures for private helpers.
- Extract a helper only when it removes real duplication or hides a genuinely
  complex graph/control detail. Do not add extension points for unimplemented
  A/B or N-input behavior.
- Work in red → green vertical slices. Do not write the whole test suite before
  implementation, and defer nonessential refactoring to final review.

## Task 1: Establish the Replay Artifact and Value Contracts

Files:

- Add `demos/replay/replay.py`.
- Add `demos/replay/tests/test_seek_table.py`.
- Add `demos/replay/tests/conftest.py` only if import setup cannot remain local
  to the test files.

### Slice 1.1: Read a known packed seek table

1. Write a failing test that supplies two literal 16-byte records with known
   timestamps and byte offsets.
2. Add immutable `SeekTableEntry` values and a public
   `read_seek_table(path) -> tuple[SeekTableEntry, ...]` function.
3. Decode native-endian, packed signed-64/unsigned-64 records without depending
   on the output implementation.
4. Run:

   ```sh
   python3 -m pytest -q demos/replay/tests/test_seek_table.py
   ```

### Slice 1.2: Reject invalid recording metadata

For each behavior, add one failing test followed by the smallest
implementation:

- missing file;
- size not divisible by 16;
- empty table;
- decreasing timestamps; and
- decreasing byte offsets.

Expose one validation function returning recording limits needed by the player
rather than separate shallow validation helpers.

### Slice 1.3: Read and validate timestamp history

1. Write a failing test with literal packed 32-byte history records.
2. Add immutable `TimestampHistoryEntry` values and a public reader.
3. Validate record size, ordering, and the frame-zero wallclock mapping.
4. Add literal conversion cases proving media timestamp to UTC and UTC to
   media timestamp without using the implementation formula in expected
   values.

### Slice 1.4: Define playback values

Add public, immutable definitions for:

- `PlaybackOperation`;
- `PlaybackStatus`;
- transcode configuration;
- replay-slot configuration; and
- Janus video configuration with a fixed 4 Mbit/s encoder contract.

Tests should construct valid and invalid configurations through their public
initializers. Validate frame rate, ports, payload type, SSRC, speed range, and
scrubbing range using independent literal examples.

Checkpoint: commit the replay format parser and public value contract.

## Task 2: Build the Transcode Graph and CLI

Files:

- Update `demos/replay/replay.py`.
- Add `demos/replay/transcode.py`.
- Add `demos/replay/tests/test_graph.py`.

### Slice 2.1: Build a video-only all-intra CUDA graph

1. Write a failing graph test through `build_transcode_application(config,
   api=...)` using a lightweight AVPlumber adapter.
2. Require the returned application to expose only lifecycle and output
   artifact information.
3. Build this graph:

   ```text
   input -> demux(v:0) -> dec_video(CUDA) -> force_fps -> force_keyframe
         -> enc_video(h264_nvenc) -> mux -> output(mpegts)
   ```

4. Assert the user-visible format contract:

   - no audio routing or audio nodes;
   - video is the only and therefore stream-zero mux input;
   - CUDA hardware context and CUDA decoder output;
   - baseline profile, GOP size one, no B-frames, `rc=vbr`, `cq=17`, and no
     bitrate cap;
   - `ts_sort_wait` zero;
   - binary seek, text seek, and binary history paths derived from the output
     path; and
   - no realtime pacing in the offline graph.

5. Run the graph test before adding additional CLI behavior.

### Slice 2.2: Generate wallclock history

1. Add failing parser tests for `--wallclock-start now`, timezone-qualified
   ISO-8601 values, and rejection of timezone-less values.
2. Default to `now` and resolve it exactly once in UTC before graph start.
3. After transcode, read the first seek timestamp `P0` and write the literal
   packed record `(0, 0, P0-W0, 0)`, where `W0` is the selected UTC origin.
4. Validate the generated history through the public artifact reader.

### Slice 2.3: Protect and stage output artifacts

1. Add failing CLI/parser tests for missing input, invalid fps, and an existing
   output family.
2. Implement `--input`, `--output`, required `--fps`, optional
   `--wallclock-start` defaulting to `now`, and `--force`.
3. Treat the MPEG-TS, canonical sidecars, and rotated backing files as one
   output family. Without `--force`, report every collision and make no
   changes. With `--force`, remove only the explicitly resolved output family.

4. Transcode into a sibling staging location. Validate every staged artifact,
   publish sidecars with same-filesystem atomic replacement, and publish the
   MPEG-TS path last as the completion marker.

### Slice 2.4: Finish a finite transcode cleanly

1. Add a lifecycle test using an AVPlumber adapter that signals output
   completion and records shutdown.
2. Register the output-finished event before starting groups.
3. Wait for output completion, capture graph exceptions, shut down nodes, and
   validate the resulting sidecar before returning zero.
4. Return nonzero for graph failure, missing output, or invalid sidecar.

Checkpoint: commit the runnable transcode application before changing C++.

## Task 3: Reproduce and Fix Finite Seek-Table Finalization

Files:

- Update `demos/replay/tests/test_seek_table.py` or add
  `demos/replay/tests/test_transcode_integration.py`.
- Update `src/nodes/output.cpp`.

### Slice 3.1: Prove the short-recording failure

1. Add an integration fixture that uses public FFmpeg commands to generate a
   deterministic, video-only VOD shorter than 60 frames.
2. Run `transcode.py` through its executable interface.
3. Assert that canonical `+seek` and `+txt` paths exist and contain every video
   packet reported independently by `ffprobe`.
4. Confirm this test fails against the current output node because the
   canonical sidecar is absent or incomplete.

The integration test may skip with an explicit reason when the built
`_avplumber` module, `ffmpeg`, or `ffprobe` is unavailable. It must not silently
pass.

### Slice 3.2: Publish the pending batch during flush

1. Extract one output-node helper that commits pending binary/text entries to
   the current backing files.
2. Extract or reuse one helper that flushes backing files and publishes only
   configured canonical paths.
3. Use the helpers from normal 60-entry rotation and graceful finalization.
4. Make finalization idempotent so EOF followed by shutdown cannot write a
   trailer or seek entries twice.
5. Preserve:

   - the 16-byte packed record format;
   - the four rotating backing files;
   - the 60-entry live publication interval; and
   - video stream zero as the recorded stream.

6. Rebuild the CUDA/NVENC Python module on the configured NVIDIA Fedora host
   using the same CUDA/neural/TensorRT feature set as the binary build.
7. Re-run the short-recording test until it passes.

### Slice 3.3: Cover exact and later partial batches

1. Add one recording with exactly 60 video packets and one whose packet count
   exceeds 60 but is not divisible by 60.
2. Require both canonical sidecar families to include the final packet without
   duplication.
3. Check independently with `ffprobe` that every encoded video frame is a
   keyframe and seek timestamps/offsets are monotonic.

Checkpoint: commit the output-node correction and its artifact-level
regressions together. Call out this shared output-node behavior change in the
commit message and review notes.

## Task 4: Implement the Playback Controller One Behavior at a Time

Files:

- Update `demos/replay/replay.py`.
- Add `demos/replay/tests/test_controller.py`.

Use a recording command adapter at the AVPlumber control seam. Tests call the
real controller interface and assert resulting status plus complete command
strings; they do not mock internal controller methods.

### Slice 4.1: Play, pause, and toggle

For each operation:

1. write a failing behavior test;
2. implement the minimal state transition and AVPlumber command;
3. assert `PlaybackStatus`; and
4. run the controller tests.

Required behavior:

- pause sends immediate pause and retains configured speed;
- play retains forward/reverse direction and resumes only when configured
  speed is nonzero;
- play at zero speed remains truthfully paused with an actionable status; and
- toggle selects play or pause from current state.

### Slice 4.2: Absolute, frame, and second seeking

Add separate red → green cycles for:

- absolute millisecond seek;
- signed frame seek; and
- signed seconds converted to literal milliseconds.

Use known literal cases including `-30`, `-5`, `-1`, `+1`, `+5`, and `+30`.
Reject booleans, non-finite floats, and out-of-range values before issuing a
command.

### Slice 4.3: Playback speed and reverse

Add cycles for:

- speed get through returned status;
- speed set from 0 through 400 percent;
- zero speed pausing;
- nonzero speed respecting current direction; and
- reverse play using the negative configured magnitude.

Use percent at the controller interface and multiplier in the AVPlumber
command. Expected conversions must be literal values in tests, not recomputed
with the implementation formula.

### Slice 4.4: Scrubbing and restoration

Add cycles for:

- signed scrubbing from -500 through 500 percent;
- the Gateway-compatible 20 percent dead zone and 100 ms throttle;
- zero scrubbing cancellation;
- configured playback speed remaining unchanged while scrubbing; and
- restoring the pre-scrub play/pause and direction state.

### Slice 4.5: Tail seek

Add known duration/start examples proving that tail seek targets duration minus
3000 ms and clamps to the recording start. Label the operation `TAIL -3s` in
user-facing data while retaining the Gateway semantic name in documentation.

Checkpoint: commit the complete v2 single-source playback controller.

## Task 5: Build the Single-Slot Player and Janus Graph

Files:

- Update `demos/replay/replay.py`.
- Update `demos/replay/tests/test_graph.py`.
- Add `demos/replay/player.py` with headless startup first.

### Slice 5.1: Build a controllable slot

1. Write a failing graph test through `build_player_application(config,
   api=...)`.
2. Build:

   ```text
   input_rec -> demux(v:0) -> dec_video(CUDA) -> speed_video -> force_fps
             -> pause -> realtime<VideoFrame> -> position probe
   ```

3. Require `input_rec` to use adjacent `+seek` and `+history` sidecars, zero
   preseek, `timestamp_source=wallclock`, looping by default, and the
   controller's pause/speed/realtime teams.
4. Infer FPS from the artifact, verify it against seek-table cadence, and fail
   on missing or inconsistent values. Support `--no-loop` and expose loop state
   in public status.
5. Keep all node and team names inside the slot implementation. Return only
   the output video edge, controller, and application lifecycle.

### Slice 5.2: Report observed position

1. Add a failing test feeding frames with literal `frame_no`, `frame_ts`, and
   wallclock metadata through the probe.
2. Implement a thread-safe pass-through `PythonNode` that forwards each frame
   once and updates observed status.
3. Prove missing metadata does not overwrite the last valid observation or
   block the graph.

### Slice 5.3: Add the video-only Janus output

1. Add a failing graph test for the public Janus configuration.
2. Build:

   ```text
   slot edge -> force_keyframe -> enc_video(h264_nvenc, 4 Mbit/s CBR)
             -> bsf(dump_extra=freq=keyframe) -> mux -> output(rtp)
   ```

3. Require RTP/RTCP port pairing, payload type 96 by default, configured SSRC,
   `skip_rtcp`, and no audio or OBS sink nodes. Match the mixer encoder contract:
   `b=maxrate=bufsize=4000k`, one-second GOP, no B-frames, baseline, CBR,
   ultra-low latency, forced IDRs, and strict GOP.
4. Reuse `pyplumber.rtcp_feedback.RtcpFeedbackListener`; PLI/FIR invokes the
   force-keyframe node trigger.
5. Add an NVIDIA-host integration check with a UDP receiver. Require at least
   one RTP packet with the configured payload type and SSRC, independently of
   Janus availability.
6. Validate CLI options for recording, Janus host, video port, payload type,
   SSRC, loop mode, and control timeout. FPS is inferred, not a player option.

### Slice 5.4: Lifecycle and failure reporting

1. Add a lifecycle test proving start order, listener start, listener stop,
   and graph shutdown.
2. Route PyPlumber node exceptions into application status.
3. Make repeated stop safe for Textual exit, headless completion, and error
   cleanup paths.

Checkpoint: commit a headless single-slot player that can publish to Janus.

## Task 6: Add the Headless v2 Regression Exercise

Files:

- Update `demos/replay/replay.py`.
- Update `demos/replay/player.py`.
- Add `demos/replay/tests/test_exercise.py` if it keeps scenario tests clearer
  than `test_controller.py`.

### Slice 6.1: Wait for observed state with bounded time

1. Write a failing test with a deterministic in-memory status sequence.
2. Implement one bounded wait primitive returning diagnostic before/after
   statuses on timeout.
3. Use monotonic time and never sleep indefinitely.

### Slice 6.2: Exercise each playback behavior

Add one scenario at a time, keeping each red → green cycle executable:

1. play advances;
2. pause remains within one-frame tolerance;
3. absolute seek reaches a target;
4. configured frame nudges reach expected seek-table indices;
5. configured second nudges move in the requested direction;
6. 50%, 100%, and 200% playback have distinguishable observed rates;
7. reverse play decreases timestamps;
8. positive/negative scrub changes direction and cancellation restores state;
9. tail seek reaches duration minus three seconds; and
10. absolute UTC seek reaches the history-predicted frame; and
11. rapid paused seeks followed by shutdown finish before the timeout.

For recordings too short to exercise a 30-second nudge, report an explicit
skip for that value while continuing the remaining checks. Unit tests still
cover its command conversion.

### Slice 6.3: Expose the executable mode

Implement:

```text
player.py --no-tui --exercise-v2 ...
```

Print a compact per-operation result and exit nonzero if any non-skipped check
fails. Include the last command, expected condition, and observed status in a
failure.

Checkpoint: commit the headless decoder/seek regression harness.

## Task 7: Add the Icon-First Textual Player

Files:

- Update `demos/replay/player.py`.
- Add `demos/replay/requirements.txt` with the same compatible Textual floor as
  the mixer demo unless current dependency resolution requires a higher one.
- Add `demos/replay/tests/test_tui.py`.

### Slice 7.1: Render actual playback status

1. Write a failing Textual pilot test using the real controller with an
   in-memory AVPlumber command adapter and position source.
2. Render source, Janus destination, current/duration timeline, frame, state,
   direction, speed, scrubbing speed, mapped UTC, loop mode, and latest error.
3. Poll `controller.status()` without blocking Textual's event loop.

### Slice 7.2: Transport controls

Add a failing pilot action then implement each control:

- play;
- pause;
- current play toggle through Space;
- reverse play;
- negative/positive scrubbing;
- stop scrubbing; and
- tail minus three seconds.

Use icon-plus-label buttons and tooltips. Test operations received through the
controller interface, not widget implementation details.

### Slice 7.3: Nudge controls

Add parameterized pilot tests for all twelve buttons:

```text
-30f -5f -1f +1f +5f +30f
-30s -5s -1s +1s +5s +30s
```

Add keyboard shortcuts for single-frame and single-second nudges without
shadowing text entry.

### Slice 7.4: Speed, absolute seek, and regression controls

Add tests and implementation for:

- 0.25x, 0.5x, 1x, 2x, and 4x presets;
- absolute timestamp entry with validation;
- timezone-qualified UTC entry with a `GO TO UTC` action;
- live display of speed/scrubbing get semantics; and
- starting the headless v2 exercise in a Textual worker with incremental
  pass/fail results.

### Slice 7.5: Error and responsive states

Add tests for:

- red graph/command failure banner;
- amber pause, violet reverse, blue seek, and green forward-play states;
- disabled controls until the source is ready; and
- clean quit while an operation is pending.

Checkpoint: commit the Textual interface and its interaction tests.

## Task 8: Documentation and Operator Workflow

Files:

- Add `demos/replay/README.md`.

Document:

- NVIDIA/CUDA/NVENC Python-module prerequisites;
- Textual installation;
- one-input transcode example using `<path>` placeholders;
- produced MPEG-TS and sidecar files;
- default synthetic `now` wallclock origin and explicit ISO-8601 usage;
- Janus video-only mountpoint requirements;
- TUI launch and keyboard map;
- all v2 playback controls and their AVPlumber meaning;
- headless regression usage and exit behavior;
- why finite VOD uses `TAIL -3s` rather than claiming to be live;
- the absence of audio, OBS, recording, clips, playlists, and A/B switching;
- the future composition seam for N inputs and A/B outputs; and
- troubleshooting for invalid sidecars, missing video, no Janus picture, and
  RTCP/keyframe issues.

Keep private endpoints, remote hostnames, credentials, and instance-specific
paths out of tracked documentation.

Checkpoint: commit the documented two-application workflow.

## Task 9: Final Validation and Review

### Pure Python tests

Run:

```sh
python3 -m pytest -q demos/replay/tests/test_seek_table.py
python3 -m pytest -q demos/replay/tests/test_controller.py
python3 -m pytest -q demos/replay/tests/test_graph.py
python3 -m pytest -q demos/replay/tests/test_tui.py
python3 -m pytest -q demos/replay/tests
python3 -m pytest -q tests/test_rtcp_feedback.py
```

### NVIDIA integration

On the configured NVIDIA Fedora host with a matching FFmpeg and PyPlumber
CUDA/NVENC build:

1. generate deterministic short and partial-batch VOD fixtures;
2. run both transcoder integration cases;
3. inspect packet/frame data with `ffprobe`;
4. run `player.py --no-tui --exercise-v2`; and
5. repeat the rapid paused-seek/shutdown case enough times to expose a wakeup
   race rather than relying on one pass.

Do not run the media integration, CUDA/NVENC build, or Janus checks locally
when `nvidia-smi` is unavailable. Pure Python graph/controller/history/TUI
tests remain suitable for local execution.

### Janus acceptance

On the user-provided remote Janus environment:

1. configure a video-only H.264 streaming mountpoint with the selected RTP,
   RTCP, and payload-type values;
2. run the generated replay file through the Textual player;
3. watch the full v2 exercise;
4. confirm immediate keyframes after PLI/FIR and seek operations; and
5. record any visual freeze, stale frame, excessive seek delay, or RTP
   discontinuity as a failure.

Do not store the remote hostname, credentials, media location, or Janus
instance details in tracked files.

### Review

- Run `git diff --check`.
- Confirm unrelated worktree changes are untouched.
- Confirm no audio, OBS, Gateway, A/B, playlist, or recording behavior entered
  version one.
- Confirm tests target the agreed seams and do not lock down private helpers.
- Confirm the demo has no avoidable wrapper classes, one-call indirection, or
  repetitive one-case tests that can be expressed parametrically.
- Confirm the output-node behavior change is isolated and documented.
- Perform a final code review only after all red → green slices are complete;
  avoid broad cleanup during feature implementation.

## Definition of Done

- Both executable PyPlumber applications work from documented commands.
- The transcoder emits an all-intra video-only MPEG-TS and complete canonical
  seek tables plus valid wallclock history for short, exact-batch, and
  partial-batch lengths.
- The player publishes fixed 4 Mbit/s NVENC video directly to Janus without
  OBS or Gateway.
- Textual exposes every agreed v2 single-source playback operation, including
  all frame and second nudge values.
- Headless mode verifies playback from observed timestamps/frames and exits
  nonzero on regression.
- Rapid paused seeking and shutdown do not hang.
- All targeted pure, integration, TUI, RTCP, and remote Janus checks pass.
- The implementation remains composable for a future N-input A/B orchestrator
  without carrying unused A/B behavior in version one.
