# Playlist Regression Harness Reset

Date: 2026-08-12

## Purpose

Replace the current playlist proof of concept with a small, deterministic video
regression harness. It must make the playlist control contract easy to exercise
from a clickable terminal UI while keeping one Janus RTP output alive for the
application's entire ready lifetime.

The implementation keeps the existing pybind GIL change. It does not preserve
the current two-worker preload/scheduled-switch design merely because it already
exists.

## Success criteria

The harness is successful when:

- one enabled playlist element produces the initial frame before the application
  reports ready;
- after readiness, the shared encode, mux, and Janus RTP output never stop in
  response to an element or playlist transport action;
- Stop and source replacement continuously transmit the most recent video frame
  at the configured output frame rate;
- the selected element and the playlist have independent playback-mode controls;
- every public element and playlist action has a controller test and a clickable
  TUI test;
- a live regression verifies that RTP packets continue across Pause, Stop,
  Next, Previous, Goto, natural EOF, and a source-load failure;
- the TUI contains one visible control surface and no command or AVPlumber logs.

An empty playlist, a playlist with no enabled elements, or failure to obtain the
first frame is a clear startup error. The always-live guarantee begins when the
application reports ready; it does not invent a black frame before any media has
been decoded.

## Non-goals

- Gapless or frame-perfect cuts. During source replacement, the previous frame
  remains visible until the new source produces a frame.
- Parallel preload workers.
- Timeline-scheduled source-switcher cuts.
- Audio playout.
- A production playlist service, persistence layer, or remote API.
- A command log, diagnostic panel, or HTML UI mockup.

## Test media

The demo uses five generated H.264 MP4 fixtures rather than the 95-second
basketball demo assets. Every fixture is 1920x1080, 30 fps, exactly 300 frames
and 10 seconds long. Each uses a distinct FFmpeg test pattern and burns in its
clip identity, zero-based frame number, and `HH:MM:SS.mmm` PTS clock.

Generated MP4 files are local artifacts rather than source-controlled binaries.
A deterministic generator script is part of the demo so the same fixtures can
be recreated in the local workspace or on the remote NVIDIA test environment.

## Playback model

`PlaylistMode` controls automatic behavior after an element completion:

- `PlayAll`: advance through enabled elements and stop at the end.
- `PlayCurrent`: stop after the current element.
- `LoopAll`: advance through enabled elements and wrap at the end.
- `LoopCurrent`: restart the current element after completion.

`ElementMode` controls what completes an individual element:

- `PlayToEnd`: complete at EOF or at `play_to_ms` when configured.
- `Timed`: complete after `duration_ms` of wall-clock playback.
- `LoopSelf`: restart the element at EOF without invoking playlist advancement.

Automatic completion and manual navigation use separate functions. Manual
Play/Goto always selects the requested enabled element. Manual Next and Previous
always leave a `LoopSelf` element and move among enabled elements; playlist loop
modes determine whether navigation wraps at a boundary.

The controller exposes two explicit action scopes:

- selected-element actions: Play, Pause, Stop, enable/disable, set element mode,
  update cue-in/cue-out/duration/speed, add, remove, and reorder;
- playlist actions: Play, Pause, Stop, Next, Previous, and set playlist mode.

No action calls application shutdown. Stopping an inactive element records it as
stopped without disturbing the active source. Stopping the active element, or
stopping playlist transport, tears down or parks only the source group and leaves
the last-frame/output group running. A subsequent Play starts the selected
element at its configured cue-in.

## Architecture

The graph has two lifecycle domains:

1. A replaceable `source` group contains the one active input, demux, CUDA decode,
   speed, optional CUDA normalization, and an EOF/readiness probe.
2. A permanent `output` domain contains realtime timestamping, a shared frame-rate
   holder, keyframe forcing, NVENC, muxing, and Janus RTP output.

Next, Previous, Goto, LoopCurrent, and LoopSelf replacement follow one operation:

1. Keep the permanent output running on its cached frame.
2. Stop and remove only the source group.
3. Build the selected element using its URL, cue range, speed, and loop behavior.
4. Start the source group and wait asynchronously for its first valid frame.
5. Replace the cached frame naturally when that frame reaches the permanent path.

There is no `source_switcher` and no worker-index bookkeeping. A failed replacement
leaves the prior frame transmitting, reports the element error in controller
status, and permits another selection or retry.

The EOF/readiness probe only records events. TUI/controller polling performs graph
mutations, so node-processing callbacks do not recursively start or stop groups.

## Narrow framework additions

Two additive interfaces are justified by requirements that cannot be implemented
cleanly in a Python-only graph:

1. `force_fps` gains an opt-in `repeat_on_stall` parameter. After the first valid
   frame, it continues emitting that cached frame with monotonically increasing
   timestamps when its input is empty. The default is false, preserving current
   behavior. Unit coverage includes disabled compatibility, repeat cadence,
   replacement by a new input frame, output backpressure, and EOF handling.
2. The embedded API gains a captured command-execution method that returns the
   command response rather than writing it to process stdout. Existing command
   execution retains its behavior. The live playlist backend uses the captured
   form, while `setLogFile` routes AVPlumber logs to a user-selected file.

These are explicit framework changes. They must be documented and tested outside
playlist-specific assertions. No graph-management or global stop semantics change.

## Controller boundary

The controller depends on a small backend protocol rather than AVPlumber command
strings. The protocol covers source load/replacement, Play, Pause, Stop, speed,
and permanent-output health. A fake backend drives deterministic unit tests; the
live backend translates the same calls into AVPlumber group and control commands.

Controller status is the only UI state source. It identifies the selected index,
active index, playlist mode, each element's transport/mode/settings, pending load,
last error, and Janus health. Successful backend command text is never UI state.

## Terminal UI

The Textual application has three unambiguous areas:

- playlist status and Janus health;
- the selectable element table;
- one explicit button panel, separating playlist transport/mode controls from
  selected-element controls.

Keyboard bindings call the same action methods as buttons but are hidden from the
Textual Footer; the Footer is not rendered. Neither stdout command responses nor
the last command appear in the application. Failures use the status area and
Textual notifications.

While Textual owns the terminal, playlist backend code must write no bytes to the
terminal's stdout or stderr. AVPlumber logs are routed to the configured log file,
embedded command replies are captured, and TUI dry-run mode records commands in
memory instead of printing them. A non-TUI diagnostic invocation may print only
when explicitly requested.

## Test seams

Tests exercise public behavior at four agreed seams:

1. Pure transition functions: the full 4 playlist modes by 3 element modes
   automatic matrix, plus manual boundary/wrap behavior.
2. `PlaylistController` with a fake backend: every selected-element and playlist
   action, Stop-to-Play, edits during playback, disabled elements, EOF, load
   failure, and output-health reporting.
3. Textual Pilot: every visible button and keyboard binding, selected-row
   targeting, mode indication, absence of Footer duplication, and absence of
   command-log text. A terminal-output capture asserts that backend actions emit
   nothing to stdout or stderr while the TUI is active.
4. The remote NVIDIA/Janus environment: packet cadence and decoded-frame identity
   across all transport/replacement actions, including an intentional failed URL.

Command-shape assertions may supplement these tests but cannot substitute for
state, frame, or RTP observations.

## Delivery boundary

The reset removes the diagnostic launcher, live launcher with machine-specific
paths, HTML mockup, two-worker controller, scheduled-cut logic, and tests that
only protect those discarded mechanisms. It retains reusable Janus/RTCP setup,
media node construction, playlist data types where they match this design, and
the pybind GIL change.

Implementation is complete only after targeted local tests pass and the live
remote Janus regression passes. If the shared stalled-frame repetition cannot
maintain packet cadence under source teardown, the work stops at that failed
technical gate rather than expanding into another playlist architecture.
