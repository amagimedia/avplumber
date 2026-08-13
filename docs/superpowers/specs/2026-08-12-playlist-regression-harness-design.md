# Playlist Regression Harness Reset

Date: 2026-08-12

## Purpose

Replace the current playlist proof of concept with a small, deterministic
regression harness for AVPlumber's existing C++ nodes. It approximates the OBS
MSE source-switcher and Stream Studio gateway control contract without creating
a second production playlist implementation.

The harness must be easy to exercise from one clickable terminal UI and keep
one Janus output graph alive for the application's entire ready lifetime.

## Success criteria

The harness is successful when:

- one enabled playlist element produces the initial frame before the application
  reports ready;
- after readiness, playlist and element actions never stop the shared realtime,
  encode, mux, or Janus RTP nodes;
- the selected element and the playlist have independent playback controls;
- every public element and playlist action has controller and clickable TUI
  coverage;
- a live regression exercises five distinct 1920x1080, 30 fps, ten-second
  elements through Pause, Stop, Stop-to-Play, Next, Previous, Goto, natural EOF,
  all playback modes, and a source-load failure;
- the TUI has one visible control surface, remains responsive while backend
  operations execute, and contains no command or AVPlumber logs.

An empty playlist, a playlist with no enabled elements, or failure to obtain the
first frame is a clear startup error.

## Existing-node constraint

The harness must not modify `force_fps`, sentinel, graph management, the control
protocol, or any other framework/node behavior. `force_fps` only normalizes
arriving frames; it is not a stalled-source generator. Sentinel is excluded
because its backup-media contract does not model a stopped VOD element.

The harness uses Python edge-wiretap callbacks for readiness and EOF events.
The separately committed pybind call guards remain necessary so those callbacks
can acquire the GIL while another Python thread is waiting in a control command
or group start/stop request. This is a binding-liveness change only; playlist
policy does not depend on new graph or node semantics.

## Test media

The demo uses five generated H.264 MP4 fixtures rather than the 95-second
basketball assets. Every fixture is 1920x1080, 30 fps, exactly 300 frames and ten
seconds. Each uses a distinct FFmpeg test pattern and burns in its clip identity,
zero-based frame number, and `HH:MM:SS.mmm` PTS clock.

Generated MP4s are ignored local artifacts. A deterministic generator script is
part of the demo so the same fixtures can be recreated locally and on the remote
NVIDIA test environment.

## Playback model

`PlaylistMode` controls automatic behavior after an element completes:

- `PlayAll`: advance through enabled elements and stop at the end.
- `PlayCurrent`: stop after the current element.
- `LoopAll`: advance through enabled elements and wrap at the end.
- `LoopCurrent`: restart the current element after completion.

`ElementMode` controls completion of one element:

- `PlayToEnd`: complete at EOF or configured cue-out.
- `Timed`: complete after `duration_ms` of unpaused wall-clock playback.
- `LoopSelf`: the file input loops without playlist advancement.

`Timed` maps to the MSE switcher's time-switch behavior. `LoopSelf` is an
explicit composite over the underlying source's loop setting; it is not claimed
to be a native gateway playlist enum.

Automatic completion and manual navigation are separate. Manual Play/Goto,
Next, and Previous escape current-only and element-loop modes. Playlist loop
modes alone determine manual boundary wrapping.

The controller exposes two scopes:

- selected-element: Play, Pause, Stop, enable/disable, set element mode, update
  cue-in/cue-out/duration/speed, add, remove, and reorder;
- playlist: Play, Pause, Stop, Next, Previous, and set playlist mode.

Per-element Pause and Stop are explicitly regression-only source-lifecycle
operations; the current gateway playlist API exposes item Play/select but not
item Pause/Stop. No action shuts down the application or output graph.

## Graph architecture

The graph uses the existing nodes without behavioral changes:

```text
item N: input_rec(pause_team=item_N_pause_team)
        -> demux -> dec_video(cuda) -> speed_video
        -> force_fps -> item_N_normalized

item_0_normalized .. item_15_normalized
        -> source_switcher<av::VideoFrame>
        -> realtime -> position probe
        -> force_keyframe -> h264_nvenc -> bsf -> mux -> Janus RTP
```

Sixteen fixed switcher slots permit add/remove and active-item rebuilds without
recreating the permanent switch/output graph. Only used slots have source
nodes. Changed and removed source groups are stopped, every source node is
observed non-working, and then the fixed switch edge can be reused by a uniquely
named source generation. Stopped generations remain inert until application
shutdown. This avoids both the asynchronous group-stop race and unsafe
node/team-name reuse without touching the permanent graph. Five slots are
populated by the default scenario. Reordering changes policy order, not graph
slot identity.

Element Play holds the destination `input_rec` through its existing pause team,
starts the source group, observes one valid frame on a non-consuming normalized
edge wiretap, then pauses the input again. It ensures the permanent output is
started, resumes and selects the ready C++ switcher slot, requests a keyframe,
and waits for the new frame to reach the shared path before stopping the previous
source group. A failed destination retains the previous active source and frame.

URL, cue, loop-mode, and speed edits rebuild through the same generation handoff.
In particular, the harness does not issue runtime `speed.set` across the
multi-source switch graph; the selected speed is an existing `speed_video`
construction parameter.

Element Pause uses only that source's input pause team. Element Stop stops only
that source group and waits for every node in it to leave the working state
before a restart is accepted. The output graph stays started. As in
`demos/replay`, no new video frames are sent while the active source is stopped,
so the viewer retains its last decoded frame. Continuous RTP during Stop is not
claimed or synthesized; the live Janus regression must prove that the mountpoint
resumes correctly.

Readiness wiretaps observe normalized item edges, the EOF wiretap observes the
selected switch edge before `realtime` consumes the EOF marker, and the
downstream position probe records frame position. These callbacks only record
events. TUI/controller polling performs policy changes so processing callbacks
never recursively mutate graph groups.

## Controller and backend boundary

The controller depends on a narrow asynchronous backend protocol. Public
backend methods only enqueue work and update cached intent; blocking AVPlumber
group operations execute on one serialized worker. Completion, EOF, error, and
output-health events return through a queue and are applied on the UI/controller
thread. Superseded activation requests are discarded before their visible cut.

Controller status is the only UI state source. It reports selected, active and
pending elements; playlist and element modes; transport; error; and cached Janus
health. Successful backend command text is never UI state.

## Terminal UI

The Textual application has three unambiguous areas:

- playlist transport and Janus health;
- the selectable element table;
- one button panel with separate playlist and selected-element rows.

Keyboard bindings call the same methods as buttons and are hidden. No Textual
Footer is rendered. AVPlumber logging is routed through the existing
`AVPlumber.setLogFile` API before graph threads start. Dry-run actions are
recorded in memory and never printed while Textual owns the terminal.

## Test seams

1. Pure functions: the complete four playlist modes by three element modes
   matrix, disabled-item behavior, and manual boundary/wrap behavior.
2. Controller with a fake asynchronous backend: every action, Stop-to-Play,
   edits during playback, EOF, superseded loads, failure, and output health.
3. Graph/backend fakes: existing-node shapes, fixed slot identity, source-stop
   completion before slot release/reuse, readiness before cut, source-only Stop,
   permanent output lifetime, command serialization, and non-blocking public
   calls.
4. Textual Pilot: every button and binding, selected-row targeting, no Footer,
   no duplicate controls, main and Add/Edit visibility at 80x24, no log text,
   and responsiveness during slow backend work.
5. Remote NVIDIA/Janus: five generated clips, decoded-frame identity and
   output-node health across the complete action sequence, actual Timed
   completion, a native LoopSelf interval without controller restart, PlayToEnd
   natural EOF, plus explicit proof that Janus survives and resumes after quiet
   Stop intervals.

## Delivery boundary

The mixed two-worker/single-source controller, command-history UI, diagnostic
launcher, machine-specific live launcher, unsupported stalled-frame design, and
obsolete tests are replaced. Reusable replay Janus/RTCP setup, the exact
four-mode matrix, and deterministic fixture media remain.

Completion requires the complete local playlist suite and the live remote Janus
regression. If the existing quiet Pause/Stop behavior makes Janus fail to resume,
that is a failed feasibility gate; it must not be hidden with per-clip sentinels,
synthetic black media, or framework changes.
