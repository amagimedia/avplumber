# Playlist regression harness

A deliberately small simulator for the OBS MSE source-switcher and Stream
Studio playlist controls. It drives AVPlumber's existing C++ nodes and sends one
video-only H.264 RTP stream to an existing Janus Streaming mountpoint.

This is a regression harness, not a second production playlist service. It does
not modify `source_switcher`, `force_fps`, sentinel, graph management, or the
control protocol.

## Fixed regression scenario

The default playlist has five distinct generated clips. Every clip is H.264,
1920x1080, 30 fps, 300 frames, and exactly ten seconds. The picture contains a
unique clip label, zero-based frame number, and PTS clock.

Generate or regenerate them with:

```sh
demos/playlist/test-media/generate.sh
```

The MP4 files are ignored build artifacts. The generator is the authoritative
fixture definition; the 95-second basketball demo files are not used.

## Existing-node graph

Each loaded element owns a stable source group:

```text
input_rec(pause_team=item_N_pause_team)
    -> demux -> dec_video(cuda) -> speed_video
    -> force_fps -> item_N_normalized
```

Sixteen fixed item edges feed the existing typed C++ switcher. Only used slots
have source nodes. Before a changed or removed source releases its slot, the
backend stops its group and waits until every source node reports non-working.
Reuse creates a uniquely named source generation feeding the same fixed switcher
edge; stopped generations remain inert until application shutdown. This avoids
unsafe node/team-name reuse while leaving the permanent switch/output graph
untouched:

```text
item_0_normalized ... item_15_normalized
    -> source_switcher<av::VideoFrame>
    -> realtime -> position probe
    -> force_keyframe -> h264_nvenc -> bsf -> mux -> Janus RTP
```

Element Play holds `input_rec` through its existing pause team, starts the source,
observes one non-consuming readiness frame, and pauses it again. It then starts
the permanent output if necessary, resumes and selects the ready slot, and waits
for shared output before stopping the old source. EOF is observed on the selected
switch edge before `realtime` consumes the marker. A failed target does not replace
the previous active source. Element or playlist Stop waits for only that source
group to stop; it never stops the switch, realtime, encoder, mux, or RTP output.

URL, cue, loop-mode, and speed edits are construction changes in this harness.
They use the same ready-source handoff instead of issuing `speed.set` across the
multi-source switch graph; `speed_video` receives the requested speed when the
new generation is constructed.

Like the working replay demo, Pause/Stop makes the push graph quiet and a viewer
normally retains its last decoded frame. The harness does not synthesize a
cadence with sentinel or a modified `force_fps`. The live regression verifies
that the real Janus setup survives the quiet interval and resumes on Play.

## Controls

The Textual UI has separate, explicitly labelled rows:

- playlist: Play, Pause, Stop, Previous, Next, and `PlayAll`, `PlayCurrent`,
  `LoopAll`, `LoopCurrent`;
- selected element: Play, Pause, Stop, `PlayToEnd`/`Timed`/`LoopSelf`, edit cue
  and speed, enable/disable, add, remove, and reorder.

Per-element Pause and Stop are regression-only source-lifecycle controls. The
current gateway playlist API exposes item Play/select, cue, duration, disable,
remove, and reorder, while whole-playlist Pause/Stop are media controls.

There is no Footer and no command/log panel. Keyboard bindings are hidden and
call the same action methods as buttons. AVPlumber logs go to `--log-file`, and
native control replies are redirected to that file while Textual owns the
terminal. All graph operations run on a serialized backend worker so the TUI
thread never waits for group start/stop.

The only associated framework-level change is the separately committed Python
binding call guard that releases the GIL during control commands and group
start/stop requests. It permits the readiness/EOF callbacks to run; it does not
change graph or node behavior.

## Run

Install Textual:

```sh
python3 -m pip install -r demos/playlist/requirements.txt
```

On the NVIDIA/Janus host:

```sh
python3 demos/playlist/player.py \
  --janus-host 127.0.0.1 \
  --janus-video-port 5004 \
  --log-file playlist-demo.log
```

The default RTP payload type is 96 and SSRC is `0x41565001`. Use the matching
CLI options when the mountpoint differs.

Run the complete five-item action sequence without Textual with:

```sh
python3 demos/playlist/regression.py \
  --janus-host 127.0.0.1 \
  --janus-video-port 5004 \
  --log-file playlist-regression.log
```

It emits one JSON result after shutdown. A 10 ms background sampler checks that
the native RTP output stays working while every playlist mode runs the complete
playlist Play/Pause/Stop/Stop-to-Play/Next/Previous cycle, every item runs its
transport cycle, and the harness exercises edits, source failure,
add/remove/reorder, actual Timed completion, a native LoopSelf interval, and
PlayToEnd natural EOF. Twenty alternating graph-affecting mode edits additionally
prove stopped source slots are reusable without exhausting the fixed switcher.
Observe the matching Janus mountpoint during the run to verify the decoded last
frame remains visible while a source is stopped and that playback resumes without
recreating the mountpoint.

A host-independent clickable dry run records actions in memory and prints
nothing while Textual owns the terminal:

```sh
python3 demos/playlist/player.py --dry-run
```

## Tests

```sh
python3 -m pytest demos/playlist/tests -q
```

Coverage includes:

- the complete four playlist modes by three element modes matrix;
- every playlist and selected-element action;
- stable C++ switcher-slot graph shape and source-only Stop;
- superseded and failed source loads;
- a deliberately slow backend operation proving public calls remain quick;
- every visible Textual button, primary keyboard bindings, no Footer, unique
  control IDs, main and Add/Edit control visibility at 80x24, and zero
  command/log output;
- FFprobe validation of all generated fixtures when present.

The remote live regression remains the acceptance gate for CUDA/NVENC and Janus
resume behavior; local tests do not claim to substitute for decoded Janus video.
