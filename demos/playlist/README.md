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
input_rec -> demux -> dec_video(cuda) -> speed_video
          -> force_fps -> pause -> item_N_out
```

Eight stable item edges feed the existing typed C++ switcher, leaving room for
add/remove and active-element edits without rebuilding the output graph:

```text
item_0_out ... item_7_out -> source_switcher<av::VideoFrame>
    -> realtime -> position/EOF probe
    -> force_keyframe -> h264_nvenc -> bsf -> mux -> Janus RTP
```

Element Play readies its source before selecting the slot. A failed target does
not replace the previous active source. Element or playlist Stop only stops
source groups; it never stops the switch, realtime, encoder, mux, or RTP output.

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

It emits one JSON result after shutdown and checks that the native RTP output
stays working through playlist and per-item Play/Pause/Stop, modes, navigation,
edits, source failure, add/remove/reorder, and natural EOF. Observe the matching
Janus mountpoint during the run to verify the decoded last frame remains visible
while a source is stopped and that playback resumes without recreating the
mountpoint.

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
  control IDs, and zero command/log output;
- FFprobe validation of all generated fixtures when present.

The remote live regression remains the acceptance gate for CUDA/NVENC and Janus
resume behavior; local tests do not claim to substitute for decoded Janus video.
