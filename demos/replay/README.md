# Replay VOD player demo

[![Watch Replay: video beside terminal controls](https://github.com/amagimedia/avplumber/releases/download/replay-demo-media-2026-09/replay-demo.jpg)](https://amagimedia.github.io/avplumber/demos/replay/docs/)

[Watch the demo](https://amagimedia.github.io/avplumber/demos/replay/docs/) · [MP4](https://github.com/amagimedia/avplumber/releases/download/replay-demo-media-2026-09/replay-demo.mp4) · [Full processing graph](https://amagimedia.github.io/avplumber/demos/graph.html?demo=replay)

The 49-second, 1.7 MB recording shows frame/time/UTC seeks, pause, 0.25×–2× speed,
scrubbing, reverse, and seeking after EOF. Click the first-frame preview to play
it with chapter buttons and inspect the web UI graph. The page is hosted on GitHub Pages; media comes from public release assets,
outside Git history.

## Features

This directory contains two small Python applications that drive the **Rust
avplumber** over its control protocol (`doc/control_protocol.md`):

- `transcode.py` converts one video-on-demand file into AVPlumber's seekable
  replay format, running the Rust executable as a batch job.
- `player.py` controls one replay slot and sends video-only H.264 RTP to an
  existing Janus Streaming mountpoint, over a Rust avplumber it spawns (or one
  already serving a port).

Nothing runs inside the media process: the graph is sent as `node.add` lines,
playback is driven with `seek`, `pause`, `resume` and `speed.set` on one
playback group, and the position comes back from `playback.status`. The design
is in `doc/specs/rust-refactor/rust_refactor_playback.md`.

The player provides:

- Play, Pause, Play/Pause Toggle, and Reverse controls: **PLAY** and
  **REVERSE** are the two directions, each turns playback around when it was
  going the other way, and the toggle (Space) resumes in the current direction;
- forward and backward scrubbing;
- relative seeks of 1, 5, or 30 frames and 1, 5, or 30 seconds;
- `0x`, `0.25x`, `0.5x`, `1x`, and `2x` playback speeds;
- absolute media-time and timezone-qualified UTC seeks;
- **TAIL -3s**, which jumps to three seconds before the recording ends;
- optional looping, enabled by default; and
- a built-in **RUN V2** playback regression exercise.

The status panel shows the recording, Janus destination, RTP payload type and
SSRC, Play/Pause state, direction, configured and scrub speeds, current frame,
position and duration, mapped UTC time, loop state, an `END` marker when the
recording has run out, last command, and errors. Controls remain disabled until
the first source frame is ready.

Keyboard controls are Space for Play/Pause, Left/Right for one frame, Down/Up
for one second, and `q` to quit. The footer displays these bindings.

## Requirements

Build the Rust executable and point the scripts at it:

```sh
cargo build -p avplumber_nodes --features ffmpeg7_1,async --bin avplumber
export AVPLUMBER_BIN=$PWD/target/debug/avplumber
```

(Pick the `ffmpeg*` feature matching the FFmpeg the crate links against, see
`avplumber_nodes/Cargo.toml`.) The FFmpeg libraries must provide software H.264
decoding and `libx264`; hardware acceleration is not used. Both scripts also
take `--avplumber PATH`; nothing is taken from `PATH`, because the C++ binary
of the same name does not run these graphs.

Install the TUI dependency into the same Python environment:

```sh
python3 -m pip install -r demos/replay/requirements.txt
```

## Create a replay recording

Choose the output frame rate explicitly:

```sh
python3 demos/replay/transcode.py \
  --input <path>/source.mp4 \
  --output <path>/replay.ts \
  --fps 30
```

The frame rate must be an integer from 1 to 240. The output is all-intra H.264
baseline video encoded by libx264 at constant quality (`crf=17`). It preserves
the source dimensions and creates four files that must stay together:

```text
replay.ts
replay.ts+seek
replay.ts+txt
replay.ts+history
```

By default, frame zero maps to the UTC time at which conversion starts. Supply
a timezone-qualified value to choose another origin:

```sh
python3 demos/replay/transcode.py \
  --input <path>/source.mp4 \
  --output <path>/replay.ts \
  --fps 30 \
  --wallclock-start 2026-08-10T12:00:00Z
```

The converter refuses to overwrite any member of an existing output family.
Pass `--force` to replace it. Conversion uses a sibling staging directory and
publishes the validated `.ts` file last, so an incomplete conversion does not
look ready to the player.

## Configure Janus

Create a video-only Janus Streaming mountpoint that receives H.264 RTP. This
demo uses the following defaults:

| Setting | Default | Player option |
| --- | --- | --- |
| RTP destination | `127.0.0.1:5004` | `--janus-host`, `--janus-video-port` |
| RTCP destination | `127.0.0.1:5005` | follows the video RTP port |
| RTP payload type | `96` | `--janus-video-pt` |
| SSRC | `0x41565001` | `--janus-video-ssrc` |
| output bitrate | 4000 kbit/s CBR | fixed by the demo |
| local RTCP listener | `0.0.0.0` on an automatic port | `--janus-rtcp-bind`, `--janus-rtcp-port` |

The demo sends RTCP sender announcements and listens for PLI/FIR feedback;
either feedback request forces an immediate encoder keyframe. It does not
create or destroy the Janus mountpoint through the Janus API.

## Run the player

```sh
python3 demos/replay/player.py \
  --recording <path>/replay.ts \
  --janus-host 127.0.0.1 \
  --janus-video-port 5004
```

The player validates the recording and its sidecars, infers the integer frame
rate from the seek table, spawns the Rust avplumber on a free local port, sends
it the graph, waits for the first decoded frame, starts the RTCP listener, and
opens the TUI. It rejects inconsistent frame cadence.

Useful options are:

- `--no-loop` to stop at the end instead of looping;
- `--no-tui` to keep the player running without the terminal interface;
- `--control-timeout <seconds>` to change the five-second operation timeout;
- `--connect HOST:PORT` to drive an avplumber already serving its control
  protocol instead of spawning one;
- `--avplumber-log PATH` to keep the spawned avplumber's log; and
- the Janus options in the table above when the mountpoint differs.

Run `python3 demos/replay/player.py --help` for the complete option list.

### Scrubbing and speed

Scrubbing has a 20% dead zone and sends updates at most once every 100 ms. When
scrubbing stops, the player restores the previous Play/Pause state and
forward/reverse direction.

Choosing `0x` pauses playback. Play then reports that a nonzero speed must be
selected. The configured speed remains separate from the temporary scrub
speed.

## Regression exercise

Run the playback checks without the TUI:

```sh
python3 demos/replay/player.py \
  --recording <path>/replay.ts \
  --no-tui \
  --exercise-v2
```

The exercise requires fresh, exact frames after seeks and nudges and a stable
paused frame. It also checks Play, speed changes, Reverse, scrubbing, Tail,
UTC seek, and the final target after rapid paused seeks. It prints `PASS`, `FAIL`, or an explicit `SKIP` when the recording is too short for
a nudge, and exits nonzero on failure. The TUI's **RUN V2** button runs the same
checks.

## Troubleshooting

- If startup rejects the recording, confirm that its `+seek` and `+history`
  files exist and belong to the same conversion.
- If the Rust avplumber fails to start or decode, run it by hand with
  `--log debug` (or pass `--avplumber-log` to the player) and check that its
  FFmpeg libraries provide software H.264 decoding and `libx264`.
- If Janus has no picture, compare its H.264 codec, payload type, SSRC, RTP port,
  and RTCP port with the values in the TUI header.
- If video does not recover after a seek, check RTCP reachability and confirm
  that PLI/FIR requests reach the configured RTCP bind address.

## Processing graph

<a href="https://amagimedia.github.io/avplumber/demos/graph.html?demo=replay" target="_blank" rel="noopener noreferrer"><img src="https://github.com/amagimedia/avplumber/releases/download/webui-graphs-2026-09/replay-graph.png" alt="Current replay graph from recording input through transport controls to NVENC and RTP, with routed arrows." width="640"></a>

**14 nodes / 13 queues**. Click the current WebUI capture to open the full graph
in an HTML viewer with Fit and zoom controls. It is a static capture of graph
structure and queue state.

## Tests

```sh
python3 -m pytest demos/replay/tests -q
```

The unit tests cover the controller's command translation, the graphs the
scripts send, the player's lifecycle over a fake connection, the TUI (when
`textual` is installed), the recording sidecars, and the paused-picture
oracle. The end-to-end tests run the Rust executable: they transcode
generated `testsrc2` clips (all-intra and with B-frames, odd frame counts
included) and check every packet and seek entry, play a recording through
the same **RUN V2** exercise the TUI offers, check that RTP keeps arriving
after seeks, reversal, scrubbing and nudges, and that the player stops
promptly whatever it was doing. They need `AVPLUMBER_BIN` and an `ffmpeg`
CLI with libx264, and skip otherwise:

```sh
AVPLUMBER_BIN=$PWD/target/debug/avplumber python3 -m pytest demos/replay/tests/test_rust_integration.py -q
```

Frame-exact playback itself is verified in Rust, against the frames the
`ffmpeg` CLI decodes from the same recording: `avplumber_nodes/tests/playback.rs`
(point checks), `playback_scenarios.rs` (the seek/pause/speed sweeps of the
demo's old native suite at 24, 25, 30 and 60 fps) and `encode_after_seek.rs`
(the live encoder across discontinuities). They pace at real time:

```sh
cargo test -p avplumber_nodes --features ffmpeg7_1,async -- --test-threads=1
```

## Format and implementation notes

Seeks resolve to the nearest indexed frame (ties select the later one) and the
source repositions with a byte seek to that frame, so a paused seek shows
exactly that frame. The source never announces an end: at the tail it holds the
last frame, or loops, and stays seekable. Everything the C++ demo did with
`input_rec`, `speed`, `pause`, the realtime team and the position probe is the
Rust core's playback service plus its `realtime` node. The Janus encoder runs
with `flush: keep`: a seek must not touch a live encoder (libavcodec's flush
stops libx264 for good), and the paced frames keep monotonic timestamps
across it anyway.

The binary seek table contains native-endian `(int64 timestamp_ms, uint64
byte_offset)` records. The history contains native-endian `(int64 changed_at,
int64 input_offset, int64 wallclock_offset, int64 output_offset)` records.

The player always has one input, one replay slot, and one output. It
intentionally omits live recording, audio, clips, bins, playlists, transitions,
and A/B switching. `build_player_application` returns one slot's controller
and client so a future multi-input application can compose several slots
without changing their control semantics.
