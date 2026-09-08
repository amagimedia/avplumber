# Replay VOD player demo

[![Watch Replay: video beside terminal controls](https://amagimedia.github.io/avplumber/demos/replay/docs/replay-demo.jpg)](https://amagimedia.github.io/avplumber/demos/replay/docs/)

[Watch the demo](https://amagimedia.github.io/avplumber/demos/replay/docs/) · [MP4](https://github.com/amagimedia/avplumber/releases/download/replay-demo-media-2026-09/replay-demo.mp4) · [Full processing graph](https://amagimedia.github.io/avplumber/demos/graph.html?demo=replay)

The 49-second, 1.7 MB recording shows frame/time/UTC seeks, pause, 0.25×–2× speed,
scrubbing, reverse, and seeking after EOF. Click the first-frame preview to play
it with chapter buttons and inspect the web UI graph. The page is hosted on GitHub Pages; media comes from public release assets,
outside Git history.

## Features

This directory contains two small PyPlumber applications:

- `transcode.py` converts one video-on-demand file into AVPlumber's seekable
  replay format.
- `player.py` controls one replay slot and sends video-only H.264 RTP to an
  existing Janus Streaming mountpoint.

Both applications run directly through PyPlumber with no external control
service.

The player provides:

- Play, Pause, Play/Pause Toggle, and Reverse controls;
- forward and backward scrubbing;
- relative seeks of 1, 5, or 30 frames and 1, 5, or 30 seconds;
- `0x`, `0.25x`, `0.5x`, `1x`, and `2x` playback speeds;
- absolute media-time and timezone-qualified UTC seeks;
- **TAIL -3s**, which jumps to three seconds before the recording ends;
- optional looping, enabled by default; and
- a built-in **RUN V2** playback regression exercise.

The status panel shows the recording, Janus destination, RTP payload type and
SSRC, Play/Pause state, direction, configured and scrub speeds, current frame,
position and duration, mapped UTC time, loop state, last command, and errors.
Controls remain disabled until the first source frame is ready.

Keyboard controls are Space for Play/Pause, Left/Right for one frame, Down/Up
for one second, and `q` to quit. The footer displays these bindings.

## Requirements

For a build from public sources, use the [Docker instructions](#run-in-docker)
and [shared NVIDIA setup guide](../README.md).

Use an NVIDIA host with `pyplumber` built with CUDA and NVCC support. FFmpeg
must provide CUDA decoding and `h264_nvenc`. Neural models and TensorRT are
not required. If you also build the standalone AVPlumber binary, use the same
feature settings and FFmpeg libraries for it and the Python module.

There is no software fallback, audio output, or CPU
`hwdownload`/`hwupload` path.

Install the TUI dependency into the same Python environment:

```sh
python3 -m pip install -r demos/replay/requirements.txt
```

## Run in Docker

Build the [shared runtime and start Janus](../README.md), then put an input
video named `source.mp4` in a local `media/` directory. Convert it:

```sh
docker run --rm --gpus all \
  -v "$PWD/demos/replay:/demo:ro,z" \
  -v "$PWD/media:/media:z" \
  --entrypoint python3 avplumber-mixer:local \
  /demo/transcode.py --input /media/source.mp4 --output /media/replay.ts --fps 30
```

Start the player and open the Janus preview at <http://127.0.0.1:8080>:

```sh
docker run --rm -it --gpus all --network host \
  -v "$PWD/demos/replay:/demo:ro,z" \
  -v "$PWD/media:/media:ro,z" \
  --entrypoint python3 avplumber-mixer:local \
  /demo/player.py --recording /media/replay.ts
```

The recording and its sidecars remain in `media/` after the containers exit.
The following sections describe the same tools when running Python directly.

## Create a replay recording

Choose the output frame rate explicitly:

```sh
python3 demos/replay/transcode.py \
  --input <path>/source.mp4 \
  --output <path>/replay.ts \
  --fps 30
```

The frame rate must be an integer from 1 to 240. The output is all-intra H.264
baseline video encoded by NVENC in VBR constant-quality mode (`cq=17`). It
preserves the source dimensions and creates four files that must stay together:

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
rate from the seek table, waits for the first decoded frame, starts the Janus
output, and opens the TUI. It rejects inconsistent frame cadence.

Useful options are:

- `--no-loop` to stop at the end instead of looping;
- `--no-tui` to keep the player running without the terminal interface;
- `--control-timeout <seconds>` to change the five-second operation timeout;
  and
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
- If decoding or NVENC fails, confirm that the process loads the same custom
  FFmpeg libraries as the working `ffmpeg` command. Do not add a CPU round trip.
- If Janus has no picture, compare its H.264 codec, payload type, SSRC, RTP port,
  and RTCP port with the values in the TUI header.
- If video does not recover after a seek, check RTCP reachability and confirm
  that PLI/FIR requests reach the configured RTCP bind address.

## Processing graph

<a href="https://amagimedia.github.io/avplumber/demos/graph.html?demo=replay" target="_blank" rel="noopener noreferrer"><img src="https://amagimedia.github.io/avplumber/demos/replay/docs/replay-graph.png" alt="Current replay graph from recording input through transport controls to NVENC and RTP, with routed arrows." width="640"></a>

**14 nodes / 13 queues**. Click the current WebUI capture to open the full graph
in an HTML viewer with Fit and zoom controls. It is a static capture of graph
structure and queue state.

## Tests

```sh
python3 -m pytest demos/replay/tests -q
```

The same native integration suite supports two codec backends. It runs the real
reader, decoder, pause gates, speed control, and encoder, generates its own clips,
and does not need Janus:

```sh
# CUDA build: NVDEC decoding and NVENC encoding
python3 -m pytest demos/replay/tests/test_playback_integration.py demos/replay/tests/test_transcode_integration.py --replay-backend=nvidia -q

# CPU build: software H.264 decoding and libx264 encoding; no GPU needed
python3 -m pytest demos/replay/tests/test_playback_integration.py demos/replay/tests/test_transcode_integration.py --replay-backend=cpu -q
```

Playback coverage includes 24/25/30/60 fps; forward and reverse at 25/50/100/200%;
absolute, relative, and UTC seeks around frame boundaries; exact frame steps;
repeated and seeded random seeks; both recording boundaries; looping and EOF
recovery; and pause/resume around repeated active speed changes. Assertions
require a freshly observed exact source frame and a stable paused position.
Timestamp seeks use raw AVPlumber commands to exercise the native seek path
independently of the controller.

Finite transcode tests also exercise the regular `input` node and default decoder
EOF behavior, with both all-intra and B-frame H.264 sources. They verify that every
frame reaches the output and seek table. An RTP test checks the configured stream
headers on both backends.

The demux lifecycle suite uses the real native node with video and audio packet
streams. It checks blocked reads, stop/EOF overlap, full output queues, repeated
stops, ordinary EOF completion, and resumed packets after retained EOF. Each
case runs in a child process with a shutdown deadline so a hang fails the test:

```sh
python3 -m pytest demos/replay/tests/test_demux_shutdown_integration.py -q
```

Tests skip if the native module or selected backend's prerequisites are missing.
Both backends need `pytest`, `ffprobe`, and an `ffmpeg` with `libx264` for fixture
generation. The native module must load its matching FFmpeg libraries: CUDA-enabled
libraries for NVIDIA, or libraries with `libx264` for CPU. CPU selection changes
only test graph codec settings; the shipped demo remains NVIDIA-based.

## Format and implementation notes

Indexed seeks resolve the target frame before setting the decoder's discard
cutoff, so a paused seek cannot discard the only frame selected by the seek table.
Replay keeps its reader and demultiplexer available at EOF (`stop_on_eof=false`)
and drains its decoder without ending playback (`hold_at_eof=true`). These are
opt-in settings; regular `input` and default finite-stream EOF behavior are unchanged.

The binary seek table contains native-endian `(int64 timestamp_ms, uint64
byte_offset)` records. The history contains native-endian `(int64 changed_at,
int64 input_offset, int64 wallclock_offset, int64 output_offset)` records.

The player always has one input, one replay slot, and one output. It
intentionally omits live recording, audio, clips, bins, playlists, transitions,
and A/B switching. `build_player_application` returns one slot's controller and
video output path so a future multi-input application can compose several slots
without changing their control semantics.
