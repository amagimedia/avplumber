# Replay VOD and single-slot player demo

This directory contains two small PyPlumber applications:

- `transcode.py` converts one VOD to the video-only replay MPEG-TS format.
- `player.py` controls one replay slot and sends H.264 RTP directly to a
  preconfigured Janus Streaming mountpoint.

Neither application uses OBS or Stream Studio Gateway at runtime. Version one
has one input, one slot, one output, and no audio.

## Requirements

Use an NVIDIA host with an AVPlumber Python module built with the same CUDA,
NVCC, neural, and TensorRT options as the AVPlumber binary. FFmpeg must expose
CUDA decoding and `h264_nvenc`, and AVPlumber must load the matching FFmpeg
libraries. There is deliberately no software fallback and no CPU
`hwdownload`/`hwupload` path.

Install the TUI dependency into the same Python environment:

```sh
python3 -m pip install -r demos/replay/requirements.txt
```

## Create a replay recording

The frame rate is required rather than guessed:

```sh
python3 demos/replay/transcode.py \
  --input <path>/source.mp4 \
  --output <path>/replay.ts \
  --fps 30
```

The output is all-intra H.264 baseline video encoded by NVENC in VBR
constant-quality mode (`cq=17`). It preserves the source dimensions and emits:

```text
replay.ts
replay.ts+seek
replay.ts+txt
replay.ts+history
```

The binary seek table contains packed native-endian `(int64 timestamp_ms,
uint64 byte_offset)` records. The history contains packed native-endian
`(int64 changed_at, int64 input_offset, int64 wallclock_offset, int64
output_offset)` records.

By default, frame zero is mapped to the UTC time at which the command starts.
Use a timezone-qualified value to select a different origin:

```sh
python3 demos/replay/transcode.py \
  --input <path>/source.mp4 \
  --output <path>/replay.ts \
  --fps 30 \
  --wallclock-start 2026-08-10T12:00:00Z
```

The application refuses any existing member of the output family. `--force`
allows replacement. Conversion happens in a sibling staging directory;
validated sidecars are published first and the `.ts` file last.

## Configure Janus

Create a video-only Janus Streaming mountpoint that receives H.264 RTP. The
defaults are:

```text
RTP host        127.0.0.1
video RTP port  5004
video RTCP port 5005
payload type    96
SSRC            0x41565001
bitrate         4000 kbit/s CBR
```

The demo does not create or destroy the mountpoint through the Janus API. It
sends RTCP sender announcements and listens for PLI/FIR; either request forces
an immediate encoder keyframe.

## Run the player

```sh
python3 demos/replay/player.py \
  --recording <path>/replay.ts \
  --janus-host 127.0.0.1 \
  --janus-video-port 5004
```

The player infers the integer frame rate from the seek table and rejects
inconsistent cadence. Playback loops by default; pass `--no-loop` for finite
playback. The header always shows `LOOP=ON` or `LOOP=OFF`.

The TUI provides the current/duration timeline, frame, mapped UTC timestamp,
direction, configured speed, active scrub speed, source readiness, Janus RTP
configuration, last command, and graph errors. Its controls reproduce the
Stream Studio Gateway v2 single-source playback operations:

- play, pause, play/pause toggle, and reverse play;
- scrub backward, scrub forward, and stop scrubbing;
- relative `-30`, `-5`, `-1`, `+1`, `+5`, and `+30` frame seeks;
- relative `-30`, `-5`, `-1`, `+1`, `+5`, and `+30` second seeks;
- `0x`, `0.25x`, `0.5x`, `1x`, and `2x` speed;
- absolute zero-based media-time seek;
- timezone-qualified absolute UTC seek; and
- `TAIL -3s`, the finite-recording equivalent of Gateway's go-to-live action.

Scrubbing keeps the configured playback speed separate, uses the Gateway's
20% dead zone and 100 ms throttle, then restores both the previous pause state
and the previous forward/reverse direction. A configured speed of zero is
truthfully paused; Play reports that a nonzero speed must be chosen.

Keyboard controls are Space for play/pause, Left/Right for one frame,
Down/Up for one second, and `q` to quit. The buttons expose every larger nudge.

## Regression exercise

Run the playback checks without the TUI:

```sh
python3 demos/replay/player.py \
  --recording <path>/replay.ts \
  --no-tui \
  --exercise-v2
```

This observes frames after play, pause, absolute and relative seeks, speed
changes, reverse, scrubbing, tail, UTC seek, and rapid paused seeks. It prints
`PASS`, `FAIL`, or an explicit `SKIP` for a nudge that the recording is too
short to exercise, and exits nonzero on failure. The TUI's `RUN V2` button
runs the same checks.

## Troubleshooting

- An invalid or absent `+seek`/`+history` file is rejected before CUDA startup.
- If decoding or NVENC fails, confirm the process loads the same custom FFmpeg
  libraries as the working `ffmpeg` command; do not insert a CPU round trip.
- If Janus has no picture, confirm its mountpoint payload type, SSRC, RTP port,
  RTCP port, and H.264 codec match the TUI header.
- If video does not recover after a seek, inspect RTCP reachability and verify
  PLI/FIR messages reach the player's configured RTCP bind address.

This demo intentionally omits recording live inputs, audio, clips, bins,
playlists, transitions, and A/B switching. `build_player_application` returns
one slot's controller and video output path; a future N-input A/B application
can compose multiple slots without changing their control semantics.
