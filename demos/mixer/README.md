# Generic manual mixer

This demo builds one video-only 1080x1920 program from any positive number of
inputs. It exposes fullscreen scenes and paged 2/4/8/16-box views through the
generic mixer control protocol. There is no audio path, speaker detection,
automatic switching, face analysis, or source-specific policy.

The video-frame pipeline is zero-copy CUDA from hardware decode through NVENC:
there are no `hwdownload`, `hwupload`, or `hwupload_cuda` stages. Every
supported geometry is created at startup. A `preheat_video_router` feeds fixed
CUDA scale/pad graphs for both mixer slots, and both the `cuda_rect_overlay`
compositors and permanent `transition_cuda` filter are warmed before the
control server reports ready. Fades and left/right/up/down wipes therefore
change runtime parameters without rebuilding a filter graph.

## Run

Build AVPlumber with CUDA, NVCC, and the Python module using the same FFmpeg
installation, then run:

```sh
LD_LIBRARY_PATH=/usr/local/lib python3 demos/mixer/mixer.py \
  --input <input-1> \
  --input <input-2> \
  [--input <input-N> ...] \
  [--output <output-url-or-path>] \
  [--janus-output] \
  [--fps 30] \
  --remote-control-port 7777
```

Each file is decoded through its own `realtime` node, which paces frames from
their timestamps. Input order determines the source numbers shown by the TUI.
Repeat `--input` for any number of configurable file paths and use
`--loop-inputs` for continuous playback of finite test files. Network input
URLs remain supported. Output format is inferred for RTMP, SRT, `.flv`, `.ts`,
`.mp4`, `.mkv`, and `.webm`; otherwise pass `--output-format`.
`--fps` controls input normalization, mixer timing, GOP length, and every
enabled output. Its default is 30; use `--fps 60` only when the decoder can
sustain all selected source streams in real time.

At least one output is required. `--janus-output` publishes H.264 video over
RTP to Janus and may be used alone or alongside `--output`. It defaults to
`127.0.0.1`, RTP port 5004, RTCP port 5005, payload type 96, and includes RTCP
PLI/FIR handling for immediate keyframe requests. Override the destination with
`--janus-host` and `--janus-video-port`. The Janus mountpoint must be configured
as video-only; this demo never creates an audio stream.

## Container

The demo-specific image builds FFmpeg 7.1 with the patches in
`deps/ffmpeg-patches`, verifies the patched CUDA overlay and transition
filters, and builds the CUDA-enabled AVPlumber Python module against that
FFmpeg installation:

```sh
docker build -f demos/mixer/Dockerfile -t avplumber-mixer:local .

docker run --rm --gpus all --network host \
  -v /path/to/media:/media:ro \
  avplumber-mixer:local \
  --input /media/camera-1.mp4 \
  --input /media/camera-2.mp4 \
  --loop-inputs \
  --janus-output
```

Host networking makes the RTP/RTCP and TCP control paths explicit and matches
the existing Janus demos. The input paths are container paths supplied at
runtime; the image contains no media locations or endpoints.

Install Textual and start the separate control UI:

```sh
python3 -m pip install -r demos/mixer/requirements.txt
python3 demos/mixer/tui.py --host 127.0.0.1 --port 7777 \
  --wipe-style wipe_left
```

The TUI uses the production switcher layout: program and preview buses, a
scrollable scene strip, explicit layout/page controls, and cut/fade/CUDA-wipe
controls. Selecting a scene normally loads it into preview and waits for
readiness. Enable `Direct: ON` (or press `t`) to make scene and layout choices
cut directly to program; F1–F9 always perform a direct cut. CUT, FADE, and WIPE
are not sent until the preheated preview slot reports ready when Direct mode is
off. Disconnected or incomplete inputs retain their assigned positions; they
never cause the remaining sources to reorder.

## Layouts

| View | Columns | Rows | Cell |
| --- | ---: | ---: | ---: |
| 2-box | 1 | 2 | 1080x960 |
| 4-box | 1 | 4 | 1080x480 |
| 8-box | 2 | 4 | 540x480 |
| 16-box | 2 | 8 | 540x240 |

Inputs keep their aspect ratio and are centered on black within each cell.
Additional inputs use consecutive pages. Empty cells and unavailable sources
remain black.

## Tests

Pure layout, graph-construction, and control tests do not require a GPU:

```sh
python3 -m pytest -q demos/mixer/tests
```

The runtime acceptance check requires the configured NVIDIA environment. It
must cover every layout, a disconnected input, preview plus cut/fade/all four
procedural wipes, and the deterministic `overlay_many_cuda` matrix in
`demos/cuda-overlay`.

With the mixer running, the protocol smoke test covers the required scenes and
reports transition command and settle latency:

```sh
python3 demos/mixer/smoke_test.py --port 7777
```
