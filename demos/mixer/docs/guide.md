# Mixer reference

[Quick start and demo](../README.md)

## Features

This demo manually mixes up to 192 video inputs into one
1080x1920 portrait program. It provides a separate terminal interface for
choosing what is on air, preparing the next view, and changing between views.

The interface provides:

- separate **Program** (on-air) and **Preview** (ready) buses;
- a scrollable strip of fullscreen and grid scenes;
- fullscreen and paged 2-box, 4-box, 8-box, and 16-box layouts;
- immediate Cut, timed Fade (a mix, or a dip through a colour), and transparent media-file Wipe transitions;
- editable fade duration and media-wipe file path;
- Direct mode for putting scene and layout selections straight on air; and
- connection status, transition status, and manual Reconnect.

With Direct mode off, selecting a scene or layout loads it into Preview.
Cut, Fade, and Media Wipe take the selected scene to Program. With Direct mode
on, scene and layout selections use the current transition immediately (Cut by default). The
take buttons and adjacent selector share this choice. Fade uses the configured
duration; Media Wipe uses the file path and the clip's duration.

A fade mixes, or dips: program fades to a solid colour, shown alone at the
midpoint, then the colour fades to the new scene, in the same fade duration and
at no more GPU cost than a mix. The web UI's colour menu beside the fade curve picks Mix,
Black, White or a custom colour and remembers the choice per browser; the TUI
uses `control.fade_color` from the config (see config.md).

A new take interrupts an unfinished transition. The mixer retains its current
output picture, including a partial fade or wipe, while preparing the new scene.
The replacement transition starts from that picture; cancelled transitions
cannot switch the output back later. This may briefly hold motion during
preparation. Capture happens at the mixer output, before encoding and browser
playback, so it is ahead of the delayed picture on the viewer's screen.

Input order determines source numbers. An input that stalls after startup holds
its last frame in its assigned position; remaining inputs never shift to fill
the gap. Empty cells on the last grid page are also black.

This is deliberately a video-only manual mixer. It has no audio, speaker
detection, automatic switching, face analysis, or source-specific policy.

### Keyboard controls

| Key | Action |
| --- | --- |
| `1`-`9` | Select one of the first nine scenes; Preview normally, Program in Direct mode |
| F1-F9 | Take one of the first nine scenes to Program using the Direct transition selector |
| `c` | Cut |
| `f` | Fade |
| `w` | Media Wipe using the configured transparent clip |
| `t` | Toggle Direct mode |
| `r` | Reconnect |
| `q` | Quit |

Keyboard scene shortcuts are ignored while a fade-duration or wipe-file
field has focus, so numbers and text can be entered normally.

## Requirements

The mixer backend requires:

- an NVIDIA host;
- AVPlumber and `pyplumber` built with matching CUDA and NVCC support;
- the same FFmpeg installation for the binary and Python module;
- FFmpeg with the patched CUDA overlay and transition filters; and
- at least one video input and one output.

The graph accepts only an NVENC encoder and keeps hardware-decoded inputs on
the GPU through output. Raw v210 recipes explicitly exercise software decode
and upload for 4:2:2 inputs; media wipes also upload their alpha frames.

The control TUI requires Textual but may run in a separate terminal or on
another host that can reach the backend's TCP control port:

```sh
python3 -m venv .venv-tui
.venv-tui/bin/python -m pip install -r demos/mixer/requirements.txt
```

## Start the mixer backend

Repeat `--input` for every media file or network URL. At least one of
`--output` and `--janus-output` is required:

```sh
LD_LIBRARY_PATH=/usr/local/lib python3 demos/mixer/mixer.py \
  --input <input-1> \
  --input <input-2> \
  --output <output-url-or-path> \
  --remote-control-port 7777
```

Use `--loop-inputs` to repeat finite files. Each file is paced from its own
timestamps, so input decoding must keep up in real time.

The default frame rate is 30 fps. `--fps` controls input normalization, mixer
timing, GOP length, and every enabled output:

```sh
python3 demos/mixer/mixer.py \
  --input <input-1> \
  --input <input-2> \
  --loop-inputs \
  --output program.mp4 \
  --fps 60
```

Use 60 fps only when the NVIDIA host can decode all inputs in real time.
`--preheat-timeout` controls how long startup waits for every required source
and transition path; its default is 60 seconds.

Output format is inferred for RTMP, SRT, `.flv`, `.ts`, `.mp4`, `.mkv`, and
`.webm`. Pass `--output-format` when the target is ambiguous. The default
recording/stream encoder is `h264_nvenc` at `8M`; another `*_nvenc` codec and
bitrate may be selected with `--codec` and `--bitrate`.

Run `python3 demos/mixer/mixer.py --help` for the complete backend option list.

## Start the control TUI

In another terminal, connect to the backend's control port:

```sh
.venv-tui/bin/python demos/mixer/tui.py \
  --host 127.0.0.1 \
  --port 7777 \
  --wipe-file /path/on/mixer/host/wipe.mov
```

The defaults are mixer name `mixer` and fade duration `0.5` seconds. Use
`--mixer` and `--fade-duration` to change them.

**MEDIA WIPE** plays a video with an alpha channel over the program and changes
the underlying scene at the clip's midpoint. Enter its path in **Wipe file** or
pass `--wipe-file`. The path must be readable by the mixer backend, including
inside its container; it need not exist on the machine running the TUI. The
backend uses the clip's duration, independently of **Fade seconds**. Use a clip
that covers the picture at its midpoint to hide the scene cut.

MOV is a container: the video codec must preserve alpha, for example QTRLE/ARGB
or ProRes 4444. The wipe branch decodes these assets on the CPU and uploads
alpha frames at their native size; scaling and compositing happen together
on the GPU. The demo caches decoded wipes in GPU memory (`--wipe-cache-mb 640`);
`0` decodes on each take instead. The recipe generates its own moving alpha wipes; custom
clips are supplied separately.

The TUI polls Program, Preview, and transition state twice per second. If the
connection fails or is lost, it shows the error in the connection bar; use
**RECONNECT** or press `r` after the backend becomes reachable.

## Janus output

`--janus-output` publishes the video-only Program over RTP: H.264 from an 8-bit canvas, HEVC Main10 from a 10-bit one. It may be used
alone or together with `--output`:

```sh
LD_LIBRARY_PATH=/usr/local/lib python3 demos/mixer/mixer.py \
  --input <input-1> \
  --input <input-2> \
  --loop-inputs \
  --janus-output \
  --remote-control-port 7777
```

Create the video-only Janus Streaming mountpoint before starting the mixer. The
demo does not create or destroy it through the Janus API.

| Setting | Default | Option |
| --- | --- | --- |
| RTP destination | `127.0.0.1:5004` | `--janus-host`, `--janus-video-port` |
| RTCP destination | `127.0.0.1:5005` | follows the video RTP port |
| RTP payload type | `96` | `--janus-video-pt` |
| SSRC | `0x41565001` | `--janus-video-ssrc` |
| bitrate | 4500 kbit/s | `--janus-video-bitrate-kbps` |
| local RTCP listener | `0.0.0.0` on an automatic port | `--janus-rtcp-bind`, `--janus-rtcp-port` |

The mixer sends RTCP sender announcements and listens for PLI/FIR feedback;
either feedback request forces an immediate keyframe.

## Layouts

Every scene uses a 1080x1920 portrait canvas:

| View | Columns | Rows | Cell size |
| --- | ---: | ---: | ---: |
| Fullscreen | 1 | 1 | 1080x1920 |
| 2-box | 1 | 2 | 1080x960 |
| 4-box | 1 | 4 | 1080x480 |
| 8-box | 2 | 4 | 540x480 |
| 16-box | 2 | 8 | 540x240 |

Inputs retain their aspect ratio and are centered on black within each cell.
When there are more inputs than cells, **Page** moves between consecutive
groups without reordering sources.

## Docker

For recipes with generated media, browser sources and WebRTC, use the
[Compose quick start](../README.md#run). The commands below run the mixer alone
with your own video files, after the [shared NVIDIA setup](../../README.md).

The demo image defaults to FFmpeg 8.1 with the shared `deps/ffmpeg/8` series, verifies the
patched CUDA overlay and transition filters, and builds the CUDA-enabled
AVPlumber Python module against that FFmpeg installation:

```sh
docker build -f demos/mixer/Dockerfile.fedora44 -t avplumber-mixer:local .

docker run --rm --gpus all --network host \
  -v <path-to-media>:/media:ro,z \
  avplumber-mixer:local \
  --input /media/camera-1.mp4 \
  --input /media/camera-2.mp4 \
  --loop-inputs \
  --janus-output
```

Host networking exposes the RTP/RTCP and TCP control paths in the same way as
the other Janus demos. Input paths are supplied at runtime; the image contains
no media locations or endpoints.

Start `demos/mixer/tui.py` separately and connect it to port 7777.

### The two images

`demos/mixer/Dockerfile.fedora44`, the Compose default, builds the image on
Fedora 44 with CUDA 13.4, GCC 16, Python 3.14 and Boost 1.90, in a builder stage
plus a slim runtime stage that installs no CUDA packages (FFmpeg and avplumber
load the driver directly). It needs a host driver of R615 or newer: FFmpeg's
CUDA filters are PTX that the driver JIT-compiles, and older drivers reject CUDA
13.4 PTX. The image sets `NVIDIA_REQUIRE_CUDA=cuda>=13.4`, so the container
runtime refuses an older host at start (`requirement error: unsatisfied
condition: cuda>=13.4`). Like the Ubuntu toolchain, the build hardens explicitly
(`-fstack-protector-strong -D_FORTIFY_SOURCE=2 -fcf-protection`), so the two
images differ in toolchain, not hardening; [capacity.md](capacity.md) compares
them on the same 68-input 60 fps show.

The driver keeps the kernels it compiled in `CUDA_CACHE_PATH`, which
`compose.yaml` points at `/media/.nv-cache` on the media volume. Without it the
cache lives in the container (`~/.nv/ComputeCache`) and every new container
compiles again on first use, stalling GPU work during startup; with it only the
first start on an empty media directory does
([deploy/l4/README.md](../deploy/l4/README.md)).

`demos/mixer/Dockerfile` (Ubuntu 22.04 / CUDA 11.7) builds the same `/build`
layout, entrypoint, ports and environment for a host whose driver is older than
R615. `compose.yaml` selects it with one variable, which every later `up --build`
needs as well (or put the line in `demos/mixer/.env`):

```sh
MIXER_DOCKERFILE=Dockerfile docker compose -f demos/mixer/compose.yaml up --build
```

Both build the tag `avplumber-mixer:local`, so switching rebuilds and replaces
the image in place.

Build arguments of `Dockerfile.fedora44`:

| Argument | Default | Purpose |
| --- | --- | --- |
| `WITH_TENSORRT` | `0` | `1` builds the neural nodes (`NEURAL_NET=1 HAVE_TENSORRT=1`) and ships TensorRT in the runtime stage. |
| `--secret id=tensorrt_url` | none | Required with `WITH_TENSORRT=1`: the URL of your tensorrt-minimal tgz, a TensorRT build for CUDA 13 with `bin/trtexec`, `include/`, the `libnvinfer*` and `libnvonnxparser` libraries, and a builder resource for each GPU (`sm75` T4, `sm89` L4). Passed as a build secret (`--secret id=tensorrt_url,env=TENSORRT_URL`), so it stays out of the image history. `.plan` engines are TensorRT-version and GPU specific and are mounted at run time. |
| `CUDA_VERSION` | `13.4` | Must not exceed the host driver's CUDA version (13.4 needs R615). For an R595 host: `13.2` with `CUDA_REPO_URL` set to NVIDIA's fedora43 repository, `gcc15-c++` and `NVCC_PREPEND_FLAGS='-ccbin g++-15'` (nvcc 13.2 supports GCC up to 15). |
| `CUDA_REPO_URL` | NVIDIA's fedora44 repository | Carries CUDA 13.3 and newer. |
| `NVCC_PREPEND_FLAGS` | empty | Extra nvcc flags without editing the Dockerfile. |
| `FFMPEG_TAG`, `BUILD_JOBS`, `AVPLUMBER_REVISION` | `n8.1`, `4`, `workspace` | As in `Dockerfile`. |

## Tests

[Click-to-picture latency](../docs/latency.md) explains the browser measurement,
its initial baseline, and how to reproduce it. Protocol timings below only
measure acknowledgment and state settlement.

Pure layout, graph-construction, control, and TUI tests do not require a GPU:

```sh
python3 -m pytest -q demos/mixer/tests
```

With the mixer backend running, the protocol smoke test exercises the required
scenes and reports transition-command and settle latency:

```sh
python3 demos/mixer/smoke_test.py --port 7777 --wipe-file /path/on/mixer/host/wipe.mov
```

The runtime acceptance check requires the configured NVIDIA environment. It
must cover every layout, a disconnected input, Preview plus Cut/Fade/media
wipes, and the deterministic
`overlay_many_cuda` matrix in
`demos/cuda-overlay`.

Two standalone NVIDIA regressions use the numbered 640×360/60 fixtures from
`demos/mixer/tests/frame_codes.py` and a freshly built Python module:

```sh
python3 demos/mixer/tests/frame_codes.py /tmp/fixtures \
  --sources 2 --width 640 --height 360 --fps 60 --seconds 60
python3 demos/mixer/tests/check_transition_recovery.py \
  /tmp/fixtures/source-0.mp4 /tmp/fixtures/source-1.mp4
python3 demos/mixer/tests/check_rendered_interruptions.py \
  /tmp/fixtures/source-0.mp4 /tmp/fixtures/source-1.mp4 \
  --wipe-file /path/to/wipe.mov --wipe-seconds 2 --output /tmp/interruptions.json
```

Set `--wipe-seconds` to the clip's duration. Recovery checks target timeout,
invalid scenes, missing wipe media and cancelled callbacks. The interruption
suite covers all Cut/Fade/media-wipe pairs and both sides of the wipe midpoint,
comparing retained luma pixels against the last observed pre-command picture.
Chroma is not compared. These tests own their graphs; they do not control a
running demo. A test-only download makes pixel assertions possible.

## Implementation notes

The program pipeline stays on CUDA from hardware decode through NVENC; the
separate alpha-media branch uploads its software-decoded wipe frames.
Scene geometry is applied inside the two CUDA compositors. Sources feed both
slots through fanouts; a larger catalogue uses a router to select its visible
positions. The compositor resolves each frame's dimensions, crop and destination
rectangle without rebuilding a filter graph. Existing callers can still supply
explicit FFmpeg preprocessing graphs.

The compositor scaler uses bilinear interpolation over four neighbours. It does not widen its filter for downscaling, so fine patterns can alias.
Image-quality parity with FFmpeg has not been established.

For 16 inputs with Janus output and media wipes enabled, the optimized graph
has 122 nodes / 140 defined queues, down from 185 / 248. All FPS stages remain.
The compositors preserve source letterboxing without intermediate full-size
normalization frames. CPU/GPU savings and full frame-level equivalence are still
being compared; these counts alone do not establish a resource or latency
improvement.

For a demo screenshot, open the WebUI graph and choose **Expand / restore
graph**. The grouped overview shows inputs, both compositor slots, transitions,
media wipe and output, with bundled queues between them. Click a group to inspect
its nodes; the input group opens a picker for all 16 source chains. Breadcrumbs
show the current location; **← Back** moves one level up. Disable **Grouped
overview** to inspect the complete native graph. Pan and zoom preserve the layout; hover a queue label for its full
name. Displayed node and queue totals always describe the native graph.

Both `mixer_compositor` slots and the permanent `transition_cuda` filter
are warmed before the control server reports ready. Cut and Fade therefore
change runtime parameters without rebuilding their FFmpeg filter graphs. Media
Wipe uses a separate, predeclared graph. With the clip cache (the default) its
player and compositor run from startup and idle between wipes; a take arms them
in place and an interruption parks them, so wipe spam creates, starts and stops
nothing under the program. With `--wipe-cache-mb 0` the orchestrator starts the
decode graph for the selected clip and stops it after its tail has drained.

## Demo recording

The [15-second demonstration](https://amagimedia.github.io/avplumber/demos/mixer/docs/)
combines the host's encoded program output with a separate browser capture of
the real TUI. Direct mode remains enabled through Cut, Fade and transparent
media-file Wipe selections. The final silent H.264 MP4 is 1600×900 at 60 fps,
contains 900 frames and is 947,203 bytes (about 0.51 Mbit/s).

All 900 program frames have evenly spaced 1/60-second RTP timestamps, with no
packet gaps or consecutive identical decoded frames. The final MP4 also has
evenly spaced frame timestamps. Terminal updates come from the separate
browser capture; this recording is not a click-to-display latency measurement.
See the [latency measurements](latency.md) for the measurement method and
its published baseline.

The poster is the MP4's first frame. The grouped WebUI overview opens the
full ungrouped graph in a new tab on the demo page. Video and images are public
release assets; [media.sha256](media.sha256) records their checksums.
