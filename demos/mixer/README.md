# Live mixer demo

**60 fps throughout: all 16 native 1920×1080 inputs → 1080×1920 portrait output at 60 fps.**
Input files and encoded output are 60 fps; per-source frame continuity is qualified below.
Measured on Tesla T4, 30 seconds per scene, with all sixteen sources active:

| Scene | GPU compute | NVDEC | NVENC | AVPlumber CPU |
| --- | ---: | ---: | ---: | ---: |
| **16-box** | **8.0%** | **76.7%** | **23.0%** | **0.58 cores** |
| Fullscreen | 7.0% | 76.5% | 22.2% | 0.53 cores |

[Samples and conditions](docs/runtime-load-1080p.json).

[![Watch the mixer: program output beside the TUI](https://amagimedia.github.io/avplumber/demos/mixer/docs/mixer-demo.jpg)](https://amagimedia.github.io/avplumber/demos/mixer/docs/)

[Watch the demo](https://amagimedia.github.io/avplumber/demos/mixer/docs/) · [MP4](https://github.com/amagimedia/avplumber/releases/download/mixer-demo-media-2026-09/mixer-demo.mp4) · [Full processing graph](https://amagimedia.github.io/avplumber/demos/graph.html?demo=mixer)

The 15-second, 0.95 MB recording shows Direct editing with Cut, Fade and
transparent media wipes, switching between fullscreen, 16-box, 8-box, 4-box
and 2-box. Program output is on the left and real terminal controls on the
right, in a 1600×900, 60 fps MP4. Click the first-frame preview to play it with
chapter buttons and inspect the web UI graph. This earlier recording uses 360p
sources; the current 1080p benchmark below was measured separately. The page is hosted on GitHub
Pages; media comes from public release assets, outside Git history.

## Run

Use a Linux NVIDIA host with hardware decode and NVENC. Follow the
[shared Docker/NVIDIA setup](../README.md); no company account, private runtime,
neural models or TensorRT is needed.

From the repository root, build the image and start the shared Janus preview:

```sh
docker build -f demos/mixer/Dockerfile -t avplumber-mixer:local .
docker compose --env-file demos/dmabuf-browser/.env.example \
  -f demos/dmabuf-browser/compose.yaml up -d --build janus janus-preview
```

Put your input clips and an optional alpha wipe in `media/`, then run:

```sh
docker run --rm --gpus all --network host \
  -v "$PWD/media:/media:ro,z" \
  avplumber-mixer:local \
  --input /media/camera-1.mp4 --input /media/camera-2.mp4 \
  --loop-inputs --fps 60 --janus-output \
  --janus-video-bitrate-kbps 8000 --remote-control-port 7777
```

Repeat `--input` for each source; the recording uses sixteen generated clips.
Open the output at <http://127.0.0.1:8080>. In another terminal:

```sh
python3 -m venv .venv-tui
.venv-tui/bin/python -m pip install -r demos/mixer/requirements.txt
.venv-tui/bin/python demos/mixer/tui.py --host 127.0.0.1 --port 7777 \
  --fade-duration 0.8 --wipe-file /media/wipe.mov
```

The wipe path is resolved by the mixer backend, including inside its container.
Pass the same clip to the mixer as `--wipe-file` (compose: `MIXER_WIPE_FILE`) and
it runs the wipe chain once, invisibly, at start, so the first wipe is as fast as
the following ones instead of paying for file open, decoder and GPU filter setup.
Use a clip with alpha, such as QTRLE/ARGB or ProRes 4444. The clip plays over the
program and the scene changes at its midpoint. The demo has no audio.

## Browser pages as sources

Any input can be a live page rendered by the
[DMA-BUF browser demo](../dmabuf-browser/README.md) instead of a clip:
`--input dmabuf://<window-id>` takes the window's DRM PRIME frames straight
into the compositor with no decoder, using the same chain as that demo. File
and browser inputs mix freely and share every layout, transition and control.

The compose override runs this mixer inside the DMA-BUF stack and opens the
pages itself. From the repository root:

```sh
cd demos/dmabuf-browser
cp .env.example .env
MIXER_SOURCE_COUNT=4 docker compose --env-file .env   -f compose.yaml -f compose.mixer.yaml up --build
```

Open <http://127.0.0.1:8080> and drive it with `tui.py` as above. `HTML_OVERLAY_URL`
selects the page (default: the bundled animation), `MIXER_SOURCE_COUNT` the
number of windows, `MIXER_SOURCE_WIDTH`/`MIXER_SOURCE_HEIGHT` their size.
Outside compose, the switches are `--dmabuf-open URL` (open the windows through
the browser's REST API, `--dmabuf-rest`), `--dmabuf-size WxH` and
`--dmabuf-socket-dir`.

Sixteen Singular.live pages on the Tesla T4 host (16 vCPU), 60 fps, all
sixteen windows painting at 60 fps with zero capture drops and the encoder at
60 fps, ten one-second samples after warm-up:

| Page size | GPU busy | GPU memory | Host CPU busy |
| --- | ---: | ---: | ---: |
| 480x270 | 33.7 % | 584 MiB | 52 % |
| 960x540 | 37.4 % | 1308 MiB | 60 % |
| 1280x720 | 40.8 % | 1394 MiB | 72 % |
| 1920x1080 | 51.1 % | 3585 MiB | 78 % |

The same sixteen clips decoded by NVDEC cost 2 % GPU; the browser rendering and
capture is the load. Each DMA-BUF allocation is imported once and copied per
frame; browser frames are converted from RGB to the NV12 canvas inside the
compositor's draw pass, so video and browser sources mix on one canvas without
extra passes.

## Generated input size and FPS

Generate sixteen native-resolution test clips (requires NumPy and FFmpeg with
`libx264`; generation does not require a GPU):

```sh
python3 demos/mixer/tests/frame_codes.py media \
  --sources 16 --width 1920 --height 1080 --fps 60 --seconds 30
```

`--width`, `--height` and `--fps` control the generated source files. Defaults
are **1920×1080 at 60 fps**; `--ffmpeg` selects the FFmpeg executable. Existing
files are not overwritten. For example, use `--width 1280 --height 720 --fps 30`
for a 720p30 input test. Supplied video files retain their encoded dimensions.
The mixer's separate `--fps` option sets the processing/output rate; keep it at
60 when comparing different source sizes.

To run all sixteen generated inputs from Bash:

```sh
demo_inputs=()
for i in {0..15}; do demo_inputs+=(--input "/media/source-$i.mp4"); done
docker run --rm --gpus all --network host \
  -v "$PWD/media:/media:ro,z" avplumber-mixer:local \
  "${demo_inputs[@]}" --loop-inputs --fps 60 --janus-output \
  --janus-video-bitrate-kbps 8000 --remote-control-port 7777
```

## Controls

| Control | Result |
| --- | --- |
| Scene, layout and Page | Prepare Preview; in Direct mode, put the selection on air |
| Cut / `c` | Immediate take |
| Fade / `f` | Blend for the selected duration |
| Media Wipe / `w` | Play the selected transparent clip |
| Direct / `t` | Use the current Cut, Fade or Media Wipe for each selection |
| `1`–`9` | Select one of the first nine scenes |
| `r` / `q` | Reconnect / quit the TUI |

A new take can interrupt a transition, starting from the current mixer picture.
Sources retain their positions if one stalls. Output is 1080×1920 portrait;
inputs keep their aspect ratio. More sources than cells create additional pages.

## Processing graph

<a href="https://amagimedia.github.io/avplumber/demos/graph.html?demo=mixer" target="_blank" rel="noopener noreferrer"><img src="https://amagimedia.github.io/avplumber/demos/mixer/docs/mixer-graph-grouped.png" alt="Grouped mixer graph: inputs, two compositor slots, transitions, media wipe and output. Click for the full ungrouped graph." width="640"></a>

**122 nodes / 140 queues**, grouped into six blocks. Click the overview for the
full-resolution ungrouped graph in an HTML viewer with Fit and zoom controls.
On the [demo page](https://amagimedia.github.io/avplumber/demos/mixer/docs/#graph)
it opens in a new tab. In the live WebUI, click a group to enter it and use
**← Back** to return. Static edge colours and fill markers show sampled flow
without animated edges.

## Timing and measured load

Both compositor slots use the native timing shared with the
[DMA-BUF demo](../dmabuf-browser/README.md). `--mixer-latency-ms` sets the jitter
budget; the default is two output frame periods (about 33 ms at 60 fps).
Late sources repeat their previous image and discard overdue frames to recover.
This budget is not the full click-to-display latency.

With **sixteen native 1920×1080@60 H.264 inputs**, a Tesla T4 / 16-vCPU
host measured the following over 30 seconds in the steady 16-box scene.
Output was **1080×1920@60**, H.264 NVENC at 8 Mbit/s; the two-frame jitter
budget and automatic GPU clocks were retained.

| Measurement | Average |
| --- | ---: |
| GPU compute utilization | **8.0%** (min/max: 8%/8%) |
| NVDEC decoder utilization | **76.7%** (min/max: 76%/79%) |
| NVENC encoder utilization | **23.0%** |
| GPU memory activity | **16.0%** |
| AVPlumber CPU | **0.58 cores** |
| Whole-device GPU memory | **1,958 MiB** |

These are separate engine/activity measurements; **8% does not mean the entire
GPU is only 8% occupied**. Decoder capacity matters for scaling source count.
The input clips contain native-resolution animated test patterns and visible
frame IDs. No fixture generation or DMA-BUF browser workload ran during the
measurement. See [samples and exact conditions](docs/runtime-load-1080p.json).
A separate 11.4-second decoded-output check had regular 60 fps RTP timestamps
and no packet loss, but source 14 repeated **33.43%** and skipped **33.28%**
of source frames (229 repeats / 228 skipped frames across 685 intervals).
The other fifteen sources had **0%** repeats and skips. An earlier short
capture also showed smaller continuity errors on other sources. This is an
unresolved per-source timing issue: the load figures do **not** establish
frame-perfect delivery. See [both pixel-level checks](docs/continuity-1080p.json).

The earlier **4%** measurement used **640×360 inputs** and is retained only as
historical data in [the earlier report](docs/runtime-load.json).

Click latency was measured separately with the earlier 360p inputs: eleven
warm Cut trials had **182.3 ms minimum, 237.7 ms mean, 188.8 ms median and
503.9 ms maximum**. The viewer ran on the same host; these measurements include
browser presentation and probe overhead. They are not a new 1080p latency
measurement. See [raw samples and method](docs/latency.md).
Content, resolution, transitions and viewer/network conditions affect results.

For output files, Janus settings, keyboard details, layouts and tests, see the
[full reference](docs/guide.md). GPU program frames remain on the GPU; the
separate alpha-media wipe branch decodes on CPU and uploads its frames.
