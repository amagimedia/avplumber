# Live mixer demo

[![Watch the mixer: program output beside the TUI](https://github.com/amagimedia/avplumber/releases/download/mixer-demo-media-2026-09/mixer-demo.jpg)](https://amagimedia.github.io/avplumber/demos/mixer/docs/)

[Watch the demo](https://amagimedia.github.io/avplumber/demos/mixer/docs/) · [MP4](https://github.com/amagimedia/avplumber/releases/download/mixer-demo-media-2026-09/mixer-demo.mp4) · [Full processing graph](https://github.com/amagimedia/avplumber/releases/download/mixer-demo-media-2026-09/mixer-graph-ungrouped.png)

The 15-second, 0.95 MB recording shows Direct editing with Cut, Fade and
transparent media wipes, switching between fullscreen, 16-box, 8-box, 4-box
and 2-box. Program output is on the left and real terminal controls on the
right, in a 1600×900, 60 fps MP4. Click the first-frame preview to play it with
chapter buttons and inspect the web UI graph. The page is hosted on GitHub
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
python3 -m pip install -r demos/mixer/requirements.txt
python3 demos/mixer/tui.py --host 127.0.0.1 --port 7777 \
  --fade-duration 0.8 --wipe-file /media/wipe.mov
```

The wipe path is resolved by the mixer backend, including inside its container.
Use a clip with alpha, such as QTRLE/ARGB or ProRes 4444. The clip plays over the
program and the scene changes at its midpoint. The demo has no audio.

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

<a href="https://github.com/amagimedia/avplumber/releases/download/mixer-demo-media-2026-09/mixer-graph-ungrouped.png" target="_blank" rel="noopener noreferrer"><img src="https://github.com/amagimedia/avplumber/releases/download/mixer-demo-media-2026-09/mixer-graph-grouped.png" alt="Grouped mixer graph: inputs, two compositor slots, transitions, media wipe and output. Click for the full ungrouped graph." width="640"></a>

**122 nodes / 140 queues**, grouped into six blocks. Click the overview for the
full-resolution ungrouped graph. On the [demo page](https://amagimedia.github.io/avplumber/demos/mixer/docs/#graph)
it opens in a new tab. In the live WebUI, click a group to enter it and use
**← Back** to return. Static edge colours and fill markers show sampled flow
without animated edges.

## Timing and measured load

Both compositor slots use the native timing shared with the
[DMA-BUF demo](../dmabuf-browser/README.md). `--mixer-latency-ms` sets the jitter
budget; the default is two output frame periods (about 33 ms at 60 fps).
Late sources repeat their previous image and discard overdue frames to recover.
This budget is not the full click-to-display latency.

A 20-sample, steady 16-box run on a Tesla T4 / 16-vCPU host used **4% GPU,
0.50 CPU cores and 509 MiB GPU memory**, with sixteen generated 640×360 H.264
sources and 1080×1920@60 output. This was an exploratory automatic-clock sample,
not a capacity guarantee or matched before/after benchmark. Page/file content,
resolution and transitions change load.

[Click-to-picture measurements](docs/latency.md) distinguish displayed-frame
latency from command acknowledgment; the published initial baseline predates
the current graph optimization. The demonstration video is not a latency test.

For output files, Janus settings, keyboard details, layouts and tests, see the
[full reference](docs/guide.md). GPU program frames remain on the GPU; the
separate alpha-media wipe branch decodes on CPU and uploads its frames.
