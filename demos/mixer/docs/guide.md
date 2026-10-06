# Running the mixer

[Quick start](../README.md) · [Configuration](config.md) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)

## Start the mixer backend

Use an NVIDIA host and a Python module built against the same patched FFmpeg
as avplumber, with CUDA/NVCC and DRM/GL for browser sources.

```sh
python3 demos/mixer/mixer.py --config mixer.json
```

The config must declare outputs, or supply `--output <path>` / `--janus-output`.
For a simple file-only show:

```sh
python3 demos/mixer/mixer.py \
  --input <input-1> --input <input-2> --loop-inputs \
  --output program.mp4 --fps 30 --remote-control-port 7777
```

Only this `--input` mode uses fixed 1080×1920 fullscreen and 2/4/8/16-box
layouts. `--config` supplies its own canvas and scenes. Use `--output-format`
for an ambiguous target; `--codec` and `--bitrate` select the NVENC output.
Run `--help` for current defaults and the full option list.

## Controls

The Compose stack starts the web UI. With a standalone backend:

```sh
python3 demos/mixer/webui.py --host 127.0.0.1 --port 7777 --http-port 7681
```

For the optional terminal UI:

```sh
python3 -m venv .venv-tui
.venv-tui/bin/pip install -r demos/mixer/requirements.txt
.venv-tui/bin/python demos/mixer/tui.py --host 127.0.0.1 --port 7777
```

TUI keys: `1`–`9` select scenes, `c` cuts, `f` fades, `w` wipes, `t` toggles
Direct, `r` reconnects. Wipe paths are on the backend host. Clips must retain
alpha and cover the scene at their midpoint. `--wipe-cache-mb` controls the
GPU cache; zero decodes each take. Startup fails on an incomplete cached clip.

## Janus output

`--janus-output` sends video RTP to `127.0.0.1:5004`, RTCP to port 5005 by
default. Set `--janus-host` / `--janus-video-port` for another destination.
Each rendition needs its own port pair and a matching Streaming mountpoint.
Setup manages mountpoints through `--janus-api`; the standalone backend does not.
PLI/FIR feedback requests keyframes, limited by `--keyframe-min-interval-ms`.

## Docker

Compose builds the mixer and imports browser/Wayland/Janus service definitions
from the shared [DMA-BUF stack](../../../docker-compose/dmabuf/compose.yaml).

### The two images

| Dockerfile | Stack | Host |
| --- | --- | --- |
| `Dockerfile.fedora44` (default) | Fedora 44 / CUDA 13.4 / FFmpeg 8.1 | R615+ driver |
| `Dockerfile` | Ubuntu 22.04 / CUDA 11.7 | Older compatible NVIDIA driver |

Select the second with `MIXER_DOCKERFILE=Dockerfile`. The [L4 preset](../deploy/l4/README.md)
pins its own FFmpeg build. Keep `media/.nv-cache` to reuse driver-compiled kernels.
TensorRT is optional (`WITH_TENSORRT=1` and the `tensorrt_url` build secret in the
Fedora image); the manual mixer needs no neural models.

## Behind a reverse proxy

Serve controls at `/` and the preview server at `/preview/`, stripping that
prefix. Set `MIXER_PREVIEW_BASE=/preview/` (or `webui.py --preview-base /preview/`).
Forward WebSocket upgrades for the preview's Janus path as well as normal HTTP.
The [L4 proxy template](../deploy/l4/nginx.conf.template) is a complete example.
WebRTC also needs the configured UDP range reachable from the viewer.

## Tests

```sh
python3 -m pytest -q demos/mixer/tests
```

NVIDIA-dependent cases require the configured GPU host. Against a running backend:

```sh
python3 demos/mixer/smoke_test.py --port 7777 --wipe-file <alpha-clip>
python3 demos/mixer/tests/cut_spam.py --url http://127.0.0.1:7681
```

These change Program. See [latency](latency.md) for the cut probe and browser
measurement, and [capacity](capacity.md) for the limits of recorded results.
