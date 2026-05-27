# Auto Mixer Docker Runtime

This directory contains the Docker Compose stack for the auto mixer demo/runtime.
Janus, the WebRTC preview page, the headless Wayland compositor, and the
DMA-BUF HTML browser sidecar are support services for the auto mixer, so their
Dockerfiles and config live here.

Copy `.env.example` to `.env` and fill in host-local paths before building.
The Compose file intentionally has no local placeholder bind-mount defaults for
media, model, cache, or TensorRT paths; missing variables should fail fast
instead of creating empty directories under this repo.
`TENSORRT_CONTEXT` points at the directory containing the TensorRT archive;
`TENSORRT_ARCHIVE` is the archive filename inside that directory.
Set `JANUS_HOST_IP` to the address browsers use to reach the Docker host when
using the Janus preview from another machine. Do not leave it at `127.0.0.1`
for VPN/browser preview; WebRTC sessions may connect but will not receive media.
Set `MEDIA_INPUT_DIR` to the host directory containing input media for
file-based demo runs. Inside the container the directory is mounted at
`/media-inputs`.
Set `MEDIA_WIPE_DIR` to the host directory containing media wipe files.
Inside the container the directory is mounted at `/media-wipes`; a relative
`--auto-switch-wipe-file` is resolved under `--media-wipe-dir`.
Set `REMOTE_CONTROL_PORT` for the mixer/TUI control socket. The compose default
is `7777`, and the auto-mixer container exports it as `AVP_REMOTE_CONTROL_PORT`
so Docker runs start with TUI and auto-switch controls enabled by default.
Set `WEBUI_PORT` for the avplumber Web UI. The compose default is `22222`.
The auto-mixer container registers itself with `http://127.0.0.1:$WEBUI_PORT`
by default, using `AUTO_MIXER_INSTANCE_NAME` and `AUTO_MIXER_LOGFILE`.
Set `FACE_ENGINE_CACHE_DIR` to a persistent host directory for the generated
TensorRT face engine and timing cache. Inside the container it is mounted at
`/models-cache`, where `AVP_FACE_ENGINE` defaults to
`/models-cache/face.fp16.plan`. Reusing this directory prevents rebuilding the
plan on every container run.
Set `HTML_OVERLAY_URL` when the browser sidecar should auto-open an HTML
overlay. The sample value renders a 9:16 Singular overlay:

```sh
HTML_OVERLAY_URL=https://app.singular.live/output/6W76ei5ZNekKkYhe8nw5o8/Output?aspect=9:16
```
Set `HTML_OVERLAY_DRM_DEVICE` to the render device visible inside the
auto-mixer container. The default is `/dev/dri/renderD128`, and the compose file
passes `/dev/dri` into the container so the auto mixer can create the same DRM
source hwaccel context as the working host graph.

Start support services:

```sh
docker compose up -d janus janus-preview web-ui wayland dma-browser
```

Or use the stack wrapper to start the support services, wait for health checks,
and launch the auto mixer as a stable named container:

```sh
./mixer-stack.sh start -- \
  --inputs /media-inputs/<input-0> /media-inputs/<input-1> \
  --janus-output \
  --janus-host 127.0.0.1 \
  --media-wipe-dir /media-wipes \
  --html-overlay-url "$HTML_OVERLAY_URL"
```

The wrapper reads `.env`, uses `AUTO_MIXER_CONTAINER` as the container name
(`avp-auto-mixer` by default), and also accepts auto-mixer arguments from
`AUTO_MIXER_ARGS` when no arguments are passed after `--`. Stop the whole stack
with:

```sh
./mixer-stack.sh stop
```

Use `./mixer-stack.sh status` for service state, `./mixer-stack.sh logs
auto-mixer` for the runtime container, and `./mixer-stack.sh restart -- ...` to
recreate the mixer with new arguments. Add `--build` to `start` or `restart`
when the images should be rebuilt before launch.

For deployment hosts, `rebuild-restart-stack.sh` can sync a configurable branch,
rebuild the mixer images, recreate the stack, and prune old project containers,
images, and builder cache:

```sh
./rebuild-restart-stack.sh --branch <branch>
```

Use `AVP_GIT_BRANCH=<branch>` instead of `--branch` when the branch belongs in
the host-local `.env`. Add `--discard-local-changes` for disposable deployment
worktrees, and `--no-prune` only when old images or build cache should be kept.

Open the Web UI from the VPN/browser host at `http://<host-ip>:22222/` unless
`WEBUI_PORT` is changed. It uses the same main auto-mixer control port as the
TUI, so the auto-mixer must have `REMOTE_CONTROL_PORT` enabled.

Run the auto mixer with explicit CLI arguments. Pass `--html-overlay-url` only
when the overlay is required; when omitted, the auto mixer skips the HTML
overlay graph and does not consume `/tmp/dma-page/overlay.sock`. When the URL is
provided, the overlay branch starts enabled and consumes the sidecar DMA-BUF
socket by default. The Docker entrypoint appends Web UI registration arguments
from env unless they are supplied explicitly.

```sh
set -a
. ./.env
set +a

docker compose build auto-mixer dma-browser
docker compose run --rm auto-mixer \
  --inputs /media-inputs/<input-0> /media-inputs/<input-1> \
  --janus-output \
  --janus-host 127.0.0.1 \
  --media-wipe-dir /media-wipes \
  --html-overlay-url "$HTML_OVERLAY_URL"
```

Six-camera demo runs should pass the media files explicitly in the order used by
the role-index options. The example below uses generic camera filenames; adapt
the paths and indexes for each input set. The role rules are not tied to names:
`--program-audio-input` selects the audio source, `--special-speaker-index`
gives one input a dB-margin rule, `--vad-only-priority-speaker-index` lets one
input win from audio VAD without visual speech, and `--static-face-crop-input`
uses a fixed face crop for that input.

```sh
MEDIA_INPUT_DIR=/path/to/media \
JANUS_HOST_IP=<host-ip> \
docker compose run --rm auto-mixer \
  --inputs \
    /media-inputs/cam0.ts \
    /media-inputs/cam1.ts \
    /media-inputs/cam2.ts \
    /media-inputs/cam3.ts \
    /media-inputs/cam4.ts \
    /media-inputs/cam5.ts \
  --program-audio-input 0 \
  --special-speaker-index 0 \
  --special-speaker-margin-db 3.0 \
  --vad-only-priority-speaker-index 1 \
  --static-face-crop-input 2 \
  --auto-switch-transition wipe \
  --auto-switch-wipe-file alpha_1773690550_12480_qtrle_argb.mov \
  --janus-output \
  --janus-host 127.0.0.1 \
  --media-wipe-dir /media-wipes \
  --html-overlay-url "$HTML_OVERLAY_URL"
```

The `wayland` service exposes `WAYLAND_DISPLAY=wayland-1` through the shared
`wayland-runtime` volume. The `dma-browser` service shares that display and
writes the overlay DMA-BUF socket to `/tmp/dma-page/overlay.sock` through the
`dma-browser-sockets` volume, which the auto mixer mounts at the same path.
The browser REST API is available on `http://127.0.0.1:9009/status` by default.

For the VAAPI/NVIDIA path the browser defaults to a retained DMA-BUF pool of
10 frames (`DMA_BROWSER_DMABUF_POOL_SIZE=10`) and enables the Chromium NVIDIA
sync checks listed in `DMA_BROWSER_CHROMIUM_EXTRA_FEATURES`.

The current image still builds `face.fp16.plan` at auto-mixer startup when the
cache directory is empty. The intended next step is to move that generation
into the Docker build so a TensorRT plan failure fails image build, not runtime
startup.
