# Portable L4 mixer

[Quick start](../../../../demos/mixer/README.md) · [Capacity](../../../../demos/mixer/docs/capacity.md#nvidia-l4-16-vcpu-nvidia_l4_cuarray) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)

Requires a Linux L4 host with working NVIDIA DRM/GBM/EGL, R615+ driver,
Docker Compose and NVIDIA Container Toolkit. The measured host has 16 vCPUs,
64 GiB RAM and a 256 GB disk. See [GCP provisioning](../gcp/README.md).

```sh
cp docker-compose/mixer/deploy/l4/.env.example docker-compose/mixer/deploy/l4/.env
# Set destination addresses and optional authentication in .env.
docker-compose/mixer/deploy/l4/stack.sh build
docker-compose/mixer/deploy/l4/stack.sh up -d
docker-compose/mixer/deploy/l4/stack.sh logs -f mixer
```

`stack.sh` pins FFmpeg from [bases.env](../../../../deps/ffmpeg/9/bases.env)
and selects `nvidia_l4_cuarray`. Browser services come from the shared
[DMA-BUF integration](../../../../doc/dmabuf.md); no other demo is required.

The mixer uses 32 CUDA compute work queues and eight copy work queues on L4.
This reduces serialization between independent source and compositor streams under
moving-browser load. Override `CUDA_DEVICE_MAX_CONNECTIONS` or
`CUDA_DEVICE_MAX_COPY_CONNECTIONS` in `.env` to compare workloads; changing them
requires recreating the mixer container. These settings do not change the browser
capture pool or the mixer's frame deadline.

Browser workers hold up to ten windows each, with frame staggering enabled.
Setup caps browser windows at 40 for 25/30/50 fps and 30 for 60 fps, including
DSKs. Changing the frame rate recalculates the allocation on the page and server.
All AUX outputs default to `p1`, including the initial deployment settings;
explicit saved encode choices remain editable and can be reset with Defaults.
Override `DMA_BROWSER_WINDOWS_PER_PROCESS` and `DMA_BROWSER_STAGGER_FRAMES` in
`.env` for other workloads. For the optional patched exporter, `.env.example`
also shows how to select a fixed nine-buffer pool and a single asynchronous
CUDA copy on import. That option requires the version-2 browser build described
in the [browser configuration](../../../../deps/dma-browser/README.md).

Open `http://<host>:7681/setup/`, controls at `/`, outputs at `/wall`.
Each source count is editable, including zero. Count edits recalculate the total
without refilling other types; editing **Unique sources** restores the balanced
proportions. Per-type controls enforce the remaining total, shared NVDEC, HDR
surface, browser/DSK and weighted upload budgets. Rate or format changes reduce
custom counts where necessary, keeping disabled types at zero.
Allow the public proxy port and `JANUS_RTP_PORT_RANGE` (UDP 20000–20100 by
default). Set `MIXER_HTPASSWD` or use an authenticated reverse proxy.
The included proxy forwards HTTP and WebSocket paths.

The [current demo recording](https://amagimedia.github.io/avplumber/demos/mixer/docs/)
shows this profile at 192 inputs / 256 scenes / 24 outputs, 1080p30 SDR.
Its two viewers show Program/Preview and dirty Program with four keys. The
Program/Preview and Multiviewer outputs in that show use HEVC; use a browser
with HEVC WebRTC reception support to view them. The recording itself is H.264.

First start seeds [settings.json](settings.json) and generates media. Later
starts preserve saved settings; change an existing show through Setup.
Keep the writable media directory, including `.nv-cache`, across restarts.
Do not mount the generated show file separately: Setup replaces it atomically.

The image carries FFmpeg's CUDA filters as machine code for sm_75/80/86/89, so an
L4 or T4 host starts without JIT-compiling them; `.nv-cache` then only serves GPUs
newer than sm_89. The `dma-browser-cache` volume keeps each browser worker's
compiled shader programs across container recreation. Keep GPU persistence mode on
(`sudo systemctl enable --now nvidia-persistenced`); `stack.sh up` warns when it is off.

## Output codecs and bitrates

Setup selects codec, preset and bitrate per program/own AUX, with one shared
setting for extra AUX. SDR supports H.264/HEVC; HDR uses HEVC Main10.
Codec/preset changes recalculate capacity and restart the mixer/mountpoints.
HEVC playback requires browser support. Current defaults are in `settings.json`.

## Fresh-host validation

Deployment checks recorded zero missed deadlines and drops at 192 sources / 256 scenes
at 1080p25 (26 outputs) and 1080p30 (24 outputs), and at 88 sources / 17 outputs in
HDR 4:2:2 at 60 fps. See the [capacity page](../../../../demos/mixer/docs/capacity.md).

## Move an instance

Keep source revision, image release and media together. Set destination choices
in the untracked `.env`; `MIXER_MEDIA_DIR` selects the host media directory.
Stop cleanly before copying media, or start empty to use packaged defaults.

```sh
docker save $(docker-compose/mixer/deploy/l4/stack.sh config --images) | gzip > mixer-images.tar.gz
# Transfer images, source and optional media to the destination.
docker load < mixer-images.tar.gz
docker-compose/mixer/deploy/l4/stack.sh up -d --no-build
```

Use the same `MIXER_RELEASE`; alternatively push/pull with `MIXER_IMAGE_PREFIX`.
Verify Setup reaches `running`, every wall stream advances, WebSocket reconnection
works, and cached SDR/HDR and FPS changes recover. Measure fresh deadline/drop deltas.
