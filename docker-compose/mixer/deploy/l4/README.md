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
`.env` for other workloads. The browser image runs the patched Electron with a
fixed, prewarmed pool of nine export buffers per window, and the mixer copies each
frame once on import and returns it at once (see the
[browser configuration](../../../../deps/dma-browser/README.md)). Set
`DMA_BROWSER_CHROMIUM_EXTRA_FEATURES` to change the pool size; changing it, or
`AVP_BROWSER_COPY_ON_IMPORT`, requires recreating the browser and mixer containers.

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

## GPU time-sharing collapse (measured 2026-10-15)

With 40 moving browser windows the show can tip into a state where every compositor's
GPU wait rises from 2 ms to 33-66 ms while no engine is saturated (SM about 60 %, NVDEC
about 75 %, NVENC about 40 %) and the host has idle CPU. Program and AUX then miss a
third of their deadlines and NVENC's average latency goes from 8 ms to about 45 ms. The
state is entered under perturbation (fade bursts, browser worker restarts, heavy host
activity such as Docker log scans) and does not leave on its own: the mixer's CUDA
context and Chromium's GL contexts time-share the GPU and the CUDA backlog never drains.

What resets it, measured: stopping Chromium's GPU processes (`electron --type=gpu-process`)
for 2-3 seconds with SIGSTOP/SIGCONT; misses fall to zero within seconds and stay there.
What did not matter: viewer PLIs (blocked for 60 s, no change), the fade implementation,
source kind on program, DSK keys. Two GPU processes of 20 windows each were worse than
four of ten (two clean starts failed).

Operating rules that follow: keep the browser container's CPU weight below the mixer's
(`cpu_shares: 512` above; Chromium raises its GPU processes to nice -8 on its own), avoid
CPU-bound work on the host while the show runs, and if an operator-side watchdog is used,
have it act only in the setup's `running` phase, because a freeze during the 60 s
clean-start window makes that start fail. The clean-start gate restarts its 10 s window on
missed deadlines; output drops are reported but do not restart it, because in the collapsed
state every bus drops about 15 frames a minute and a start would loop until the setup manager
gave up. A start that begins inside the collapsed state still needs the GL side paused once
before its window (an operator-side watchdog that freezes Electron's GPU processes for 2.5 s
about 25 s after a new mixer process appears made every such start settle on 2026-10-15).
An output node now gives up only after its writes have failed for 3 s; a burst of failed UDP
writes after a stall used to stop the whole show through the program outputs' panic policy.

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
