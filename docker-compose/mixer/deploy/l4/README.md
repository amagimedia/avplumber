# Portable L4 mixer

[Quick start](../../../../demos/mixer/README.md) · [Capacity](../../../../demos/mixer/docs/capacity.md#nvidia-l4-16-vcpu-nvidia_l4_cuarray) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)

For a GeForce desktop, the [desktop launcher](../../../../demos/mixer/README.md#geforce-desktops)
uses this stack with a first show sized to the GPU and its own media directory.

The L4 preset requires a Linux L4 host with working NVIDIA DRM/GBM/EGL, R615+ driver,
Docker Compose and NVIDIA Container Toolkit. The measured host has 16 vCPUs,
64 GiB RAM and a 256 GB disk. See [GCP provisioning](../gcp/README.md).

```sh
cp docker-compose/mixer/deploy/l4/.env.example docker-compose/mixer/deploy/l4/.env
# Set destination addresses and optional authentication in .env.
docker-compose/mixer/deploy/l4/stack.sh build
docker-compose/mixer/deploy/l4/stack.sh smoke
docker-compose/mixer/deploy/l4/stack.sh up -d
docker-compose/mixer/deploy/l4/stack.sh logs -f mixer
```

The same CUDA 13.4 mixer image runs on R615 and R620 hosts. The smoke check
compares raw frames uploaded from page-locked host RAM byte for byte, then
exercises HEVC encoding, decoding and CUDA composition. The raw loops in the
show stay in pinned host RAM and require a successful CUDA host allocation.

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
`.env` for other workloads. The browser image runs stock Electron with Chromium's
ordinary capture, and the mixer samples each captured buffer in place (see the
[browser configuration](../../../../deps/dma-browser/README.md)).

The image is built with MXL (`WITH_MXL=1`, set `WITH_MXL=0` in `.env` to leave it out), so Setup
offers MXL sources among the HW uploads, up to the profile's `mxl_sources`. The mixer container
runs their writers beside the mixer and holds their flows in its own `/dev/shm` (`shm_size`).

## Optional NDI input

NDI® is a registered trademark of Vizrt NDI AB ([ndi.video](https://ndi.video/)). The NDI SDK and its runtime are proprietary and licensed by Vizrt under the SDK's own terms, which
`setup-ndi.sh` accepts for you: an image built with `ENABLE_NDI=1` contains the NDI runtime and must not
be pushed to a public registry or redistributed.

Set `ENABLE_NDI=1` in the L4 stack's `.env`. On the Ubuntu L4 host run:

```sh
sudo docker-compose/mixer/deploy/l4/setup-ndi.sh
docker-compose/mixer/deploy/l4/stack.sh build mixer
docker-compose/mixer/deploy/l4/stack.sh run --rm --no-deps --entrypoint ffmpeg mixer -hide_banner -devices
```

Enabling NDI installs the checksum-pinned official Linux SDK 6.4 under
`/opt/ndi/6.4` and accepts its SDK license. The mixer image builds FFmpeg
with the AVP binary and Python module and carries the NDI runtime library.
Host networking and an Avahi-only D-Bus proxy provide local source discovery.
For a source outside the local discovery domain, set `extra_ips` to the
sender address. An AVP `input` node takes the discovered source name as
`url`, `libndi_newtek` as `format`, and device options in `options`.

To verify AVP against a live sender with audio and video:

```sh
docker-compose/mixer/deploy/l4/stack.sh run --rm --no-deps --entrypoint ffmpeg mixer \
  -hide_banner -f libndi_newtek -find_sources 1 -wait_sources 3 -i dummy
docker-compose/mixer/deploy/l4/stack.sh run --rm --no-deps --entrypoint python3 \
  -v "$PWD/tests/smoke_ndi_input.py:/tmp/smoke_ndi_input.py:ro" mixer \
  /tmp/smoke_ndi_input.py '<discovered source name>'
```

Open `http://<host>:7681/setup/`, controls at `/`, outputs at `/wall`.
The controls' **Graph** link opens the read-only Sigma.js viewer at `/graph/`.
If viewers use a DNS name instead of the configured public address, set
`WEBUI_ALLOWED_HOSTS` to that name (comma-separated for multiple names).
The graph backend remains on loopback; the proxy serves its page and WebSocket.
Each source count is editable, including zero. Count edits recalculate the total
without refilling other types; editing **Unique sources** restores the balanced
proportions. Per-type controls enforce the remaining total, shared NVDEC, HDR
surface, browser/DSK and weighted upload budgets. Rate or format changes reduce
custom counts where necessary, keeping disabled types at zero.
Allow the public proxy port and `JANUS_RTP_PORT_RANGE` (UDP 20000–20100 by
default). On a cloud VM keep `JANUS_NAT_1_1="auto"` (`.env.example`): Janus and the
graph viewer read the public address from the instance metadata at every start, so
a new ephemeral address after a stop and start needs no change. Set `MIXER_HTPASSWD` or use an authenticated reverse proxy.
The included proxy forwards HTTP and WebSocket paths.

The [current demo recording](https://amagimedia.github.io/avplumber/demos/mixer/docs/)
shows this profile at 192 inputs / 256 scenes / 24 outputs, 1080p30 SDR.
Its two viewers show Program/Preview and dirty Program with four keys. The
Program/Preview and Multiviewer outputs in that show use HEVC; use a browser
with HEVC WebRTC reception support to view them. The recording itself is H.264.

First start seeds [settings.json](settings.json), the show of that recording, and generates media. Later
starts preserve saved settings; change an existing show through Setup.
Keep the writable media directory, including `.nv-cache`, across restarts.
Do not mount the generated show file separately: Setup replaces it atomically.

The image carries FFmpeg's CUDA filters as machine code for sm_75/80/86/89, so an
L4 or T4 host starts without JIT-compiling them; `.nv-cache` then only serves GPUs
newer than sm_89. The `dma-browser-cache` volume keeps each browser worker's
compiled shader programs across container recreation. Keep GPU persistence mode on
(`sudo systemctl enable --now nvidia-persistenced`); `stack.sh up` warns when it is off.

## GPU time-sharing collapse (2026-10-15)

With 40 moving browser windows the show could tip into a state where every compositor's
GPU wait rose from 2 ms to 33-66 ms while no engine was saturated: Program and AUX missed a
third of their deadlines and NVENC's average latency went from 8 ms to about 45 ms, until
something paused Chromium. The mixer was then copying every browser frame on import. With
the mixer sampling Chromium's capture buffers in place, the only path now, the same show ran
clean with the browser uncapped, through hours of takes, fades and DSK keys.

The host keeps a long compute time-slice (`nvidia-smi compute-policy --set-timeslice=3`), with
which the collapsed state ended on its own instead of persisting. The policy resets on every
boot: `gcp/host-setup.sh` sets it on GCE, and `stack.sh up` warns when it is not set.

Operating rules: keep the browser container's CPU weight below the mixer's (`cpu_shares: 512`
above; Chromium raises its GPU processes to nice -8 on its own), avoid CPU-bound work on the
host while the show runs, and do not pause containers during the 60 s clean-start window,
because a freeze there makes that start fail. The clean-start gate restarts its 10 s window
on missed deadlines; output drops are reported but do not restart it. An output node counts a full send queue (ENOMEM from FFmpeg's paced UDP
sender) as a dropped packet and gives up only after other write errors persist for 3 s.

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
