# Portable L4 mixer

[Quick start](../../../../demos/mixer/README.md) · [Capacity](../../../../demos/mixer/docs/capacity.md#nvidia-l4-linear-and-cuarray-profiles) ·
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

Open `http://<host>:7681/setup/`, controls at `/`, outputs at `/wall`.
Allow the public proxy port and `JANUS_RTP_PORT_RANGE` (UDP 20000–20100 by
default). Set `MIXER_HTPASSWD` or use an authenticated reverse proxy.
The included proxy forwards HTTP and WebSocket paths.

The [current demo recording](https://amagimedia.github.io/avplumber/demos/mixer/docs/)
shows this profile at 192 inputs / 256 scenes / 26 outputs, 1080p25 SDR.
Its two viewers show Program/Preview and dirty Program with four keys. The
Program/Preview and Multiviewer outputs in that show use HEVC; use a browser
with HEVC WebRTC reception support to view them. The recording itself is H.264.

First start seeds [settings.json](settings.json) and generates media. Later
starts preserve saved settings; change an existing show through Setup.
Keep the writable media directory, including `.nv-cache`, across restarts.
Do not mount the generated show file separately: Setup replaces it atomically.

## Output codecs and bitrates

Setup selects codec, preset and bitrate per program/own AUX, with one shared
setting for extra AUX. SDR supports H.264/HEVC; HDR uses HEVC Main10.
Codec/preset changes recalculate capacity and restart the mixer/mountpoints.
HEVC playback requires browser support. Current defaults are in `settings.json`.

## Fresh-host validation

2026-10-05 deployment checks recorded 60 seconds with zero missed deadlines or
drops at 88 sources / 256 scenes / 17 outputs (HDR 4:2:2, 60 fps) and
192 / 256 / 26 (SDR25). These are short checks, not sustained-load guarantees;
the [capacity report](../../../../demos/mixer/docs/capacity.md) includes less successful sweeps.

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
