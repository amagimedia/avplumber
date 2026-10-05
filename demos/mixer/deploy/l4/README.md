# Portable L4 mixer

This preset packages the CUarray demo without changes inside a running container.
It builds the pinned FFmpeg revision and all patches listed in
[`deps/ffmpeg/9/bases.env`](../../../../deps/ffmpeg/9/bases.env), then avplumber,
Janus, the browser capture services, preview player and graph UI. The proxy serves
the mixer and player on one origin and forwards both WebSocket paths.

Use a Linux x86-64 host with an L4, 16 vCPUs, 64 GiB RAM, a 256 GB balanced
persistent disk and NVIDIA driver R615 or newer,
Docker Compose and the NVIDIA Container Toolkit. Browser capture needs working
NVIDIA DRM/GBM/EGL and `/dev/dri`; the [host setup](../../../README.md#nvidia-host-setup)
and [GCP host scripts](../gcp/README.md) cover provisioning. A different GPU needs
its own measured capacity profile.

From a recursive checkout of the repository:

```sh
cp demos/mixer/deploy/l4/.env.example demos/mixer/deploy/l4/.env
# Edit the interface/public addresses in .env.
demos/mixer/deploy/l4/stack.sh build
demos/mixer/deploy/l4/stack.sh up -d
demos/mixer/deploy/l4/stack.sh logs -f mixer
```

Open `http://<host>:7681/setup/`, the controls at `/`, or all outputs at `/wall`.
Set `MIXER_HTPASSWD` to an existing password file to retain the demo's HTTP Basic
authentication, or use your authenticated reverse proxy. The password file stays
outside the repository. WebRTC uses the configured
`JANUS_RTP_PORT_RANGE` (UDP 20000–20100 by default); the advertised address must
match the destination host. The graph UI uses TCP 22222. Internal service ports
are 17681 for Setup, 18080 for the preview server, 8088/8188 for Janus and 9009
for browser control.

The first start seeds `media/demo.json` from [settings.json](settings.json), then
generates missing clips. It starts 88 sources including four key pages, 256
scenes and 17 outputs on an HDR 4:2:2, 1080p60 canvas. Assets remain cached in the
writable media directory. Subsequent starts keep saved settings and layouts;
editing `settings.json` only affects an empty media directory. Use Setup to
change an existing instance.

`nvidia_l4_cuarray` is an opt-in profile: generated SDR and HLG 4:2:0 clips use
HEVC with fixed CUarray pools (`extra_hw_frames: 12`). Setup preserves those
contracts when switching frame rate or canvas format. The ordinary `nvidia_l4`
and `tesla_t4` profiles retain their existing input contracts and FFmpeg 8.1
compatibility. At SDR25 this preset admits 192 sources; output limits include
the programs and reserved clean feed, and vary by format and rate. See the
[L4 measurements](../../docs/capacity.md#nvidia-l4-nvidia_l4), including the
recorded deadline misses and the rates whose limits are derived.

## Output codecs and bitrates

Setup has separate controls for program SDR, program HDR, clean program,
program preview and multiviewer, plus **one shared setting for every extra AUX**.
Bitrates are in Mbit/s in the UI, or `bitrate_kbps` in JSON, from 250 to 20000.
SDR outputs offer `h264_nvenc` and `hevc_nvenc`; HDR uses HEVC Main10 on L4.
The defaults preserve the measured show:

| Setting | Codec | kbit/s | Preset |
| --- | --- | ---: | --- |
| `sdr` | H.264 | 6000 | p5 |
| `hdr` | H.265 Main10 | 8000 | p3 |
| `sdr_clean` | H.264 | 6000 | p3 |
| `mv` (program preview) | H.264 | 4000 | p3 |
| `mv2` (multiviewer) | H.264 | 4000 | p3 |
| `extra` (all extra AUX) | H.264 | 3000 | p3 |

Codec and preset changes recalculate the NVENC allowance; lowering bitrate
reduces network traffic but does not increase the admitted output count. Apply
restarts the mixer and updates Janus mountpoints to match the chosen codecs.
HEVC playback depends on browser support. The allowance uses the profile's
existing HEVC Main10 costs for SDR HEVC too; a separate Main8 capacity sweep has
not been measured.

## Move an instance

Keep the source revision, image release and media together. `.env` contains only
destination-specific choices and is excluded from Git and the build context.
`MIXER_MEDIA_DIR` selects another absolute host directory; only that directory
holds the saved recipe, generated show and clips. Copy it after a clean stop
when preserving an existing show, or start with an empty directory to use the
packaged defaults. Do not bind-mount `mixer.demo.json` individually: Setup
atomically replaces it.

To avoid recompiling, set the same `MIXER_RELEASE` in both hosts' local `.env`
and export/import the stack's images:

```sh
docker save $(demos/mixer/deploy/l4/stack.sh config --images) | gzip > mixer-images.tar.gz
# Transfer the archive, checkout and optional media to the destination.
docker load < mixer-images.tar.gz
demos/mixer/deploy/l4/stack.sh up -d --no-build
```

For a registry, set `MIXER_IMAGE_PREFIX` to its repository prefix, push the built
service images, and pull that same release on the destination. The optional
compute price file is runtime data under `/media`; supply the new instance's
region and rate rather than copying a stale price.

After a move, verify `/api/setup` reports `running`, each wall tile advances,
WebSocket reconnection works, and a cached 25↔60 / SDR↔HDR Setup change returns
to `running`. Check deadline and output-drop counters over a fresh interval;
the lifetime counters also include startup. The deployment preset does not
turn the existing measurements into a guarantee for arbitrary content.
