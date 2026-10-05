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

Allow the public proxy port and Janus UDP range through the destination firewall.
The internal service ports stay behind the proxy. For GCP, an instance tag can
scope the viewer rule to this mixer:

```sh
gcloud compute instances add-tags <name> --project <project> --zone <zone> --tags mixer-viewer
gcloud compute firewall-rules create <rule-name> --project <project> --network <network> \
  --target-tags mixer-viewer --allow tcp:7681,udp:20000-20100 --source-ranges <viewer-cidr>
```

Use the ports configured in `.env` if changing the defaults. Viewer access
includes WebRTC UDP; making the Setup page reachable alone does not test video.

The first start seeds `media/demo.json` from [settings.json](settings.json), then
generates missing clips. It starts 88 sources including four key pages, 256
scenes and 17 outputs on an HDR 4:2:2, 1080p60 canvas. Assets remain cached in the
writable media directory. Subsequent starts keep saved settings and layouts;
editing `settings.json` only affects an empty media directory. Use Setup to
change an existing instance.

The first start on a new media directory also compiles FFmpeg's CUDA filter
kernels: the NVIDIA driver compiles them on first use, and on this host that
stalled GPU work for about 15 s during startup. The compiled kernels are kept
in `media/.nv-cache` (`CUDA_CACHE_PATH` in [compose.yaml](../../compose.yaml)),
so a recreated container or a new image starts without the stall for as long as
the media directory is kept. The driver decides when an entry no longer fits,
for example after a driver upgrade, and then compiles once more. Deleting
`media/.nv-cache` is safe and costs one such start. A stalled start is slower,
not wrong: each wipe clip is cached whole or the start fails
([cookbook](../../docs/cookbook/wipe-preload.html)).

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

## Fresh-host validation

On 2026-10-05, all six application images built from revision `899de927c21f`
on a fresh public Ubuntu 26.04 `g2-standard-16` host prepared by the
[GCP installer](../gcp/README.md). All 16 FFmpeg patches applied; no images or
media from an existing mixer were needed. The generated shows passed these
public WebRTC checks:

| Canvas | Sources | Scenes | Outputs | Mean VRAM | 60-second deadline misses / output drops |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1080p60 HDR 4:2:2 | 88 | 256 | 17 | 14.2 GiB | 0 / 0 |
| 1080p25 SDR 4:2:0 | 192 | 256 | 26 | 19.4 GiB | 0 / 0 |

Every wall stream advanced before and after WebSocket reconnection in both
shows. Setup changed from HDR60 to SDR25 and returned to `running`; the first
SDR asset generation took about 287 seconds, followed by 25 seconds to restart.
The final SDR show used HEVC at 2.5 Mbit/s for preview and multiviewer, with the
remaining outputs at the defaults above. These are short deployment checks,
not a sustained-load qualification or a sweep of every format/rate combination.

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
