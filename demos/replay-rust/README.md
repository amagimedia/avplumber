# Replay on the Rust core

The [replay demo](../replay/README.md) as three containers: Janus with its
streaming plugin and the browser preview, both from the shared images the
other demos use (`docker-compose/images`), and a player image with the Rust
`avplumber` executable and the Python TUI that drives it over the control
protocol.

The player uses the GPU when it can reach one: NVDEC decodes into CUDA
surfaces and NVENC encodes them, and no frame ever reaches host memory. On a
machine without a usable NVIDIA card it runs the same graph with software
H.264 and libx264 instead. Nothing else changes: same nodes, same edges, same
controls.

## Run

From the repository root, with Docker and Compose installed:

```sh
demos/replay-rust/run.sh
```

That starts Janus and the preview in the background, builds the player image
on first use, generates a 20-second `testsrc2` clip and transcodes it into
`demos/replay-rust/media/replay.ts` with its seek table and history if there
is no recording yet, and opens the TUI. Open <http://127.0.0.1:8080> in a
browser on the same host to watch. Press `q` in the TUI to leave; Janus keeps
running until `run.sh down`.

To play your own file:

```sh
demos/replay-rust/run.sh transcode /path/to/source.mp4 30
demos/replay-rust/run.sh play
```

| Command | What it does |
| --- | --- |
| `run.sh up` / `run.sh down` | start or stop Janus and the preview |
| `run.sh build` | build the player image (and Janus) |
| `run.sh sample [seconds] [fps] [WxH]` | a `testsrc2` clip, transcoded into `media/` (default 20 s, 30 fps, 640x360) |
| `run.sh transcode <vod> [fps]` | convert a file into `media/replay.ts` (replaces an existing one) |
| `run.sh play [player options]` | the TUI; options go to `player.py`, e.g. `--no-loop` |
| `run.sh exercise` | the **RUN V2** checks headless, exit code 1 on any failure |
| `run.sh shell` | a shell in the player image |

Both `sample`/`transcode` and `play` follow the same rule, so a recording made
on the GPU is played on the GPU.

## The GPU

`run.sh` decides how the container reaches the card, and says which way it
chose:

| `REPLAY_GPU` | Wiring |
| --- | --- |
| `auto` (default) | the toolkit when it works, otherwise the devices by hand, otherwise the CPU |
| `toolkit` | `gpus: all`, the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| `devices` | pass `/dev/nvidia*` and the host's driver libraries in directly, no toolkit needed |
| `off` | no GPU; the player runs on the CPU |

The `toolkit` mode asks for `gpus: all` and for the `video` driver capability,
which is what puts `libnvcuvid` and `libnvidia-encode` in the container. The
`devices` mode bind-mounts the running driver's own libraries instead, because
their version has to match the loaded kernel module, and the entrypoint runs
`ldconfig` over them; it is generated into `.compose.gpu-devices.yaml`, which
is not committed. Both were exercised on an RTX 5080: the headless checks pass
either way, and while the player runs the host reports one NVENC session at the
recording's frame rate.

`REPLAY_BACKEND` overrides what the player then asks for: `auto` (default),
`nvidia` or `cpu`. Naming `nvidia` on a machine that cannot run it fails
instead of quietly using the CPU, which is what you want in a check.

### Watching the card work

The default clip is small enough to leave the engines near idle. For something
visible in `nvtop`, make a bigger one and raise the bitrate to match:

```sh
JANUS_BITRATE=40M demos/replay-rust/run.sh sample 20 60 3840x2160
JANUS_BITRATE=40M demos/replay-rust/run.sh play
```

On an RTX 5080, playing at 1x:

| Recording | Encoder | Decoder |
| --- | --- | --- |
| 640x360, 30 fps | ~1% | ~0% |
| 1920x1080, 60 fps | ~8% | ~1% |
| 3840x2160, 60 fps | ~28% | ~4% |

Playback speed does not change those numbers. At 2x the source reads every
other frame, so the same number of pictures per second is decoded and the
pacing node still releases one per tick; what changes is how fast the position
moves, not how much pixel work there is.

The recording is all-intra, which is what makes a frame-exact seek cheap and
the files large: about 15 MB per second at 4K60, 5 MB at 1080p60. `media/` is
not committed, and `run.sh sample` replaces what is there.

Every run says which codecs it resolved to, so a silent fallback to the CPU is
visible without digging:

```text
[replay] GPU wired in through demos/replay-rust/compose.gpu.yaml
[replay] codecs: nvidia (h264_cuvid -> h264_nvenc, frames stay on the GPU)
```

The first line is `run.sh` saying how the container reaches the card, the
second is the player saying what it then asked for. The TUI repeats it in its
status panel as `CODECS=…`, and `sample`/`transcode` end with the same wording
in their summary.

Whether the card was really used is in the avplumber log
(`/tmp/replay-demo/avplumber.log` inside the container):

```text
hwaccel `replay_gpu`: opened cuda device, frames are cuda
replay_decode: decoding h264 on hardware device `replay_gpu`
janus_encoder: encoding 640x360 nv12 surfaces on hardware device `replay_gpu`,
               no round trip through host memory
```

The same through Compose directly:

```sh
docker compose -f demos/replay-rust/compose.yaml up -d janus janus-preview
docker compose -f demos/replay-rust/compose.yaml run --rm replay sample
docker compose -f demos/replay-rust/compose.yaml run --rm replay
```

Everything shares the host network: Janus listens on 8088 (REST), the preview
on 8080, the video mountpoint on RTP 5004 and RTCP 5005, ICE on UDP
20000–20100, and the player sends RTP to 127.0.0.1. Set `JANUS_HOST_IP` to the
address your browser reaches when it is not on this host, and allow those
ports through the firewall:

```sh
JANUS_HOST_IP=192.168.1.20 demos/replay-rust/run.sh
```

`REPLAY_MEDIA` picks the media directory. The other Janus port variables are
the ones the [browser demo](../dmabuf-browser/README.md) uses.

## The player image

`Dockerfile` builds the Rust workspace against RPM Fusion's FFmpeg 7.1 on
Fedora (`--features ffmpeg7_1,async`) with a current stable toolchain from
rustup, then copies the binary into a runtime image with FFmpeg's libraries
and CLI, Python and Textual, and `demos/replay`. That FFmpeg already carries
`h264_cuvid` and `h264_nvenc`, so the image needs nothing NVIDIA-specific: the
driver comes from the host at run time. Rebuilds are incremental:
the cargo registry and the build directory are BuildKit cache mounts, the
dependencies are fetched in a layer keyed on the lock file before the sources
are copied, and the pip install sits before the demo sources.

The images build on the host network (`build.network: host` in
`compose.yaml`), which keeps package downloads working on hosts whose firewall
does not forward Docker's bridge. The bindgen step gets clang's own include
directory explicitly, because Fedora's libclang does not find it by itself.

`entrypoint.sh` checks that Janus answers on its REST port, then runs the
requested mode (`play`, `exercise`, `transcode`, `sample`, `avplumber`,
`shell`). The avplumber log of a session is `/tmp/replay-demo/avplumber.log`
inside the container. Files that `sample` and `transcode` write into `media/`
are handed to the owner of that directory, so the host user can remove them.

## Troubleshooting

- `docker compose -f demos/replay-rust/compose.yaml logs janus` for Janus.
- No picture in the browser: check that `JANUS_HOST_IP` is the address the
  browser uses, and that UDP 20000–20100 reach the host.
- `run.sh exercise` prints one `PASS`/`FAIL`/`SKIP` line per check; a short
  recording skips the 30-second nudges by design.
- The player runs on the CPU when you expected the GPU: the two lines above
  say which half went wrong — no GPU wiring, or wiring but no codecs. Then try
  `nvidia-smi` inside the container (`run.sh shell`). `REPLAY_BACKEND=nvidia`
  turns a silent fallback into an error that says what is missing.
