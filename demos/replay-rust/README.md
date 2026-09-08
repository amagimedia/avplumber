# Replay on the Rust core

The [replay demo](../replay/README.md) as three containers: Janus with its
streaming plugin and the browser preview, both from the shared images the
other demos use (`docker-compose/images`), and a player image with the Rust
`avplumber` executable and the Python TUI that drives it over the control
protocol. Software H.264 in and libx264 out; no GPU, no CUDA, no NVIDIA
container toolkit.

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
| `run.sh sample [seconds] [fps]` | a `testsrc2` clip, transcoded into `media/` |
| `run.sh transcode <vod> [fps]` | convert a file into `media/replay.ts` (replaces an existing one) |
| `run.sh play [player options]` | the TUI; options go to `player.py`, e.g. `--no-loop` |
| `run.sh exercise` | the **RUN V2** checks headless, exit code 1 on any failure |
| `run.sh shell` | a shell in the player image |

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
and CLI, Python and Textual, and `demos/replay`. Rebuilds are incremental:
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
