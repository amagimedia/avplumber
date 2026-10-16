# Running the mixer

[Quick start](../README.md) · [Configuration](config.md) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)

## Start the mixer backend

Use an NVIDIA host and a Python module built against the same patched FFmpeg
as avplumber, with CUDA/NVCC and DRM/GL for browser sources.

```sh
python3 -m pyplumber.mixer.cli --config mixer.json
```

The config must declare outputs, or supply `--output <path>` / `--janus-output`.
For a simple file-only show:

```sh
python3 -m pyplumber.mixer.cli \
  --input <input-1> --input <input-2> --loop-inputs \
  --output program.mp4 --fps 30 --remote-control-port 7777
```

Only this `--input` mode uses fixed 1080×1920 fullscreen and 2/4/8/16-box
layouts. `--config` supplies its own canvas and scenes. Use `--output-format`
for an ambiguous target; `--codec` and `--bitrate` select the NVENC output.
Run `--help` for current defaults and the full option list.

File inputs retain their native cadence; `--fps` selects the canvas/output rate.
For encoded files, fractional rates come from their timestamps. The compositor
selects, repeats or drops source frames as needed, while `realtime` still paces
input delivery. Browser pacing follows its configured paint rate. See
[configuration](config.md#input-cadence-and-colour-tag-guarantees).

## Controls

The Compose stack starts the web UI. With a standalone backend:

```sh
python3 -m pyplumber.mixer.gui --host 127.0.0.1 --port 7777 --http-port 7681
```

For the [recorded two-viewer layout](https://amagimedia.github.io/avplumber/demos/mixer/docs/),
select **Viewers 2**, then **Program preview** in one viewer and **Program dirty**
in the other. The first contains PVW, PGM and eight scene slots; the second
shows the program with its enabled keys. Viewer selection is local to the
browser. Scene takes, key buttons, AUX paging and AUX layout controls operate
the running mixer. **Direct cut** always cuts a selected scene to Program.
Otherwise scene clicks select Scene Preview. **Fade / Wipe** choose the effect;
**AUTO / Space** runs it, and **CUT / C** cuts the displayed Preview. An empty
Preview cannot be taken. Scene Preview shows the background only, without
downstream keys; it is not a keyed look-ahead view.

The **AUX bus to edit** selector delegates slot edits independently of the
monitors. An armed slot belongs to that bus until assigned or cancelled.
Changing the editing bus cancels the armed slot; changing a monitor does not.

**1–9** use the selected saved source bank, regardless of browsing filters.
Three banks are stored in this browser. **Save browsed scenes to bank** explicitly
replaces a bank with the first nine scenes in the current filter. Missing scenes
remain labelled unavailable instead of being silently replaced. Arrow keys still
browse Scene Preview. Fade and key durations accept seconds or whole frames at
the canvas rate (for example, 12 frames at 25 fps is 0.48 seconds).

Each downstream key has separate **CUT** and **AUTO** controls. Its tally comes
from the last frame emitted by the keyer: on, fading-in, fading-out, off, or
unknown if telemetry is unavailable. The requested on/off target is separate;
pending commands never turn the tally on or off speculatively. Visibility means
a key frame contributes with nonzero envelope opacity, not that its HTML contains
nontransparent pixels. The UI polls this small tally separately every 200 ms.

**Toggle all keys / K** sends one group operation: all off if any target is on,
otherwise all on, using the key duration. **With next take** is a persistent
selection: selected keys toggle on the first emitted frame of each new background
take. CUT cuts those keys; AUTO runs their independent key fades at the configured
key duration and curve. Pausing the background fade does not pause independent
DSK fades. CUT completing that background transition also finishes its selected
key fades. Preview selection and failed takes that never reach air do not fire keys.

Each viewer must support its output codec. The recorded L4 show uses HEVC for
Program preview and H.264 for dirty Program; an H.264-capable browser alone
cannot display both. Setup can change output codecs, but applying Setup
restarts the mixer. The downloadable demonstration MP4 is H.264.

For the optional terminal UI:

```sh
python3 -m venv .venv-tui
.venv-tui/bin/pip install -r pyplumber/mixer/gui/requirements.txt
.venv-tui/bin/python -m pyplumber.mixer.gui.tui --host 127.0.0.1 --port 7777
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

Select the second with `MIXER_DOCKERFILE=Dockerfile`. The [L4 preset](../../../docker-compose/mixer/deploy/l4/README.md)
pins its own FFmpeg build. Keep `media/.nv-cache` to reuse driver-compiled kernels.
TensorRT is optional (`WITH_TENSORRT=1` and the `tensorrt_url` build secret in the
Fedora image); the manual mixer needs no neural models.

## Behind a reverse proxy

Serve controls at `/` and the preview server at `/preview/`, stripping that
prefix. Set `MIXER_PREVIEW_BASE=/preview/` (or pass `--preview-base /preview/`
to `python3 -m pyplumber.mixer.gui`).
Forward WebSocket upgrades for the preview's Janus path as well as normal HTTP.
The [L4 proxy template](../../../docker-compose/mixer/deploy/l4/nginx.conf.template) is a complete example.
WebRTC also needs the configured UDP range reachable from the viewer.

## Tests

```sh
python3 -m pytest -q tests/mixer
```

NVIDIA-dependent cases require the configured GPU host. Against a running backend:

```sh
python3 tests/mixer/smoke_test.py --port 7777 --wipe-file <alpha-clip>
python3 tests/mixer/cut_spam.py --url http://127.0.0.1:7681
```

These change Program. See [latency](latency.md) for the cut probe and browser
measurement, and [capacity](capacity.md) for the limits of recorded results.
