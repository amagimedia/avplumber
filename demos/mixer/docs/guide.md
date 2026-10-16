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
the running mixer.

The take bar follows a vision switcher's transition area. A scene tile or **1–9**
selects Preview; with **Direct cut / D** a pick cuts straight to Program. **Cut / C**
cuts Preview to Program and completes a running transition; **Auto / Space** runs
the selected transition, **Fade / F** at its length in seconds or a wipe clip
(**W**), and on a running fade holds it and runs it on again. The **lever** mixes
by hand from the end it rests at: the far end completes the mix and is where the
next one starts, back at its start the mix is cancelled. After an AUTO or CUT ended
a mix the lever left half way, it acts again once moved to an end. The bar under
the lever shows the transition's progress.

**Next** in the Preview box chooses what the next take changes, as on a GV panel:
**BKGD** the scene, **1–4** the keys, numbered as in the Keys card, lit while
armed. A red dot marks a key on air, which an armed take removes. With BKGD the
armed keys go with the background transition on its clock: they mix at its rate,
hold when it holds, follow the lever, cut with a cut and switch at a wipe clip's
cut point; a cancelled transition leaves them as they were, and they stay armed
for the following takes. Without BKGD, Cut cuts the armed keys and Auto fades them
at the Fade length; the lever, which mixes the background only, is off. BKGD is
the page's own; the armed keys are the mixer's, the same for every controller.

Each key in the Keys card cuts on or off air, or fades at the card's key fade. Its
red border is the tally of the last frame the keyer emitted, never a pending
command; its number turns amber while it is armed. **K** changes every key on one
frame: all off if any is on, otherwise all on. The slots card edits the Program
preview's scene slots: click a slot, then a scene, or **Shift+1–8**.

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
