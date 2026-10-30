# Running the mixer

[Quick start](../README.md) · [Configuration](config.md) ·
[Operator controls on Pages](https://amagimedia.github.io/avplumber/demos/mixer/docs/controls.html) ·
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

The [20-second Retina demonstration](https://amagimedia.github.io/avplumber/demos/mixer/docs/)
shows two viewers, four downstream keys, a manual fade, halfway hold and return stroke.
Its native pictures were captured before Janus; controls and measured performance meters
are replayed in the actual UI. The controls below describe the live operator page.

### Preview, Program and automatic takes

A scene tile or **1–9** selects Preview. **Direct / Shift+D** makes a scene pick cut
straight to Program; the arrow keys still browse Preview without taking it. Scene filters
and page buttons let you browse the catalogue. After a completed background take, Preview
becomes the scene that left Program.

Choose **Fade / F**, **Dip / D**, or a media wipe (**W**, repeated to cycle the list), then
press **Auto / Space** to run it. Selecting a transition prepares Auto; it does not take.
Fade mixes the scenes. Dip goes through black, white or a chosen colour. The curve buttons
shape a fade or each half of a dip. A wipe uses its clip's duration and cut point; clip
paths refer to files on the mixer host.

**Cut / C** takes immediately, or completes a transition already on air. The **Halt / Complete**
selector controls a second Auto press during its own transition: Halt pauses it with both
sources still moving, and another press resumes; Complete finishes it immediately.

The rate field uses the show's frame rate: `15f` means 15 frames, `1:15` means one second
and 15 frames. Bare `45` is 45 frames, displayed as `1:15` at 30 fps. Presets, the small
arrows and the mouse wheel change the rate; Shift with the wheel steps one second.

### Manual T-bar

Use the rotate button to place the T-bar horizontally or upright. Drag from the end where
it rests to mix toward Preview. Stop anywhere to hold the mixture while both sources keep
moving. Reach the far end to complete the take; the next take starts from that end in the
opposite direction. Return to the starting end to cancel the take. The grip, percentage,
LEDs and trail show its position and progress.

Auto can run a manual fade on from its current position; Cut completes it. If an automatic
take finishes while the physical lever is between endpoints, the lever must return to an
endpoint before starting another take. The page parks its on-screen lever as needed.

A selected wipe can follow the lever forward and backward once its complete clip is decoded
in the GPU wipe cache. The lever is unavailable while that wipe is not ready. Without a
selected wipe, manual operation mixes the scenes. The T-bar requires **NEXT → BKGD**; it
cannot perform a key-only take.

### DSK: independent key transitions and ALL rate

Downstream keys are overlays applied after the scene transition: lower third, ticker and
corner bugs in this demo. **Program dirty** includes them; the clean feed omits them while
retaining the scene transition. Off-air keys remain subscribed so they can be brought in.

Click a key's name to fade it on or off at its effective rate, using the selected Fade curve.
**Shift+click** cuts that key regardless of its saved rate. Each key can have its own rate,
or inherit the shared **ALL** rate. Rates and inheritance belong to the running mixer and
are shared by every controller; changing a rate does not itself put a key on or off air.

| Rate control | Action |
| --- | --- |
| Click the ALL rate tag | Cycle CUT, ¼ second, ½ second and 1 second |
| Click a key's rate tag | Cycle its own CUT, ¼ second, ½ second and 1 second, then return to following ALL |
| Wheel over a rate tag | Step one frame; Shift steps one second. On a key, this sets its own rate |
| Focus a tag and press F2 | Type frames (`15` or `15f`), seconds:frames (`1:15`), or `0`, `c`, `cut` or an empty field for CUT |
| Type `all` in a key's rate editor | Restore inheritance from ALL |
| Enter or leave the editor / Escape | Save / cancel the edit |

Key rates are limited to 0–10 seconds. The ALL tag changes only keys that follow it;
explicit key rates remain independent. **ALL / K** takes all keys off if any is on,
otherwise takes all on, each at its own effective rate. This is different from a key-only
NEXT take, which uses one common take rate.

For example, set ALL to `12f`, the lower third to `30f`, and a bug to CUT. The ticker and
other following keys use 12 frames, the lower third uses 30, and the bug cuts. Changing
ALL to `15f` leaves those two explicit overrides intact. Type `all` on the lower third's
tag to make it follow ALL again.

The red tally reflects the keyer's reported on-air state (a click responds immediately,
then the mixer confirms it). An amber key number means it is armed for the next take;
arming alone does not make it visible.

### NEXT: combine background and key takes

The **NEXT** row in Preview arms **BKGD** and individual keys. An armed key toggles on the
take: off-air keys come in, on-air keys go out. The red dot marks a key currently on air.

| Operation | Background | Armed keys | Rate / clock |
| --- | --- | --- | --- |
| BKGD on, Cut | Cuts to Preview | Cut to their opposite state | Same take |
| BKGD on, Auto fade or dip | Runs the selected transition | Follow its progress and hold/resume | Background take's clock |
| BKGD on, T-bar | Follows the lever | Follow the manual take; cancel restores their original state | Lever progress |
| BKGD on, media wipe | Runs or manually scrubs the ready wipe | Switch at the wipe's cut point | Wipe clock / position |
| BKGD off, Cut | Unchanged | Cut to their opposite state | Immediate |
| BKGD off, Auto | Unchanged | Fade together | Main Fade rate and curve, regardless of individual key rates |
| Direct key click or ALL / K | Unchanged | Operates the clicked key or all keys, irrespective of NEXT | Each key's own rate or inherited ALL rate |

NEXT selection stays armed for subsequent takes. BKGD selection belongs to this page;
armed keys are shared by all controllers. With BKGD off the T-bar is disabled. Arm at
least one key for a key-only take.

To remove a lower third during a scene change, arm BKGD and key 1, select the next scene
in Preview and press Auto. The lower third follows that scene transition even if its
independent key button has a different rate. To fade only the lower third and ticker
together, turn BKGD off, arm keys 1 and 2, set the main Fade rate and press Auto.

### Fade to black

**FTB** in Program fades the finished keyed program to black at the main Fade rate and
curve; press it again to fade back. It flashes while Program is black. The clean feed
is unaffected. This is distinct from Dip, which transitions between scenes through a colour.

### Viewers, AUX and output wall

For the recorded layout, select **Viewers 2**, **Program preview** in the first viewer
and **Program dirty** in the second. Program preview contains PVW, PGM and eight scene
slots. Drag the dividers to resize the deck and viewers; double-click a viewer or press
Enter while it has focus for full screen. The T-bar orientation, viewer choices and
split sizes are local browser preferences.

The slots card controls the multiview last clicked, marked “· slots” when applicable.
Click a slot and then a scene to assign it, or use **Shift+1–8** for the selected scene.
AUX layout selectors and source-page buttons change that bus on the running mixer;
**[ / ]** page a source multiview. These operations are shared, not local viewer preferences.
**Outputs** opens the output wall for monitoring the configured feeds and renditions.

Each browser must support the selected output's codec and profile. The approved October
28 recording used H.264 for both viewers; other setups can offer HEVC or multiple colour
renditions. **Setup** configures the show, sources, scenes, keys and outputs; applying it
restarts the mixer. Selecting a viewer does not change the output's encoding.

### Performance bar and viewer statistics

Hover or focus a top-bar meter for details; Escape dismisses its tip.

| Meter | What it shows |
| --- | --- |
| LAT | Measured cut latency; its tip includes cut p95, deadline misses, AUX drops and input repeat/skip counters |
| GPU / ENC / DEC | GPU work, NVENC and NVDEC utilization separately; tips include device and codec/session details |
| VRAM / POWER | Used/total GPU memory and power draw/limit |
| WIPES | GPU wipe-cache use and available clip details |
| LOAD / CPU | One-minute host load relative to vCPU count, and host CPU utilization; CPU's tip shows mixer cores and sources/core |
| THREAD | Busiest mixer thread, in percent of one CPU core; its tip groups mixer threads and Chromium processes |

On multiple GPUs the utilization meters show the busiest device and VRAM the fullest;
the tips list devices. A viewer's **RTT**, **FPS** and **BUF** describe its WebRTC connection,
decoded frames per second and jitter-buffer residence. Missing or stale player statistics show a dash. Viewer
network statistics and the LAT cut measurement describe different parts of delivery.

### Keyboard reference

| Key | Action |
| --- | --- |
| 1–9 | Pick a visible scene; Direct mode takes it immediately |
| Arrow keys | Browse Preview only; up/down move by a scene-grid row |
| < / > or , / . | Change scene page |
| Shift+D | Toggle Direct |
| C / Space | Cut / Auto |
| F / D / W | Select Fade / Dip / wipe; repeated W cycles wipes |
| K | All keys off if any is on, otherwise all on |
| Shift+1–8 | Assign selected scene to a multiview slot |
| [ / ] | Previous / next source-multiview page |
| F2 on a rate tag | Edit that key's or ALL's transition rate |

Shortcuts do not take over while typing in a text field or choosing in an open menu.

### Terminal UI

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
