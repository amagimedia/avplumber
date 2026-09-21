# Live mixer demo

<table>
<tr>
<td width="74%" valign="top"><img src="docs/webui.png" alt="The mixer web UI: program and preview panels, a tile per scene with the one on air lit, and take buttons — Cut, Fade, one per cached wipe clip, the Direct toggle and the fade length."></td>
<td width="26%" valign="top"><img src="docs/program-16box.png" alt="The 1080x1920 program: a sixteen-box grid, two columns of eight, each cell a different source — live browser pages down the left, video clips down the right."></td>
</tr>
</table>

Sixteen sources — **live browser pages and video files together** — composited
on one GPU canvas into a 1080×1920 portrait program, cut, faded and wiped from
the browser control surface on the left, with the whole show described by one
JSON document.

[**Watch it run** (21 s)](https://github.com/amagimedia/avplumber/releases/download/mixer-demo-media-2026-09/mixer-webui-demo.mp4) ·
[Demo page](https://amagimedia.github.io/avplumber/demos/mixer/docs/) ·
[Configuration reference](docs/config.md) ·
[Processing graph](https://amagimedia.github.io/avplumber/demos/graph.html?demo=mixer)

On a Tesla T4, that run costs **17–18% GPU** with the canvas at 30 fps and
sends **2.86 Mbit/s** from a 2 700 kbit/s WebRTC rendition, with five media
wipes held decoded in GPU memory.

## Run

A Linux NVIDIA host with hardware decode and NVENC; see the
[shared Docker/NVIDIA setup](../README.md). Nothing private is needed.

Build the image and start the Janus preview:

```sh
docker build -f demos/mixer/Dockerfile -t avplumber-mixer:local .
docker compose --env-file demos/dmabuf-browser/.env.example \
  -f demos/dmabuf-browser/compose.yaml up -d --build janus janus-preview
```

Prepare the default recipe, then run its generated show. Preparation creates
synthetic SDR/HDR clips and an alpha wipe; no private assets or downloads are needed.

```sh
mkdir -p media
docker run --rm --gpus all --entrypoint python3 \
  -v "$PWD/media:/media:rw,z" avplumber-mixer:local \
  demos/mixer/prepare_demo.py demos/mixer/demo.example.json --media-dir /media

docker run --rm --gpus all --network host \
  -v "$PWD/media:/media:ro,z" avplumber-mixer:local \
  --config /media/mixer.demo.json --janus-output --remote-control-port 7777
```

The SDR program plays at <http://127.0.0.1:8080>. The recipe also emits HDR HEVC
on RTP port 5006; viewing that output needs a matching HEVC Janus mountpoint and
compatible client. Synthetic media is cached in `media/assets/`, and wipes in
`media/media_wipes/`. Rerunning preparation reuses the media and replaces the
expanded `media/mixer.demo.json`.

### Choose or customize a preset

Two mixed-source example presets are provided:

| Preset | Source proportions |
| --- | --- |
| [`demo.equal.json`](demo.equal.json) | Equal weights for generated SDR 420, HLG 420, HLG 422, SDR 422, Bunny and browser sources. |
| [`demo.cinematic.json`](demo.cinematic.json) | The same source types, with two generated HLG 420 inputs replaced by public PQ and HLG movie clips at the default count. |

Both make **16 independent sources and 32 scenes at 1080×1920, 60 fps**,
retain the existing synthetic patterns, and use steady colour bars behind
transparent browser overlays. Both need the
[DMA-BUF browser stack](docs/guide.md#docker) and `--wipe-cache-mb 2048` for
the 1080p60 wipe. The synthetic-only `demo.example.json` used above is the
minimal startup example for the plain mixer image; it makes 16 sources and
24 scenes without downloads or browsers.

To add your own preset, copy either mixed-source recipe:

```sh
mkdir -p media
cp demos/mixer/demo.cinematic.json media/demo.custom.json
```

Edit **`inputs[].weight`** for source proportions, **`source_count`** for the
total (for example 8, 16, 32 or 42), **`scene_count`** for the number of scenes,
and **`canvas.fps`** for 25, 30, 50 or 60 fps. Weights are relative; zero
disables an input entry. URLs and pattern definitions stay in the copied JSON.
No Python changes or preset registration are needed.

Prepare it with the same Docker command above, replacing the recipe argument
with `/media/demo.custom.json`, then run the generated show in the browser-capable
mixer environment. See the [worked custom preset](docs/recipe.md#custom-preset)
for exact proportions, commands and transparency background requirements.

With the build and dependencies already installed on an NVIDIA host:

```sh
python3 demos/mixer/prepare_demo.py demos/mixer/demo.example.json
python3 demos/mixer/mixer.py --config media/mixer.demo.json --janus-output --remote-control-port 7777
```

## Faster cuts and latency

For a catalogue of fixed, filter-free scenes, append these options to the
Janus mixer command:

```sh
--prewarm-cut-scene '*' --cut-latency-encoder janus_encoder
```

Prewarm retains recent decoded source frames for direct cuts, sharing bounded
queues across scene definitions without rendering every hidden scene. It is
off by default; repeat `--prewarm-cut-scene SCENE` to select only some scenes.
The cost is extra queue handling and potentially more retained GPU surfaces.

The preview can show **avplumber latency** beside **WebRTC RTT**, outside the
video. The AVP value is the median of the last up to three measured CUTs, from
command receipt to the first matching encoded frame—not capture-to-browser
latency. See [measurement setup and prewarm limits](../../doc/mixer_cut_latency.md)
for the WebUI connection and deployment-local `metrics.json` configuration.

## Control it

The web page puts scene controls on the left and the portrait WebRTC player on
the right. Drag the divider to resize the panes; the split persists in your
browser. The player uses port 8080 on the same host and defaults to H.265/HDR;
open the control page with `?codec=h264` for SDR. Port 8080 remains available
as a standalone preview, including the latency and RTT readouts.

Two surfaces speak the same protocol and can run at once. Cut is the default
transition; explicit show settings and operator choices can select Fade or Wipe.

```sh
# browser: scene tiles, a button per wipe clip, Cut / Fade / Direct
python3 demos/mixer/webui.py --host 127.0.0.1 --port 7777 --http-port 7681

# terminal
python3 -m venv .venv-tui && .venv-tui/bin/python -m pip install -r demos/mixer/requirements.txt
.venv-tui/bin/python demos/mixer/tui.py --host 127.0.0.1 --port 7777
```

| Control | Result |
| --- | --- |
| A scene tile | Loads Preview; in Direct mode, takes it to Program |
| Cut / `c` | Immediate take |
| Fade / `f` | Blend over the chosen duration |
| A wipe button / `w` | Play that transparent clip over the change |
| Direct / `d` (`t` in the TUI) | Picks go straight to Program |
| `1`–`9` | Pick one of the first nine scenes |

A new take interrupts a running transition from the current picture. Wipe
clips are decoded once at start and replayed from GPU memory, so the first
wipe is as quick as the tenth; use a clip with alpha (QTRLE/ARGB, ProRes 4444).

## Browser pages as sources

Any input can be a live page rendered by the
[DMA-BUF browser demo](../dmabuf-browser/README.md) instead of a clip:
`--input dmabuf://<window-id>` takes the window's frames straight into the
compositor with no decoder. Pages and files mix freely and share every layout,
transition and control. Pages that only paint once are held, so a static
graphic keeps feeding the mixer.

The compose override runs the mixer inside the DMA-BUF stack and opens the
pages itself:

```sh
cd demos/dmabuf-browser && cp .env.example .env
MIXER_SOURCE_COUNT=4 docker compose --env-file .env \
  -f compose.yaml -f compose.mixer.yaml up --build
```

`HTML_OVERLAY_URL` picks the page, `MIXER_SOURCE_COUNT` the number of windows,
`MIXER_SOURCE_WIDTH`/`MIXER_SOURCE_HEIGHT` their size. Outside compose:
`--dmabuf-open URL`, `--dmabuf-size WxH`, `--dmabuf-socket-dir`.

## Describe a show in one file

`--config mixer.json` replaces `--input` and the built-in layouts with a
document of sources, scenes, wipes, control defaults and encoded outputs — no
2/4/8/16-box layout lives in code. Every field is documented in
**[docs/config.md](docs/config.md)**; [`config.example.json`](config.example.json)
is a worked example.

The example uses placeholder locations. Supply your own source IDs, URLs, media
paths and scenes in a deployment configuration kept outside the public checkout.
The running show's source list is not a mixer default.

```sh
docker run ... avplumber-mixer:local --config /media/mixer.json --janus-output
```

`make_config.py` writes the demo's own layouts out in that form:

```sh
python3 demos/mixer/make_config.py --wipe /media/wipe.mov \
  cam1=/media/camera-1.mp4 page=https://example.org/page@1920x1080 > mixer.json
```

## Test sources

Sixteen generated clips with visible frame IDs (needs NumPy and FFmpeg):

```sh
python3 demos/mixer/tests/frame_codes.py media --sources 16 --width 1920 --height 1080 --fps 60 --seconds 30
```

Eight colorful SDR patterns (bars, mandelbrot, life, ...) as NVENC clips:

```sh
python3 demos/mixer/tests/sdr_patterns.py media/assets/patterns --fps 60 --seconds 20
```

## An HDR show

The [default recipe](demo.example.json) uses an HLG P210 (10-bit 4:2:2) canvas,
synthetic H.264 SDR 4:2:0, HEVC HLG 4:2:0 and raw SDR/HLG 4:2:2 inputs.
It outputs a tone-mapped H.264 SDR preview on port 5004 and HEVC Main10 HDR
on port 5006. Change their proportions and total source count in the recipe;
there is no separate asset list to assemble on the command line.

Optional movie inputs can be downloaded from HTTP(S) or read from
`media/assets/`. A disabled Big Buck Bunny download is included as an example.
HDR movie downloads are optional too: specify a direct playable media URL and
its `pq` or `hlg` color contract. See [public assets and recipe options](docs/recipe.md).
Browser inputs require the `dma-page` service and DMA-BUF container flags from
[docs/guide.md](docs/guide.md#docker); their default weight is zero.

## Under the hood

Janus output limits forced keyframes to one per 150 ms by default (9 frames at
60 fps). Override with `--keyframe-min-interval-ms 200`; `0` disables the limit.
The option also applies to Janus renditions loaded with `--config`. Cuts and
ordinary frames are not delayed: pending cut/RTCP requests coalesce until the
next eligible frame. Periodic keyframes share the same limit.

<a href="https://amagimedia.github.io/avplumber/demos/graph.html?demo=mixer" target="_blank" rel="noopener noreferrer"><img src="https://amagimedia.github.io/avplumber/demos/mixer/docs/mixer-graph-grouped.png" alt="Grouped mixer graph: inputs, two compositor slots, transitions, media wipe and output. Click for the full ungrouped graph." width="640"></a>

Two compositor slots draw every scene; a transition filter blends them and the
wipe is one more layer in the same kernel, so nothing round-trips through the
CPU. Browser frames are converted from RGB to the NV12, P010 or P210 canvas inside the
draw pass. The program is composited once and each rendition re-times and rescales
it, so a second output costs an encode, not another composite.

Limits: 64 sources per show, no runtime source changes. Recipe grids support up
to 64 boxes; the legacy `--input` layouts support up to 16;
see [docs/config.md](docs/config.md#known-limitations).

Output files, Janus settings, layouts and tests: [docs/guide.md](docs/guide.md).
Measured samples and conditions: [runtime-load-1080p.json](docs/runtime-load-1080p.json),
[latency.md](docs/latency.md).
