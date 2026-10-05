# Demo recipe

A recipe describes the workload to generate. The preparer creates missing media,
then writes an ordinary [mixer show](config.md). The [Compose quick start](../README.md#run)
prepares it automatically. For a standalone NVIDIA environment:

```sh
python3 demos/mixer/prepare_demo.py media/demo.json --media-dir media
python3 demos/mixer/mixer.py --config media/mixer.demo.json --janus-output
```

Run preparation on the NVIDIA host or inside the mixer Docker image. It needs
Python, NumPy and FFmpeg with NVENC, included in that image: generated clips
are encoded with `h264_nvenc` (SDR) and `hevc_nvenc` (HDR). `--runtime-media-dir /media`
lets host preparation write paths for a later `/media` container mount.

## Counts and proportions

| Recipe field | Meaning |
| --- | --- |
| `source_count` | Total independent input chains, 1–192 (up to 191 with aux), each with its own decoder or browser window. Every generated source also has its own media; see [generated media](#generated-media). This is separate from boxes visible in a scene. |
| `scene_count` | Total named scenes to generate, independent of source count. |
| `inputs[].weight` | Relative share of the input total. Zero disables the entry, including download/preparation. |
| `layouts` | Layout names mapped to relative shares of the scene total. |
| `seed` | Integer controlling random compositions. The same recipe and seed reproduce the same scenes. |
| `alpha_background` | Optional allocated video source ID for every transparency scene. The equal recipe uses steady colour bars. |
| `canvas` | Normal mixer canvas settings: dimensions, fps, working format and color. Also sets synthetic clip dimensions and cadence. |
| `generation.seconds` | Length of each synthetic input loop; default 2 seconds. |
| `generation.width`, `generation.height` | Optional synthetic source dimensions, independent of the canvas. Defaults to the canvas dimensions. |
| `renditions` | Normal mixer output definitions, copied to the generated show. Use separate RTP/RTCP port pairs. |
| `clean_rendition` | Fields in which the clean SDR copy, added by the setup's clean feed, differs from the first rendition, such as `preset` and `bitrate_kbps`. |
| `browser_ring_size` | Maximum outstanding DMA-BUF frames per browser; 1–64, default 6 at 25/30 fps and 9 otherwise. The import-cache capacity is at least 32; obsolete idle imports expire. |

Weights need not add to 100. Largest-remainder rounding makes counts add to the
requested total; ties follow recipe order. A positive weight can round to zero
when the total is small. For exact counts, make weights sum to the desired total.
Preparation prints the allocated input counts before generating media.

These are graph limits. The setup page also applies the host's
[source limits by mode and frame rate](cookbook/source-limits.html) (110 sources at
25 and 30 fps, 82 at 50 and 75 at 60 on an SDR canvas, keys included) and at most
192 scenes; custom recipes must fit the host's GPU, decoder, upload and browser capacity.

The setup page's Balanced mix includes **SDR · 4:2:0 · HW upload** sources beside the
H.264/NVDEC ones, in all three canvas modes. These cached NV12 patterns
are paced on the CPU, then uploaded once per frame; they use no NVDEC and need no
pixel-format conversion before upload. Disk traffic and CPU-to-GPU bandwidth
increase compared with encoded clips.
In a custom recipe, add an input such as:

```json
{"id": "sdr420_raw", "kind": "generated", "color": "sdr", "chroma": "420",
 "storage": "nv12", "weight": 4}
```

It uses the same SDR pattern pool as encoded sources, as `.nv12` files. The
generated show declares these sources as `kind: "nv12"`.
`canvas.raw_upload: "pinned"` in the recipe (copied to the show) uploads raw
sources through `raw_to_cuda` instead; see [config](config.md#canvas).

Generated, encoded 4:2:0 inputs accept `codec: "h264"` or `codec: "hevc"`.
SDR defaults to H.264; HLG requires HEVC. An encoded input group can also set
`decode_storage: "cuarray"` and `extra_hw_frames: 3`; preparation preserves these
options on every generated video source. CUarray requires the patched upstream
FFmpeg build and compatible GPU consumers; see [decoder options](config.md#sources).
Other input groups keep their existing decoder storage.

Both 10-bit canvas modes also offer **HDR · 4:2:0 · raw upload**. This generates
cached HLG P010 patterns and uploads them directly, without NVDEC or a runtime
SDR-to-HDR conversion. Use `storage: "p010"` and `color: "hlg"` in a generated
4:2:0 recipe input; the resulting show uses `kind: "p010"` and `.p010` files.
Assets are prepared only when missing. P010 doubles raw storage and upload bytes
relative to NV12; the shared upload budget is a conservative estimate, not a
validated HDR capacity limit.

The synthetic-only startup example's weights 8:4:2:2 at 16 sources give eight SDR 4:2:0, four HLG
4:2:0, two HLG 4:2:2, and two SDR 4:2:2 inputs. Increasing `scene_count` creates
more scene definitions, not additional input chains. Increasing `source_count`
opens more independent chains and generates media for each new source.

## Example presets

`demo.equal.json` is the mixed-source preset: a 1080×1920 canvas at 60 fps,
32 scenes and equal weights over five entries. Its default 16-source allocation is:

| Input entry | Sources |
| --- | ---: |
| `sdr420` — generated SDR 4:2:0 | 4 |
| `hlg420` — generated HLG 4:2:0 | 3 |
| `hlg422` — generated HLG 4:2:2 | 3 |
| `sdr422` — generated SDR 4:2:2 | 3 |
| `browser` — transparent browser pattern | 3 |

Equal weights are rounded to fit 16 sources. The preset uses the existing
pattern generators and requires the DMA-BUF browser stack.

## Custom preset

Copy a preset to `media/demo.json` and edit `source_count`, `scene_count`,
`canvas.fps` and the source/layout weights. Apply it with
`docker compose -f demos/mixer/compose.yaml restart mixer`.

For example, set `source_count` to **42**, `scene_count` to **64**, keep
`canvas.fps` at **60**, and change the existing input weights to:

| Input ID | Weight | Sources at this total |
| --- | ---: | ---: |
| `sdr420` | 20 | 20 |
| `hlg420` | 8 | 8 |
| `hlg422` | 4 | 4 |
| `sdr422` | 0 | 0 |
| `browser` | 10 | 10 |

These weights sum to 42, so they are exact counts at this total. Changing only
`source_count` scales the same proportions with rounding. Weights such as
50:25:25 work as percentages too; no specific sum is required. Each allocated
source has its own input chain and its own clip. Its allocation is a workload
description, not a performance guarantee.

`demo.equal.json` sets `alpha_background` to `sdr420_001`, the steady bars pattern.
Keep at least two allocated `sdr420` sources to retain that background. For a
smaller total or different mix, you can set that entry's `pattern` to `"bars"`
and `alpha_background` to `"sdr420_000"`; all sources in that entry then use
steady bars. If you disable browsers, also set `layouts.alpha_overlay` to zero.
Remove `alpha_background` if its source entry is disabled. Allocated alpha
scenes always need at least one browser and one video source.

The generated show is `media/mixer.demo.json`; keep your editable recipe separate.
Scene-only changes reuse cached assets; changing FPS creates new synthetic clips
and wipes.
To prepare media without starting the services, use the image from the quick start:

```sh
docker run --rm --gpus all --entrypoint python3 \
  -v "$PWD/media:/media:rw,z" avplumber-mixer:local \
  demos/mixer/prepare_demo.py /media/demo.json --media-dir /media
```

## Input entries

Every entry has a unique `id`, `kind` and `weight`.

| Kind | Other fields | Behavior |
| --- | --- | --- |
| `generated` | `color`: `sdr` or `hlg`; `chroma`: `"420"` or `"422"`; optional `pattern` or `patterns` | Prepares synthetic clips. `pattern` selects one; `patterns` supplies a non-empty list to cycle through, e.g. `["bars", "gradients"]`. |
| `download` | `url`; optional `color`: `sdr`, `hlg`, `pq` | Downloads a direct media file once into `media/assets/downloads/`; does not unpack archives. Untagged entries use decoded color metadata. |
| `file` | `path`, relative to `--media-dir`; optional `color` | Uses an existing encoded clip, e.g. `assets/my-movie.mp4`. |
| `browser` | `url` or `graphic`; optional `width`, `height` | Opens live SDR browser windows using the DMA-BUF browser stack. `graphic` names a graphic under [`graphics/`](../graphics/README.md) and embeds its page, with no HTTP server or external website; without `url` and `graphic` it is `browser_alpha`, the bundled transparency test page (`pattern: "alpha"` is the older way to ask for that one). Dimensions default to the canvas. |

SDR 4:2:0 defaults to seven H.264 patterns: `testsrc2`, `bars`, `rgbtest`,
`mandelbrot`, `gradients`, `life`, `sierpinski`. The cellular-noise pattern
`cellauto` remains available through an explicit `pattern` or `patterns` choice
for encoder stress testing; it is excluded from the default rotation because
large noisy regions can produce substantial keyframe bursts. The other categories have patterns
`"0"` and `"1"`. HLG uses the smooth wide-gamut bars, luminance sweep and orbiting
highlights demo, generated in scene-linear light through the HLG transfer
function; HLG 4:2:0 is encoded HEVC Main10. Both 4:2:2 categories are
raw 10-bit v210. SDR v210 uses steady SMPTE colour bars; it is not HDR.

### Generated media

Every generated source gets its own file: its source ID, e.g. `SDR420_RAW_007`,
is burned into every frame on a plate that circles a spot of its own and
closes on the loop. No two sources share a file, a picture or page cache, and
a frozen source shows even on steady bars. Browser sources using the alpha
pattern show their own ID in the page header. Clips have the generation size
(the setup page uses 1920×1080, shown contained on portrait canvases) and the
canvas frame rate.

| Storage | Per source, 1920×1080, 2 s at 25 fps | At 50 fps | Read rate at 25 fps |
| --- | ---: | ---: | ---: |
| H.264 / HEVC (NVDEC) | ≤ 3 MB (12 Mb/s) | ≤ 3 MB | — |
| NV12 | 156 MB | 311 MB | 78 MB/s |
| P010 | 311 MB | 622 MB | 156 MB/s |
| v210 | 276 MB | 553 MB | 138 MB/s |

Raw inputs are read every frame, so the host either keeps all their files in
the page cache or reads them from disk at the combined rate. Short loops limit
disk space, not playback load. HLG patterns render on the CPU, a few seconds
per source; other patterns take about a second. Only allocated input categories
are prepared. File names hold the source ID and everything that shapes the
content, so changing layout weights or scene count reuses media, while size,
cadence, duration and storage choices get new files under
`media/assets/synthetic_v3_<size>_<fps>fps_<seconds>s/`. Nothing is pruned:
delete unused files, or older `synthetic_v1_*`/`synthetic_v2_*` directories,
to reclaim space; delete an asset to regenerate it. An interrupted preparation does
not publish a partial asset. `media/media_wipes/` holds two generated alpha
wipes: **Diagonal sweep** and **Sliding panels**. Moving colour bands make
frame pacing visible even while the graphic covers the scene-switch midpoint.
Both last two seconds, use the selected FPS and preserve the canvas aspect
ratio, with the longer edge capped at 960 pixels to limit decode and upload
cost. The compositor scales them to the canvas. No external wipe assets or
downloads are required, and the mixer caches both decoded in GPU memory by
default (`--wipe-cache-mb 640`).

## Scene compositions

| Layout | Composition |
| --- | --- |
| `fullscreen` | One source, rotating through the source pool. |
| `grid_2`, `grid_4`, `grid_8`, `grid_16`, `grid_32`, `grid_64` | Even box grids; orientation follows the canvas. Fewer sources leave unused cells. Portrait `grid_32` is 4 columns × 8 rows; `grid_64` is 8 × 8. |
| `pip` | A full-frame background with up to three randomly positioned small insets. |
| `random` | A full-frame background with up to seven overlapping boxes of varied sizes and positions. |
| `alpha_overlay` | A video background with a full-frame browser graphic blended using its alpha. Requires allocated video and browser inputs. |

All boxes stay inside the canvas with even coordinates and dimensions. Later
items draw above earlier items; sources are not repeated within a scene.
The mixed-source presets include `grid_32` and `grid_64` with zero weights.
Set those weights above zero when using larger source pools; scene allocation
still respects `scene_count`, and a 64-source grid shows all 64 inputs at once.
If only one source is allocated, PiP/random layouts reduce to fullscreen.
Scenes have distinct IDs, but repeated geometric arrangements are possible.
`alpha_overlay` needs a page with an actually transparent background; an ordinary
opaque website stays opaque. It uses source alpha, not adjustable video opacity.
The synthetic-only `demo.example.json` disables browsers and alpha-overlay scenes.
The Compose stack supplies browser capture for `demo.equal.json`.

For a browser transparency check, [`demo.browser-alpha.json`](../demo.browser-alpha.json)
creates SDR and HLG backgrounds and one transparent browser source, with four
scenes to compare the backgrounds with and without the overlay. The page shows
0/25/50/75/100% opacity patches and a moving half-opacity marker. Empty areas
should show the background unchanged; 100% patches should hide it completely.

## Movie inputs

The presets ship no movies. A `download` or `file` entry feeds exactly one
source: the preparer rejects a weight that allocates two sources to one clip,
or two entries naming the same clip. For a fair workload use clips of at least
1920×1080 at 25 fps or more, one per source; the preparer does not probe them.
Record attribution and license beside each URL in the recipe.
[Apple's HDR streaming examples](https://developer.apple.com/streaming/examples/)
are HLS manifests, not direct movie downloads; use a cached encoded clip as a
`file` input.
