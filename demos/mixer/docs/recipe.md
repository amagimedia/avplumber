# Demo recipe

A recipe describes the workload to generate. The preparer creates missing media,
then writes an ordinary [mixer show](config.md). The [Compose quick start](../README.md#run)
prepares it automatically. For a standalone NVIDIA environment:

```sh
python3 demos/mixer/prepare_demo.py media/demo.json --media-dir media
python3 demos/mixer/mixer.py --config media/mixer.demo.json --janus-output
```

Run preparation on the NVIDIA host or inside the mixer Docker image. It needs
Python, NumPy and FFmpeg, included in that image. For CPU-only preparation,
set `generation.sdr_encoder` to `libx264` and `generation.hdr_encoder` to
`libx265` in a copy of the recipe; FFmpeg must have those encoders. Playback
still requires the NVIDIA mixer environment. `--runtime-media-dir /media`
lets host preparation write paths for a later `/media` container mount.

## Counts and proportions

| Recipe field | Meaning |
| --- | --- |
| `source_count` | Total independent input chains, 1–64. Several may read the same cached clip, but each has its own decoder or browser window. This is separate from boxes visible in a scene. |
| `scene_count` | Total named scenes to generate, independent of source count. |
| `inputs[].weight` | Relative share of the input total. Zero disables the entry, including download/preparation. |
| `layouts` | Layout names mapped to relative shares of the scene total. |
| `seed` | Integer controlling random compositions. The same recipe and seed reproduce the same scenes. |
| `alpha_background` | Optional allocated video source ID for every transparency scene. The equal recipe uses steady colour bars. |
| `canvas` | Normal mixer canvas settings: dimensions, fps, working format and color. Also sets synthetic clip dimensions and cadence. |
| `generation.seconds` | Length of each synthetic input loop; default 2 seconds. |
| `generation.width`, `generation.height` | Optional synthetic source dimensions, independent of the canvas. Defaults to the canvas dimensions. |
| `generation.sdr_encoder`, `generation.hdr_encoder` | Preparation encoders; default `h264_nvenc`, `hevc_nvenc`. |
| `renditions` | Normal mixer output definitions, copied to the generated show. Use separate RTP/RTCP port pairs. |

Weights need not add to 100. Largest-remainder rounding makes counts add to the
requested total; ties follow recipe order. A positive weight can round to zero
when the total is small. For exact counts, make weights sum to the desired total.
Preparation prints the allocated input counts before generating media.

The synthetic-only startup example's weights 8:4:2:2 at 16 sources give eight SDR 4:2:0, four HLG
4:2:0, two HLG 4:2:2, and two SDR 4:2:2 inputs. Increasing `scene_count` creates
more scene definitions, not additional input chains. Increasing `source_count`
opens more independent chains, reusing asset files when a pattern pool repeats.

## Example presets

The two mixed-source presets have the same canvas (1080×1920 at 60 fps),
32 scenes and layout weights. Their default 16-source allocations are:

| Input entry | `demo.equal.json` | `demo.cinematic.json` |
| --- | ---: | ---: |
| `sol_pq` — downloaded PQ movie | 0 | 1 |
| `sol_hlg` — downloaded HLG movie | 0 | 1 |
| `sdr420` — generated SDR 4:2:0 | 3 | 3 |
| `hlg420` — generated HLG 4:2:0 | 3 | 1 |
| `hlg422` — generated HLG 4:2:2 | 3 | 3 |
| `sdr422` — generated SDR 4:2:2 | 3 | 3 |
| `bunny` — downloaded SDR movie | 2 | 2 |
| `browser` — transparent browser pattern | 2 | 2 |

`equal` gives its six entries equal weights, with rounding to fit 16 sources.
`cinematic` includes movies, patterns and browsers; it is not a movies-only
preset. Both use the existing pattern generators and require the DMA-BUF browser
stack. Movie URLs, attribution and license links live beside their input weights
in each recipe. Downloads are cached under `media/assets/downloads/`.

## Custom preset

Copy either preset to `media/demo.json` and edit `source_count`, `scene_count`,
`canvas.fps` and the source/layout weights. Apply it with
`docker compose -f demos/mixer/compose.yaml restart mixer`.

For example, set `source_count` to **42**, `scene_count` to **64**, keep
`canvas.fps` at **60**, and change the existing input weights to:

| Input ID | Weight | Sources at this total |
| --- | ---: | ---: |
| `sol_pq` | 5 | 5 |
| `sol_hlg` | 2 | 2 |
| `sdr420` | 10 | 10 |
| `hlg420` | 1 | 1 |
| `hlg422` | 4 | 4 |
| `sdr422` | 0 | 0 |
| `bunny` | 10 | 10 |
| `browser` | 10 | 10 |

These weights sum to 42, so they are exact counts at this total. Changing only
`source_count` scales the same proportions with rounding. Weights such as
50:25:25 work as percentages too; no specific sum is required. Each allocated
source has its own input chain, even when multiple sources reuse one clip.
This example uses the public Sol Levante downloads, not a pre-existing local
movie file. Its allocation is a workload description, not a performance guarantee.

Both presets set `alpha_background` to `sdr420_001`, the steady bars pattern.
Keep at least two allocated `sdr420` sources to retain that background. For a
smaller total or different mix, you can set that entry's `pattern` to `"bars"`
and `alpha_background` to `"sdr420_000"`; all sources in that entry then use
steady bars. If you disable browsers, also set `layouts.alpha_overlay` to zero.
Remove `alpha_background` if its source entry is disabled. Allocated alpha
scenes always need at least one browser and one video source.

The generated show is `media/mixer.demo.json`; keep your editable recipe separate.
Scene-only changes reuse cached assets; changing FPS creates new synthetic clips
and wipes. Downloaded films retain their original cadence and are adapted at playback.
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
| `browser` | `url` or `pattern: "alpha"`; optional `width`, `height` | Opens live SDR browser windows using the DMA-BUF browser stack. The alpha pattern embeds the bundled test page, with no HTTP server or external website. Dimensions default to the canvas. |

SDR 4:2:0 defaults to seven H.264 patterns: `testsrc2`, `bars`, `rgbtest`,
`mandelbrot`, `gradients`, `life`, `sierpinski`. The cellular-noise pattern
`cellauto` remains available through an explicit `pattern` or `patterns` choice
for encoder stress testing; it is excluded from the default rotation because
large noisy regions can produce substantial keyframe bursts. The other categories have patterns
`"0"` and `"1"`. HLG uses the smooth wide-gamut bars, luminance sweep and orbiting
highlights demo, generated in scene-linear light through the HLG transfer
function; HLG 4:2:0 is encoded HEVC Main10. Both 4:2:2 categories are
raw 10-bit v210. SDR v210 uses steady SMPTE colour bars; it is not HDR.

Raw v210 consumes substantial disk space and read bandwidth: roughly 75 MB/s
per 1280×720 input at 30 fps. Short loops limit disk space, not playback load.
Only allocated input categories are prepared. Changing layout weights or scene
count reuses media; size, cadence, duration and encoder choices have separate
cache paths. Delete an asset to regenerate it. An interrupted preparation does
not publish a partial asset. `media/media_wipes/` holds the generated alpha wipe.

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
The Compose stack supplies browser capture for the two mixed presets.

For a browser transparency check, [`demo.browser-alpha.json`](../demo.browser-alpha.json)
creates SDR and HLG backgrounds and one transparent browser source, with four
scenes to compare the backgrounds with and without the overlay. The page shows
0/25/50/75/100% opacity patches and a moving half-opacity marker. Empty areas
should show the background unchanged; 100% patches should hide it completely.

## Public movie examples

The mixed presets include a short Big Buck Bunny MP4 from
[test-videos.co.uk](https://test-videos.co.uk/bigbuckbunny/mp4-h264). Its weight
controls how many independent inputs read the cached file. The film is public under CC BY 3.0; retain the
[Blender Foundation attribution](https://peach.blender.org/about/) when sharing
it. The recipe records the attribution and license beside the URL.

[`demo.cinematic.json`](../demo.cinematic.json) keeps the same portrait 60 fps,
16-source, 32-scene workload and includes direct HTTP downloads of **Sol Levante**
in PQ and HLG, plus Bunny. The two HDR clips use the HEVC 10-bit compatible base
layers of Dolby's profile 8.1 and 8.4 test files; this does not apply Dolby Vision
dynamic metadata. They occupy two of the three HDR 4:2:0 inputs, alongside one
generated HLG input; smooth HLG 4:2:2 patterns remain included. The first two
fullscreen scenes show the film. Downloads are cached once.

[Netflix Open Content](https://opencontent.netflix.com/) supplies Sol Levante
under CC BY 4.0; the compatible test encodes are hosted by
[Dolby Laboratories](https://github.com/DolbyLaboratories/dolby-vision-contents).
Attribution and license links are in the recipe.
[Apple's HDR streaming examples](https://developer.apple.com/streaming/examples/)
are also available for testing, but their HLS manifests are not direct movie
downloads. Use a cached encoded clip as a `file` input. Synthetic-only startup
remains available through `demo.example.json`.
