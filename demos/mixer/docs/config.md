# Mixer configuration

[Quick start](../README.md) · [Example show](../config.example.json) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)

`--config mixer.json` supplies sources, scenes, canvas and outputs. `canvas`,
`sources` and `scenes` are required. The [parser](../../../pyplumber/mixer/config.py)
is the validation authority; the example is the full editable starting point.

Top-level `initial_scene` defaults to the first scene. `max_compositor_layers`
defaults to 640. `browser_ring_size` is 1–64 (6 at 25/30 fps, otherwise 9).
`wipe_color: "sdr"` overrides wipe tags; `--wipe-color` takes precedence.

## canvas

HLG/PQ require 10-bit storage. Browser graphics are SDR; the compositor converts them to the canvas color.

| field | default | meaning |
| --- | --- | --- |
| `width`, `height` | — | the program raster the compositor draws into |
| `fps` | `30` | **how often the compositor renders**, and the clock the whole mixer runs on: inputs are re-timed to it and browser pages are asked to paint at it |
| `working_format` | `nv12` | compositor and transition pixel storage: `nv12` (8-bit 4:2:0), `p010le` (10-bit 4:2:0) or `p210le` (10-bit 4:2:2). 8-bit sources are promoted onto a 10-bit canvas; `p210le` keeps 4:2:2 through conversion and compositing. Renditions are 4:2:0 for NVENC, subsampled once |
| `raw_upload` | `hwupload` | `hwupload` or `pinned` (`hwupload_cuda=pinned=1`, requires the FFmpeg patch); Setup uses pinned staging. [Details](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/raw-uploads.html) |
| `color` | `sdr` | canvas color contract: `sdr` (BT.709), `hlg` or `pq` (BT.2020). HLG/PQ need a 10-bit `working_format`. Every source is converted to it on the GPU; renditions convert from it |
| `latency_ms` | 2 frames up to 30 fps, 3 at 50/60 | playout buffer between a source frame's arrival and its tick (80 ms at 25 fps, 50 ms at 60). A frame later than that is skipped and the previous one repeated, so set it just above the worst source jitter; must stay below six frames. `--mixer-latency-ms` on the command line overrides it |

## renditions

The program is composited once. Each rendition adds an encode. With no renditions, CLI output flags apply. Each Janus output needs a separate RTP/RTCP pair and matching mountpoint; Setup manages these, standalone callers must supply them.

| field | default | meaning |
| --- | --- | --- |
| `id` | — | unique; names the nodes and edges of this output |
| `target` | `"janus"` | `"janus"` for the WebRTC RTP output, or a file path to record |
| `width`, `height` | canvas size | when they differ, both must be even; scaled on the GPU by `cuda_transform`, stretched to the given size (bilinear over four neighbours, as the compositor: a large reduction can alias). Renditions of one size and feed share one scaled picture; color conversion follows on that smaller picture |
| `aspect` | — | optional, e.g. `"9:16"`; checked against `width:height` |
| `fps` | canvas fps | may only re-time **downwards**; a higher rate is rejected |
| `bitrate_kbps` | `3000` | CBR target, also the `maxrate` and `bufsize` |
| `codec` | from canvas depth | `h264_nvenc` for 8-bit or `hevc_nvenc` for 10-bit by default; applies to Janus and file targets |
| `profile` | from codec/depth | HEVC `main`/`main10`; H.264 `baseline` for Janus, `high` for files. No B-frames |
| `preset` | `"p7"` | NVENC quality preset. The setup page sets p1, p3 or p5 on every output it generates ([NVENC costs](capacity.md#nvenc-and-extra-aux-outputs)) |
| `port` | — | Janus target: overrides the RTP port from the command line |
| `color` | automatic | `sdr`, `hlg`, or `pq`; H.264 always requires SDR, HEVC otherwise inherits the canvas |
| `feed` | `"dirty"` | `"clean"` encodes the program without the [downstream keys](#dsk); without keys both are the program |
| `tonemap` | none | requests an SDR output from an HDR canvas: `clip` (exact SDR, hard-clipped highlights), `mobius` (see `tonemap_param`), `hable`, `reinhard`, `gamma`, `linear`, `none` |
| `tonemap_peak` | `10` | HDR peak in units of 100 nits, minimum `2.03` |
| `tonemap_desat` | `0` | highlight desaturation; `0` keeps saturation |
| `max_cll`, `max_fall` | derived | PQ metadata in nits: MaxCLL = `tonemap_peak` × 100, MaxFALL = 40% of MaxCLL; MaxFALL cannot exceed MaxCLL. Requires compatible NVENC headers/driver. |
| `tonemap_param` | `0` | operator knee in reference-white units; `0` = operator default (0.3 mobius/reinhard, 1.8 gamma). mobius must be below 1.0; `0.9` keeps 90% of SDR white untouched |
| `dpb_size` | `0` | NVENC reference frames kept, 0–16; `0` lets NVENC choose. Without B-frames `1` is enough and frees the other reference surfaces; it may cost quality at low bitrates, so measure before setting it on a program output. Aux monitors default to `1` |

## sources

One source ID opens one decoder or browser window; scenes and aliases reuse it. Repeated locations require `independent: true` on every declaration. Browser import is provided by the [shared DMA-BUF integration](../../../doc/dmabuf.md). CUarray requires compatible FFmpeg/consumers and a fixed surface budget; it has no copy fallback or decoder auto-restart. See the [CUarray cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/nvdec-cuarray.html).

| field | applies to | meaning |
| --- | --- | --- |
| `id` | both | referenced from scenes; no `#` |
| `kind` | all | `"video"`, `"browser"`, `"nv12"` (raw SDR 8-bit 4:2:0), `"p010"` (raw 10-bit 4:2:0), or `"v210"` (headerless packed 10-bit 4:2:2, e.g. generated HDR test content) |
| `path` | video, v210, nv12, p010 | file or stream |
| `width`, `height` | v210, nv12, p010 | required: raw bytes carry no header; NV12/P010 dimensions must be positive and even |
| `color` | browser, v210, nv12, p010 | required color contract (`sdr`, `hlg`, `pq`); browser and NV12 sources must be `sdr`. Optional for `video`: by default the decoded frame tags decide and untagged files are treated as BT.709 SDR |
| `url`, `width`, `height` | browser | page and the window it is rendered in (all three required) |
| `fps` | browser | paint rate; defaults to the canvas rate |
| `hold_last_frame` | browser | default `true`: while a failed or crashed page reloads, repeat its last frame. `false` shows Chromium's empty error page (transparent or black) instead |
| `width`, `height` | video | optional; probed with ffprobe at load when absent |
| `loop` | both | default `true` |
| `decode_storage` | video | default `cuda`; experimental `cuarray` requires the patched upstream FFmpeg build and compatible array consumers, and selects zero-copy NVDEC without CPU fallback |
| `extra_hw_frames` | CUarray video | default `3`: fixed application headroom beyond codec and FFmpeg working surfaces; size it for the graph's retained-frame demand |
| `transform` | video/v210/nv12/p010 | optional geometry applied once before the source filter, color and fan-out: `{"width", "height", "fit", "crop", "sw_format"}`. The frame, or its `crop` `[x, y, w, h]`, is scaled onto a canvas of that even size by `cuda_transform`; `fit` is `contain` (default, black bars) or `stretch`. `sw_format` (`nv12`, `p010le`, `p210le`) is the source's own storage: raw kinds supply it, a `video` source must state it. A frame that already has the size and storage is passed on undrawn; one of lower depth is promoted, one of higher depth is an error |
| `filter` | video/v210/nv12/p010 | optional CUDA source graph, before automatic normalization; preserve dimensions and correct output metadata |
| `filter_output_format` | custom filters | required CUDA YUV output storage, e.g. `p010le` or `p210le` |

## wipes

Alpha clips must cover the picture at the switch midpoint. `wipe_dir` adds clips by filename. `--wipe-cache-mb` defaults to 640; zero decodes per take. Incomplete clips or an insufficient cache budget fail startup.

| field | default | meaning |
| --- | --- | --- |
| `id` | — | referenced by `control.default_wipe` |
| `path` | — | clip with alpha, on the mixer host |
| `name` | the id | label shown on the web UI button and in the TUI picker |
| `duration_seconds` | probed | how long the take runs |

## control

Starting defaults for the web UI and TUI; operator choices take precedence. Published as `mixer.settings`.

| field | default | meaning |
| --- | --- | --- |
| `direct` | `true` | a scene pick goes straight to program rather than loading preview |
| `transition` | `"cut"` | what a direct-mode pick takes with: `cut`, `fade` or `wipe` |
| `fade_seconds` | `0.5` | length of a fade |
| `fade_curve` | `linear` | `linear`, `ease-in`, `ease-out`, `ease-in-out`; per-take `curve` overrides it. |
| `fade_color` | `null` | `null` mixes; `#RRGGBB` dips through a color at the midpoint. Per-take `color` overrides it. |
| `default_wipe` | first wipe | the clip a direct-mode wipe uses |

## scenes

Each scene has an `id` and ordered `items`; later items draw above earlier ones. Padding is black. `cover` requires a known source size.

| field | default | meaning |
| --- | --- | --- |
| `source` | — | a source id |
| `dst` | — | `x`, `y`, `w`, `h` on the canvas |
| `fit` | `"contain"` | `stretch`, `contain` (letterbox/pillarbox), `cover` (fill and crop the overflow) |
| `blend` | `false` | honour source alpha, for example a transparent browser graphic over video; opaque sources remain opaque |
| `crop` | whole frame | region of the source in source pixels, applied before the fit |

## dsk

Up to four browser keys over the finished program. `feed: "clean"` omits them. Keys count as sources and stay subscribed while off. Change a key with `mixer.dsk {"key": "bug", "on": true}`. The first table describes the DSK block; the second describes each entry in `keys`.

| field | default | meaning |
| --- | --- | --- |
| `fade_seconds` | `0.4` | how long a key change fades, 0 to 10; `0` cuts |
| `fade_curve` | `"linear"` | easing of a key fade, the same presets as `control.fade_curve` |


| field | default | meaning |
| --- | --- | --- |
| `id` | — | unique key name, used by `mixer.dsk` and the control page |
| `source` | — | a **browser** source id; its alpha is always kept. Scenes may use the same source |
| `dst` | whole canvas | `x`, `y`, `w`, `h` on the canvas; the window is scaled into it |
| `on` | `false` | on air at start |

## aux_buses

Each bus has an `id`, layouts, scene assignments and Janus renditions (`port`, optional `codec`, `preset`, `bitrate_kbps`). It reuses source frames and drops output rather than blocking Program. SDR only; half canvas rate above 30 fps unless `full_rate`. Commands: `mixer.aux_layout`, `mixer.aux`, `mixer.aux_page`, `mixer.aux_status`. Assignment updates include the current `expected_revision`. Layout changes stage new sources; status exposes `composition_pending` and `composition_error`. See [timing details](../../../doc/mixer.md#aux-bus-follower).

| role | draws |
| --- | --- |
| `pvw` | the mixer's preview scene, changed with the program on a take (see `pvw_align`) |
| `pgm` | the finished, keyed program, `pgm_delay_frames` behind the other cells |
| `slot` (+ `"slot": n`) | the scene assigned to slot `n` |
| `source` (+ `"source": i`) | source `i` (its index in `sources`) |


| layout | cells |
| --- | --- |
| `{"preset": "pgm_pvw_grid"}` (default) | PVW and PGM on top, slots 0–7 below in two rows of four |
| `{"preset": "source_pages", "page": 0}` | one page of source cells, 12 per page (2 × 6 portrait, 4 × 3 landscape): page `page`, 0 by default |
| `{"cells": [{"role": "slot", "slot": 0, "x": 0, "y": 0, "w": 540, "h": 960}, ...]}` | exactly these: even `x`, `y`, `w`, `h` inside the canvas, slots numbered 0 to n−1, each once; any number of `pvw` and `pgm` cells, including none |


| field | default | meaning |
| --- | --- | --- |
| `layout` | `{"preset": "pgm_pvw_grid"}` | the layout the bus starts with |
| `layouts` | `source_pages`, and `pgm_pvw_grid` when `layout` has a `pgm` cell | further layouts the control page offers besides `layout` (at most 16) |
| `scenes` | none | slot assignments by slot index, `null` for an empty slot; padded with `null` to the layout's slot count |
| `max_layers` | `max_compositor_layers` | Fixed layer budget for PVW reserves, slots, sources and PGM; oversized layouts are rejected. |
| `latency_ms` | program latency | AUX playout buffer; combined with PGM delay must remain below six AUX frames. |
| `pgm_delay_frames` | 1; 2 at 50/60 fps AUX | Delay only the PGM tile. Bus latency + delay must exceed program latency by at least one program frame, and remain below six AUX frames. |
| `pvw_align` | `program` | Change PVW alongside Program, or `pgm_tile` to align with the delayed PGM cell. |
| `label` | `Program preview`, `Multiviewer` or `Aux <id>` by its initial layout | the bus's name on the control page |
| `full_rate` | `false` | Keep canvas rate above 30 fps; increases compositor/encoder work. |

## Known limitations

- **193 sources per show, or 192 with an aux bus that can draw the program.** Every
  source is a pad on the compositor; a bus with a `pgm` cell in its `layout` or
  `layouts` (by default, a `pgm_pvw_grid` bus) reserves one additional pad for PGM. 8-bit or 10-bit makes no difference;
  scenes and aliases are free. Sources cannot be added while running: the
  pads are wired at build time. A document with more sources is rejected at load.
- **16 boxes per scene** in the built-in `--input` layouts; a `--config` scene has no box limit.
- **All sources run all the time.** A source not on screen is still decoded
  and converted. Budget GPU for the whole catalogue, not the scene.
- **`--input` with more than 32 files** switches to a router that converts
  only the 16 on-screen sources. It needs all inputs in one size, format and
  frame rate, and it does not take `v210` or browser sources.
- **NVENC encodes 4:2:0.** A `p210le` canvas keeps 4:2:2 through compositing;
  every rendition is 4:2:0.
- **HLG and PQ need a 10-bit canvas.** Browser pages are SDR only.

More than 193 pads and runtime source addition are not supported.

## Generating one

[make_config.py](../../../pyplumber/mixer/tools/make_config.py) emits the built-in layouts as JSON:

```sh
python3 -m pyplumber.mixer.tools.make_config --fps 30 cam0=/media/camera.mp4 > mixer.json
```

For generated workloads use a [recipe](recipe.md). Other mixer applications use
`pyplumber.mixer.build_application` directly; see the [base API](../../../doc/mixer.md).

## Input cadence and colour-tag guarantees

Mixer file sources retain native cadence; the compositor samples them at the
canvas rate. Keep rational source rates exact (for example `30000/1001` or
`60000/1001`) instead of substituting decimal approximations. The input
`realtime` node still paces/rebases timestamps, using a 1/120000 time base.
Per-input `force_fps` is omitted; output/AUX normalization remains. The shared
input helper's default normalized behavior for other applications is unchanged.
Browser sources retain their separate timestamp smoothing/repeat chain.

`canvas.raw_upload: "pinned"` uses patched FFmpeg's `hwupload_cuda`; the old
custom raw uploader nodes are retired. NV12/P010 and v210 builders stamp declared
YUV tags in the upload graph. The application bypasses a standalone tag-only
colour filter only when no user filter or processing callback can invalidate
that promise. Real colour conversion and alpha processing remain. A source
`transform` preserves the tags, while an arbitrary `filter_graph` or
`process_source` callback disables automatic eligibility.

The library's `color_tagged` source contract is not a request to infer colour
from content. See [the mixer API](../../../doc/mixer.md#pacing-loops-and-redundant-colour-tags)
when integrating custom inputs in another repository.
