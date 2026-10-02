# Mixer configuration file

One JSON document describes a mixer run: the sources it opens, the wipe clips
it plays, the scenes an operator can take, the defaults its control surfaces
start from, and the encoded outputs it produces. Nothing about 2/4/8/16-box
layouts lives in code — a grid is a scene somebody wrote or generated.

```sh
python3 demos/mixer/mixer.py --config mixer.json --janus-output
```

`--config` replaces `--input` and the built-in layouts. The design rationale is
in the [schema note](../../../doc/research/2026-09-08-mixer-config-schema.md);
this file is the reference.

[`config.example.json`](../config.example.json) is a generic schema example, not
a deployed source list. Replace its placeholder URLs and paths in your own show
file, kept outside the public checkout. Source IDs, source count and scenes are
supplied by that file; the runtime does not depend on the recorded demo's inputs.

## Shape

```json
{
  "canvas":       { "width": 1080, "height": 1920, "fps": 30 },
  "renditions":   [ ... ],
  "sources":      [ ... ],
  "wipes":        [ ... ],
  "wipe_dir":     "/media/media_wipes",
  "control":      { "direct": true, "transition": "cut", "fade_seconds": 0.5 },
  "scenes":       [ ... ],
  "initial_scene": "fullscreen_0"
}
```

`canvas`, `sources` and `scenes` are required; everything else has a default.

`max_compositor_layers` is the per-compositor draw budget (default 256), and
an aux bus's default `max_layers`. Eight 64-input slots of a
`pgm_pvw_grid` aux bus need 577 layers: 512 for the slots, 64 reserved for PVW
and one for the already-composited PGM. Set the budget to `640`, or
override a show with `--max-compositor-layers 640`. Recipes accept the same field; setup changes
preserve it. Native compositor nodes call this parameter `max_layers`.
The CUDA layer table allocates 128 bytes per configured layer on both the GPU
and pinned host memory. The culling mask and per-frame work use the actual
layer count; raising the budget does not allocate more video frames.

AUX scene changes briefly subscribe to the union of the current and latest requested
layout. The old layout stays live until new inputs are ready for the output clock,
then all tiles switch together. Preparation times out after the larger of 250 ms
and twice the AUX latency budget, keeping the previous layout. Superseded requests
and unused subscriptions are released; this does not prewarm every source. The bus
compositor's status reports `composition_pending` while a layout waits and
`composition_error` when one timed out.

`browser_ring_size` is a top-level integer (1–64, default 6 at 25/30 fps and 9
otherwise) limiting outstanding DMA-BUF frames per browser. The CUDA import cache
keeps room for at least 32 allocation handles independently of the outstanding
frame limit. Returning allocations refresh their registrations before expiry;
frames still in use keep their registration discoverable. Idle imports expire
after four ring cycles (at least one second), so replaced allocations are not
pinned indefinitely.
Cached handles do not withhold frame-release acknowledgments. For example,
`"browser_ring_size": 6` uses a six-frame ceiling for every browser source.
Smaller limits can reduce VRAM but drop paints when downstream retains all slots.
The browser service must support the `ringSize` and `holdLastFrame` window options; update it together
with the mixer. The setup page exposes the same setting and resets it to the default when FPS changes. Without a JSON config,
use `--browser-ring-size` with `--dmabuf-open`.

## Complete example

Every meaningful option in one HDR show: three source kinds, two outputs, a wipe
library and a scene with each fit mode. Paths and URLs are placeholders.

```json
{
  "canvas": {"width": 1920, "height": 1080, "fps": 60,
             "working_format": "p210le", "color": "hlg", "latency_ms": 50},
  "sources": [
    {"id": "movie", "kind": "video", "path": "/media/hdr-movie.mp4", "loop": true},
    {"id": "cam", "kind": "video", "path": "srt://10.0.0.5:9000?mode=caller", "loop": false,
     "color": "sdr", "width": 1920, "height": 1080},
    {"id": "graded", "kind": "video", "path": "/media/graded.mp4",
     "filter": "tonemap_cuda=transfer_in=pq:transfer_out=hlg,scale_cuda=format=p210le",
     "filter_output_format": "p210le"},
    {"id": "bars", "kind": "v210", "path": "/media/bars.v210", "width": 1920, "height": 1080,
     "color_trc": "arib-std-b67", "color_primaries": "bt2020", "colorspace": "bt2020nc", "color_range": "tv"},
    {"id": "page", "kind": "browser", "url": "https://example.org/lower-third",
     "width": 1920, "height": 1080, "fps": 60, "color": "sdr"}
  ],
  "wipes": [{"id": "ribbons", "path": "/media/media_wipes/ribbons.mov", "name": "Ribbons", "duration_seconds": 2.0}],
  "wipe_dir": "/media/media_wipes",
  "wipe_color": "sdr",
  "control": {"direct": true, "transition": "wipe", "fade_seconds": 0.5, "default_wipe": "ribbons"},
  "renditions": [
    {"id": "program", "target": "janus", "port": 5006, "width": 1920, "height": 1080, "aspect": "16:9",
     "fps": 60, "bitrate_kbps": 8000, "codec": "hevc_nvenc", "profile": "main10", "preset": "p5"},
    {"id": "sdr", "target": "janus", "port": 5004, "fps": 60, "bitrate_kbps": 6000,
     "codec": "h264_nvenc", "profile": "baseline", "preset": "p5",
     "tonemap": "mobius", "tonemap_param": 0.9, "tonemap_peak": 10, "tonemap_desat": 0},
    {"id": "archive", "target": "/recordings/program.ts", "fps": 30, "bitrate_kbps": 12000,
     "codec": "hevc_nvenc", "profile": "main10", "color": "pq", "max_cll": 1000, "max_fall": 400}
  ],
  "scenes": [
    {"id": "pip", "items": [
      {"source": "movie", "dst": {"x": 0, "y": 0, "w": 1920, "h": 1080}, "fit": "cover"},
      {"source": "cam", "dst": {"x": 1180, "y": 620, "w": 680, "h": 400}, "fit": "contain",
       "crop": {"x": 240, "y": 0, "w": 1440, "h": 1080}},
      {"source": "page", "dst": {"x": 0, "y": 0, "w": 1920, "h": 1080}, "fit": "stretch"}
    ]},
    {"id": "bars_full", "items": [{"source": "bars", "dst": {"x": 0, "y": 0, "w": 1920, "h": 1080}}]}
  ],
  "initial_scene": "pip"
}
```

## canvas

| field | default | meaning |
| --- | --- | --- |
| `width`, `height` | — | the program raster the compositor draws into |
| `fps` | `30` | **how often the compositor renders**, and the clock the whole mixer runs on: inputs are re-timed to it and browser pages are asked to paint at it |
| `working_format` | `nv12` | compositor and transition pixel storage: `nv12` (8-bit 4:2:0), `p010le` (10-bit 4:2:0) or `p210le` (10-bit 4:2:2). 8-bit sources are promoted onto a 10-bit canvas; `p210le` keeps 4:2:2 through conversion and compositing. Renditions are 4:2:0 for NVENC, subsampled once |
| `raw_upload` | `hwupload` | how `nv12`/`p010` sources reach the GPU: `hwupload` (rawvideo frames paced on the CPU, then FFmpeg `hwupload`) or `pinned` (`raw_to_cuda`: pinned staging on a private CUDA stream, one uploaded frame held ahead of pacing). The library default stays `hwupload`; the generic setup's recipe selects `pinned`, which at 68 inputs and 60 fps missed no program deadlines where `hwupload` missed 69% ([raw uploads](cookbook/raw-uploads.html)) |
| `color` | `sdr` | canvas color contract: `sdr` (BT.709), `hlg` or `pq` (BT.2020). HLG/PQ need a 10-bit `working_format`. Every source is converted to it on the GPU; renditions convert from it |
| `latency_ms` | 2 frames up to 30 fps, 3 at 50/60 | playout buffer between a source frame's arrival and its tick (80 ms at 25 fps, 50 ms at 60). A frame later than that is skipped and the previous one repeated, so set it just above the worst source jitter; must stay below six frames. `--mixer-latency-ms` on the command line overrides it |

The canvas rate is the single biggest load knob. Halving it from 60 to 30 on
the sixteen-source demo took the T4 from ~33% to ~17% GPU.

## renditions

Each entry is one encoded output. The program is composited **once**; a
rendition re-times and rescales that picture for its own target, so a second
output costs an encode, not another composite.

| field | default | meaning |
| --- | --- | --- |
| `id` | — | unique; names the nodes and edges of this output |
| `target` | `"janus"` | `"janus"` for the WebRTC RTP output, or a file path to record |
| `width`, `height` | canvas size | scaled on the GPU (`scale_cuda`) when they differ |
| `aspect` | — | optional, e.g. `"9:16"`; checked against `width:height` |
| `fps` | canvas fps | may only re-time **downwards**; a higher rate is rejected |
| `bitrate_kbps` | `3000` | CBR target, also the `maxrate` and `bufsize` |
| `codec` | from canvas depth | `h264_nvenc` for 8-bit or `hevc_nvenc` for 10-bit by default; applies to Janus and file targets |
| `profile` | from codec/depth | HEVC `main`/`main10`; H.264 `baseline` for Janus, `high` for files. No B-frames |
| `preset` | `"p7"` | NVENC quality preset |
| `port` | — | Janus target: overrides the RTP port from the command line |
| `color` | automatic | `sdr`, `hlg`, or `pq`; H.264 always requires SDR, HEVC otherwise inherits the canvas |
| `feed` | `"dirty"` | `"clean"` encodes the program without the [downstream keys](#dsk); without keys both are the program |
| `tonemap` | none | requests an SDR output from an HDR canvas: `clip` (exact SDR, hard-clipped highlights), `mobius` (see `tonemap_param`), `hable`, `reinhard`, `gamma`, `linear`, `none` |
| `tonemap_peak` | `10` | HDR peak in units of 100 nits, minimum `2.03` |
| `tonemap_desat` | `0` | highlight desaturation; `0` keeps saturation |
| `max_cll`, `max_fall` | derived | HDR10 static metadata for **PQ** outputs, nits. Defaults: MaxCLL = `tonemap_peak`×100, MaxFALL = 40% of it; `max_fall` may not exceed MaxCLL. Needs nv-codec-headers 13 and driver ≥ 570 (`Dockerfile.cuda`), else the SEIs are silently absent. HLG needs none |
| `tonemap_param` | `0` | operator knee in reference-white units; `0` = operator default (0.3 mobius/reinhard, 1.8 gamma). mobius must be below 1.0; `0.9` keeps 90% of SDR white untouched |
| `dpb_size` | `0` | NVENC reference frames kept, 0–16; `0` lets NVENC choose. Without B-frames `1` is enough and frees the other reference surfaces; it may cost quality at low bitrates, so measure before setting it on a program output. Aux monitors default to `1` |

With no `renditions` the demo builds its usual single output from the command
line flags.

```json
"renditions": [
  {"id": "program", "target": "janus", "width": 1080, "height": 1920,
   "aspect": "9:16", "fps": 30, "bitrate_kbps": 2700,
   "profile": "baseline", "preset": "p7"}
]
```

Multiple Janus renditions need separate RTP ports and matching Janus
mountpoints. Each has its own encoder and RTCP feedback. The first keeps the
`janus_encoder` name used by cut-latency measurement; additional outputs use
`janus_<id>_encoder`.

Janus output paces RTP transmission at three times the configured average
bitrate, with a four-packet burst allowance. Packetization remains RTP; the
output uses FFmpeg's `udp://` transport because its `rtp://` wrapper does not
expose UDP pacing. RTCP feedback uses the separate feedback listener. Pacing
spreads large keyframes over time without changing encoded detail or bitrate;
their delivery can take tens of milliseconds. See the shared stack's
[socket buffer guidance](../../../docker-compose/README.md#rtp-burst-headroom)
if Janus drops packets before forwarding them.

For a P010 HLG canvas, these outputs provide simultaneous HDR and SDR previews:

```json
"renditions": [
  {"id": "hdr", "target": "janus", "port": 5006,
   "codec": "hevc_nvenc", "profile": "main10"},
  {"id": "sdr", "target": "janus", "port": 5004,
   "codec": "h264_nvenc", "profile": "baseline", "tonemap": "clip"}
]
```

`clip` preserves SDR content embedded into HLG with the same 203-nit white;
HDR highlights above that white are clipped. Operators such as `hable` compress
highlights and also change midtone brightness. A preview selector chooses
between the two continuously running outputs; the canvas remains HDR.

## sources

One entry per **unique** clip or page: two entries with the same `url` or
`path` are an error, because a source is decoded or captured exactly once
however many scenes and slots show it. A scene that shows the same source
twice fans the frames out under alias names (`id#2`, `id#3`), never a second
decoder.

```json
{"id": "cam0", "kind": "video",   "path": "/media/camera-0.mp4", "loop": true}
{"id": "page", "kind": "browser", "url": "https://example.org/live",
 "width": 1920, "height": 1080, "fps": 30, "color": "sdr"}
```

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
| `filter` | video/v210/nv12/p010 | optional CUDA source graph, before automatic normalization; preserve dimensions and correct output metadata |
| `filter_output_format` | custom filters | required CUDA YUV output storage, e.g. `p010le` or `p210le` |

Browser sources arrive over DMA-BUF from the `dma-page` service and are
imported straight into CUDA; they mix with video sources on the same canvas.

Color normalization is automatic in `MixerGraphBuilder`, before alias and scene
fan-out. Video sources use decoded frame metadata, including live SRT streams.
A `video` source without frame color tags is treated as BT.709 SDR. A declared
contract is either the `color` preset (which supplies all four fields) or a complete,
consistent set of `color_trc`, `color_primaries`, `colorspace` and `color_range`.
Raw `v210` and browser inputs must declare one.

For SDR Bunny on an HLG canvas, no manual tone-map filter is needed:

```json
"canvas": {"width": 1920, "height": 1080, "fps": 60,
           "working_format": "p010le", "color": "hlg"},
"sources": [{"id": "bunny", "kind": "video", "path": "<path>", "color": "sdr"}]
```

Supported video contracts are limited-range BT.709/BT.1886 SDR and BT.2020
non-constant-luminance HLG/PQ. Full-range YUV, other gamuts/matrices and missing
metadata fail explicitly. SDR uses a 203-nit reference white and HLG a 1000-nit
peak. On an HLG/PQ P210 canvas, 4:2:0 video stays P010 after colour conversion;
the compositor resamples chroma while scaling into the 4:2:2 canvas, without an
extra conversion pass. Native 4:2:2 inputs stay P210. Matching colour and storage
pass without GPU copies. Other declared CUDA YUV storage uses `scale_cuda`
around conversion. Custom source filters receive the
explicit source override first; their output metadata drives normalization, so
a manually converted source is not converted from its original transfer again.

Packed SDR RGB graphics retain alpha and convert in the compositor to its target
transfer/gamut. This path requires NV12, P010 or P210 canvas storage. HDR alpha
sources are unsupported. Missing transfer and primaries tags on RGB(A) wipes
default to SDR BT.709; explicit tags are preserved and validated. Set top-level
`wipe_color: "sdr"` to override the wipe library's color tags. The CLI option
`--wipe-color sdr` provides the same override and takes precedence over the JSON
setting. Neither the fallback nor the override changes alpha.

H.264 renditions and the default output path automatically convert HDR to SDR.
HEVC renditions inherit the canvas unless `color` requests another supported
contract. HDR outputs require 10-bit storage. The `clip` default preserves the
brightness of SDR embedded in HDR; it clips highlights above SDR white. Select
another operator explicitly when highlight compression is preferred.

Repeated file/URL declarations are rejected by default. For load testing, mark
**every declaration of the repeated location** with `"independent": true`:
each ID then opens its own input chain (or browser window). Referencing one ID
in several scenes still shares that one chain. The demo recipe uses this opt-in
for downloaded and file inputs, which several sources may read.

## wipes

The media-wipe library. By default startup decodes each clip once into a GPU
cache (`--wipe-cache-mb 640`; the demo's two 2 s 540x960 clips take about
0.5 GB), so a take neither decodes nor uploads: decoding QTRLE/ProRes alpha on
the CPU per take missed program deadlines at 64+ sources at 60 fps. The cached
player and the wipe compositor stay running and idle between wipes; a take
arms them in place, so wipe spam creates, starts and stops no node. `0`
decodes on each take, keeping GPU memory bounded by the playback queues. The
web UI's WIPES meter shows how full the cache is and lists its clips on hover.

```json
"wipes": [{"id": "ribbons", "path": "/media/media_wipes/ribbons.mov",
           "name": "Ribbons", "duration_seconds": 2.03}]
```

| field | default | meaning |
| --- | --- | --- |
| `id` | — | referenced by `control.default_wipe` |
| `path` | — | clip with alpha, on the mixer host |
| `name` | the id | label shown on the web UI button and in the TUI picker |
| `duration_seconds` | probed | how long the take runs |

`"wipe_dir": "/media/media_wipes"` adds every clip in a directory under its file
name; entries declared above keep their own id, label and duration.

## control

Where the control surfaces (web UI and TUI) start. An operator's own choice
always outranks these; they are the state a fresh browser tab picks up.

| field | default | meaning |
| --- | --- | --- |
| `direct` | `true` | a scene pick goes straight to program rather than loading preview |
| `transition` | `"cut"` | what a direct-mode pick takes with: `cut`, `fade` or `wipe` |
| `fade_seconds` | `0.5` | length of a fade |
| `fade_curve` | `"linear"` | easing of a fade: `linear`, `ease-in` (t²), `ease-out` (1−(1−t)²) or `ease-in-out` (3t²−2t³); a take may pick its own (`mixer.fade {..., "curve": "ease-in"}`); a `mixer.fade` without `curve` is linear |
| `fade_color` | `null` | `null` mixes; `"#RRGGBB"` makes a fade a dip through that colour, fully shown at the midpoint, with `fade_curve` shaping each half; converted for the canvas color (HLG/PQ white is 203-nit graphics white); a take may pick its own (`mixer.fade {..., "color": "#000000"}`); a `mixer.fade` without `color` mixes |
| `default_wipe` | first wipe | the clip a direct-mode wipe uses |

The mixer publishes this over the control protocol as `mixer.settings`.
The web bridge's optional `--transition cut` overrides its page's starting
choice without reconfiguring or restarting the media pipeline. It does not
override an operator's subsequent transition selection.

## scenes

A scene is an ordered list of items; **order is z-order**, later items draw
over earlier ones.

```json
{"id": "pip", "items": [
  {"source": "cam0", "dst": {"x": 0, "y": 0, "w": 1080, "h": 1920}, "fit": "cover"},
  {"source": "page", "dst": {"x": 40, "y": 100, "w": 1000, "h": 562}, "fit": "contain"},
  {"source": "cam1", "dst": {"x": 600, "y": 420, "w": 420, "h": 236},
   "fit": "stretch", "crop": {"x": 480, "y": 270, "w": 960, "h": 540}}
]}
```

| field | default | meaning |
| --- | --- | --- |
| `source` | — | a source id |
| `dst` | — | `x`, `y`, `w`, `h` on the canvas |
| `fit` | `"contain"` | `stretch`, `contain` (letterbox/pillarbox), `cover` (fill and crop the overflow) |
| `blend` | `false` | honour source alpha, for example a transparent browser graphic over video; opaque sources remain opaque |
| `crop` | whole frame | region of the source in source pixels, applied before the fit |

Padding is always black. `cover` needs the source size, which is declared or
probed. `initial_scene` names the scene on program at start (default: the
first one).

## dsk

Downstream keys: up to four alpha browser graphics (bugs, lower thirds)
drawn over the finished program, above transitions and wipes. Renditions with
`"feed": "clean"` receive the program without them; the multiview PGM tile
shows the keyed program.

```json
"dsk": {"fade_seconds": 0.5, "fade_curve": "ease-in-out", "keys": [
  {"id": "bug", "source": "logo_page", "dst": {"x": 32, "y": 32, "w": 152, "h": 152}},
  {"id": "strap", "source": "strap_page", "dst": {"x": 32, "y": 1640, "w": 1016, "h": 172}, "on": true}
]}
```

| field | default | meaning |
| --- | --- | --- |
| `fade_seconds` | `0.4` | how long a key change fades, 0 to 10; `0` cuts |
| `fade_curve` | `"linear"` | easing of a key fade, the same presets as `control.fade_curve` |

Each key:

| field | default | meaning |
| --- | --- | --- |
| `id` | — | unique key name, used by `mixer.dsk` and the control page |
| `source` | — | a **browser** source id; its alpha is always kept. Scenes may use the same source |
| `dst` | whole canvas | `x`, `y`, `w`, `h` on the canvas; the window is scaled into it |
| `on` | `false` | on air at start |

Keys are ordinary sources and count toward the source limit. Size the browser
window to the graphic (and allow that size in `DMA_BROWSER_ALLOWED_DIMS`),
not to the canvas: Chromium then paints and exports only the graphic's pixels.

The keyer is one compositor pass clocked by the program: every program frame
renders at once over each key's frame stamped for that tick (matched by
timestamp, as scene sources are), so it adds no playout latency, steady motion
stays steady, and a late browser paint never delays the program. Off keys keep
receiving frames (without drawing them), so a key switched on appears on the
very next program frame. The cost is browser ring slots, not copies: a key,
on or off, holds its source's frames stamped ahead of the program (usually the
program latency plus about two frames, at most seven), so a key source also
shown in scenes or on a multiview may need a larger `browser_ring_size`. With
all keys off the program frame passes through without GPU work. With no keys declared the graph has no keyer.

`mixer.dsk {"key": "bug", "on": true, "fade_seconds": 0.5, "curve": "ease-out"}`
puts a key on or off air; `fade_seconds` and `curve` default to the `dsk` ones
above. A fade starts on the first program frame after the command, even if the
key's browser has not painted yet: a late first paint joins the ramp part-way
rather than delaying it. Keys fade independently; a key switched back
mid-fade turns around from the level it reached, at the same pace. A command
that leaves a key as it is changes nothing, even a cut sent while that key is
still fading. A cut (`0`) also asks the preview encoder for a keyframe, as an
M/E cut does; a fade does not. Like the M/E's, that keyframe goes to the first
Janus rendition only, even when that rendition is a clean feed.

## aux_buses

Extra monitor outputs, any number of them, each its own compositor and H.264
encoder (SDR, canvas size, program rate halved above 30 fps unless `full_rate`)
sent to Janus. They subscribe to the sources the main mixer already decodes: a
source reaches a bus only while its layout draws it, and a bus never delays the
program.

```json
"aux_buses": [
  {"id": "mv", "scenes": ["grid_4_000", null, null, null, null, null, null, null],
   "renditions": [{"id": "monitor", "port": 5008}]},
  {"id": "mv2", "layout": {"preset": "source_pages"},
   "renditions": [{"id": "monitor", "port": 5012}]}
]
```

A bus draws a layout: a list of cells in canvas pixels, each with a role.

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

Every `pvw` cell draws the preview scene below the rest, each reserving the
largest scene's item count of layers; the `slot` and `source` cells follow in
list order and the `pgm` cells last, over any cell they overlap.

| field | default | meaning |
| --- | --- | --- |
| `layout` | `{"preset": "pgm_pvw_grid"}` | the layout the bus starts with |
| `layouts` | `source_pages`, and `pgm_pvw_grid` when `layout` has a `pgm` cell | further layouts the control page offers besides `layout` (at most 16) |
| `scenes` | none | slot assignments by slot index, `null` for an empty slot; padded with `null` to the layout's slot count |
| `max_layers` | `max_compositor_layers` | the bus compositor's layer budget, fixed at build: a layout's PVW reserve, slots, sources and PGM. A layout or assignment over it is refused. Each frame's work follows the layers it draws; the budget sizes only the layer table, 128 bytes per layer on the GPU and in pinned host memory |
| `latency_ms` | the mixer's `latency_ms` | the bus compositor's playout buffer. The sources reach a bus when they reach the program compositors, so the program's buffer leaves it the same slack (1.5 aux frames at 60 fps: 50 ms); a bus at the program's buffer can change its PVW cells when the program changes. Bus latency plus `pgm_delay_frames` must stay below six aux frames |
| `pgm_delay_frames` | `1`, `2` at a 50/60 fps bus | aux frames the PGM pad is matched back. The finished program leaves the main compositor `latency_ms` after its timestamp, when the bus would already be drawing that frame's tick, and needs a margin to cross the output chain (snapshot, selectors, keyer, tap) to the bus: `pgm_delay_frames × aux frame + latency_ms` must exceed the mixer's `latency_ms` by at least one program frame (checked at build for a bus whose layouts draw the program). The default gives about 33 ms (40 at 25/50) at any rate, which is why a `full_rate` bus at 50/60 takes two of its frames; `1` there leaves one program frame (16.7 ms at 60), and `0` needs a bus `latency_ms` at least a program frame above the mixer's, which delays every cell instead of the PGM cells alone |
| `pvw_align` | `"program"` | when the PVW cells change on a take. `program`: on the bus frame leaving the bus when the program frame of the take leaves the mixer, so the operator sees both at once; the PGM cells of the same frame follow `pgm_delay_frames` later. `pgm_tile`: together with those PGM cells, `pgm_delay_frames` after the program |
| `label` | `Program preview`, `Multiviewer` or `Aux <id>` by its initial layout | the bus's name on the control page |
| `full_rate` | `false` | run the bus at the canvas rate at 50/60 fps instead of half. Costs about twice the bus's compositor and encoder work on the GPU (a second bus's worth at 1080p60), and the rendition's `bitrate_kbps` then covers twice the frames, so raise it to keep the quality per frame; no change at 25/30 |

A bus gets a PGM pad (the program tap's subscribed output) only when its
`layout` or one of its `layouts` has a `pgm` cell, as the default
`pgm_pvw_grid` has. A `source_pages` bus has none by default: no program
latency check, 128 sources. A bus without the pad can switch to any layout
without a `pgm` cell and refuses one with it; list `{"preset": "pgm_pvw_grid"}`
in its `layouts` to give it the pad.
Each bus needs its own Janus RTP/RTCP port pair and a Janus mountpoint whose ID
equals its RTP port; the bundled Janus config has one for 5008 (RTCP 5009), and the setup
page's extra aux outputs create their own ([README](../README.md)).

Control commands, each `{"bus": <id>, ...}`:

- `mixer.aux_layout {"bus", "layout"}` switches the bus to any valid layout
  within its `max_layers`; slot assignments keep their slot numbers (a layout
  with fewer slots hides the rest until one with more shows them again).
- `mixer.aux {"bus", "expected_revision", "scenes"}` assigns slots: the
  complete `scenes` list as `mixer.aux_status` reports it (`null` clears a slot)
  and that status's `revision`; a stale revision returns the current
  assignments without applying the change.
- `mixer.aux_page {"bus", "page"}` or `{"bus", "step": ±1}` shows a page of a
  `source_pages` layout; a step past the last page wraps around.
- `mixer.aux_status` reports every bus: `id`, `layout`, `layouts`, `cells` (in
  canvas pixels, source cells with the source's `id` and `kind`), `scenes`,
  `revision`, `max_layers`, `composition_pending`, `composition_error`,
  `pvw_scene` and the follower's status under `follower`, and for a
  `source_pages` layout `page`, `pages`, `first` and `total`. A change is
  `composition_pending` until the bus compositor draws its revision: its new
  sources are staged until they have frames, and dropped with
  `composition_error` when they do not arrive in time.

The control page offers each bus as an output, with a menu of its layouts when
it has several, and M1… slot buttons for the bus a viewer shows: arm a slot,
then click a scene; × clears it, and Shift+1–9 assigns the selected scene.
Recipes accept the same block. Setup changes keep each bus's live layout,
`layouts` (with them its PGM pad) and assignments: slots whose scene no longer
exists or no longer fits the bus's budget are cleared, and source pages follow
the new source list from page 0.
With a bus configured, scene definitions are fixed until the next setup,
`mixer.scene` included. A bus suspends after sustained encoder backpressure;
the next assignment, layout switch or page turn resumes it.

The PGM cells run `pgm_delay_frames` (one aux frame, two at a 50/60 fps bus)
behind the other cells: the finished program reaches the bus after the sources
it is made of, so that pad alone is shown later instead of delaying every pad.
At half rate the bus receives every other program frame (the first of each aux
tick), so a take whose first program frame falls between aux ticks shows in the
PGM cells from the next frame.

Every bus's composition is set by its `mixer_pvw_follow` node
(`aux_<id>_pvw`), which draws the PVW cells on the bus frame `pvw_align` picks.
With the defaults, a take whose first new program frame has timestamp K leaves
the mixer at K + `latency_ms` (50 ms at 60 fps, 66.7 at 30). The bus frame with
the new PVW cells leaves its bus at the same instant, or 16.7 ms later at 60 →
30 for the takes whose K falls between aux ticks (always the same instant at
25/30 fps or with `full_rate`), and one aux frame later when the change misses
its tick. The PGM cells show the take at K + 83.3 ms (K even) or K + 100 ms (K
odd) at 60 → 30, and at K + 100 ms at 30 fps. A fade's PVW change lands one tick
after the program by construction; wipes and explicit `mixer.preview` changes
draw on the next frame. A bus `latency_ms` above the mixer's moves the whole
bus later by the difference. `mixer.status` `pvw_latency` reports every
follower's last timed change; the
[mixer notes](../../../doc/mixer.md#aux-bus-follower) give the error bound,
the lock order and what each latency field measures.

## Known limitations

- **128 sources per show, or 127 with an aux bus that can draw the program.** Every
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

More than 128 pads and runtime source addition are not supported.

## Generating one

`make_config.py` writes the demo's own fullscreen and 2/4/8/16-box layouts out
as a document, so a config-driven run starts from what the demo already does:

```sh
python3 demos/mixer/make_config.py --fps 30 --bitrate-kbps 2700 \
    --wipe /media/media_wipes/ribbons.mov \
    cam0=/media/camera-0.mp4 page=https://example.org/live@1920x1080 > mixer.json
```

Repeated locations collapse to one source, and grid slots take every distinct
source before any repeat — a 16-box of sixteen unique sources shows all
sixteen rather than one of them three times.

`--color hlg|pq` with `--working-format p010le|p210le` emits an HDR canvas, a
HEVC Main10 program rendition and, with `--sdr-port`, a tone-mapped H.264
rendition (`--sdr-tonemap`, `--sdr-knee`). Source arguments take a color
suffix (`clip=/m/c.mp4:hlg`) and raw v210 a size (`bars=/m/b.v210@1920x1080:hlg`);
see the README for a 16-source HDR example.
