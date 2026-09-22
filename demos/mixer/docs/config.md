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
| `color` | `sdr` | canvas color contract: `sdr` (BT.709), `hlg` or `pq` (BT.2020). HLG/PQ need a 10-bit `working_format`. Every source is converted to it on the GPU; renditions convert from it |
| `latency_ms` | two frames | playout buffer between a source frame's arrival and its tick (33 ms at 60 fps). A frame later than that is skipped and the previous one repeated, so set it just above the worst source jitter; must stay below six frames. `--mixer-latency-ms` on the command line overrides it |

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
| `tonemap` | none | requests an SDR output from an HDR canvas: `clip` (exact SDR, hard-clipped highlights), `mobius` (see `tonemap_param`), `hable`, `reinhard`, `gamma`, `linear`, `none` |
| `tonemap_peak` | `10` | HDR peak in units of 100 nits, minimum `2.03` |
| `tonemap_desat` | `0` | highlight desaturation; `0` keeps saturation |
| `max_cll`, `max_fall` | derived | HDR10 static metadata for **PQ** outputs, nits. Defaults: MaxCLL = `tonemap_peak`×100, MaxFALL = 40% of it; `max_fall` may not exceed MaxCLL. Needs nv-codec-headers 13 and driver ≥ 570 (`Dockerfile.cuda`), else the SEIs are silently absent. HLG needs none |
| `tonemap_param` | `0` | operator knee in reference-white units; `0` = operator default (0.3 mobius/reinhard, 1.8 gamma). mobius must be below 1.0; `0.9` keeps 90% of SDR white untouched |

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
| `kind` | all | `"video"`, `"browser"` or `"v210"` (headerless packed 10-bit 4:2:2, e.g. generated HDR test content) |
| `path` | video, v210 | file or stream |
| `width`, `height` | v210 | required: the packed bytes carry no header |
| `color` | browser, v210 | required color contract (`sdr`, `hlg`, `pq`); browser pages must be `sdr`. Optional for `video`: by default the decoded frame tags decide and untagged files are treated as BT.709 SDR |
| `url`, `width`, `height` | browser | page and the window it is rendered in (all three required) |
| `fps` | browser | paint rate; defaults to the canvas rate |
| `width`, `height` | video | optional; probed with ffprobe at load when absent |
| `loop` | both | default `true` |
| `filter` | video/v210 | optional CUDA source graph, before automatic normalization; preserve dimensions and correct output metadata |
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
to vary the independent input count while reusing cached media files.

## wipes

The media-wipe library. Clips decode on each take by default, keeping GPU
memory bounded by the playback queues. Startup warms the wipe path but does
not retain whole decoded clips. Opt in with `--wipe-cache-mb 256` to retain
clips in a GPU cache with a 256 MiB budget; `0` disables caching.

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

## Known limitations

- **64 sources per show.** Every source is a pad on the compositor, and the
  active-pad mask is a 64-bit word. 8-bit or 10-bit makes no difference;
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

Larger config-driven shows and runtime source addition require a different
pad allocation strategy; they are not supported.

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
