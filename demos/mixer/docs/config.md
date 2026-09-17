# Mixer configuration file

One JSON document describes a mixer run: the sources it opens, the wipe clips
it caches, the scenes an operator can take, the defaults its control surfaces
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
  "wipe_dir":     "/media/wipes",
  "control":      { "direct": true, "transition": "cut", "fade_seconds": 0.5 },
  "scenes":       [ ... ],
  "initial_scene": "fullscreen_0"
}
```

`canvas`, `sources` and `scenes` are required; everything else has a default.

## canvas

| field | default | meaning |
| --- | --- | --- |
| `width`, `height` | — | the program raster the compositor draws into |
| `fps` | `30` | **how often the compositor renders**, and the clock the whole mixer runs on: inputs are re-timed to it and browser pages are asked to paint at it |
| `working_format` | `nv12` | compositor and transition pixel storage: `nv12` (8-bit 4:2:0), `p010le` (10-bit 4:2:0) or `p210le` (10-bit 4:2:2). 8-bit sources are promoted onto a 10-bit canvas; `p210le` keeps 4:2:2 content (`v210` sources) native, renditions subsample once for NVENC |
| `color` | `sdr` | canvas colour contract: `sdr` (BT.709), `hlg` or `pq` (BT.2020). HLG/PQ need a 10-bit `working_format`. Every source is converted to it on the GPU; renditions convert from it |

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
| `profile` | from codec/depth | H.264 baseline, HEVC main or main10; B-frames stay off |
| `preset` | `"p7"` | NVENC quality preset |
| `port` | — | Janus target: overrides the RTP port from the command line |
| `color` | automatic | `sdr`, `hlg`, or `pq`; H.264 always requires SDR, HEVC otherwise inherits the canvas |
| `tonemap` | `clip` when conversion is needed | Explicit operator requests SDR; 203-nit reference white, selectable highlight compression |
| `tonemap_peak` | `10` | HDR display peak in units of 100 nits |
| `tonemap_desat` | `0` | highlight desaturation; zero preserves saturation |
| `color` | canvas | output contract: `sdr`, `hlg` or `pq`. H.264 and any `tonemap` imply `sdr`; HEVC defaults to the canvas contract |
| `max_cll`, `max_fall` | derived | HDR10 static metadata for **PQ** outputs, in nits: MaxCLL defaults to `tonemap_peak` × 100, MaxFALL to 40% of it; the mastering display is BT.2020/D65 at that peak. HLG carries none |
| `tonemap_param` | no | operator knee in reference-white units (mobius/reinhard). mobius `0.9` keeps everything up to 90% of SDR white untouched and folds brighter HDR highlights into the top 10% of the SDR range; must be below 1.0 (1.0 is a plain clip). `0` keeps the filter default (0.3) |

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
| `color` | browser, v210 | required colour contract (`sdr`, `hlg`, `pq`); browser pages must be `sdr`. Optional for `video`: by default the decoded frame tags decide and untagged files are treated as BT.709 SDR |
| `url`, `width`, `height` | browser | page and the window it is rendered in (all three required) |
| `fps` | browser | paint rate; defaults to the canvas rate |
| `width`, `height` | video | optional; probed with ffprobe/ffmpeg when a `cover` item needs them |
| `loop` | both | default `true` |
| `color` | all sources | explicit `sdr`, `hlg`, or `pq` override; required for untagged/raw inputs |
| `filter` | video/v210 | optional CUDA source graph, before automatic normalization; preserve dimensions and correct output metadata |
| `filter_output_format` | custom filters | required CUDA YUV output storage, e.g. `p010le` or `p210le` |

Browser sources arrive over DMA-BUF from the `dma-page` service and are
imported straight into CUDA; they mix with video sources on the same canvas.

Color normalization is automatic in `MixerGraphBuilder`, before alias and scene
fan-out. Video sources use decoded frame metadata, including live SRT streams.
Missing transfer, primaries, matrix or range is an error; bit depth never selects
a color space. An explicit `color` preset supplies all four fields. Individual
`color_trc`, `color_primaries`, `colorspace` and `color_range` fields must be complete
and consistent. Raw `v210` and browser inputs require an explicit setting.

For SDR Bunny on an HLG canvas, no manual tone-map filter is needed:

```json
"canvas": {"width": 1920, "height": 1080, "fps": 60,
           "working_format": "p010le", "color": "hlg"},
"sources": [{"id": "bunny", "kind": "video", "path": "<path>", "color": "sdr"}]
```

Supported video contracts are limited-range BT.709/BT.1886 SDR and BT.2020
non-constant-luminance HLG/PQ. Full-range YUV, other gamuts/matrices and missing
metadata fail explicitly. SDR uses a 203-nit reference white and HLG a 1000-nit
peak. Matching NV12/P010 frames pass without GPU copies. Other declared CUDA YUV
storage uses `scale_cuda` around conversion. Custom source filters receive the
explicit source override first; their output metadata drives normalization, so
a manually converted source is not converted from its original transfer again.

Packed SDR RGB graphics retain alpha and convert in the compositor to its target
transfer/gamut. This path requires NV12, P010 or P210 canvas storage. HDR alpha
sources are unsupported. Set top-level `wipe_color: "sdr"` to explicitly declare
an untagged SDR wipe library; otherwise wipes must carry usable metadata.

H.264 renditions and the default output path automatically convert HDR to SDR.
HEVC renditions inherit the canvas unless `color` requests another supported
contract. HDR outputs require 10-bit storage. The `clip` default preserves the
brightness of SDR embedded in HDR; it clips highlights above SDR white. Select
another operator explicitly when highlight compression is preferred.

## wipes

The media-wipe library. Every declared clip is decoded once at start and held
as frames in the clip cache, so a take costs no file open and no decoder.

```json
"wipes": [{"id": "ribbons", "path": "/media/wipes/ribbons.mov",
           "name": "Ribbons", "duration_seconds": 2.03}]
```

| field | default | meaning |
| --- | --- | --- |
| `id` | — | referenced by `control.default_wipe` |
| `path` | — | clip with alpha, on the mixer host |
| `name` | the id | label shown on the web UI button and in the TUI picker |
| `duration_seconds` | probed | how long the take runs |

`"wipe_dir": "/media/wipes"` adds every clip in a directory under its file
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
| `crop` | whole frame | region of the source in source pixels, applied before the fit |

Padding is always black. `cover` needs the source size, which is declared or
probed. `initial_scene` names the scene on program at start (default: the
first one).

## Generating one

`make_config.py` writes the demo's own fullscreen and 2/4/8/16-box layouts out
as a document, so a config-driven run starts from what the demo already does:

```sh
python3 demos/mixer/make_config.py --fps 30 --bitrate-kbps 2700 \
    --wipe /media/wipes/ribbons.mov \
    cam0=/media/camera-0.mp4 page=https://example.org/live@1920x1080 > mixer.json
```

Repeated locations collapse to one source, and grid slots take every distinct
source before any repeat — a 16-box of sixteen unique sources shows all
sixteen rather than one of them three times.
