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
| `codec` | `"h264_nvenc"` | file targets only; the Janus target is always H.264 |
| `profile` | `"baseline"` | WebRTC negotiates constrained baseline; B-frames stay off |
| `preset` | `"p7"` | NVENC quality preset |
| `port` | — | Janus target: overrides the RTP port from the command line |

With no `renditions` the demo builds its usual single output from the command
line flags.

```json
"renditions": [
  {"id": "program", "target": "janus", "width": 1080, "height": 1920,
   "aspect": "9:16", "fps": 30, "bitrate_kbps": 2700,
   "profile": "baseline", "preset": "p7"}
]
```

## sources

One entry per **unique** clip or page: two entries with the same `url` or
`path` are an error, because a source is decoded or captured exactly once
however many scenes and slots show it. A scene that shows the same source
twice fans the frames out under alias names (`id#2`, `id#3`), never a second
decoder.

```json
{"id": "cam0", "kind": "video",   "path": "/media/camera-0.mp4", "loop": true}
{"id": "page", "kind": "browser", "url": "https://example.org/live",
 "width": 1920, "height": 1080, "fps": 30}
```

| field | applies to | meaning |
| --- | --- | --- |
| `id` | both | referenced from scenes; no `#` |
| `kind` | both | `"video"` or `"browser"` |
| `path` | video | file or stream |
| `url`, `width`, `height` | browser | page and the window it is rendered in (all three required) |
| `fps` | browser | paint rate; defaults to the canvas rate |
| `width`, `height` | video | optional; probed with ffprobe/ffmpeg when a `cover` item needs them |
| `loop` | both | default `true` |

Browser sources arrive over DMA-BUF from the `dma-page` service and are
imported straight into CUDA; they mix with video sources on the same canvas.

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
