# Mixer configuration file: design

One JSON document drives a mixer run: which sources exist, which wipe clips
are available, and what every scene looks like. Nothing about 2/4/8/16 grids
lives in code; grids are just scenes somebody wrote (or generated) in the file.
The loader maps the document onto the existing engine: one input chain per
unique source, `mixer.scene` per scene, `mixer.wipe` with the named clip.

## Principles

- **Decode once per unique source.** Sources are declared once and referenced
  by id from scenes. Two declarations with the same `url`/`path` (and the same
  size and fps for browsers) are an error; the loader never builds two chains
  for one URL. A scene item that references a source adds no decoder and no
  copy: it is a reference to the frames the compositor already receives.
- **Scenes are data.** A scene is an ordered list of items. Order is z-order:
  later items draw over earlier ones. The same scene can be used on either
  slot; both slots read the same frames.
- **The composer rate and the output rates are separate.** `canvas.fps` is how
  often the compositor renders; each rendition re-times and rescales that one
  program for its own target, so a second rendition costs an encode rather than
  another composite. A rendition may not ask for more frames than the composer
  produces.
- **Placement is explicit.** Every item says where it goes on the canvas and
  what to do with the aspect: `stretch`, `contain` (letterbox/pillarbox, always
  black), `cover` (fill the box, crop the overflow), plus an optional crop in
  source pixels applied first. No pad colours: padding is black, period.

## Schema (JSON Schema 2020-12, abridged)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["canvas", "sources", "scenes"],
  "properties": {
    "canvas": {"type": "object", "required": ["width", "height", "fps"],
      "properties": {"width": {"type": "integer"}, "height": {"type": "integer"},
                     "fps": {"type": "integer"}}},
    "sources": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/source"}},
    "wipes": {"type": "array", "items": {"$ref": "#/$defs/wipe"}},
    "wipe_dir": {"type": "string", "description": "directory scanned for further wipe clips, named after their files"},
    "control": {"type": "object", "description": "defaults for the control surface",
      "properties": {"direct": {"type": "boolean", "default": true},
                     "fade_seconds": {"type": "number"}, "default_wipe": {"type": "string"}}},
    "renditions": {"type": "array", "description": "encoded outputs; the compositor renders once",
      "items": {"$ref": "#/$defs/rendition"}},
    "scenes": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/scene"}},
    "initial_scene": {"type": "string"}
  },
  "$defs": {
    "rect": {"type": "object", "required": ["x", "y", "w", "h"],
      "properties": {"x": {"type": "integer"}, "y": {"type": "integer"},
                     "w": {"type": "integer"}, "h": {"type": "integer"}}},
    "source": {"type": "object", "required": ["id", "kind"],
      "properties": {
        "id": {"type": "string", "pattern": "^[a-z0-9_-]+$"},
        "kind": {"enum": ["browser", "video"]},
        "url": {"type": "string", "description": "browser: page URL"},
        "path": {"type": "string", "description": "video: file path or stream URL"},
        "width": {"type": "integer"}, "height": {"type": "integer",
                  "description": "browser: window size (required); video: optional, probed with ffprobe"},
        "fps": {"type": "integer", "description": "browser paint rate, default canvas fps"},
        "loop": {"type": "boolean", "default": true},
        "audio": {"type": "boolean", "default": false}
      },
      "allOf": [
        {"if": {"properties": {"kind": {"const": "browser"}}},
         "then": {"required": ["url", "width", "height"]}},
        {"if": {"properties": {"kind": {"const": "video"}}}, "then": {"required": ["path"]}}
      ]},
    "wipe": {"type": "object", "required": ["id", "path"],
      "properties": {"id": {"type": "string"}, "path": {"type": "string"}, "name": {"type": "string"},
                     "duration_seconds": {"type": "number", "description": "default: probed"}}},
    "item": {"type": "object", "required": ["source", "dst"],
      "properties": {
        "source": {"type": "string", "description": "id of a declared source"},
        "dst": {"$ref": "#/$defs/rect", "description": "box on the canvas, pixels"},
        "fit": {"enum": ["stretch", "contain", "cover"], "default": "contain"},
        "crop": {"$ref": "#/$defs/rect", "description": "source pixels, applied before fit"}
      }},
    "rendition": {"type": "object", "required": ["id"],
      "properties": {"id": {"type": "string"},
                     "target": {"type": "string", "description": "\"janus\" or a file path"},
                     "width": {"type": "integer"}, "height": {"type": "integer"},
                     "aspect": {"type": "string", "description": "checked against width:height"},
                     "fps": {"type": "integer", "description": "at most the canvas rate"},
                     "bitrate_kbps": {"type": "integer"}, "codec": {"type": "string"},
                     "profile": {"type": "string"}, "preset": {"type": "string"},
                     "port": {"type": "integer"}}},
    "scene": {"type": "object", "required": ["id", "items"],
      "properties": {"id": {"type": "string"}, "items": {"type": "array", "items": {"$ref": "#/$defs/item"}}}}
  }
}
```

## Example

```json
{
  "canvas": {"width": 1080, "height": 1920, "fps": 30},
  "renditions": [{"id": "program", "target": "janus", "width": 1080, "height": 1920,
                  "aspect": "9:16", "fps": 30, "bitrate_kbps": 3000,
                  "profile": "baseline", "preset": "p7"}],
  "sources": [
    {"id": "cam1", "kind": "video", "path": "/media/cam-1.mp4"},
    {"id": "cam2", "kind": "video", "path": "/media/cam-2.mp4"},
    {"id": "score", "kind": "browser", "url": "https://app.singular.live/output/…/Output?aspect=16:9",
     "width": 1920, "height": 1080},
    {"id": "ticker", "kind": "browser", "url": "https://example.org/ticker", "width": 1920, "height": 120}
  ],
  "wipes": [
    {"id": "swoosh", "path": "/media/swoosh-alpha.mov"},
    {"id": "stinger", "path": "/media/stinger.mov", "duration_seconds": 1.2}
  ],
  "control": {"direct": true, "fade_seconds": 0.8, "default_wipe": "swoosh"},
  "scenes": [
    {"id": "cam1_full", "items": [{"source": "cam1", "dst": {"x": 0, "y": 0, "w": 1920, "h": 1080}, "fit": "cover"}]},
    {"id": "two_up", "items": [
      {"source": "cam1", "dst": {"x": 0, "y": 270, "w": 960, "h": 540}},
      {"source": "cam2", "dst": {"x": 960, "y": 270, "w": 960, "h": 540}}]},
    {"id": "cam1_with_graphics", "items": [
      {"source": "cam1", "dst": {"x": 0, "y": 0, "w": 1920, "h": 1080}, "fit": "cover"},
      {"source": "cam2", "dst": {"x": 1440, "y": 60, "w": 420, "h": 236},
       "crop": {"x": 240, "y": 0, "w": 1440, "h": 810}},
      {"source": "ticker", "dst": {"x": 0, "y": 960, "w": 1920, "h": 120}, "fit": "stretch"}]}
  ],
  "initial_scene": "cam1_full"
}
```

`cam1` appears in three scenes and twice in the layouts loaded on the two
slots at any time; it is decoded once. Grids are generated the same way a
person would write them: a small script emitting `scenes` entries, kept out
of the engine.

## Mapping onto the engine

| Document | Engine today | Gap |
| --- | --- | --- |
| `sources[]` video | `avpmixer.inputs.build_input` (NVDEC chain) | none |
| `sources[]` browser | `avpmixer.dmabuf_inputs` (window open, DMA-BUF import) | none |
| duplicate `url`/`path` | — | loader rejects; alias support (one chain, two ids) is a second `one_to_many` output, ~10 lines |
| `item.dst`, `crop`, `fit: stretch\|contain` | `cuda_rect_overlay` layer: `dst_*`, `crop`, `fit` | none |
| `fit: cover` | loader computes the crop from the aspect; clip sizes probed with ffprobe | none |
| item order = z-order | draw order is source registration order | order ops by item index instead; small compositor change |
| same source twice in one scene | one layer per source | alias source (above) |
| `wipes[]` + `wipe_dir` | `mixer.wipe` takes any path; `mixer.wipe.warmup` | the library is warmed at start and published in `mixer.settings`; the web UI shows one button per clip, the TUI a picker |
| `control.*` | TUI arguments | the mixer publishes them as `mixer.settings`; the TUI applies them on connect |
| `initial_scene` | `set_initial_scene` | none |

Scaling is done in the compositor's draw pass (bilinear, with RGB→NV12 for
browsers in the same kernel), so crop, fit and placement never add a pass.
