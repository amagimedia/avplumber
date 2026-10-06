# Demo recipes

A recipe generates test media and a [mixer show](https://github.com/amagimedia/avplumber/blob/ad371df0145fa9d9f1ed1e10397e1952eb359bfa/demos/mixer/docs/config.md). Setup handles this
automatically; for manual preparation on the NVIDIA host:

```sh
python3 -m demos.mixer.prepare_demo media/demo.json --media-dir media
python3 -m pyplumber.mixer.cli --config media/mixer.demo.json --janus-output
```

Edit `media/demo.json`; `media/mixer.demo.json` is generated. Preparation needs
NumPy and FFmpeg with NVENC. `--runtime-media-dir /media` writes container paths.

## Counts and proportions

| Field | Meaning |
| --- | --- |
| `source_count` | Independent source chains; hardware/profile limits also apply |
| `scene_count` | Number of named scenes, independent of source count |
| `inputs[].weight` | Relative share of sources; zero disables the entry |
| `layouts` | Layout names and relative shares of scenes |
| `seed` | Reproduces scene selection and geometry |
| `canvas` | Dimensions, rate, format and color; see [config](https://github.com/amagimedia/avplumber/blob/ad371df0145fa9d9f1ed1e10397e1952eb359bfa/demos/mixer/docs/config.md#canvas) |
| `generation` | `seconds` (default 2), optional source `width` and `height` |
| `alpha_background` | Allocated video source used behind transparent pages |
| `renditions`, `clean_rendition`, `aux_buses`, `dsk` | Output/key settings copied into the show |
| `browser_ring_size` | Outstanding frames per browser: 1–64; default 6 at 25/30 fps, otherwise 9 |

Largest-remainder rounding makes the weighted allocations sum to the total;
ties follow recipe order. Weights summing to `source_count` give exact counts.
The graph accepts 193 sources, or 192 with an AUX program pad. Setup also
applies the selected [instance profile](https://github.com/amagimedia/avplumber/blob/ad371df0145fa9d9f1ed1e10397e1952eb359bfa/demos/mixer/instance_profiles.py).

## Example presets

- [demo.example.json](https://github.com/amagimedia/avplumber/blob/ad371df0145fa9d9f1ed1e10397e1952eb359bfa/demos/mixer/demo.example.json): generated video, no browsers.
- [demo.equal.json](https://github.com/amagimedia/avplumber/blob/ad371df0145fa9d9f1ed1e10397e1952eb359bfa/demos/mixer/demo.equal.json): video and browser sources with equal weights.
- [demo.browser-alpha.json](https://github.com/amagimedia/avplumber/blob/ad371df0145fa9d9f1ed1e10397e1952eb359bfa/demos/mixer/demo.browser-alpha.json): transparent overlay checks.

Copy a preset to `media/demo.json`, edit it, then restart the Compose mixer.
Disable `alpha_overlay` layouts if there are no browser inputs; remove
`alpha_background` if its source no longer exists.

## Input entries

Every entry has an `id`, `kind` and `weight`.

| Kind | Fields |
| --- | --- |
| `generated` | `color`: `sdr`/`hlg`; `chroma`: `420`/`422`; optional `pattern` or `patterns` |
| `browser` | `url` or `pattern: "alpha"`; optional `width`, `height` |
| `file` | `path` relative to media directory; optional `color` |
| `download` | Direct media-file `url`; optional `color` |

Generated 4:2:0 media defaults to H.264 for SDR and HEVC for HLG. Set `codec`
to select encoded storage, or `storage: "nv12"` / `"p010"` for raw uploads.
Generated 4:2:2 uses v210. Encoded entries may select `decode_storage: "cuarray"`
and `extra_hw_frames`; see [decoder requirements](https://github.com/amagimedia/avplumber/blob/ad371df0145fa9d9f1ed1e10397e1952eb359bfa/demos/mixer/docs/config.md#sources).

## Generated media

Each generated source has a separate file with its ID burned into moving frames.
Assets and two alpha wipes are cached under `media/`. Scene-only changes reuse
assets; changing dimensions, rate or storage generates new ones. Unused assets
are not pruned. Raw loops consume disk/cache space and upload bandwidth every frame.

## Scene compositions

Layouts: `fullscreen`, `grid_2`, `grid_4`, `grid_8`, `grid_16`, `grid_32`,
`grid_64`, `pip`, `random`, `alpha_overlay`. Grid orientation follows the canvas;
later items draw above earlier ones. Alpha layouts require video and browser inputs.

## Movie inputs

A `file` or `download` entry may allocate only one source; duplicate clips are
rejected. For capacity comparisons use separate clips of at least 1080p25 and
record attribution/license with each URL. See the [HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)
for workload rules and implementation notes.

## Capacity fixtures and mixed-rate correctness

The balanced capacity recipe generates a show at its chosen canvas rate. It
does not by itself test a mixture of native source rates. Use
[the mixed-rate smoke](https://github.com/amagimedia/avplumber/blob/ad371df0145fa9d9f1ed1e10397e1952eb359bfa/tests/cuda/smoke_mixed_fps.py) on an NVIDIA host
for that check, and keep its small barcoded fixture separate from full-resolution
capacity results. Generated source counts include reserved browser keys when
comparing them with profile totals; the 192-source CUarray SDR25 profile has
188 catalogue sources plus four keys.
