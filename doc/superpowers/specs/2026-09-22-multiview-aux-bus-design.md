# Multiview aux bus

Status: approved 2026-09-22, ready for implementation. Step 0 is done and
the text below matches `mixer-improv` at 6f584c2 (same day). Companions: proposal
https://claude.ai/artifact/5RLau7GkWSHPuWnv3q59QG and control UI mock
https://claude.ai/artifact/1K4ZuwSxcJAP6hAgN3Nm3o.

Design constraint: multiple aux buses from the start. One bus ships first,
but nothing may assume there is only one.

## Goal

Encode a second, differently composed picture at the same time as the program:
one or more aux buses, each with its own NVENC session. The first bus is a
multiview with PGM and PVW in the top row and eight operator-selected scene
previews below. The existing two-slot mixer, its
transitions, snapshot, cut-latency probe and renditions stay untouched.

A single M/E cannot do this today: every rendition re-times and rescales the
one `final_out` picture (`demos/mixer/mixer.py`, `_build_renditions`).

## Layout

The bus canvas follows the program canvas: same size and aspect unless the
config says otherwise. The `pgm_pvw_grid` preset is fractional, so every
place has the program aspect and the picture fills the monitor pane, which
is sized to that aspect like the program pane is today. For a bus canvas
`W x H` with `rows` and `cols`:

| Place | Rect | Content |
|---|---|---|
| PVW | 0, 0, W/2, H/2 | current PVW scene, flattened |
| PGM | W/2, 0, W/2, H/2 | real `final_out` picture, one compositor pad |
| grid r,c | c*W/cols, H/2 + r*H/(2*rows), W/cols, H/(2*rows) | selected scenes, flattened |

Today's 9:16 demo at 1080x1920 gets PVW/PGM at 540x960 and eight tiles at
270x480; a 16:9 show gets 960x540 and 480x270. Rects are rounded to even
pixels. Explicit `places` bypass the preset. Nothing assumes an aspect.

"Flattened" means the referenced scene's layer list remapped into the tile:

```
s  = min(tile.w / canvas.w, tile.h / canvas.h)
ox = tile.x + (tile.w - canvas.w * s) / 2
oy = tile.y + (tile.h - canvas.h * s) / 2
dst_x' = ox + dst_x * s,  dst_w' = dst_w * s   (same for y/h)
crop, fit, blend and relative z carry over
```

Sixteen previews are `rows: 2, cols: 8`; 32 are `rows: 4, cols: 8`. The
canvas decides how big they come out. Every layer is drawn in one kernel
launch per frame (step 0, done), so draw cost follows canvas pixels, not
layer count. The rect table caps a frame at 256 layers
(`AVP_RECT_MAX_LAYERS`; `cuda_rect_draw` throws above it): sixteen grid_16
tiles plus PVW and PGM exceed it, so the bus builder rejects such a bus
unless the cap is raised. A 4K aux canvas also means a 4K NVENC session.

Each source is scaled once, straight into its final tile rect. There is no
intermediate frame per tile, no extra latency and no extra GPU pass per scene.

## Graph changes

1. **Fan-out.** Every source pad's `one_to_many` (`otm_<pad>`, dsts `[a, b]`)
   gains a third dst `mv`, bit 2, enabled at build time for every pad.
   - Needs one native change. Only `rewriteCameraOutputsForSlot`
     (`scene.cpp:224`) does read-modify-write; `applyPostTransitionRouting`
     (`scene.cpp:199`), fade cleanup (`fade.cpp:162`) and the wipe midpoint
     (`wipe.cpp:147`) write absolute masks and would clear bit 2 on every
     take. All four go through `MixerState::sourceOutputMask`, which already
     ORs the prewarm bits in. Add `uint32_t aux_output_mask` to `MixerState`,
     OR it there, and accept it as `aux_outputs` in `mixer.init`
     (`src/avplumber.cpp:1079`). About ten lines.
   - `one_to_many` drops on a full unconsumed edge when a timeline is set, so
     a pad the multiview is not drawing cannot stall the program.
2. **Compositor.** One new `cuda_rect_overlay` `aux_<id>_comp` at the bus
   canvas size (defaults to the program canvas), same working format and
   color as the canvas, inputs = the bus's source pads plus one PGM pad. `layers` and `active_inputs` are set at runtime; both are already
   accepted by `setObject`.
3. **PGM pad.** The finished 1080p program frame is reused as-is: the
   rendition `Split` on `final_out` gains one more output feeding the aux
   compositor, which scales it into the PGM box like any other source. Fades,
   wipes and the HTML overlay show because it is the real output.
   - Timing: a `final_out` frame for tick `q` leaves the main compositor at
     about `q + D_main`; the aux playout accepts it only if
     `arrival + C <= q + D_aux`. Set the aux bus `latency_ms` to
     `D_main + 3 frames` (about 166 ms at 30 fps with the default two-frame
     main delay). Verify with the playout repeat/discard counters.
   - Escape hatch if the extra delay is unwanted: a metadata-only
     `setpts=PTS+<D_main in ticks>` on the aux branch keeps the default
     delay at the cost of the PGM tile trailing the other tiles by D_main.
   - The aux branch must drop on full so a stalled aux compositor can never
     backpressure the program encoders.
   - Not chosen: flattening the PGM scene from source pads. No timing
     issue, but no transitions or overlays in the tile.
4. **Output.** `_build_renditions` on the bus compositor's output, one
   chain per bus rendition (`scale/format -> NVENC -> janus or file`),
   nodes named `aux_<bus>_<rendition>_*`.
5. **Same source in several tiles.** Required native change: one compositor
   input carries a list of layers (see below). An aux bus then has one pad
   per distinct source plus the PGM pad, whatever the tiles repeat. Python
   aliases stay for the main mixer's existing scenes but are not used by aux
   buses.
6. **Start order.** Unchanged main sequence (sources, preheat, routes, slots,
   transition warm-up, `start_output`), then start the multiview group. It
   needs no preheat; first output appears once all active pads delivered.

## Control

- New demo command `aux {"bus": id, "scenes": [ids]}` (web UI
  `POST /api/command`, TUI key later). It fills the bus's `scene` places in
  order, computes the flattened layer array and active mask and issues
  `node.object.set` for `layers` then `active_inputs` on that bus's
  compositor.
- The PVW tile is refreshed by the control surface after `preview`, `cut`,
  `fade` and `wipe`, from the scene it just requested. It is the flattened
  scene, not the slot compositor's pixels; outside a transition these are the
  same picture.
- Changing what a tile shows is a cut; only the PGM tile shows fades and wipes.
- Web UI (mock: https://claude.ai/artifact/1K4ZuwSxcJAP6hAgN3Nm3o):
  - A "Multiview" strip under the PGM/PVW buses with one slot per `scene`
    place (`M1`..`M8`). Assign by dragging a scene tile onto a slot, or by
    clicking a slot (armed, amber) then a scene, or `Shift+1..8` for the
    selected scene. A scene sits in one slot at a time; `x` clears a slot.
    Scene tiles show an `M<n>` badge when assigned, next to the existing
    PGM/PVW colouring.
  - Every change sends one `aux` command with the full slot list, so the
    strip is the single source of truth and reconnects re-sync from
    `/api/state`, whose bridge adds an `aux` section per bus from a
    demo-registered command (like `mixer.settings`); `mixer.status` is
    native and cannot be extended from Python.
  - The monitor is the existing preview page in an iframe. Its `Stream`
    select (H.264/SDR, H.265/HDR) gains a `Multiview` entry; the footer
    metrics (cut latency, RTT, playback fps, buffer, GPU/NVDEC/VRAM) stay
    as they are. Add `preview_outputs: [{id, codec, port}]` to
    `mixer.settings` next to `preview_codecs` (the flat codec list both
    pages consume today) so the aux rendition appears there without
    special-casing. Choosing Multiview keeps the last program color
    choice; the aux rendition follows it (see Config).

## Config

A bus is a list of places; the standard multiview is a preset that expands
to places. Place kinds: `pgm` (the real output pad), `pvw` (flattened current
PVW scene), `scene` (flattened, filled from `scenes` in order, re-targetable
at runtime).

```json
"aux_buses": [
  {"id": "multiview",
   "layout": {"preset": "pgm_pvw_grid", "rows": 2, "cols": 4},
   "scenes": ["cam1_full", "two_up", "grid_4", "pip_cam2",
              "grid_8", "grid_16", "cam1_with_graphics", "cam3_full"],
   "renditions": [{"id": "monitor", "target": "janus", "color": "follow",
                   "ports": {"h264": 5008, "h265": 5009}}]},
  {"id": "wall", "width": 3840, "height": 2160,
   "places": [
     {"kind": "pgm",   "dst": {"x": 0,    "y": 0,    "w": 2560, "h": 1440}},
     {"kind": "scene", "dst": {"x": 2560, "y": 0,    "w": 1280, "h": 720}},
     {"kind": "scene", "dst": {"x": 2560, "y": 720,  "w": 1280, "h": 720}},
     {"kind": "scene", "dst": {"x": 0,    "y": 1440, "w": 1280, "h": 720}},
     {"kind": "scene", "dst": {"x": 1280, "y": 1440, "w": 1280, "h": 720}},
     {"kind": "scene", "dst": {"x": 2560, "y": 1440, "w": 1280, "h": 720}}],
   "scenes": ["grid_16", "cam2_full", "cam3_full", "two_up", "pip_cam2"],
   "renditions": [{"id": "rec", "target": "/rec/wall.mp4", "bitrate_kbps": 20000}]}
]
```

`width`/`height` default to the program canvas. A list from day one, even
if the first implementation ships the multiview bus only. Bus `i` owns OTM bit `2 + i`, nodes are named `aux_<id>_*`, and
the `aux` command takes the bus id. Each bus has its own compositor size,
source subset, `latency_ms`, renditions and, if it shows PGM, its own output
on the `final_out` split.

Color: the aux compositor uses the canvas working format and color contract
(HLG P210 on an HLG show), fed by the same per-source conversion as PGM. SDR
or HDR is decided per rendition exactly as for program: H.264 with tonemap
for SDR, HEVC Main10 for HDR.

Default for the multiview: one rendition that follows the operator's monitor
choice. Switching Program SDR/HDR in the UI sends `aux.rendition {bus,
color}`; the demo stops the aux rendition group, rebuilds scale/format and
encoder for that color and codec, and restarts it. A Janus mountpoint pins
its codec and port (`janus.plugin.streaming.jcfg`: id 1 = H.264, id 2 =
H.265), so a follow rendition declares one port per codec (`ports`) and the
restarted encoder targets the port matching the codec; the preview page
reloads onto that mountpoint as it does today when the program Stream
select changes. The bus compositor and the program path are upstream and
keep running (program renditions all run at once and are never rebuilt);
the monitor shows about a second of black. NVENC cannot change codec or bit
depth on a live session, so the only glitch-free alternative is two
renditions and a second NVENC session, which the list allows when a show
needs it.

Parsing lives in `pyplumber/mixer/config.py`; the flattening and preset
expansion are pure functions next to `scene_layers`.

## Cost

| Item | Value |
|---|---|
| GPU | one compositor tick per frame at the bus canvas size, one NVENC session |
| Pads per aux bus | distinct sources + 1 (PGM), hard limit 64 |
| Multiview latency | main + 3 frames |
| Main program | unchanged |

Estimates for the T4 at 1080x1920p30; measured compositor numbers are in
implementation step 0. The composite kernel (`composite_planes` in
`cuda_rect_scale.cu`, one launch per frame) samples bilinearly, four taps
per output sample, so compositor cost scales with output pixels, not with
source size or how many tiles a source appears in.

| Per bus per frame | Amount | Share of T4 at 30 fps |
|---|---|---|
| canvas write, NV12 | 3.1 MB | 0.03% of 320 GB/s |
| bilinear reads | ~12 MB, mostly cached | ~0.1% |
| arithmetic | ~2 MP x tens of FLOPs | ~0.01% of 8 TFLOPS |
| kernel launch, one per frame plus the rect-table upload | measured 0.108 ms for grid-16 at 1080x1920 | ~0.3% of the 30 fps frame budget |
| NVENC session | H.264 5-8%, HEVC Main10 10-15% of the encoder engine | separate engine |
| VRAM | 250-550 MB: pool 25-65, retention 150-300, filter 10-30, NVENC 60-200 | |

Scaling: linear per bus in SM time and one NVENC session each. Buses share
decode, conversion and frames. NVENC saturates first: with program holding
two sessions, four to six aux buses fit on a T4.

Retention is the VRAM item to watch: the aux playout holds decoder frames
longer than PGM, so a tight decoder pool shows up as decoder stalls.

### Keeping the impact minimal

Done before any aux code (step 0):

1. **One launch per frame.** `composite_planes` draws the whole canvas in
   one launch from a host z-sorted rect table: each 128x8 tile masks the
   layers touching it, every thread walks those layers bottom to top
   starting from the clear value (opaque layers overwrite, RGBA blends) and
   stores once per sample. No clear pass, no per-layer launch. Launches
   fell from two per layer plus two memsets to one; grid-16 on the program
   path went from 0.160 to 0.108 ms per frame, under 1% of the mixer's GPU
   time. Every `cuda_rect_overlay` instance, so every aux bus, gets it.

Sources are assumed to deliver a new frame every tick, so no static-tick
skipping is planned.

Config options, off by default:

2. Per-bus `fps` (integer divisor of the program rate): halves SM and NVENC
   at half rate.
3. Per-bus `width`/`height`: cost is proportional to area; 720x1280 is 45%
   of 1080x1920 for compositor and encoder alike.
4. Aux renditions default to NVENC preset `p1`.

Later, if more than two buses become normal: **wall mode**, one compositor
drawing all bus canvases side by side and one crop per rendition. One
thread and one launch set for all buses; encoders unchanged.

Quality: bilinear at 4-16x downscale aliases. If grid tiles shimmer on the
T4, render a shared quarter-res thumbnail per source once per frame (about
1% more) and let aux layers read it. Decide after the live test.

## Native changes, complete list

| Change | Where | Size |
|---|---|---|
| `aux_output_mask` ORed into every camera OTM mask | `MixerState.hpp`, `avplumber.cpp` (`mixer.init`) | ~10 lines, required |
| several layers per compositor pad | `compositor_layers.hpp`, `cuda_rect_overlay.cpp`, `routing.hpp`, `SceneDefinition` | moderate, required, see below |
| rect table cap | `AVP_RECT_MAX_LAYERS` is 256 ops per frame (`cuda_rect_table.h`; `CudaRectDraw::draw` throws above it): raise it for the largest bus, and the Python bus builder rejects a bus whose flattened layer count exceeds it | small |
| nothing else | `one_to_many` masks are 32-bit; `cuda_rect_overlay` already takes `layers` and `active_inputs` at runtime; `node.object.set` exists; `split` has a `drop` option | |

### Multi-layer pads instead of aliases

An alias is Python-only today: `cam1#2` is an unrelated source to the
orchestrator, and the sharing is graph plumbing (one color filter, one
`color_alias_cam1` fan-out). With several aux buses the alias count would be
the max occurrences across every bus's tiles, every compositor would pay for
pads it never draws, and 64 is reached with a handful of grids. Porting the
alias concept to C++ would not fix that: a pad per occurrence is the cost.

So aux buses use a different mechanism: one compositor input carries a list
of layers. A scene is an ordered list of `(source, rect)` items, the
compositor draws a pad as many times as it is referenced, and pads, OTM
masks and `active_inputs` stay per source. Touch points:

- `SceneDefinition.sources` map to an ordered item list (`MixerState.hpp`),
  and the `mixer.scene` JSON accordingly.
- `compositorLayersFromScene` groups items by input index
  (`src/mixer/routing.hpp:81`).
- `parseLayersArray`, `applyLayerMetadata` and `resolveDrawOps` iterate
  layers per input (`compositor_layers.hpp`), as do `default_layers_` and
  `mergeLayersForTick` in `cuda_rect_overlay.cpp`. `CudaRectDraw::draw`
  only fills the rect table from the resolved ops and is untouched. Playout
  is untouched; frame selection is per input.
- `scene_layers` in `config.py` can stop generating alias names for main
  scenes too, in a follow-up; the main mixer keeps working with aliases
  meanwhile because a one-element layer list is the current behaviour.

Order of work: multi-layer pads first, since the aux compositor is built on
it; then `aux_output_mask`; then the Python bus builder and control.

## Out of scope for v1

- A second bus is built and tested in v1 only as a graph test (two entries
  in `aux_buses`, distinct bits, distinct encoders); live validation of two
  buses on the T4 is a follow-up. Each bus costs one compositor tick and one
  NVENC session; bits are 32-bit so up to 30 buses fit.
- Tally borders. Needs a solid-color source or a compositor fill layer; the
  compositor has no rect-fill primitive today.
- Transitions inside tiles, or feeding the multiview back into PGM.
- Snapshot and cut-latency measurement on the aux bus.

## Verification

- Unit: flattening math (contain/stretch, crop and z carried, a repeated
  source yields one pad with several layers, per-bus pad-limit and rect
  table cap rejection) in `demos/mixer/tests/test_graph.py`; the per-show
  64-source limit is already covered by `test_source_mask_capacity`.
- Graph: builder test that every `otm_<pad>` has three dsts with bit 2 set
  and that a cut/fade leaves bit 2 set.
- Live on the T4 host: record program and multiview together, confirm the PGM
  tile follows a fade with no repeats in the multiview's playout counters,
  and that re-targeting tiles does not disturb program timing.

## Implementation plan

Ordered so that every step leaves the tree green and the existing demo
unchanged in behaviour. Steps 1 and 2 are native, 3 to 7 Python and web.
One PR per step is fine; steps 3 to 6 may be combined.

### 0. Single-launch compositor draw, validated on PGM alone (native)

DONE 2026-09-22 on `mixer-improv`: commits 7a23cdd and 29a34cd (single
launch), bbfdb73 and 470bd58 (zero-copy browser frames), 6f584c2 (cleanup).
Every `cuda_rect_overlay` instance (slot A/B, wipe overlay, later each aux
bus) draws through `CudaRectDraw::draw`, so the existing 9:16 demo was the
test bed and no aux-bus code was involved.

As built (`cuda_rect_scale.cu`, `cuda_rect_draw.{hpp,cpp}`,
`cuda_rect_table.h`): the host resolves the z-sorted layer list into a rect
table (`AvpRectLayer`: source planes or texture, crop, destination rect per
canvas plane, kind and blend flags), uploads it once per frame and launches
`composite_planes` once (`composite_planes_yuv` when no packed-RGB layer is
present, `composite_planes_packed4` for rgb0 canvases), with `gridDim.z`
spanning the planes. Each 128x8 tile masks the layers touching it; each
thread owns four lane groups of one row and walks those layers in draw
order from the clear value: YUV and promoted layers overwrite with bilinear
samples, opaque RGB converts in place, RGBA blends over what is below; one
vector store per sample. No clear pass, no per-layer launch, no per-layer
stream sync. The per-layer path and its `draw=layered` flag were removed
once parity held (29a34cd); the old kernels live in
`tests/cuda/legacy_rect_kernels.cuh` as the oracle for
`tests/cuda/test_rect_composite.cu` (mixed, clear, odd-width, grid16, P210
HLG, P010 PQ, packed RGB0 and texture-backed RGBA layer sets, ±1 code on a
few blended samples from FMA contraction, plus grid16 timing).

Measured on the T4: grid-16 at 1080x1920, old 0.160 ms per frame (2
memsets + 32 launches) vs new 0.108 ms (1 launch); with one RGBA layer
0.170 vs 0.129 ms. Live 64-source show: mixer process SM 28.0% before,
28.5% after, i.e. no visible change, because the compositor is under 1% of
the mixer's GPU time. `nsys` on the live show (110 s): 84% of kernel time
is the NVDEC surface unmap kernel `ConvertNV12BLtoNV12` (38 streams), 11%
the compositor, 5% NVENC-side conversions; kernels total only ~4% of wall
time. Memory copies dominated: 26 DMA-BUF browsers x 8.3 MB RGBA x 30 fps =
5 GB/s of array-to-device copies (78 µs each) plus the decoder's per-plane
copies (1.7 MB, 187k per 110 s). NVDEC sits at 92%. So the levers for that
show are the browser import and decode count, not compositing.

Zero-copy browser frames (bbfdb73): `drm_prime_to_cuda` `zero_copy` hands
out frames that reference the cached EGL mapping, pitch-linear imports as
plain device-pointer frames and tiled imports through a texture object
described in `opaque_ref` (`cuda_rect_texture.h`; descriptor built by
`rectTextureDesc` in `cuda_rect_sampler.h`). The frame pins the DRM input
so the producer's release ack follows the last consumer, and the compositor
samples `AVP_RECT_KIND_RGB_TEX`/`RGBA_TEX` layers with the same taps. Live
on the same 64-source show: card 48% -> 39%, mixer process SM 28.0% ->
15.4%, mem 19.3% -> 10.9%, VRAM 7.36 -> 7.06 GB, NVDEC/NVENC unchanged.
Producer pool `DMA_BROWSER_DMABUF_POOL_SIZE=11` bounds outstanding buffers;
the playout keeps at most 8 queued + 1 held per input, and the producer
drops rather than overwrites when the pool is exhausted.

Follow-ups the same day: 470bd58 sets `CU_TRSF_READ_AS_INTEGER` on the
zero-copy texture (flags 0 made the `uchar4` fetch return bytes of the
float bit pattern: 255 read as 0, 127 as 255, so opaque white vanished and
Chrome's 50% dither became a dot grid); the parity test now builds its
textures through the production descriptor and fails without the flag.
6f584c2 drops the `scale` parameter from `cuda_rect_overlay` (kernels load
at init), keeps the canvas contract only in `CudaRectDraw::canvas()`, and
stops the DRM node's destructor from terminating the process-global EGL
display. Live after both: card 37-38%, mixer SM 16-18%, VRAM 6.9 GB.

### 1. Multi-layer compositor pads (native)

- `src/mixer/primitives/compositor_layers.hpp`: `parseLayersArray` accepts,
  per input, either one layer object (today) or an array of layer objects;
  `resolveDrawOps` emits one `DrawOp` per layer, in array order, still
  sorted by `z`. `LayerSpec` gains nothing.
- Draw path already single-launch from step 0; multi-layer pads only
  change how the op list is resolved (`resolveDrawOps`), not the rect table
  or the kernel.
- `src/nodes/hwaccel/cuda_rect_overlay.cpp`: `default_layers_` becomes
  `std::vector<std::vector<LayerSpec>>`; `input_eof_`, `held_` and friends
  stay per input. The `layers` `setObject` path and the composite loop use
  the nested form. A one-element array is exactly today's behaviour.
- `src/mixer/primitives/MixerState.hpp`: `SceneDefinition.sources` becomes an
  ordered `std::vector<SceneItem{source, SourceLayout}>`;
  `computeActiveInputsMask` and the inline `scene.sources.count(name)`
  checks (`scene.cpp`, `fade.cpp`, `wipe.cpp`, `routing.hpp`) adapt, behind
  a new `usesSource(name)` helper. `mixer.scene` JSON
  accepts the current object form and a new `items` array form.
- `src/mixer/routing.hpp`: `compositorLayersFromScene` groups items by input
  index into arrays.
- Tests: `tests/cpp/test_mixer_source_mask.cpp` and a new
  `tests/cpp/test_compositor_layers.cpp` covering one-vs-many layers per
  input, z order across inputs, and unchanged output for the object form.
- Acceptance: existing demo configs render identically; `pytest tests` and
  the C++ tests pass.

### 2. Aux output mask (native)

- `MixerState.hpp`: `uint32_t aux_output_mask = 0;` and
  `sourceOutputMask()` ORs it in.
- `src/avplumber.cpp` `mixer.init`: accept `aux_outputs` (integer mask);
  `mixer.status` reports it.
- Test: `tests/cpp/test_mixer_source_mask.cpp` asserts the mask survives
  `applyPostTransitionRouting`, fade cleanup and wipe midpoint values.
- Acceptance: with `aux_outputs: 0` nothing changes.

### 3. Config: `aux_buses`, places, preset, flattening (Python)

- `pyplumber/mixer/config.py`: dataclasses `AuxBus`, `Place`, parse
  `aux_buses`, `layout.preset` expansion (`pgm_pvw_grid`, fractional, even
  pixels), explicit `places`, per-bus `renditions` with `color: "follow"`,
  `latency_ms` default main + 3 frames. Validate scene ids, distinct
  sources + 1 <= 64 per bus, at most one `pgm`/`pvw` place per bus.
- Pure functions next to `scene_layers`: `flatten_scene(cfg, scene, dst)`
  and `bus_layers(cfg, bus, slots, pvw_scene)` returning the per-input
  layer arrays and the active mask.
- Update `doc/research/2026-09-08-mixer-config-schema.md`.
- Tests in `demos/mixer/tests/test_graph.py`: preset geometry for 9:16 and
  16:9, remap math with crop/fit/blend/z, repeated source across tiles
  yields one pad with several layers, pad-limit rejection, bad references.

### 4. Graph builder (Python)

- `pyplumber/mixer/graph.py`: `add_aux_bus(bus)` before `build()`. Per
  source pad, `otm_<pad>` gets one more dst per bus; initial `outputs` and
  `mixer.init` `aux_outputs` carry the bus bits. Per bus: `CudaRectOverlay`
  `aux_<id>_comp` (bus size, canvas format/color, `latency_ms`, timeline,
  group `aux_<id>`), initial `layers` from `bus_layers`, `active_inputs`.
  The PGM pad edge is returned so the demo can wire the split.
- `demos/mixer/mixer.py` `_build_from_config`: call `add_aux_bus`, extend
  the `final_out` split (or add a `one_to_many` with `drop: true` on the aux
  branch) with one output per bus that has a `pgm` place, run
  `_build_renditions` on each bus output with names `aux_<bus>_<rendition>_*`,
  and start group `aux_<id>` after `start_output` in `MixerApplication.start`.
- Tests: builder test that every `otm_<pad>` has 2 + N dsts and the init
  carries `aux_outputs`; a two-bus config yields distinct bits, compositors
  and encoders; the split has one extra output per PGM-showing bus.

### 5. Control (Python)

- `pyplumber/mixer/control.py` / demo: `aux {"bus", "scenes"}` computes
  `bus_layers` and issues `node.object.set` `layers` then `active_inputs`
  on `aux_<id>_comp`; `aux.rendition {"bus", "color"}` stops the bus
  rendition group, rebuilds it for the color, restarts it.
- `/api/state` gains `aux: {<id>: {scenes, pvw_scene, color}}` from a
  demo-registered command (`mixer.aux_status`, registered like
  `mixer.settings`); `mixer.status` is native (`MixerOrchestrator::status`)
  and stays unchanged.
- PVW place refresh after `preview`, `cut`, `fade`, `wipe`.
- `mixer.settings`: `preview_outputs: [{id, codec, port}]` replacing
  `preview_codecs` (keep the old key for one release).
- Tests: `demos/mixer/tests/test_control.py` for command payloads, order
  of object sets, and PVW refresh.

### 6. Web UI

- `demos/mixer/webui/index.html`: multiview strip (slots `M1..Mn` from the
  bus's `scene` places), drag/drop, armed-slot click, `Shift+1..8`, `x` to
  clear, `M<n>` badges on scene tiles, one `aux` command per change, state
  re-sync from `/api/state`.
- `docker-compose/images/preview/index.html`: `Stream` select lists
  `preview_outputs`; choosing the multiview keeps the program color choice
  and the UI sends `aux.rendition` when the program color changes.
- Tests: `demos/mixer/tests/test_webui.py` for `/api/state` shape and the
  `aux` command path; `docker-compose/images/preview/tests` for the select.

### 7. Live validation on the T4

- Run the 9:16 demo with one multiview bus, SDR rendition. Record program
  and multiview; verify the PGM tile follows a fade and a wipe, no repeats
  or discards in the aux playout counters at `latency_ms = main + 3
  frames`, program cut latency unchanged, re-targeting all eight slots
  during a fade leaves program timing untouched.
- Record `nvidia-smi` VRAM and SM/NVENC utilisation before and after
  enabling the bus; replace the estimates in Cost with the numbers. Target
  under 1% SM per bus at 1080x1920p30; record the aux playout
  `repeats`/`discarded` counters with browser sources.
- Judge tile aliasing on grid_16 tiles; decide on the shared thumbnail.
- Switch the rendition SDR/HDR/SDR three times; confirm the program never
  glitches and the multiview returns within about a second.
- Two-bus config: graph test only in v1.

### Done when

- `demos/mixer` runs with and without `aux_buses` and all tests pass.
- The multiview shows PGM with transitions, PVW, and eight operator-chosen
  scenes on its own Janus port, selectable in the preview page.
- The spec's Cost table carries measured numbers.
