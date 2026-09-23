# Multiview aux bus

Status: v1 scope agreed; implementation plan revised 2026-09-23 against
`mixer-improv` at cff498d. The completed compositor work is recorded in the
appendix. This document supersedes the earlier follow-color, drag/drop and
main-scene migration proposals; it is the implementation handoff.

Companions: [original proposal](https://claude.ai/artifact/5RLau7GkWSHPuWnv3q59QG)
and [control UI mock](https://claude.ai/artifact/1K4ZuwSxcJAP6hAgN3Nm3o).
The decisions below take precedence over those mockups.

## Agreed v1

- Ship one multiview: PVW and PGM above eight operator-selected **scenes**.
  Lower tiles can contain grids, PiP and alpha compositions, not just sources.
- Only the PGM tile shows live transitions and program overlays. PVW and
  lower tiles show scene compositions and change by cuts.
- One fixed SDR/H.264 aux rendition. Program SDR/HDR viewing changes never
  restart it. On an HDR show, tonemap the composed aux output to SDR.
- Program 50/60 fps produces aux 25/30 fps respectively; program 25/30 fps
  keeps the same aux rate. Same canvas size/aspect as program by default.
- PGM frame pacing and render performance are the top priority, ahead of aux
  smoothness or throughput. Aux may drop/repeat or be suspended under overload; it
  must not introduce blocking dependencies into the program path.
- Two rendering streams in v1: the existing M/E/PGM stream and one dedicated
  nonblocking aux stream, sharing the CUDA context and source frames. Aux may
  wait for source/program readiness; PGM never waits for aux completion or
  queue space. Do not split the existing M/E internals into further streams.
- Click a slot, then a scene to assign it; x clears it. With a scene
  selected, Shift+1..8 assigns it to that slot. No drag/drop.
- Keep the 256 draw-layer ceiling for v1 and reject invalid assignments
  before changing the running composition.
- No layout editor, bus selector, Master DSK, extra M/E, nested-scene
  authoring or rendered-scene cache in this implementation.

## Architectural boundaries

The fixed v1 UI must not become a singleton assumption in the graph.

1. **Sources own ingest.** Decode/upload/normalize each configured source
   once, then share reference-counted GPU frames. Aliases or occurrences do
   not create decoders. A scene, M/E or aux bus does not own the source.
2. **Scenes are definitions.** A scene reference resolves to ordered source
   placements. Flattening means drawing instructions, not a rendered texture.
   Render it directly at the tile size; no full-canvas intermediate per tile.
   Repeated occurrences share frames but still incur drawing work.
3. **Outputs are finished frames.** Keep an output reference distinct from a
   scene reference. The PGM tile reuses the actual program output, including
   transitions, rather than reconstructing its scene.
4. **M/E state is independent of scenes.** A future M/E owns its own PGM/PVW
   selection and transitions while sharing sources and scene definitions.
   Do not put transition state into a scene or the aux tile renderer.
5. **Aux buses own presentation.** Keep `aux_buses[]`, stable bus/place IDs,
   layout expansion, timing and renditions per bus. Multiple layouts and
   buses should require extending configuration/UI, not replacing the model.
6. **Routing indices are plumbing.** For v1, destinations 0/1 are the existing
   M/E slots and bus i uses bit 2+i. Allocate these in the builder, not in
   public references or render kernels. Future M/Es need additional named
   destinations. Validate the current 32-bit routing limit (30 aux buses)
   rather than implying unlimited capacity; multiple live buses remain deferred.

Use a small content resolver with tagged references: a named scene, the
current preview scene of a named M/E, or a named rendered output. V1 registers
only the existing M/E and program output. Logical examples are
`scene(cam1_full)`, `preview(main)` and `output(program)`; finalize their JSON
spelling in config work, without introducing a general graph framework.

Future Master DSK is a branch after the M/E: the base output is clean, the
DSK composition yields dirty. Both share the base render and decoding. Named
output taps can then expose clean/dirty or another M/E's PGM. A clean feed
still includes scene content and M/E transitions; only downstream layers are
excluded. Do not implement those taps/controllers now or equate an output
reference with a scene. Future nested scene references belong in the resolver
with cycle detection and expanded-layer accounting, not recursive GPU callbacks.

### Future Vulkan backend

V1 remains CUDA-only. Preserve a replaceable rendering boundary without
building a universal GPU framework or adding a Vulkan implementation now:

- Keep scene/reference resolution, M/E state, transition timing, geometry,
  playout deadlines, routing and overload policy independent of GPU APIs.
  Existing configurations, control commands and UI should survive a backend change.
- Pass ordered draw descriptions to the renderer: source references,
  destination/crop geometry, draw order, blend and color requirements. Keep
  CUDA pointers, texture handles and streams inside the CUDA implementation.
- The backend owns GPU allocation/import, sampling/composition, conversion
  and readiness/completion synchronization. Retain shared frames until GPU
  consumers finish; do not assume CPU submission means a buffer can be reused.
  Reuse existing frame ownership and scheduling interfaces where sufficient.
- Express independent output submission and one-way dependencies as the
  contract. The two CUDA rendering streams implement it today; a future
  Vulkan backend may use queues, semaphores and barriers without exposing
  their arrangement in scene or aux configuration. PGM priority and bounded
  aux retention/work remain mandatory for either backend.
- Preserve the existing separation between playout/geometry and CUDA draw
  code. Introduce only small interfaces needed by this implementation; do not
  refactor unrelated nodes, add backend switches, or generalize the framework
  merely to anticipate Vulkan. Decoder/encoder integration and browser frame
  imports will still require backend-specific work and capability validation.

Backend parity would be tested on identical geometry, color, alpha and timing
semantics. Do not assume CUDA is intrinsically faster or that Vulkan removes
shared-resource contention; performance comparisons require matched workloads.

## Layout and composition

`pgm_pvw_grid` expands to ordinary places, not a special compositor mode.
V1 uses rows=2, cols=4:

| Place | Rectangle on W x H canvas | Content |
|---|---|---|
| PVW | 0, 0, W/2, H/2 | current preview scene |
| PGM | W/2, 0, W/2, H/2 | named program output |
| M1..M8 | c*W/4, H/2+r*H/4, W/4, H/4 | assigned scene or empty |

For 1080x1920: PVW/PGM are 540x960; lower tiles are 270x480.
For 1920x1080: 960x540 and 480x270. Round shared boundaries to chroma
alignment so adjacent rectangles do not acquire gaps or overlap.

A referenced scene's destination rectangles are transformed into the tile:

```
s = min(tile.w / scene_canvas.w, tile.h / scene_canvas.h)
ox = tile.x + (tile.w - scene_canvas.w * s) / 2
oy = tile.y + (tile.h - scene_canvas.h * s) / 2
dst_x' = ox + dst_x * s; dst_w' = dst_w * s  # likewise y/h
```

Preserve source crop, fit, blend and scene draw order. Resolve omitted sizes
and fit against the original scene geometry before remapping. Clip at the
scene/tile boundary with corresponding sampling coordinates: an off-canvas
layer must not spill into the next tile. Define place order and stable layer
order separately so future overlapping places do not interleave scene layers.
Empty/letterbox regions use the compositor background. Custom geometry needs
no v1 editor; keep the normalized place representation able to express it.

Same source in several tiles uses **one input pad with several layers**.
PGM uses one pad/layer regardless of its internal scene complexity. No
intermediate scene texture is required. Shared rendered thumbnails are a
possible later optimization, not a prerequisite or an automatic cache.

## Capacity and validation

There are two different limits:

- Current main scenes have up to 128 input pads, including occurrence aliases.
  The show's complete pad allocation can constrain a scene further.
- A compositor draws at most 256 layers (`AVP_RECT_MAX_LAYERS`). Multi-layer
  pads decouple draw count from input count; drawing remains bounded.

For aux, count all lower scene layers + PVW scene layers + one PGM layer.
Repeated placements count even when they share a decoder. Reserve the maximum
expanded PVW layer count across scenes selectable by the existing M/E so a
normal preview/take cannot invalidate aux:

- 8 grid-16 tiles + grid-64 PVW + PGM = 193: fits.
- 8 grid-16 tiles + grid-128 PVW + PGM = 257: rejected.
- 8 grid-64 tiles + grid-64 PVW + PGM = 577: rejected.

Do not raise the cap in v1. Report the required count, reserved PVW count and
limit, leaving the old assignment active on rejection. Revalidate definitions
on setup reload. 256 is an implementation ceiling, not a permanent schema
restriction or a promise that every valid layout meets the frame budget.

For v1, scene definitions are fixed for the lifetime of a setup with aux enabled.
Build native scenes and aux resolution from the same validated configuration;
geometry/source-placement changes require setup reload. Enforce this in native
scene mutation handling, including `mixer.scene` calls from other clients, with
a clear reload-required error. Reject additions/replacements/removals before
changing state. Establish the guard during setup before exposing control, and
keep it across aux suspension or restart. No-aux setups retain existing scene
editing behavior. Cuts, PVW selection and aux tile assignments remain dynamic.
Live definition synchronization and scene revision tracking are deferred.

Separately validate the bus's source universe: distinct decoded source pads
plus referenced output pads must fit the 128-bit `SourceMask`. Map them to
stable bus-local indices, deduplicating main-scene aliases. Reserve all sources
needed by permitted assignments/PVW; do not silently grow/reindex a running
compositor. A show with 128 distinct sources plus PGM cannot all fit this bus
without a later mask/capacity change or an explicit restricted source universe.

## Graph, timing and PGM isolation

1. Add a destination per aux bus to existing per-source `one_to_many` nodes.
   Keep live aux subscriptions separate from the existing A/B timeline masks:
   apply them at fan-out execution, not in scheduled `sourceOutputMask()`
   values that a later fade/wipe cleanup could restore. Subscribe each bus to
   the union of sources in its current PVW and assigned scene tiles; subscribe
   its PGM tap while the bus runs. Default to no aux subscriptions. Aux changes
   must never modify main A/B routing, `prewarm_source_mask`, main playout
   history or source lifecycle. Existing main prewarming remains unchanged.
2. Build one `aux_<id>_comp` with the program color/working-format contract,
   source frames shared after their existing conversion, and a PGM output pad.
   One source pad can resolve to multiple draw operations. Only active pads
   should retain frames; exclude main aliases from aux pad accounting.
3. Tap the real program output. Existing `Split::drop` and `OneToMany::drop`
   apply to every destination, not per destination. Merely enabling `drop`
   on the shared program split changes program semantics. Specify and test
   the smallest reusable fan-out change supporting a nonblocking aux destination
   while preserving current program destinations. A downstream dropping node
   alone does not prove isolation if its upstream queue can fill. Cover EOF
   and shutdown too; they must not block behind a stopped aux consumer.
4. Lower aux cadence before expensive drawing and bound retained references.
   Validate PTS-based 50->25 and 60->30 selection for sources and PGM. At a
   lower output rate, intentionally discarded input frames are not failures.
   Do not feed full-rate frames into an eight-entry playout buffer with an
   unexamined longer delay: it can evict needed frames or pin producer pools.
   Use reference-only rate selection/decimation if needed, with no decode,
   pixel conversion, upload or full-size intermediate added.
5. Implemented latency: `D_aux = D_main + T_aux`, in wall-clock units,
   where a bus consumes program output. The initial three-frame allowance
   retained unnecessary source frames; one aux frame passed the live 60/30 fps
   run and compositor restart check. This is not a universal no-repeat
   guarantee. Current playout allows six output-frame
   periods with eight queue entries. Also budget input-rate retention and
   producer/decoder surfaces. Larger configured main delay may exceed that
   budget and must produce a validation error, not silent eviction.
6. Give aux composition its own nonblocking CUDA stream. Keep the existing
   M/E/PGM stream and shared CUDA context. Preserve existing producer-readiness
   guarantees; where a producer is asynchronous, use frame-ready events for
   cross-stream dependencies. The program-ready event is recorded on the
   program stream and waited on by aux, never the reverse. Keep input frames
   and draw-table storage alive until aux GPU work completes. A wait on the
   aux stream by its own CPU worker is acceptable in v1; never synchronize the
   entire context or introduce a PGM wait for aux. Audit aux filters/conversion
   paths for accidental submission back onto the program/default stream.
7. Feed aux output through its own SDR/H.264 rendition. Use per-bus names and
   an independent lifecycle and a bounded, nonblocking encoder handoff. Skip
   aux work on congestion; sustained congestion suspends the bus and closes
   its subscriptions. Aux renderer/encoder failures or stalls release references
   safely as below and do not stop program. Start it after existing program
   startup; bound missing-input warmup rather than waiting forever for every tile.

Aux input ownership must follow subscription lifetime. On removal, suspension
or stop, first disable delivery and acknowledge that the fan-out has applied
the change, then drain that aux input's edge and clear its playout history and
held frame. Serialize this with the aux consumer; a publisher already in flight
must not repopulate the queue after cleanup. Keep references used by submitted
draws until their CUDA completion, handled by the aux worker without making
PGM wait. Re-enabling an input must not revive frames from its old subscription.
New inputs may warm up with background in their tiles; main cuts stay warm.

An inactive aux input owns zero producer frames after outstanding GPU use
completes. An active input has a fixed retention budget covering its edge,
playout history, held frame and in-flight draws. Nonblocking writes alone do
not satisfy this: dropping new frames leaves old queued surfaces pinned.
Decimate before retaining aux history, but preserve enough timestamp history
for the chosen PGM alignment delay; a latest-frame slot alone cannot replace it.
Release only aux-owned references, never main-mixer prewarm references.

Default main playout tolerance remains unchanged. Aux may repeat the last
available tile or show background for a missing source, while other tiles
continue. Never overwrite producer buffers still in GPU use to meet a budget.
Use measurements to reduce aux workload or refuse activation if PGM cannot
be protected; a shared GPU has no absolute real-time isolation guarantee.
Compare PGM p50/p95/p99/max frame times, missed deadlines, repeats and cut latency
against an aux-off baseline under the same source/scene load. Average FPS or GPU
utilization alone is insufficient. Reproducible aux-induced deadline misses or
frame-time tail regressions beyond baseline variability fail acceptance: bound
aux work in flight, skip work or suspend aux before sacrificing PGM pacing.
An overloaded aux must not build a GPU submission backlog ahead of PGM.

## Control and output discovery

Keep commands scoped by bus, even with only one exposed:

```
aux {"bus":"multiview","expected_revision":12,"scenes":["cam1_full","two_up",null,null,null,null,null,null]}
```

The complete eight-slot list makes replacement and clearing unambiguous.
Require `expected_revision` for assignment commands. The backend owns a revision
per bus; atomically compare it with the current revision, accept the validated
assignment and increment the revision. Concurrent requests based on the same
revision cannot both succeed. Missing/stale revisions leave state unchanged and
return an error with current assignments and revision. Clients refresh on
conflict rather than automatically resubmitting a stale eight-slot list.
Expose assignments/revision in command responses and `/api/state`; reconnect
reads them before editing. Invalidate old revision tokens on setup reload or
backend restart so requests from an earlier instance cannot match fresh state.
Automatic PVW changes do not increment the assignment revision.

Validate/resolve the entire update, then atomically publish layers and active
inputs as one composition snapshot at a frame boundary. Two separate
`node.object.set` calls are not sufficient: current masks and layers have
separate locks and the render thread can see an intermediate combination.
Keep critical sections short; do not hold configuration locks while drawing,
waiting for CUDA, making HTTP calls, rebuilding encoders or sending frames.

PVW follows authoritative M/E status, including transition completion and
commands from other clients; it must not be inferred only from UI clicks or
from the last requested destination. Backend state is authoritative for aux
assignments too. Invalid commands leave existing state/output unchanged.
`/api/state` adds per-bus state from a demo-registered status command; native
`mixer.status` is not a Python extension point.

UI has M1..M8 assignment slots, click-to-arm, click-to-assign, clear, scene
badges and Shift+1..8. Reconnect reads backend state. No drag/drop, bus selector
or layout editor. Preserve assignments while the backend remains running;
restart defaults come from the config unless existing persistence provides more.

`mixer.settings.preview_outputs` identifies bus/rendition, codec, color and
mountpoint/port. Keep `preview_codecs` during migration. The v1 selector adds
one SDR multiview entry without changing program HDR/SDR selection. Provision
its distinct Janus mountpoint; a new RTP port alone does not create one.
There is no `aux.rendition` color-follow command or encoder rebuild in v1.

## Config outline

```json
"aux_buses": [
  {"id": "multiview",
   "layout": {"preset": "pgm_pvw_grid", "rows": 2, "cols": 4},
   "scenes": ["cam1_full", "two_up", "grid_4", "pip_cam2",
              "grid_8", "grid_16", "cam1_with_graphics", "cam3_full"],
   "renditions": [{"id": "monitor", "target": "janus", "color": "sdr",
                   "codec": "h264", "port": 5008}]}
]
```

This is a schema outline; align rendition fields with the existing parser
rather than introducing a second output schema. Canvas defaults to program,
fps follows the agreed mapping. Normalize the preset to stable place IDs,
rectangles and tagged content references. Keep bus IDs, rendition IDs and
output/port uniqueness validated. The model accepts a list; only the agreed
single-bus layout is exposed/tested live in v1. No hardcoded single-bus storage.

## CUDA versus libobs: synchronization review

Source review on 2026-09-23 used latest stable OBS **32.2.2**, confirmed by
[the release](https://github.com/obsproject/obs-studio/releases/tag/32.2.2).
This is a code comparison, not a matched performance benchmark or a diagnosis
of any historical OBS incident.

- In [libobs graphics.c](https://github.com/obsproject/obs-studio/blob/32.2.2/libobs/graphics/graphics.c),
  `gs_enter_context` takes the graphics object's mutex until the matching outer
  leave. In [obs-video.c](https://github.com/obsproject/obs-studio/blob/32.2.2/libobs/obs-video.c),
  one graphics loop renders output mixes, then displays; output mixes are
  traversed under `mixes_mutex`. Slow work on that path can delay other work.
- [obs-scene.c](https://github.com/obsproject/obs-studio/blob/32.2.2/libobs/obs-scene.c)
  holds a scene video lock while traversing/rendering items.
  [obs-display.c](https://github.com/obsproject/obs-studio/blob/32.2.2/libobs/obs-display.c)
  invokes display callbacks under their mutex. Linux
  [X11/EGL](https://github.com/obsproject/obs-studio/blob/32.2.2/libobs-opengl/gl-x11-egl.c)
  and [Wayland/EGL](https://github.com/obsproject/obs-studio/blob/32.2.2/libobs-opengl/gl-wayland-egl.c)
  bind/unbind the GL context on context entry/exit. These are serialization
  points; their existence does not prove an observed stall was mutex contention.
- OBS also reuses compatible mix textures (`can_reuse_mix_texture`); do not
  characterize it as always rerendering or decoding each reference.
- Our `cuda_rect_overlay.cpp` snapshots layer/mask state with short locks and
  draws without holding those locks. Compositors are separate graph nodes;
  flat draw operations avoid recursive scene/plugin rendering callbacks.
- However, `CudaRectDraw::stream()` in `cuda_rect_draw.hpp` returns the shared
  `AVCUDADeviceContext::stream`. Compositors using the same hardware device
  therefore submit to the same stream. `processComposite` synchronizes it
  every frame before a blocking sink put. We do **not** currently have fully
  independent GPU scheduling merely because nodes have independent threads.
- Shared stream work, default-stream interactions, driver waits, output queues
  and retained NVDEC/DMA-BUF surfaces can still couple aux to PGM. Browser
  capture still uses Chromium/GL; CUDA composition does not remove its stalls.

V1 must separate aux from the existing M/E/PGM rendering stream as specified
above; this is an agreed requirement, not an optional optimization. Measure
kernel execution separately from CUDA queue/wait time and CPU mutex/queue
waits with aux on/off and stalled. Validate producer readiness, consumer
completion and DMA-BUF release lifetime before enabling the new stream. Do not
remove synchronization or add `glFinish` / `cuCtxSynchronize` as a workaround.
Stream priorities alone do not guarantee preemption or protect against
exhausted shared frame pools. "PGM never waits for aux" is the dependency and
backpressure contract, not a claim that shared GPU execution/memory resources
provide hard real-time isolation.

## Cost and current jitter evidence

Per bus: one canvas composition at aux fps, one SDR conversion when required,
and one H.264 NVENC session. Source decoding/conversion are shared; each
placement is sampled again at its final tile size. A single launch does not
make layers free: layer-table scanning, overlap, blending and source format
matter in addition to canvas area. Keep the 256 cap and profile worst valid
layouts. Additional live bus count and VRAM cost require measurement; do not
promise a fixed number of buses from rough FLOP or encoder percentages.

The passive five-minute 20-NVDEC/28-browser, 60-fps capture found:

| Metric | p99 | Maximum |
|---|---|---|
| Browser frame arrival spacing | 21.80 ms | 101.50 ms |
| Browser handoff-to-receipt transport | 0.30 ms | 2.99 ms |
| Decoder frame-return spacing, before pacing | 29.06 ms | 47.71 ms |
| Encoded output spacing | 20.15 ms | 35.46 ms |

Browsers averaged 60 fps with zero transport sequence gaps/reported drops,
but 437 per-source gaps over 50 ms formed 114 clusters, each confined to one
Electron worker (four workers, seven pages each). Catch-up bursts hide stalls
in average FPS. The exact blocked operation was not established. NVDEC was
92-96%; file pacers repeated 0.52% of output frames overall. This was a single
fullscreen program scene, not proof of smooth all-source grid rendering.

Do not promise zero aux repeats in browser tests or attribute all discards
to overload: half-rate selection deliberately discards frames. Compare against
baseline, separate transport loss, timestamp corrections, intended decimation,
late frames and repeated output. Do not increase global mixer latency just to
hide upstream browser stalls. Adding aux must not create another decoder.

## Implementation sequence and acceptance

1. **Native compositor support.** Accept one or several layers per input,
   preserve the existing single-layer form and deterministic global draw order.
   Implement one atomic composition update for layers + active mask, plus tile
   clipping support if current geometry cannot express it safely. Keep frame
   selection per input. Do not migrate `SceneDefinition.sources`, native
   `mixer.scene` or main aliases merely to support Python-generated aux layers.
2. **Routing/isolation.** Add independent live aux subscriptions and tested
   per-destination nonblocking fan-out as above. These are reusable node/control
   changes, not a graph-management framework workaround. Test cut/fade/wipe
   paths and stopped/slow aux consumers; default no-aux behavior stays unchanged.
3. **Config/resolver.** Add bus/place data and typed content resolution in
   `pyplumber/mixer/config.py`, normalized geometry, source/output deduplication,
   128-pad and 256-layer validation with PVW reserve. Keep presets outside the
   renderer. Enforce fixed native scene definitions for aux-enabled setups.
   Update the existing mixer config-schema document.
4. **Graph builder.** Build per-bus compositor/rendition groups, shared-source
   routes and named program tap in `pyplumber/mixer/graph.py` and the demo.
   Implement the dedicated nonblocking aux stream and one-way readiness
   dependencies. Establish rate selection, bounded warmup/retention, latency
   validation and independent stop/restart before exposing the control UI.
5. **Control/UI.** Implement validated full-slot replacement with per-bus
   revision checks and conflict refresh, authoritative PVW synchronization,
   state/discovery and one Janus SDR output. Add clicks, clear, badges and
   keyboard shortcuts only; no drag/drop or color-follow path.
6. **Remote validation.** Use the configured NVIDIA host, not local GPU builds
   or local tests. Run relevant Python mixer/demo tests and native compositor
   tests. No live changes during unrelated monitoring without coordination.

Required tests:

- Old single-layer configs render identically; repeated references share one
  decoder/pad but yield the expected drawing operations. Test z ties, crop,
  contain/stretch, omitted geometry, negative/off-canvas layers and tile bounds.
- Boundary cases at 128 pads / 256 layers, PVW reservation, invalid assignments
  leaving state intact, and reload validation of changed scene definitions.
- Direct native scene mutations are rejected without changing main or aux
  definitions while aux is configured, including during suspension/restart.
  No-aux scene editing and normal cuts/PVW selection remain functional.
- 50->25, 60->30, 25->25, 30->30 PTS cadence; count intended selection separately.
  PGM tile shows fade/wipe/overlays while PVW/lower tiles remain direct scenes.
- Live SDR and HDR programs both yield SDR/H.264 multiview. Changing program
  viewer color does not restart aux. Existing program controls still work.
- Aux off/on/stopped/encoder-stalled: compare program frame times, cut latency,
  missed deadlines, decoder pool availability, browser retained/quarantined
  buffers and memory. Release aux references safely on failure/stop after GPU
  use completes. Verify the two distinct rendering streams and absence of any
  program-stream wait on an aux event. Profile event/stream waits and accidental
  shared-stream work; address measured coupling before declaring PGM protected.
- Repeatedly subscribe/unsubscribe while sources publish and fade/wipe cleanup
  runs. Verify no delivery after unsubscribe acknowledgement, no stale frames
  on reactivation, bounded active retention and zero inactive aux references
  after GPU completion. Main A/B routing, prewarm history and source lifecycle
  must remain unchanged. Include full aux queues and EOF during shutdown.
- Assign tiles during a transition and from two clients; reject invalid changes
  atomically, track actual PVW after completion, and restore state on reconnect.
- Two clients replacing assignments from the same revision: exactly one
  succeeds; the other receives current state without overwriting it. Cover
  missing revisions, reconnect and stale requests after setup/backend restart.
- A two-bus graph test proves distinct IDs, destination bits, layouts and output
  names without singleton assumptions. Live multi-bus and editable-layout UI
  validation remain deferred. No requirement to implement extra M/Es or DSK.

Done when the agreed single multiview works, no-aux configurations stay intact,
program isolation is measured under aux failure/load, and actual incremental
GPU/CPU/VRAM/frame-time costs are recorded. Historical numbers below are not
acceptance results for aux.

## Appendix: completed compositor work and historical measurements

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
