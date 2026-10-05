# Video Mixer

AVPlumber's mixer is a two-slot program/preview video switcher. The reusable
graph builder is `pyplumber/mixer/graph.py`; the native control implementation is in
`src/mixer/`; the maintained example is `demos/mixer/`.

## Native code layout

| Module | Role |
|---|---|
| `src/mixer/orchestrator/MixerOrchestrator.hpp`, `core.cpp` | `MixerOrchestrator` core: node adapters, interruption/abort, status |
| `src/mixer/orchestrator/{scene,cut,fade,wipe,overlay}.cpp` | one transition kind per file |
| `src/mixer/routing.hpp` | state → router route tables and compositor layer arrays (pure) |
| `src/mixer/graph_ops.{hpp,cpp}` | node/edge lookups, deferred `setObject`, readiness polls |
| `src/mixer/TransitionScheduler.{hpp,cpp}` | worker thread for scheduled transition steps |
| `src/mixer/Playout.hpp` | clocked playout: per-input queues, deadlines, frame pick (unit-tested) |
| `src/mixer/primitives/` | headers with no graph or CUDA dependency: `TickGrid`, `Cadence`, `MonotonicClock`, `CutLatency`, `CutLatencyProbe`, `Snapshot`, `OutputSnapshot`, `MixerState`, `TransitionGuard`, `PreviewFollow` (unit-tested where they carry logic), plus the compositor geometry headers below |
| `src/nodes/mixer_compositor.cpp`, `mixer_keyer.cpp` | mixer-owned compositor nodes: the clocked playout of the scene slots, wipe and AUX buses (`mixer_compositor`); the DSK (`mixer_keyer`) |
| `src/nodes/hwaccel/cuda_rect_compositor.hpp`, `cuda_rect_overlay.cpp` | the compositors' shared base (canvas, layers, control, drawing one frame); the generic unclocked `cuda_rect_overlay` |
| `src/nodes/mixer_snapshot.cpp`, `mixer_pvw_follow.cpp` | mixer-owned nodes: the output hold and slot substitution; an AUX bus's composition writer, which draws its PVW cells on the bus frame the bus's `pvw_align` picks ([AUX bus follower](#aux-bus-follower)) |
| `src/nodes/hwaccel/cuda_rect_draw.{hpp,cpp}`, `cuda_rect_scale.cu` | kernel module, canvas clear, per-layer draw |
| `src/mixer/primitives/compositor_layers.hpp`, `pixel_layout.hpp`, `compositor_geometry.hpp` | layer parsing and draw-op resolution, format geometry, placement (pure; geometry and layout unit-tested) |

The mixer carries video frames only. It has no audio routing, VAD, speaker
selection, face tracking, or camera policy.

## Backend boundary and lifetime

Shared code owns scene geometry, color intent, routing, subscriptions and timing.
`pyplumber/mixer/backend.py` defines node/filter construction; `backends/cuda.py`
implements it. Native fade requests use `src/mixer/transition_control.hpp`, with
CUDA command translation in `src/mixer/backends/cuda/`. Registered CUDA nodes
and kernels remain under `src/nodes/hwaccel/`.

A future Vulkan backend belongs alongside those CUDA implementations. PGM, AUX
and wipes use the same backend; its hardware handles and synchronization stay
private. Device creation, input interop and encoders remain explicit integration
boundaries. CUDA is currently the only implemented backend.

The demo uses the instance-owned device name `mixer_gpu`. Names prefixed with
`@` are process-global and need host-managed teardown before GPU libraries exit;
they must not be introduced as a shortcut for sharing PGM/AUX resources.

## Downstream compatibility

Applications outside this repository use two mixer pieces directly rather than
`MixerGraphBuilder`: the compositor node `pyplumber.node.CudaRectOverlay`
(`cuda_rect_overlay`) and `pyplumber.mixer.dmabuf_inputs.dmabuf_cuda_input_nodes`.
Keep the node name, its Python import and parameters, and the helper's
signature and return structure stable. The compositor in particular keeps:

- unclocked composition: upstream owns pacing. The clocked modes are the
  mixer's own nodes, `mixer_compositor` (`fps`, `aux_mode`) and `mixer_keyer`
  (`clock_input`); `cuda_rect_overlay` rejects those parameters rather than
  silently dropping a caller's clock;
- per-frame `metadata_key` layers, including moving crops, contain and two-box
  layouts, with the existing source-index and z-order semantics;
- program metadata propagation with the program placed last in the input list
  but drawn first, beneath the browser overlay;
- premultiplied browser alpha, texture-backed inputs and browser surfaces
  shared across output aspects, without added copies or retained frames;
- acceptance of legacy parameters such as `scale=True`: ignored parameters do
  not become errors as incidental cleanup.

Builder-level tests do not cover these uses; check a change against the
downstream graph and layout contracts, including an unclocked metadata/alpha
case on the GPU.

Geometry on one stream (crop, scale, placement, choice of frames by rate) is the
node `cuda_transform`. Applications build its parameters with
`pyplumber.transform.transform_output` and `transform_params` instead of
writing the JSON, so the engine changes in one place; keep both signatures and
the emitted keys stable (`tests/test_transform_params.py` pins them).

## Graph

Each source supplies one CUDA video-frame edge. A source can either fan out to
the two mixer slots through `one_to_many`, or use `preheat_video_router` outputs
for a catalogue that exceeds the compositor input limit.

```text
CUDA source frames
  -> source fanout (or catalogue router)
  -> mixer_compositor A / mixer_compositor B
  -> permanent transition_cuda
  -> source_switcher
  -> NVENC
```

Each slot has its own layer rectangles because an outgoing scene and incoming
scene must remain live at the same time during a transition. Scene changes are
scheduled against a shared timeline so router selection, compositor inputs,
and the program selector change at consistent frame timestamps.

`MixerGraphBuilder` returns the final video-frame edge. The application owns
input decode, output encoding/muxing, startup order, and shutdown.

## Preheating

Applications that need immediate transitions must preheat the complete path:

1. Start and pace all input groups; wait for input frames.
2. Start the catalogue router if used, then publish initial scene routes.
3. Start both compositors with the scaling kernel loaded and fixed output pools.
4. Temporarily feed both compositor outputs to `transition_cuda`, wait for a
   transition output frame, then restore steady routing.
5. Open the output gate for fresh frames, start the encoder/output group and
   declare the graph ready.

Sources registered with `default_graph=""` feed the compositors directly.
Their scene layers use `dst_x`, `dst_y`, `dst_w`, `dst_h`, optional `crop`, and
`fit` (`contain` or `stretch`). Dimensions and pitch are resolved from each
frame. Optional `source_canvas: {"w": 1920, "h": 1080}` preserves letterboxing
into that virtual source canvas without an intermediate GPU frame. Explicit
FFmpeg preprocessing graphs remain available to existing callers.

There is no useful cold fallback for a low-latency production graph. The
generic demo fails startup when any required preheat stage times out.

## Transitions

The control protocol accepts JSON objects:

```text
mixer.preview {"mixer":"mixer","scene":"grid_4_page_0"}
mixer.cut {"mixer":"mixer","scene":"grid_4_page_0"}
mixer.fade {"mixer":"mixer","scene":"grid_4_page_0","duration_sec":0.5}
mixer.fade {"mixer":"mixer","scene":"grid_4_page_0","duration_sec":1,"color":"#000000"}
mixer.wipe {"mixer":"mixer","scene":"grid_4_page_0","wipe_file":"<path>/wipe.mov"}
```

Cut, Fade and transparent media-file Wipe are supported. Fade uses the permanent
CUDA transition filter. The media wipe graph is predeclared. Decoding per take,
the orchestrator starts it on the selected clip and stops it after the wipe.
With the clip cache (`mixer.init` `wipe_cache_store` and `wipe_overlay`) the
player group runs for the life of the graph: the clip player and the wipe
compositor idle between wipes, and a take arms them in place (`play` on the
`clip_cache` node, a reset and `active_inputs` on the compositor), so a wipe
creates, starts or stops nothing. A new take can interrupt an ongoing
transition using the current output picture; interrupting a wipe parks the
chain the same way.

A fade with `"color"` (opaque RGB such as `"#000000"`, in libavutil colour
syntax) is a dip: program fades to that colour over the first half and the
colour fades to the new scene over the second. Between them the colour holds
alone for one frame period, so a frame shows it at any rate and start phase;
the halves share the rest of the duration. The `curve` shapes each half. The
colour is converted once per take for the canvas (`mixer.init` `"color"`:
`sdr`, `hlg` or `pq`, default `sdr`) with the compositor's RGB-graphics
maths, so it is never raw RGB in YUV planes: black is
Y 16 and white Y 235 (8-bit SDR), and on HLG white is graphics white (203 nits,
75% signal), not peak. The same `transition_cuda` pass does the dip, reading
only the picture still visible (none during the hold), so a dip costs no more
than a crossfade: two launches per frame, no extra buffers or passes, and
nothing while no transition runs. Readiness, timing, interruption and cleanup
are the crossfade's; an interrupted dip keeps the picture it had reached,
which at the midpoint is the solid colour.

After a completed take the preview is the scene that left program (OBS's
"Swap Preview/Program Scenes After Transitioning", always on; a failed take,
or one dropped by `mixer.interrupt`, clears the preview instead, while a take
that replaces a pending one leaves the preview as it is until its own switch,
so an aux bus's PVW cells do not blank between takes). The swapped preview
is only shown: the slot it came from has its sources routed away, so
`mixer.status` distinguishes `pvw_scene`, what the
operator and the AUX buses see, from `pvw_slot_scene`, what is loaded in
the PVW slot (`""` while cold). A cut reuses the slot only for `pvw_slot_scene`;
any other scene, a swapped preview or `mixer.init`'s `initial_pvw_scene`
(published to the followers, not loaded) included, is loaded first, warm when it is in `mixer.prewarm`.
An AUX bus shows each preview change as described in
[AUX bus follower](#aux-bus-follower).

`mixer.status <name>` returns the current PGM/PVW scene and transition state,
and under `playout` each slot compositor's (`A`, `B`) running `frames`, `repeats`
and `missed_deadlines`, published every 60 output frames. When `mixer.init`
names a `wipe_cache_store`, `wipe_cache` holds that clip cache's `bytes`,
`budget_bytes` and `clips` (`path`, `frames`, `bytes`, `complete`).
`mixer.scenes <name>` lists registered scenes.

The alpha media path decodes the wipe in software and uploads it to the mixer
CUDA device. This is separate from the GPU-native program video path.

## AUX bus follower

Every AUX bus (`demos/mixer/docs/config.md`, `aux_buses`) has a
`mixer_pvw_follow` node, `aux_<id>_pvw`, the only writer of its compositor's
`composition`. The control side (`pyplumber/mixer/aux.py`) turns the bus's
layout, a list of cells by role (`pyplumber/mixer/aux_layout.py`), into the
node's `layout` object: every scene's layers in the `pvw` cells (empty when
there are none) and the base, everything else, with a revision. The node sets
pvw[shown] followed by the base with that revision; the compositor reports the
revision it draws (`composition_revision`), and the bus is
`composition_pending` until it is the bus's own. A slot assignment, a layout
switch and a page turn are each a new layout object; one Python thread for all
buses resends the layout to a follower that restarted with the one it was built
with. The timing of the PVW cells is
`src/mixer/primitives/PreviewFollow.hpp`.

Every preview change is one `MixerState::PreviewChange` (revision, the pts of
the first program frame of the new program, the take command's receipt and
kind), published under the feed's own `preview_mutex`, which wakes the
`preview_followers`. A cut or fade publishes it right after the selector
switch, before its routing; `mixer.status` still reports the swap and the ended
transition together, since the take holds the mixer's `mutex` throughout. A
change therefore reaches the bus compositor without waiting on that `mutex`.
While the node is unreachable (its group still starting) a layout object sent
to it is dropped and the bus keeps its previous composition until the resend.

The PGM pad carries the program frames with the selector's pts. The bus draws
the frame stamped with main tick K on aux tick `aux.nearestIndex(pts)` +
`pgm_delay_frames`, at that tick's deadline (aux time plus the bus latency).
The compositor swaps a pending composition at the top of every iteration,
before it prepares a tick, so a composition set inside
(deadline(N−1), deadline(N)] is first drawn on tick N; the follower sets it
half an aux tick before deadline(N). For a take whose first new program frame
is K, the bus's `pvw_align` picks N:

- `program` (the default): the first aux tick whose deadline is at or after
  the program frame's own, main.time(K) + main latency, the instant that frame
  leaves the main compositor. The PVW cells then change with the program if
  that frame sits on an aux tick (every K at equal rates and latencies, an even
  K at 60 → 30), and half an aux tick later otherwise (16.7 ms at 60 → 30, 20
  at 50 → 25). The PGM cells of the same bus frame trail the program by
  `pgm_delay_frames` aux ticks (plus the latency difference), so the PVW cells
  lead them.
- `pgm_tile`: the tick whose PGM cells show frame K, so the PVW and PGM cells
  change together, `pgm_delay_frames` later than the program (33 ms at 30 aux
  fps with the default 1).

The error, in aux frames, for a cut or fade is 0 nominally, and:

- +1 when the follower is woken more than half an aux tick late (16 ms at 30
  aux fps, 20 ms at 25) or the compositor's render thread is that late to tick
  N; −1 when the render thread draws tick N−1 more than half a tick late (the
  set lands in that iteration).
- ±1 main tick of the change's own pts when the first new program frame is not
  one main tick after the selector's last output (a missed main deadline before
  the cut, or a frame emitted between the selector switch and the read that
  follows it at once): half an aux tick at 50/60 fps, a whole one at 25/30.
- In `program` mode, +1 when the set misses a tick whose deadline equals the
  program frame's. The change is published at a random phase between the
  emission of frames K−1 and K, so the follower has what is left of one main
  tick (0 to 16.7 ms at 60 fps) minus the main compositor's render time, its
  own wake and the composition set. The host's `cut_spam.py` `late` count
  measures how often; no fraction is claimed.
- A fade's change is published from a frame already presented, so such a tick
  has passed by construction (`target_unreachable`) and the change lands on
  the next: a tick after the program, not a miss.
- The timed composition only drops inputs (the previewed scene's are active;
  the program scene's stay warm for the swap), so the compositor applies it at
  once. When it does add one the bus was not receiving (a source that stalled,
  or takes faster than the warm-up settle, about one aux tick), the compositor
  stages it until that input has a frame for the tick, and past its staging
  deadline (max(250 ms, 2× latency)) keeps the previous layout: the change is
  dropped, not late, until the next preview change.

A PGM frame that misses the aux deadline moves the PGM cells, not the PVW cells.
Wipes and explicit `mixer.preview` changes are not timed: they draw on the next
tick. After a take the program scene's sources keep flowing to the bus, so the
swapped preview is warm: one subscription push per source per aux frame per
bus, no compositing.

Every timed change is measured from the take command's receipt:
`pvw_latency_ms` to the deadline of the bus frame the change is first
drawn on, `pgm_latency_ms` to the program frame's deadline at the main
compositor, and `pvw_minus_pgm_ms` their difference (0 aligned; one aux frame
when the change missed its tick, `last_target_error_ticks` 1, or when its tick
had passed at publish, `target_unreachable`), with the take's `kind` (`cut`,
`fade`; a fade's latencies include the fade). Both end at compositor
deadlines, before the encoders; the cut probe's `mixer.status` `cut_latency`
ends at the program encoder's output, so it exceeds a cut's `pgm_latency_ms`
by the encoder's share. `mixer.aux_status` reports them under `follower` with
`align`, `last_change_to_apply_ms`, `last_target_error_ticks`, the
`layout_revision` it holds and `error`; `mixer.status` `pvw_latency` carries every follower's
last timed change keyed by its node name, for a script polling the status
alone (`demos/mixer/tests/cut_spam.py` reports them per kind, `late` counting
reachable misses only).

## Zero-copy contract

The generic demo's production video-frame path stays in CUDA memory from
hardware decode through normalization, routing, geometry, composition,
transition, and NVENC. It must not contain `hwdownload`, `hwupload`, or
`hwupload_cuda` filters. Packet edges before decode and after encode, RTP/RTCP,
muxing, and control messages are not video-frame memory paths.

The demo uses these CUDA operations:

- optional `scale_cuda` and `pad_cuda` for homogeneous catalogue inputs;
- `mixer_compositor` for per-layer scaling and scene composition;
- `cuda_transform` for a rendition whose size differs from the canvas: one node per
  feed, one draw per distinct size, before the rendition's color conversion;
- `transition_cuda` for fades and dips;
- NVDEC and NVENC at the graph boundaries.

## Generic demo

`demos/mixer/mixer.py` accepts a repeated `--input` option. File paths and URLs
are runtime configuration and are never embedded in the repository. File
inputs are paced in realtime and can be looped with `--loop-inputs`.

The fixed portrait canvas is 1080x1920. The scene set contains one fullscreen
scene per input plus paged 2-, 4-, 8-, and 16-box layouts. Manual control is
available through `demos/mixer/tui.py`; output can be a video-only recording or
a video-only Janus RTP mountpoint.
