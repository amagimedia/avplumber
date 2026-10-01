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
| `src/nodes/hwaccel/cuda_rect_overlay.cpp` | compositor node: scheduling, control, output plumbing |
| `src/nodes/mixer_snapshot.cpp`, `mixer_pvw_follow.cpp` | mixer-owned nodes: the output hold and slot substitution; the AUX multiview's PVW tile, applied to the bus compositor on the frame its PGM tile shows the take (`demos/mixer/docs/config.md`, `aux_buses`) |
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

## Graph

Each source supplies one CUDA video-frame edge. A source can either fan out to
the two mixer slots through `one_to_many`, or use `preheat_video_router` outputs
for a catalogue that exceeds the compositor input limit.

```text
CUDA source frames
  -> source fanout (or catalogue router)
  -> cuda_rect_overlay A / cuda_rect_overlay B
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
so a multiview's PVW tile does not blank between takes). The swapped preview
is only shown: the slot it came from has its sources routed away, so
`mixer.status` distinguishes `pvw_scene`, what the
operator and the AUX multiview see, from `pvw_slot_scene`, what is loaded in
the PVW slot (`""` while cold). A cut reuses the slot only for `pvw_slot_scene`;
any other scene, a swapped preview or `mixer.init`'s `initial_pvw_scene`
(shown only) included, is loaded first, warm when it is in `mixer.prewarm`.
Every preview change is one `MixerState::PreviewChange` (revision, the pts of
the first program frame of the new program, the take command's receipt and
kind), under the feed's own `preview_mutex`, which wakes the
`preview_followers` (`mixer_pvw_follow` nodes; a change reaches the bus
compositor without waiting on the mixer's `mutex`: the follower reads the
compositor's status, which that mutex guards, only between takes). A cut or
fade publishes it right after the selector switch, before
its routing, so a multiview's PVW tile changes on the multiview frame that
leaves its bus when that program frame leaves the mixer (or, per bus, on the
frame whose PGM tile shows the take); `mixer.status` still reports the swap
and the ended transition together, since the take holds `mutex` throughout.
Each follower writes its
last timed change back into `mixer.status` `pvw_latency` (keyed by node
name): `pvw_latency_ms` and `pgm_latency_ms` from the command's receipt to the
multiview's and the program's compositor deadlines, `pvw_minus_pgm_ms`,
`kind` and `target_unreachable`; `cut_latency` ends at the encoder's output
instead (`demos/mixer/docs/config.md`, `aux_buses`).

`mixer.status <name>` returns the current PGM/PVW scene and transition state,
and under `playout` each slot compositor's (`A`, `B`) running `frames`, `repeats`
and `missed_deadlines`, published every 60 output frames. When `mixer.init`
names a `wipe_cache_store`, `wipe_cache` holds that clip cache's `bytes`,
`budget_bytes` and `clips` (`path`, `frames`, `bytes`, `complete`).
`mixer.scenes <name>` lists registered scenes.

The alpha media path decodes the wipe in software and uploads it to the mixer
CUDA device. This is separate from the GPU-native program video path.

## Zero-copy contract

The generic demo's production video-frame path stays in CUDA memory from
hardware decode through normalization, routing, geometry, composition,
transition, and NVENC. It must not contain `hwdownload`, `hwupload`, or
`hwupload_cuda` filters. Packet edges before decode and after encode, RTP/RTCP,
muxing, and control messages are not video-frame memory paths.

The demo uses these CUDA operations:

- optional `scale_cuda` and `pad_cuda` for homogeneous catalogue inputs;
- `cuda_rect_overlay` for per-layer scaling and scene composition;
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
