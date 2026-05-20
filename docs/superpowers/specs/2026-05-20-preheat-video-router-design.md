# Slot-Aware Preheat Video Router Design

Date: 2026-05-20
Status: approved design, pending implementation plan

## Context

The auto mixer currently keeps all talkshow scene geometry hot by creating one
preheated source per geometry slot. For six cameras the preheated geometry set is:

- `face_full` x 1
- `face_square` x 1
- `face_conf_thumb` x 5
- `orig_stack` x 3
- `orig_pip_thumb` x 1

That is 11 logical hot geometry slots. The current Python graph generation in
`pyplumber/auto_mixer/preheated.py` fans every camera into every hot slot:

```text
face_i/orig_i
  -> split per camera
  -> one queue per camera x hot geometry slot
  -> source_switcher per hot geometry slot
  -> mixer source one_to_many
  -> filter_video A/B
  -> compositor A/B
```

For six cameras, this creates 66 camera-to-hot-slot queues before the selected
hot sources. Including the selected-source edges and per-slot mixer source
queues, there are 99 pre-filter routing queues. The routing/scaling portion has
roughly 56 nodes: 12 splits, 11 source switchers, 11 source OTMs, and 22
`filter_video` nodes.

The optimization must not remove scene families. Full-face, videoconference,
PiP, vstack2, and vstack3 geometry must all remain available.

## Goals

- Keep the full existing scene vocabulary and geometry coverage.
- Replace the complete `camera x geometry-slot` fanout with a small native
  router layer.
- Preserve correct `cut`, `fade`, and `wipe` behavior.
- Leave room for a future move transition where one camera can feed multiple
  geometry slots during an animation.
- Keep the non-preheated fallback path available for debugging and rollback.
- Improve graph scale for six cameras from 99 pre-filter routing queues to 22
  router-to-filter queues.

## Non-Goals

- Do not fuse routing with crop/scale work. Existing `filter_video` nodes remain
  responsible for geometry.
- Do not make a generic ATD router yet. The first node is video-only:
  `av::VideoFrame` input and output.
- Do not remove PiP, vstack, full-face, or videoconference scenes.
- Do not require a third geometry copy beyond the existing mixer A/B slot model.
- Do not change `--disable-preheated-scenes` behavior.

## Core Invariant: Slot-Aware Outputs

The router must expose one output per `(geometry slot, mixer slot)` pair, not
one global output per geometry slot.

Global geometry routing is wrong for transitions. During a fade, slot A may
still render the old PGM scene while slot B renders the new PVW scene. If both
scenes use `face_square_0`, the outgoing scene may need camera 0 while the
incoming scene needs camera 1. A single global `face_square_0` route cannot
represent both at the same time.

The routed outputs for six cameras are:

```text
face router:
  face_full_A, face_full_B
  face_square_A, face_square_B
  face_conf_thumb_0_A, face_conf_thumb_0_B
  face_conf_thumb_1_A, face_conf_thumb_1_B
  face_conf_thumb_2_A, face_conf_thumb_2_B
  face_conf_thumb_3_A, face_conf_thumb_3_B
  face_conf_thumb_4_A, face_conf_thumb_4_B

orig router:
  orig_stack_0_A, orig_stack_0_B
  orig_stack_1_A, orig_stack_1_B
  orig_stack_2_A, orig_stack_2_B
  orig_pip_thumb_A, orig_pip_thumb_B
```

This gives 14 face outputs and 8 original-feed outputs. The 22 scaling filters
remain hot, but they are fed directly by two routers instead of by 66 fanout
queues, 11 source switchers, and 11 preheated source OTMs.

## Native Router Node

Add a native C++ node, tentatively named `preheat_video_router`.

The node shape is:

```text
preheat_video_router
  src: [camera_0_edge, camera_1_edge, ...]
  dst: [hot_slot_0_A, hot_slot_0_B, hot_slot_1_A, hot_slot_1_B, ...]
  routes: output_index -> input_index or -1
```

The node is video-only in the first implementation. It should register as a
concrete `av::VideoFrame` node, not through ATD type detection.

The router owns only routing. It does not crop, scale, overlay, synchronize, or
generate synthetic frames.

## Router Runtime Semantics

The router continuously drains all inputs. For whichever input has the earliest
available frame:

1. Read the frame PTS.
2. Resolve the route table active at that PTS.
3. Copy the frame to every output whose route equals that input index.
4. Drop outputs routed to `-1`.
5. Drop on full output queues instead of blocking.
6. Pop the input frame.

There is no cross-camera synchronization inside the router. The mixer and
compositor paths remain responsible for downstream timing.

The router supports duplicate routes. Multiple outputs may select the same
input at the same timestamp. This is required for future move transitions and
also happens naturally when old and new slots route the same camera.

Each output enforces monotonic PTS independently. If a route changes from one
camera to another and the candidate frame would move that output backward in
PTS, the router drops that frame for that output. It does not block or rewrite
timestamps.

EOF handling is local. If an input emits EOF, the router propagates EOF only to
outputs routed to that input at that frame PTS, pops that input, and does not
stop the whole router unless all inputs are finished. Input group restart policy
remains outside the router.

## Timeline and Batching

Routes must be timeline-driven, not only immediate `setObject()` mutations.
The mixer already schedules source outputs, compositor activity, and output
selector state by PTS. Router routes need the same timing model to avoid
camera leaks during transition warmup.

The router should support an immediate route table for startup and debugging,
and timeline snapshots for scheduled changes. A snapshot sets the complete
route table for a router at a PTS, avoiding partial route updates.

The high-level mixer path should batch route updates per router and per PTS.
Conceptually:

```json
{
  "node": "face_preheat_router",
  "at_pts_ms": 46357,
  "routes": [-1, 1, -1, 0, 2, -1]
}
```

The C++ representation can use `SharedTimeline::setBatch()` and a `routes`
key per router node. The router resolves the latest `routes` snapshot at each
frame PTS.

## Mixer Scene Model

Router route plans should be explicit scene data, not opaque generic controls.

Generic `controls` may remain for non-routing side effects, but routing affects
scene identity and transition lifetime. Keeping it inside generic controls is
too easy to schedule at the wrong transition phase.

Extend the mixer scene model with explicit route requirements. A route
requirement maps a routed mixer source to a camera input index:

```text
SceneRoute:
  source_name: hot_face_square_0
  input_index: 1
```

The mixer state maps each routed source to:

```text
router node name
route output index for slot A
route output index for slot B
filter_video node for slot A
filter_video node for slot B
compositor input index
```

The orchestrator applies the route requirements to the target slot by choosing
the correct route output index for A or B.

## Python Graph Generation

`pyplumber/auto_mixer/preheated.py` remains the authority for the geometry
resource set. It should build two routers:

- one face router consuming the per-input face crop edges,
- one orig router consuming the per-input original 16:9 edges.

The preheated source builder should register routed mixer sources rather than
normal mixer sources. A routed source supplies the per-slot pre-filter edges
directly:

```text
hot_face_square_0:
  slot A pre-filter edge: face_square_A
  slot B pre-filter edge: face_square_B
  default graph: crop_cuda + scale_cuda
```

`pyplumber/mixer.py` should add a routed-source path alongside the current
normal source path. For routed sources it creates the two `filter_video` nodes
directly from the router output edges and does not create the per-source OTM.

The non-preheated path continues to use normal mixer sources unchanged.

## Transition Semantics

### Initial Startup

The initial PGM slot receives routes for the initial scene before graph start.
The other slot starts with all routes disabled (`-1`). This avoids warming an
unused default PVW scene at startup.

### Cut

When a cut targets the PVW slot:

1. Apply target-scene routes to the PVW slot at prep time.
2. Configure PVW filters and compositor layers.
3. Wait for the PVW direct edge to produce a fresh frame when needed.
4. Switch the output selector at the cut PTS.
5. Disable old-slot routes during cleanup.

The old PGM slot routes remain unchanged until the visible switch is complete.

### Fade

Fade needs old and new scenes alive simultaneously:

1. Keep old PGM slot routes unchanged for the whole fade.
2. Apply target-scene routes to the PVW slot during fade prep.
3. Render both slot branches during the transition.
4. Switch output selector back to direct new PGM at fade end.
5. Disable old-slot routes during cleanup.

This is why global geometry outputs are disallowed.

### Wipe

Wipe should use the same slot-aware route model:

1. Keep old PGM slot routes unchanged while the old scene is visible under the
   wipe overlay.
2. Apply target-scene routes to the PVW slot during wipe prep.
3. Perform the hidden scene switch at the wipe midpoint.
4. Disable old-slot routes during cleanup.

The implementation should not rely on the wipe being fully opaque at the exact
frame where global routes would change.

### Future Move Transition

The router exposes stable geometry-slot resources. Future move transitions can
route one input into multiple geometry outputs at the same time and animate
layers or filter parameters above the router. No router change should be needed
for that class of transition.

## Rollout

Add a rollout flag before making the router default:

```text
--preheat-routing source-switcher
--preheat-routing router
```

The initial default should remain `source-switcher`. After remote validation,
flip the default to `router`.

## Observability

The router should expose route state and counters through `getObject()` or an
equivalent status command:

- current route per output,
- output labels if provided in params,
- frames drained per input,
- frames enqueued per output,
- frames dropped per output,
- backward-PTS drops per output.

This is required for debugging live queue and route behavior.

## Acceptance Criteria

Functional:

- Six-camera Janus auto mixer runs with router preheating enabled.
- The registered scene families still include full-face, videoconference, PiP,
  vstack2, and vstack3.
- Auto-switch cut behavior still works.

Transition correctness:

- Manual cut, fade, and wipe tests with visibly different cameras do not route
  the outgoing scene to the incoming camera early.
- During fade, old and new slots can render different cameras for the same
  logical geometry family.
- During wipe, old scene routes remain stable until the hidden midpoint switch.

Graph scaling:

- Six-camera preheated routing drops from about 56 routing/scaling nodes to
  about 24: two routers plus 22 `filter_video` nodes.
- Six-camera pre-filter routing queues drop from 99 to 22.

Reliability:

- No native `Node factory returned nullptr` exceptions from preheated geometry
  startup in the router path.
- Hidden preheat routing must not backpressure input decode or AI nodes.
- `--disable-preheated-scenes` and `--preheat-routing source-switcher` remain
  available for comparison and rollback.

## Implementation Touch Points

- `src/nodes/preheat_video_router.cpp` or equivalent new C++ node file.
- Build/node factory registration for `preheat_video_router`.
- `src/MixerState.hpp`: add routed-source metadata and explicit scene route
  requirements.
- `src/mixer_orchestrator.cpp`: apply route plans for initial load, cut, fade,
  wipe, and cleanup.
- `pyplumber/node.py`: add `PreheatVideoRouter`.
- `pyplumber/mixer.py`: add routed-source construction and scene route emission.
- `pyplumber/auto_mixer/preheated.py`: generate routers, routed sources, and
  explicit scene route requirements.
- `pyplumber/auto_mixer/cli.py`: add rollout flag.

## Validation Plan

Remote validation should run on the Fedora GPU host, not locally.

1. Build `python_module` with the CUDA/TensorRT flags from the project remote
   workflow.
2. Run a short six-camera Janus smoke test with `--preheat-routing router`.
3. Query auto-switch status and native exception status while running.
4. Exercise manual `mixer.cut`, `mixer.fade`, and `mixer.wipe` commands.
5. Compare `queues.json` and node count against the current source-switcher
   path.
6. Leave a live Janus preview running only when explicitly requested.
