# Video Mixer / Scene Switcher

The mixer is a 2-slot PGM/PVW video production switcher built entirely from avplumber nodes, coordinated by the `MixerOrchestrator` class. Multiple scenes can be defined, but only two compositors exist. The orchestrator reconfigures the idle slot before each transition, so GPU work is proportional to visible content only.

Source files:

- `src/SharedTimeline.hpp` -- shared timeline + `TimelineReader` mixin
- `src/MixerState.hpp` -- mixer state (instance-shared)
- `src/mixer_orchestrator.hpp` / `src/mixer_orchestrator.cpp` -- orchestrator
- `src/nodes/one_to_many.cpp` -- 1-to-N output bitmask node
- `src/nodes/source_switcher.cpp` -- N-to-1 input selector node
- `src/nodes/hwaccel/cuda_rect_overlay.cpp` -- compositor (`active_inputs` extension)
- `src/nodes/force_fps.cpp` -- fixed-FPS normalizer/reset point used around slot and wipe switches
- `examples/sync_mixer.avplumber` -- current confirmed working mixer example
- `examples/mixer.avplumber` -- smaller two-camera example
- `tools/mixer_tui/mixer_tui.py` -- Textual control app used with the working example

## TL;DR: Signal flow

The mixer graph is built from ordinary avplumber nodes. The orchestrator does not carry frames; it only rewrites node parameters/objects and schedules frame-PTS state changes in `SharedTimeline`.

Required flow:

1. **Each input is decoded to video frames in wallclock timescale.** A source goes through `input`/`demux`/`dec_video`, then `realtime(set_pts=true)` and `force_fps`. This gives every camera a stable PTS grid that timeline-controlled nodes can compare against.

2. **Each source fans out to two slot renderers.** A per-source `one_to_many` sends frames to slot A and/or slot B crop-scale chains. (Its `outputs` bitmask is the first major GPU-saving switch: cameras not needed by the current PGM/PVW scene are not fed downstream.)

3. **Each slot renders a full scene.** For every source-slot combination, a `filter_video` crop-scale node prepares that source's rectangle. One `cuda_rect_overlay` compositor per slot combines active inputs into a scene frame. Its `active_inputs` bitmask tells it exactly which inputs to wait for.

`cuda_rect_overlay` does not scale the sources itself, as it is only able to copy rectangular regions between frames. Scaling needs to be handled before, and that's what `filter_video` is for.

There are only 2 slots - the minimum needed for video crossfade. If more than 2 scenes are defined, the slot which is not currently on program is treated as a scratch space for preparing to subsequent scene switch - called `PREVIEW`/`PVW` in `mixer_tui.py` but not actually rendered. When scene is switched, it becomes the rendered program.

4. **The two slot outputs feed the program selector.** Each compositor is normalized by `force_fps`, then a post-scene `one_to_many` can route it to a direct branch and/or a transition branch. The main `source_switcher` (`out_sel`) selects slot A direct, slot B direct, or the dynamic crossfade output.

5. **Cuts, fades, and wipes are only routing changes.** A cut schedules `out_sel.active` to the incoming direct branch. A fade temporarily creates `transition_cuda`, feeds both slot transition branches, then schedules `out_sel.active` through the transition output and finally to the incoming direct branch. A wipe routes final output through the pre-created wipe branch and performs the hidden cut while the wipe covers the frame.

6. **Final output is normalized and encoded.** `wipe_base_fps`, `otm_final`, and `wipe_sel` produce `final_out`.

In `sync_mixer.avplumber`, an optional HTML overlay layer then either bypasses directly to encode or feeds `html_overlay_filter`; `otm_html_overlay_src` prevents dmabuf frames from entering libavfilter framesync while overlay is disabled. This overlay is totally independent from `MixerOrchestrator` and is not a regular video input for the mixer, so scene changes using mixer commands do not affect it.

Minimum node families needed for the mixer core: per-input `realtime`, `force_fps`, `one_to_many`, per-slot crop-scale `filter_video`, per-slot `cuda_rect_overlay`, post-slot `one_to_many`, main `source_switcher`, final wipe-path `one_to_many`/`source_switcher`, encoder/mux/output. Crossfade additionally needs dynamic `transition_cuda`; media wipe additionally needs the pre-created `mixer_wipe` group.



## Graph topology

```
Input Layer (static, per camera)
  input -> demux -> dec_video -> realtime(set_pts=true) -> force_fps -> one_to_many

Per-input crop/scale chains (one per camera per slot)
  one_to_many -> [cam_a] filter_video(crop+scale) -> compositor A
  one_to_many -> [cam_b] filter_video(crop+scale) -> compositor B

Scene Rendering Layer (fixed 2 slots)
  compositor A -> force_fps -> one_to_many(scA) -> [scA_direct] \
  compositor B -> force_fps -> one_to_many(scB) -> [scB_direct]  > source_switcher
                                                                 /
  [scA_trans] + [scB_trans] -> transition_cuda (dynamic) -------'

Output Layer (static)
  source_switcher(out_sel) -> force_fps(wipe_base_fps) -> one_to_many(otm_final) -> [final_direct] \
                                                                                                     > source_switcher(wipe_sel) -> final_out
  [final_wipe_in] + wipe_chain(mixer_wipe) -> overlay_many_cuda -> [wipe_overlay_out] ---------------'

Optional post-mixer overlay/output layer in `sync_mixer.avplumber`
  final_out -> one_to_many(otm_html_overlay) -> [no_overlay] \
                                                            > source_switcher(overlay_sel) -> enc_input_fps -> enc -> mux -> output
  [pre_overlay] + HTML DMA-BUF source -> one_to_many(otm_html_overlay_src) -> overlay_many_cuda --'
```

Nodes labeled "dynamic" are created and destroyed by the orchestrator during crossfade. The wipe subgraph (`mixer_wipe` group) is declared in the config but is not started with the rest of the graph; it is started/stopped around each wipe transition. The HTML overlay layer in `sync_mixer.avplumber` is outside `MixerOrchestrator` and is controlled directly by the TUI with `node.object.set`.

### GPU-saving mechanisms

1. **`one_to_many` output bitmask** (per camera): in steady state only the PGM slot bit is set. Cameras not used by a scene have that bit cleared -- their crop/scale chain receives no frames, no GPU cycles. After each PVW slot load, `MixerOrchestrator` rewrites **all** cameras' bitmasks for that slot so `outputs` never routes a source into a compositor input that `active_inputs` would skip (avoids queue growth and needless `scale_cuda` work).

2. **`cuda_rect_overlay` `active_inputs` bitmask** (per compositor): tells the compositor which inputs to wait for. Inactive inputs are skipped in the sync loop entirely (no peek, no wait, no blit).

3. **`transition_cuda` does not exist in steady state.** The orchestrator creates it dynamically when a crossfade starts and deletes it after.

4. **Post-compositor `one_to_many`** routes scene output to direct path only in steady state. The transition path bit is set only during crossfade. On hard cuts and wipes, the incoming slot's direct branch is prewarmed before the visible selector switch so the selected path is not one pipeline latency late.

5. **Resettable `force_fps` nodes** are used at discontinuity boundaries. The orchestrator resets the two slot normalizers (`norm_a` / `norm_b`) after hard cuts and wipe midpoints, and resets `wipe_base_fps` before each wipe. It deliberately does not reset the final encoder-side FPS guards in the example output chain.

### Steady-state cost

- N active crop/scale filters (only cameras in PGM scene)
- 1 compositor kernel
- 0 transition filter
- PVW compositor, its crop/scale chains, and `transition_cuda` are all idle (no frames flowing, no GPU work)
- Wipe subgraph (`mixer_wipe` group) is stopped: no threads running, no GPU work. `otm_final` outputs=1, so only `final_direct` is fed; `final_wipe_in` and `wipe_overlay_out` stay empty. `wipe_base_fps` normalizes the selected mixer output before the direct/wipe split, and `wipe_rt_fps` normalizes the wipe media leg so `wipe_overlay` receives matching 30fps grids on both inputs.

## SharedTimeline

`SharedTimeline` (`src/SharedTimeline.hpp`) is an instance-shared current-value store. The orchestrator writes entries like "at wallclock PTS T, key K for channel C becomes value V." Nodes query it every frame using the frame's PTS.

### Data model

```
channel -> key -> [ {at_pts_ms, value}, {at_pts_ms, value}, ... ]  (sorted by at_pts_ms)
```

`channel` is the node name (set by `TimelineReader` from the `name` parameter). `key` is a parameter name like `"outputs"` or `"active"`. `at_pts_ms` is wallclock milliseconds.

### Lookup semantics

`get(channel, key, frame_pts)` returns the value from the latest entry with `at_pts_ms <= frame_pts`. Comparison uses `av_compare_ts`, so a frame with PTS 300 in timebase {1,30} correctly matches an entry at 10000ms. Returns `nullopt` if no entry exists at or before `frame_pts`.

Reads are idempotent. Nothing is consumed. If a node restarts, it immediately reads the correct current state on the next frame.

### Atomicity

`set()` locks per call. `setBatch()` applies multiple entries under a single lock, so readers never see a partially-applied batch.

### Garbage collection

`gc(before_pts_ms)` removes old entries but keeps the latest entry at or before the threshold as the "current" value. Entries after the threshold are untouched.

### TimelineReader mixin

Nodes opt in by taking a `timeline` parameter and mixing in `TimelineReader`:

```cpp
class TimelineReader {
    void initTimeline(NodeCreationInfo& nci);  // call in create()
    T tlGet<T>(key, pts, fallback);            // typed read with default
    optional<Parameters> tlGetRaw(key, pts);   // raw read
    bool hasTimeline();
};
```

The channel is automatically set to the node's `name` parameter. If no timeline entry exists at or before the frame PTS, the fallback (typically an internal atomic) is returned. If a matching timeline entry does exist, it wins over `node.object.set`; this is why the orchestrator clears stale timeline keys before publishing fresh immediate state for camera OTMs and compositor `active_inputs`.

## Node types

### `one_to_many`

1 input, N outputs: anything (auto-detected)

Parameters:

- `outputs` (string or uint32, default `1`) -- output bitmask. Accepts two formats:
  - **String** in dst-list order: `"10"` = dst[0] on, dst[1] off. Character at index i maps to output i.
  - **Integer** as raw bitmask: `1` = bit 0 set = dst[0] on.
- `timeline` (string, optional) -- name of `SharedTimeline` instance-shared object
- `drop` (bool, default false) -- drop frames if output queue is full

Each frame, the node reads the `outputs` key from the timeline (if configured). If no timeline entry exists, it falls back to the internal value set by `node.object.set` or the `outputs` parameter.

Immediate control: `node.object.set otm_cam1 outputs 3`

Runtime read: `node.object.get otm_cam1 outputs`

### `source_switcher`

N inputs, 1 output: anything (auto-detected)

Parameters:

- `active` (int, default `0`) -- which input index to forward
- `timeline` (string, optional) -- name of `SharedTimeline` instance-shared object

Each `process()` call iterates all inputs once: for each input that has a frame, it peeks, forwards to output if `i == active`, and pops. One frame per input per tick, not a drain loop. In the mixer, non-active inputs typically have no data at all because their upstream `one_to_many` bitmask is cleared, so `peek()` returns null and they are skipped at zero cost.

### `cuda_rect_overlay` extensions

The compositor gained two new capabilities:

- `active_inputs` (string or uint32, default `0xFFFFFFFF`) -- bitmask of which inputs to wait for. Accepts the same two formats as `one_to_many`'s `outputs` (string in src-list order like `"110"`, or integer). Bit N corresponds to src[N]. Inactive inputs are skipped in every loop (EOF check, min_ts scan, frame discard, frame matching, consume). `processComposite` already handles `nullptr` entries.

- `layers` can be updated at runtime via `node.object.set comp_a layers [...]`. The new layers array is parsed and swapped under a mutex. Per-frame metadata under `metadata_key` (default `rect_overlay_v1`) can override layer fields for that frame.

`active_inputs` is controllable via `node.object.set` (immediate) or `SharedTimeline` (PTS-synchronized). `layers` are immediate object state plus optional per-frame metadata; they are not read from `SharedTimeline`.

## MixerState

`MixerState` (`src/MixerState.hpp`) is an instance-shared object holding all bookkeeping:

- **Source registry**: maps logical camera name to `{otm_node_name, input_index, cs_node_a, cs_node_b}`
- **Scene registry**: maps scene name to `SceneDefinition {sources, width, height}`
- **Slot tracking**: `pgm_is_slot_a`, `pgm_scene_name`, `pvw_scene_name`
- **Transition state**: atomic `TransitionMode` enum: `Idle`, `Cut`, `Crossfade`, `Wipe`
- **Node name registry**: slot A/B compositor, norm, post-otm names; output selector name; wipe path names (`wipe_otm`, `wipe_base_fps`, `wipe_selector`, `wipe_group`, `wipe_input_node`, `wipe_tail_edge`, `wipe_flush_edges`)

Helper methods: `pgmSlot()`, `pvwSlot()`, `pgmSourceSwitcherIndex()`, `pvwSourceSwitcherIndex()`, `pgmOutputBit()`, `pvwOutputBit()`, `computeActiveInputsMask(scene)`.

The `pgmOutputBit()` / `pvwOutputBit()` convention: slot A = bit 0 (value 1), slot B = bit 1 (value 2). So when `pgm_is_slot_a=true`, `pgmOutputBit()=1`, `pvwOutputBit()=2`.

## MixerOrchestrator

The orchestrator (`src/mixer_orchestrator.cpp`) does not process media. It manages graph topology and pushes state into nodes and the timeline.

It is constructed per-command (not long-lived): each control command creates a temporary `MixerOrchestrator` with references to `NodeManager`, `MixerState`, and `SharedTimeline`, calls the relevant method, then discards it.

### Timeline `clearKey` on camera OTMs and compositors

When loading a PVW scene, the orchestrator calls `SharedTimeline::clearKey` for **`outputs` only** on each registered camera `one_to_many` channel, and for **`active_inputs` only** on that slot’s compositor. Other timeline keys on those channels (if any) are left alone; unrelated channels (`out_sel`, `otm_scene_*`, …) are untouched.

That wipe is intentional: any older scheduled `outputs` / `active_inputs` rows for those keys—including not-yet-fired `T_cleanup` from a **previous** transition—would otherwise keep winning over `node.object.set` while frames advance. New `T_cleanup` rows for the **current** cut or fade are always written **after** `loadSceneIntoSlot` returns, so they are not removed by that clear.

Do not rely on independent `timeline.set` on a camera’s `outputs` to survive a later `mixer.cut` / `mixer.fade` / wipe midpoint load: the next PVW load replaces the schedule for that key.

### Immediate `node.object.set` vs `SharedTimeline`

Anything that must line up with decoded **frame PTS** (program path, camera fan-out, post-scene `one_to_many` during a crossfade, wipe routing, `source_switcher` selection) is written as **timeline** entries on the mixer timeline so nodes resolve keys from `SharedTimeline` per frame.

**Immediate** `node.object.set` (and `node.param.set` / `node.auto_restart`) is reserved for work where frame-perfect alignment with transport time is unnecessary. The main case is **preparing the broadcast-inactive PVW slot** before it is shown: crop/scale `graph`, compositor `layers` / `active_inputs`, and the orchestrator's full rewrite of camera `outputs` for that slot. Those updates are safe while no PGM-visible frame should depend on them yet.

### Hard cut flow

`mixer.cut {"mixer":"mixer","scene":"scene_name","start_pts_ms":1234567}`

All graph mutations and timeline writes happen synchronously in the calling thread. Only the internal state flip is deferred.

1. `ensureIdle()` -- reject if transition already in progress
2. Set `transition_mode = Cut`
3. If the requested scene is not already preloaded in PVW, `loadSceneIntoSlot(pvw_slot, scene_name)`:
   - For each camera in scene: update the PVW crop/scale chain graph and restart that filter only if the graph string changed (PVW slot prep; not frame-critical to PGM air)
   - `node.object.set` compositor layers and `active_inputs`
   - Publish compositor `active_inputs` and **every** camera `one_to_many` `outputs` for the PVW slot bit to both the node atomics and the mixer timeline: clear prior `outputs` / `active_inputs` entries for those channels (so old `T_cleanup` values cannot win over `node.object.set`), then write current wallclock entries. Nodes that use `TimelineReader` for these keys would otherwise keep applying stale scheduled masks after a previous transition.
4. Prewarm the incoming direct branch immediately: clear the incoming slot post-OTM `outputs`, set its object value to `1`, and write a timeline entry at current wallclock.
5. Write timeline at `T_cut`: source_switcher `active` = PVW direct index. When `start_pts_ms` is omitted or set to `-1`, this is `wallclock + 200ms`; otherwise it is the requested `start_pts_ms`.
6. Write timeline at `T_cleanup = T_cut + 100ms`: camera OTM `outputs` converge to PVW-only for the new scene; old compositor `active_inputs` = 0; old slot post-OTM `outputs` = 0; new slot post-OTM `outputs` = 1.
7. Reset both slot `force_fps` nodes (`norm_a`, `norm_b`) so an idle slot that has fallen behind does not emit a duplicate burst when selected.
8. Spawn detached thread: sleep `T_cleanup - now + 300ms`, then flip `pgm_is_slot_a`, set `pgm_scene_name`, clear `pvw_scene_name`, set `transition_mode = Idle`.

The 200ms margin ensures all timeline entries are written before any node processes a frame at `T_cut`. The deferred cleanup thread exists because node deletion and internal bookkeeping flip cannot be expressed as timeline entries.

### Crossfade flow

`mixer.fade {"mixer":"mixer","scene":"scene_name","duration_sec":3.0,"start_pts_ms":1234567}`

1-3. Same as cut through `loadSceneIntoSlot` (including camera `outputs` rewrite for the PVW slot).

4. Create `transition_cuda` dynamically:
   ```
   filter_video name=mixer_transition src=["scA_trans","scB_trans"] dst=trans_out
     graph="transition_cuda=alpha='clip(n/FRAMES,0,1)':eval=frame"
   ```
   The `n` variable starts at 0 when the filter is created, producing a 0-to-1 ramp over `FRAMES = round(duration_sec * fps)` frames.

   When PVW is slot A (meaning slot A is the *incoming* scene), the expression is `clip(1-n/FRAMES,0,1)` so that the blend goes from showing slot B (the outgoing PGM) to showing slot A (the incoming scene). The src order is always `["scA_trans","scB_trans"]`; only the alpha expression direction changes.

5. Write timeline at `T_prep = T_start - 200ms`: both post-scene otm `outputs` = `0b11` (direct + trans) to prime queues. For scheduled fades this prevents the `transition_cuda` frame counter from advancing long before the transition is visible.

6. Write timeline at `T_start`:
   - source_switcher `active` = 2 (transition output)
   - both post-scene otm `outputs` = `0b10` (trans only, stop wasting frames on direct)

7. Write timeline at `T_end = T_start + duration_ms`:
   - source_switcher `active` = new PGM direct index
   - new PGM post-scene otm `outputs` = `0b01` (direct only)
   - old slot post-scene otm `outputs` = `0b00` (idle)

8. Write timeline at `T_cleanup = T_end + 100ms`:
   - All camera otm `outputs` converge to new-PGM-only bitmask
   - Old compositor `active_inputs` = 0

9. Spawn detached thread: sleep, delete `mixer_transition` node, flip internal state.

Between `T_prep` and `T_start`, `transition_cuda` warms up but `source_switcher` still reads the current PGM direct input (index 0 when PGM is slot A, index 1 when PGM is slot B). The first visible blend frame has alpha near 0, visually indistinguishable from the outgoing scene.

### Media wipe flow

`mixer.wipe {"mixer":"mixer","scene":"scene_name","wipe_file":"/path/to/wipe.mov","duration_sec":2.0,"start_pts_ms":1234567}`

The wipe video has an alpha channel that goes transparent -> opaque -> transparent. The fully opaque midpoint covers an invisible hard cut.

If `duration_sec` is omitted, the command probes the wipe file with `avformat_open_input` and reads its duration from container metadata.

The output path uses two static nodes (`otm_final` and `wipe_sel`) that exist in the graph but are normally bypassed (`otm_final` outputs=1 direct only, `wipe_sel` active=0 direct).

The wipe subgraph (`mixer_wipe` group) is declared in the config but not started with the main graph. In `sync_mixer.avplumber` it contains:
```
input_rec(url="", loop=false) -> demux(v:0) -> dec_video(?cuda)
  -> filter_video(format=yuva420p,scale=1920:1080,hwupload_cuda)
  -> realtime(set_pts=true) -> force_fps(wipe_rt_fps) -> [wipe_rt_fps_out]

out_sel -> force_fps(wipe_base_fps) -> otm_final -> [final_wipe_in]
[final_wipe_in] + [wipe_rt_fps_out] -> filter_video(overlay_many_cuda) -> [wipe_overlay_out]
```
The `url` is empty in steady state. `otm_final` outputs=1, so `final_direct` is fed and `final_wipe_in` is not. `wipe_base_fps` normalizes the main/direct path before it is split, while `wipe_rt_fps` normalizes the wipe media path; together they give `wipe_overlay` two inputs on the same 30fps PTS grid. The wipe overlay graph converts the NV12 main input to YUV420P, overlays the alpha wipe, then converts back to NV12.

1. Reset the pre-created `wipe_input` node object (via `stop(true)`) so that `createNode()` will pick up the updated `url` when the group starts.

2. Set `wipe_input` `url` parameter to `wipe_file`, flush configured `wipe_flush_edges`, reset `wipe_base_fps`, then start the `mixer_wipe` group. For scheduled wipes this prep is delayed until `T_start - 200ms` so the wipe media begins near the requested visible start.

3. Route output through wipe: timeline entries at `T_prep` / `T_start` -- `otm_final` `outputs` = 3 (both direct and wipe input), `wipe_sel` `active` = 1 (overlay).

4. Spawn background thread with two phases:
   - **Midpoint** (`T_start + duration/2`): load target scene into PVW slot (immediate prep + camera `outputs` rewrite as in a cut), set the new slot post-OTM to direct-only, set the old slot post-OTM to idle, then timeline `source_switcher` `active` = PVW direct at current wallclock. This is hidden by the opaque midpoint of the wipe.
   - **End**: wait until the planned wipe duration or until `wipe_input` reaches EOF. If EOF arrives early and `wipe_tail_edge` is configured, wait for that edge to drain (up to 1000ms) plus a short overlay tail grace (120ms). Then switch `otm_final` back to direct-only, switch `wipe_sel` back to direct, converge camera OTMs, and set old compositor `active_inputs` = 0. After an additional 250ms switch grace, stop `mixer_wipe`, flush `wipe_flush_edges`, and flip internal state.

## Timestamp flow

1. Camera inputs: arbitrary PTS, arbitrary timebase
2. `realtime(set_pts=true)`: rewrites PTS to wallclock ms, timebase {1, 1000}
3. `force_fps(30/1)`: rescales to {1, 30}; PTS values are frame counts on that grid
4. `one_to_many`: PTS/timebase unchanged
5. `filter_video` (crop+scale): PTS preserved (FFmpeg filter passthrough)
6. `cuda_rect_overlay`: output PTS = min PTS of active inputs, same timebase
7. Post-compositor `force_fps`: cleans jitter, ensures monotonic PTS
8. `transition_cuda` (framesync, only during crossfade): matches frames by PTS
9. `source_switcher`: passes through active input PTS unchanged
10. `wipe_base_fps`, optional overlay/output `force_fps` nodes, and encoder-side normalizers: keep selector switches and the encoder input on a stable frame grid

Timeline entries store `at_pts_ms` in wallclock milliseconds. Frame timebases don't matter because `SharedTimeline::get()` wraps each entry as `av::Timestamp(at_pts_ms, {1,1000})` and uses `av_compare_ts` for comparison.

## Control commands

### Timeline commands (general-purpose)

```timeline.set <json_object>```

Write a single entry. Fields: `name` (timeline instance), `ch` or `channel`, `at` (wallclock ms), `key`, and `val`.

Example: `timeline.set {"name":"mixer_tl","ch":"otm_cam1","at":1234567,"key":"outputs","val":3}`

```timeline.batch <name> <entries_json_array>```

Write multiple entries atomically (single lock). Each entry: `{"ch":"...","at":int64,"key":"...","val":...}`. Example: `timeline.batch mixer_tl [{"ch":"otm_cam1","at":1234567,"key":"outputs","val":3},{"ch":"out_sel","at":1234567,"key":"active","val":2}]`

```timeline.clear <name> [channel]```

Remove entries. If channel omitted, clears entire timeline.

```timeline.gc <name> <before_pts_ms>```

Garbage-collect old entries (keeps the latest entry at or before the threshold as current value).

```timeline.dump <name>```

Print all entries as JSON for debugging.

### Mixer commands

```mixer.init <mixer_name> <json_config>```

Initialize `MixerState` with node names and global settings. Must be called before other mixer commands.

Config fields:
- `timeline` (string) -- SharedTimeline instance name
- `hwaccel` (string) -- hwaccel device name
- `fps_num`, `fps_den` (int) -- output frame rate for frame count calculations
- `source_switcher` (string) -- source_switcher node name
- `initial_pgm_slot` (string, "A" or "B") -- which slot starts as PGM
- `initial_pgm_scene` (string) -- scene name initially on PGM
- `initial_pvw_scene` (string, optional)
- `wipe_otm` (string) -- otm_final node name (for wipe output path)
- `wipe_base_fps` (string, optional) -- name of the `force_fps` node that feeds the wipe main input (e.g. `wipe_base_fps`). The orchestrator resets it at wipe start so the wipe path does not replay a stale PTS grid after being idle.
- `wipe_selector` (string) -- wipe_sel node name
- `wipe_group` (string) -- name of the pre-created wipe subgraph group (started/stopped per wipe)
- `wipe_input_node` (string) -- name of the `input_rec` node inside the wipe group (its `url` is set before each wipe start)
- `wipe_tail_edge` (string, optional) -- edge feeding the wipe overlay input; when set, wipe cleanup waits for it to drain after wipe input EOF
- `wipe_flush_edges` (array of strings, optional) -- wipe pipeline edges to clear before each wipe and after stopping the wipe group
- `slot_a` -- `{"compositor":"comp_a", "norm_ts":"norm_a", "post_otm":"otm_scene_a"}`
- `slot_b` -- same structure for slot B

```mixer.source <mixer_name> <src_name> <otm_node> <input_index> <cs_node_a> <cs_node_b>```

Register a camera source. `input_index` is the camera's position in the compositor's `src` array (0-based). `cs_node_a` and `cs_node_b` are the crop/scale filter_video node names for slot A and B respectively.

```mixer.scene <mixer_name> <scene_name> <json_definition>```

Define a scene composition. Definition fields:
- `sources` (object) -- logical source name (same strings as `mixer.source`) -> object containing:
  - `graph` (string) -- FFmpeg filter chain for that camera's crop/scale `filter_video` (`node.param.set` / `auto_restart`).
  - Any other keys (`dst_x`, `dst_y`, …) -- passed as that compositor source's `cuda_rect_overlay` layer entry (same names as in `node.object.set comp_* layers`).
- `width`, `height` (int, optional) -- canvas dimensions (not currently used by orchestrator, reserved)

Sources not listed in `sources` are off this scene: their `one_to_many` slot bit is cleared and the orchestrator fills their layer slot with `{"dst_x":0,"dst_y":0}` when building the compositor `layers` array. The array length matches registered `mixer.source` input indices (one layer per compositor `src` order); inactive compositor inputs ignore their layer at runtime.

```mixer.cut <json_object>```

Hard cut. Required fields: `mixer` (string), `scene` (string). Optional fields: `start_pts_ms` (int64, default `-1` = now + prep margin).

Example: `mixer.cut {"mixer":"mixer","scene":"multiviewer"}`

```mixer.preview <json_object>```

Preload a scene into the hidden PVW slot without taking it to program. Required fields: `mixer` (string), `scene` (string). The next `mixer.cut` or `mixer.fade` to the same scene reuses the warmed slot instead of flushing/reloading it; a cut to that scene switches at its scheduled PTS instead of waiting for a new post-cut readiness frame.

Example: `mixer.preview {"mixer":"mixer","scene":"multiviewer"}`

```mixer.fade <json_object>```

Crossfade to scene. Required fields: `mixer` (string), `scene` (string). Optional fields: `duration_sec` (number, default `1.0`), `start_pts_ms` (int64, default `-1` = now + prep margin).

Example: `mixer.fade {"mixer":"mixer","scene":"synctest","duration_sec":3.0}`

```mixer.wipe <json_object>```

Media wipe to scene. Required fields: `mixer` (string), `scene` (string), `wipe_file` (string). Optional fields: `duration_sec` (number; if omitted, probed from wipe file metadata), `start_pts_ms` (int64, default `-1` = now + prep margin). Wipe file must have an alpha channel (e.g., ProRes 4444, VP9 with alpha).

Example: `mixer.wipe {"mixer":"mixer","scene":"fullsync1","wipe_file":"/path/with spaces/wipe.mov"}`

For all transition commands, `start_pts_ms` is wallclock milliseconds in the same `{1,1000}` domain used by `SharedTimeline`. If provided, it must be at least the mixer prep margin (currently 200ms) in the future. While a future transition is armed, the mixer is considered busy and later transition commands are rejected until cleanup completes.

```mixer.status <mixer_name>```

Returns JSON: `{"pgm_scene":"fullsync1","pvw_scene":"","pgm_slot":"A","transition":"idle"}`

```mixer.scenes <mixer_name>```

Returns a JSON array of all registered scene names, sorted alphabetically. Example: `["fullmp4_1","fullmp4_2","fullsync1","fullsync2","fullsync3","multiviewer","synctest"]`

## Control app

`tools/mixer_tui/mixer_tui.py` is the current control app used with `examples/sync_mixer.avplumber`.

- It connects to the avplumber TCP control port and fetches scene names from `mixer.scenes <mixer_name>` on connect, then refreshes the list roughly every 10 seconds.
- It polls `mixer.status <mixer_name>` every 500ms and treats any non-`idle` transition as busy; cut/fade/wipe commands are not sent while busy.
- Selecting a scene with `1`-`9` or a scene button sends `mixer.preview` so avplumber preloads the hidden PVW slot. Pressing `c`, `x`, or `w` sends `mixer.cut`, `mixer.fade`, or `mixer.wipe` for the selected scene.
- After a take completes, the UI polls until `transition == "idle"` and then puts the previous PGM scene on the local PVW bus to mimic a production switcher bus swap.
- `F1`-`F9` sends a direct `mixer.cut` to that scene, skipping the local preview selection.
- The HTML overlay buttons are outside `MixerOrchestrator`. Enabling first opens the HTML-source gate (`otm_html_overlay_src outputs 1`) and prewarms both final-output legs (`otm_html_overlay outputs 3`), then selects `overlay_sel active 1` and settles on `otm_html_overlay outputs 2`. Disabling prewarms the direct leg, selects `overlay_sel active 0`, settles on `otm_html_overlay outputs 1`, and closes the HTML-source gate (`otm_html_overlay_src outputs 0`). Closing the source gate drains and drops dmabuf frames before they reach `html_overlay_filter`, avoiding libavfilter framesync buffering while bypassed.

## Example setup

See `examples/sync_mixer.avplumber` for the current confirmed working setup. It contains three synchronized SRT sources, two local MP4 loop sources, the mixer core, a media wipe path, an optional HTML DMA-BUF overlay path, and the encoder/output chain. `examples/mixer.avplumber` is a smaller two-camera variant.

- The three SRT sources use `realtime(set_pts=true, team="sync_team")`, so their outputs are wall-clock aligned as a group. The MP4 loops use independent `realtime(set_pts=true)`.
- Crop/scale `filter_video` nodes have `auto_restart: "on"` so `node.param.set` + `node.auto_restart` works
- Compositor `active_inputs=1` (only `sync1`) for initial fullscreen scene on slot A; `active_inputs=0` for idle slot B
- Camera `one_to_many` `outputs=1` (bit 0 = slot A only) for initial PGM-on-A state
- `mixer.init` declares `initial_pgm_scene: "fullsync1"` so the orchestrator knows what's on PGM before the first transition
- `wipe_base_fps`, `wipe_rt_fps`, `wipe_tail_edge`, and `wipe_flush_edges` are part of the working wipe setup.
- The optional HTML overlay path is controlled by `node.object.set` on `otm_html_overlay`, `otm_html_overlay_src`, and `overlay_sel`, not by mixer commands.

Usage after startup (via TCP socket):

```
mixer.cut {"mixer":"mixer","scene":"multiviewer"}
mixer.fade {"mixer":"mixer","scene":"synctest","duration_sec":3}
mixer.wipe {"mixer":"mixer","scene":"fullsync1","wipe_file":"/path/to/wipe.mov"}
mixer.status mixer
timeline.dump mixer_tl
```

## Constraints and limitations

- **Video only.** Audio mixing (crossfade audio during transitions) is not implemented. Audio should be handled separately.
- **One transition at a time.** All transition commands (`cut`, `fade`, `wipe`) reject with an error if a transition is already in progress or already armed for the future.
- **Fixed 2-slot architecture.** You cannot have more than 2 compositors. More scenes can be defined, but only 2 render simultaneously (PGM + PVW during transition).
- **Wipe timing is partly sleep-based.** Wipe group prep, midpoint switch, cleanup, EOF polling, tail-drain polling, and group stop are driven by a detached thread and wallclock sleeps/polls, not purely by PTS-synchronized timeline entries. This is acceptable because the wipe video covers the midpoint switch; the cleanup path also waits for the wipe tail to drain when configured.
- **Deferred cleanup uses detached threads.** The internal state flip and group stop after transitions happen on detached threads. If avplumber shuts down during a transition, these threads are abandoned.
- **Wipe subgraph must be declared.** The `mixer_wipe` group must be present in the avplumber config; `mixer.wipe` will not create that subgraph dynamically. In the current examples it has 7 nodes: `wipe_input`, `wipe_demux`, `wipe_dec`, `wipe_fmt`, `wipe_rt`, `wipe_rt_fps`, and `wipe_overlay`. The group must not be started by the config because the orchestrator manages its lifecycle.
