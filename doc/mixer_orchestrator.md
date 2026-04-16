# Video Mixer / Scene Switcher

The mixer is a 2-slot PGM/PVW video production switcher built entirely from avplumber nodes, coordinated by the `MixerOrchestrator` class. Multiple scenes can be defined, but only two compositors exist. The orchestrator reconfigures the idle slot before each transition, so GPU work is proportional to visible content only.

Source files:

- `src/SharedTimeline.hpp` -- shared timeline + `TimelineReader` mixin
- `src/MixerState.hpp` -- mixer state (instance-shared)
- `src/mixer_orchestrator.hpp` / `src/mixer_orchestrator.cpp` -- orchestrator
- `src/nodes/one_to_many.cpp` -- 1-to-N output bitmask node
- `src/nodes/source_switcher.cpp` -- N-to-1 input selector node
- `src/nodes/hwaccel/cuda_rect_overlay.cpp` -- compositor (`active_inputs` extension)
- `examples/mixer.avplumber` -- full working example

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
  source_switcher -> one_to_many(final) -> [direct] \
                                                     > source_switcher(wipe_sel) -> force_fps -> enc -> mux -> output
  [wipe_in] + wipe_chain -> overlay_many_cuda ------'
```

Nodes labeled "dynamic" are created and destroyed by the orchestrator. Everything else is static.

### GPU-saving mechanisms

1. **`one_to_many` output bitmask** (per camera): in steady state only the PGM slot bit is set. Cameras not used by a scene have that bit cleared -- their crop/scale chain receives no frames, no GPU cycles. After each PVW slot load, `MixerOrchestrator` rewrites **all** cameras' bitmasks for that slot so `outputs` never routes a source into a compositor input that `active_inputs` would skip (avoids queue growth and needless `scale_cuda` work).

2. **`cuda_rect_overlay` `active_inputs` bitmask** (per compositor): tells the compositor which inputs to wait for. Inactive inputs are skipped in the sync loop entirely (no peek, no wait, no blit).

3. **`transition_cuda` does not exist in steady state.** The orchestrator creates it dynamically when a crossfade starts and deletes it after.

4. **Post-compositor `one_to_many`** routes scene output to direct path only in steady state. The transition path bit is set only during crossfade.

### Steady-state cost

- N active crop/scale filters (only cameras in PGM scene)
- 1 compositor kernel
- 0 transition filter
- PVW compositor, its crop/scale chains, and `transition_cuda` are all idle (no frames flowing, no GPU work)

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

`gc(before_pts_ms)` removes entries older than the threshold but keeps the latest entry before it (the "current" value). Entries at or after the threshold are untouched.

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

The channel is automatically set to the node's `name` parameter. If no timeline entry exists, the fallback (typically an internal atomic) is returned. This means `node.object.set` still works as an immediate override.

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

- `layers` can be updated at runtime via `node.object.set comp_a layers [...]`. The new layers array is parsed and swapped under a mutex.

Both are controllable via `node.object.set` (immediate) or `SharedTimeline` (PTS-synchronized).

## MixerState

`MixerState` (`src/MixerState.hpp`) is an instance-shared object holding all bookkeeping:

- **Source registry**: maps logical camera name to `{otm_node_name, input_index, cs_node_a, cs_node_b}`
- **Scene registry**: maps scene name to `SceneDefinition {sources, layers, width, height}`
- **Slot tracking**: `pgm_is_slot_a`, `pgm_scene_name`, `pvw_scene_name`
- **Transition state**: atomic `TransitionMode` enum: `Idle`, `Cut`, `Crossfade`, `Wipe`
- **Node name registry**: slot A/B compositor, norm, post-otm names; source_switcher name; wipe path names

Helper methods: `pgmSlot()`, `pvwSlot()`, `pgmOutputBit()`, `pvwOutputBit()`, `computeActiveInputsMask(scene)`.

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

`mixer.cut mixer scene_name`

All graph mutations and timeline writes happen synchronously in the calling thread. Only the internal state flip is deferred.

1. `ensureIdle()` -- reject if transition already in progress
2. Set `transition_mode = Cut`
3. `loadSceneIntoSlot(pvw_slot, scene_name)`:
   - For each camera in scene: `node.param.set` crop/scale chain graph + `node.auto_restart` (PVW slot prep; not frame-critical to PGM air)
   - `node.object.set` compositor layers and `active_inputs`
   - Publish compositor `active_inputs` and **every** camera `one_to_many` `outputs` for the PVW slot bit to both the node atomics and the mixer timeline: clear prior `outputs` / `active_inputs` entries for those channels (so old `T_cleanup` values cannot win over `node.object.set`), then write current wallclock entries. Nodes that use `TimelineReader` for these keys would otherwise keep applying stale scheduled masks after a previous transition.
4. Write timeline at `T_cut = wallclock + 200ms`: source_switcher `active` = PVW direct index
5. Write timeline at `T_cleanup = T_cut + 100ms`: camera otm `outputs` converge to PVW-only; old compositor `active_inputs` = 0
6. Spawn detached thread: sleep `T_cleanup - now + 300ms`, then flip `pgm_is_slot_a`, set `pgm_scene_name`, clear `pvw_scene_name`, set `transition_mode = Idle`

The 200ms margin ensures all timeline entries are written before any node processes a frame at `T_cut`. The deferred cleanup thread exists because node deletion and internal bookkeeping flip cannot be expressed as timeline entries.

### Crossfade flow

`mixer.fade mixer scene_name duration_sec`

1-3. Same as cut through `loadSceneIntoSlot` (including camera `outputs` rewrite for the PVW slot).

4. Create `transition_cuda` dynamically:
   ```
   filter_video name=mixer_transition src=["scA_trans","scB_trans"] dst=trans_out
     graph="transition_cuda=alpha='clip(n/FRAMES,0,1)':eval=frame"
   ```
   The `n` variable starts at 0 when the filter is created, producing a 0-to-1 ramp over `FRAMES = round(duration_sec * fps)` frames.

   When PVW is slot A (meaning slot A is the *incoming* scene), the expression is `clip(1-n/FRAMES,0,1)` so that the blend goes from showing slot B (the outgoing PGM) to showing slot A (the incoming scene). The src order is always `["scA_trans","scB_trans"]`; only the alpha expression direction changes.

5. Write timeline at `T_prep = wallclock` (same command time): both post-scene otm `outputs` = `0b11` (direct + trans) to prime queues.

6. Write timeline at `T_start = wallclock + 200ms`:
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

Between `T_prep` and `T_start`, `transition_cuda` warms up but `source_switcher` still reads `active=0` (old PGM direct). The first visible blend frame has alpha near 0, visually indistinguishable from the outgoing scene.

### Media wipe flow

`mixer.wipe mixer scene_name wipe_file [duration_sec]`

The wipe video has an alpha channel that goes transparent -> opaque -> transparent. The fully opaque midpoint covers an invisible hard cut.

If `duration_sec` is omitted, the command probes the wipe file with `avformat_open_input` and reads its duration from container metadata.

The output path uses two static nodes (`otm_final` and `wipe_sel`) that exist in the graph but are normally bypassed (`otm_final` outputs=1 direct only, `wipe_sel` active=0 direct).

1. Create wipe playback chain (all dynamically):
   ```
   input_rec(url=wipe_file, loop=false) -> demux(v:0) -> dec_video(?cuda) -> filter_video(format=yuva420p,hwupload_cuda) -> realtime(set_pts=true)
   ```

2. Create overlay node: `filter_video` with `graph=overlay_many_cuda`, `src=["final_wipe_in","wipe_rt_out"]`, `dst=wipe_overlay_out`

3. Route output through wipe: timeline entries at current wallclock — `otm_final` `outputs` = 3 (both), `wipe_sel` `active` = 1 (overlay)

4. Spawn background thread with two phases:
   - **Midpoint** (duration/2): load target scene into PVW slot (immediate prep + camera `outputs` rewrite as in a cut), then timeline `source_switcher` `active` = PVW direct at current wallclock. Invisible because the wipe fully covers the screen.
   - **End** (duration + 500ms buffer): timeline `otm_final` / `wipe_sel` / camera otms / old compositor `active_inputs`, then delete all 6 wipe chain nodes, flip internal state.

## Timestamp flow

1. Camera inputs: arbitrary PTS, arbitrary timebase
2. `realtime(set_pts=true)`: rewrites PTS to wallclock ms, timebase {1, 1000}
3. `force_fps(30/1)`: rescales to {1, 30}, PTS values are frame counts (0, 1, 2, ...)
4. `one_to_many`: PTS/timebase unchanged
5. `filter_video` (crop+scale): PTS preserved (FFmpeg filter passthrough)
6. `cuda_rect_overlay`: output PTS = min PTS of active inputs, same timebase
7. Post-compositor `force_fps`: cleans jitter, ensures monotonic PTS
8. `transition_cuda` (framesync, only during crossfade): matches frames by PTS
9. `source_switcher`: passes through active input PTS unchanged
10. Final `force_fps`: cleanup for encoder

Timeline entries store `at_pts_ms` in wallclock milliseconds. Frame timebases don't matter because `SharedTimeline::get()` wraps each entry as `av::Timestamp(at_pts_ms, {1,1000})` and uses `av_compare_ts` for comparison.

## Control commands

### Timeline commands (general-purpose)

```timeline.set <name> <channel> <at_pts_ms> <key> <value_json>```

Write a single entry. Example: `timeline.set mixer_tl otm_cam1 1234567 outputs 3`

```timeline.batch <name> <entries_json_array>```

Write multiple entries atomically (single lock). Each entry: `{"ch":"...","at":int64,"key":"...","val":...}`. Example: `timeline.batch mixer_tl [{"ch":"otm_cam1","at":1234567,"key":"outputs","val":3},{"ch":"out_sel","at":1234567,"key":"active","val":2}]`

```timeline.clear <name> [channel]```

Remove entries. If channel omitted, clears entire timeline.

```timeline.gc <name> <before_pts_ms>```

Garbage-collect entries older than threshold (keeps latest entry before threshold as current value).

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
- `wipe_selector` (string) -- wipe_sel node name
- `slot_a` -- `{"compositor":"comp_a", "norm_ts":"norm_a", "post_otm":"otm_scene_a"}`
- `slot_b` -- same structure for slot B

```mixer.source <mixer_name> <src_name> <otm_node> <input_index> <cs_node_a> <cs_node_b>```

Register a camera source. `input_index` is the camera's position in the compositor's `src` array (0-based). `cs_node_a` and `cs_node_b` are the crop/scale filter_video node names for slot A and B respectively.

```mixer.scene <mixer_name> <scene_name> <json_definition>```

Define a scene composition. Definition fields:
- `sources` (object) -- camera_name -> `{"graph":"crop=...,scale_cuda=..."}` (the `graph` parameter for the crop/scale filter_video)
- `layers` (array) -- `cuda_rect_overlay` layers array (one per compositor input, in src order)
- `width`, `height` (int, optional) -- canvas dimensions (not currently used by orchestrator, reserved)

Cameras not listed in `sources` have their compositor input bit cleared. The `layers` array must have one entry per compositor input (matching the compositor's `src` array length), even for inactive inputs -- those entries are ignored by the compositor when the input is inactive.

```mixer.cut <mixer_name> <scene_name>```

PTS-scheduled hard cut. Always requires a scene name.

```mixer.fade <mixer_name> <scene_name> <duration_sec>```

Crossfade to scene over `duration_sec` seconds.

```mixer.wipe <mixer_name> <scene_name> <wipe_file> [duration_sec]```

Media wipe to scene. If `duration_sec` omitted, probed from wipe file metadata. Wipe file must have an alpha channel (e.g., ProRes 4444, VP9 with alpha).

```mixer.status <mixer_name>```

Returns JSON: `{"pgm_scene":"fullcam1","pvw_scene":"","pgm_slot":"A","transition":"idle"}`

## Example setup

See `examples/mixer.avplumber` for a complete 2-camera setup with fullscreen and PiP scenes. Key points:

- Each camera has `realtime(set_pts=true)` with no shared team -- input timestamps are not guaranteed synchronized
- Crop/scale `filter_video` nodes have `auto_restart: "on"` so `node.param.set` + `node.auto_restart` works
- Compositor `active_inputs=1` (only cam1) for initial fullscreen scene on slot A; `active_inputs=0` for idle slot B
- Camera `one_to_many` `outputs=1` (bit 0 = slot A only) for initial PGM-on-A state
- `mixer.init` declares `initial_pgm_scene: "fullcam1"` so the orchestrator knows what's on PGM before the first transition

Usage after startup (via TCP socket):

```
mixer.cut mixer pip
mixer.fade mixer fullcam1 3
mixer.wipe mixer pip /path/to/wipe.mov
mixer.status mixer
timeline.dump mixer_tl
```

## Constraints and limitations

- **Video only.** Audio mixing (crossfade audio during transitions) is not implemented. Audio should be handled separately.
- **One transition at a time.** All transition commands (`cut`, `fade`, `wipe`) reject with an error if a transition is already in progress.
- **Fixed 2-slot architecture.** You cannot have more than 2 compositors. More scenes can be defined, but only 2 render simultaneously (PGM + PVW during transition).
- **Wipe timing is sleep-based.** The wipe midpoint cut and cleanup are driven by `std::this_thread::sleep_for`, not PTS-synchronized timeline entries. This is acceptable because the wipe video fully covers the screen at the midpoint, hiding any timing imprecision.
- **Deferred cleanup uses detached threads.** The internal state flip and node deletion after transitions happen on detached threads. If avplumber shuts down during a transition, these threads are abandoned.
