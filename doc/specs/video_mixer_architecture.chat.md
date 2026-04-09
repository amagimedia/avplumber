# Video mixer development and scene management
_Exported on 4/9/2026 at 17:30:07 GMT+2 from Cursor (3.0.13)_

---

**User**

We're building video mixer (switcher, compositor, transitions, media wipes - like "micro OBS") in avplumber (@avplumber/README.md @avplumber/doc/developing_nodes.md ). We already have most needed nodes (@avplumber/examples/compositor.avplumber ) for making composition scenes, but more glue logic is needed.

We need to switch between scenes, also having fade (video crossfade) and media wipe (playing overlay with alpha channel and switching scenes exactly in the middle of this video). Fade is already implemented in libavfilter in @FFmpeg/libavfilter/vf_transition_cuda.c , media wipe should be done using already existing primitives (@avplumber/src/nodes/input_rec.cpp > @avplumber/src/nodes/demux.cpp > @avplumber/src/nodes/decoders.cpp > @avplumber/src/nodes/realtime.cpp ). Some timestamp-aware logic will be needed for media wipe playback synchronized with scenecut.

Remember that inputs (mostly cameras) may be present on more than 1 scene, and, during crossfade, frames need to be duplicated into separate scene renderers and then merged back into vf_transition_cuda.

Remember that there needs to be some timestamp correction, because both libav filters and our multi-input nodes expect synchronized timestamps on inputs.

Make it modular. Do not invent new nodes if not necessary. We already have @FFmpeg/libavfilter/vf_overlay_many_cuda.c and @avplumber/src/nodes/hwaccel/cuda_rect_overlay.cpp (for non-alpha overlays, e.g. PIP, or stack). An orchestrator for these nodes may be needed - it's better to manage multiple nodes than write one big node with spaghetti code.

---

**Cursor**

Let me explore the codebase to understand the architecture and existing primitives before creating a plan.

Now let me explore the existing nodes and graph management infrastructure more deeply.

Let me also check a few more details about the graph interfaces and existing commands.

I now have a thorough understanding of the entire system. Let me create the plan.

---

**User**

does transition_cuda need to be controlled via commands? won't something like `transition_cuda=alpha=t/${duration}` suffice?

---

**Cursor**

Good point -- the expression engine is definitely sufficient, and `process_command` is overkill. The only nuance is that `t` is the frame's PTS in seconds (not time since filter creation), so with `realtime(set_pts=true)` feeding wallclock-aligned PTS, `t` would be something like 5000 seconds, making `t/2` = 2500 rather than a 0-to-1 ramp.

Two clean alternatives that work:

**1. Use `n` (frame count) -- starts from 0 when the filter node is (re)created:**

```
transition_cuda=alpha='clip(n/FRAMES, 0, 1)':eval=frame
```

where `FRAMES = fps * duration_sec`. The orchestrator bakes the computed integer into the expression string before (re)starting the node.

**2. Bake current `t` into the expression:**

```
transition_cuda=alpha='clip((t-5000.3)/2, 0, 1)':eval=frame
```

The orchestrator reads the current PTS (or derives it from wallclock) and embeds it as a constant. A bit more fragile than `n`.

**Option 1 (`n`) is simpler and self-contained.** The whole crossfade flow becomes:

1. Both scenes always render (A/B slots)
2. `transition_cuda` is always in the graph with `auto_restart: "on"`
3. To fade A->B: `node.param.set mixer_transition graph "transition_cuda=alpha='clip(n/90,0,1)':eval=frame"` then `node.auto_restart mixer_transition` (one restart, ~1 frame covered by sentinel)
4. After 90 frames (3s at 30fps): alpha stabilizes at 1.0 -- done, B is now program
5. Next fade: load new scene into slot A, use `alpha='clip(1-n/90,0,1)'` (reverse direction)

This alternating-direction pattern means only **one restart per transition** and **no `process_command` at all**.

This eliminates the entire FFmpeg change (Component 1) and the `filter.command` plumbing (Component 2) from the plan. The orchestrator just needs `node.param.set` + `node.auto_restart`, which already exist.

Shall I update the plan to use this simpler approach?

---

**User**

yes, use option 1. remember to add force_fps, maybe before first compositor? and remove the sentinel, the output is buffered anyway.

---

**Cursor**

Updating the plan to use the `n`-based expression approach, add `force_fps` before the compositor, and remove the sentinel.

Now replace Components 1-2 (FFmpeg/filter.command) with the simplified crossfade mechanism, and update Components 3-4:

Good, the renumbering happened automatically. Now let me update the orchestrator section (Component 3 -> Component 3) and the remaining sections:

Now update the Component 5 numbering and the files section:

Plan updated. Key changes:

1. **Crossfade uses `n`-based expressions** -- `transition_cuda=alpha='clip(n/FRAMES,0,1)':eval=frame` with `node.param.set` + `node.auto_restart`. The `n` variable resets to 0 on restart, giving a natural 0-to-1 ramp. Alternating direction (forward/reverse expression) avoids ever needing to swap `src` order.

2. **`force_fps` added after `realtime`, before `split`** -- normalizes all inputs to the target frame rate before they enter the compositor. This ensures `cuda_rect_overlay` gets frames at a consistent cadence and the `FRAMES` calculation for crossfade duration is reliable.

3. **Sentinel removed** -- the output is buffered through encoder/mux queues, which absorbs the ~1 frame gap from `node.auto_restart`.

4. **No FFmpeg modifications needed** -- the entire FFmpeg `process_command` component and the `filter.command` plumbing are gone. The plan is now purely avplumber-side: `MixerState`, `MixerOrchestrator`, mixer commands, and an example script.

---

**User**

how is scene selection for output done?

---

**Cursor**

`transition_cuda` itself **is** the scene selector. It's always in the graph with both scene slots wired to it:

- Input 0 ("main") = Slot A output
- Input 1 ("overlay") = Slot B output

The blend formula is: `output = alpha * overlay + (1 - alpha) * main`

So the alpha value controls everything:

| State | Alpha | Output |
|---|---|---|
| PGM is Slot A (steady) | `0` | `0 * B + 1 * A = A` |
| Crossfade A→B (mid) | `0.5` | `0.5 * B + 0.5 * A` (blend) |
| PGM is Slot B (steady) | `1` | `1 * B + 0 * A = B` |
| Crossfade B→C (mid) | `0.5` | `0.5 * B + 0.5 * A` where A is now C |

There's no separate router or selector node. In steady state, `transition_cuda` with a constant alpha (0 or 1) is effectively a passthrough of the active scene. It does a trivial per-pixel multiply (alpha * 0 or alpha * 1) which is negligible GPU cost compared to encoding.

The orchestrator never needs to rewire edges -- it only changes the alpha expression string via `node.param.set` and restarts the filter with `node.auto_restart`. The `src` order stays fixed forever: `["scene_a_sm", "scene_b_sm"]`.

---

**User**

Make it waste less GPU resources:
Create 1-to-many and many-to-1 switcher nodes
managed by orchestrator
1-to-many:
* should be actually "split-or-switch" because we need to handle that edge case with fade transition.
* place it on inputs (cameras) after signal is normalized to our fps

many-to-1:
used for selecting scene or fade transition that will land on the output

efficient fade transition:
place 1-to-many switcher (with 2 outputs) after @avplumber/src/nodes/hwaccel/cuda_rect_overlay.cpp to allow switching between passthrough (scene -> output) and transition ([scene1, scene2] -> transition filter -> output)

---

**Cursor**

Good idea -- this is significantly more efficient. Let me read the graph base classes to nail down the implementation details for the new nodes, then update the plan.

Now I have a clear picture of the base classes and the existing switcher. Let me update the plan with the new architecture.

Now update the input layer, scene rendering, and transition sections:

Now replace the crossfade mechanism and orchestrator sections:

Now update the MixerState and MixerOrchestrator sections:

Now update the files list:

Plan updated with the efficient GPU architecture. Here's what changed:

**Two new lightweight nodes:**
- **`split_or_switch`** -- replaces `split`. In switch mode, routes to only one output (zero wasted copies). The orchestrator flips it to split mode only during crossfade for shared cameras. Placed both on inputs (after `force_fps`) and after each scene compositor.
- **`output_selector`** -- many-to-1 selector. Forwards only the active input, drains the rest. Has 3 inputs: Scene A direct, Scene B direct, and transition output.

**`transition_cuda` is now fully dynamic:**
- Does NOT exist in steady state (zero GPU overhead when not transitioning)
- Created fresh by the orchestrator for each crossfade, with the `n`-based alpha expression baked in
- Deleted after the crossfade completes

**Steady-state efficiency:**
- Only the PGM scene's compositor runs on GPU
- No frame duplication (input switches in switch mode)
- No transition filter
- The inactive scene's compositor blocks on empty queues (costs nothing)

**The crossfade sequence** (steps 1-13) is explicit about when each switch happens and in what order, ensuring no visible glitch while keeping GPU work minimal.

---

**User**

what about timestamping? the split_or_switch and output_selector need to switch at the same frame time, but because of queues, not necessarily at the same wallclock time.

---

**Cursor**

Excellent catch. The commands arrive at different wall-clock times, but frames in queues between those nodes were stamped earlier. A frame already sitting in a queue downstream of `split_or_switch` might get processed by `output_selector` AFTER the selector switches but BEFORE the transition filter has started producing -- causing a gap or stale frames.

The fix: **PTS-based scheduled switching**. Instead of instant mode changes, both nodes accept a scheduled change that takes effect at a specific frame PTS. The orchestrator sends all schedules at once, all targeting the same PTS. Each node applies the change when it sees a frame with `PTS >= target`.

Since `realtime(set_pts=true)` maps PTS to wallclock time, the orchestrator can compute `target_pts = wallclock_now + margin` (e.g., +200ms) to ensure all commands arrive before the target frame does.

Let me update the plan with this mechanism.

Now let me also update the crossfade flow section to reflect the PTS-scheduled approach.

Now let me update the Crossfade Flow to use PTS-scheduled switches instead of the sequential approach.

Now let me also update the Hard Cut section to use PTS-scheduled switching, and update the file sizes to reflect the added PTS scheduling code.

Now let me also check how `av::Timestamp` works to make sure the PTS comparison approach is sound.

Good -- `av::Timestamp` has a raw `timestamp()` returning `int64_t` and a `timebase()` returning a `Rational`. The comparison operators use `av_compare_ts` which handles different timebases. Now let me check what timebase `realtime(set_pts=true)` produces to understand the PTS values.

Important finding: `force_fps` rescales PTS to its own timebase (default `{1, fps}` for fps=30/1, meaning PTS values are frame counts: 0, 1, 2, ...). This is different from `realtime`'s `{1, 1000}` millisecond timebase. The PTS-scheduling must handle this timebase mismatch.

The fix: use `av::Timestamp` with explicit timebases. The schedule stores values in milliseconds (`{1, 1000}`, matching `wallclock`), and the comparison uses `av_compare_ts` which handles different timebases automatically.

Now update the `output_selector` similarly.

Now update the PTS-Synchronized Switching section to include the concrete timebase details.

Also update the Timestamp Flow section to reflect the force_fps timebase change.

Here's a summary of what was added to the plan:

**PTS-Scheduled Switching** -- Both `split_or_switch` and `output_selector` now support a `schedule` command that takes `at_pts_ms` (wallclock milliseconds). Instead of applying state changes immediately, they apply them when a frame's PTS reaches the target value. Key details:

1. **The problem:** Queue depth between nodes means wall-clock-timed `node.object.set` commands hit different nodes at different times, but the frames in those queues were stamped earlier. A naive instant switch could cause gaps or stale frames.

2. **The solution:** All schedule values are in milliseconds (`{1, 1000}` timebase), matching `wallclock.pts()`. The comparison uses `av::Timestamp::operator>=` which calls `av_compare_ts` internally -- this correctly handles the timebase mismatch between the schedule (ms) and frames after `force_fps` (which rescales to `{1, fps}` timebase, where PTS values are frame counts).

3. **Orchestrator flow:** Compute `target_pts_ms = wallclock.pts() + margin_ms`, then send schedule commands to all affected nodes in a batch. Each node independently triggers at the same frame boundary. The crossfade END can be pre-scheduled too (`target + duration*1000`), so the entire transition is dispatched as one batch with no timer thread needed for completion.

4. **Crossfade setup phase** uses immediate switches (split mode on input and post-compositor switches) to prime the pipeline before the scheduled PTS arrives. Between the immediate setup and the PTS trigger, `transition_cuda` warms up while its output is still ignored by the output_selector.

---

**User**

what about more general mechanism of shared timeline with planned state changes wrt ptses, that could be used by all nodes that need it?

---

**Cursor**

Great idea -- a general-purpose shared timeline decouples the PTS-scheduling logic from individual nodes. Instead of each node reimplementing its own scheduling queue, there's one instance-shared object that any node can check. Let me think through the design.

Now let me check how existing nodes reference instance-shared objects (like `RealTimeTeam`) to understand the pattern for the shared timeline.

Now let me check how `node.object.set` dispatches to nodes.

Now let me also check NodeWrapper's setObject/getObject to understand the dispatch.

Good thinking. Here's how a general `SharedTimeline` would work:

**Core concept:** A single instance-shared object holding a priority queue of `{pts, node_channel, key, value}` events. Any node that opts in (via a `timeline` parameter) checks it during `process()` and applies pending events to itself via the existing `setObject`/`IInputsObjects` interface. The individual nodes don't need any scheduling logic at all.

Let me draft this in the plan.

Now I have full context. Let me replace the per-node scheduling with a shared timeline.

Now update the "Files to Create/Modify" section to include the new SharedTimeline file and remove the inflated LOC estimates.

Now renumber the MixerState and MixerOrchestrator component headings since SharedTimeline is now Component 1.

Now update the Mixer Control Commands section to include the `timeline.schedule` and `timeline.batch` commands.

Now update the orchestrator description to reference SharedTimeline.

Now update the todo list in the frontmatter.

Also update the Modify section for avplumber.cpp to mention the timeline commands.

Now update the control commands section to include the timeline commands.

Now let me add the timeline commands to the control commands section.

Here's a summary of the refactored design:

**`SharedTimeline`** is a new general-purpose instance-shared object (`src/SharedTimeline.hpp`) that replaces the per-node scheduling code. The key changes:

1. **`SharedTimeline`** -- a per-channel priority queue of `{at_pts_ms, key, value}` events. Any node can push events, any node can consume them. Events are stored in wallclock milliseconds and compared using `av_compare_ts` for correct cross-timebase handling.

2. **`TimelineConsumer<Derived>`** -- a thin CRTP mixin. Nodes that inherit it call `applyTimeline(frame_pts)` at the top of `process()`. The mixin calls `timeline->consumeUpTo(channel, pts)` and dispatches each event through the node's existing `setObject()` method. No per-node scheduling code needed.

3. **`split_or_switch` and `output_selector`** are now simpler (~60 and ~50 LOC) -- they just implement `setObject` for their parameters (`mode`/`active`) and inherit `TimelineConsumer` for PTS-synchronized control. All scheduling logic lives in `SharedTimeline`.

4. **New control commands**: `timeline.schedule`, `timeline.batch`, and `timeline.cancel` give external control over the shared timeline, independent of the mixer. The orchestrator uses the same `SharedTimeline` API in C++.

5. **Reusable beyond the mixer**: Any future node that needs PTS-synchronized state changes (e.g., scheduled filter parameter changes, timed audio routing) can opt in just by adding `TimelineConsumer` and a `timeline` parameter.

---

**User**

I'm also thinking of architecture for the timeline with node asking for current value within its process(). what would be better in your opinion? (just asking, be honest)

---

**Cursor**

Honestly, I think the **current-value / query model** is better than the event-consume model for this use case. Let me lay out both and explain why.

## Event-consume model (current plan)

```cpp
// Timeline stores: queue of {at_pts, key, value} events
// Node calls: timeline->consumeUpTo(channel, pts) → returns & removes events
// Node applies them via setObject, maintaining its own state
void process() {
    T* data = this->source_->peek();
    applyTimeline(data->pts()); // consume events, call setObject for each
    int active = active_.load(); // read internal state set by setObject
    ...
}
```

**Pros:** Low per-frame overhead (usually no events pending, just a quick check). Natural for side-effects on change. Simple data structure.

**Cons:**
- Events are consumed (destroyed). If the node restarts via `node.auto_restart`, it wakes up with default state and no way to recover -- the events are gone.
- Nodes must maintain their own state copies. The timeline is just a transport mechanism.
- Non-idempotent: reading twice gives different results.
- Hard to debug: you can't inspect "what should the state be at time T?" because past events are erased.

## Current-value / query model

```cpp
// Timeline stores: sorted history of {at_pts, value} per (channel, key)
// Node calls: timeline->get(channel, key, pts) → returns the value active at that PTS
// No internal state needed for timeline-driven parameters
void process() {
    T* data = this->source_->peek();
    int active = timeline_->get<int>(channel_, "active", data->pts()); // query directly
    ...
}
```

**Pros:**
- **Restart-resilient.** This is the big one. You use `node.auto_restart` as a core mechanism. A restarted node just queries the timeline and gets the correct state -- nothing was consumed or lost.
- **Nodes are nearly stateless** for timeline-driven parameters. No `setObject`, no internal `std::atomic<int> mode_`. The timeline IS the source of truth.
- **Idempotent.** Query it ten times, same answer. Makes reasoning about correctness trivial.
- **Debuggable.** You can dump the timeline at any time and see the full state history -- what each parameter should be at any point.
- **Late-joining nodes.** If `transition_cuda` is created mid-crossfade, it can immediately query the timeline for the correct state. No need to replay missed events.

**Cons:**
- Per-frame lookup cost: binary search over the history per key per frame. But for 2-3 entries per key (typical: one for start, one for end of crossfade), this is `O(1)` in practice.
- No built-in "on change" notification. If a node needs to perform a side effect when a value changes (e.g., log a message), it has to detect the change itself by comparing against the last value it read.
- Needs garbage collection of old entries (entries before the earliest possible PTS in the system). But this can be lazy -- just trim occasionally.

## My recommendation

**Go with the query model.** For this use case:

1. The parameters are trivially simple (`mode`: string, `active`: int). Per-frame lookup is negligible.
2. `node.auto_restart` is a core mechanism -- restart resilience matters a lot.
3. The "no side effects on change" downside barely applies. The split_or_switch and output_selector don't need side effects -- they just need to know the current mode/active value when processing a frame.
4. The timeline becomes a proper "state-over-time" model, not just a message queue. This is conceptually cleaner and more general.

The data structure would look like:

```cpp
class SharedTimeline : public InstanceShared<SharedTimeline> {
    struct Entry { int64_t at_pts_ms; Parameters value; };
    // channel -> key -> sorted vector of entries
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<Entry>>> data_;
    std::mutex mutex_;
public:
    // Set the value of key for channel starting at at_pts_ms
    void set(const std::string& channel, const std::string& key,
             int64_t at_pts_ms, const Parameters& value);

    // Get the value of key for channel at frame_pts (latest entry <= frame_pts)
    // Returns nullopt if no entry exists at or before this PTS
    std::optional<Parameters> get(const std::string& channel, const std::string& key,
                                   const av::Timestamp& frame_pts);

    // Convenience typed getter with default
    template<typename T>
    T get(const std::string& channel, const std::string& key,
          const av::Timestamp& frame_pts, const T& default_val);
};
```

And nodes become trivially simple:

```cpp
void process() {
    T* data = this->source_->peek();
    if (!data) return;
    // No setObject, no internal state -- just ask the timeline
    auto mode_str = tl_->get<std::string>(channel_, "mode", data->pts(), "switch");
    Mode mode = (mode_str == "split") ? Mode::Split : Mode::Switch;
    int active = tl_->get<int>(channel_, "active", data->pts(), 0);
    // ... route frames based on mode/active
}
```

The nodes that don't use the timeline (or that also support immediate `node.object.set`) can keep their internal state as a fallback -- the timeline value just overrides when present.

Want me to update the plan with this model?

---

**User**

yes of course

---

**Cursor**

Now I have the full plan. Let me rewrite the SharedTimeline, the node components, the orchestrator scheduling section, and the crossfade flow to use the current-value query model.

Now update the `split_or_switch` node to use the query model.

Now update the `output_selector` node similarly.

Now update the PTS-Synchronized Switching / orchestrator scheduling section to use `set` instead of `schedule`.

Now update the crossfade flow to reference `tl->set()` consistently and mention the query model.

Now update the todo list and the overview line to reflect the new approach.

Also update the files list for SharedTimeline.

Also update the control commands section since we renamed `timeline.schedule` to `timeline.set`.

Let me also update the node.add examples to include the `timeline` parameter.

Done. Here's a summary of the change:

**SharedTimeline is now a current-value store, not an event queue.** The key differences:

1. **`set()` instead of `schedule()`** -- The orchestrator writes entries with `tl->set(channel, key, at_pts_ms, value)`. These entries persist in the timeline (not consumed).

2. **Nodes query, not consume** -- In `process()`, nodes call `tlGet<int>("active", frame_pts, fallback)` which does a binary search for the latest entry at or before the frame PTS. Same call, same answer, every time.

3. **`TimelineReader` instead of `TimelineConsumer`** -- The mixin just provides `tlGet<T>(key, pts, fallback)`. No CRTP needed since there's no dispatch to `setObject`.

4. **Dual control paths** -- Timeline values override the internal fallback state. `node.object.set` still works for immediate changes (sets the fallback), but timeline entries take priority when present. This gives maximum flexibility.

5. **GC for cleanup** -- Old entries are pruned by `timeline.gc` or `tl->gc(before_pts_ms)`. The GC keeps the latest entry before the threshold (it's the "current" value) and removes everything older.

---

**User**

rename 'split_or_switch' to 'one_to_many', 'output_selector' to 'source_switcher'

---

**Cursor**


Updating the plan document to rename `split_or_switch` → `one_to_many` and `output_selector` → `source_switcher`.




Updating the plan: applying systematic replacements and adding a note distinguishing the new `source_switcher` from the `_unfinished` stub.




Updates are in [video_mixer_architecture_d6ffbd2a.plan.md](file:///home/teowoz/.cursor/plans/video_mixer_architecture_d6ffbd2a.plan.md):

**Renames**
- Node type **`split_or_switch`** → **`one_to_many`** (file `src/nodes/one_to_many.cpp`, C++ class **`OneToMany`**).
- Node type **`output_selector`** → **`source_switcher`** (file `src/nodes/source_switcher.cpp`, C++ class **`SourceSwitcher`**).

**Also adjusted**
- Overview, todos (`one-to-many-node`, `source-switcher-node`), JSON `node.add` examples, mermaid (node id **`srcSw`** instead of **`outSel`**), timeline/orchestrator snippets, `MixerState` field **`source_switcher_name`**, and the files list.
- **Note** under Component 3: the new **`source_switcher`** is separate from [src/nodes/_unfinished/source_switcher.cpp](file:///home/teowoz/tly/avplumber/src/nodes/_unfinished/source_switcher.cpp).

Example node names like **`out_sel`** are unchanged on purpose (they are instance names, not types).

---

**User**

Now let's think about scenes. There may be multiple (more than 2) scenes defined (e.g. single cam full screen, PiP, side-by-side), but our mixer has PGM+PVW 2-scene architecture. It means that we could either:
* configure the PVW compositor appropriately to prepare it for upcoming scene, before each scene change. (note that changes in cuda_rect_overlay may be necessary to support discarding some inputs, or null_sinks need to be dynamically inserted, NOTE: remember about timeline and timestamp sync)
* increase number of compositors (remember about needed changes down the graph)
Which would be better?

Also note that cuda_rect_overlay does not support resizing by itself (it doesn't use opengl after all! only simple cumemcpy), so between one_to_many and cuda_rect_overlay there need to be inserted: crop, scale_cuda. In the "always 2 scene" architecture these filters will also need to be updated on scene change. Remember about frame duplication edge case during fade transition.

---

**Cursor**

Great question. Let me think through both approaches carefully, especially considering the crop/scale requirement and the crossfade edge case.

## Analysis: Reconfigure 2 slots vs. N compositors

### The crop/scale problem kills Option 1

Consider a fade from Scene A (cam1 fullscreen 1920x1080) to Scene B (PIP: cam1 at 480x270, cam2 at 1920x1080):

During crossfade, cam1 is duplicated via `one_to_many(mode=split)`. But each copy needs **different** crop/scale parameters:
- Scene A path: cam1 → (no scale) → compositor A
- Scene B path: cam1 → scale_cuda(480x270) → compositor B

With the "2 slots, reconfigure" approach:
- Slot B's crop/scale chain needs to be reconfigured before the crossfade starts
- But `filter_video` reconfiguration (`node.auto_restart`) causes a brief gap -- frames are lost during restart
- You'd need to do this for ALL inputs that change size between scenes
- The reconfiguration happens at wallclock time, but it needs to align with PTS-based timeline switching -- a timing nightmare
- Even worse: if you try to reconfigure the crop/scale **during** a crossfade (because PVW was already showing something else), you'd lose frames mid-transition

With N compositors:
- Each scene has its own pre-configured crop/scale chain: `cam1 → scale_cuda(480x270)` lives permanently in the Scene B chain
- No reconfiguration. Ever.
- `one_to_many` just routes cam1's frames to both chains during crossfade
- Timeline switching works naturally because no nodes are being restarted

### GPU waste concern is a non-issue

With `one_to_many(mode=switch)` in steady state, inactive scene chains receive zero frames. The crop/scale and compositor nodes block on empty input queues -- no GPU cycles consumed. Only the active scene processes frames. During crossfade, exactly 2 scene chains run (same as the 2-slot approach).

The only "waste" is memory: inactive filter_video/compositor nodes hold allocated CUDA contexts. For 5-10 scenes this is negligible.

### My recommendation: N compositors

**Option 2 is clearly better.** Here's the updated per-camera chain:

```
cam → realtime → force_fps → one_to_many
                                ├─ [scene_fullcam] → (passthrough, no scale) → compositor_fullcam
                                ├─ [scene_pip]     → scale_cuda(480x270)     → compositor_pip
                                └─ [scene_sbs]     → scale_cuda(960x1080)    → compositor_sbs
```

Each scene's crop/scale chain is static. The `one_to_many` selects which scene(s) receive frames.

The downstream also changes:

```
compositor_fullcam → smooth_ts → one_to_many_fullcam ──→ [direct] ──→ source_switcher
compositor_pip     → smooth_ts → one_to_many_pip     ──→ [direct] ──→ source_switcher
compositor_sbs     → smooth_ts → one_to_many_sbs     ──→ [direct] ──→ source_switcher
                                                          [trans]  ──→ transition_cuda ──→ source_switcher
```

`source_switcher` has N+1 inputs: one direct per scene + one transition output.

**N-compositor architecture details**

> **When should scene chains be built? This affects startup time vs. latency of first scene switch.**
> - Eager: build all scene chains at definition time (faster scene changes, more upfront memory)
> - Lazy: build on first PVW assignment, keep alive for reuse, explicit teardown (less memory, slight first-switch delay)
> - Lazy with TTL: build on first PVW, auto-teardown after idle timeout (least memory, re-creation delay if reused after timeout)

> **With N scenes, each camera's one_to_many needs one output per scene that uses it. These outputs are added when scenes are defined. Should we support dynamic output addition to one_to_many?**
> - Yes, support add/remove outputs at runtime (needed if scenes are defined dynamically)
> - No, outputs are fixed at node creation (scenes must be defined before starting the graph)
> - Recreate the one_to_many node when scenes change (simplest, brief frame loss on scene definition)

---

**User**

what about compositor outputs? multiple compositors need more complications with routing to transition filter. also, remember that in the "always 2 compositors" architecture we can have one_to_many split-or-switch BEFORE cropping and scaling and have always 2 scale_cuda filters changing their parameters before transitions. but remember about timestamps, too

---

**Cursor**

This is a critical architectural decision. Let me think through both approaches carefully, including the crop/scale pipeline and timestamp implications you raised.

## Approach A: Always 2 compositors (reconfigure PVW before each transition)

Pipeline per camera, per slot:

```
cam1 → force_fps → one_to_many ─→ [crop_a → scale_cuda_a] → compositor_a
                                 └→ [crop_b → scale_cuda_b] → compositor_b
```

**Scene change lifecycle (steady state → prepare PVW → crossfade → cleanup):**

1. **Steady state:** one_to_many in switch mode, only active slot receives frames. PVW slot's crop/scale/compositor are idle.
2. **Prepare PVW:** Reconfigure PVW slot's crop/scale `graph` params + compositor `layers` via `node.param.set` + `node.auto_restart`. Safe because no frames are flowing to this slot.
3. **Crossfade start:** one_to_many switches to split → PVW crop/scale/compositor starts receiving fresh frames with correct PTS. Parameters are already set.
4. **Crossfade end:** one_to_many switches back to switch. Tear down nothing -- just reconfigure the now-idle slot when the next scene change is requested.

**Problem: unused cameras.** If the scene uses cam1 but not cam2, cam2's one_to_many doesn't route to this slot. But `cuda_rect_overlay` is wired to both cam1_a and cam2_a -- it blocks waiting for cam2_a to produce frames.

**Solutions for unused inputs:**
- **(a)** Add `active_inputs` bitmask to `cuda_rect_overlay` (timeline-driven). Inactive inputs are not waited on, not peeked, and render as transparent/black. Small change (~20 LOC in the sync loop).
- **(b)** Route unused cameras' one_to_many to both slots anyway, but put the crop/scale off-screen. Wastes GPU on scale_cuda for invisible layers.
- **(c)** Dynamically insert/remove null_sinks. Messy, timestamp-fragile.

Option (a) is cleanest. The compositor's existing min_ts synchronization loop just skips inactive inputs.

**Routing to transition filter:** Simple. Always `[slot_a_out, slot_b_out] → transition_cuda`. source_switcher has exactly 3 inputs: `direct_a`, `direct_b`, `trans_out`. Never changes.

## Approach B: N compositors (one per scene definition)

```
cam1 → force_fps → one_to_many(N) ─→ [crop_scene1 → scale_scene1] → compositor_scene1
                                     ├→ [crop_scene2 → scale_scene2] → compositor_scene2
                                     └→ [crop_scene3 → scale_scene3] → compositor_scene3
```

**Problems:**
1. **Transition routing.** Which two compositor outputs feed `transition_cuda`? Need a switching layer before the transition filter -- essentially a second one_to_many after each compositor, plus the source_switcher needs N+1 inputs. Or dynamically rewire `transition_cuda`'s src list, which means teardown/recreate.
2. **GPU footprint.** N compositors, N*cameras crop/scale chains. Even idle, they hold allocated resources.
3. **one_to_many fan-out.** Each camera needs N outputs instead of 2. During crossfade, split mode duplicates to all N (wasteful -- only 2 are needed).
4. **Dynamic scene additions.** Adding a new scene means wiring a new output on every one_to_many, new crop/scale chains, a new compositor, and rewiring the source_switcher. Much harder than reconfiguring an idle slot.

## My recommendation: Approach A (always 2 compositors)

The routing simplicity alone is decisive. With 2 compositors:
- The transition path is fixed: always `[slot_a, slot_b] → transition_cuda`.
- source_switcher has exactly 3 fixed inputs.
- one_to_many has exactly 2 outputs per camera.
- Adding a new scene definition is just JSON -- no graph topology change.

The only cost is reconfiguring the idle slot's crop/scale/compositor before each transition. But this happens while no frames flow to that slot, so it's safe and instant.

## Crop/scale pipeline with timestamps

The chain between one_to_many and compositor, per camera per slot:

```
one_to_many → filter_video(graph="crop=...,scale_cuda=...") → compositor
```

Using `filter_video` to wrap the `crop` + `scale_cuda` filter chain means:
- PTS is preserved through the filter (FFmpeg filters don't modify PTS by default)
- `filter_video` supports `node.param.set` on `graph` + `node.auto_restart` to change crop/scale parameters
- The `n` variable resets on restart, but that only matters for expression-based filters (not crop/scale)

**Timestamp flow during crossfade:**
1. one_to_many splits frame (PTS=T) to both slots
2. Slot A's crop/scale processes it → output PTS=T (unchanged)
3. Slot B's crop/scale processes it → output PTS=T (unchanged)
4. Compositor A: output PTS = min(input PTSes) = T
5. Compositor B: output PTS = min(input PTSes) = T
6. transition_cuda framesync: matches by PTS → both at T → correct blend

**Frame duplication edge case during fade:** When a camera is shared between PGM and PVW scenes, one_to_many is in split mode. Both crop/scale chains get the same frame (same PTS). Each chain may have different crop/scale parameters (e.g., fullscreen in scene A, small PiP in scene B). The frame data is read-only (GPU surface) -- both chains read the same CUDA memory, no conflict. PTS is preserved through both paths. transition_cuda sees matching PTSes from both compositors.

**Reconfiguration timing (critical):** The PVW slot's crop/scale are reconfigured while one_to_many is in switch mode (no frames flowing). The sequence:

```
1. node.param.set crop_cam1_b graph "crop=...,scale_cuda=..."
2. node.auto_restart crop_cam1_b          ← filter rebuilds with new graph
3. node.param.set crop_cam2_b graph "..."
4. node.auto_restart crop_cam2_b
5. node.param.set compositor_b layers [...]
6. Set active_inputs bitmask on compositor_b via timeline
7. THEN: switch one_to_many nodes to split → frames start flowing to slot B
```

All reconfiguration happens before any frames arrive. No timestamp issues.

**Change needed in `cuda_rect_overlay`:** Add `active_inputs` (bitmask or JSON array) as a parameter readable from SharedTimeline. In the main sync loop, skip inactive inputs:

```cpp
// In the frame synchronization loop:
for (int i = 0; i < num_inputs; i++) {
    bool active = active_inputs & (1 << i);  // from timeline or param
    if (!active) {
        // Don't wait for this input, use transparent/black
        continue;
    }
    // existing peek/wait logic...
}
```

Want me to update the plan with this 2-compositor architecture, including the crop/scale pipeline and `cuda_rect_overlay` `active_inputs` change?

---

**User**

yes. also, update one_to_many definition, to allow arbitrary output mask - this way, crop/scale_cuda is not run for cameras that are not needed right now.

---

**Cursor**

Let me read the current state of the plan sections that need updating.

Now let me check `cuda_rect_overlay` to understand its current sync loop and input handling.

Now I have full context on the sync loop. Let me also check whether `cuda_rect_overlay` implements `IInputsObjects`.

Good. Now I have full context. Let me do a comprehensive update of the plan to incorporate:

1. 2-compositor (PGM/PVW slot) architecture with scene reconfiguration
2. Crop/scale pipeline between `one_to_many` and compositor
3. `one_to_many` with output bitmask instead of mode/active
4. `active_inputs` bitmask on `cuda_rect_overlay` (+ TimelineReader)
5. Updated crossfade flow, MixerState, orchestrator methods

Now update the todos to include the new cuda_rect_overlay change.

Now rewrite the Architecture Overview and diagram.

Now update the Input Layer section with crop/scale and bitmask-based one_to_many.

Now update the Scene Rendering Layer.

Now update the Timestamp Flow section.

Now update the Steady-State Efficiency section.

Now update Component 2: `one_to_many` node with the output bitmask design.

Now add the new cuda_rect_overlay active_inputs component after component 3 (source_switcher). Let me update accordingly.

Now renumber the remaining components (MixerState was 4, now becomes 5; MixerOrchestrator was 5, now 6; Control Commands was 6, now 7). Let me update them.

Now update MixerState to include the crop/scale scene definition and per-slot tracking.

Now update the Orchestrator section with the new loadScene, crossfade flow, etc.

Now update the crossfade flow with the bitmask-based approach.

Now update the Hard Cut section.

Now update the PGM/PVW Slot Tracking section for the bitmask model.

Now update the Orchestrator's PTS-Synchronized Switching example code.

Now update the Files to Create/Modify section.

Now let me also update the todo list to sync it.

The plan has been updated. Here's a summary of the changes:

**`one_to_many` -- output bitmask model:**
- Replaced the `mode` (split/switch) + `active` (int) design with a single `outputs` bitmask (`uint32_t`). Each bit corresponds to an output. This is more general: `0b01` = only slot A, `0b10` = only slot B, `0b11` = both, `0b00` = discard. The orchestrator controls exactly which downstream paths receive frames per camera.
- This means crop/scale chains for cameras not needed by a scene never run -- the output bit is simply never set.

**`cuda_rect_overlay` -- `active_inputs` bitmask:**
- Added Component 4 describing minimal changes (~30 LOC) to the existing `CudaRectOverlay` class: inherit `TimelineReader` + `IInputsObjects`, add an `active_inputs_` atomic, and gate every loop in `process()` with `isActive(i)`. This lets the compositor run with a subset of its wired inputs, which is essential when a scene doesn't use all cameras.

**2-slot architecture with crop/scale pipeline:**
- The diagram and layer descriptions now show per-camera `filter_video(crop+scale_cuda)` chains between `one_to_many` and the compositor.
- Scene changes reconfigure the idle slot's crop/scale params via `node.param.set` + `node.auto_restart` while no frames flow (safe).
- `MixerState` now tracks per-camera `CameraInfo` (otm node name, input index, crop/scale node names per slot).

**Crossfade flow:**
- Updated with concrete bitmask values, showing how shared cameras get both bits set while exclusive cameras keep only their slot's bit.

