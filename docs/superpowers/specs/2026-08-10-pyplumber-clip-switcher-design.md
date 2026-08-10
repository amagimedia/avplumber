# PyPlumber Clip Switcher (Growable Playlist, Frame-Perfect Cuts, Mixer Transitions)

## Goal

Provide a PyPlumber application that reproduces the runtime behavior of the OBS
`mse-source-switcher` plugin without OBS: play an ordered, **runtime-growable**
playlist of media clips through a single video output, switching between clips
with **frame-perfect** cuts and optional **fade/wipe transitions** identical to
those the mixer already produces.

The design is deliberately a composition of primitives that already exist in the
tree — `source_switcher`, `SharedTimeline`, the worker-group lifecycle, and the
`MixerGraphBuilder` transition scheduling — rather than a new node type. The
core mechanics were validated empirically before this note was written (see
[Validation](#validation)).

## Motivation

The OBS `mse-source-switcher` plugin
(`obs2/plugins/mse-source-switcher/mse-source-switcher.cpp`) drives a sequence
of clips through a small pool of reusable OBS source workers. Two structural
properties make it work:

- a **playlist of clip metadata** that can grow while playing; and
- a **fixed-size pool of source workers** (typically 2–3), each of which is
  reloaded (`obs_source_update` with `force_restart:true`) to point at the next
  clip when it becomes free.

Downstream products (eka-recorder-ai, streamstudio-gateway) already run
AVPlumber as their media core, so reproducing this behavior as a PyPlumber
script removes OBS from the runtime path for sequenced-clip playout while
keeping the exact switch semantics.

## Non-Goals

- This is not a new C++ node. No `switcher.cpp` is proposed; the behavior is
  assembled from existing nodes plus a Python coordinator.
- Audio sequencing is out of scope for the first version (video only), matching
  the initial scope of the replay-player work.
- Gapless concatenation into a single continuous PTS timeline is **not** a goal;
  each clip keeps its own PTS domain and the output realtime node reclocks.

## Architecture

The OBS clips-vs-sources split maps directly onto AVPlumber concepts:

| OBS concept | AVPlumber concept |
|---|---|
| `clips[]` (growable metadata list) | plain Python `list` in the coordinator |
| `sources[]` (fixed worker pool) | fixed set of **worker groups** |
| `obs_source_update(force_restart)` | `group.stop` + `node.delete`/`node.add` + `group.start` |
| program/preview source selection | `source_switcher` `active` input |
| `obs_transition_*` fade/wipe | `MixerGraphBuilder` `fade`/`cuda_wipe`/`wipe` |
| transition scheduling | PTS-keyed `SharedTimeline` entries |

### Data flow

```text
worker_A: input_rec -> demux -> dec_video -> rescale/scale -> workerA_out ---+
                                                                             |
worker_B: input_rec -> demux -> dec_video -> rescale/scale -> workerB_out ---+--> source_switcher[sel] -> realtime -> <output>
                                                                             |        (timeline seq_tl,
worker_C: ... (optional third worker for transition overlap) --------------+          channel sel, key active)
```

For **hard cuts** two worker groups suffice (one playing, one preloading the
next clip). For **fade/wipe transitions** both the outgoing and incoming clips
must render simultaneously for the transition duration, so the transition path
routes both live edges into a `MixerGraphBuilder` scene instead of (or in
addition to) `source_switcher`. A third worker group gives headroom to preload
the clip *after* the transition while two are still mixing.

### Worker pool lifecycle

Each worker group is `input_rec -> demux -> dec_video -> {rescale_video|scale_cuda} -> <group>_out`,
where `<group>_out` is a **named edge** consumed by the selector/mixer. The
coordinator keeps the edge names stable and only ever tears down and rebuilds
the *nodes* inside a worker group:

1. mark a worker free when its clip's EOF is observed (edge wiretap on
   `<group>_out`, detecting the EOF marker frame — `pts().isNoPts()`, i.e.
   `frame.pts.timestamp == AV_NOPTS`);
2. `group.stop <worker>` — nodes exit their thread loops via the
   `ReportsFinishByFlag`/`markFinished` finish path;
3. `node.delete` the worker's nodes (the `<group>_out` **edge is not deleted**;
   `EdgeManager::findInternal` reuses it by name);
4. `node.add` the worker's nodes pointed at the next clip's URL, wired back to
   the same `<group>_out` edge;
5. `group.start <worker>` — the rebuilt worker refills the surviving edge.

The selector/mixer downstream of `<group>_out` is never torn down, so it never
loses its input edges and never restarts.

### Frame-perfect cuts

`source_switcher` selects its active input from a PTS-keyed `SharedTimeline`
channel (`tlGet<int>("active", pts, ...)`). The coordinator schedules a cut by
writing a timeline entry with the **PTS of the boundary frame**, not a
wall-clock time:

```text
timeline.set {"name":"seq_tl","channel":"sel","key":"active","at":<boundary_pts_ms>,"val":<input_index>}
```

Because the switch decision is keyed on the frame's own PTS, the cut lands on an
exact frame regardless of when the control command arrives, as long as it
arrives before that frame is processed. This is the load-bearing property for
frame-accurate playlist edits and was validated (see below).

### Transitions (fade / wipe)

Transitions reuse `MixerGraphBuilder` (`pyplumber/mixer.py`) rather than
reimplementing blending. The builder's `cut`, `fade`, `cuda_wipe`, and `wipe`
all take a `start_pts_ms` and schedule their effect through the same PTS-keyed
timeline mechanism as `source_switcher`, so the frame-perfect timing property
carries over by construction.

The mixer is **build-once static** (`add_source`/`add_scene` raise after
`build()`). Therefore the transition scene graph — its source inputs, one per
worker edge — is constructed at startup for the fixed worker pool. Only the
*scheduling* calls (`fade(start_pts_ms=...)`, etc.) happen at runtime. The
growable playlist lives entirely in the coordinator's clip list; the graph
topology is fixed by pool size.

Two composition options:

- **Selector-primary**: `source_switcher` is the normal output; transitions are
  performed by briefly routing through a mixer scene, then handing back to the
  selector. Simpler cut path, more moving parts at transition time.
- **Mixer-primary**: the mixer is always the output; a hard cut is just a
  `cut(start_pts_ms=...)` and a fade is `fade(start_pts_ms=...)`. Uniform code
  path; requires the mixer (CUDA) image for all playout, including plain cuts.

The recommendation is **mixer-primary when a GPU is available** (uniform,
already validated scheduling semantics) and **selector-primary on CPU-only
deployments** (cuts only, no blend). The coordinator abstracts this behind a
single `schedule_transition(kind, at_pts_ms, from_idx, to_idx)` call.

## Coordinator responsibilities

A `ClipSequencer` Python class (proposed `pyplumber/switcher.py`) owns:

- the growable `clips` list (URL + per-clip metadata);
- the fixed worker-group pool and each worker's current assignment/free state;
- an edge wiretap on each `<group>_out` to detect clip EOF and mark workers free;
- clip preloading (assign next unplayed clip to a freed worker, rebuild nodes);
- transition scheduling via the selector timeline or the mixer builder;
- a small control surface (`append_clip`, `insert_clip`, `next`, current index).

The class must not assume the playlist is complete at start: `append_clip` is
valid at any time, mirroring the OBS plugin's mid-playout `switcher_update`
diff-merge of the clips array.

## Risks and Mitigations

| Risk | Mitigation | Status |
|---|---|---|
| Edge lost when worker nodes rebuilt | Reuse edge by name; never delete `<group>_out` | **Validated** |
| Cut lands on wrong frame under control latency | PTS-keyed timeline, not wall-clock | **Validated** |
| Selector starves during worker rebuild | Downstream never torn down; other worker keeps feeding | **Validated** |
| Worker won't stop cleanly | `ReportsFinishByFlag`/`markFinished` finish path | **Validated** |
| Fade/wipe timing drifts from cut timing | Mixer transitions share the PTS-keyed scheduler | **Design-level only (needs GPU spike)** |
| Preload too late for gapless-feeling cut | Pool size ≥ 2 (≥ 3 with overlapping transitions) | Design choice |

## Validation

A CPU-only spike (`avp-spin:blocking` image) exercised two worker groups feeding
`source_switcher[sel] -> realtime -> null_sink`, driven over the TCP control
server. Results:

- **Frame-perfect cut**: after
  `timeline.set {"name":"seq_tl","channel":"sel","key":"active","at":9000,"val":1}`
  the log showed `selected input 1 at pts 228352*1/12800` — the switch fired on
  the frame crossing the scheduled PTS boundary, not at the ~35 ms-later
  wall-clock arrival of the command.
- **Edge survives teardown/rebuild**: with `worker_A` stopped and its nodes
  deleted, the `workerA_out` edge sat at `0/63` (idle, alive); after
  `node.add`/`group.start` it refilled to `63/63`. Throughout, `sel_out` and
  `workerB_out` stayed at `63/63` — the selector never starved.
- **Clean self-termination**: `group.stop` produced
  `wA_in/wA_demux/wA_dec ... reported that it finished processing` and clean
  `end of threadFunction` for each node.

Not yet validated empirically: mixer `fade`/`cuda_wipe`/`wipe` transitions,
which require a CUDA/python image. They reuse the same PTS-keyed timeline
scheduling as the validated `source_switcher` cut path, so the frame-perfect
timing property is expected to carry over, but this should be confirmed with a
GPU spike before the transition code path is relied upon.

## Directory Shape (proposed)

```text
pyplumber/
  switcher.py          # ClipSequencer coordinator (new)
demos/switcher/
  play_playlist.py     # example driver: growable playlist -> one output
```

No changes to C++ sources are required by this design.
