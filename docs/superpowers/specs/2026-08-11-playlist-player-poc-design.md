# Playlist Player PoC (Single Playlist, Two-Level Playback Modes, Hard Cuts, Janus Output)

## Goal

A minimal PyPlumber demo — built on the `demos/replay` template — that plays a
**single, runtime-editable playlist** of media clips through **one** video
output to a preconfigured Janus Streaming mountpoint, switching between clips
with **gapless, never-black hard cuts** (frame-accurate; frame-perfection
validated by test, not asserted). It faithfully reproduces the two-level
playback model of the Stream Studio Gateway playlist API and the OBS
`mse-source-switcher` plugin, but only the CPU-validated cut path (no mixer,
no GPU transitions) so the PoC runs anywhere.

This composes existing primitives (`source_switcher`, PTS-keyed
`SharedTimeline`, worker-group lifecycle) exactly as
`2026-08-10-pyplumber-clip-switcher-design.md` describes. This note narrows that
design to a shippable PoC and adds the two-level mode model and a Textual TUI.

### Purpose: a playlist-scenario simulator / tester

This demo is first and foremost a **simulator and test harness for playlist
playout scenarios** — a way to drive and observe the exact runtime behavior of
the gateway/OBS playlist engine (mode transitions, add/remove/reorder mid-play,
out-of-order jumps, mixed-fps content, boundary timing, no-black-frame
guarantee) *without* OBS or the gateway in the loop. The TUI is the interactive
driver; the `--no-tui` exercise runners (mirroring the replay demo's
`--exercise-v2`) turn scripted scenarios into pass/fail regressions. Fidelity to
the switcher/gateway semantics matters more than feature breadth.

**Primary objective: shake out bugs in the C++ backend.** The simulator's job
is to stress the existing C++ nodes (`source_switcher` selection under load,
worker `group.stop`/`node.delete`/`node.add` edge-reuse lifecycle,
`SharedTimeline` PTS scheduling, `force_fps`/`rescale` cadence, realtime
starve behavior — verified: emits nothing on starve, does *not* synthesize
black — `pause`/`speed` teams) through realistic playlist
scenarios and **surface any backend defects** — races, starvation, edge
leaks, wrong-frame switches, black-frame emission, cadence drift, leaks on
repeated rebuild. When the Python layer is provably correct but a scenario
still fails, the finding is a **C++ backend bug to report** (not something to
paper over in Python). Each exercise scenario doubles as a backend regression
probe; failures should point at the offending node/command with enough context
(status snapshot, last command, observed frames) to file against the C++ tree.

### Hard constraints (per explicit instruction)

1. **Python-only, on top of existing C++ infrastructure.** This demo is
   PyPlumber script + Textual TUI. No C++ node is added or modified. Every
   graph node used already exists (`input_rec`, `demux`, `dec_video`, `scale`/
   `scale_cuda`, `rescale_video`, `force_fps`, `source_switcher`, `pause`,
   `realtime`, `enc_video`, `bsf`, `mux`, `output`, plus `MixerGraphBuilder`
   for the fade follow-up).
2. **Do not blindly change framework/decoder/node behavior.** Reuse the nodes
   and parameters as `demos/replay`/`demos/mixer` already use them. If a
   scenario appears to *need* a change to a decoder/framework/node's output
   semantics, or a regression is spotted, **stop and prompt the user** — do not
   silently alter framework/decoder output.

## Source of truth (analyzed)

- `obs2/plugins/mse-source-switcher/mse-source-switcher.cpp` — clip list,
  fixed worker (`sources[]`) pool, `current_clip`/`next_clip`, diff-merge
  `switcher_update`, `switcher_next_clip`/`switcher_prev_clip` (skip `disabled`,
  wrap on `loop`).
- `obs2/plugins/mse-source/avplumber-source.cpp` — per-clip AVPlumber graph =
  our worker group.
- `streamstudio-gateway`:
  - `modules/playouts/interfaces/playouts.ts` — the two-level data model
    (`PLAYOUTS_Playlist` vs `PLAYOUTS_PlaylistSource`).
  - `modules/playouts/utils/getPlaylistMode.ts` — the 4 named modes.
  - `modules/player/backend/actions/buildPlaylist.ts` — maps modes to
    `setLoop` × `setCurrentClipOnly`.
  - `modules/player/backend/processPlayerAction.ts` — API verb surface.

## Two-level playback model (required, both levels)

### Level 1 — playlist mode (the 4 modes)

Four playlist booleans collapse to one mode (`getPlaylistMode.ts`), which is
exactly the `loop` x `current_clip_only` 2x2 matrix (`buildPlaylist.ts`):

| Mode | Gateway flag | loop | current_clip_only | On element end |
|---|---|---|---|---|
| **PlayAll** | `stopAll` | false | false | advance; stop after last |
| **PlayCurrent** | `stopSingle` | false | true | stop |
| **LoopAll** | `loop` | true | false | advance; wrap after last |
| **LoopCurrent** | `loopSingle` | true | true | replay same element |

### Level 2 — per-element (per-clip) mode + settings

The gateway/OBS switcher does not expose one named element enum; the per-clip
behavior is composed from `PLAYOUTS_PlaylistSource` + switcher
`switcher_clip_info` fields. For a symmetric two-level model we collapse the
load-bearing axis (the hand-off trigger + self-loop) into an explicit
**`ElementMode`** enum, parallel to the 4 playlist modes:

| Element mode | Composed from | Hand-off trigger |
|---|---|---|
| **PlayToEnd** | `media_state_switch`, not looping | advance when media reaches EOF (or `play_to`) |
| **Timed** | `time_switch` + `duration` | advance after a fixed duration (required for stills/live sources that have no natural end) |
| **LoopSelf** | `looping` = true | loop this clip's own media; only advances on manual `next`/`goto` or under playlist LoopCurrent/PlayCurrent |

Remaining per-element settings (orthogonal to the mode):

- `play_from` / `play_to` — in/out trim (switcher `start`/`stop`).
- `duration` — only meaningful for **Timed** (switcher `time_switch_duration`).
- `disabled` — skip element; `resolve_next` walks past it.
- `speed` — per-clip playback rate (switcher clip `speed`).

Not modeled in the PoC (OBS interactive-scene semantics, moot in sequential
playout because we reload the worker for the next clip): `playback_behavior`
(`always_play` / `stop_restart` / `pause_unpause`), `restart_on_activate`,
`clear_on_media_end`.

### Interaction (single `resolve_next` state machine)

The element mode decides *when* an element ends; the playlist mode decides
*what happens* on that end.

```text
element hand-off fires, per ElementMode:
   PlayToEnd -> media EOF (or play_to)
   Timed     -> duration elapsed
   LoopSelf  -> never on its own (media loops); only manual next/goto ends it
      |
      +-- current_clip_only?  yes --+-- loop? yes -> replay same element   (LoopCurrent)
      |                              +-- loop? no  -> STOP                 (PlayCurrent)
      |
      +-- current_clip_only?  no  --+-- next enabled element exists -> CUT to it
                                     +-- no next -> loop? yes -> wrap to first (LoopAll)
                                                    loop? no  -> STOP          (PlayAll)
```

A LoopSelf element under PlayAll/LoopAll would never advance automatically, so
it is only meaningful under PlayCurrent/LoopCurrent or as an explicitly-cut
element; the TUI shows this pairing but does not forbid it.

The same `resolve_next(direction)` serves natural end-of-element, manual
`next`, and manual `prev`. Modes are pure coordinator logic; they add no graph
nodes.

## Architecture

Fixed **worker pool of 2** (one playing, one preloading the next clip). The
playlist is a growable Python list; graph topology is fixed by pool size, never
by list length.

**A worker is one copy of the proven `demos/replay` player chain.** We do not
design a new pipeline — we replicate the exact node sequence replay already
wires (`demos/replay/replay.py` lines 727–751) per worker, then feed them into a
`source_switcher` and share one post-switch reclock:

```text
worker_i:  InputRec(url,preseek=play_from,loop,stop_delay,timeout)
             -> Demux -> DecVideo
             -> SpeedVideo(team=speed_i, sync_team, sync_node, speed)   # per-clip speed
             -> scale_cuda/rescale_video                                # resolution normalize
             -> ForceFPS(fps=out/1)                                     # cadence normalize
             -> Pause(worker_i_pause)                                   # FREEZE while preloaded
             -> worker_i_out --+
                               +--> source_switcher[sel] (timeline seq_tl, key active)
                               |      -> RealtimeVideoFrame(set_pts=True, tick_period=1/out,
                               |                            negative_time_tolerance=1/out)  # single output clock
                               |      -> PositionProbe -> <janus output group (replay, verbatim)>
worker_j_out --+ (as above) --+
```

Every node here is one replay already uses:

- **`SpeedVideo` on a per-worker `team`** for per-clip speed (`speed.set <team>`),
  wired with `sync_team`/`sync_node` exactly as replay does — we don't invent
  speed handling.
- **`ForceFPS(fps=out/1)`** + `scale_cuda`/`rescale_video` so **every**
  `worker_i_out` edge carries frames at the single output resolution and fps,
  regardless of the clip's native format (see "Mixed-fps / mixed-resolution").
- **`Pause`** (verified `src/nodes/pause.cpp`) — replay uses two Pause nodes
  already (`replay_transition_gate`, `replay_pause`); while paused it `return`s
  without popping, so the decoder **backpressures** and the worker sits frozen on
  decoded frames near frame 0. This is how a preloaded worker is *held ready*
  (below). It is a real freeze — unlike the mixer's `OneToMany outputs` gate,
  which **discards** non-selected frames every tick (`one_to_many.cpp` line 31
  pops unconditionally; `drop_dynamic_` forced under a timeline) and so cannot
  hold a VOD clip.
- **`RealtimeVideoFrame(set_pts=True, tick_period=1/out, …)`** — the same reclock
  replay puts after its slot; we relocate it to *after* the switcher (one shared
  copy) so both workers land on one monotonic, continuous output clock. The
  mixer does exactly this (`pyplumber/mixer.py` 672).

The **Janus output group** (`ForceKeyFrame -> EncVideo h264_nvenc -> Bsf
dump_extra -> Mux -> Output rtp`), `RtcpFeedbackListener`, and `PositionProbe`
are reused **verbatim** from replay (lines 763–778).

The Janus output group (`EncVideo h264_nvenc -> Bsf dump_extra -> Mux ->
Output format=rtp`) plus `RtcpFeedbackListener` and the `PositionProbe` are
reused verbatim from `demos/replay/replay.py`.

### Priorities: gapless first, frame-perfect where realistic

Two guarantees, in strict priority order:

1. **Gapless + never-black (hard requirement, always).** Every transition — cut,
   sequential boundary, goto, random, loop point — must produce a continuous
   output with no black frame outside an explicit fade. This is non-negotiable
   and applies to *every* scenario.
2. **Frame-perfect (best-effort, where realistically possible).** A transition
   is frame-perfect **iff both**: (a) the target is already **preloaded on a
   frame-locked worker**, and (b) the **boundary PTS is known ahead of time**.
   Where both hold we get it for free (same mechanism); where they don't we
   fall back to gapless-but-not-frame-perfect. We never sacrifice (1) to chase
   (2).

Which scenarios can be frame-perfect (both conditions met):

| scenario | preloaded? | boundary known? | frame-perfect |
|---|---|---|---|
| sequential next (PlayAll / LoopAll) | yes (natural next) | yes (`last_pts+Δ`) | **yes** |
| loop point (LoopCurrent / LoopSelf) | yes (self) | yes (known duration) | **yes** |
| Timed element -> next | yes | yes (`start+duration`) | **yes** |
| **cued/armed goto** (`goto @ pts` or `goto when-ready`) | yes (we preload first) | yes (we pick it) | **yes** |
| scripted playlist w/ rolling preload window | yes | yes | **yes** |
| **immediate goto** ("jump NOW", not preloaded) | no | no | no — gapless only |

The only inherently non-frame-perfect path is **immediate goto** to an
un-preloaded target: physical open+decode latency means there is no
predetermined frame to be perfect *to*. It is still gapless + blackless (keep
the outgoing clip active until the target's edge is ready, then flip).

### Frame-perfect cut mechanism (preload + cued switch)

`source_switcher` selects its active input from a PTS-keyed `SharedTimeline`
channel (verified in `src/nodes/source_switcher.cpp`: with
`timeline_reference_input >= 0` it evaluates `tlGet<int>("active", ref.pts, …)`
against the reference input's frame PTS; the reference input must keep producing
every tick or scheduled switches never fire). A cut is scheduled by:

```text
timeline.set {"name":"seq_tl","channel":"sel","key":"active","at":<boundary_pts_ms>,"val":<worker_index>}
```

Because the decision is keyed on the frame's own PTS, a scheduled flip is
deterministic w.r.t. control latency — **but only if the target worker is
actually emitting a ready frame when the flip takes effect** (see the preload
freeze below). So the cut is a two-command sequence at the boundary, **unpause
first, then flip**:

```text
resume <workerB_pauseteam>                      # release the frozen incoming worker
timeline.set {"name":"seq_tl","channel":"sel","key":"active",...,"val":B}
```

The `pause` node is driven by its shared `PauseControlTeam`, not a per-node
object: `pause <team> now` freezes and `resume <team>` releases (verified in
`src/avplumber.cpp` command table and `src/nodes/pause.cpp`; this is exactly how
`demos/replay` drives its pause teams). Each worker gets its own pause team.

Unpausing *before* the flip guarantees B's already-decoded frame 0 is flowing on
the tick the switcher selects it, so the select never lands on an empty input.
This yields a **gapless** join. It is **frame-accurate to ~±1 frame**; whether
it is exactly zero-gap/zero-dup depends on the unpause-tick vs. flip-tick
alignment at runtime, which `test_frame_perfect.py` measures — we do not claim
provable frame-perfection from the node contracts.

### Worker preload / rebuild (edge reuse)

When a worker becomes free (its clip reached its end trigger):

1. edge wiretap on `<group>_out` detects the end (EOF marker frame
   `pts().isNoPts()`, or the scheduled `play_to`/`duration` boundary);
2. `group.stop <worker>` (clean finish via `ReportsFinishByFlag`);
3. `node.delete` the worker's nodes — **never** delete `<group>_out`;
   `EdgeManager::findInternal` reuses it by name;
4. `node.add` the worker's replay-chain nodes pointed at the next clip URL
   (`InputRec` with `preseek`=`play_from`, `loop`, `stop_delay`, `timeout`;
   `SpeedVideo` `speed`; `ForceFPS`; the worker's `Pause` **paused**), rewired to
   the same `<group>_out`;
5. `group.start <worker>` — it decodes then freezes on frame 0 at its `Pause`.

Downstream (`source_switcher` -> realtime -> output) is never torn down.

### Gapless preloading — no freeze between clips (crucial)

The switch must never wait on a decoder cold-start; a freeze at the boundary is
the primary failure mode. The whole point of the fixed pool of 2 is that the
**next clip is already decoding on the idle worker before the boundary**. The
invariant:

> At any moment there is exactly one *playing* worker feeding the switcher and
> one *preloaded* worker whose next clip is open, decoded, and **frozen (paused)**
> on its frame 0, ready to release the instant the cut fires.

**How a preloaded worker is held ready — verified from the C++.** Two dead ends
we ruled out by reading source:
(a) buffering inside the switcher's input edge fails — `source_switcher` pops
*every* input every tick and discards non-selected frames
(`src/nodes/source_switcher.cpp` 133–138);
(b) the mixer's `OneToMany outputs` gate does **not** hold a source — it pops
unconditionally (`one_to_many.cpp` line 31) and forces `drop_dynamic_` under a
timeline (61–65), so an inactive VOD clip would just play through and be
discarded. Likewise `realtime` has **no pause/hold** (only a `delay` object)
and cannot be told to start at a chosen PTS.
The primitive that actually freezes a stream is the **`pause` node**
(`src/nodes/pause.cpp`): while paused it `return`s without popping (lines
14–20), so its upstream decoder backpressures and the worker sits frozen on
already-decoded frames. So each worker carries a `pause` node; a preloaded
worker is paused, holding frame 0 decoded and waiting.

Preload lifecycle, decoupled from the switch:

1. As soon as a worker becomes the *current* one, the coordinator immediately
   picks `resolve_next()` and rebuilds+starts the **other** worker on that clip
   (the rebuild steps above), with that worker's `pause` node **paused**. This
   begins the moment playback of the current clip starts, not near its end — so
   there is a whole clip's worth of lead time.
2. The preloaded worker opens and decodes until its `pause` backpressures the
   decoder; it then sits **frozen on frame 0**, spending no CPU and discarding
   nothing.
3. The switch is *unpause the preloaded worker* then *`timeline.set`* the flip
   (order matters, see above). Because frame 0 is already decoded, the switcher
   reads a ready frame on
   the very next tick — **zero decoder latency at the cut**.
4. Only *after* the cut fires does the now-free ex-current worker get rebuilt
   for the *following* clip. So preload always runs during steady-state
   playback, never in the critical switch window.

Preload readiness gates the boundary. `_schedule_cut` is only armed once the
incoming edge reports a ready frame (wiretap/`peek` non-null). If a clip is slow
to open (network source), the boundary is deferred and the current clip holds
(or loops, per mode) rather than cutting to black — the switch waits for
readiness, it never cuts into an unprimed input.

### NEVER a black frame — hold last frame on unpreloaded / out-of-order jumps

A sequential `next` (and any cued jump to an already-preloaded target) is
frozen (paused) in advance and cuts with zero decoder latency — gapless and
frame-accurate (see caveat above). But
an **immediate / out-of-order jump** (`goto(index, when="immediate")`, or
`next`/`prev` to a clip the idle worker was not preloading) *cannot* be primed
in advance — the target must be opened and decoded on demand, so **some latency
is unavoidable** and the cut is gapless but not frame-perfect. That latency must
never surface as a black frame. Hard rule for this demo:

> **Black frames are forbidden outside of an explicit fade transition.** When
> the incoming clip is not yet ready, the outgoing clip keeps playing; if it too
> has ended, the output simply stops emitting new frames and the downstream
> decoder freezes on the last presented frame until the target's first real
> frame arrives. Under no circumstance is black emitted.

How it is guaranteed (based on verified C++ behavior — see "Backend
verification" below):

- **No node in the playlist graph synthesizes black.** `source_switcher` only
  ever emits frames it peeked from an input edge; when the active input has no
  frame it emits *nothing* that tick (it stalls), it does not produce black.
  `realtime` likewise emits nothing when its input starves. The only black/card
  generator in the tree is `sentinel`, which is a **live-input** node and is
  **not used here** (and must not be added). So black is impossible by
  construction as long as we never wire a black fallback and never add sentinel.
- A freeze on the last frame is therefore a *downstream display effect*: no new
  frames reach NVENC, RTP stalls, and the Janus/WebRTC decoder keeps showing the
  last decoded frame. We do not, and must not, rely on any node "repeating" the
  last frame.
- The switch remains a PTS-keyed `timeline.set`; it only *arms*, and
  `source_switcher` only *flips* active input, once the target edge has a ready
  frame (`_schedule_cut` gates on incoming-edge readiness). Until then the
  previous input stays active.
- On `goto`/random jump: (1) rebuild the idle worker onto the target with its
  `pause` node paused; (2) **keep the current worker active and playing** until
  the target's first frame is decoded (worker frozen-ready) — so real frames of
  the current clip keep flowing during the target's open+decode, not a freeze;
  (3) unpause the target worker, then flip. Only if the current clip has *also*
  ended before the target is ready does a true freeze gap appear, whose length =
  remaining target open+decode time, bounded by
  `source_ready_timeout`.
- If the target ultimately fails to open within timeout, we **never flip** to
  it, keep the outgoing clip active (or the decoder-held last frame if it ended)
  and surface an error in status — still never black.

This applies to hard cuts. During a real fade (follow-up) a momentary blend is
expected; that is the *only* sanctioned deviation from "hold last frame."

### Backend verification (read the C++ before relying on it)

The no-black guarantee rests on how the existing C++ nodes behave on
starvation. All three were read in full and confirmed:

- **`source_switcher`** (`src/nodes/source_switcher.cpp`): only ever `put`s a
  frame it `peek`ed from an input edge. When the active input has no frame it
  emits nothing that tick (it stalls), then advances what it can. It has **no
  black/blank generator**. `fallback_input_` defaults to **-1 (off)** and
  `fallback_when_active_missing_` only routes to a *configured fallback input*
  (another real source), never to black. → We keep no fallback configured; a
  starve is a stall, not a black.
- **`realtime`** (`src/nodes/realtime.cpp`): on starve, `peek(0)` returns
  `nullptr` and the node **returns without emitting** — it does *not*
  duplicate/repeat the last frame. (This corrects an earlier assumption that
  realtime holds-last-frame; it does not.)
- **`sentinel`** (`src/nodes/sentinel.cpp`): this *is* the hold-last-frame /
  backup-card primitive. It is a **live-input** node and is **excluded** from
  the playlist graph by explicit instruction; it must not be used or altered.

Conclusion: "never black" is achieved by *construction* — there is no black
source in the graph — plus keeping the outgoing clip active until the target is
ready. The freeze a viewer sees on a true gap is the downstream decoder holding
its last picture, not a node output. **No C++ change is required or requested;**
if a scenario ever appears to need one, stop and prompt (per hard constraints).

### Mixed-fps / mixed-resolution content — normalize to output

Clips will have different frame rates and dimensions (24/25/30/60 fps, SD/HD).
The switcher output must be a single, stable cadence and size, so **each worker
normalizes to the output format before the switcher**, on the worker's own
thread — the switcher and everything downstream only ever see uniform frames.

Per worker, after decode:

- `scale` / `rescale_video` (or `scale_cuda`) -> the fixed output WxH (with
  aspect handling: letterbox/pad or crop, configurable; default letterbox so we
  never distort).
- `force_fps` -> the fixed output fps. This is the **same `ForceFPS` node the
  replay/transcode graphs already use** (drops/duplicates frames to retarget
  cadence), so mixed input fps converges to one output fps.

Because normalization happens per worker *upstream* of `source_switcher`, both
edges present frames on the same grid; the PTS-keyed cut lands on an output-fps
frame boundary and the realtime node reclocks a single steady cadence out to
NVENC. This also keeps the boundary math simple: `play_to`/`duration` boundaries
are computed in output-fps frames.

Output format (WxH + fps) is a launch parameter (with a sane default), not
inferred per clip. Per-clip native fps/size is informational only.

> Constraint (per your instruction): this reuses the **existing** `ForceFPS` /
> `rescale`/`scale_cuda` nodes and their current parameters as used in
> `demos/replay`. If faithful mixed-fps normalization turns out to need a
> change to a decoder/framework/node's behavior or exposes a regression, I will
> **stop and prompt you** rather than modifying framework/decoder output
> semantics on my own.

Tunables mirrored from the switcher: `source_preload_time` /
`live_source_preload_time` (how early to start preload — for the PoC "as early
as possible" = at current-clip start) and `source_ready_timeout` (how long to
wait for a slow source before surfacing an error). Edge `planCapacity` on the
preloaded worker's edge *upstream of its `pause` node* sets how many decoded
frames sit primed behind the paused node (the `<group>_out` edge downstream of
`pause` stays empty for the inactive worker, since the switcher would otherwise
pop and discard it).

Pool sizing: 2 is sufficient for hard cuts (one playing + one frozen-primed) and
makes **sequential playback gapless and frame-accurate** (the natural next is
always the primed worker). A fade needs both endpoints live simultaneously *and*
somewhere to preload the clip after the fade, hence pool of 3 (see
Transitions/fade below).

Preload depth (`preload_depth`, default 1). Depth 1 = pool of 2 = only the
immediate next is frozen-primed, so only *sequential*/`next` and a cued jump to
that same clip get the zero-latency (frame-accurate) join; any other jump is
gapless-but-higher-latency. Raising depth (pool = depth+1) freezes several
upcoming targets at once, so a **scripted playlist with known switch points**
gets the primed join end to end. This is a memory/GPU-footprint lever, not a new
mechanism — the freeze + cut path is identical for each preloaded worker. For
the PoC we ship
depth 1 (gapless guaranteed everywhere; zero-latency frame-accurate join for the
realistic sequential/cued cases) and expose `preload_depth` for stress scenarios.

## Add / remove / reorder clips — effort: small

The switcher's `switcher_update` (lines 928-1016) already defines a clean
diff-merge that we port to plain Python list ops on the coordinator's `clips`
list. Because graph topology is fixed by pool size, editing the list never
touches the graph. The only real logic is two guard cases lifted straight from
the C++:

- removed clip == the preloaded `next` -> cancel that worker's preload and
  re-preload `resolve_next`;
- removed clip == `current` -> force an immediate cut to `resolve_next`.

`reorder` and `insert` just mutate the list and, if they change what `next`
should be, re-preload the idle worker. No node rebuild is forced unless the
*currently preloaded* target changed.

Fixed-size note: "fixed size" means the **worker pool** (2), not a cap on the
playlist. The clip list stays growable, matching the OBS/gateway behavior.

## Coordinator (proposed `demos/playlist/playlist.py`)

Mirrors `replay.py` structure (dataclasses + one controller + a graph builder).

```python
class PlaylistMode(str, Enum):      # level 1
    PLAY_ALL = "PlayAll"
    PLAY_CURRENT = "PlayCurrent"
    LOOP_ALL = "LoopAll"
    LOOP_CURRENT = "LoopCurrent"

    @property
    def loop(self) -> bool: ...             # LoopAll, LoopCurrent
    @property
    def current_clip_only(self) -> bool: ...# PlayCurrent, LoopCurrent

class ElementMode(str, Enum):       # level 2 hand-off trigger
    PLAY_TO_END = "PlayToEnd"        # media_state_switch: advance on EOF/play_to
    TIMED = "Timed"                  # time_switch + duration: advance after N ms
    LOOP_SELF = "LoopSelf"           # looping: media loops; only manual/goto ends

@dataclass
class Clip:                          # level 2 (PLAYOUTS_PlaylistSource subset)
    url: str
    name: str = ""
    element_mode: ElementMode = ElementMode.PLAY_TO_END
    play_from_ms: int = 0
    play_to_ms: int | None = None    # None -> natural media end (PlayToEnd)
    duration_ms: int | None = None   # required when element_mode is Timed
    disabled: bool = False
    speed: float = 1.0
    # derived for the worker/switcher:
    #   PlayToEnd -> media_state_switch=True
    #   Timed     -> time_switch=True, time_switch_duration=duration_ms
    #   LoopSelf  -> looping=True

@dataclass
class PlaylistStatus:                # what the TUI renders
    mode: PlaylistMode
    playing: bool
    current_index: int | None
    next_index: int | None
    clip_position_ms: int
    clip_duration_ms: int | None
    clips: tuple[Clip, ...]
    ready: bool
    message: str = ""
    error: str = ""
    last_command: str = ""

class PlaylistController:
    # transport
    def play(self): ...
    def pause(self): ...
    def toggle(self): ...
    def stop(self): ...
    def next(self): ...              # resolve_next(+1); target is preloaded -> frame-perfect
    def prev(self): ...              # resolve_next(-1); usually not preloaded -> gapless goto
    def goto(self, index, *, when: str = "immediate", at_pts_ms: int | None = None): ...
        # when="immediate": jump NOW; if target not preloaded -> gapless, not frame-perfect
        #   (keep outgoing active until target edge ready, then flip)
        # when="ready":     preload target first, cut the instant it is frame-locked -> frame-perfect
        # when="at":        preload target, schedule cut at at_pts_ms -> frame-perfect
    # playlist edits (small)
    def append_clip(self, clip): ...
    def insert_clip(self, index, clip): ...
    def remove_clip(self, index): ...          # applies the two guard cases
    def reorder_clip(self, src, dst): ...
    def set_disabled(self, index, disabled): ...
    def set_element_mode(self, index, mode: ElementMode, duration_ms=None): ...
    # modes
    def set_mode(self, mode: PlaylistMode): ...   # playlist-level (level 1)
    # internal
    def _resolve_next(self, direction=+1) -> int | None: ...
    def _preload(self, worker_index, clip): ...  # rebuild worker, pause it (freeze on frame 0)
    def _pause_worker(self, worker_index, paused: bool): ...
        # pause <worker_pauseteam> now / resume <worker_pauseteam>
        #   (team-controlled; verified src/avplumber.cpp + src/nodes/pause.cpp)
    def _cut(self, worker_index, boundary_pts_ms=None): ...
        # unpause target worker FIRST, then timeline.set the active flip (order matters)
    def _on_worker_eof(self, worker): ...       # wiretap callback (realtime isEof / EOF marker)
    def status(self) -> PlaylistStatus: ...
```

Each worker owns a `pause` node (paused while preloaded). `_cut` always
**unpauses the incoming worker before flipping `active`** so the switcher never
selects an unproduced input. A `play`/`pause`/`toggle` at the transport level
pauses/unpauses the *currently active* worker's `pause` node.

## API verb mapping (gateway -> controller)

| Gateway `processPlayerAction` verb | Controller method |
|---|---|
| `playouts.playPause` | `toggle()` |
| `playouts.stop` | `stop()` |
| `playouts.playlists.sources.play` | `goto(index)` (immediate; gapless, frame-perfect only if that index was the preloaded next) |
| `playouts.playlists.sources.play` + cue (PoC) | `goto(index, when="ready"/"at")` (frame-perfect cued jump) |
| `playouts.playlists.sources.play.next` | `next()` |
| `playouts.playlists.sources.play.prev` | `prev()` |
| `playouts.playlists.sources.add` | `append_clip()` |
| `playouts.playlists.sources.remove` | `remove_clip()` |
| `playouts.playlists.sources.reorder` | `reorder_clip()` |
| `playouts.playlists.sources.disable.set` | `set_disabled()` |
| `playouts.playlists.sources.cue.set` | clip `play_from` |
| `playouts.playlists.sources.duration.set` | clip `duration_ms` |
| `playouts.playlists.sources.mode.set` (PoC verb) | `set_element_mode()` (level 2) |
| `playouts.playlists.mode.set` | `set_mode()` (level 1) |
| `playouts.playlists.transition.set` | out of PoC scope (cuts only) |

`sources.mode.set` is a PoC-local verb (the gateway spreads level-2 across
several `sources.*` verbs); it sets one clip's `ElementMode` and optional
`duration_ms`.

### Transport (matches gateway's single toggle)

The gateway exposes one `playouts.playPause` toggle plus `playouts.stop`. The
TUI's **TOGGLE** button and Space key both map to `toggle()` -> `playPause`;
explicit **PLAY**/**PAUSE** buttons are convenience wrappers over
`play()`/`pause()` and are optional.

## TUI design (`demos/playlist/player.py`, Textual)

Single-playlist view; reuses the replay TUI shell (Header/Footer, 0.2s status
refresh, `_execute` wrapper, thread-safe notify). One-column layout:

An interactive HTML stand-in for iterating on this layout lives at
`demos/playlist/tui-mockup.html` (serve it and click through the controls).

```text
+ AVPlumber Playlist -> Janus -----------------------------------------------+
| JANUS 127.0.0.1:5004 PT=96 SSRC=0x41565001   MODE=LoopAll   > PLAY         |
| clip 2/5  intro.mp4   00:07.3 / 00:12.0                                    |
| last: sources.play.next            <error line if any>                     |
+---------------------------------------------------------------------------+
| PLAYLIST (DataTable, current row highlighted, NEXT row marked)            |
|  st  #  name           element mode   from    to/dur    spd               |
|  ··  1  bumper.png     [Timed]        0.0     3.0(dur)  1.0               |
| >>>  2  intro.mp4      [PlayToEnd]    0.0     end       1.0   <- current   |
|  ··  3  ad_break.mp4   [PlayToEnd]    (disabled)                          |
|  ->  4  sports.mp4     [LoopSelf]     2.0     end       1.0   <- next      |
|  ok  5  outro.mp4      [PlayToEnd]    0.0     end       1.0               |
+---------------------------------------------------------------------------+
| transport:  [<> TOGGLE][# STOP][|< PREV][>| NEXT]   (PLAY/PAUSE optional)  |
| mode:       [PlayAll][PlayCurrent][LoopAll][LoopCurrent]                   |
| edit: [+ ADD][- REMOVE][^ UP][v DOWN][o EN/DISABLE][<> ELEM MODE][>> PLAY] |
+---------------------------------------------------------------------------+
```

Textual "icons" (unicode glyphs on Buttons, matching replay's `>`/`||` style):

- Transport: `<> TOGGLE` (= Space = gateway `playPause`), `# STOP`,
  `|< PREV`, `>| NEXT`. Optional convenience `> PLAY` / `|| PAUSE`.
- Mode (level 1): `PlayAll` / `PlayCurrent` / `LoopAll` / `LoopCurrent`,
  radio-like, active highlighted.
- Edit: `+ ADD`, `- REMOVE`, `^ UP` / `v DOWN` (reorder),
  `o EN/DISABLE`, `<> ELEM MODE` (cycle level-2 mode of selected row),
  `>> PLAY SELECTED` (goto).
- Per-row status glyph: `>>>` current, `->` next/preloaded, `··` disabled,
  `ok` idle.
- Element-mode cell renders a colored pill (`PlayToEnd` / `Timed` / `LoopSelf`).

Widgets:

- `DataTable` (id `playlist`) — clip list; selected row is the edit target;
  current/next rows marked via the status glyph + row style. The **element-mode
  column is interactive**: activating that cell (Enter / cell-selected), the
  `<> ELEM MODE` button, or the `m` key cycles the selected clip's `ElementMode`
  PlayToEnd -> Timed -> LoopSelf (switching to Timed seeds a default duration).
- Button rows: transport, mode, edit.

Key bindings: `space` toggle, `n` next, `p` prev, `m` cycle element mode of
selected row, `Delete` remove selected, `a` open add dialog, `Up`/`Down` move
DataTable cursor, `q` quit. The larger edit actions live on buttons.

### ADD -> file-selector modal

`+ ADD` opens a Textual `ModalScreen` with a `DirectoryTree` (Textual's
built-in filesystem browser) rooted at the configured media directory, plus a
manual URL `Input` (for `rtsp://`/`http://` sources the tree can't show), an
**element-mode selector** (PlayToEnd / Timed / LoopSelf), and the per-element
fields (`play_from`, `speed`, plus `duration` shown only for Timed). The modal
returns a `Clip`, which the app appends (or inserts after the selected row).
This mirrors the file-picker pattern used in the mixer examples; the
`DirectoryTree` is filtered to media extensions.

```text
+ ADD CLIP -------------------------------------------------+
| media dir: /media                                         |
|  > /media                                                 |
|    - bumper.mp4                                            |
|    - intro.mp4                                             |
|    > ads/                                                  |
| or URL: ( rtsp://.................................. )     |
| element mode: [PlayToEnd][Timed][LoopSelf]                |
| from ( 0.0 )  dur ( 5.0 )*  speed ( 1.0 )   *Timed only   |
|                                  [ CANCEL ]  [ + ADD ]     |
+-----------------------------------------------------------+
```

Refresh: `set_interval(0.2, refresh_status)` reads `controller.status()`,
rebuilds the DataTable rows (or updates in place), sets the MODE/transport
header, and disables edit buttons when `not status.ready`.

## Directory shape (proposed)

```text
demos/playlist/
  playlist.py        # Clip, PlaylistMode, PlaylistStatus, PlaylistController,
                     # build_playlist_application, Janus output (reused), pool
  player.py          # Textual TUI + --no-tui + arg parsing (mirrors replay)
  requirements.txt   # textual (same pin as replay)
  README.md
  tests/
    test_modes.py        # 4-mode x resolve_next truth table; ElementMode triggers
    test_controller.py   # transport + edit verbs, the two remove guard cases
    test_graph.py        # pool build, edge names, cut scheduling command text
    test_sequencer.py    # preload/rebuild edge-reuse + pause-freeze/unpause ordering against a fake command sink
    test_gapless.py      # gapless (hard req): every transition continuous, no gap/dup
    test_frame_perfect.py # frame-perfect where preloaded+known-boundary; goto degrades gracefully
    test_no_black_frame.py  # never-black guarantee (see below)
    test_mixed_fps.py    # per-clip cadence normalization to output fps
```

### `test_no_black_frame.py` (explicit, load-bearing)

These encode the "never black outside a fade" rule so it can't regress:

- **sequential cut is gapless**: preloaded `next()` -> observed frames are
  contiguous across the boundary, no repeated-forever/null frame.
- **out-of-order goto keeps the outgoing clip playing until ready**:
  `goto(far_index)` on an un-preloaded target -> the outgoing worker stays
  `active` and keeps emitting *its own real frames* while the target opens; the
  active flip only happens once the incoming edge has a ready frame; after
  ready, frames become the target's. No black is emitted at any point.
- **switch arms only on readiness**: with a stubbed slow source, assert the
  `timeline.set`/active flip is not issued until the incoming edge reports a
  ready frame (fake command sink + fake edge `peek`).
- **source open failure**: target that never opens -> the active input is never
  flipped, so the outgoing clip keeps playing (or, if it too has ended, the
  output simply stops receiving new frames and the downstream decoder holds the
  last decoded frame); after `source_ready_timeout` status carries an error.
  At no point is a black frame emitted.
- **no black source in the graph** (the load-bearing invariant): a static
  assertion over the built graph that it contains **no** `sentinel` node and no
  black/blank generator, and that `source_switcher` has no black `fallback`
  configured. This is what actually guarantees "never black" — verified from
  the C++: `source_switcher`/`realtime` emit *nothing* on starve, they never
  synthesize black (see "Backend verification").

Assertions use the `PositionProbe`/observed-frame stream (as in the replay
demo's `observed_frames_since`) plus, for integration, a cheap "is this frame
black" check (mean luma above a floor) on a captured frame at/after the
boundary. Note the freeze on a true starvation gap is a **downstream display
effect** (no new RTP frames -> Janus decoder holds the last picture), not
something any node emits — so the integration check verifies the last *emitted*
frame was non-black and that no black frame is ever pushed.

## Optional: simple fade — how much complexity?

Question raised: what does adding a plain crossfade cost on top of the hard-cut
PoC? Answer: **moderate, and mostly reuse** — because the transition scheduling
is already PTS-keyed exactly like our cut, and `MixerGraphBuilder`
(`pyplumber/mixer.py`) already implements the blend. We do not write blending
code; we reuse `builder.fade(scene, start_pts_ms=...)`.

Reuse from the existing mixer (`pyplumber/mixer.py`,
`demos/mixer/*`, `src/mixer_orchestrator.cpp`):

- `MixerGraphBuilder.add_source()` per worker edge, `add_scene()` for the two
  crossfade endpoints, `build()` once at startup.
- `fade(scene, start_pts_ms=...)` / `cut(scene, start_pts_ms=...)` — same
  PTS-keyed `SharedTimeline` scheduling our hard cut already uses, so the
  frame-perfect timing property carries over unchanged.
- The mixer status/scene plumbing (`MixerProgramSceneReader` pattern from the
  talkshow project) for reporting the active scene.

What actually changes (the added complexity):

1. **Output path swaps selector -> mixer** ("mixer-primary"): the mixer is
   always the output; a hard cut becomes `cut(start_pts_ms)` and a fade becomes
   `fade(start_pts_ms)`. Uniform code path, but every playout (even plain cuts)
   now goes through the mixer image. This is the main structural change.
2. **Static topology, one input per worker**: the mixer is build-once; sources
   are fixed at startup for the pool. Only *scheduling* calls happen at runtime
   (fits our fixed-pool design already).
3. **Pool grows 2 -> 3**: during a fade both endpoints must render for the fade
   duration, so we need a third worker to preload the clip *after* the fade
   while two are still mixing. Preload logic is unchanged, just one more slot.
4. **GPU requirement**: `cuda_wipe`/blended `fade` want the CUDA/python image;
   pure `cut` works CPU-only. So fades make the demo GPU-only.
5. **Unpause timing differs from a cut**: for a cut we unpause the incoming
   worker *at* the boundary; for a fade the incoming clip must be **live and
   producing for the whole fade**, so we **unpause it `duration` before** the
   fade's end point and let both feed the blend. Same `pause` primitive, earlier
   release.
6. **Reconcile `pause`-preload with the mixer's realtime/discard model** (the
   real work): `MixerGraphBuilder` is built for *live cameras that legitimately
   run in real time*, and its inactive `OneToMany` slots **discard** frames
   (verified `one_to_many.cpp`). A frozen VOD worker cannot simply be an idle
   mixer slot or it would play through and be dropped. So the frame-lock +
   preload stays *upstream* of the mixer (our per-worker `pause`), and a worker
   is handed to a mixer source only when it is unpaused and producing. This
   integration — not the blend — is the bulk of the fade effort.
7. **Coordinator abstraction**: one `schedule_transition(kind, at_pts_ms,
   from_idx, to_idx)` that emits either a `timeline.set` (selector cut) or a
   `builder.fade(...)`/`cut(...)`. TUI gains a fade duration field + a
   cut/fade toggle (the mockup already reserves the transition row).

Bottom line: no new blending code and no new scheduling model — the added work
is the selector->mixer output swap, pool 2->3, the GPU image, the earlier
unpause, and reconciling the `pause`-based preload with the mixer's realtime
slot model. It is a clean follow-up, not a rewrite, which is why the PoC keeps
it out of v1 but leaves the seams for it. **Verdict: nice-to-have, moderate
effort, all reuse** — fades stay a **Phase 2 follow-up** (need a GPU spike).

## Scope / non-goals (PoC)

- Video-only, hard cuts only (selector-primary). No mixer, no fade/wipe, no GPU
  requirement for switching (encoder still uses whatever the replay path uses).
- Single playlist, single output, single Janus mountpoint.
- No audio sequencing; no gapless single-PTS timeline (each clip keeps its PTS
  domain; realtime reclocks) — same non-goals as the parent design.
- Transitions (`transition.set`) are explicitly deferred to a follow-up that
  swaps `source_switcher` for `MixerGraphBuilder` (`cut`/`fade`, already
  PTS-scheduled) on a GPU build — see "Optional: simple fade" above for the
  cost estimate.

## Reuse map (do not reinvent)

To keep the PoC minimal, everything below is reused, not rewritten:

| Need | Reused from |
|---|---|
| Janus RTP output group (`EncVideo h264_nvenc -> Bsf -> Mux -> Output rtp`) | `demos/replay/replay.py` (`build_player_application`, `_rtp_url`) |
| PLI/FIR keyframe handling | `demos/replay` `RtcpFeedbackListener` |
| Position/frame observation | `demos/replay` `PositionProbe` |
| App start/stop lifecycle, `_add_node`, `_init_cuda`, `planCapacity` | `demos/replay/replay.py` |
| TUI shell (Header/Footer, 0.2s refresh, `_execute`, notify) | `demos/replay/player.py`, `demos/mixer/tui.py` |
| Gapless cut (PTS-keyed timeline flip) | validated parent design; `src/nodes/source_switcher.cpp` |
| Whole per-worker chain (InputRec->Demux->DecVideo->SpeedVideo->scale->ForceFPS->Pause->Realtime) | `demos/replay/replay.py` 727–751 — replicated per worker |
| Freeze/hold a preloaded worker on frame 0 (backpressure) | `Pause` node — verified `src/nodes/pause.cpp`; replay already uses two (`replay_transition_gate`, `replay_pause`) |
| Per-clip speed | `SpeedVideo` on a `team` + `speed.set <team>` — as replay's `SPEED_TEAM` |
| Cadence + resolution normalize | `ForceFPS(fps=out/1)` + `scale_cuda`/`rescale_video` — replay's `replay_fps` |
| Continuous monotonic output clock across clips | single post-switch `RealtimeVideoFrame set_pts=True,tick_period=1/out` — verified `realtime.cpp` 268–273; as replay's `replay_realtime` |
| EOF / boundary detection | `realtime` `isEof()` / EOF marker + `getCurrentFrameTimestamp` — verified `realtime.cpp` 252–257, 386–397 |
| Crossfade/wipe blending + PTS scheduling (fade follow-up) | `pyplumber/mixer.py` `MixerGraphBuilder`, `src/mixer_orchestrator.cpp` |
| Mixer status/active-scene reporting | `MixerProgramSceneReader` pattern (talkshow `control_status.py`) |
| Clip diff-merge / next-prev / preload semantics | ported from `mse-source-switcher.cpp` |

## Risks

| Risk | Mitigation | Status |
|---|---|---|
| Cut lands on wrong frame | PTS-keyed timeline, not wall-clock | Validated (parent) |
| Edge lost on worker rebuild | reuse `<group>_out` by name | Validated (parent) |
| Selector starves during rebuild | downstream never torn down; other worker feeds | Validated (parent) |
| play_to/duration boundary PTS mapping | derive boundary PTS from element start + trim/duration; wiretap confirms EOF marker | PoC to validate |
| Remove of current/next clip races the cut | apply the two switcher guard cases under the controller lock | PoC to validate |
| **Freeze/black at clip boundary** | preload next clip on the idle worker at current-clip *start*, frozen on frame 0 via a paused `pause` node (decoder backpressures); `_cut` unpauses before flipping so a slow source defers the boundary instead of cutting to black | PoC to validate |
| Unpause/flip tick misalignment -> 1-frame gap or dup at cut | unpause **before** flip; `test_frame_perfect.py` measures actual seam; claim is gapless + frame-accurate (not provably frame-perfect) | PoC to validate |
| `pause` doesn't backpressure decoder as assumed (frames still advance) | verified `pause.cpp` returns-without-pop while paused; validate on bench that the worker truly holds frame 0 | PoC to validate |
| Slow-opening network source blows the boundary | `source_ready_timeout`; hold/loop current until primed, then cut | PoC to validate |
| **Out-of-order/random jump can't be preloaded -> freeze** | freeze is accepted and bounded by open+decode time; must show as **held last frame, never black** | PoC to validate |
| **Black frame outside a fade** (hard no-go) | no black/card source exists in the graph (no `sentinel`, no black `fallback`); switch only flips once target edge is ready; on any starve `source_switcher`/`realtime` emit *nothing* (never black) and the downstream decoder holds the last picture; failure keeps outgoing clip active + status error | **must be covered by tests (incl. "no black source in graph" static check)** |

No C++ changes required.
