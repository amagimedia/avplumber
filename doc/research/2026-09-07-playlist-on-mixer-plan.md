# Playlist demo on the mixer engine

Planned 2026-09-07. Rebuild `demos/playlist` so it follows the mixer
architecture one to one: the same input chains, the same `MixerGraphBuilder`
and native `MixerOrchestrator`, the same Janus output path, the same recording
and publishing procedure. The playlist keeps only its policy layer.

## Why

The current playlist drives a bare `source_switcher` from Python with an
immediate `active` set after the previous clip has already ended. Findings from
the review:

- No pre-roll. The next clip is opened, decoded and switched only after EOF is
  observed through a 50 ms poll, so every automatic transition freezes the
  output for the full decoder start-up.
- The switcher pops frames from all inputs, so the one preheated frame is
  discarded and clips start several frames after cue-in.
- Slots are stopped after use; every play reopens the file and re-creates the
  NVDEC decoder.
- Readiness is measured as "two frames on the shared output", regardless of
  which clip produced them.

The mixer already solves each of these: preheated PVW slot, wallclock-scheduled
native cut with a fresh-frame readiness condition, fade and media wipe with
instant interruption, and a measured click-to-picture path.

## Acceptance criteria

Ordered by importance, per the owner's direction.

1. **No black or frozen frames between clips.** Automatic transitions are
   scheduled natively at the outgoing clip's cue-out wallclock time. The
   incoming clip's cue-in frame is the first frame after the cut. Verified by
   decoding program output and reading the frame counters burned into the
   generated clips: the counter sequence across a boundary must be
   `..., last_of_A, first_of_B, first_of_B + 1, ...` with no repeat and no gap,
   on every boundary of a full playlist pass.
2. **Click-to-output latency.** Manual ITEM PLAY, Next, Previous, Pause and
   Stop are measured with the mixer's method (`measure_click_latency.cjs`) and
   published like `demos/mixer/docs/latency.md`. Target: same order as the
   mixer's warm Cut (about 190 ms median on a T4), because the path is the same.
3. **Transitions.** Per item: Cut (default), Fade, Media Wipe, with duration.
   A manual action during a transition interrupts it instantly, as in the mixer.
4. **Uncached clip.** A clip that was never played or was edited pays only its
   own decoder start-up, overlapped with the clip currently on air, never on
   the output.

## Architecture

### Sources and slots

Sixteen fixed source slots are registered on the builder before `build()`,
because `add_source` is closed after build. Slot `i` owns input group
`pl_item_i` and a fixed pre-OTM edge `pl_item_i_fps`. A playlist item is bound
to a slot; editing an item rebuilds its chain in the same group (the mixer's
OTM lives in the input group and restarts with it). This replaces today's
generation-suffixed node names.

Per slot, the chain is the mixer's `_build_input` plus the playlist's controls:

```
input_rec(url, start_ts=cue_in, stop_ts=cue_out, loop=LoopSelf,
          pause_team=pl_item_i_pause, timestamp_source=wallclock, send_eof)
  -> demux -> dec_video(cuda, h264_cuvid)
  -> speed_video(speed, team=pl_item_i_speed)
  -> realtime(set_pts=true)          # rebases to the shared wallclock
  -> force_fps(fps)                  # output grid
  -> [mixer OTM, slot A / slot B]    # created by MixerGraphBuilder
```

`realtime(set_pts)` before the mixer gives every slot the common time base the
native scheduler keys on. It is exactly where the mixer places it.

### Scenes

One scene per slot, `item_i`, full canvas, `fit: contain`. No layouts, no
router. With at most 16 slots the compositor mask is not a constraint.

### Item lifecycle

| Item state | Graph state |
| --- | --- |
| Cold | Chain nodes exist, group stopped. Only for slots never bound. |
| Parked | Group running, paused on cue-in frame, decoder resident. Default after Stop, after leaving air, and after load. |
| Preview | Parked and loaded into the PVW slot via `mixer.preview(item_i)`. Its cue-in frame is visible in PVW. |
| On air | Resumed, its scene is PGM. |
| Paused | On air but the pause team is paused; last frame holds. |

Leaving air never stops the group. The item is re-parked: pause, then
`seek <team> now <cue_in>`, the same input control the replay demo uses. Play
on a parked item is therefore a resume plus a cut, both warm.

### Automatic transition (the frame-perfect path)

When item A goes on air at wallclock `T_a`:

1. Compute `T_end = T_a + (cue_out_A - cue_in_A) / speed_A`, or `T_a +
   duration_A` for Timed. For PlayToEnd without cue-out, use the container
   duration probed once at load (the input node exposes it; fallback to
   `ffprobe` at playlist load). LoopSelf schedules nothing.
2. Resolve B with `resolve_automatic` and preview it: `mixer.preview(item_B)`.
   B is parked on its cue-in frame, so PVW shows that frame. Nothing is decoded
   ahead and nothing is dropped.
3. Arm the transition natively, ahead of time:
   `mixer.cut(item_B, start_pts_ms=T_end)` or `fade`/`wipe` with the item's
   duration, ending at `T_end`. The orchestrator's ready-cut task then owns
   the switch; Python is not on the critical path.
4. Schedule `resume pl_item_B_pause` at `T_end` minus the measured pipeline
   lead so that B's cue-in frame reaches the compositor as the first frame at
   or after `T_end`. The cut requires a fresh frame on the incoming edge, so
   an early resume by one grid step at most costs one B frame, never a repeat
   or black frame. The lead is calibrated once at startup from the first
   preheat and stored; the regression run asserts the boundary counters.
5. On the orchestrator's completion, mark B active, re-park A, and go to 1.

EOF from A remains a fallback trigger only. If it arrives before `T_end` the
scheduled cut is pulled forward to "now plus switch margin", and the
regression counts that as a timing miss.

Python side, the `resume ... at` form does not exist today; `pause <team> at`
does. Either add `resume <team> at <ts>` to the pause team (small C++ change,
symmetric with the existing pause-at) or schedule the resume on the backend
worker with a monotonic timer. The native form is preferred for jitter.

### Manual actions

- ITEM PLAY, Next, Previous: preview the target if it is not already in PVW,
  resume it, `mixer.cut` now (start at `now + switch_margin_ms`). Fade or wipe
  if configured for the incoming item.
- During a transition: the new command interrupts it, mixer semantics.
- Pause: pause team pause now. Resume: resume. Stop: park.
- Edit active item: rebuild chain in its group, park, and re-cut to itself.

### Output

The Janus output is the mixer's `_build_janus_output` (force_fps, keyframe on
PLI, assume format, NVENC, `dump_extra`, RTP mux, RTCP listener). Move it and
`_build_input` from `demos/mixer/mixer.py` into `avpmixer` so both demos import
one implementation instead of carrying copies.

### Code layout

| File | Change |
| --- | --- |
| `demos/playlist/playlist.py` | Keep policy and controller. Drop `plan_item_nodes`/`plan_switch_nodes`; add transition type and duration to `Clip`; add `scheduled_end_ms` to status. |
| `demos/playlist/playlist_app.py` | Rewrite backend on `MixerGraphBuilder`. Slot pool, parking, native scheduling, orchestrator completion events. |
| `avpmixer/inputs.py`, `avpmixer/janus.py` | Extracted from the mixer demo, shared. |
| `src/...` pause team | Optional `resume <team> at <ts>`. |
| `demos/playlist/player.py` | TUI gains transition selector per item and a countdown to the scheduled cut; poll loop stays for status only. |
| `demos/playlist/tests/` | Controller tests unchanged. Backend tests against a fake builder recording `preview`/`cut`/`fade` calls and their `start_pts_ms`. |
| `demos/playlist/regression.py` | Add boundary counter check and click latency run; publish JSON like the mixer's `latency-current.json`. |

## Publishing, same as the mixer

1. Side-by-side recording, program left, TUI right, 1600x900, 60 fps if the
   playlist runs 60 fps clips, following the procedure in
   `demos/mixer/docs/guide.md` "Demo recording": encoded program output plus a
   separate browser capture of the TUI. Content: a full pass with Cut, one
   Fade, one Wipe, a manual Next during a fade, Pause, an edit of cue points,
   a random ITEM PLAY on a cold clip.
2. `demos/playlist/docs/index.html` with chapters, download link and graph
   figure, from the replay/mixer page template; `media.sha256`.
3. WebUI captures: grouped overview and the full ungrouped graph (16 chains
   plus mixer internals is wide). Two assets like `mixer-graph-ungrouped.png`.
4. Release `playlist-demo-media-2026-09` for the MP4, poster and graphs.
   `playlist` entry in `demos/graph.html`, which now has a demo dropdown.
5. README rewritten around the mixer requirements and Docker base.

## Verification plan

- Local: pytest suites for controller and fake-builder backend.
- T4: full playlist pass with counter check on every boundary, 0 repeats and
  0 gaps; regression matrix for all four playlist modes and three item modes;
  interruption of fade and wipe by Cut; edit-while-playing; latency samples.
- Record numbers in `docs/` JSON files with the same schema the mixer uses.

## Decisions (owner, 2026-09-07)

- Generated clips stay 1920x1080 at 30 fps.
- One playlist-wide transition setting (Cut, Fade, Wipe with duration) for
  now; per-item override deferred.
- Add native `resume <team> at <ts>` to the pause team.

## Implementation status (2026-09-07)

Done locally, 74 unit tests green, not yet run on the NVIDIA host:

- `src/PauseControlTeam.hpp`, `src/avplumber.cpp`: `resume <team> at <ms>`;
  an explicit pause cancels a scheduled resume.
- `avpmixer/inputs.py`, `avpmixer/janus.py`, `avpmixer/control.py`: shared
  decode chain, RTP output and TCP client (the mixer demo keeps its own copies).
- `demos/playlist/playlist.py` policy with wallclock scheduling;
  `engine.py` mixer-backed backend; `control.py` JSON protocol;
  `server.py` backend process; `player.py` redesigned TUI.
- Fixtures carry a 32-bit frame code strip; `tests/verify_recording.py`
  checks every boundary of a `--record` run.

Also done without the host: `regression.py` (8 live checks, passes against the
in-memory backend), `tests/verify_recording.py` proven on synthetic splices
(clean PASS; injected repeat, late cue-in and fade blend detected), recording
harness (`tests/capture_tui.cjs`, `tests/record_demo.py`,
`tests/compose_recording.sh`), README, `docs/index.html` with placeholders for
the `playlist-demo-media-2026-09` release, fresh `docs/tui*.svg`.

First T4 run (2026-09-07 21:26 UTC): backend started, first element on air,
Janus RTP alive. The armed second element failed with "no frame decoded":
a chain started paused decodes nothing, and the mixer input chain's
`auto_restart: group` restarted element one at EOF. Fixed locally: chains are
primed to their first frame then parked, every chain loops between cue points
with no auto-restart, and a failed arm falls back to an immediate cut. Not yet
re-run on the host.

Deviation from the plan: EOF is no longer used at all. The scheduled end time
is authoritative and elements whose length cannot be probed report an error
until a cue-out is set. Elements that leave air are parked, never stopped.

## Second T4 run (2026-09-07 22:10 UTC)

After the review fixes (sync-team seek for parking, native transition armed
600 ms before its start, immediate takes with `start_pts_ms=-1`,
`mixer.interrupt` on disarm, on-air confirmed from `mixer.status`): the
automatic LoopAll pass ran through all five elements repeatedly with every
ready cut firing at its scheduled second (waited_ms=20 in the mixer log), no
errors, and the owner drove the TUI live over ttyd. Fixed on the way: the
`playlist.status` reply lacked the trailing newline the control protocol
uses to terminate a 201 response, so the TUI client timed out.

Not done yet because the GPU was handed to another task: the boundary
verification of a recorded program, the side-by-side recording, WebUI graph
captures and release assets.

## Verification and publishing (2026-09-08)

Measured on the T4 with `tests/verify_recording.py` over a recorded 50 s
LoopAll pass (1,509 frames, 0 unreadable): cue points inside a file land
exactly (element 3: frame 60 in, frame 239 out); elements cued at 0 land on
frame 0 or 1 depending on the mixer's ready-cut race (it fires on the first
fresh incoming frame and switches on the next one; for a cue at 0 the consumed
frame must come from the loop wrap, where the realtime resync decides). One
repeated and one skipped frame follow each transition by about 0.8 s.

Fixes found by measurement: the decoder discard target was cleared by a stale
NVDEC frame after a seek (flush magic now runs first, `src/nodes/decoders.cpp`);
fixtures now have no B-frames and the RGB fixture's code strip is drawn in YUV;
seeking to timestamp 0 stalls the decoder after the resume, so elements cued at
0 park on their last frame. Rejected after measurement: switching by the mixer
timeline key (`active` at PTS) caused post-cut repeats, deeper chain queues
made continuity worse, and resuming the incoming chain after the cut showed the
decoder's warm-up on air.

Published: `playlist-demo-media-2026-09` (movie, poster) and the graph PNGs in
`webui-graphs-2026-09`; `demos/graph.html` has the playlist entry; the TUI is
ASCII-only so browser terminals without symbol fonts render it.

## Determinism experiments after the merge (2026-09-08)

Goal: make elements cued at frame 0 land on frame 0 every time. Three
mechanisms were built and measured on the T4, each with a clean 50 s pass:

1. `source_switcher` **hold input** (`mixer.cut hold_incoming`): the pending
   input's newest frame is kept instead of drained, so the frame that makes
   the ready cut fire is the first one on air. Result: element 3 parked on
   frame 60 went on air on 61, i.e. the frame is lost *upstream* of the
   selector, not in it.
2. **Seek then `pause at`** (content-based park): the seek on a running chain
   never returned (control path hung inside flushAndSeek); not safe.
3. `pause` node **`pass_on_seek` off** so the seeked frame stays behind the
   pause until the resume, park exactly on cue-in, hold on: still +1, and
   parking at the file start reproduces the decoder tear after the resume.

Conclusion: the first fresh frame of a resumed element is consumed between
the chain and the output selector, most likely by the compositor's cadence
logic for the first frame after a long idle. A frame-exact cut for a cue at 0
needs either a compositor change (do not drop the first frame of a
re-activated input) or a content-addressed switch. All three experiments were
reverted; develop keeps the measured behaviour: exact for cue points inside a
file, ±1 frame for elements cued at 0.
