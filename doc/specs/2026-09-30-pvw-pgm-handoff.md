# PVW and PGM switch together: handoff

Status on 2026-09-30: implemented and reviewed on branch `pvw-latency`. It has not been built, run on a
GPU, or merged into `mixer-improv`.

2026-10-01: merged onto `mixer-improv` as branch `pvw-merge` (a clean merge; the Python suite passes)
and the swap decision below is taken. Still not built or run on a GPU.

## Goal

Under cut spam, the multiview PVW tile and the program must change at the same visual instant, with
the lowest latency the mixer allows.

Constraints from the owner:
- The design must be event-driven inside mixer code.
- No changes to generic avplumber C++: `src/nodes/hwaccel/cuda_rect_overlay.cpp` and the other generic
  nodes, `src/graph_*`, `src/EventLoop.hpp`, `src/avplumber.cpp`.
- No GPU, VRAM, RAM or CPU regression with default settings.
- The aux multiview stays at half rate at 50/60 fps by default; full rate is opt-in.
- Stability first: A/B before any default changes, and announce restarts on the live demo.

## Before this branch

- **How the tile followed the preview.** A Python thread in `pyplumber/mixer/aux.py`
  (`AuxMultiview.run`) polled the aux compositor's status every 50 ms. When the preview changed it
  republished the whole multiview composition. Take commands reach the C++ orchestrator directly, so
  Python learned of a change 0-50 ms late.
- **What the tile was aligned to.** The PGM tile, which is matched one aux tick back
  (`pgm_delay_frames = 1`).
- **Resulting latency.** At 60 fps the PVW tile left the mixer about K + 100 ms, where K is the pts of
  the first new program frame. The program leaves at K + 50 ms.
- **After a take.** The orchestrator cleared the preview, so the PVW tile went empty.

## What the branch does

Commits on `pvw-latency`: `452f22c`, `4cb2392`, `e9f4491` and `e1d9d9b`.

- **Emitter.** The orchestrator publishes each preview change as `MixerState::PreviewChange`: a
  revision, the scene and the effective time, under its own `preview_mutex` and condition variable.
  - A cut or fade publishes right after the selector switch, before the per-source routing
    (`applyPostTransitionRouting`'s switched hook, `publishTakePreview`).
  - A wipe publishes in `completeTransition`.
  - Interrupt and abort clear it (`clearTakePreview`).
  - One helper replaces the three copies of the state flip in cut, fade and wipe.
- **Absorber.** The new mixer-owned node `src/nodes/mixer_pvw_follow.cpp` runs one per `pgm_pvw_grid`
  bus.
  - It is a threaded node that waits on the condition variable. The wait is bounded to one aux tick,
    because node `process()` must return regularly.
  - It holds the per-scene PVW-tile layouts that Python publishes once and again on tile reassignment.
  - It applies them through the aux compositor's existing `setObject("composition")`.
  - `src/mixer/primitives/PreviewFollow.hpp` holds the tick math. The follower never takes the mixer's
    `mutex`.
- **OBS-style swap.** After a take, the preview becomes the scene that left program, matching OBS
  Studio's default "Swap Preview/Program Scenes After Transitioning". `pvw_slot_scene` (the slot is
  warm) is separate from `pvw_scene_name` (what the operator sees), so a cut back to a cold preview
  still reloads it. There is no GPU cost: the old slot is not kept compositing.
- **Aux bus options** (`demos/mixer/docs/config.md`):

  | Option | Default | Meaning |
  | --- | --- | --- |
  | `pvw_align` | `program` | Target the first aux tick at or after the program frame's departure. `pgm_tile` restores the old alignment. |
  | `latency_ms` | the program's latency (50 ms at 60 fps) | Was 2 aux ticks (66.7 ms). Revert per bus with `"latency_ms": 66.7`. |
  | `pgm_delay_frames` | 1; 2 on a `full_rate` bus at 50/60 fps | The build rejects settings that leave the PGM pad less than one program frame of margin. |
  | `full_rate` | `false` | Runs the bus at canvas rate at 50/60 fps, about x2 GPU for that bus's compositor and encoder. |

- **Probe.**
  - The follower's status reports `pvw_latency_ms`, `pgm_latency_ms`, `pvw_minus_pgm_ms`, `kind` and
    `target_unreachable`. Fades land one aux tick after program by construction and are reported as
    `target_unreachable`, not late.
  - `mixer.status` carries `pvw_latency`.
  - `demos/mixer/tests/cut_spam.py` prints a `pvw` line.
- **Web UI.** `?lowlat=1` on the control page stays opt-in. When set, it now reaches every viewer
  identically.

### Latency budget

K is the pts of the first new program frame.

| Setup | PGM leaves mixer | PVW tile leaves bus |
| --- | --- | --- |
| 60 fps, defaults | K + 50 ms | K + 50 (even K) or K + 66.7 (odd K); +33.3 on a missed tick |
| 60 fps, `full_rate` | K + 50 | K + 50; +16.7 on a missed tick |
| 30 fps, defaults | K + 66.7 | K + 66.7; +33.3 on a missed tick |
| Before this branch, 60 fps | K + 50 | K + 100 |

## Decision: the swap is always on

As left on `pvw-latency`, `control.swap_preview` was parsed and documented in Python, but C++ never
read it, so swap was always on. `mixer.init` is parsed in `src/avplumber.cpp`, which the constraints
exclude. The options were:

- (a) one line in the `mixer.init` parser;
- (b) the flag as a `mixer_pvw_follow` parameter, which makes it configurable only when a multiview
  exists;
- (c) remove the switch and keep swap always on.

The owner chose (c) on 2026-10-01: the swap after a take is always on, as in OBS Studio's default,
and not configurable. `control.swap_preview` is gone from the config parser, `MixerConfig`,
`settings()`, the builder's keyword arguments and `mixer.init`; `MixerState::publishTakePreview`
swaps unconditionally and `PreviewChange` carries no `swap` flag, so nothing is inert.

## Not verified

- Nothing was built with the real toolchain. The syntax checks and the C++ tests ran on macOS with stub
  `sys/eventfd.h` and `sys/prctl.h`.
- The Python suite passes locally: `demos/mixer/tests` and `tests/test_mixer_color.py`.
- These need a GPU host:
  - the PVW/PGM alignment and the fraction of missed ticks;
  - PGM-tile repeats at the 50 ms bus buffer;
  - `full_rate` cost;
  - the H.265 viewer with `?lowlat=1`.

## Next steps

1. **Merge.** Bring `pvw-latency` onto the current `mixer-improv`. It branched before these changes,
   which will conflict:
   - the web UI host CPU meters (`demos/mixer/webui/index.html`, `webui.py`);
   - the resident wipe chain merge (`pyplumber/mixer/graph.py`, `demos/mixer/mixer.py`);
   - the Fedora 44 image and the Boost `io_context` change;
   - the docs refresh.
2. **`swap_preview`**: decided, (c), see above.
3. **Build.** Build the mixer image with `demos/mixer/Dockerfile.fedora44` (CUDA 13.4, which needs host
   driver R615 or newer). Run `tests/test_mixer_preview_follow.py` and
   `tests/test_mixer_preview_swap.py` in the builder, then deploy by recreating the mixer container.
   Announce the restart.
4. **Measure** on the 68-input 60 fps show with a `pgm_pvw_grid` bus.
   - Run `cut_spam.py`: check the `pvw` line (PVW - PGM p50, p95 and max, and `late` for cuts only)
     and the bus's `playout.repeats` and `missed_deadlines` in `mixer.aux_status`.
   - Repeat with the bus at `"latency_ms": 66.7`, and once with `"full_rate": true`.
   - A/B CPU, GPU, VRAM and RAM against the baseline below. A regression of 5% or more keeps the old
     defaults.
5. **Ship.** Only then merge into `mixer-improv` and push. Update the cookbook (`multiviewer.html`,
   `pgm-delay.html`) and republish.

## Baseline

Fedora 44 image, CUDA 13.4, driver R615, 68 inputs at 60 fps (18 NVDEC + 29 browser + 17 raw NV12 +
4 keys):

| Metric | Value |
| --- | --- |
| Host CPU idle | 52% |
| avplumber | 4.38 cores |
| Electron | 2.91 cores |
| CPU PSI "some" | 6.9% |
| GPU | 50% |
| NVDEC | 81% |
| NVENC | 61% |
| VRAM | 7.7 GB |
| Mixer anonymous RSS | 2.29 GB |

The cut-spam gate (`--mix 6:2:2`) passes: spam p95 46 ms, max 52 ms, recovery p50 37 ms, and 0 missed
program deadlines in 7320 frames.
