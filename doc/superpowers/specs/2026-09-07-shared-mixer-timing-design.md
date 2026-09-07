# Shared mixer timing and frame continuity

Status: approved for implementation. Use bounded-latency recovery, with explicit
repeat/drop accounting, and the configurable two-frame default.

## Objective

Use one native timing implementation for the regular CUDA mixer and the
EGL/CUDA DMA-BUF compositor. With complete equal-rate inputs and readiness jitter
within the configured budget, each expected input frame must appear exactly once
in the output. A constant output frame count or consecutive output PTS alone is
not sufficient evidence.

First validate sixteen browser processes with one page each through the DMA-BUF
compositor at 60 fps. Then integrate the same timing implementation into the
regular mixer and verify its graph startup, prewarming, routing, and transitions.
The timing module must support rational frame rates rather than hardcoding 60.

## Preserve existing behavior deliberately

Keep the working Python mixer graph construction and prewarming algorithms.
Move mixer-specific Python code out of the generic `pyplumber` package into a
dedicated mixer package, splitting graph construction, prewarming, and scene
configuration into focused files. This is a relocation and targeted adaptation,
not a wholesale rewrite of orchestration in C++.

Preserve the established startup sequence: normalize sources, start the preheat
router, warm geometry paths, initialize routes, start scene slots, warm the
transition path, restore normal routing, then start output and declare readiness.
Preserve timeout reporting and avoid exposing partially warmed scene images.

The native scene orchestrator retains responsibility for transitions and routing
commands. The timing change must respect the existing shared scene timeline and
the alignment of A/B slots. Rendering kernels, pixel-format conversions, and
GPU ownership rules remain in the rendering adapters.

## Native module and responsibilities

Shared implementation belongs under `src/mixer/`. Proposed responsibilities:

- A public timing interface accepts source timing information and a supplied
  monotonic time, and returns the next presentation decision or wake deadline.
- Input timeline logic distinguishes source presentation time from arrival time,
  handles phase/cadence information, and detects discontinuities and sustained
  rate mismatch.
- Bounded frame queues retain ordered frame references and track consumption.
  The same queue/selection implementation must serve both frame representations.
- The two existing compositor nodes adapt graph edges and GPU frame types to
  this interface, then render its selected frames.

Use multiple implementation files where responsibilities warrant them, with a
small shared interface. Do not duplicate selection algorithms in the adapters
or move per-frame timing into Python. Do not change graph management to work
around a mixer-local problem.

## Timing contract

1. Calculate presentation times from an integer frame index and rational rate.
   Keep scheduled media time distinct from the worker's actual wake time.
2. Preserve valid presentation timestamps. Arrival-stamped live inputs require
   explicit cadence interpretation; do not independently round every noisy
   arrival timestamp and treat resulting collisions/gaps as content changes.
3. Keep future frames queued. A processing call that sees several ready frames
   must not silently discard unshown frames by retaining only the newest.
4. Apply an explicit playout delay and composition allowance. For output media
   time `q[n]`, readiness `A[i,n]`, delay `D`, and processing budget `C`, require
   `A[i,n] + C <= q[n] + D` for every selected input to meet its deadline.
5. Treat real rate conversion separately from jitter. Different-rate sources
   necessarily have a different frame-selection contract; buffering cannot
   create missing source frames or absorb indefinite frequency mismatch.
6. Report underruns, repeats, discarded frames, overflow, discontinuities, and
   missed output deadlines separately. Account for startup separately from the
   steady-state acceptance interval.
7. Preserve frame ownership through GPU completion. Buffer capacity must include
   queued, held, and in-flight references; metadata smoothing must not release
   a producer texture early or exhaust the producer pool silently.

Existing `Realtime`, `ForceFPS`, and output timestamp-normalization stages need
an explicit integration audit. Retain genuine rate conversion where needed, but
avoid multiple independent stages making jitter-related drop/repeat decisions.
The existing `smooth_timestamps` node is not a playout buffer and is not a
standalone solution for either compositor.

## Configurable delay

Provide one native `latency_ms` option shared by both compositors. If omitted,
the native implementation chooses two output frame periods: approximately
33.3 ms at 60 fps or 66.7 ms at 30 fps. This is the starting default to validate,
not a measured universal jitter bound. Allow a nonnegative explicit millisecond
override and reject invalid/nonfinite values before starting the graph.

Expose this through `--mixer-latency-ms` in the regular mixer demo and
`MIXER_LATENCY_MS` in the DMA-BUF demo. Python forwards the optional value;
native code owns default calculation, validation, and scheduling semantics.
Report the effective delay with the configured output rate.

Keep the configured delay fixed during a run in the first implementation.
Compare one, two, and three frame periods in validation. Changing the option
requires restarting the demo graph, not the VM.

## Deadline-miss policy

After an input exceeds the delay budget, repeat only the affected tile as
needed and discard overdue frames to restore alignment. Count every repeat
and discarded frame. Unaffected inputs and the fixed-rate output continue.
Frames arriving within the declared budget must remain continuous.

Adaptive changes during playback are a separate policy decision: increasing
content delay at fixed output cadence requires holds, while reducing delay
requires skipping or changing content progression. A smooth control signal does
not eliminate that tradeoff.

## Verification

The current local mixer suite has 24 passing cases. Graph tests use fake mixer
and node objects; the live protocol smoke test checks state and transition
completion. They do not establish rendered-frame continuity.

Add deterministic native tests at the shared timing interface used by both
adapters. Cover clean equal-rate streams, many phases, exact deadline boundaries,
timestamp zero, rational rates, receiver wake jitter, burst delivery, correlated
sixteen-input stalls, backpressure, and long simulated runs. Assert explicit
expected frame IDs, not calculations copied from the implementation.

Add policy tests for budget overruns, missing inputs, source restart, EOF,
discontinuity, rate mismatch, overflow, and inactive/reactivated scene inputs.
Output backpressure must not advance timing state for an uncommitted frame.

For GPU integration, record the normal NVENC output and analyze it offline.
Decode the recording once, inspect all sixteen cropped tiles, and verify actual
source frame IDs plus output PTS. Use moving deterministic patterns so intentional
static content does not look like a timing failure. `mpdecimate` can supplement
frame-ID checks, but cannot detect skipped source frames by itself.

Do not insert sixteen download/filter branches into the live performance test.
The existing live per-input `mpdecimate` mode requires the legacy CUDA backend
and changes the workload; it cannot certify the production EGL path.

After DMA-BUF passes, exercise regular-mixer cold start, all preheated layouts,
preview/program changes, repeated cuts, fades, wipes, and input restart/stall
during transitions. Verify rendered output as well as reported control state.
Retain and rerun the existing Python suite and protocol smoke tests.

Compare CPU, sustained GPU usage, GPU memory, host memory, and latency only
between configurations meeting the same correctness and throughput criteria.

## Delivery

One combined PR contains the shared native timing implementation, mixer Python
organization, both demo integrations, tests, and documentation. Consolidate the
implementation after validation; do not publish a sequence of small commits.
Keep unrelated work out of this PR. Keep recorded binaries outside Git.

See `doc/research/2026-09-07-live-mixer-playout.md` for the OBS/GStreamer/WebRTC
comparison, the existing AVPlumber paths, and the source-throughput measurements.

## Mixer media, included in this PR

After timing and transition validation passes, record a short side-by-side view
of the program output and the connected mixer TUI. Capture the Web UI graph to
show nodes and frame flow. Use the established replay demo page template, with
an MP4 poster fallback, graph screenshot, README links, and GitHub Pages page.
Publish media as external assets, never as binary files in Git. Include the
mixer media page and documentation in the same combined PR.


## Interrupting a take

A new Cut, Fade, or Media Wipe replaces an in-flight take. It must not wait for
an earlier transition or its cleanup. Capture the exact last committed mixer
output, including partially blended scenes and media overlays, as a retained
frame on its existing device. Hold it while preparing a fresh target scene,
then use it as the outgoing picture for the replacement transition. A second
interruption captures the current result again. No CPU download or extra scene
compositor is required.

The shared C++ orchestrator invalidates old callbacks and their scheduled route
changes before reusing the existing slots. Output capture and publication share
a short lock; neither GPU work nor a blocking queue write holds that lock.
The TUI sends each take directly, including corrections to the old Program.
Preview remains optional preparation. Verify all nine pairs of transitions,
interruptions before/after a wipe midpoint, repeated corrections, monotonic
output timestamps, final visible scene identity, and stale cleanup rejection.
