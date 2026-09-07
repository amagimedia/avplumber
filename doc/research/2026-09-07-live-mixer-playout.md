# Live mixer playout: delay versus frame continuity

Reviewed 2026-09-07. This is a design investigation, not an implemented fix or
a claim that any particular historical OBS incident has been reproduced.

## What other implementations do

### OBS Studio

The requested [libobs directory](https://github.com/obsproject/obs-studio/tree/master/libobs)
was inspected at commit `6b3e550729f125b6c5b3767df88c08f5aef9d264`.
In `ready_async_frame`, unbuffered mode explicitly removes every queued frame
except the newest. Buffered mode advances a source timeline by the output clock
increment and walks past older candidates. A fixed 2 ms guard limits advancement
near a boundary, but only after a candidate has already been selected. These
are deliberate frame-selection policies, not a promise to display each input.
[Source](https://github.com/obsproject/obs-studio/blob/6b3e550729f125b6c5b3767df88c08f5aef9d264/libobs/obs-source.c#L4124)

`get_closest_frame` returns early on an empty queue, while `async_tick` updates
the last system timestamp even then. Timestamp zero also acts as an initialization
sentinel. At 30 queued frames, `cache_video` clears the async cache and resets
timing state. These boundary and recovery behaviors deserve explicit tests.
[Source](https://github.com/obsproject/obs-studio/blob/6b3e550729f125b6c5b3767df88c08f5aef9d264/libobs/obs-source.c#L3535)

Inference: clean producer cadence does not exclude drops inside OBS. Internal
delivery bursts or timeline changes can trigger its selection policies. This
does not establish the cause of an unspecified past incident. Copying its
thresholds would not establish frame continuity in this mixer.

### GStreamer

`GstVideoAggregator` maps timestamps through each input segment to running time.
Its queue selection compares the input presentation interval with the output
interval: overlapping input is selected, future input is retained, and stale
input can be consumed while requesting more data. Timeout processing can drain
late data. A separate QoS calculation permits dropping an entire late output
frame. Merely finding `drop_buffer` in this code is insufficient evidence of a
discarded image: it also removes a buffer from the queue after retaining it for
composition.
[Source at commit 633eae4](https://github.com/GStreamer/gstreamer/blob/633eae4c7c4b25b9043372a8a91cf7a386d5168b/subprojects/gst-plugins-base/gst-libs/gst/video/gstvideoaggregator.c#L1753)

GStreamer's latency model makes synchronization delay explicit: live data needs
time to become available, and the pipeline accounts for source/processing latency
before scheduling presentation. The useful idea is a shared presentation timeline
with an explicit readiness budget.
[Latency design](https://gstreamer.freedesktop.org/documentation/additional/design/latency.html)

### WebRTC

The inspected `VCMTiming` separates timestamp extrapolation, estimated jitter,
decode time, render allowance, and configured playout bounds. Its target includes
jitter plus decode/render allowances, subject to a minimum delay; actual render
time uses the mapped source timestamp plus bounded current delay. Delay changes
are rate limited rather than applied as an immediate large jump. This revision
also has a distinct low-latency rendering path, so these observations do not
describe every WebRTC mode.
[Source at commit 459e9df](https://webrtc.googlesource.com/src/+/459e9dffeae8c42f9a495295a7aaa1c9d032ddf4/modules/video_coding/timing/timing.cc)

Its jitter estimator includes receiver operating-system jitter in addition to
network-derived estimation. This supports measuring readiness at the consumer,
not assuming regular producer timestamps imply regular availability downstream.
Its packet-size, retransmission and network-specific estimator machinery is not
directly needed for a local DMA-BUF mixer.
[Jitter estimator at commit c42162c](https://webrtc.googlesource.com/src/+/c42162cacb762b3a343a0b7bd398ce77dd4ff69e/modules/video_coding/timing/jitter_estimator.cc)

## Mathematical model and proposed policy

The following is a design recommendation inferred from those approaches and the
requirements here, not an algorithm attributed to any one project.

The agreed implementation direction is a shared C++ timing module under
`src/mixer/`, used by both `CudaRectOverlay` and `EglImageCudaOverlay`. Retain
the working Python graph construction and prewarming behavior in a dedicated
mixer package outside the generic `pyplumber` bindings. Split responsibilities
into focused files; do not create independent timing algorithms in the demos.
Validate the sixteen-process DMA-BUF demo first, then migrate and regression-test
the regular mixer, including its prewarming and transition paths.

### Existing AVPlumber timing paths

The mixer demo builds `Realtime(set_pts=true) -> ForceFPS -> CUDA normalization`
for each input, followed by the PTS-driven `CudaRectOverlay` and another
`ForceFPS` stage. `CudaRectOverlay` selects the minimum available input PTS,
waits for required inputs, and retains prior images for inputs whose next PTS
is later. This differs from the production DMA-BUF compositor's arrival-driven
`drainLatest` plus independent output ticks. See `demos/mixer/mixer.py`,
`pyplumber/mixer.py`, and `src/nodes/hwaccel/cuda_rect_overlay.cpp`.

The DMA-BUF browser currently timestamps frames inside its paint handler using
`monotonicTimeNs()`. The receiver converts that value to microsecond PTS, and
the EGL conversion preserves it. Thus shared clock origin does not make those
timestamps an ideal animation presentation grid: paint-handler scheduling is
part of their timing. See `deps/dma-browser/src/main/capture/FrameCaptureChannel.ts`,
`src/nodes/hwaccel/ipc_dmabuf_source.cpp`, and
`src/nodes/hwaccel/drm_prime_to_egl_image.cpp`.

`smooth_timestamps` already produces nominal consecutive frame timestamps with
optional drift-based resynchronization. It neither waits for presentation nor
provides a multi-input playout buffer. `ForceFPS` explicitly duplicates or
discards input when normalizing timestamps. Neither alone fixes `drainLatest`.
Before reusing the smoother, test its backpressure behavior: code inspection
shows it advances timing/drift state before a nonblocking output write and
restores only the frame timestamp if that write fails. This is an untested
additional hazard, not the cause established for the current EGL path, which
does not instantiate this node. See `src/nodes/smooth_timestamps.cpp` and
`src/nodes/force_fps.cpp`.

### Presentation timeline and delay

Let `T = 1/60 s` and define the ideal output media timeline as `q[n] = q[0] + nT`.
Compute timestamps from the integer frame index and rational rate, rather than
repeatedly adding a rounded 16.667 ms period. Keep the scheduled media time
separate from the actual worker wakeup time.

For a clean, equal-frequency source, establish a frame-index offset `k[i]` during
warmup. Output frame `n` consumes source frame `n + k[i]`. Retain pending GPU
frame references in order; a late wakeup must not silently replace that scheduled
frame with the newest arrival. Different fixed phases do not themselves require
recurring repeats or drops. Alignment to an arbitrary output phase can introduce
up to one period of quantization delay.

Let `A[i,n]` be the time that the selected frame is usable by the compositor,
including internal scheduling and GPU-fence readiness. With output presentation
deadline `q[n] + D` and remaining composition budget `C`, a sufficient condition
for frame availability is:

```text
A[i,n] + C <= q[n] + D     for every active i and n
D >= sup(i,n) (A[i,n] - q[n]) + C
```

This is a conditional guarantee: it assumes correct clock/index mapping, no
missing frames, adequate retained-buffer capacity, and bounded processing time.
A fixed buffer cannot guarantee uninterrupted playback through unbounded stalls.
Regular input timestamps do not bound receiver scheduling jitter by themselves.

For measured rather than hard bounds, record `R[n] = max_i(A[i,n] - q[n])` and
choose `D = quantile_(1-epsilon)(R) + C`. This estimates the probability that
**any tile** misses its deadline and captures correlations between sources.
It is preferable to multiplying independent per-source probabilities on a shared
CPU/GPU. For intuition only, independent per-source miss probability `p` gives
scene miss probability `1 - (1-p)^16`; `p = 0.001` is about 1.59% per scene frame.
Observed quantiles need sufficiently long runs and do not promise future bounds.

Start with a fixed, configurable delay and compare 0, 1, 2, and 3 frame periods
(0, 16.7, 33.3, 50 ms). Choose the smallest measured delay meeting the continuity
target. Two frames are a candidate to test, not an established answer.

## Limits and recovery choices

- **Ordinary equal-rate jitter:** consume one ordered frame per tick. Do not
  drain the queue merely because multiple frames are ready after a wakeup.
- **Source missing its deadline:** hold only that tile and count the reason.
  Decide explicitly whether to preserve the late frame at increased tile age or
  discard it to recover alignment. Continuing the FIFO after a hold preserves
  order but increases delay; at equal rates that extra occupancy does not drain
  without another policy change.
- **True rate mismatch:** finite buffering cannot eliminate corrections forever.
  A 60.006 fps source against 60 Hz accumulates one surplus frame in roughly
  167 seconds. Clock synchronization, producer pacing, interpolation, or explicit
  occasional drop/repeat is required. Do not mistake arrival jitter for drift.
- **Adaptive delay:** increasing content delay at fixed output cadence requires
  holds or slower content progression; decreasing it requires skips or faster
  progression. Smoothing the control reduces abruptness but cannot remove this
  conservation constraint. Prefer adjustment during startup/reconfiguration,
  then add bounded, hysteretic adaptation only if measurements justify it.
- **Overflow/discontinuity:** use a bounded, observable recovery policy. Avoid
  silent whole-queue resets. Timestamp reset, restart, and format change establish
  a new epoch, using explicit state rather than treating timestamp zero as unset.
- **GPU ownership:** queued references retain actual producer resources. Size
  the producer pool for pending, held, and in-flight frames, and release only
  after GPU use completes. Zero-copy buffering still consumes resource capacity.

## Tests that distinguish a fix from concealment

Use frame IDs as well as timestamps and frame-rate counters. Assert exact
one-frame increments for each tile under the declared bounded-jitter model.

1. Perfect 60 fps input at many fixed phases, including exact tick boundaries,
   timestamp zero, and long runs exposing rounded-period drift.
2. Perfect producer cadence with injected receiver wakeup, dispatch, and GPU
   readiness jitter; arrivals and presentation timestamps remain distinct.
3. Sixteen independent phases, correlated scheduler stalls, and bursts that
   deliver two or more frames between processing calls.
4. Delays just below, equal to, and just above the configured readiness budget;
   missing frames and overflow must produce the documented recovery counts.
5. Slow frequency mismatch, source restart, timestamp discontinuity, late joining
   inputs, and one stalled input while the other fifteen continue.
6. GPU lifetime checks and end-to-end encoded pixel/frame-ID verification.

For pixel validation, record the normal NVENC output and analyze it offline.
Decode the composite once, then examine each of its sixteen cropped tiles.
The existing live per-input `mpdecimate` diagnostic requires the legacy CUDA
backend and adds scaling, downloads, conversion, and CPU filtering on every
input; it is not suitable for measuring production EGL-path throughput.
`mpdecimate` is a threshold-based visual-similarity check, so static content
can be classified as duplicate and skipped source frames are not detected.
Use known moving patterns and source frame IDs as the continuity oracle, with
`mpdecimate` as a supplementary check. See the
[FFmpeg filter documentation](https://ffmpeg.org/ffmpeg-filters.html#mpdecimate).

Report source production rate, consumer arrivals, selected IDs, repeats,
discard reasons, input age, queue occupancy, fence readiness, worker lateness,
and output deadline misses separately. A 60 fps encoder alone proves none of
the per-source continuity properties.

## Source-throughput experiment

Thirty-second samples of sixteen Singular pages on a Tesla T4 with sixteen
Broadwell vCPUs, using the same unmodified mixer and browser build:

| Browser workers | Pages per worker | Per-source paint/send fps | Host CPU busy |
| --- | --- | --- | --- |
| 2 | 8 | 56.10–57.17 | 54.2% |
| 4 | 4 | 58.36–59.29 | 79.8% |
| 8 | 2 | 59.32–59.78 | 80.3% |
| 16 | 1 | 59.95–59.98 | 65.9% |

All runs reported zero DMA-BUF transport drops; mixed output stayed near 60 fps.
The final run's native input queues measured 59.98–60.02 fps, consistent with
60 fps at the sampling precision. Its GPU utilization snapshot was 44%, with
3410 MiB allocated. These are workload-specific observations, not capacity limits.

Use one worker per page as the next validation baseline: it delivered more frames
with lower measured CPU usage. Compared with four workers, GPU utilization
snapshots were similar (44% versus 43%), while GPU memory increased by 181 MiB
(3410 versus 3229 MiB). Sustained resource usage and total host memory still need
measurement. The results also motivate investigating shared-process overhead.

These rates do not certify consecutive source frame IDs or fix the independently
observed mixer repeat/skip behavior. The diagnostic rate probe's permissive
59 fps threshold must not be used as the frame-perfect acceptance criterion.
