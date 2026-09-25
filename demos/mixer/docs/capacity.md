# Mixer demo capacity

These numbers describe 1920×1080 SDR 8-bit inputs on a 16 GiB NVIDIA T4,
with PGM and one AUX output. They are not decoder-only limits or guarantees
for HDR, larger frames, arbitrary browser pages, or additional outputs.

The setup uses a 100-source-at-25-fps baseline. Higher rates below are linear
estimates rounded down, with a separate raw NV12 budget of 700 frames/second.

| Input fps | Total baseline | NVDEC | Browser | Raw NV12 upload maximum | Validation |
| --- | ---: | ---: | ---: | ---: | --- |
| 25 | 100 | 40 | 32 | 28 | Two healthy starts; several minutes of transitions |
| 30 | 83 | 33 | 27 | 23 | Healthy starts and transitions; long soak still needed |
| 50 | 50 | 20 | 16 | 14 | Linear estimate, not a measured capacity result |
| 60 | 41 | 16 | 14 | 11 | Linear estimate, rounded to keep upload headroom |

The browser service defaults to four workers with eight windows each (32 total).
The setup allows 192 scenes; scenes describe layouts and do not each allocate a
running compositor. Active layers and AUX outputs have separate limits.

A temporary six-worker, 42-browser experiment at 110 sources kept the 25-fps
upload and NVDEC counts unchanged:
**40 NVDEC + 42 browser + 28 raw NV12**, six browser workers and a six-frame
browser ring. It tests additional browser capacity rather than raising upload
traffic. The first run maintained 25 fps between two multi-second stalls,
with a 2.9-second cut, VRAM peaking near 14.1 GiB and up to 145 browser imports
pending cleanup. It recovered without intervention. Remote validation overlapped
the stalls, so this is not an isolated capacity measurement; 110 is not yet a
validated reliable limit. The instance was returned to the 100-source baseline,
and the browser limit was restored to 32 across four workers.
Do not extrapolate this experiment to 110 inputs at 30 fps.

Both stalls began during the same early phase of two remote validation runs.
Fresh browser delivery fell to zero for several seconds while raw uploads
continued near 25 fps. GPU utilization stayed around 61–67%; decoder utilization
fell during the stall rather than remaining saturated. No CUDA out-of-memory
error was recorded. Repeating validation with the mixer stopped showed no
additional CUDA compute process and constant GPU memory, so the tests did not
directly allocate the extra VRAM. Their contribution to scheduling or driver
contention still needs an isolated trace.

The import path can amplify a delay: idle cache entries expire after one second
at ring size six / 25 fps, and all browser nodes share a cleanup worker. A cache
miss waits until that shared cleanup backlog is empty before creating an import.
A burst of expiry can therefore stop new imports across otherwise independent
browser sources. Import statistics were sampled every ten seconds; they establish
the backlog during the stall, but cannot prove that expiry caused its onset.

### Same workload at 25 and 30 fps

A repeat with nonblocking import admission and corrected PVW publication kept
1080p SDR inputs, ring six, 192 scenes, one random PGM scene and eight 64-input
AUX tiles. No profiler or build ran during sampling.

| Sources / fps | NVDEC / browser / upload | Fresh browser mean fps | PGM / AUX fps | GPU busy | Mean VRAM |
| --- | --- | ---: | --- | ---: | ---: |
| 100 / 25 | 40 / 32 / 28 | 25.0 | 25.0 / 25.0 | 55% | 9.4 GiB |
| 100 / 30 | 40 / 32 / 28 | 18.7 | 3.7 / suspended | 65% | 9.4 GiB |
| 82 / 30 | 33 / 26 / 23 | 30.0 | 30.0 / 30.0 | 56% | 7.5 GiB |

The failed 30-fps run lasted 91 seconds; its row excludes the first 15 seconds.
It rejected 26,585 import admissions during the remaining window, and AUX
suspended after encoder backpressure. The 82-source run sustained two minutes
without new import-admission drops or pending cleanup; 12 cuts measured
54–82 ms to encoded output (median 66 ms). This is a short validation, not a
guaranteed maximum. The healthy loads request about 2,500 input frames/second,
versus 3,000 for the failed run. Source count alone cannot explain this boundary;
the comparison does not separate pixel-transfer cost from per-frame driver work.

## Why uploads limit this mix

At 100 sources / 30 fps (40 NVDEC, 32 browser, 28 raw NV12), Nsight showed a
roughly 0.4 ms compositor kernel, but around 32 ms in its CPU-side CUDA table
upload and kernel submission calls. Browser import retirement also slowed,
and expired imports caused further registration work.

In a controlled run, stopping only the 28 raw-upload inputs in the same process
restored PGM from about 13.5 to 29.8 fps and fresh browser frames from about 1.3
to 29.8 fps. Compositor submission fell from about 32 ms to 0.08 ms; new browser
imports fell to zero in the following 48-second sample. The NVDEC and browser
inputs remained running. This identifies the raw-upload workload as the trigger
for shared CUDA submission contention, not an expensive compositor shader.
It does not identify a specific NVIDIA internal lock or prove that a separate
stream or pinned staging alone would solve it.

28 NV12 inputs at 1080p25 upload about 2.18 GB/s of image payload. On this host,
a single-stream copy benchmark with the mixer stopped measured roughly
3.84 GB/s from ordinary host memory and 7.63 GB/s from pinned memory at that
frame size. These are measured transfer rates, not usable mixer budgets:
decode, encoding, browser interop and driver submission need headroom too.

The setup enforces 28 raw NV12 uploads at most, reducing that to 23/14/11 at
30/50/60 fps. Excess allocation is redistributed among enabled source types;
a raw-only request above its limit is rejected. This cap concerns the raw NV12
CPU-to-GPU path. P010 uploads share the same budget at two units per source,
based on double the bytes per frame; this is not a measured HDR capacity. Combined
SDR/HDR NVDEC inputs are capped at 40 for 25/30 fps and 20 for 50/60 fps.
The separate v210 upload/unpack path is not calibrated by this
measurement. Higher browser capacity does not increase the upload allowance.

### Upload headroom and browser allocation bursts

A subsequent 25-fps sweep kept 40 NVDEC inputs and 42 browser windows fixed,
then changed active raw NV12 uploads through 28, 26, 24, 22, 18, 14 and back to
28. PGM displayed a 64-input grid and AUX eight 64-input grids. Each later phase
included a refresh of all browser pages, deliberately replacing browser buffers.
This is a recovery stress test, not normal steady-state browsing.

The initial untraced 28-upload interval delivered 25 fps from every browser.
After refresh, 26 and 24 uploads each left one browser no longer painting
(different windows in the two phases). At 22 and 18 uploads, every browser
returned to 25 fps. Thus 22 is a candidate upload ceiling for this particular
42-browser workload, with 18 providing more headroom; neither is a certified
long-running capacity limit. The generic setup's 28-upload maximum must not be
interpreted as safe independently of browser load and allocation churn.

CUDA uprobes attached during the 14-upload phase and the return to 28 exposed
submission waits, but also affected performance. With 14 uploads, the compositor
table upload averaged 0.081 ms; with 28 it averaged 5.31 ms. Slow table uploads,
kernel submissions, texture destruction and browser registration spent roughly
98–99.7% of their duration in futex waits. The compositor and deferred cleanup
thread waited on the same futex address. This identifies shared CUDA
synchronization as the blocking point; it does not identify NVIDIA's internal
lock implementation or establish a GPU-memory-bandwidth limit. Stream
synchronization averaged about 0.63–0.67 ms in both phases.

After all tracing detached, the same process recovered without removing sources.
A further untraced refresh at 28 uploads caused Chromium GPU-buffer allocation
failures and left ten of the 42 browsers no longer painting, although PGM output
returned to 25 fps. GPU memory sampled once per second peaked near 14 GiB during
that refresh; sampled occupancy alone cannot rule out allocation failure between
samples. An earlier refresh had queued 324 retired imports for cleanup.

There are consequently two related failure modes: CUDA submission/cleanup waits
reduce ingestion throughput, and overlapping old/new browser allocations exhaust
buffer headroom during a burst. Import expiry and the shared cleanup admission
gate amplify a delay. Encoded PGM fps is insufficient to validate recovery: check
fresh browser delivery and the slowest individual source as well. A fixed source
count or GPU-core utilization percentage does not describe this boundary.

A separate untraced comparison held 40 NVDEC inputs and 28 uploads fixed, using
32, 37 and 42 browsers for 100, 105 and 110 total sources respectively. All runs
used six available browser workers, the same PGM/AUX layout sizes, and one page
refresh after 30 seconds. The 75-second observations ended with every browser
near 25 fps at 100 and 105 sources. At 110, two browsers remained at zero fps;
peak sampled VRAM was 14.29 GiB and pending import cleanup reached 219. The
corresponding peaks at 100/105 were 12.02/13.42 GiB and 24/50 cleanup jobs.
This locates a recovery boundary for this test, not an exact universal maximum
between 105 and 110 sources.

### Paced cleanup experiment

`drm_prime_to_cuda.cleanup_interval_us` optionally spaces retired-import releases
on the shared cleanup worker. The interval is a minimum pause after a release
finishes, so a slow release never triggers catch-up bursts. Closing an importer
bypasses pacing for its retired jobs. Live and retired imports retain the same
per-source budget; cache hits do not wait for cleanup. The default remains `0`
(unpaced), because a 1000 µs interval did not make 110 sources reliable.

The experiment retained 40 NVDEC + 42 browser + 28 raw NV12 inputs at 25 fps,
six browser workers, ring six, 192 scenes, the 640-layer budget, a PGM 64-input
grid and eight AUX 64-input tiles. Each 75-second observation refreshed all
browser pages at second 30, without tracing or concurrent builds/tests.

An initial variant allowed replacement imports to consume byte credits as
individual cleanup jobs completed. It left browser delivery averaging only
3.2 fps in the final 20 seconds with AUX active. That admission change was
removed; new registrations still wait for the shared cleanup backlog to drain.

With pacing alone, AUX suspended during startup. After the first refresh,
six browser windows remained frozen. AUX was explicitly resumed and another
75-second refresh observation ended with three browsers frozen, despite fresh
PGM and AUX output both averaging 25 fps. Sampled VRAM peaked at 14.27 GiB,
and Chromium logged GPU-buffer allocation failures. The resumed observation
started with six frozen browsers, so it is not a clean before/after capacity
comparison with the earlier 110-source run. Both observations failed recovery.

Pacing cannot interrupt a CUDA call already blocked in the driver, nor bound
Chromium's own allocation bursts. Delaying destruction also retains retired
buffers longer. Keep this option experimental; the instance was restored to
100 sources with four browser workers and pacing disabled.

### Locating the global browser stall

A replay with pacing disabled added importer-phase, admission-wait, cleanup-time
and receiver ACK counters. It used the same 110-source recipe and browser refresh
with PGM and AUX active. At 31.4 seconds, 21 importers were in CUDA registration.
At 32.7 seconds, 38 of 42 importers were waiting in `cleanup_admission`. They were
still waiting at 38.7 seconds, with individual uninterrupted waits over 5 seconds.
The shared backlog reached 321 retired imports. Between the 32.7- and 38.7-second
samples, 329 imports were destroyed, taking 5.9 seconds of cleanup-worker time
(about 18 ms each). These timings include driver waits and scheduling, not just
GPU execution.

During the sustained stall, the 38 affected sources each held six frames. Their
queues immediately before the importer held 114 frames (three per source), while
the queues after import were empty. A representative source accounted for all
six: three queued before import, one being imported, one queued further upstream,
and one last displayed frame. The receiver had received 937 frames and released
931. Pending ACK messages were zero: already released frames were acknowledged.
`repeat_last_frame` continued emitting references to the last image at 25 fps,
which explains output frame rates recovering before browser images moved.

This identified the sustained global stall with blocking admission: every cache-missing importer
waited for **all** pending cleanup in the instance to drain, regardless of its own
remaining budget. Expiry after the refresh increases that shared backlog. The
gate holds incoming frames, fills bounded upstream queues, and exhausts browser
capture slots. It is not an ACK transport blockage or unbounded graph buffering.
The initial registration burst and Chromium allocation failures also occurred;
the counters do not identify which NVIDIA operation inside destruction dominates.

Admission is now nonblocking: an incoming frame needing a new import is dropped
when cleanup is pending or its import budget is full. The input reference is
released so the sender can reuse its capture slot. Cached imports continue normally;
new registrations retain the same conservative memory admission rule. The check
runs before EGL import work. `admission_dropped` counts rejected frames; monitor
the CUDA output edge as well as browser transmission, since a transmitted frame
may now be dropped before import. This does not remove waits inside an admitted
EGL/CUDA registration call.

Two subsequent runs used nonblocking admission with 110 sources and **ring nine**,
restarting both the mixer and browser workers between runs. Other source counts,
layouts and the refresh at 30 seconds were unchanged. Ring nine also increases
the derived import-cache expiry from 1000 to 1440 ms, so these are combined
configuration/code experiments rather than isolated admission comparisons.

| Final 20-second observation | Run 1 | Run 2 |
| --- | ---: | ---: |
| Browser frames transmitted, mean per source | 22.5 fps | 21.9 fps |
| Browser frames imported into CUDA, mean per source | 22.5 fps | 7.1 fps |
| Fresh PGM output | 24.9 fps | 2.4 fps |
| Fresh AUX output | 24.9 fps | 23.4 fps |
| Browser windows no longer painting | 4 | 5 |
| Peak sampled VRAM over the run | 13.89 GiB | 14.12 GiB |
| Admission drops after refresh | 3,416 | 26,760 |

The nonblocking path released frames while cleanup was busy. In run 1 at 34.9
seconds, 210 imports were pending cleanup, all importers were back waiting for
input, the queues immediately before them were empty, and pending ACKs were zero.
This removed frame retention at the admission gate. It did not establish reliable
110-source operation: run 2 remained in import/expiry churn, with browser
transmission substantially exceeding successful imports. EGL/CUDA registration
can still block after admission, and allocation failures persisted. PGM/AUX
encoded frame rates and browser transmission rates alone conceal these failures.


## AUX layout changes

The measurements in the table below preceded staged AUX layout changes.

On the 100-input / 25-fps baseline, a draw budget of 640 admitted eight 64-input
tiles plus a 64-input PVW and the composited PGM (577 layers). The full layout
held 25 fps with about 9.5–9.8 GiB total VRAM and 55–58% GPU utilization in the
short observation windows. No validation jobs ran alongside these measurements.

| Operation | Fresh AUX output | Encoded AUX / PGM | Observation |
| --- | ---: | ---: | --- |
| Hold all eight grids | 25 fps | 25 / 25 fps | No browser drops in steady windows |
| Permute the same grids, four edits/second | 25 fps | 25 / 25 fps | 32 edits, zero browser drops |
| Replace grids with different source sets, four edits/second | About 18 fps | 25 / 25 fps | Brief hitches; recovered after edits stopped |
| Alternate sparse and dense layouts, four edits/second | About 20 fps | 25 / 25 fps | Brief hitches; recovered after edits stopped |

These are short edit stress tests, not a long-term capacity certification.
The worst sampled one-second AUX interval during source-set changes was about
10 fps; VRAM briefly reached about 11.5 GiB. PGM stayed near 25 fps. The AUX
encoder repeated frames to maintain its nominal rate, so encoded fps alone did
not reveal the reduced fresh-output rate.

Previously, a source-set change cleared new-input history and restarted a 250 ms
warm-up window for the entire AUX. At 25 fps its usual 120 ms latency budget
could therefore delay every tile while a newly subscribed frame became eligible.

AUX now keeps rendering the current layout while subscribing only to the latest
requested layout's additional sources. It switches the complete composition once
all required inputs have a frame eligible for the next output tick. Superseded
requests release their extra subscriptions. Preparation is bounded to the larger
of 250 ms and twice the AUX latency budget; on timeout the previous layout stays
live and pending subscriptions are released. Frame history and queue sizes are
unchanged. Native AUX status exposes `composition_pending` and `composition_error`.

Repeating the same 32 dense-layout edits and 16 sparse/dense edits at four edits
per second after this change measured approximately 25 fps for fresh AUX output
and PGM, with zero browser drops and no sampled import-cleanup backlog. Total
VRAM remained about 9.4 GiB and GPU utilization 55–59% over the 100-second run.
Settled subscription sets matched the requested layouts, rather than every input.
Remote tests also covered an unavailable source, timeout, cancellation, resumed
input, encoder backpressure, and independent compositor restart.
