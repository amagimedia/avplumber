# Mixer demo capacity

These numbers come from a 16 GiB NVIDIA T4 host (16 vCPU) with 1920×1080 inputs. The
30 and 60 fps baselines below stand on the source mixes listed beside them; the older
25 and 30 fps measurements that follow used SDR 8-bit inputs with PGM and one AUX
output. None of them are decoder-only limits or guarantees for HDR, larger frames,
arbitrary browser pages, or additional outputs.

The setup (`setup_runtime.py`, mirrored in `setup.html`) allows 110 inputs at 25 and
30 fps, 82 at 50 and 75 at 60 fps, downstream-key pages included. Within that total it caps
NVDEC streams at min(40, 1100 ÷ fps), browser windows at 40 (five workers of eight) and
raw NV12 uploads at 30, 34, 20 or 17 units at 25, 30, 50 or 60 fps (a P010 upload costs two).
50 fps takes the 60 fps total and upload budget scaled by frame rate; the total never exceeds
what the three caps carry together, 82 at 50 fps, since the 40 browser windows do not scale.
[Source limits by mode and frame rate](cookbook/source-limits.html) has every mode in one table.
On a 10-bit canvas the total is scaled by `MODE_CAPACITY`: 0.82 for HLG 4:2:0 (90 at 25/30 fps,
73 at 50, 61 at 60) and 0.74 for HLG 4:2:2 (81, 66, 55); see [10-bit canvases](#10-bit-canvases).
A request above the limit is scaled down to it.

| Input fps | Setup limit | NVDEC | Browser | Raw NV12 upload maximum | Validation |
| --- | ---: | ---: | ---: | ---: | --- |
| 25 | 110 | 40 | 40 | 30 | 100 (40 / 32 / 28) validated: two healthy starts, several minutes of transitions; 110 not measured at this rate |
| 30 | 110 | 36 | 40 | 34 | The declared baseline: 36 NVDEC + 36 browser + 34 raw NV12 + 4 keys, pinned uploads; not pushed further |
| 50 | 82 | 22 | 40 | 20 | The 60 fps upload budget scaled by frame rate, NVDEC from the 1100 frames/s rule; the total is the three caps (the scaled 90 exceeds them); not measured |
| 60 | 75 | 18 | 40 | 17 | 18 NVDEC + 36 browser + 17 raw NV12 + 4 keys: every cap filled, cut-spam gate passes (see [60 fps on the current stack](#60-fps-on-the-current-stack)); 68 had a 19.5 h soak |

At 60 fps the 68-input show (18 NVDEC + 29 browser + 17 raw NV12 + 4 downstream-key
pages) currently measures about 50% host CPU idle, 62–64% GPU utilisation, 84% NVDEC,
61% NVENC and 8 GB of the 15.4 GB VRAM, with 0 missed playout deadlines in steady state;
a 19.5 h run missed 24 (0.0006%). Cut latency, from the command to the first encoded
frame of the new scene, is about 43 ms at the median. History: before the mostly-still
browser test page (commit 5068a50) and Electron 44, the same show measured GPU SM about
88%, NVDEC 88% and 19% host CPU idle, and 70 inputs saturated the GPU (SM 92-94%,
NVDEC 95%). Two defaults make it hold under cut, fade and wipe spam
(`demos/mixer/tests/cut_spam.py`): a 3-frame playout deadline at 50/60 fps (50 ms at 60; two frames
left 17 ms of slack), and the wipe clip cache (`--wipe-cache-mb 640`, which holds the
demo's two wipes, about 0.5 GB, in VRAM), without which each wipe take decoded QTRLE on
the CPU and missed deadlines.
The cached wipe chain also stays running between wipes: stopping and re-creating its CUDA
compositor on every take freed and reallocated GPU memory under the program and missed
deadlines under wipe spam.

**Current stack, measured 2026-09-30.** The shows below ran on the Fedora 44 mixer image
(`Dockerfile.fedora44`, CUDA 13.4), NVIDIA driver R615 and Electron 44 browsers. Each figure
is from two or three 10 s samples after the show settled. Every show started in about 14 s and
missed 0 playout deadlines while measured.

| Show | Host CPU idle | avplumber | Electron | CPU PSI "some" | GPU | NVDEC | NVENC | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1080p60, 68 = 18 NVDEC + 29 browser + 17 raw + 4 keys | 52% | 4.38 cores | 2.91 cores | 6.9% | 50% | 81% | 61% | 7.7 GB |
| 1080p30, 110 = 36 NVDEC + 36 browser + 34 raw + 4 keys | 55% | 4.42 cores | 2.20 cores | 4.6% | 55% | 86% | 39% | 8.6 GB |
| 1080p25, 110 = 40 NVDEC + 36 browser + 30 raw + 4 keys | 64% | 3.54 cores | 1.78 cores | 10% | 45% | 79% | 32% | 8.7 GB |

On this stack the cut-spam gate (`--mix 6:2:2`) passes at 68@60. Cut latency was p95 46 ms and
max 52 ms during spam, and p50 37 ms in recovery. The program missed 0 of 7320 deadlines.

The same 68@60 show on the Ubuntu 22.04 image (CUDA 11.7) with driver R595 measured:
- avplumber 4.58 cores and host CPU idle 50%;
- CPU PSI 16%;
- GPU 62-64%.

The mixer's anonymous memory was 2.29 GB on both stacks. At 1080p30 the limit is NVDEC, near
the setup's ~90% budget. The busiest single thread, the program compositor, used about 30% of
one core.

### 60 fps on the current stack

The 68 limit dated from before the mostly-still browser page and Electron 44. Measured again on
2026-10-01 (SDR, PVW follower on, cut-spam gate `--mix 6:2:2` at 4/s; then a 10 s CPU sample
and a 15 s GPU sample):

| Total | Mix | Gate | Input repeats | Host CPU idle | CPU PSI | Electron | avplumber | GPU | NVDEC | VRAM |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 68 | 18 NVDEC + 29 browser + 17 raw + 4 keys | pass, 0 missed | 156 | 52% | 8.0% | 2.81 cores | 4.50 cores | 64% | 85% | 8.2 GB |
| 72 | 18 + 33 + 17 + 4 | pass, 0 missed | 6 125 (start-up) | 49% | 9.9% | 3.05 cores | 4.67 cores | 70% | 86% | 7.9 GB |
| 75 | 18 + 36 + 17 + 4 | pass, 0 missed | 113 | 45% | 12.5% | 3.39 cores | 4.85 cores | 72% | 86% (max 90) | 8.8 GB |

75 fills every per-rate cap (NVDEC 18, browser windows 40, raw upload units 17), so the setup
allows 75 at 60 fps, and 82 at 50 (the caps scaled by frame rate). Cut latency at 75 was p95 51 ms; PVW followed PGM within one frame.
The GPU keeps headroom; host CPU is the margin. At 75 about 8.5 of the 16 vCPUs were busy:

| Inputs | CPU per input | Total | Where |
| --- | ---: | ---: | --- |
| 40 browser windows (keys included) | about 0.10 core | 3.9 cores | Electron renderers, compositor and GPU process 3.45; avplumber receive and import 0.4 |
| 17 raw NV12 | about 0.135 core | 2.3 cores | the file read (`input_N`, 3.1 MB a frame) 1.25; the upload (`upload_N`) 1.0 |
| 18 NVDEC | about 0.025 core | 0.45 core | demux and decode threads; decoding itself runs on NVDEC |
| shared mixer work | | 1.3 cores | EventLoop, the two scene compositors, colour, aux and encoders |

The 1-minute load average reached 19 with 45% of the CPU idle: about 200 threads wake on every
60 Hz tick, so runnable threads queue in bursts (CPU PSI) rather than the CPU running out.

At 30 fps the 110-input ceiling is the declared baseline for this host, keys included:
**36 NVDEC + 36 browser + 34 raw NV12 + 4 key pages**, with pinned raw uploads
(`canvas.raw_upload: "pinned"`, the setup default) and a fifth browser worker; it has not
been pushed further. At 25 fps the setup allows the same 110 as **40 NVDEC + 40 browser +
30 raw NV12**, which has not been measured as a whole. NVDEC is capped at about 1100 decoded
frames/s (near 90%; 40 streams at 25 fps measured 81%): 40 at 25 fps, 36 at 30, 22 at 50 and
18 at 60.

### 10-bit canvases

Measured 2026-10-01 at 1080p30 on the current stack: portrait canvas, 4 key pages included,
PGM rendered in SDR and HLG. The table first takes the SDR 110 show's mix (36 : 36 : 34 SDR
NVDEC : browser : raw NV12) to each total, every input converted to HLG, the costliest mix a
10-bit canvas can carry; the last row is the setup page's Balanced HDR mix, which turns half
of the decodes and uploads HLG. Each point ran the cut-spam gate (`--mix 6:2:2`, 60 s at 4/s)
and then a 15 s GPU sample. "Input repeats" counts inputs showing their previous frame on a
program tick over the gate; the SDR 110 show has about 360.

| Canvas | Total | Gate | Input repeats | GPU avg / max | NVDEC avg / max | NVENC | VRAM | Host CPU idle |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P010 (HLG 4:2:0) | 82 | pass | 319 | 79% | 81% | 63% | 8.3 GB | 66% |
| P010 | 90 | pass | 267 | 83 / 87% | 90 / 96% | 66% | 9.5 GB | 62% |
| P010 | 95 | pass, sources fall behind | 28 316 | 87 / 91% | 99 / 100% | 69% | 11.4 GB | 56% |
| P010 | 100 | pass, sources fall behind | 76 995 | 92 / 96% | 100% | 73% | 12.3 GB | 54% |
| P210 (HLG 4:2:2) | 60 | pass | 343 | 58% | 48% | 55% | 6.4 GB | 74% |
| P210 | 80 | pass | 227 | 76 / 80% | 74 / 75% | 62% | 8.3 GB | 66% |
| P210 | 90 | pass | 235 | 83 / 86% | 89 / 94% | 66% | 9.3 GB | 63% |
| P210 | 110 | fail: CUDA out of memory under spam, 59 missed deadlines | 23 052 | — | 90% | — | 12.8 GB after the failure | — |
| P010, Balanced: 16 + 16 HLG NVDEC, 31 browser, 12 NV12 + 11 P010 | 90 | pass | 186 | 75 / 78% | 65 / 68% | 55% | 10.2 GB peak in spam | 59% |

Per source, both 10-bit canvases cost about the same GPU time, roughly 1.8x an SDR source
(every SDR input is converted to HLG). Above 90 the GPU-side work slows NVDEC to saturation:
the program still meets its deadlines, but decoded sources fall behind and frames back up
in VRAM, which at 4:2:2 and 110 ran the 15 GB T4 out of memory. HLG inputs skip that
conversion: the Balanced HDR mix at 90 leaves GPU and NVDEC headroom. The limit holds for any
mix the page allows, so the setup caps HLG 4:2:0 at 90 / 110 = 0.82 of the SDR total, and
applies the same share at 50 fps (73) and 60 fps (61).

At 60 fps the 61-input Balanced HDR 4:2:0 mix (9 SDR + 9 HLG NVDEC, 28 browser, 6 NV12 + 5 P010,
4 keys, clean feed on) first failed on NVENC, not on its inputs. An HDR show encodes the program
in SDR (H.264) and HLG (HEVC Main10) plus the SDR clean feed, three 1080p60 encodes and two 30 fps
multiviews on the T4's single NVENC. At preset p5 NVENC saturated: the program met every deadline
but encoded output fell seconds behind (cut p95 1.6-2.9 s). The demo recipes now use p3, which
keeps the same low-latency settings (`tune=ull`, CBR, no lookahead, no B-frames) and trades a
little quality per bit for encoder time:

| 60 fps, HDR 4:2:0, 61 | Gate | Cut p95 | NVENC p50 / p95 | GPU p50 / p95 | NVDEC p50 / p95 | Peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| p5, clean feed | fail | 1.6-2.9 s | 100% | 72 / 87% | 86 / 89% | 10.6 GB |
| p5, no clean feed | pass, 0 missed (twice) | 53 ms | 73% | 82 / 89% | 77 / 83% | 10.1 GB |
| p3, clean feed | pass, 0 missed | 51 ms | 70 / 76% | 83 / 91% | 78 / 84% | 9.5 GB |

Above 61 the GPU's CUDA compute runs out (p3, clean feed): 64 passed once and then missed 1
deadline with a 123 ms cut (GPU p50 87-89%, p95 94%); 66 missed 2 (GPU p50 88%, p95 95%). The
limit stays at 61.

HLG 4:2:2 at 60 fps (P210 canvas, four HLG v210 inputs unpacked on the GPU, p3, clean feed) costs
more GPU per source than 4:2:0, which at 30 fps did not show because the GPU had room:

| 60 fps, HDR 4:2:2 | Gate | Input repeats | GPU p50 / p95 | NVDEC p50 / p95 | NVENC | Peak VRAM |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 61 | fail, 4 missed | 4 500 | 87 / 96% | 77 / 83% | 73% | 10.2 GB |
| 57 | pass, 0 missed (twice) | 177; 5 750 | 87-88 / 92-93% | 78-80 / 83-84% | 71-72% | 9.9 GB |
| 55 | pass, 0 missed | 98 | 80 / 86% | 77 / 81% | 71% | 9.2 GB |

The setup takes 55, the margin 61 has at 4:2:0, as the 4:2:2 share: 55 / 75 = 0.74 at every rate
(81 at 25/30 fps; 90 passed at 30 fps but is no longer allowed).

The browser service defaults to five workers with eight windows each (40 total); the setup
allows all 40 at every rate, downstream-key pages included.
The setup allows 192 scenes; scenes describe layouts and do not each allocate a
running compositor. Active layers and AUX outputs have separate limits.

A temporary six-worker, 42-browser experiment at 110 sources kept the 25-fps
upload and NVDEC counts unchanged:
**40 NVDEC + 42 browser + 28 raw NV12**, six browser workers and a six-frame
browser ring. It tests additional browser capacity rather than raising upload
traffic. The first run maintained 25 fps between two multi-second stalls,
with a 2.9-second cut, VRAM peaking near 14.1 GiB and up to 145 browser imports
pending cleanup. It recovered without intervention. Remote validation overlapped
the stalls, so this is not an isolated capacity measurement; it does not validate
110 at 25 fps. The instance was returned to the 100-source baseline, and the
browser limit was restored to 32 across four workers (since raised to 40 across
five). The 30 fps baseline above was measured separately (36 NVDEC + 36 browser +
34 raw NV12 + 4 keys, pinned uploads); this 25 fps experiment says nothing about it.

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

These runs used FFmpeg `hwupload` from pageable memory and up to 40 NVDEC streams, before
pinned uploads and the 36-stream cap at 30 fps; the 30 fps baseline above supersedes
them.

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

The setup enforces 30 raw NV12 uploads at 25 fps and 34 at 30 fps (28 and 23 were
measured with FFmpeg hwupload; 34 at 30 fps is part of the measured baseline with pinned
uploads, 30 at 25 fps relies on them), 20 at 50 and 17 at 60 fps.
Excess allocation is redistributed among enabled source types; a raw-only
request above its limit is rejected. This cap concerns the raw NV12
CPU-to-GPU path. P010 uploads share the same budget at two units per source,
based on double the bytes per frame; this is not a measured HDR capacity. Combined
SDR/HDR NVDEC inputs are capped at 40, 36, 22 and 18 at 25, 30, 50 and 60 fps.
The separate v210 upload/unpack path is not calibrated by this
measurement. Higher browser capacity does not increase the upload allowance.
The measurements in this section used FFmpeg `hwupload` from pageable memory, still the
mixer's own default. `canvas.raw_upload: "pinned"`, which the generic setup's recipe
selects, uploads through `raw_to_cuda` instead (pinned staging, a private stream per
source); the 30 fps baseline above is measured on that path, the browser-refresh
ceiling below is not.

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
long-running capacity limit. The generic setup's 25 fps upload maximum (28 then,
30 now with pinned uploads) must not be interpreted as safe independently of
browser load and allocation churn.

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
