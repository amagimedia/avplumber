# Mixer demo capacity

These numbers come from a 16 GiB NVIDIA T4 host (16 vCPU) with 1920×1080 inputs. The
per-rate baselines below stand on the source mixes listed beside them; the older
25 and 30 fps measurements that follow used SDR 8-bit inputs with PGM and one AUX
output. None of them are decoder-only limits or guarantees for HDR, larger frames,
arbitrary browser pages, or additional outputs.

The setup applies these limits as the `tesla_t4` profile in `instance_profiles.py`
(`webui.py --instance-type tesla_t4`): `setup_runtime.py` enforces it and serves it to
`setup.html`. [Source limits by mode and frame rate](cookbook/source-limits.html) lists
them; this page has the measurements behind them. Another machine needs its own measured
profile, not these numbers scaled.

| Input fps | Measured show, keys included | Result |
| --- | --- | --- |
| 25 | 110 = 40 NVDEC + 36 browser + 30 raw NV12 + 4 keys | 0 missed playout deadlines in steady state on the current stack (below) |
| 30 | 110 = 36 NVDEC + 36 browser + 34 raw NV12 + 4 keys, pinned uploads | The declared baseline; not pushed further |
| 50 | 82 = 22 NVDEC + 36 browser + 20 raw NV12 + 4 keys | Cut-spam gate passes, 0 missed (see [50 fps](#50-fps)) |
| 60 | 75 = 18 NVDEC + 36 browser + 17 raw NV12 + 4 keys | Cut-spam gate passes, 0 missed (see [60 fps on the current stack](#60-fps-on-the-current-stack)); 68 inputs ran a 19.5 h soak with 24 missed deadlines (0.0006%); that show's median cut latency is about 43 ms |

Before the mostly-still browser test page (commit 5068a50) and Electron 44, the 68-input
60 fps show measured GPU SM about 88%, NVDEC 88% and 19% host CPU idle, and 70 inputs
saturated the GPU (SM 92-94%, NVDEC 95%). Two defaults make 60 fps shows hold under cut,
fade and wipe spam (`demos/mixer/tests/cut_spam.py`): a 3-frame playout deadline at 50/60 fps (50 ms at 60; two frames
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

With every browser page moving all the time (the test page's marker never resting) the same 75@60
show still passed the cut-spam gate with 0 missed deadlines, but without margin: Electron rose from
3.4 to 6.5 cores (host CPU 24% idle, CPU PSI 19%), and Chromium rasterising and compositing 40
repainting windows on the same T4 took GPU to p50 96% / p95 99% and NVDEC to 91-97%. The limits
assume pages that rest like real graphics; constantly animated pages need fewer browser sources.

At 30 fps the 110-input ceiling is the declared baseline for this host, keys included:
**36 NVDEC + 36 browser + 34 raw NV12 + 4 key pages**, with pinned raw uploads
(`canvas.raw_upload: "pinned"`, the setup default) and a fifth browser worker; it has not
been pushed further. NVDEC is capped at about 1100 decoded
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

### 50 fps

50 fps takes the 60 fps limits scaled by frame rate ([how the limit is computed](cookbook/source-limits.html)). Measured 2026-10-01
with the Balanced mixes, clean feed on, NVENC p3:

| 50 fps | Mix with 4 keys | Gate | Cut p95 | Input repeats | GPU p50 / p95 | NVDEC p50 / p95 | NVENC | Peak VRAM | Host CPU idle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HLG 4:2:2, 66 | 11 + 11 NVDEC, 3 HLG v210, 24 browser, 7 NV12 + 6 P010 | pass, 0 missed | 57 ms | 5 278 | 79 / 85% | 76 / 80% | 59% | 11.5 GB | 47% |
| HLG 4:2:0, 73 | 10 + 12 NVDEC, 34 browser, 7 NV12 + 6 P010 | pass, 0 missed | 56 ms | 2 626 | 85 / 91% | 77 / 83% | 59% | 10.9 GB | 48% |
| SDR, 82 | 22 NVDEC, 36 browser, 20 NV12 | pass, 0 missed | 51 ms | 3 262 | 69 / 92% | 86 / 90% | 35% | 12.0 GB | 47% |

Input repeats are higher than at 30 or 60 fps (100-600 per gate there): browser pages paint on
a 60 Hz rhythm that does not divide into a 50 fps grid. The program met every deadline.

The browser service defaults to five workers with eight windows each (40 total); the setup
allows all 40 at every rate, downstream-key pages included.
The T4 profile allows 192 scenes; the L4 profile allows 256. Scenes describe layouts and do not each allocate a
running compositor. Active layers and AUX outputs have separate limits.

### NVENC and extra aux outputs

Every encoded output has its own NVENC preset (p1, p3 or p5) and CBR bitrate (2–20 Mbit/s),
set under **Outputs** on the setup page; the **Extra aux** outputs fill what the others leave of
the profile's NVENC budget (`nvenc` in the `tesla_t4` entry). An encode takes its encoded frames/s
times its codec's share per frame/s at its preset, scaled by its pixels against 1920×1080:

| 1920×1080 encode, % of the T4's NVENC per frame/s | p1 | p3 | p5 |
| --- | ---: | ---: | ---: |
| H.264 | 0.204 | 0.219 | 0.466 |
| HEVC Main10 (HLG program) | 0.153 | 0.305 | 0.442 |

Measured on 2026-10-02 at 1080p30, `tune ull`, CBR, no B-frames. On the live show five H.264
encodes took 32.8% at p3 (nine 58.2%, about 6.5% each) and 70.0% at p5. Two real-time 1080p30 test
encodes beside it took, per encode, 6.0%, 6.5% and 14.9% for H.264 p1, p3 and p5, and 4.5%,
9.0% and 13.1% for HEVC Main10; the table anchors them to the show's 0.219 for H.264 p3. p2,
p4, p6 and p7 are not offered (the renditions' own default, p7, saturated NVENC with seven
encodes). The bitrate does not enter: CBR NVENC time is roughly independent of the bitrate,
which is still to be verified.

Every encode counts: the H.264 program, the HEVC HLG program on a 10-bit canvas, the H.264 clean
feed, counted even while off, and each aux bus, which encodes at the program rate at 25/30 fps and
at half of it at 50/60 (a `full_rate` bus at the program rate). Extra outputs, each at the
**Extra aux** preset, take what is left of 80% (`setup_runtime.extra_aux_limit`); a setup whose
encodes exceed 80% without them is refused, with the share they need. With the live show's two aux
buses (Program preview and Multiviewer) at the defaults, the programs and the clean feed at p3 and
every aux bus at p1:

| Canvas | 25 fps | 30 fps | 50 fps | 60 fps |
| --- | ---: | ---: | ---: | ---: |
| SDR, 8-bit | 11 | 8 | 9 | 6 |
| HLG, 10-bit | 10 | 7 | 6 | 3 |

Every encode at p3 leaves 10, 8, 8 and 6 extra outputs on an SDR canvas and 9, 6, 5 and 3 on a
10-bit one. The programs and the clean feed at p5, the aux buses at p1, leave 9, 6, 4 and 1 on an
SDR canvas; on a 10-bit one 6, 4 and 0 at 25, 30 and 50 fps, and at 60 fps the encodes need
94.7%, so that setup is refused (the SDR program at p3 fits).

Each aux bus also composites on the GPU, which has not been measured per bus: the source
limits do not change with the number of extra outputs.

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

## NVIDIA L4 (`nvidia_l4`)

<!-- L4-CAPACITY-2026-10-05:BEGIN -->
### CUarray reconfiguration and output capacity, 2026-10-05

The [L4 capacity JSON](capacity-l4.json) records the current FFmpeg development build
with direct CUarray decode, fixed `extra_hw_frames: 12` pools and three-frame decoded queues.
These are 1920×1080 sources on one L4 / g2-standard-16. Program renditions run at the
canvas rate; AUX outputs run at 25 fps for a 25 fps show and 30 fps for a 60 fps show.
Totals include four browser keys. HDR 4:2:2 uses a P210 canvas and native v210 inputs;
the L4 HEVC inputs in this test are 4:2:0.

| Mode | Inputs / scenes | Extra AUX / total outputs | Observation | Missed deadlines / output drops | Result |
| --- | ---: | ---: | --- | ---: | --- |
| HLG 4:2:0, 25 fps | 136 / 128 | 25 / 30 | 30 s cuts | 0 / 0 | zero-miss observation |
| HLG 4:2:0, 25 fps | 136 / 256 | 15 / 20 | 80 s cuts | 0 / 0 | zero-miss observation |
| HLG 4:2:0, 25 fps | 136 / 128 | 25 / 30 | 20 s steady | 0 / 0 | zero-miss observation |
| HLG 4:2:0, 60 fps | 88 / 128 | 13 / 18 | 30 s cuts | 0 / 0 | zero-miss observation |
| HLG 4:2:0, 60 fps | 88 / 128 | 15 / 20 | 30 s cuts | 2 / 0 | gate not met |
| HLG 4:2:0, 60 fps | 88 / 256 | 13 / 18 | 80 s cuts | 1 / 0 | gate not met |
| HLG 4:2:0, 60 fps | 88 / 128 | 15 / 20 | 20 s steady | 0 / 0 | zero-miss observation |
| HLG 4:2:2, 25 fps | 132 / 256 | 15 / 20 | 30 s cuts | 0 / 0 | zero-miss observation |
| HLG 4:2:2, 25 fps | 132 / 128 | 25 / 30 | 20 s steady | 0 / 0 | zero-miss observation |
| HLG 4:2:2, 60 fps | 88 / 128 | 12 / 17 | 30 s cuts | 0 / 0 | zero-miss observation |
| HLG 4:2:2, 60 fps | 88 / 128 | 15 / 20 | 30 s cuts | 51 / 0 | gate not met |
| HLG 4:2:2, 60 fps | 88 / 128 | 15 / 20 | 20 s steady | 4 / 0 | gate not met |
| SDR 4:2:0, 25 fps | 192 / 256 | 22 / 26 | 80 s cuts | 0 / 1 | gate not met |
| SDR 4:2:0, 25 fps | 192 / 256 | 22 / 26 | 80 s cuts | 0 / 0 | zero-miss observation |
| SDR 4:2:0, 25 fps | 192 / 128 | 26 / 30 | 30 s cuts | 0 / 0 | zero-miss observation |
| SDR 4:2:0, 25 fps | 192 / 256 | 26 / 30 | 80 s cuts | 693 / 856 | gate not met |
| SDR 4:2:0, 60 fps | 99 / 128 | 18 / 22 | 30 s cuts | 0 / 0 | zero-miss observation |
| SDR 4:2:0, 60 fps | 99 / 128 | 18 / 22 | 30 s cuts | 1 / 0 | gate not met |

**256-scene coverage:** HLG 4:2:0 at 25 fps, 136 sources and
20 outputs completed 80 seconds with all 256 scenes requested,
256 successful cut requests and 0 failed requests. The capture had zero missed
deadlines, output drops or stalled captured edges; VRAM peaked at 21463 MiB.

**256-scene coverage:** SDR 4:2:0 at 25 fps, 192 sources and
26 outputs completed 80 seconds with all 256 scenes requested,
256 successful cut requests and 0 failed requests. The capture had zero missed
deadlines, output drops or stalled captured edges; VRAM peaked at 19295 MiB.

The 192-input SDR25 show with 30 outputs passed an earlier six-cut probe but failed
the full 256-scene sweep: 693 missed deadlines across program/AUX timelines,
856 AUX output drops and NVENC p95 of 100%. At 26 outputs the first sweep had one
AUX drop; the repeat completed all 256 requests with zero missed deadlines or drops.
Both observations are retained. The SDR output ceiling is now 26 at 25/30 fps and
22 at 50/60 fps; 30/50 fps values remain derived rather than newly tested.


The final HLG 4:2:0 / 60 fps sweep used 88 sources, 18 outputs and all 256 scenes.
It recorded 1 missed deadline and 0 output drops, with no stalled captured edges.
Decoder and program rates stayed near 60 fps and AUX rates near 30 fps, but this
**did not meet the zero-miss gate**. No further sweep was run before handoff.


Configured admission limits also bound 10-bit NVDEC input count and total encoded outputs:

| fps | Maximum 10-bit NVDEC inputs | SDR outputs | HLG 4:2:0 outputs | HLG 4:2:2 outputs | Evidence |
| ---: | ---: | ---: | ---: | ---: | --- |
| 25 | 53 | 26 | 20 | 20 | specific combinations measured here |
| 30 | 53 | 26 | 20 | 20 | derived; not tested in this update |
| 50 | 32 | 22 | 18 | 17 | derived; not tested in this update |
| 60 | 27 | 22 | 18 | 17 | specific combinations measured here |

These are configuration guards, not measurements of every allowed combination. The largest
10-bit decode counts in completed zero-miss captures here are 53 at 25 fps and 26 at 60 fps;
the configured 27-input 60 fps bound was not directly exercised. The 30/50 fps settings are
derived and have no completed captures in this update.


The initial 30-second, six-cut HDR60 probes passed with 13 extra AUX (18 total outputs) at 4:2:0 and
12 extra AUX (17 total) at 4:2:2. The earlier 15-extra-AUX setting missed 2 and 51
deadlines respectively during cuts, despite finishing startup. The setup now reserves
more encoding headroom for HDR. Steady observations and cut tests are identified separately;
none is a long soak. SDR runs are listed separately, including
the first one-miss 60 fps result; a later passing run does not invalidate that observation.

The JSON includes measured decoder and encoder rates, CPU and GPU statistics, VRAM,
power, counter deltas and cut latency. Expected rates are labelled separately. Cut events
are deduplicated across samples, with pre-existing events excluded. Browser freshness is
not covered by the captured queue counters; encoded output alone cannot prove every source is fresh.
Missed deadlines sum independent program and AUX timelines. An encoder can retain its target
FPS by repeating after backpressure; the gate also requires zero output drops.

Preparation and restart timings are separate in the JSON. Uncached assets are prepared
while the previous show runs. The restart pauses output; the reported restart-to-ready
interval comes from setup polling and is not a measured browser blackout duration.

Startup failures are recorded separately from completed captures. The source mix matters:
a balanced HDR show passing does not qualify the same count of native HDR decodes.

| Failed configuration | Scenes | Intended outputs | Observed VRAM | Result |
| --- | ---: | ---: | ---: | --- |
| HLG 25 fps: 96 HLG HEVC + 0 SDR HEVC + 40 browser | 128 | 25 | 22643 MiB | NVENC initialization out of memory |
| HLG 25 fps: 96 HLG HEVC + 0 SDR HEVC + 40 browser | 256 | 15 | 22637 MiB | NVENC initialization out of memory |

The failed startup snapshot is not a sampled peak or a successful capacity result.
The 43-SDR / 53-HLG / 40-browser show completed the 25 fps cut test with 30 outputs
and a sampled VRAM maximum of 22002 MiB; changing all 96 decodes to HLG exhausted
startup headroom even with only 25 intended outputs. Automatic rollback to the previous
balanced 136-input / 30-output show also ran out of memory and required a manual
container stop. A second attempt with 256 scenes and only 15 intended outputs
also failed at encoder initialization; immediate graph shutdown then blocked in
native code, requiring another manual stop. Those failed configurations do not validate
automatic recovery; the later balanced 256-scene pass is a separate configuration.
Completed steady and cut captures do not cover all startup or
rollback allocation transients.

<!-- L4-CAPACITY-2026-10-05:END -->

### Earlier measurements, 2026-10-02

One L4 (two NVENC and four NVDEC engines, 24 GiB) on a GCP g2-standard-16 (16 vCPU), measured
2026-10-02. Every show ran the full outputs: the program at p5 (H.264, plus HEVC Main10 on an HLG
canvas), the clean feed, Program preview, Multiviewer and three extra aux at p3, and four keys. The
inputs are synthetic 1920×1080 patterns with grain at 3–10 Mbit/s (16 peak). Each row passed the
cut-spam gate (60 s, four cuts, fades or wipes per second).

A limit keeps NVDEC and the GPU at 90% of the idle show and 10% of the CPU idle. NVDEC fills first,
then the 40 browser windows (keys included), raw uploads last.

| Canvas | fps | Inputs | NVDEC | Browser | Raw upload | NVDEC idle | NVDEC cutting (mean, p95) | GPU idle | GPU cutting (p95, max) | CPU idle | Cut p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SDR 4:2:0 | 30 | 150 | 83 | 40 | 27 NV12 | 88% | | 42% | | 59% | 74 ms |
| SDR 4:2:0 | 30 | 170 | 83 | 40 | 47 NV12 | 88% | | 54% | | 46% | 75 ms |
| SDR 4:2:0 | 60 | 85 | 41 | 40 | 4 NV12 | 84% | | 39% | | 64% | 42 ms |
| SDR 4:2:0 | 60 | 96 | 41 | 40 | 15 NV12 | 91% | 85%, 90% | 58% | 61%, 67% | 50% | 43 ms |
| SDR 4:2:0 | 60 | 100 | 41 | 40 | 19 NV12 | 91% | | 63% | | 47% | 44 ms |
| HLG 4:2:0 | 60 | 85 | 44 | 40 | 1 NV12 | 78% | | 68% | | 64% | |
| HLG 4:2:0 | 60 | 88 | 48 | 40 | none | 89% | 85%, 94% | 70% | 77%, 84% | 62% | 46 ms |
| HLG 4:2:0 | 60 | 93 | 52 | 40 | 1 NV12 | 98% | | 74% | | 54% | 147 ms, gate failed |
| HLG 4:2:2 | 60 | 88 | 48 | 40 | none | 91% | | 74% | | 62% | 46 ms |
| HLG 4:2:2 | 60 | 88 | 44 | 40 | 4 v210 | 84% | 82%, 95% | 75% | 85%, 91% | 58% | 47 ms |
| HLG 4:2:2 | 60 | 92 | 48 | 40 | 4 v210 | 95% | | 79% | | 56% | 49 ms |
| HLG 4:2:2 | 60 | 92 | 44 | 40 | 8 v210 | 86% | 88%, 99% | 81% | 95%, 98% | 49% | 49 ms |

What the profile takes from them:

- **SDR**: 170 at 25/30 fps, where the CPU is the limit. 41 H.264 decodes at 60 fps read 91% beside
  15 and beside 19 uploads, so the profile takes 40 and the 19 uploads: 99 inputs.
- **HLG 4:2:0**: half the decodes are HEVC Main10, which loads NVDEC less than H.264, so 48 fit at
  60 fps. No raw upload fits beside them: 88 inputs.
- **HLG 4:2:2**: NVDEC decodes 4:2:0 only, so the native 4:2:2 inputs are v210 uploads and the
  canvas takes no 4:2:0 upload. Every upload raises the NVDEC load of the same decodes (48 decodes:
  91% beside none, 95% beside four v210), so four decodes make room for four v210: 88 inputs. Eight
  v210 pass at rest and take the GPU to 98% while cutting.
- **Scaled, not measured**: 50 fps takes the 60 fps decodes and uploads by frame rate, the HLG
  canvases at 25/30 fps twice their 60 fps decodes.
- **NVENC**: not measured per preset; the profile has the T4's costs over 2.2. The full outputs took
  22% on SDR at 30 fps, 30% at 60 and 49% on HLG at 60, within 5 points of that model.

| Canvas | 25/30 fps | 50 fps | 60 fps |
| --- | --- | --- | --- |
| SDR 4:2:0 | 170 = 83 NVDEC + 40 browser + 47 raw | 110 = 48 + 40 + 22 | 99 = 40 + 40 + 19 |
| HLG 4:2:0 | 136 = 96 + 40 | 97 = 57 + 40 | 88 = 48 + 40 |
| HLG 4:2:2 | 132 = 88 + 40 + 4 v210 | 96 = 52 + 40 + 4 v210 | 88 = 44 + 40 + 4 v210 |

The limits a canvas measured apart are the profile's `mode_limits`; `setup_runtime.for_mode`
applies them.
