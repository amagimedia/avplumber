# CUarray mixer capacity validation

Pinned FFmpeg NVDEC CUarray output reduces decoded-image storage enough to run
192 inputs with 28 encoded outputs on the validation L4. The previous FFmpeg 8.1
configuration ran 20 outputs. This is a measured configuration, not a maximum
capacity rating for all source formats, scenes or GPUs.

The upstream revision, SDK headers and complete patched tree are pinned in
[`deps/ffmpeg/9/bases.env`](../../deps/ffmpeg/9/bases.env). The stable 8.x patch
series and default build remain unchanged. No separate decoder implementation,
AVP API change or edit to `src/nodes/decoders.cpp` was needed.

## Workload and comparison

All runs used the same 192 physical sources: 120 independent low-DPB HEVC
1080p25 clips, 32 raw inputs and 40 DMA-BUF browser inputs. Encoded demo clips
were already optimized before the baseline. Those earlier savings are excluded.
Decoded queues stay at three frames and AUX latency stays at 80 ms. Output
resolution, frame rate and encoder settings are unchanged; extra AUX outputs
use 1080p25 H264 NVENC with the existing p3 preset and bitrate.

The baseline has two program renditions, two monitors and 16 extra AUX buses.
The 28-output run adds eight AUX buses. All 128 program scenes are prewarmed.
The matched 20-output measurements use the same program and preview scenes.
No compilation or unrelated GPU tests run during the performance captures.

| 60-second warmed run | Whole-GPU VRAM mean | Mixer CPU, logical cores | Threads | GPU SM mean | NVDEC mean |
|---|---:|---:|---:|---:|---:|
| FFmpeg 8.1, linear CUDA, 20 outputs | 20.65 GiB | 6.47 | 1430 | 49.0% | 68.0% |
| Pinned upstream, CUarray, 20 outputs | 15.77 GiB | 6.01 | 1431 | 43.3% | 65.8% |
| Pinned upstream, CUarray, 28 outputs | 18.13 GiB | 6.42 | 1527 | 44.4% | 65.6% |

The matched 20-output comparison saves **4.88 GiB (23.7%)** and reduces mixer
CPU by about 7.2%. The accepted 28-output capture has **40% more outputs** while
using 2.52 GiB less VRAM than the old 20-output baseline. Whole-GPU figures include
the browser service and other resident GPU consumers, so small changes between
restarts are not decoder-pool growth. CPU is summed over mixer-container threads;
100% means one logical CPU. Per-thread and whole-host CPU samples are retained
in the private measurement reports.

`tests/cuda/nvdec/capture.py --control-port` also checks decoded, paced and encoded
edge progress. Deadline counters alone are insufficient: a failed decoder can
leave repeated pictures while downstream clocks continue normally.

## Fixed pools and zero-copy boundary

For these clips, FFmpeg requires nine codec/working surfaces plus the explicit
`extra_hw_frames` budget. Twenty outputs passed with eight extras (17 surfaces);
28 outputs needed twelve extras (21 surfaces). The default three extras is a
source-level option, not a full-show recommendation. Twelve total surfaces
exhausted at 20-output startup; seventeen exhausted with 28 outputs. Those runs
are rejected and excluded from the results above. Other bitstreams have different
reference-picture requirements. No pool grows dynamically or silently switches
to copying.

NVDEC writes registered arrays directly. Matching `scale_cuda` normalization
preserves the same array handles, and the mixer samples the planes as textures
into its output canvas. There is no intermediate decoded-image GPU copy. Per-
decoder producer streams and consumer events order these reads; frame references
survive until consumer work completes. Rendering a new output canvas remains
necessary, and optional pad/crop/transition filters produce linear CUDA output.
Crop copies its selected rectangle directly into that final output, without an
intermediate staging image.

The 32-output attempt failed the existing `one_to_many` limit of 32 destinations
for downstream-key fan-out before becoming a valid mixer run. It did not establish
a VRAM or encoder ceiling. This work leaves that graph/node limit unchanged.

## Validation and remaining boundaries

- All 28 browser wall players advanced frames and recovered after forced
  WebSocket disconnection; no page or HTTP errors occurred.
- The 28-output cut/fade/wipe stress passed: 136 cuts, 57 fades and 52 wipes,
  zero missed program deadlines, cut p95 111.2 ms and maximum 119.6 ms.
  The matched 20-output CUarray cut p95 was 96.5 ms versus 99.4 ms on FFmpeg 8.1.
  Fade/wipe first-encoded-frame latency is not instrumented by the cut probe.
- A fresh container from the built image completed a five-minute quiet capture:
  all 120 decoders and 28 encoded edges advanced at 25.00–25.01 fps, with zero
  output drops, missed deadlines or overflows. Whole-GPU memory averaged
  18.01 GiB and peaked at 18.11 GiB; mixer CPU averaged 6.50 logical cores.
  Source repeat/discard counters are also recorded; they are not output-drop
  counters and include browser cadence and changing scene subscriptions.
- The first AUX page test applied and restored 42 changes on each of three
  eligible buses, but its wider capture recorded three real output drops on
  other AUX buses about 19 seconds after cycling ended. Docker image work also
  occurred during that capture, so it is not a clean scheduling comparison;
  attribution to that work is unproven. Preserve that result as a diagnostic,
  rather than treating clean deadline counters as evidence of zero stutter.
- The subsequent quiet AUX repeat applied and restored 43 page changes on each
  of those three buses. Its two-minute capture had zero output drops, missed
  deadlines or overflows, with all 120 decoders and 28 encoders advancing at
  24.99–25.01 fps. Memory averaged 18.03 GiB and peaked at 18.13 GiB. This and
  the quiet soak are the accepted scheduling checks; they do not identify the
  cause of the earlier transient drops or guarantee arbitrary host contention.
- Real NVDEC/compositor pixels and PTS match linear CUDA. H264, HEVC Main10
  and AV1 GPU regressions match FFmpeg 8.1, and a mixed-codec graph passes.
  The 80-case CUDA filter matrix covers NV12/P010/P210, including mixed storage.
- AVP/avcpp builds and native mixer tests pass against both library versions;
  changed C++ and FFmpeg CUDA filter code pass Clang checks. Required Python
  mixer tests pass: 633 passed, 5 skipped; four native tests ran on the GPU host.
- Actual HEVC 10-bit 4:2:2 decode still needs Blackwell validation. P210 array
  sampling is tested with real synthetic GPU arrays on L4.
- Replay seeking, held frames across flush, corrupt streams and sequence changes
  remain separate milestones. An idle compositor can retain its former scene's
  pool until its next draw or destruction; it does not retain surface-index refs.
  Decoder auto-restart is disabled for CUarray to avoid overlapping replacement
  pools. CPU and linear CUDA decoder behavior and existing flush handling remain
  unchanged.

See the [implementation record](../specs/2026-10-02-nvdec-fixed-pools.md) for the
complete regression matrix and the [source configuration](../../demos/mixer/docs/config.md#sources)
for opt-in storage and fixed-budget parameters. Runtime measurements, source
paths and deployment details remain outside the repository.
