# Recorded mixer capacity

Setup admission limits live in [instance_profiles.py](../instance_profiles.py).
They depend on source mix, canvas rate/format, output count and encoder preset.
Do not scale one GPU's profile to another GPU. This page summarizes historical
runs; raw evidence and caveats remain linked below.

[Source-limit cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/source-limits.html) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)

## Current demo recording: 2026-10-09

The [10-second UI recording](https://amagimedia.github.io/avplumber/demos/mixer/docs/)
shows a running `nvidia_l4_cuarray` show on one NVIDIA L4 / 16-vCPU host:
192 inputs, 256 scenes and 26 outputs at 1920×1080p25 SDR. The input total is
120 NVDEC + 36 browser scene sources + 32 raw NV12 + four browser keys;
Setup reports 188 catalogue sources before adding the keys. Outputs are
Program, clean Program and 24 AUX buses, including the Program/Preview monitor
and source multiviewer.

The capture records both encoded outputs on the instance before Internet
transport, then replays them with recorded UI state and cuts at 25 fps. Four
keys stay enabled through direct cuts across several scene types. The native
ticker advances four pixels on each of 249 consecutive frame pairs in the
10-second excerpt; this checks capture cadence, not long-running capacity. It is an interface demonstration, not a new capacity,
latency or soak benchmark. The [capture metadata](https://amagimedia.github.io/avplumber/demos/mixer/docs/capture-20261009.json)
records its scope; the historical qualification results below remain separate.

## T4

16 GiB T4, 16 vCPUs, 1080p inputs; totals include four browser keys.
The September 2026 tests used mostly-still browser test pages and raw uploads.

| fps | Sources | Mix: NVDEC / browser including keys / raw |
| ---: | ---: | --- |
| 25 | 110 | 40 / 40 / 30 |
| 30 | 110 | 36 / 40 / 34 |
| 50 | 82 | 22 / 40 / 20 |
| 60 | 75 | 18 / 40 / 17 |

These are tested/declared profile baselines, not decoder-only limits. Short
50/60 fps cut-spam checks passed; a separate 68-source 60 fps, 19.5-hour soak
recorded 24 missed deadlines. Browser content can change CPU/GPU demand greatly.
See [browser cost](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/browser-paint-cost.html), [raw uploads](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/raw-uploads.html)
and [CPU pressure](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/cpu-pressure.html) for the measurement context.

## NVENC and extra aux outputs

Every rendition, reserved clean feed and AUX consumes encoder capacity.
Setup applies both modeled NVENC costs and measured output ceilings. Codec,
preset, resolution and FPS affect admission; lowering bitrate does not raise
that allowance. AUX normally runs at half rate above 30 fps. Use Setup's
calculated limit rather than a copied table of extra-output counts.

## NVIDIA L4: linear and CUarray profiles

`nvidia_l4` and `nvidia_l4_cuarray` are distinct profiles. The CUarray SDR25
profile admits 192 total sources: 120 NVDEC, 32 raw, 36 browser scene sources
and four browser keys (188 sources in the catalogue). The linear SDR25 profile
uses 170 total: 83 NVDEC, 47 raw and 40 browser including keys.

Current SDR output ceilings are 26 at 25/30 fps and 22 at 50/60 fps, below the
hard 30-output encoder ceiling. HLG 4:2:0 uses 20/20/18/18, and HLG 4:2:2 uses
20/20/17/17 at 25/30/50/60 fps. These are admission rules; not every entry has a
separate measured qualification. Read the profile comments and trial scope.

The [2026-10-05 raw report](https://amagimedia.github.io/avplumber/demos/mixer/docs/capacity-l4.json) covers 1080p inputs on one L4 /
16-vCPU host, patched FFmpeg, HEVC CUarray decode, `extra_hw_frames: 12` and
three-frame decoded queues. Totals include keys; HDR 4:2:2 inputs use v210.

| Show | Sources / scenes / outputs | Observation |
| --- | --- | --- |
| SDR25 | 192 / 256 / 26 | First sweep: 1 AUX drop; repeat: 0 missed deadlines / 0 drops |
| SDR25 | 192 / 256 / 30 | 693 missed deadlines / 856 drops; failed |
| HLG 4:2:0, 25 fps | 136 / 256 / 20 | 80-second sweep: 0 missed / 0 drops |
| HLG 4:2:0, 60 fps | 88 / 256 / 18 | 1 missed deadline; failed zero-miss gate |
| HLG 4:2:2, 60 fps | 88 / 128 / 17 | 30-second cut probe: 0 missed / 0 drops |

The raw report retains all trials, including failed startup/rollback when 96
HDR decoders exhausted VRAM. A balanced show passing does not qualify the same
count of HDR decoders. Short probes do not qualify every scene or a long soak;
30/50 fps admission values in that update were derived, not newly measured.

The [CUarray cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/nvdec-cuarray.html) explains storage savings and
fixed-pool sizing. Use the [portable L4 stack](../../../docker-compose/mixer/deploy/l4/README.md) for that build.
After deployment, exercise actual content, every scene and Setup restart; check
fresh deadline/drop counter deltas, not just encoder FPS.

## Node reductions reviewed 2026-10-06

For the captured 188-source catalogue plus four keys, native input pacing
removes 152 `force_fps` nodes/queues and the guarded raw colour-tag bypass
removes another 32. Applying those changes alone to the old 1,514-node graph
would yield 1,330; other routing/wipe changes can affect the deployed total.
The four pacing shards remain, now with 38 file pacers and ten browser pacers
each rather than 86 nodes each. These are not 184 eliminated threads.

The small mixed-rate GPU smoke verifies cadence, cuts and stall recovery; it
does not qualify 188-source throughput. No admission cap was raised based on
these node reductions. See [cadence evidence](latency.md#native-input-cadence-check-2026-10-06).
