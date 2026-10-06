# Recorded mixer capacity

Setup admission limits live in [instance_profiles.py](https://github.com/amagimedia/avplumber/blob/mixer-improv/demos/mixer/instance_profiles.py).
They depend on source mix, canvas rate/format, output count and encoder preset.
Do not scale one GPU's profile to another GPU. This page summarizes historical
runs; raw evidence and caveats remain linked below.

[Source-limit cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/source-limits.html) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)

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

## NVIDIA L4 (`nvidia_l4`)

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
fixed-pool sizing. Use the [portable L4 stack](https://github.com/amagimedia/avplumber/blob/mixer-improv/docker-compose/mixer/deploy/l4/README.md) for that build.
After deployment, exercise actual content, every scene and Setup restart; check
fresh deadline/drop counter deltas, not just encoder FPS.
