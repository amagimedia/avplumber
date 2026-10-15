# Mixer capacity

The L4 CUarray Setup profile now caps browser windows (DSKs included) at
40/40/40/30 for 25/30/50/60 fps; the initial deployment and new AUX outputs use
`p1`. Saved explicit encode choices remain until edited or reset with Defaults.
The moving-browser measurements below require the optional fixed-pool exporter
and copy-on-import configuration; a browser count alone does not enable it.

Setup admits what [instance_profiles.py](../instance_profiles.py) allows for the GPU, the canvas
rate and format, the outputs and their encoder presets. A profile is measured on its own GPU; do
not scale one GPU's numbers to another. All figures are 1080p inputs and include four browser keys.

[Demo recording](https://amagimedia.github.io/avplumber/demos/mixer/docs/) ·
[Source-limit cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/source-limits.html) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)

## NVIDIA L4, 16 vCPU: `nvidia_l4_cuarray`

25 and 30 fps carry the same 192 sources.

| Canvas | fps | Sources | NVDEC / browser / raw | Outputs |
| --- | ---: | ---: | --- | ---: |
| SDR | 25 | 192 | 120 / 40 / 32 | 26 |
| SDR | 30 | 192 | 105 / 40 / 47 | 24 |
| SDR | 50 | 110 | 48 / 40 / 22 | 22 |
| SDR | 60 | 99 | 40 / 40 / 19 | 22 |
| HLG 4:2:0 | 25, 30 | 136 | 96 / 40 / 0 | 20 |
| HLG 4:2:0 | 60 | 88 | 48 / 40 / 0 | 18 |
| HLG 4:2:2 | 25, 30 | 132 | 88 / 40 / 4 v210 | 20 |
| HLG 4:2:2 | 60 | 88 | 44 / 40 / 4 v210 | 17 |

256 scenes. Outputs are Program, the clean feed and AUX buses, at the default presets. At 1080p30
the 192-source show ran 40 minutes with no missed deadline on Program or on any of its 22 AUX
buses and no AUX drop (2026-10-12): a draw takes about 3 ms of its 33 ms, the GPU is at 60 %
and VRAM at 19.5 of 23 GB. The SDR 25 and 30 fps rows are the measured 192-source shows; the other
rows are the profile's admission limits. The linear `nvidia_l4` profile (no CUarray decode) carries
170 sources at 25 and 30 fps.

Use the [portable L4 stack](../../../docker-compose/mixer/deploy/l4/README.md). The historical
ordinary-capture/direct-import runs used paced browser returns
([`return_queue`](../../../doc/NODES.md#ipc_dmabuf_source)) to prevent allocation churn.
The optional fixed-pool/copy-on-import path below instead releases each browser frame
after its CUDA copy and retains the full prewarmed import cache; it uses no return queue.

### Moving pages and transitions

Short transition checks on a 12-vCPU L4 (2026-10-15) used a 1080p HLG 4:2:0
canvas, 57 video sources, 18 outputs and a fixed, fully prewarmed nine-buffer
browser pool with a single CUDA copy on import. Browser counts include four
cropped keys; every other page was animated at 1920×1080 and the canvas rate.
One-second fades and cuts between two 32-layer grids were spaced six seconds
apart. These checks do not establish capacity for 64-layer transitions.

Main presets stayed SDR `p5`, HDR `p3`, clean `p3`. Each row measured four
transitions; counts exclude warm-up and intentional static holds.

| fps | Browser windows | Workers | AUX preset | Browser repeats / skipped frames | Key repeats / skipped frames |
| ---: | ---: | --- | --- | ---: | ---: |
| 25 | 40 | 4 × 10 | `p3` | 0 / 0 | 0 / 0 |
| 30 | 40 | 4 × 10 | `p3` | 0 / 1 | 0 / 0 |
| 50 | 40 | 4 × 10 | `p3` | 0–3 / 0–3 | 0 / 0 |
| 60 | 30 | 3 × 10 | `p1` | 3 / 3 | 0 / 0 |

These runs had no compositor misses or new imports. A second recorded four-fade
pass at 50 fps had zero observed browser/key repeats or drops. At 60 fps it had
one browser repeat/drop, no key repeats and two skipped key frames. Both
recordings had regular RTP cadence and changing ticker, corner and grid regions
on every decoded frame sampled. A final 50 fps rerun had three isolated browser
repeats/skips and no key issues. These are short smoke results, not zero-drop
guarantees; `p1` trades AUX encoding quality for throughput.

At 60 fps, 32 windows repeated hundreds of frames; 36 and 40 collapsed into
persistent repeats. Increasing the 32-window pool to eleven did not rescue it,
nor did reducing the main SDR preset to `p3` sustain 40 windows.

Use the L4 stack's 32 compute queues, eight copy queues and frame staggering for
this workload. Three larger browser groups failed the 40-window/50-fps test;
the grouping is part of the tested configuration. The optional fixed-pool build
and startup flags are described in the [browser configuration](../../../deps/dma-browser/README.md).
Its advertised pool size synchronizes sender retention, import caching and prewarm;
it is independent of the number of CUDA work queues.

In a matched 30-window/60-fps transition test with AUX `p1`, reducing the pool
from nine to six saved about 1 GiB but increased browser repeats from 3 to 223.
Neither pool churned. Keep nine unless the smaller pool passes the actual show.

## NVIDIA T4, 16 vCPU: `tesla_t4`

| Canvas | fps | Sources | NVDEC / browser / raw |
| --- | ---: | ---: | --- |
| SDR | 25 | 110 | 40 / 40 / 30 |
| SDR | 30 | 110 | 36 / 40 / 34 |
| SDR | 50 | 82 | 22 / 40 / 20 |
| SDR | 60 | 75 | 18 / 40 / 17 |

## Reading the limits

- Every rendition, the clean feed and every AUX bus is an encode. Codec, preset, size and rate
  count against the NVENC allowance; bitrate does not. AUX runs at half rate above 30 fps.
- Browser cost depends on what the pages paint; the figures use mostly still pages and four keys.
- After a deployment, check the show itself: the control page shows deadline misses and AUX drops
  of the last ten minutes, and `mixer.status` shows how long each bus's draws take
  ([`playout.timing`](../../../doc/NODES.md#mixer_compositor)).
