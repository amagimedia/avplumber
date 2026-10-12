# Mixer capacity

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

Use the [portable L4 stack](../../../docker-compose/mixer/deploy/l4/README.md). Browser frames must
go back to the browser one per new frame ([`ack_holdback`](../../../doc/NODES.md#ipc_dmabuf_source)),
which the mixer sets; without it the browsers keep allocating buffers and the releases stall the show.

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
