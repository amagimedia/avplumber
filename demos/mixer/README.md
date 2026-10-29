# Live mixer demo

Mix video and browser sources on a CUDA canvas, with Preview/Program,
Cut/Fade/media wipes, downstream keys and AUX monitors. Video only.

[![192-input mixer: Program/Preview and dirty Program with four downstream keys](https://amagimedia.github.io/avplumber/demos/mixer/docs/mixer-demo-poster.jpg)](https://amagimedia.github.io/avplumber/demos/mixer/docs/)

[Watch the 20-second demo](https://amagimedia.github.io/avplumber/demos/mixer/docs/) ·
[WebGL graph demo and zoom levels](https://amagimedia.github.io/avplumber/) ·
[MP4](https://github.com/amagimedia/avplumber/releases/download/mixer-demo-media-2026-10/mixer-192-input-tbar-retina-20261028-30fps.mp4) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/) ·
[Configuration](docs/config.md) · [Run manually](docs/guide.md)

Captured 2026-10-28 on the 192-input develop show: 105 NVDEC videos, 36 browser scene
sources, 47 raw NV12 feeds and four browser keys, with 256 scenes and 24 encoded outputs
at 1920×1080p30 SDR on one NVIDIA L4. All four keys are enabled.

The two viewers use native encoded pictures captured on the host before Janus. The actual
control page replays the recorded mouse-operated T-bar and sampled performance meters at
2656×1468, 2× Retina scale. The sequence includes a manual fade, halfway hold, completion,
return stroke and cut. CPU/GPU meters show captured measurements; this is an offline UI
replay, not a WebRTC screen recording. Both streams have continuous 30 fps RTP timestamps
and zero packet gaps; all 599 measured ticker steps advance by exactly four pixels.

Setup counts the 188 scene sources separately from the keys. This documentation recording
does not refresh the historical [capacity measurements](docs/capacity.md).

### Inspect the 192-input graph

The [WebGL viewer demo](https://amagimedia.github.io/avplumber/) follows the same
192-input workload from the whole show down to individual processing nodes. Its
[four zoom-level screenshots](https://amagimedia.github.io/avplumber/#zoom) and
24-second video show 999 nodes and 5,378 connected queues captured on 2026-10-29.
The viewer replays recorded telemetry at 2× Retina scale and 30 fps; graph counts
belong to this capture, independently of older architecture or capacity reports.
See the [WebUI documentation](../../tools/web-ui/README.md) to connect it to your
own mixer. The portable L4 deployment exposes it through the controls' **Graph** link.

## Run

Use a Linux NVIDIA host with NVDEC/NVENC, Docker Compose, NVIDIA Container
Toolkit and DRM/EGL/GBM for browser capture. Start from a recursive checkout;
see [host setup](../README.md#nvidia-host-setup).

```sh
docker compose -f docker-compose/mixer/compose.yaml up --build
```

The default Fedora/CUDA image requires driver R615 or newer. For the older
Ubuntu/CUDA image, prefix every build/start with `MIXER_DOCKERFILE=Dockerfile`.
The [L4 preset](../../docker-compose/mixer/deploy/l4/README.md) supplies a separate measured configuration.

Open <http://127.0.0.1:7681/setup/>, choose sources, canvas and outputs, then
**Apply setup**. Controls are at `/`, all outputs at `/wall`, and the standalone
player at <http://127.0.0.1:8080>. Select SDR if HEVC playback is unavailable.

For a remote host set `JANUS_HOST_IP=<host>`; allow TCP 7681/8080 and UDP
20000–20100. Stop with `docker compose -f docker-compose/mixer/compose.yaml down`.

Setup saves settings and generated assets in `media/`; applying changes pauses
output before asset preparation and restarts the mixer afterward. Synthetic
clip generation uses NVENC and must not compete with the live show for GPU
memory. Cached clips are reused. Keep that directory writable and mount it
whole: Setup atomically replaces the generated show. Capacity comes from the
selected [instance profile](instance_profiles.py), not a universal source limit.

## Controls

- Preview/Program: select the next scene, or enable Direct to cut on selection.
- Cut, Auto, Fade, colour Dip and media wipes; Auto can halt/resume or complete a take.
- T-bar: horizontal or upright, manual fade with a hold, cancellation or return stroke;
  cached wipes can follow the lever too.
- DSK: independent per-key fades or cuts, a shared ALL rate with per-key overrides,
  and Shift+click for an immediate key cut.
- NEXT: arm background and keys for one transition, or take only the armed keys.
- FTB: fade the keyed program to black while the clean feed remains available.
- Two resizable viewers, editable multiview slots, AUX layouts/source paging and an output wall.
- Live GPU, encoder/decoder, memory, power, CPU, thread and delivery statistics.

See the [operator controls](docs/guide.md#controls) or the
[illustrated Pages guide](https://amagimedia.github.io/avplumber/demos/mixer/docs/controls.html)
for rate editing, NEXT behavior and shortcuts.

## Configure and extend

- [Show configuration](docs/config.md) and [example](config.example.json): sources, scenes and outputs.
- [Recipes](docs/recipe.md): generate test media and scene layouts.
- [Operator guide](docs/guide.md): CLI, TUI, Docker, proxy and tests.
- [Capacity](docs/capacity.md) and [latency](docs/latency.md): scoped measurements and reproduction.
- [HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/): implementation and tuning notes.
- [DMA-BUF integration](../../doc/dmabuf.md): shared browser capture and the GBM shim.
- [Mixer base API](../../doc/mixer.md): reuse `pyplumber.mixer.build_application` in other applications; no demo imports required.
