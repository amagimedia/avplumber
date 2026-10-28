# Live mixer demo

Mix video and browser sources on a CUDA canvas, with Preview/Program,
Cut/Fade/media wipes, downstream keys and AUX monitors. Video only.

[![192-input mixer: Program/Preview and dirty Program with four downstream keys](https://amagimedia.github.io/avplumber/demos/mixer/docs/mixer-demo-poster.jpg)](https://amagimedia.github.io/avplumber/demos/mixer/docs/)

[Watch the 20-second demo](https://amagimedia.github.io/avplumber/demos/mixer/docs/) ·
[MP4](https://github.com/amagimedia/avplumber/releases/download/mixer-demo-media-2026-10/mixer-192-input-tbar-20261028-1080p30.mp4) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/) ·
[Configuration](docs/config.md) · [Run manually](docs/guide.md)

Captured 2026-10-28 on an NVIDIA L4 and replayed in the real two-viewer interface:
192 inputs (105 NVDEC, 36 browser scene sources, 47 raw NV12 and four browser
keys), 256 scenes and 24 encoded outputs at 1920×1080p30 SDR. It shows direct
cuts and manual T-bar fades, including a halfway hold and a return stroke,
with all four keys enabled. Both encoded outputs were recorded on the
instance at Janus’s RTP input before Internet transport, then combined with the matching UI
and sampled mixer status at 30 fps. The fade itself is in the recorded Program output. [Original Program recording](https://github.com/amagimedia/avplumber/releases/download/mixer-demo-media-2026-10/mixer-pgm-tbar-20261028-1080p30.mp4)
retains the encoder bitstream, including capture preroll and tail. Setup counts the 188 scene sources separately
from the keys. This documentation recording does not refresh the historical capacity benchmark.
The same show runs at 1080p25 with 26 outputs; see the
[capacity page](docs/capacity.md).

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

- Scene: select Preview, or take immediately in Direct mode.
- Cut / Fade / Wipe: take the selected scene; a new take interrupts the old one.
- T-bar: move between Program and Preview; hold midway or reverse the stroke.
- Keys: toggle browser overlays on the program; clean outputs omit them.
- AUX: choose layouts, scene slots or source pages independently of Program.

## Configure and extend

- [Show configuration](docs/config.md) and [example](config.example.json): sources, scenes and outputs.
- [Recipes](docs/recipe.md): generate test media and scene layouts.
- [Operator guide](docs/guide.md): CLI, TUI, Docker, proxy and tests.
- [Capacity](docs/capacity.md) and [latency](docs/latency.md): scoped measurements and reproduction.
- [HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/): implementation and tuning notes.
- [DMA-BUF integration](../../doc/dmabuf.md): shared browser capture and the GBM shim.
- [Mixer base API](../../doc/mixer.md): reuse `pyplumber.mixer.build_application` in other applications; no demo imports required.
