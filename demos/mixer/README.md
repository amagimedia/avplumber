# Live mixer demo

Mix video and browser sources on a CUDA canvas, with Preview/Program,
Cut/Fade/media wipes, downstream keys and AUX monitors. Video only.

[Watch the demo](https://amagimedia.github.io/avplumber/demos/mixer/docs/) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/) ·
[Configuration](docs/config.md) · [Run manually](docs/guide.md)

## Run

Use a Linux NVIDIA host with NVDEC/NVENC, Docker Compose, NVIDIA Container
Toolkit and DRM/EGL/GBM for browser capture. Start from a recursive checkout;
see [host setup](../README.md#nvidia-host-setup).

```sh
docker compose -f demos/mixer/compose.yaml up --build
```

The default Fedora/CUDA image requires driver R615 or newer. For the older
Ubuntu/CUDA image, prefix every build/start with `MIXER_DOCKERFILE=Dockerfile`.
The [L4 preset](deploy/l4/README.md) supplies a separate measured configuration.

Open <http://127.0.0.1:7681/setup/>, choose sources, canvas and outputs, then
**Apply setup**. Controls are at `/`, all outputs at `/wall`, and the standalone
player at <http://127.0.0.1:8080>. Select SDR if HEVC playback is unavailable.

For a remote host set `JANUS_HOST_IP=<host>`; allow TCP 7681/8080 and UDP
20000–20100. Stop with `docker compose -f demos/mixer/compose.yaml down`.

Setup saves settings and generated assets in `media/`; applying changes pauses
output while the mixer restarts. Keep that directory writable and mount it
whole: Setup atomically replaces the generated show. Capacity comes from the
selected [instance profile](instance_profiles.py), not a universal source limit.

## Controls

- Scene: select Preview, or take immediately in Direct mode.
- Cut / Fade / Wipe: take the selected scene; a new take interrupts the old one.
- Keys: toggle browser overlays on the program; clean outputs omit them.
- AUX: choose layouts, scene slots or source pages independently of Program.

## Configure and extend

- [Show configuration](docs/config.md) and [example](config.example.json): sources, scenes and outputs.
- [Recipes](docs/recipe.md): generate test media and scene layouts.
- [Operator guide](docs/guide.md): CLI, TUI, Docker, proxy and tests.
- [Capacity](docs/capacity.md) and [latency](docs/latency.md): scoped measurements and reproduction.
- [HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/): implementation and tuning notes.
- [DMA-BUF integration](../../doc/dmabuf.md): shared browser services, shim and Chromium patches.
- [Mixer base API](../../doc/mixer.md): reuse `pyplumber.mixer.build_application` in other applications; no demo imports required.
