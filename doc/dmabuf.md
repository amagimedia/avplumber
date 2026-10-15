# DMA-BUF browser inputs

- Applications use `pyplumber.mixer.build_application(config, options)` with
  `kind: "browser"` sources. [Base API](mixer.md); implementation:
  [dmabuf_inputs.py](../pyplumber/mixer/dmabuf_inputs.py).
- Set `MixerOptions.dmabuf_rest` and `dmabuf_socket_dir`. The base API opens
  windows, imports frames and preserves alpha for blended scene items/DSK.
  The demo and AI mixer reuse this code; each owns its browser inputs.
- Shared container services and the Wayland image live in
  [docker-compose/dmabuf](../docker-compose/dmabuf/compose.yaml), outside demos.
  The browser implementation is the [dma-browser service](../deps/dma-browser).
- Start browser services from the repository root:
  `docker compose -f docker-compose/dmabuf/compose.yaml up -d --build wayland dma-browser`.
  Consumers need the same `dma-browser-sockets` volume mounted at `/tmp/dma-page`.
  An application Compose stack can extend `cuda-consumer` and supply its image/command;
  clear its optional profile (`profiles: !reset []`) and include `wayland`,
  `dma-browser`, `janus` and their named volumes in that stack.
- Linux NVIDIA graphics/DRM/EGL/GBM is required; build AVP with CUDA, NVCC, DRM and GL.
  Frames travel Electron → DMA-BUF FD → EGL/CUDA import → AVP compositor → NVENC.
  FD transport and acknowledgment keep textures alive until downstream GPU use ends.

## Vanilla Chromium with the GBM shim

Use official Electron with its bundled Chromium and the
[GBM shim](../deps/dma-browser/native/gbm-linear-shim/gbm_linear_shim.c).
`LD_PRELOAD` clears `GBM_BO_USE_LINEAR` and adds `SCANOUT | RENDERING`, allowing
NVIDIA-renderable capture buffers. It affects matching GBM allocations in that process;
enable it only for the NVIDIA browser. `GBM_LINEAR_SHIM_LOG=1` logs changes.
The Docker image and NVIDIA launcher enable the shim automatically.

Chromium copies into its capture pool and waits for GPU completion before delivering
`paint`. The browser then retains the texture until the mixer acknowledges its release.
Keep DRM modifiers and texture-release acknowledgements intact.
The mixer samples each imported capture buffer in place; it does not copy it again.
dma-browser refuses the `AvpZeroCopyOffscreenExport` feature of a removed patched
Electron, and the mixer's setup fails on a browser service that still reports a fixed
capture pool.

[Runnable mixer](../demos/mixer/README.md) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)
