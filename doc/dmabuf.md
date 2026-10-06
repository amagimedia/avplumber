# DMA-BUF browser inputs

- Applications use `pyplumber.mixer.build_application(config, options)` with
  `kind: "browser"` sources. [Base API](mixer.md); implementation:
  [dmabuf_inputs.py](../pyplumber/mixer/dmabuf_inputs.py).
- Set `MixerOptions.dmabuf_rest` and `dmabuf_socket_dir`. The base API opens
  windows, imports frames and preserves alpha for blended scene items/DSK.
  The demo and AI mixer reuse this code; each owns its browser inputs.
- Shared container services, Wayland image and optional source patches live in
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

## Shim and Chromium patches

- Default: official Electron plus the [GBM shim](../deps/dma-browser/native/gbm-linear-shim/gbm_linear_shim.c).
  `LD_PRELOAD` clears `GBM_BO_USE_LINEAR` and adds `SCANOUT | RENDERING`, allowing
  NVIDIA-renderable buffers. It affects matching GBM allocations in that process;
  enable it only for the NVIDIA browser. `GBM_LINEAR_SHIM_LOG=1` logs changes.
- Chromium allocation support: [CL 6681354](https://chromium-review.googlesource.com/c/chromium/src/+/6681354),
  [commit a531c83a](https://chromium.googlesource.com/chromium/src/+/a531c83a9bbb552fa13ceeaf73d0a19d1203cc12).
- Chromium texture-capture fix: [CL 8220427](https://chromium-review.googlesource.com/c/chromium/src/+/8220427),
  [commit 9522ea8a](https://chromium.googlesource.com/chromium/src/+/9522ea8ad0bed3c3fd1b4f570ed9792dec8176d5),
  merged 2026-09-15 (verified 2026-10-06).
- The pinned [Electron 44.5.1 consumer](https://github.com/electron/electron/blob/v44.5.1/shell/browser/osr/osr_video_consumer.cc)
  still requests mappable textures. The Chromium fix alone does not remove the shim.
- Optional [source-build script and patches](../docker-compose/dmabuf/chromium/build-electron.sh)
  select native-handle capture with `RenderableMappableSharedImageForceScanout`.
  This is a local feature, not a stock Chrome flag. Patches target older Electron;
  do not reapply the Chromium patch to a revision already containing the fix.
- Keep this route on ARGB/BGRA/RGBA; do not assume NV12/RGBAF16 native-handle capture
  works. Preserve DRM modifiers and texture-release acknowledgments.

[Runnable mixer](../demos/mixer/README.md) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)
