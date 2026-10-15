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

## Optional fixed-pool exporter

The alternative patched browser supports a fully prewarmed capture pool. Its
version-2 build selects the size at startup; nine buffers are the current tested
starting point. The advertised size also controls sender retention and AVP's
import cache. See the [browser configuration](../deps/dma-browser/README.md) for
build requirements and flags; changing the mixer's `browser_frame_queue` alone
does not resize this pool.

With `AVP_BROWSER_COPY_ON_IMPORT=1`, AVP waits for the acquire fence, copies into
owned CUDA memory on a private nonblocking stream, then waits for that copy before
returning the browser lease and publishing the frame. The compositor holds the
owned copy. The import thread can wait; a nonblocking CUDA stream describes GPU
stream ordering, not a guarantee that the CPU thread never blocks. This path
retains all pool registrations and uses no paced return queue. Its moving-page
[capacity checks](../demos/mixer/docs/capacity.md#moving-pages-and-transitions)
are separate from historical ordinary-capture benchmarks.

[Runnable mixer](../demos/mixer/README.md) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)
