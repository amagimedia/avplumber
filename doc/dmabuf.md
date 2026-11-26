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
NVIDIA-renderable capture buffers. It affects every GBM allocation in that process.
`bin/run.sh` always preloads it on NVIDIA and refuses to start when it is not built;
`GBM_LINEAR_SHIM_LOG=1` logs changes.

Chromium copies into its capture pool and waits for GPU completion before delivering
`paint`. The browser then retains the texture until the mixer acknowledges its release.
Keep DRM modifiers and texture-release acknowledgements intact.
The mixer samples each imported capture buffer in place; it does not copy it again.

[Runnable mixer](../demos/mixer/README.md) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)

## Publishing a CUDA output

`ipc_dmabuf_sink` accepts a CUDA video edge and serves one local client on a Unix
`SOCK_SEQPACKET` socket. Use an absolute socket path in an existing directory.

```json
{"type":"ipc_dmabuf_sink","src":"program_sdr","socket":"/tmp/mixer-monitors/program.sock","pool_size":4}
```

The node converts SDR NV12 or NV16 to a pooled RGBA EGL image on the GPU. It exports
one DMA-BUF descriptor per frame with the 48-byte `TexInfo` header used by the HTML
input. The header describes the actual format, pitch, modifier and plane offset.
Send the 16-byte frame release acknowledgement from `dmabuf_ack.h` after the last
GPU reader finishes. A full pool drops viewer frames. Disconnecting retires the
pool. The default socket mode is 0600.

An optional `subscription` names a `one_to_many` subscribed output. The sink
enables that output while a receiver is connected, so an upstream monitor colour
conversion can idle between viewers. The desktop mixer uses this for Program
after keys and transitions, the configured clean feed and each AUX bus.
