# DMA-BUF Browser NVIDIA Native-Handle Opt-In

## Goal

Make the `dmabuf-browser` demo self-contained for both users who rebuild
Electron and users who run a stock Electron binary through the GBM shim. Carry
only the allocation change needed for Electron to export a renderable NVIDIA
DMA-BUF that avplumber can import.

## Upstream Chromium foundation

Credit the upstream work that made Chromium capture buffers shareable before
describing the later NVIDIA allocation fix. Reito OvO's Chromium CL 5276423
added RGBA capture into an existing native GPU texture. CL 5265077 then made
`FrameSinkVideoCapturer` support RGBA output backed by a native-texture GPU
memory buffer, explicitly enabling accelerated shared-texture consumers such
as CEF OSR, Electron, and OBS. Link the two implementation CLs directly and
link Reito's combined Chromium-author query for their test and pixel-format
correctness follow-ups.

Keep this history distinct from CL 6681354. The 2024 work established RGBA
native-texture buffer sharing; CL 6681354 later added the consumer choice that
avoids CPU-mappable/linear allocation for the NVIDIA path.

## Electron patch

Chromium CL 6681354 introduced the internal
`kPreferSharedImageWithNativeHandle` capture-buffer preference. The Electron
patch adds a disabled-by-default `RenderableMappableSharedImageForceScanout`
feature in Electron's OSR consumer. When the feature is enabled and the window
uses `useSharedTexture`, Electron selects Chromium's native-handle preference
instead of `kPreferMappableSharedImage`.

This combined design retains runtime opt-in without changing Chromium's
general renderable video-frame pool. The launcher enables the feature only on
the detected NVIDIA path. The same Electron binary behaves like upstream when
the feature is absent and remains usable on other GPU vendors.

The demo includes a script that creates an isolated Electron checkout, applies
the patch, builds `electron:electron_dist_zip`, and emits an artifact and
checksum. The patch must apply to both Electron 41.3.0 / Chromium 146 and
Electron 43.4.0 / Chromium 150.

## Stock-binary shim

The alternative is an architecture-neutral C source shim compiled on the
target host and loaded into stock Electron with `LD_PRELOAD`. It intercepts
only GBM buffer and surface allocation functions that carry usage flags,
removes `GBM_BO_USE_LINEAR`, and adds `GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING`.
It does not change Chromium's feature registry, modifier lists, CPU mapping, or
video decoding.

NVIDIA may then allocate a tiled buffer. avplumber imports it through EGL with
its DRM modifier, and `drm_prime_to_cuda` detiles it into a CUDA frame before
NVENC. The shim is NVIDIA-gated because low-level GBM flag rewriting is not a
portable policy for other vendors.

## Scaling tests

Add a parameterized scaling graph with `SOURCE_COUNT=N`, defaulting to 8 for
the requested test. It creates N independent Electron windows through the REST
API, receives N independent DMA-BUF sockets, imports each through
`ipc_dmabuf_source` and `drm_prime_to_cuda`, and composites them on the GPU.
No source count is baked into the graph.

The requested load uses 1920x1080@60 capture for every input. The grid chooses
near-square row and column counts from N, then scales and pads each 16:9 source
inside its cell so the 1920x1080 program output is not distorted. For N=8 this
produces a 3x3 grid with one empty cell. A separate single-source mode runs one
full-canvas source through the same DMA-BUF import and Janus output path as a
baseline.

Performance and duplicate detection are separate phases. The production phase
stays zero-copy through the grid mixer and records GPU, encoder, memory, and
PCIe counters. The diagnostic phase fans out every imported CUDA frame, scales
the diagnostic copy to 640x360 on the GPU by default, downloads it, and runs an
independent `mpdecimate`. This evaluates every temporal frame from every
1920x1080@60 input while limiting raw device-to-host traffic to about 0.44 GB/s
for N=8. A full-resolution override would create about 4 GB/s and can
backpressure the capture-buffer lifetime, so diagnostic PCIe/CPU measurements
are not production performance measurements.

The browser's default maximum remains 8, but `DMA_BROWSER_MAX_WINDOWS` becomes
configurable above that default so a future N is not rejected by an
application-specific hard limit. Actual capacity is bounded by GPU memory,
capture dimensions, frame rate, and the compositing node's input-mask width.

## Exclusions

Do not include unrelated browser video-decoder changes. This work is limited
to the offscreen render-target allocation needed by the DMA-BUF receiver.

Preserve the EGL/GBM, DRM render-node, DMA-BUF, CUDA detiling, and NVENC path.
Do not change the DMA-BUF receiver protocol or graph-management framework.

## Validation

- Apply the Electron patch to both supported source layouts.
- Run the focused `deps/dma-browser` unit, TypeScript, lint, and formatting
  checks.
- Compile the shim on the local architecture and the NVIDIA x86_64 target.
- Run stock Electron 43.4.0 / Chromium 150 with the shim and confirm increasing
  paint/transmit counts with no dropped frames.
- Receive, EGL-import, and CUDA-detile frames with avplumber on NVIDIA.
- Run focused pure-Python tests for dynamic grid geometry and graph generation.
- Run single-source and N=8 smoke tests on the provided NVIDIA host without
  disturbing unrelated containers.
