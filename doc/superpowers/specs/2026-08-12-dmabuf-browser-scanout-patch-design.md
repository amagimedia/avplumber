# DMA-BUF Browser NVIDIA SCANOUT Patch

## Goal

Make the `dmabuf-browser` demo self-contained for users who want to rebuild
Electron instead of using a GBM `LD_PRELOAD` shim. The demo will carry and
document only the Chromium allocation change needed for Electron to export a
renderable NVIDIA DMA-BUF that avplumber can import.

## Scope

Add one standalone Chromium patch under `demos/dmabuf-browser/`. The patch will
define the disabled-by-default `RenderableMappableSharedImageForceScanout`
feature and make the Linux renderable SharedImage pool select
`gfx::BufferUsage::SCANOUT` when either CPU access is unnecessary or the
feature is enabled.

The demo README will link Chromium CL 6681354 and its landed commit. It will
explain that the upstream change introduced the CPU-access preference and the
non-linear SCANOUT allocation path, but did not add a command-line feature that
forces Electron's offscreen capture onto that path. The bundled patch supplies
that missing opt-in; users must enable it with
`--enable-features=RenderableMappableSharedImageForceScanout` in their rebuilt
Electron.

The README will provide concise Electron rebuild instructions and distinguish
the two supported approaches:

- stock Electron with the GBM allocation shim;
- rebuilt Electron with the bundled Chromium patch and no shim.

The shim documentation will state that this means an unpatched Chromium build
inside Electron, not the standalone Google Chrome application: the demo relies
on Electron's offscreen shared-texture API. In the NVIDIA configuration used by
the demo, the `LD_PRELOAD` shim intercepts GBM buffer and surface allocation,
removes `GBM_BO_USE_LINEAR`, and adds both `GBM_BO_USE_SCANOUT` and
`GBM_BO_USE_RENDERING`. This reproduces the effective allocation selected by
the Chromium patch without changing Chromium source. NVIDIA may then allocate a
tiled buffer; avplumber imports it through EGL with its DRM modifier and
`drm_prime_to_cuda` detiles it into a CUDA frame before NVENC.

The README will not imply that the shim changes Chromium's feature registry or
that `RenderableMappableSharedImageForceScanout` exists in a stock binary. That
feature flag works only after compiling the bundled patch.

## Exclusions

Do not copy or document the NVIDIA VAAPI/NVDEC patch series from
`electron-hwaccel`. The standalone patch must contain no decoder
synchronization, buffer-pool tuning, reset handling, surface-lifetime
diagnostics, or VAAPI logging.

Remove VAAPI-specific Chromium features, launcher environment variables,
diagnostic filters, documentation, and unit-test expectations from the
`deps/dma-browser` runtime. Preserve the EGL/GBM, DRM render-node, DMA-BUF,
CUDA detiling, and NVENC configuration used by the browser-to-avplumber graph.

Do not change avplumber framework behavior or the DMA-BUF receiver protocol.

## Compatibility and validation

The demo currently uses Electron 41.3.0 with Chromium 146.0.7680.188. The
standalone patch will be checked against stock Chromium 146 and Chromium 150
source layouts because both contain the same upstream `requires_cpu_access`
allocation branch.

Run the focused `deps/dma-browser` unit tests after removing the VAAPI runtime
configuration. Run patch applicability checks for both Chromium versions and
`git diff --check`. Do not claim NVIDIA runtime validation without executing
the demo on an NVIDIA host.
