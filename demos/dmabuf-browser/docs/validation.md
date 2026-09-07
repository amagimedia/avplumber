# DMA-BUF browser validation

Validated on 2026-09-07 with an NVIDIA T4, driver 595.71.05, and an Ubuntu 26.04
host. The browser and headless Wayland containers use Fedora 42; the CUDA
consumer uses Ubuntu 22.04. Host and container distributions need not match,
but the NVIDIA runtime must expose the matching host graphics libraries.

## Results

- All five layout/output configuration tests passed.
- All five Compose images built from the public-source configuration: official
  Electron plus the bundled GBM shim, with no private Electron archive.
- A single 1920×1080 Singular page passed three fresh starts. The received Janus
  video showed its colour bars, scrolling ticker, and advancing on-page frame
  counter. The pixel checks sampled eight images per run and confirmed actual
  changing content at roughly 60 decoded frames per second.
- A 16-source 4×4 Singular grid produced real changing content in every tile.
  Each tile was checked separately in eight samples of the received video.
  A separate 10-second queue-counter measurement, without receiver screenshots,
  measured 39.0–39.8 new frames/s per source and 59.98 frames/s through the
  compositor and encoder. The output repeats the latest source image as needed;
  this run does not establish sixteen independent 60 fps captures.
- The existing MP4 decoded without errors: H.264, 1920×1080, 60 fps, 600 frames,
  10 seconds. It is retained as the demo recording; the web UI screenshot comes
  from the new Ubuntu validation run and shows all 54 nodes and 53 queues.
  The poster uses the frame at one second because the recording starts black.

The production path stays on the GPU through DMA-BUF import, EGL/CUDA
composition and NVENC. Pixel checks inspect the decoded WebRTC receiver output;
they do not add downloads to the production graph.

## Reproduced Ubuntu GBM failures

The following comparisons used the same Ubuntu host, GPU, driver, and Fedora
Wayland image. Only the container runtime or GBM backend search path changed.

| Configuration | Observed result |
| --- | --- |
| NVIDIA runtime; Ubuntu and Fedora GBM paths | NVIDIA EGL and Tesla T4 renderer; successful block-linear GBM buffer allocation. |
| NVIDIA runtime; Fedora-only GBM path | Repeated `gbm_bo_create failed` and buffer-allocation errors. A Wayland socket still appeared, so socket health alone was misleading. |
| `runc`; both GBM paths | EGL context and GLES2 renderer creation failed. |

The working configuration uses:

```yaml
runtime: nvidia
environment:
  NVIDIA_DRIVER_CAPABILITIES: all
  GBM_BACKEND: nvidia-drm
  GBM_BACKENDS_PATH: /usr/lib/x86_64-linux-gnu/gbm:/usr/lib64/gbm
```

On the tested Ubuntu host, the toolkit supplied
`/usr/lib/x86_64-linux-gnu/gbm/nvidia-drm_gbm.so`, pointing to
`../libnvidia-allocator.so.1`. The old image hardcoded a Fedora-path symlink to
`libnvidia-egl-gbm.so.1`. The image now lets the toolkit supply the backend and
EGL vendor files. NVIDIA documents that `graphics` controls the OpenGL/EGL/Vulkan
libraries exposed to containers; `nvidia-smi` alone exercises a different
capability. [NVIDIA driver capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities)

Official Electron also needs the supplied allocation shim for this NVIDIA
capture path (`GBM_LINEAR_SHIM=1`, `GBM_LINEAR_SHIM_ADD_SCANOUT=1`). This is
separate from finding the correct GBM backend. The single-source graph waits
for its first timestamp-initialized CUDA frame before starting the encoder.

These comparisons establish a container configuration cause on Ubuntu 26.04.
They are not a controlled comparison of Fedora and Ubuntu kernels or a promise
that every NVIDIA driver version supports the same interop path.
