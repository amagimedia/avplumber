# dmabuf-demo — browser overlay → avplumber → Janus (WebRTC)

Render a single HTML page (any URL — e.g. a broadcast graphics overlay)
headlessly in **stock Electron**, export each frame as a GPU **DMA-BUF**
(zero-copy), feed it into an **avplumber** graph, and stream it to **Janus** for
WebRTC preview in a browser.

No private/patched Electron build is used — stock upstream Electron works on
NVIDIA thanks to the **GBM linear shim** + one Chromium feature flag.

```
 ┌─────────┐   wayland    ┌──────────────┐  dmabuf fd (unix socket)  ┌───────────┐   RTP    ┌────────┐  WebRTC  ┌─────────────┐
 │ wayland │─ display ───▶│  dma-browser │──────────────────────────▶│ avplumber │────────▶│ janus  │─────────▶│ janus-preview│
 │ (sway)  │              │ stock Electron│   /tmp/dma-page/*.sock    │  graph    │  :5004   │        │  :8088   │  (browser)  │
 └─────────┘              │  42 + shim    │   48B TexInfo + FD        └───────────┘          └────────┘          └─────────────┘
                          └──────────────┘
```

## The settings that actually matter

### 1. Sender = stock Electron 42 (Chromium 146) + GBM shim
- `deps/dma-browser/package.json`: `"electron": "^42.0.0"` (Chromium 146). **No `.npmrc`** private mirror.
- **GBM shim — THE key to running stock Electron on NVIDIA** (`deps/dma-browser/native/gbm-linear-shim/`, built by `npm run rebuild:shim`, `LD_PRELOAD`ed from `bin/run.sh` on the NVIDIA path). Its relevant defaults are:
  - `GBM_LINEAR_SHIM=1` — enables the allocation rewrites.
  - `GBM_LINEAR_SHIM_FORCE_LINEAR=0` — strips `GBM_BO_USE_LINEAR`, allowing the NVIDIA driver to allocate a renderable tiled buffer. Forcing this to `1` commonly fails because NVIDIA cannot allocate a linear, renderable RGBA buffer.
  - **`GBM_LINEAR_SHIM_ADD_SCANOUT=1` — adds `GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING`** to renderable allocations. **This is the essential bit.** Stock Chromium's mappable-SharedImage path allocates a *scanout-only* buffer **without `RENDERING`** on NVIDIA, so it can't be a render target → `Unable to initialize SkSurface`, `paintCount: 0`, no frames. Adding `RENDERING` at the GBM layer replicates the custom Chromium **patch 0005** (`RenderableMappableSharedImageForceScanout`), so a **stock, unpatched Electron/Chromium captures**. (Verified: without it, stock Electron 41 *and* 42 both fail identically; with it, they paint. The private patched build is no longer needed.)
- `--enable-features=RenderableMappableSharedImageForceScanout` is set for parity but is a **no-op on stock Chromium** (that feature only exists in the patched build) — the shim's `ADD_SCANOUT` is what actually does the job.
- Note: `gpu_compositing=disabled_software` in `getGPUFeatureStatus()` is **expected/harmless** here (present even in the working patched build) — the offscreen `useSharedTexture` path is independent of the on-screen compositor. Don't chase it.
- **No VAAPI / no HW video decode** (graphics overlay only): all `Vaapi*` / `AcceleratedVideoDecoder*` features are in `--disable-features`, plus `--disable-accelerated-video-decode`.
- Other required Chromium switches (see `HwAccelConfigurator.ts`): `--enable-gpu --no-sandbox --run-all-compositor-stages-before-draw --use-gl=angle --use-angle=gl-egl --disable-vulkan --disable-hardware-overlays --ignore-gpu-blocklist --ozone-platform=wayland`.
- Offscreen capture: `BrowserWindow` `webPreferences.offscreen = { useSharedTexture: true }`, `transparent: true`, `backgroundThrottling: false`.
- Single page via env: `DMA_BROWSER_AUTO_OPEN_URL`, `_WIDTH=1920 _HEIGHT=1080 _FPS=30`, `DMA_BROWSER_ALLOWED_DIMS=1920x1080,1280x720`.

### 2. Host must have the NVIDIA **graphics** driver userspace (not compute-only)
The headless GL path needs the NVIDIA GL/EGL/GBM userspace on the host so the
NVIDIA container runtime can mount it. A **compute-only** driver
(`nvidia-headless-*` / `libnvidia-compute` only) is **not enough**.
- Ubuntu host: `apt install libnvidia-gl-<VER>-server` (provides `libEGL_nvidia`, `libGLX_nvidia`, `libnvidia-egl-gbm`, the `10_nvidia.json` EGL ICD, `nvidia-drm_gbm.so`).
- Fedora host: the matching `xorg-x11-drv-nvidia-cuda-libs`.
- Verify: `nvidia-smi` works **and** `ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json` exists.

### 3. Container NVIDIA EGL/GBM wiring (both wayland + dma-browser)
Each GPU container needs:
- `NVIDIA_DRIVER_CAPABILITIES=all` (must include `graphics`), `--gpus all`, `--privileged`, `--device /dev/dri`.
- The EGL vendor ICD and NVIDIA GBM backend (`nvidia-drm_gbm.so`). Current NVIDIA Container Toolkit versions inject both when graphics capabilities are enabled. Older installations may need the host's matching `libnvidia-egl-gbm` mounted manually; do not bind over paths already injected by the runtime.
- Env: `GBM_BACKEND=nvidia-drm`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`, `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.
- **Do not** bake the NVIDIA driver into the container (version must match the host kernel module — let the runtime mount it).

### 4. Headless wayland (sway) — `wayland/`
- `WLR_BACKENDS=headless WLR_RENDERER=gles2 WLR_RENDER_DRM_DEVICE=/dev/dri/renderD128`.
- The DRM render node is `root:render 0660`; sway runs as non-root `avp`, so the entrypoint `chmod o+rw /dev/dri/renderD128 /dev/dri/card*` (as root) before dropping to `avp` — a privileged container does **not** give a non-root user `CAP_DAC_OVERRIDE`.

### 5. avplumber graph → Janus RTP — `graph/dmabuf_browser_to_janus.py`
`ipc_dmabuf_source(@drm)` → `assume_video_format(drm_prime/cuda)` → `smooth_timestamps` →
`drm_prime_to_cuda(@gpu, drop_alpha)` → `force_keyframe(1s)` →
`enc_video(h264_nvenc baseline)` → `bsf dump_extra=freq=keyframe` → `mux` →
`output(rtp)` to Janus.

Set `FPS` to the dma-browser window's capture rate. `smooth_timestamps`
regularizes PTS while preserving one output frame per input frame; it does not
convert between frame rates.
- **Why CUDA detile:** on NVIDIA the browser's GPU render target is always tiled
  (block-linear); a plain DRM `hwdownload` reads sheared garbage. `drm_prime_to_cuda`
  EGL-imports the DMA-BUF honoring the tiling modifier into a linear CUDA frame,
  which NVENC encodes directly (`drop_alpha` labels it RGB0 so NVENC accepts it).
  Consumer image: `consumer/Dockerfile.cuda` (`HAVE_CUDA+GL+DRM`, no TensorRT/neural).
- Janus streaming mountpoint: video H264, **payload type 96**, ports **5004/5005**, `videofmtp=profile-level-id=42e028;packetization-mode=1`.
- RTP output opts: `rtp://JANUS:5004?pkt_size=1200&rtcp_port=5005`, `{payload_type:96, rtpflags:"skip_rtcp", ssrc:0x41565001}`.

## Run

```bash
cp .env.example .env      # set JANUS_HOST_IP + HTML_OVERLAY_URL
./up.sh                   # builds + starts wayland, dma-browser, janus, janus-preview, graph
# watch in a browser:
open http://<JANUS_HOST_IP>:8080     # janus-preview
./down.sh
```

Confirm capture is flowing: `curl http://127.0.0.1:9009/status` — `txFrameCount`
should be climbing and `droppedReasons` near-empty. Turn on `GBM_LINEAR_SHIM_LOG=1`
to see the shim strip `GBM_BO_USE_LINEAR`.

## Duplicate-frame analysis

The graph has a second output mode that measures duplicate browser frames
without changing the live Janus path. It detiles the DMA-BUF into CUDA, performs
an explicit diagnostic-only download and conversion to `yuv444p`, runs FFmpeg's
`mpdecimate` inside avplumber, and counts frames immediately before and after
the filter:

```bash
docker run --rm --network host --gpus all --privileged \
  --security-opt seccomp=unconfined --device /dev/dri:/dev/dri \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e SOCKET=/tmp/dma-page/overlay.sock \
  -e WIDTH=1280 -e HEIGHT=720 -e FPS=30 \
  -e OUTPUT_MODE=mpdecimate \
  -e MPDECIMATE_DURATION_SEC=600 \
  -e MPDECIMATE_REPORT_INTERVAL_SEC=60 \
  -v <dma-browser-socket-volume>:/tmp/dma-page \
  -v "$PWD/dmabuf-demo/graph/dmabuf_browser_to_janus.py:/opt/avplumber/dmabuf_browser_to_janus.py:ro" \
  avplumber-dmabuf-consumer-cuda \
  python3 /opt/avplumber/dmabuf_browser_to_janus.py
```

Progress and the final result use a stable, grep-friendly format:

```text
[dmabuf_mpdecimate] final input=18000 unique=17990 duplicates=10 duplicate_pct=0.056
```

Set `MPDECIMATE_FILTER` to override the default `mpdecimate` filter expression,
for example `mpdecimate=hi=768:lo=320:frac=0.33:max=0`.
