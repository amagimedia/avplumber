# DMA-BUF EGL Compositor for Browser Scaling

## Goal

Sustain eight independent 1920x1080@60 Electron browser captures and produce one
1920x1080@60 Janus program output without converting or scaling every input in
CUDA. Preserve the browser-ingest/OBS DMA-BUF ingestion model through the final
OpenGL composite; hand only that composite to the existing CUDA/NVENC pipeline.

The existing eight-input path is the performance baseline, not the design to
extend. It imports every DMA-BUF anew, copies every 1920x1080 input into an
OpenGL texture, copies it again into a CUDA frame, runs one `scale_cuda` filter
per input, and then composites the scaled CUDA frames. The controlled remote
test showed that the avplumber graph, rather than Electron capture, caused the
large frame-rate regression.

## Success criteria

The production test uses `SOURCE_COUNT=8`, eight 1920x1080@60 instances of the
configured browser source, and one 1920x1080@60 Janus output. During a 60-second
steady-state sample:

- every Electron sender reports at least 59.5 transmitted frames per second and
  no increasing capture/transmit drop counter;
- the encoded RTP stream contains one frame marker for every 60 Hz output tick,
  with a measured wall-clock rate between 59.5 and 60.5 frames per second;
- the graph reports no queue overflow, EGL import, synchronization, CUDA, or
  encoder errors;
- the production graph contains no per-input `drm_prime_to_cuda`,
  `scale_cuda`, `hwdownload`, or `hwupload` stage;
- average avplumber SM utilization is at least 50 percent lower than the prior
  eight-input CUDA graph (approximately 35.6 percent SM in the earlier sample),
  with a target of 18 percent or less on the same host and workload; and
- production PCIe traffic does not increase relative to the current graph.

Frame-uniqueness diagnostics remain a separate run because downloading probe
frames changes the performance being measured. The diagnostic must still check
all eight independent inputs; its resource measurements are not production
results.

## Architecture

The production data path is:

```text
ipc_dmabuf_source (x8)
  -> drm_prime_to_egl_image (x8, cached and synchronized)
  -> egl_image_compositor_cuda (one GL context and one render target)
  -> CUDA frame (one composite copy per output frame)
  -> NVENC
  -> Janus RTP
```

This follows the proven OBS browser-ingest path until the terminal output
representation. OBS retains an OpenGL texture. Avplumber instead registers the
single composite texture with CUDA graphics interop and copies it into one
FFmpeg CUDA frame so the existing NVENC nodes remain unchanged. The DMA-BUF to
EGLImage to input-texture path is zero-copy. The registered composite texture
is exposed to CUDA as an array, while the normal FFmpeg `AV_PIX_FMT_CUDA` path
expects pitched device memory, so the design retains one array-to-device copy
per program frame. There is no CUDA representation of an individual browser
input in the production graph.

The implementation adds a new node rather than changing graph management or
the control protocol. The existing CUDA compositor remains available as an
explicit comparison and diagnostic backend.

## DMA-BUF import cache and synchronization

`drm_prime_to_egl_image` is the reusable import boundary. Its cache will follow
the OBS invariants:

- identify a buffer by `st_dev`, `st_ino`, width, height, DRM fourcc, plane
  pitch, plane offset, and modifier; an incoming FD number may accelerate a
  lookup but is not buffer identity;
- retain a duplicated DMA-BUF FD for the lifetime of every cached EGLImage;
- reuse an EGLImage only after exporting a DMA-BUF read `sync_file` and waiting
  on it through `EGL_ANDROID_native_fence_sync`, with bounded CPU polling when
  the EGL native-fence path is unavailable;
- if synchronization fails or times out, import a fresh EGLImage and retain it
  for deferred destruction instead of reusing the possibly busy image;
- evict unused entries by TTL and evict the least recently used entry when the
  bounded cache is full; and
- invalidate an entry when its identity or import attributes change, including
  FD-number reuse and resolution changes.

Synchronized caching is the default. JSON parameters may select `off` for a
fresh-import comparison or `unsafe` for an explicit diagnostic, but the demo
must not select `unsafe`. The initial scope remains the single-plane
ABGR8888/ARGB8888 buffers produced by the Electron DMA-BUF path. Multi-plane
YUV import is not part of this optimization.

The importer continues to attach a lifetime holder to each `EglImageFrame`, so
cache eviction cannot destroy an EGLImage while a queue or compositor still
references it.

## EGL compositor

`egl_image_compositor_cuda` accepts multiple `EglImageFrame` inputs and emits
CUDA `av::VideoFrame` objects. It owns one EGL display/context, one set of input
texture names, one shader program, and one persistent 1920x1080 framebuffer
texture. It binds each current EGLImage directly to its input texture with
`glEGLImageTargetTexture2DOES`; it does not copy source pixels into intermediate
textures.

For every output frame, the node clears the framebuffer once and draws each
active source into its configured rectangle. Bilinear texture sampling scales
1920x1080 inputs to their grid cells during those draws. The eight-input demo
uses the existing near-square layout calculation and passes destination
`x`, `y`, `width`, and `height` values to the compositor. Empty cells remain
opaque black. The first implementation needs static rectangles and opaque
sources only; dynamic layout metadata and general alpha compositing are outside
this optimization.

The framebuffer texture is registered with CUDA once and reused. After the GL
draws, CUDA/GL resource mapping provides the graphics-to-CUDA synchronization.
The node copies the one completed RGBA/RGB0 canvas into a pooled FFmpeg CUDA
frame and then emits it. It must not call `glFinish` for every input; a fallback
whole-frame `glFinish` is acceptable only if remote testing proves the normal
interop synchronization insufficient, and must be reported as such.

On the desktop NVIDIA target this registration uses
[`cuGraphicsGLRegisterImage`](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html),
which exposes a GL texture as a CUDA array without copying it. It does not use
`cuGraphicsEGLRegisterImage`: NVIDIA documents that entry point as
[Tegra-only](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-for-tegra-appnote/index.html).
The standard FFmpeg `AV_PIX_FMT_CUDA`/NVENC chain expects pitched CUDA device
memory rather than the registered GL array, which is why the one final
array-to-device copy remains. Direct registration of an EGLImage-backed input
GL texture is a remote capability probe, not an assumed desktop contract; it
may inform a later fused CUDA backend but cannot be the fallback for this
design.

## Frame scheduling

The compositor uses one configurable input as the 60 Hz clock, defaulting to
input zero. On every clock-input frame it drains the other active input queues
to their newest available frames and renders exactly one composite. This is
latest-frame behavior analogous to a real-time OBS scene: individual input
arrivals update held textures but do not independently trigger additional
composites.

Startup waits until every active input has supplied a frame, subject to the
existing bounded warm-up timeout. After warm-up, a temporarily late input keeps
its last frame. A clock-input discontinuity is passed downstream and the
existing program timestamp normalization remains responsible for the outgoing
monotonic timeline. EOF from a non-clock input freezes its last frame; EOF from
the clock input ends the compositor after queued work is drained.

This scheduling removes the current behavior where several inputs advancing at
the same timestamp cause several complete GPU composites that a downstream
`fps` filter later discards.

## Demo integration

The scaling demo selects the EGL compositor for its production grid. For every
source it creates only the IPC receiver, the existing DRM format assertion and
timestamp alignment, and `drm_prime_to_egl_image`. It passes the unsized
EGLImage frames and calculated destination rectangles to the one compositor.

The old CUDA graph stays selectable by an explicit backend setting for A/B
measurements and the independent per-source `mpdecimate` diagnostic. Source
count remains parameterized as `N`, defaulting to eight; neither the compositor
nor its cache hardcodes eight or an x86_64/NVIDIA architecture. NVIDIA is the
target used for validation, not a vendor condition in the node.

## Error handling and observability

Node creation fails with a specific error if the required DMA-BUF import,
EGLImage texture binding, framebuffer, shader, CUDA device, or CUDA/GL interop
capability is missing. It must not silently download to CPU or fall back to the
old per-input CUDA path.

The importer records cache hits, fresh imports, synchronization fallbacks, and
evictions in periodic aggregate counters rather than per-frame logs. The
compositor records rendered frames, skipped pre-warm-up clock ticks, held-input
reuse, and GL/CUDA failures. Optional periodic debug logging exposes these
counters without making normal operation noisy.

## Validation

Local validation covers pure graph generation and layout behavior:

- an eight-source EGL production graph contains eight synchronized EGL
  importers and one compositor;
- it contains no per-source CUDA conversion or scale filter;
- its compositor rectangles match the expected aspect-preserving grid;
- source count remains parameterized; and
- the explicit legacy/diagnostic backend still generates the existing CUDA
  path.

The provided Fedora NVIDIA host is used for all EGL/CUDA builds and execution.
Remote validation consists of:

1. a single-source smoke test for import, composition, CUDA handoff, NVENC, and
   Janus output;
2. the eight-source 60-second acceptance run described above;
3. process-attributed GPU SM, memory, encoder, decoder, and PCIe sampling;
4. confirmation that all eight producer counters remain at full rate;
5. RTP frame-marker and timestamp accounting at the Janus input;
6. a separate eight-input `mpdecimate` uniqueness run;
7. visual inspection through the existing Janus preview; and
8. an Nsight Systems capture confirming one render pass and one final
   graphics-to-CUDA copy per output tick, with no per-input CUDA scale/copy
   chain. Nsight Compute is used only if profiling identifies a custom CUDA
   kernel as a remaining bottleneck.

Only containers created for this DMA-BUF test may be stopped or replaced.
Unrelated recorder, playlist, replay, Janus, and web UI workloads remain
untouched.

## Exclusions

- no CPU compositor or CPU/GPU round trip;
- no VAAPI or browser video-decoder changes;
- no Chromium/Electron allocation-policy changes beyond the already documented
  scanout/native-handle patch or shim;
- no vendor or CPU-architecture hardcoding in avplumber;
- no dynamic scene graph, transitions, or general-purpose alpha mixer; and
- no graph-management, sentinel, or control-protocol changes.
