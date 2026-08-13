# DMA-BUF EGL Compositor for Browser Scaling

## Goal

Sustain eight independent 1920x1080@60 Electron browser captures and produce one
1920x1080@60 Janus program output without materializing or scaling a separate
CUDA frame for every input. Import each DMA-BUF as a persistent EGLImage,
register that EGLImage directly with CUDA, and sample it from one CUDA
compositor that writes the final FFmpeg CUDA frame for NVENC.

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
- pausing any one input for five seconds does not pause or reduce the cadence of
  the program output or the other seven inputs;
- the graph reports no queue overflow, EGL import, synchronization, CUDA, or
  encoder errors;
- the production graph contains no per-input `drm_prime_to_cuda`,
  `scale_cuda`, `hwdownload`, or `hwupload` stage;
- steady state contains no `eglCreateImageKHR`, `cuGraphicsEGLRegisterImage`,
  `cuGraphicsResourceGetMappedEglFrame`, `glFinish`, or context-wide
  synchronization;
- Chromium textures are released only after the consumer acknowledges that the
  last CUDA read has completed; fixed pool depth is not used as synchronization;
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
  -> egl_image_cuda_overlay (direct EGL/CUDA registration, one CUDA compositor)
  -> final CUDA RGB0 frame (written directly)
  -> NVENC
  -> Janus RTP
```

Avplumber registers every cached EGLImage once with CUDA EGL interoperability,
gets its persistent CUDA array or pitched view, and samples that view directly.
Each output tick launches CUDA texture-sampling kernels that scale the sources
into disjoint rectangles of the final pitched `AV_PIX_FMT_CUDA` RGB0 frame.
There is no OpenGL object or render pass, no CUDA frame for an individual
browser input, no intermediate canvas, and no final canvas copy. The only
full-frame pixel write is the unavoidable write into the final CUDA output that
NVENC consumes.

The compositor keeps the existing RGB0 NVENC input. The encoder is not the
measured bottleneck, and changing output color format would not remove a source
copy. This optimization remains focused on DMA-BUF import, scaling, and
composition.

The implementation adds a new node rather than changing graph management or
the line-based graph control protocol. The existing CUDA compositor remains
available as an explicit comparison and diagnostic backend.

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

Each `EglImageFrame` carries separate allocation and per-frame lifetime holders.
The allocation holder keeps the cached EGLImage and duplicated DMA-BUF FD alive
and may be retained by the CUDA-registration cache. The per-frame holder keeps
the release acknowledgement pending and must never be retained by that cache.
Queues, the latest-frame slot, and an in-flight CUDA read retain both holders.

The DMA-BUF socket protocol also carries a release acknowledgement keyed by the
existing `frame_count`. It is a fixed 16-byte little-endian record containing a
32-bit protocol magic, a 32-bit reserved field, and the 64-bit frame number.
The avplumber receiver queues partial or temporarily blocked writes and flushes
them from its existing socket poll loop; an acknowledgement must not block a
CUDA or graph-processing thread.

The Electron sender retains each `OffscreenSharedTexture` until every client
that successfully received that frame either acknowledges it or disconnects.
The FD-pass add-on tracks the recipient set and notifies TypeScript only when it
becomes empty. The avplumber receiver attaches the acknowledgement to the
received frame's lifetime. Frames discarded before CUDA use can acknowledge
immediately. After a frame has been sampled, the compositor records a CUDA event
after its last read and retains only the per-frame holder until that event
completes; it polls events without a context-wide synchronization. Only then can
the frame lifetime callback queue the acknowledgement. This closes the
consumer-to-producer reuse direction that a producer-side `sync_file` wait does
not cover.

## Cached EGL/CUDA interop and compositor

`egl_image_cuda_overlay` accepts multiple `EglImageFrame` inputs and emits CUDA
`av::VideoFrame` objects. It owns a cache that extends every upstream cached
physical allocation into this immutable object chain:

```text
DMA-BUF allocation
  -> EGLImageKHR
  -> CUgraphicsResource
  -> CUeglFrame
  -> CUarray or pitched CUDA view
  -> CUtexObject
```

`cuGraphicsEGLRegisterImage` and `cuGraphicsResourceGetMappedEglFrame` run only
when a new physical backing allocation appears. NVIDIA's EGL registration does
not require graphics map/unmap calls. The upstream importer keys physical
allocations by `st_dev`, `st_ino`, dimensions, DRM fourcc, modifier, and every
plane's offset and pitch, so FD-number reuse cannot alias a cache entry. The
compositor's EGLImage-handle lookup is safe because it retains that upstream
entry for the entire CUDA-registration lifetime. A cache entry and its CUDA
texture object are destroyed only after all held frames stop referencing that
physical allocation.

For every output frame, the node clears the final CUDA frame once and launches
one bilinear texture-sampling kernel for each active source; those kernels
sample the cached CUDA texture objects and write disjoint configured rectangles
directly into RGB0.
The eight-input demo uses the existing near-square layout calculation and passes
destination `x`, `y`, `width`, and `height` values to the compositor. Empty
cells remain opaque black. The first implementation needs static rectangles and
opaque sources only; dynamic layout metadata and general alpha compositing are
outside this optimization.

The importer's per-frame DMA-BUF `sync_file` wait establishes producer-to-CUDA
ordering before a cached EGLImage is emitted. CUDA reads and final-canvas writes
run on the normal FFmpeg device stream, so repeated use and NVENC consumption
remain ordered without a context-wide synchronization. The frame loop must not
call `glFinish`, `cuCtxSynchronize`, register/unregister resources, or create and
destroy CUDA texture objects.

On the desktop NVIDIA target this registration uses NVIDIA's
[`cuGraphicsEGLRegisterImage` API][cuda-egl], the same pattern already used by
`src-streamer`. The API exposes the EGLImage as a `CUeglFrame` without copying
it and does not require per-frame graphics map/unmap calls. The node supports
CUDA array and pitched EGL frame types and derives RGB channel lanes from
`CUeglColorFormat`; it does not hardcode the observed buffer ordering. Failure
of direct EGL registration is a hard capability error and must not be hidden
behind a GL copy or CPU fallback.

[cuda-egl]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EGL.html

## Frame scheduling

The compositor owns a monotonic program clock with configurable frame rate,
defaulting to 60 Hz. This is implemented entirely inside the new node: it waits
on the existing multi-input edge events only until the next program deadline,
drains every ready input queue to its newest frame, and renders once when that
deadline arrives. It does not change graph management, scheduling, or the
line-based graph control protocol. A missed deadline advances to the next
future tick rather than emitting a catch-up burst.

Input arrivals update held frames but never trigger extra composites. A late,
paused, or EOF input keeps its last frame while the clock and other inputs
continue. Before an input has produced its first usable frame, its rectangle is
black. Neither compositor startup nor the program clock waits for every input
to become ready. Reuse and source-age counters identify individual stalled
inputs without coupling output cadence to them.

The compositor generates the output PTS directly from the program clock. The
production graph therefore does not need a downstream `fps` filter to discard
extra composites or to manufacture frames when an input stalls.

## Demo integration

The scaling demo selects the EGL compositor for its production grid. For every
source it creates only the IPC receiver, the existing DRM format assertion, and
`drm_prime_to_egl_image`. It passes the unsized EGLImage frames and calculated
destination rectangles to the one compositor.

The demo starts all input groups without serial readiness waits, then starts the
compositor and output. A missing initial input produces a black cell instead of
blocking the remaining sources or Janus output.

The compositor backend, source count, and frame rate remain explicit demo
settings. Source count remains `N`, defaulting to eight.

The old CUDA graph stays selectable by an explicit backend setting for A/B
measurements and the independent per-source `mpdecimate` diagnostic. Source
count is not fixed in the node; neither the compositor nor its cache hardcodes
eight or an x86_64/NVIDIA architecture. NVIDIA is the target used for
validation, not a vendor condition in the node.

## Error handling and observability

Node creation fails with a specific error if the required DMA-BUF import, CUDA
device, CUDA EGL interoperability, EGL frame type, or color format is missing.
It must not silently introduce an OpenGL copy, download to CPU, or fall back to
the old per-input CUDA path.

The importer records cache hits, fresh imports, synchronization fallbacks, and
evictions in periodic aggregate counters rather than per-frame logs. The
compositor records rendered ticks, missed program deadlines, per-input reuse and
age, and EGL/CUDA failures. Optional periodic debug logging exposes these
counters without making normal operation noisy.

## Validation

Local validation covers pure graph generation and layout behavior:

- an eight-source EGL production graph contains eight synchronized EGL
  importers and one compositor;
- it contains no per-source CUDA conversion or scale filter;
- it contains no downstream production `fps` filter;
- its compositor rectangles match the expected aspect-preserving grid;
- source count remains parameterized; and
- the explicit legacy/diagnostic backend still generates the existing CUDA
  path.

The provided Fedora NVIDIA host is used for all EGL/CUDA builds and execution.
Remote validation consists of:

1. a single-source smoke test for import, composition, CUDA handoff, NVENC, and
   Janus output;
2. the eight-source 60-second acceptance run described above;
3. a five-second pause of each input in turn, confirming that output remains at
   60 Hz and only that input's reuse counter increases;
4. release-acknowledgement accounting, including disconnect and frames dropped
   before composition, with no fixed-depth forced releases;
5. process-attributed GPU SM, memory, encoder, decoder, and PCIe sampling;
6. confirmation that all eight producer counters remain at full rate;
7. RTP frame-marker and timestamp accounting at the Janus input;
8. a separate eight-input `mpdecimate` uniqueness run;
9. visual inspection through the existing Janus preview; and
10. an Nsight Systems capture confirming EGL registration, EGL-frame lookup, and
   CUDA texture creation occur only during cache-slot creation, while steady
   state contains only the clear/layer kernels per output tick with no graphics
   map/unmap, per-input copy, or scale chain. Nsight Compute is used only if
   profiling identifies a custom CUDA kernel as a remaining bottleneck.

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
- no graph-management, sentinel, or line-based graph control-protocol changes.
