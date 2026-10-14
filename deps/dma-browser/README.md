# dma-browser

Generic offscreen HTML page renderer for Linux + Electron, with GPU dmabuf capture (via `fdpass`) and PCM audio capture (Unix socket).

`DMA_BROWSER_MAX_WINDOWS` defaults to 8 but is configurable; 8 is not an
Electron limit. The launcher groups at most `DMA_BROWSER_WINDOWS_PER_PROCESS`
windows in one Electron process (default 8) and automatically starts enough
workers for the requested maximum. All workers remain behind the same REST API
on `127.0.0.1:9009`. The project configuration uses Electron 44 / Chromium
152.

To build the runtime-gated Electron native-handle change yourself, use the
patch and script documented in
[DMA-BUF integration](../../doc/dmabuf.md#shim-and-chromium-patches).

## Quick start

```bash
npm install
npm run rebuild:addons   # builds fdpass native addon
npm start                # on NVIDIA, use the setup below before launching
```

### NVIDIA GPUs

The default install downloads stock Electron from its official releases.
Build and enable the supplied GBM shim on the NVIDIA host:

```bash
npm run rebuild:shim
LD_PRELOAD="$PWD/native/gbm-linear-shim/libgbm_linear_shim.so" \
GBM_LINEAR_SHIM=1 GBM_LINEAR_SHIM_ADD_SCANOUT=1 \
DMA_BROWSER_FORCE_NVIDIA=1 npm start
```

The shim is native to the host CPU architecture. `bin/run.sh` applies the
Wayland, EGL/GBM, and DRM settings; hardware video decoding stays disabled.

Two switches keep the GPU process stable, both on by default:

- `DMA_BROWSER_WEBGPU_OPENGLES=0` turns off `--use-webgpu-adapter=opengles`.
  With `--ignore-gpu-blocklist`, Chromium runs WebGPU on Vulkan through GL
  interop and creates a Vulkan context in the GPU process. Recreating windows
  then crashes it in `AddCleanupTaskForSkiaFlush` (electron/electron#54553).
  On OpenGL ES that Vulkan context is never created (Chromium 152+).
- `DMA_BROWSER_DISABLE_GPU_CRASH_LIMIT=0` turns off
  `--disable-gpu-process-crash-limit`. Without it, a third GPU-process crash
  within five minutes of the last makes Chromium quit the worker.
The [shared Docker stack](../../docker-compose/dmabuf/compose.yaml) builds and enables
the shim automatically. Alternatively, use the patched Electron build linked
above without the shim.

### Patched Electron: native-handle capture and the acquire fence

The [patched Electron](../../doc/dmabuf.md#shim-and-chromium-patches) can hand
the consumer a native-handle texture instead of a mappable one. It then no
longer waits for the GPU before `paint`: each frame comes with an acquire fence
(`textureInfo.handle.nativePixmap.acquireFenceFd`, a sync_file fd that becomes
readable once the frame's GPU work has completed), which the app sends to the
consumer as the frame's second fd, flagged in bit 31 of the TexInfo pixel format
word (see `addons/fdpass/texinfo.h`). That is safe only with a consumer that
waits on the fence (avplumber's `ipc_dmabuf_source` does; its `fence_timeout_ms`
parameter and `fence_*` counters are documented with the node), so the app
requests the feature only when told to. Both switches are off by default and
stock Electron ignores the features:

- `DMA_BROWSER_NATIVE_HANDLE_CAPTURE=1` enables
  `RenderableMappableSharedImageForceScanout`: native-handle capture, no GPU
  wait, a fence with every frame.
- `DMA_BROWSER_ZERO_COPY_EXPORT=1` adds `AvpZeroCopyOffscreenExport`, the
  patched build's zero-copy export. It is a native-handle capture, so it turns
  the first switch on as well; `DMA_BROWSER_NATIVE_HANDLE_CAPTURE=0` next to it
  is refused at start-up, as is either feature name in
  `DMA_BROWSER_CHROMIUM_EXTRA_FEATURES`.
- `DMA_BROWSER_EXPORT_BUFFERS=N` (2-16, default 9) is the capacity of a
  window's buffer queue, the whole per-window budget (N x 8.4 MiB at 1080p). It
  is passed to Chromium as `AvpZeroCopyOffscreenExport:buffers/N`: exactly N
  buffers exist per window, the capturer lends at most N, and a frame that
  finds no free buffer is not painted (counted as `skipped` in the GPU log),
  never blocking and never allocating more. In zero-copy mode the app keeps no
  retained pool of its own: a frame is lent until the consumer acknowledges it
  and Chromium's N is the only bound (`ringSize` is echoed, not enforced). The
  app announces the mode in `/status` and in every window snapshot
  (`captureMode: "zero-copy"`, `exportBuffers: N`); avplumber's mixer reads it,
  sets `return_queue` 0 (every frame acknowledged as soon as its last user lets
  go), keeps every import for the window's life (no idle expiry, no purge) and
  bounds every hold to N: its chain's edges hold one frame each and each
  compositor holds at most N-6 frames waiting for their tick (two buffers are
  the browser's, four are held outside that queue). Nine is the least a 30 fps
  mixer with its default two-frame latency needs without repeating frames;
  below it the mixer warns and repeats. On the stock copy path N is only the
  default `DMA_BROWSER_DMABUF_POOL_SIZE`.

In these modes Chromium's allocation is renderable on NVIDIA as it is, so the
GBM shim is unnecessary: start without `LD_PRELOAD`, or leave the preload in
place and set `GBM_LINEAR_SHIM=0`, which makes the shim pass every allocation
through unchanged. The Docker image sets `LD_PRELOAD` and `GBM_LINEAR_SHIM=1`;
override both (`-e LD_PRELOAD= -e GBM_LINEAR_SHIM=0`) when switching a container
to the patched build. The patched build may report `pixelFormat` `rgba`; the
header then carries `RGBA` and the consumer imports `DRM_FORMAT_ABGR8888`, the
same path as today's `BGRA`/`ARGB8888` with the channel offsets taken from the
format.

An avplumber that predates the fence rejects every fenced frame (it logs an
unsupported pixel format) and acknowledges it, so the window shows nothing
rather than a torn frame; but its receive buffer has room for both fds and it
closes only the plane, so it leaks one descriptor per frame until it runs out.
Update avplumber before turning the switches on, not after. The window log
notes the first fenced paint.

## Process grouping

The default keeps up to eight windows in one Electron process. For 20 windows,
the launcher therefore starts three Electron workers:

```bash
DMA_BROWSER_MAX_WINDOWS=20 npm start
```

Change `DMA_BROWSER_WINDOWS_PER_PROCESS` to tune the grouping size. Larger
groups share more Chromium GPU-process and compositor state; smaller groups
provide more fault isolation at additional GPU-memory and scheduling cost. Use
one Electron process per window when isolation is more important:

```bash
DMA_BROWSER_PROCESS_COUNT=16 \
DMA_BROWSER_WINDOWS_PER_PROCESS=1 \
DMA_BROWSER_MAX_WINDOWS=16 \
npm start
```

The public API remains on `DMA_BROWSER_REST_HOST` / `DMA_BROWSER_REST_PORT`.
The supervisor places windows across private loopback worker ports starting at
`DMA_BROWSER_WORKER_BASE_PORT` (9010 by default), enforces IDs and capacity
globally, and restores each worker's assigned windows after that worker exits.
Each worker receives a separate Chromium user-data directory. Private workers
cannot be bound to a non-loopback address.

`DMA_BROWSER_WINDOWS_PER_PROCESS` defaults to 8. Unless explicitly set,
`DMA_BROWSER_PROCESS_COUNT` is calculated as
`ceil(DMA_BROWSER_MAX_WINDOWS / DMA_BROWSER_WINDOWS_PER_PROCESS)`. An explicit
process count remains available for isolation tests, but its combined capacity
must cover `DMA_BROWSER_MAX_WINDOWS`. The public API always enforces
`DMA_BROWSER_MAX_WINDOWS`, including when the last worker has unused slots.

`DMA_BROWSER_STAGGER_FRAMES=1` (off by default) starts each worker's 60 fps
frames at a different point of the frame period: worker 0 at the start, 1 at
one half, 2 at a quarter, 3 at three quarters, and so on, so whichever workers
a show runs stay apart. Without it, every 60 fps window of every worker starts
its frame at the same instant, because Chromium hands a window's own frame
timebase to its display only when the interval differs from 1/60 s; the host
then sees one burst of runnable threads per frame. Windows at other rates
already start at their own instants and are left alone. The 60 fps windows of
one worker share a phase. Frame rate and frame count do not change, and a
frame's timestamp is still the monotonic time of its paint, which now falls at
the worker's point of the period.

## REST API

| Method | Path                | Body                                                              |
| ------ | ------------------- | ----------------------------------------------------------------- |
| POST   | `/window/open`      | `{ id, url, width, height, fps, audio, ringSize, holdLastFrame }` |
| POST   | `/window/close`     | `{ id }`                                                          |
| GET    | `/window/close/all` |                                                                   |
| POST   | `/window/refresh`   | `{ id }`                                                          |
| POST   | `/window/update`    | `{ id, url }`                                                     |
| POST   | `/window/show`      | `{ id, show }`                                                    |
| GET    | `/status`           |                                                                   |
| POST   | `/workers/recover`  | `{ ids }`                                                         |
| GET    | `/metrics/cpu`      |                                                                   |

`/metrics/cpu` reads `/proc` for this service's own process tree (Linux only; 500 elsewhere) and
returns the CPU used since the previous call, in percent of one core: by Chromium process type
(`processes`), the GPU process split into viz compositor, GPU main (the thread whose tid is the
pid), IO and other threads (`gpuThreads`), and cores' worth used with a per-core split
(`cores`, an approximation: GPU-process threads are charged to the core each last ran on, other
processes to their main thread's). To stay cheap it reads one stat file per process, each
command line once, and thread files only for GPU processes. The first call only takes the
baseline; calls within 500 ms of a sample return that sample again. The mixer polls every 5 s.

Frames go to `/tmp/dma-page/{id}.sock` (dmabuf FD + 48-byte TexInfo header, plus the
acquire fence fd when the header flags one; layout in `addons/fdpass/texinfo.h`).
Audio (when `audio: true`) goes to `/tmp/dma-page/{id}-audio.sock` (raw interleaved float32 PCM).

A main-frame load failure, a load exceeding `DMA_BROWSER_LOAD_WATCHDOG_MS` (30 s by default), or a
renderer crash reloads the page after 1 s, doubling up to 30 s while reloads keep failing. The
sockets stay open, so consumers keep their connection and held frames.
With `holdLastFrame` (default `true`), the window sends no frames until a reload loads cleanly, so
consumers repeat the last one they received; `false` sends Chromium's error page instead.

### Frame lifetime and transport counters

Rebuild the `fdpass` addon and avplumber together when updating the DMA-BUF
protocol. A consumer sends 16-byte little-endian `ACK1` records: kind `0` at
offset 4 releases the frame number at offset 8; kind `1` stops new deliveries;
kind `2` confirms that all received frames have finished their GPU reads.
`ipc_dmabuf_source` keeps the connection alive until its last frame is released,
then sends kind `2`. Shut down consumers before closing browser windows.

An unexpected disconnect cannot prove GPU reads have finished. The browser pins
unacknowledged textures, pauses the affected source, and reports
`quarantinedFrameCount` in `/status`. After stopping the affected consumer,
restart its browser worker with `POST /workers/recover`; reopening the window
alone is refused. Other windows in that worker also restart, so `ids` must list
all of them. This protocol controls buffer reuse. Whether the pixels are ready
is a separate matter: stock Electron waits for the GPU before `paint`, the
patched build sends an acquire fence with the frame instead (above).

`txFrameCount` counts paints delivered to at least one consumer. `droppedFrames`
counts paints with no consumer or any failed delivery, so a partial broadcast
increments both counters. `fdpass_backpressure`, `fdpass_disconnected`, and
`fdpass_errors` in `droppedReasons` count failed deliveries per consumer.

The DMA-BUF socket is bidirectional. The browser retains every transmitted
shared texture until the consumer acknowledges its frame number. On the stock
copy path the maximum sent-but-unacknowledged count is the window's `ringSize`,
else `DMA_BROWSER_DMABUF_POOL_SIZE` (default `DMA_BROWSER_EXPORT_BUFFERS`; both
range 1–64): if that limit is reached, the newest paint is dropped and released
(`retained_pool_full`); an older in-flight texture is never released early. In
zero-copy mode there is no such limit here: Chromium's queue of
`DMA_BROWSER_EXPORT_BUFFERS` buffers bounds the frames out, and a consumer that
disconnects or crashes still has every lent texture released or quarantined by
the same release path. `/status` reports `releasedFrameCount` and
`retainedFrameCount` for each window.

`ringSize` bounds frames in flight, not allocations. On the stock copy path the
buffers belong to Chromium's capture pool; the patched build keeps its idle
buffers, unpatched Electron keeps at most two and destroys the surplus when
frames come back in bunches, allocating new DMA-BUFs later. A consumer that
needs a stable set of allocations from unpatched Electron returns one frame for
each frame it receives (avplumber's `ipc_dmabuf_source` `return_queue`).

## Layout

- `src/main/` — Electron main process (TypeScript, service-oriented OOP)
- `src/preload/` — Renderer preload (WebGL sync mitigation, audio capture pipeline)
- `src/shared/` — IPC contracts shared between main + preload
- `addons/fdpass/` — vendored native addon for dmabuf FD passing
- `tests/unit/` — Vitest unit tests
- `bin/run.sh` — launcher that applies NVIDIA Wayland and EGL/GBM runtime args when appropriate

## Tests

```bash
npm test         # unit tests (Vitest)
npm run lint     # eslint
npm run build    # tsc -b
```
