# dma-browser

Generic offscreen HTML page renderer for Linux + Electron, with GPU dmabuf capture (via `fdpass`) and PCM audio capture (Unix socket).

`DMA_BROWSER_MAX_WINDOWS` defaults to 8 but is configurable; 8 is not an
Electron limit. The launcher groups at most `DMA_BROWSER_WINDOWS_PER_PROCESS`
windows in one Electron process (default 8) and automatically starts enough
workers for the requested maximum. All workers remain behind the same REST API
on `127.0.0.1:9009`. The project configuration uses Electron 44 / Chromium
152.

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

For the optional preallocated Chromium exporter, enable
`DMA_BROWSER_CHROMIUM_EXTRA_FEATURES=AvpZeroCopyOffscreenExport:buffers/11,RenderableMappableSharedImageForceScanout`
on a compatible patched Electron build. Run the mixer with
`AVP_BROWSER_COPY_ON_IMPORT=1`: it waits for the acquire fence, copies once
into owned CUDA memory on a private non-blocking stream per source, and returns
the browser frame before publishing downstream. The browser distribution must include `resources/avp-capture-pool.json`
with `{"version":1,"size":11}` for the hardcoded-11 binary. A binary built
with the optional startup-size patch uses `{"version":2,"size":11}` instead;
then `buffers/6`, `buffers/9`, or another integer from 2 to 16 selects the fixed
pool at worker startup. The service reports the selected size, and the mixer
uses it for sender retention, import caching, and warm-up. Changing the size
requires restarting the browser workers and mixer; an active pool never grows
or shrinks. Only mark a distribution version 2 if that patch is in its binary.
For the version-2 build, start with `buffers/9`. Six buffers use less memory
but leave less room for capture and consumer timing variation; check moving
pages during transitions before reducing the pool. Setting the mixer's
`browser_frame_queue` alone does not change this fixed pool.
The pool bounds capture allocations; it does not set the compositor's frame deadline.
The mixer registers the full pool before
going on air, retains those registrations, and uses no return queue.
The ordinary capture path remains the default.

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
the shim automatically.

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
| POST   | `/window/paint`     | `{ id, enabled }`                                                 |
| POST   | `/window/update`    | `{ id, url }`                                                     |
| POST   | `/window/show`      | `{ id, show }`                                                    |
| GET    | `/status`           |                                                                   |
| POST   | `/workers/recover`  | `{ ids }`                                                         |
| GET    | `/metrics/cpu`      |                                                                   |

`/window/paint` pauses or resumes offscreen painting without reloading the page.
Resuming also requests a fresh paint, including for static pages. A mixer can
pause after receiving its first usable frame while its output surfaces warm up,
then resume before releasing output.

`paintsSkipped` counts paints Chromium numbered but never delivered: the gaps in
the paint textures' `textureInfo.metadata.frameCount` (the capturer's
`capture_counter`). The capturer's own "capture skipped" (too many frames
outstanding) takes no number and is not counted. A
counter that starts low again (a reload) is a new baseline, not a gap.

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
all of them. This protocol controls buffer reuse. Stock Electron waits for its
GPU work before delivering `paint`.

`txFrameCount` counts paints delivered to at least one consumer. `droppedFrames`
counts paints with no consumer or any failed delivery, so a partial broadcast
increments both counters. `fdpass_backpressure`, `fdpass_disconnected`, and
`fdpass_errors` in `droppedReasons` count failed deliveries per consumer.

The DMA-BUF socket is bidirectional. The browser retains every transmitted
shared texture until the consumer acknowledges its frame number. On the stock
copy path the maximum sent-but-unacknowledged count is the window's `ringSize`,
else `DMA_BROWSER_DMABUF_POOL_SIZE` (default 9, range 1–64): if that limit is
reached, the newest paint is dropped and released
(`retained_pool_full`); an older in-flight texture is never released early. A
consumer that disconnects or crashes has every retained texture released or
quarantined by the same release path. `/status` reports `releasedFrameCount` and
`retainedFrameCount` for each window.

`ringSize` bounds frames in flight, not allocations. On the stock copy path the
buffers belong to Chromium's capture pool, which keeps at most two idle buffers
and destroys the surplus when frames come back in bunches, allocating new DMA-BUFs later. A consumer that
needs a stable set of allocations from Electron returns one frame for
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
