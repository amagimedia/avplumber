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
Build the supplied GBM shim on the NVIDIA host:

```bash
npm run rebuild:shim
DMA_BROWSER_FORCE_NVIDIA=1 npm start
```

On NVIDIA, `bin/run.sh` always preloads the shim and refuses to start when it is
not built. The shim is native to the host CPU architecture. `bin/run.sh` also
applies the Wayland, EGL/GBM, and DRM settings; hardware video decoding stays
disabled.

Two switches keep the GPU process stable, both on by default:

- `DMA_BROWSER_WEBGPU_OPENGLES=0` turns off `--use-webgpu-adapter=opengles`.
  With `--ignore-gpu-blocklist`, Chromium runs WebGPU on Vulkan through GL
  interop and creates a Vulkan context in the GPU process. Recreating windows
  then crashes it in `AddCleanupTaskForSkiaFlush` (electron/electron#54553).
  On OpenGL ES that Vulkan context is never created (Chromium 152+).
- `DMA_BROWSER_DISABLE_GPU_CRASH_LIMIT=0` turns off
  `--disable-gpu-process-crash-limit`. Without it, a third GPU-process crash
  within five minutes of the last makes Chromium quit the worker.
The [shared Docker stack](../../docker-compose/dmabuf/compose.yaml) builds the shim
into the image.

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
| POST   | `/window/open`      | `{ id, url, width, height, fps, audio, ringSize, holdLastFrame, sync }` |
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

Frames go to `/tmp/dma-page/{id}.sock` (dmabuf FD + 48-byte TexInfo header; layout in
`addons/fdpass/texinfo.h`).
Audio (when `audio: true`) goes to `/tmp/dma-page/{id}-audio.sock` (raw interleaved float32 PCM).

A main-frame load failure, a load exceeding `DMA_BROWSER_LOAD_WATCHDOG_MS` (30 s by default), or a
renderer crash reloads the page after 1 s, doubling up to 30 s while reloads keep failing. The
sockets stay open, so consumers keep their connection and held frames.
With `holdLastFrame` (default `true`), the window sends no frames until a reload loads cleanly, so
consumers repeat the last one they received; `false` sends Chromium's error page instead.

### Browser synchronization between hosts (experimental)

For two browsers rendering the same graphics, `/window/open` accepts an optional
`sync` object. Send identical settings and `fps` to both hosts:

```json
"sync": { "epochMs": 0, "phase": 0, "loopAnimations": false }
```

`epochMs` is a shared Unix wall-clock epoch in integer milliseconds; zero means
the Unix epoch. `phase` defaults to zero and is a fraction of one browser frame
period in `[0, 1)`. Matching sources on A and B need the same phase, regardless
of which worker owns them. Different source pairs can use different phases to
spread rendering work. `sync` takes precedence over worker frame staggering.
Omitting it preserves the normal browser behavior.

Both hosts must already have synchronized system clocks. The service maps the
wall-clock grid to each host's monotonic clock and requests that rendering phase
at every supported integer browser rate. It reapplies after page load and checks
the clock mapping every five seconds. It corrects a change larger than a quarter
frame (at least 2 ms), avoiding continuous rate toggles. This uses Electron's
microsecond-rounded frame interval, not the mixer's exact rational tick grid;
it does not provide NTSC genlock or set the engine's TAI clock mapping.

Set `loopAnimations: true` separately to rephase infinite CSS and Web Animations
API animations to that epoch without editing the HTML. Discovery runs every
500 ms, including after reload, and follows accessible iframe documents. Effects
retain their durations, delays and alternating directions. This intentionally
replaces their page-local start phase, so enable it only where that behavior is
wanted. Finite, paused, pending, non-unit-speed and non-document-timeline
animations are left alone. Browser clocks and network messages are unchanged.

This is best-effort alignment, not deterministic replication of arbitrary pages.
Canvas/WebGL/rAF animation code, media playback, inaccessible iframes and
independently arriving provider commands are not synchronized. A restarted page
still relies on its provider to restore content and finite animation state. Clock
steps can cause visible corrections; selecting the same output tick does not
establish that A and B rendered the same content.

`/status` includes the accepted `sync` configuration and `syncStatus` with phase
application/failure counts, last application time, and measured wall-clock drift
against monotonic. These report requested timing, not measured paint alignment
or peer health. No millisecond or frame-accuracy guarantee has been qualified.

Local validation (no native addon or GPU build required):

```bash
npm run build
npm test
node_modules/.bin/electron tests/browser/sync.cjs
```

The Electron smoke test checks CSS/WAAPI and same-origin iframe loops across
different document start times and reload, plus finite-animation and opt-out
behavior. Two-host Linux/DMA-BUF qualification remains required: compare actual
paint/output delivery and content during load, reload, clock drift and failover,
including the intended Singular/Flowics templates.

#### Isolated Linux A/B probe

`tests/browser/sync-probe.cjs` starts a separate single-window browser service
on loopback port 19099 with a fresh temporary socket/profile directory. Run it
in a disposable container on each test host, using the browser image's normal
NVIDIA/Wayland devices, libraries and environment. Do not run it inside a live
browser worker. It needs the built service and native `fdpass` addon.

Choose one future Unix timestamp in milliseconds for both hosts (allow enough
time for both processes to load), then run on each host with its own label:

```bash
BROWSER_SYNC_START_MS='<same future Unix timestamp in milliseconds>' \
BROWSER_SYNC_LABEL=A \
node_modules/.bin/electron --no-sandbox --ozone-platform=wayland \
  tests/browser/sync-probe.cjs > /tmp/browser-sync-a.json
```

Use label `B` and a separate output file on the other host. Both runs must use
the same `BROWSER_SYNC_FPS` (default 30), `BROWSER_SYNC_EPOCH_MS` (default zero)
and `BROWSER_SYNC_DURATION_MS` (default 15000). `BROWSER_SYNC_PORT` changes the
probe's private port. A bind conflict fails rather than connecting to an existing
service. Each process exits and removes its test sockets/profile on completion.

The probe exercises REST configuration, `ManagedWindow`, the preload, and the
shared-texture capture path. It records paint callback arrival times and sampled
DOM animation times. With no media consumer connected, `no_consumer` drops are
expected and confirm that valid textures reached FD broadcast. This does not
test FD reception, GPU composition, pixel identity, or encoded output continuity.

Copy both JSON results to one machine and compare them:

```bash
node tests/browser/compare-sync.cjs /tmp/browser-sync-a.json /tmp/browser-sync-b.json
node --test tests/browser/compare-sync.test.cjs
```

The comparator fails on capture errors, differing run settings, missing animation
samples, backwards clocks, paint counts outside 90–110% of the configured rate,
delivery gaps over 2.5 frames, or a 95th-percentile nearest-paint difference above
10 ms. Animation-time residuals must stay within one frame at the 95th percentile.
These are initial test thresholds, not established runtime guarantees. Record the
hosts' clock-sync health alongside the results; this probe does not verify NTP/PTP.

For a separate rejoin investigation, set `BROWSER_SYNC_RELOAD_MS=8000` on B to
reload it eight seconds into the run. The trace retains the reload time and any
evaluation errors during navigation. Delivery gaps remain in the comparison and
can make that run fail: a reload must not be hidden by reporting only post-rejoin
timestamps. Verify downstream hold/failover behavior in a subsequent consumer test.

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
shared texture until the consumer acknowledges its frame number. The maximum
sent-but-unacknowledged count is the window's `ringSize`,
else `DMA_BROWSER_DMABUF_POOL_SIZE` (default 9, range 1–64): if that limit is
reached, the newest paint is dropped and released
(`retained_pool_full`); an older in-flight texture is never released early. A
consumer that disconnects or crashes has every retained texture released or
quarantined by the same release path. `/status` reports `releasedFrameCount` and
`retainedFrameCount` for each window.

`ringSize` bounds frames in flight, not allocations. The buffers belong to
Chromium's capture pool, which keeps at most two idle buffers
and destroys the surplus when frames come back in bunches, allocating new DMA-BUFs later. A consumer that
needs a stable set of allocations from Electron returns one frame for
each frame it receives and keeps a constant number out (avplumber's `ipc_dmabuf_source` `hold_frames`).

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
