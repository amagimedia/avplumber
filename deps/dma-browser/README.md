# dma-browser

Generic offscreen HTML page renderer for Linux + Electron, with GPU dmabuf capture (via `fdpass`) and PCM audio capture (Unix socket).

`DMA_BROWSER_MAX_WINDOWS` defaults to 8 but is configurable; 8 is not an
Electron limit. The launcher groups at most `DMA_BROWSER_WINDOWS_PER_PROCESS`
windows in one Electron process (default 8) and automatically starts enough
workers for the requested maximum. All workers remain behind the same REST API
on `127.0.0.1:9009`. The project configuration uses Electron 41 / Chromium
146.

To build the runtime-gated Electron native-handle change yourself, use the
patch and script documented in
[`demos/dmabuf-browser`](../../demos/dmabuf-browser/README.md#build-patched-electron-no-shim).

## Quick start

```bash
npm install
npm run rebuild:addons   # builds fdpass native addon
npm run rebuild:shim     # optional: builds the stock-Electron NVIDIA shim
npm start                # launches Electron through bin/run.sh
```

Build the optional shim on the target host; its ignored `.so` is native to the
host CPU architecture.

### NVIDIA GPUs

`bin/run.sh` autodetects NVIDIA, enables the patched Electron OSR native-handle
feature, and applies the Wayland, EGL/GBM, and DRM render-node settings. The
same patched binary retains upstream behavior on other GPU vendors. The browser
renderer has hardware video decoding disabled. A stock Electron binary can use
the GBM shim documented in the demo README instead. Force-enable the NVIDIA
runtime path with:

```bash
DMA_BROWSER_FORCE_NVIDIA=1 npm start
```

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

## REST API

| Method | Path                | Body                                                       |
| ------ | ------------------- | ---------------------------------------------------------- |
| POST   | `/window/open`      | `{ id, url, width, height, fps, audio }`                   |
| POST   | `/window/close`     | `{ id }`                                                   |
| GET    | `/window/close/all` |                                                            |
| POST   | `/window/refresh`   | `{ id }`                                                   |
| POST   | `/window/update`    | `{ id, url }`                                              |
| POST   | `/window/show`      | `{ id, show }`                                             |
| GET    | `/status`           |                                                            |

Frames go to `/tmp/dma-page/{id}.sock` (dmabuf FD + 48-byte TexInfo header).
Audio (when `audio: true`) goes to `/tmp/dma-page/{id}-audio.sock` (raw interleaved float32 PCM).

The DMA-BUF socket is bidirectional. The browser retains every transmitted
shared texture until the consumer acknowledges its frame number. The maximum
sent-but-unacknowledged count is `DMA_BROWSER_DMABUF_POOL_SIZE` (default 11,
range 1–64). If that limit is reached, the newest paint is dropped and released;
an older in-flight texture is never released early. `/status` reports
`releasedFrameCount` and `retainedFrameCount` for each window.

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
