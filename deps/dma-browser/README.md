# dma-browser

Offscreen HTML graphics renderer for Linux + Electron, with GPU DMA-BUF capture
(via `fdpass`) and PCM audio capture (Unix socket).

Single Electron process. Up to 8 BrowserWindows. Controlled via a REST API on
`127.0.0.1:9009`. Uses stock Electron; NVIDIA capture support comes from the
GBM shim under `native/gbm-linear-shim/`.

## Quick start

```bash
npm install
npm run rebuild:addons   # builds fdpass native addon
npm run rebuild:shim     # builds the NVIDIA GBM allocation shim
npm start                # launches Electron through bin/run.sh
```

### NVIDIA GPUs

`bin/run.sh` autodetects NVIDIA, configures the Wayland/GBM environment, and
preloads the allocation shim. Hardware video decoding is deliberately disabled;
this process renders graphics overlays. Force-enable the NVIDIA runtime args
with:

```bash
DMA_BROWSER_FORCE_NVIDIA=1 npm start
```

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

## Layout

- `src/main/` — Electron main process (TypeScript, service-oriented OOP)
- `src/preload/` — Renderer preload (WebGL sync mitigation, audio capture pipeline)
- `src/shared/` — IPC contracts shared between main + preload
- `addons/fdpass/` — vendored native addon for dmabuf FD passing
- `tests/unit/` — Vitest unit tests
- `bin/run.sh` — launcher that applies NVIDIA Wayland/GBM runtime settings

## Tests

```bash
npm test         # unit tests (Vitest)
npm run lint     # eslint
npm run build    # tsc -b
```
