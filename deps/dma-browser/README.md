# dma-browser

Generic offscreen HTML page renderer for Linux + Electron, with GPU dmabuf capture (via `fdpass`) and PCM audio capture (Unix socket).

Single Electron process. Up to 8 BrowserWindows. Controlled via a REST API on `127.0.0.1:9009`. Uses the patched Electron 41 / Chrome 146 build from the project `.npmrc`.

## Quick start

```bash
npm install
npm run rebuild:addons   # builds fdpass native addon
npm start                # launches Electron through bin/run.sh
```

### NVIDIA GPUs

`bin/run.sh` autodetects NVIDIA and applies the Wayland/VAAPI launch environment for the patched Chromium build. No GBM `LD_PRELOAD` shim is used. Force-enable the NVIDIA runtime args with:

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
- `bin/run.sh` — launcher that applies NVIDIA Wayland/VAAPI runtime args when appropriate

## Tests

```bash
npm test         # unit tests (Vitest)
npm run lint     # eslint
npm run build    # tsc -b
```
