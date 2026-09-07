# DMA-BUF browser demo

[![Watch sixteen browser sources composed on the GPU](https://github.com/amagimedia/avplumber/releases/download/dmabuf-demo-media-2026-09/dmabuf-demo.jpg)](https://amagimedia.github.io/avplumber/demos/dmabuf-browser/docs/)

[Watch the demo](https://amagimedia.github.io/avplumber/demos/dmabuf-browser/docs/) · [Download MP4](https://github.com/amagimedia/avplumber/releases/download/dmabuf-demo-media-2026-09/dmabuf-demo.mp4) · [Technical reference](docs/guide.md)

Capture HTML pages in Electron, export GPU DMA-BUF frames, compose a grid in
AVPlumber and stream it through Janus/WebRTC. The default grid is **16 pages,
four Electron workers × four pages, 480×270@60 inputs → 1920×1080@60 output**.
This demo has no TUI; choose the page and dimensions through configuration.

The linked recording is the earlier 1080p-input demonstration (9.98 seconds,
8.6 MB). Its sources delivered about 39–40 new frames/s into a 60 fps output;
it predates the current configuration and 60 fps delivery results below.
The page labels this explicitly. Media is hosted as public release assets,
outside Git history.

## Run

Follow the [shared NVIDIA setup](../README.md). Browser capture requires NVIDIA
graphics/EGL/GBM support and `/dev/dri`, in addition to CUDA and NVENC.
The Docker stack builds from public sources, downloads official Electron and
compiles the supplied GBM compatibility shim. No custom Chromium download,
company account or TensorRT is required.

From the repository root:

```sh
cd demos/dmabuf-browser
cp .env.example .env
docker compose --env-file .env -f compose.yaml -f compose.scaling.yaml up --build
```

Open <http://127.0.0.1:8080>. The default is a bundled animated HTML fixture;
set `DMABUF_SOURCE_URL` to render your own page. For a remote host, configure
`JANUS_HOST_IP` in `.env` and use the host's preview address.

For one full-resolution page, use the same stack with:

```sh
DMABUF_TEST_MODE=single DMABUF_SOURCE_WIDTH=1920 DMABUF_SOURCE_HEIGHT=1080 \
docker compose --env-file .env -f compose.yaml -f compose.scaling.yaml up --build
```

Stop it with:

```sh
docker compose --env-file .env -f compose.yaml -f compose.scaling.yaml down
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DMABUF_SOURCE_COUNT` | `16` | Number of pages |
| `DMABUF_BROWSER_WINDOWS_PER_PROCESS` | `4` | Pages per worker; worker count is derived |
| `DMABUF_SOURCE_WIDTH`, `DMABUF_SOURCE_HEIGHT` | `480`, `270` | Browser capture size |
| `DMABUF_SOURCE_FPS` | `60` | Input and output rate |
| `DMABUF_SOURCE_URL` | bundled fixture | Page to render |
| `DMABUF_CANVAS_WIDTH`, `DMABUF_CANVAS_HEIGHT` | `1920`, `1080` | Program size |
| `MIXER_LATENCY_MS` | two frame periods | Input jitter budget; about 33 ms at 60 fps |

Browser resolution is a startup setting. A 480×270 page suits a tile but loses
detail if later enlarged; use 1920×1080 sources when fullscreen quality matters.
For the tested full-resolution sixteen-source configuration, also set
`DMABUF_BROWSER_WINDOWS_PER_PROCESS=1`. Benchmark your actual page workload.

## Processing graph

[View the processing graph](https://amagimedia.github.io/avplumber/demos/dmabuf-browser/docs/#graph).
The older graph screenshot is labelled with the configuration it represents.
In the current path, each source feeds cached EGL/CUDA import and the shared
native compositor, followed by NVENC and RTP output. Bilinear tile scaling
happens directly in the compositor; intermediate per-source scaling graphs
are unnecessary.

DMA-BUF imports are cached by allocation identity, not by numeric FD. Frames
remain owned until downstream GPU work completes. The native timing code is
shared with the [mixer demo](../mixer/README.md); it absorbs bounded jitter and
keeps output running when a source is late.

## Measured runtime

On a Tesla T4 / 16-vCPU Ubuntu host, the current four-worker configuration with
sixteen animated Singular pages delivered **59.91–59.99 fps per source and
59.91 fps encoded output**, with zero browser capture drops over 12 seconds.
It averaged **38.92% GPU, 576 MiB GPU memory, 8.15 browser CPU cores and
0.30 AVPlumber CPU cores**. These delivery counters do not prove frame uniqueness.
The final smoke check also held 60 Hz output with zero browser capture drops,
but two input streams accumulated compositor repeats and discards. Input jitter
can therefore reduce motion continuity even when the output rate stays at 60 fps.
The bundled page can have a different cost.

Stage comparisons used fixed 1200 MHz GPU graphics / 5000 MHz memory clocks,
twenty one-second samples, and the same page workload:

| Stage | Whole-device GPU busy time |
| --- | ---: |
| Browser rendering, capture disabled | 10.70% |
| Browser rendering and capture, AVPlumber stopped | 30.35% |
| Complete browser → AVPlumber → NVENC pipeline | 32.75% |

The large remaining cost is in browser rendering/capture, before AVPlumber's
import. These are sequential whole-device observations, not exact additive
engine costs. Automatic-clock percentages from different workloads cannot
establish that two stages cost the same. Automatic clocks were restored.

See the [reference](docs/guide.md) for the interop chain, cache ownership,
diagnostics and optional source-build route, and the [Ubuntu/GBM validation
notes](docs/validation.md) for the earlier rendering investigation.
