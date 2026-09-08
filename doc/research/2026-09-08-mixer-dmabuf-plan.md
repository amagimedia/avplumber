# Mixer with DMA-BUF browser sources: plan

Goal: run the existing mixer demo (fullscreen, 2/4/8/16-box, Cut/Fade/Wipe,
TUI, Janus output) with live browser pages from the DMA-BUF demo as inputs,
on the T4 instance, without changing the behaviour of either demo when the new
option is not used. No new code paths for compositing or transitions.

## Why a switch, not a new directory

The mixer only needs, per source, an edge carrying CUDA frames at the mixer
frame rate plus the group that owns the chain (`MixerGraphBuilder.add_source`).
The DMA-BUF demo already builds exactly that from one browser socket
(`make_dmabuf_cuda_input_nodes` in `demos/dmabuf-browser/graph/dmabuf_browser_common.py`:
`ipc_dmabuf_source -> assume_video_format -> drm_prime_to_cuda -> filter_video`
snapping the shared monotonic clock to the 1/fps grid). Everything else in
`demos/mixer/mixer.py` (`_register_sources`, `_define_scenes`, layouts,
outputs, preheat, TUI) is input-agnostic. So the change is one input scheme.

## The switch

`--input dmabuf://<window-id>` alongside the existing file/URL inputs, plus:

| Option | Default | Meaning |
| --- | --- | --- |
| `--dmabuf-socket-dir` | `/tmp/dma-page` | where `dma-browser` puts `<window-id>.sock` |
| `--dmabuf-size WxH` | `1280x720` | browser window size (must be in `DMA_BROWSER_ALLOWED_DIMS`) |
| `--dmabuf-open URL` | none | open the windows named by the inputs through the REST API before building; without it the windows must already exist |
| `--dmabuf-rest` | `http://127.0.0.1:9009` | dma-browser REST endpoint |

`_build_input` dispatches on the `dmabuf://` scheme and returns the chain's
CUDA edge; file inputs are untouched. Window fps is the mixer fps
(`--fps 60`). Inputs can be mixed: cameras from files and browser pages in the
same 16-box.

## Code changes (small)

1. Put an api-parameterised `dmabuf_cuda_input_nodes` plus `rest_request`,
   `open_browser_windows` and `wait_for_sockets` in `avpmixer/dmabuf_inputs.py`.
   The DMA-BUF demo keeps its own copy of the chain builder: its runtime image
   has no `avpmixer` on the path, and re-pointing it would change that demo's
   image or compose files, which this change must not do.
2. `demos/mixer/mixer.py`: the scheme dispatch in `_build_input`, the four
   options in `parse_args`/`GraphOptions`, and window opening in
   `build_application` when `--dmabuf-open` is given. `MixerApplication.start`
   already waits for a frame on every input edge, which is the right readiness
   check for a browser source too.
3. `demos/dmabuf-browser/compose.mixer.yaml`: an override that runs the mixer
   demo in the existing consumer image (it is the one built with DRM/EGL and
   NVIDIA), mounting `avpmixer/` and `demos/mixer/` read-only, sharing the
   `dma-browser-sockets` volume, exposing the control port 7777 for the TUI.
   The base `compose.yaml` and `compose.scaling.yaml` are unchanged.
4. Tests: a fake-API graph test in `demos/mixer/tests` asserting that a
   `dmabuf://` input produces the DMA-BUF chain nodes and that file inputs are
   byte-identical to today (the existing dump comparison). One unit test for
   the scheme parser.
5. README section in the mixer demo: "Browser pages as sources" with the run
   commands; the DMA-BUF README gets a one-line pointer.

Estimated size: about 120 lines of Python, mostly moved rather than new.

## Runtime on the T4

```sh
docker compose --env-file demos/dmabuf-browser/.env \
  -f demos/dmabuf-browser/compose.yaml -f demos/dmabuf-browser/compose.mixer.yaml \
  up -d janus janus-preview wayland dma-browser mixer
python3 demos/mixer/tui.py --host 127.0.0.1 --port 7777 --fade-duration 0.8
```

with `HTML_OVERLAY_URL` set to the Singular page (or the bundled
`smoke.html` animation for a test without external graphics) and the mixer
service command:

```
python3 /opt/avplumber/demos/mixer/mixer.py --fps 60 --janus-output \
  --dmabuf-open "$HTML_OVERLAY_URL" --dmabuf-size 1280x720 \
  --input dmabuf://page_00 ... --input dmabuf://page_15
```

## Sizing and risks

- Sixteen 1280×720 pages at 60 fps is more than the DMA-BUF demo has run
  (its sixteen-source measurements used 480×270). Start the 16-box with
  `--dmabuf-size 480x270` as in `compose.scaling.yaml`, and the 2/4-box with
  1280×720. Measure GPU, NVDEC-free this time, and Electron CPU as the
  DMA-BUF demo did (`runtime-load` JSON).
- A browser page that stops painting stalls its compositor input; the mixer
  already tolerates a dropped source (source-loss recovery is in its tests),
  and the DMA-BUF chain restarts with `auto_restart: group`.
- `DMA_BROWSER_DMABUF_POOL_SIZE` (11) bounds frames in flight per window; the
  mixer's two compositor slots plus the OTM hold at most a few, so the default
  is enough, but it is the first thing to raise if the continuity counters
  show discards.
- Frame-exact verification with the burned frame codes does not apply to
  browser pages; use the DMA-BUF demo's compositor continuity counters
  (repeats/discards per input) and the mixer's own checks.

## Sequence

1. Move the helpers and add the switch; local tests including the identical
   file-input graph.
2. On the T4: bring up the DMA-BUF stack with the bundled animation, run the
   mixer with 4 pages, drive the TUI, check the continuity counters.
3. Same with 16 pages at 480×270, then with the Singular URL.
4. Record a short movie in the mixer's recording format, publish under a new
   media release, link it from the mixer page as a second recording.

## Findings on the T4 (2026-09-08)

- **Compositor format.** `cuda_rect_overlay` required every input to match its
  NV12 canvas, so browser frames (`rgb0` CUDA) were rejected. The compositor now
  draws packed 8-bit RGB inputs through a fused scale-and-convert kernel (BT.709
  limited range, one launch per layer), so browser pages and NVDEC video share
  one NV12 canvas. The video-only demo is unchanged: sixteen clips, 2 % GPU.
- **Per-frame DMA-BUF import.** `drm_prime_to_cuda` re-imported every frame
  (EGL image, GL copy, `glFinish`, CUDA map, copy, sync, destroy). With sixteen
  pages that is 960 imports per second and the GPU sat at 74 % with the encoder
  at 54 fps regardless of page size (960x540, 640x360, 480x270 all measured the
  same), while the DMA-BUF demo's production path (`drm_prime_to_egl_image` →
  `egl_image_cuda_overlay`) caches one import per physical allocation. The node
  now keeps the same allocation-identity cache (inode + geometry, `dup` of the
  fd, TTL 3 s, 64 entries) with a CUDA registration per allocation and one
  device copy per frame.
- The DRM hwaccel on the source only annotated frames for filters; the mixer
  no longer initialises it, matching the sixteen-source configuration the
  DMA-BUF demo measured.
- The mixer demo canvas is portrait 1080x1920 by design, so 16:9 pages are
  letterboxed in fullscreen.

Measured with the cached import (sixteen Singular pages, sixteen windows at
60 fps, zero drops, encoder 59.4–59.6 fps): 480x270 33.7 % GPU / 584 MiB,
960x540 37.4 % / 1308 MiB, 1280x720 40.8 % / 1394 MiB, 1920x1080 51.1 % /
3585 MiB, host CPU 52–78 % busy. Before the cache, 74 % GPU and 54 fps encode at
every size.
